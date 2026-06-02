// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tantivy side of the FTS comparison — a second reference baseline next to
//! Apache Lucene (`LuceneFtsBench.java`). Reads the shared inputs produced by
//! `mem_wal_fts_bench gen` (corpus.txt / corpus_tok.txt / queries.txt /
//! truth.txt), builds a Tantivy index, measures build throughput, term /
//! phrase / OR query latency + QPS, recall@k vs the exact-BM25 truth, and the
//! index footprint. Emits one JSON line tagged `impl=tantivy`, the same shape
//! as the Lance and Lucene benches.
//!
//! In-memory mode (default) uses a `RamDirectory` — the apples-to-apples match
//! for Lucene's `ByteBuffersDirectory` and Lance's `FtsMemIndex`. With `--dir
//! PATH` it builds an on-disk `MmapDirectory` index — the on-disk arm of the
//! comparison (paired with Lance's flushed memtable and Lucene's FSDirectory).
//!
//! Run A (`--run a`) indexes the pre-tokenized `corpus_tok.txt` with a
//! whitespace tokenizer (no lowercasing) — isolates the inverted index + BM25
//! scorer. Run B (`--run b`) indexes raw `corpus.txt` with Tantivy's native
//! `default` tokenizer (simple + lowercase) — each engine on its own analyzer.
//!
//! Usage:
//!   tantivy_fts_bench --in-dir DIR --run a|b --k 10 --threads 64 \
//!       [--no-positions] [--dir INDEX_DIR]

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::{BooleanQuery, Occur, PhraseQuery, Query, TermQuery};
use tantivy::schema::{
    FAST, Field, IndexRecordOption, STORED, Schema, SchemaBuilder, TextFieldIndexing, TextOptions,
    Value,
};
use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer, WhitespaceTokenizer};
use tantivy::{Index, IndexWriter, TantivyDocument, Term, doc};

const TEXT_COL: &str = "text";
const WS_TOKENIZER: &str = "ws_raw";

struct Args {
    in_dir: PathBuf,
    run: char,
    k: usize,
    threads: usize,
    with_position: bool,
    /// On-disk index directory; `None` => in-memory `RamDirectory`.
    dir: Option<PathBuf>,
}

fn read_lines(path: &Path) -> Vec<String> {
    let f = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    std::io::BufReader::new(f)
        .lines()
        .map(|l| l.expect("read line"))
        .collect()
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Total size (bytes) of every file in an on-disk index directory.
fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            if let Ok(meta) = e.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_bytes(&e.path());
                }
            }
        }
    }
    total
}

struct Q {
    kind: String,
    raw: String,
    tok: String,
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let argv: Vec<&str> = argv
        .iter()
        .map(|s| s.as_str())
        .filter(|s| *s != "--bench")
        .collect();
    let get = |flag: &str, def: &str| -> String {
        argv.iter()
            .position(|a| *a == flag)
            .and_then(|i| argv.get(i + 1))
            .map(|s| s.to_string())
            .unwrap_or_else(|| def.to_string())
    };
    let args = Args {
        in_dir: get("--in-dir", "/tmp/fts_compare").into(),
        run: get("--run", "a").chars().next().unwrap_or('a'),
        k: get("--k", "10").parse().unwrap(),
        threads: get("--threads", "0").parse().unwrap(),
        with_position: !argv.contains(&"--no-positions"),
        dir: argv
            .iter()
            .position(|a| *a == "--dir")
            .and_then(|i| argv.get(i + 1))
            .map(PathBuf::from),
    };
    if args.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global();
    }
    run(&args);
}

fn build_schema(with_position: bool) -> (Schema, Field, Field) {
    let mut sb: SchemaBuilder = Schema::builder();
    // id: stored numeric so we can read back the original doc id; not indexed.
    let id_field = sb.add_u64_field("id", STORED | FAST);
    // text: indexed with our chosen tokenizer. Positions only when needed
    // (phrase support) — the apples-to-apples match for Lance `with_position`
    // and Lucene DOCS_AND_FREQS vs DOCS_AND_FREQS_AND_POSITIONS.
    let record_option = if with_position {
        IndexRecordOption::WithFreqsAndPositions
    } else {
        IndexRecordOption::WithFreqs
    };
    let indexing = TextFieldIndexing::default()
        .set_tokenizer(WS_TOKENIZER)
        .set_index_option(record_option);
    let text_opts = TextOptions::default().set_indexing_options(indexing);
    let text_field = sb.add_text_field(TEXT_COL, text_opts);
    (sb.build(), id_field, text_field)
}

/// Build the run's analyzer and the schema. Run A registers a whitespace
/// tokenizer (no lowercase) under `WS_TOKENIZER`; Run B registers the native
/// `default` pipeline (simple split + lowercase) under the same name so the
/// schema is identical and only the analyzer differs.
fn register_tokenizer(index: &Index, run: char) {
    let analyzer: TextAnalyzer = if run == 'a' {
        TextAnalyzer::builder(WhitespaceTokenizer::default()).build()
    } else {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .build()
    };
    index.tokenizers().register(WS_TOKENIZER, analyzer);
}

/// Tokenize a query string with the run's analyzer (so query tokens match the
/// indexed tokens exactly).
fn analyze(index: &Index, text: &str) -> Vec<String> {
    let mut analyzer = index.tokenizers().get(WS_TOKENIZER).unwrap();
    let mut out = Vec::new();
    let mut ts = analyzer.token_stream(text);
    while ts.advance() {
        out.push(ts.token().text.clone());
    }
    out
}

fn build_query(kind: &str, tokens: &[String], text_field: Field) -> Box<dyn Query> {
    if tokens.is_empty() {
        // Matches nothing — keeps the query set aligned across impls.
        return Box::new(TermQuery::new(
            Term::from_field_text(text_field, "__no_such_token__"),
            IndexRecordOption::Basic,
        ));
    }
    if kind == "phrase" {
        let terms: Vec<Term> = tokens
            .iter()
            .map(|t| Term::from_field_text(text_field, t))
            .collect();
        if terms.len() == 1 {
            return Box::new(TermQuery::new(
                terms[0].clone(),
                IndexRecordOption::WithFreqs,
            ));
        }
        return Box::new(PhraseQuery::new(terms));
    }
    if tokens.len() == 1 {
        return Box::new(TermQuery::new(
            Term::from_field_text(text_field, &tokens[0]),
            IndexRecordOption::WithFreqs,
        ));
    }
    // term-with-multiple / OR: SHOULD over each token (BM25-summed).
    let clauses: Vec<(Occur, Box<dyn Query>)> = tokens
        .iter()
        .map(|t| {
            let q: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(text_field, t),
                IndexRecordOption::WithFreqs,
            ));
            (Occur::Should, q)
        })
        .collect();
    Box::new(BooleanQuery::new(clauses))
}

fn run(args: &Args) {
    let corpus_file = if args.run == 'a' {
        "corpus_tok.txt"
    } else {
        "corpus.txt"
    };
    let docs = read_lines(&args.in_dir.join(corpus_file));
    let query_lines = read_lines(&args.in_dir.join("queries.txt"));
    let truth_lines = read_lines(&args.in_dir.join("truth.txt"));

    let (schema, id_field, text_field) = build_schema(args.with_position);

    // ---- create index (RAM or mmap) ----
    let on_disk = args.dir.is_some();
    let index = if let Some(dir) = &args.dir {
        fs::create_dir_all(dir).unwrap();
        Index::create_in_dir(dir, schema.clone())
            .or_else(|_| Index::open_in_dir(dir))
            .unwrap_or_else(|e| panic!("create index in {}: {e}", dir.display()))
    } else {
        Index::create_in_ram(schema.clone())
    };
    register_tokenizer(&index, args.run);

    // ---- build ----
    // Large heap so the whole corpus lands in one (or few) segments, matching
    // Lucene's single-commit build; thread count tracks the bench threads.
    let num_threads = if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    };
    let heap = 1usize << 30; // 1 GiB total writer budget
    let build_start = Instant::now();
    let mut writer: IndexWriter = index
        .writer_with_num_threads(num_threads.min(8), heap)
        .unwrap();
    for (id, text) in docs.iter().enumerate() {
        writer
            .add_document(doc!(id_field => id as u64, text_field => text.as_str()))
            .unwrap();
    }
    writer.commit().unwrap();
    let build_s = build_start.elapsed().as_secs_f64();
    drop(writer);

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();

    // ---- index footprint ----
    // `space_usage` is the index's real byte footprint and works for both the
    // RamDirectory and an mmap directory — comparable to Lucene's
    // sum-of-file-lengths. Cross-check against on-disk file sizes when present.
    let mem_bytes: u64 = searcher
        .space_usage()
        .map(|u| u.total().get_bytes())
        .unwrap_or_else(|_| args.dir.as_deref().map(dir_bytes).unwrap_or(0));

    // ---- parse queries ----
    let mut queries: Vec<Q> = Vec::new();
    for l in &query_lines {
        let mut p = l.splitn(3, '\t');
        let (kind, raw, tok) = match (p.next(), p.next(), p.next()) {
            (Some(a), Some(b), Some(c)) => (a.to_string(), b.to_string(), c.to_string()),
            _ => continue,
        };
        // Without positions, phrase search is unsupported — skip phrase queries
        // so every impl measures the same term-only workload.
        if !args.with_position && kind == "phrase" {
            continue;
        }
        queries.push(Q { kind, raw, tok });
    }
    let truth: Vec<HashSet<usize>> = truth_lines
        .iter()
        .map(|l| {
            l.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect()
        })
        .collect();

    let built: Vec<(String, Box<dyn Query>)> = queries
        .iter()
        .map(|q| {
            let text = if args.run == 'a' { &q.tok } else { &q.raw };
            let tokens = analyze(&index, text);
            (q.kind.clone(), build_query(&q.kind, &tokens, text_field))
        })
        .collect();

    let read_id = |doc: &TantivyDocument| -> usize {
        doc.get_first(id_field)
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX) as usize
    };

    // ---- warm-up ----
    for (_, q) in &built {
        let _ = searcher
            .search(q.as_ref(), &TopDocs::with_limit(args.k))
            .unwrap();
    }

    // ---- single-thread latency + top-k ----
    let mut latencies_us: Vec<f64> = Vec::with_capacity(built.len());
    let mut topk: Vec<Vec<usize>> = Vec::with_capacity(built.len());
    let st_start = Instant::now();
    for (_, q) in &built {
        let t0 = Instant::now();
        let hits = searcher
            .search(q.as_ref(), &TopDocs::with_limit(args.k))
            .unwrap();
        latencies_us.push(t0.elapsed().as_secs_f64() * 1.0e6);
        let mut ids = Vec::with_capacity(hits.len());
        for (_score, addr) in hits {
            let d: TantivyDocument = searcher.doc(addr).unwrap();
            ids.push(read_id(&d));
        }
        topk.push(ids);
    }
    let st_s = st_start.elapsed().as_secs_f64();
    let qps_1t = built.len() as f64 / st_s;

    // ---- multi-thread QPS ----
    let reps = 4usize;
    let mt_start = Instant::now();
    let _: usize = (0..reps)
        .into_par_iter()
        .map(|_| {
            built
                .par_iter()
                .map(|(_, q)| {
                    let s = reader.searcher();
                    s.search(q.as_ref(), &TopDocs::with_limit(args.k))
                        .unwrap()
                        .len()
                })
                .sum::<usize>()
        })
        .sum();
    let mt_s = mt_start.elapsed().as_secs_f64();
    let qps_nt = (built.len() * reps) as f64 / mt_s;

    // ---- recall@k vs exact BM25 (Run A only) ----
    let (mut term_recall, mut term_n) = (0.0f64, 0usize);
    let (mut phrase_recall, mut phrase_n) = (0.0f64, 0usize);
    let (mut or_recall, mut or_n) = (0.0f64, 0usize);
    if args.run == 'a' {
        for (i, (kind, _)) in built.iter().enumerate() {
            let t = truth.get(i);
            let t = match t {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };
            let hit = topk[i].iter().filter(|d| t.contains(d)).count() as f64;
            let r = hit / args.k as f64;
            match kind.as_str() {
                "phrase" => {
                    phrase_recall += r;
                    phrase_n += 1;
                }
                "or" => {
                    or_recall += r;
                    or_n += 1;
                }
                _ => {
                    term_recall += r;
                    term_n += 1;
                }
            }
        }
    }
    let term_recall_v = if term_n > 0 {
        term_recall / term_n as f64
    } else {
        f64::NAN
    };
    let phrase_recall_v = if phrase_n > 0 {
        phrase_recall / phrase_n as f64
    } else {
        f64::NAN
    };
    let or_recall_v = if or_n > 0 {
        or_recall / or_n as f64
    } else {
        f64::NAN
    };

    // ---- write top-k for the driver's mutual-overlap step ----
    let topk_path = args
        .in_dir
        .join(format!("tantivy_run{}_topk.txt", args.run));
    let mut tw = BufWriter::new(fs::File::create(&topk_path).unwrap());
    for ids in &topk {
        let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        writeln!(tw, "{}", s.join(" ")).ok();
    }
    tw.flush().ok();

    // ---- per-kind latency split ----
    let mut term_lat = Vec::new();
    let mut phrase_lat = Vec::new();
    let mut or_lat = Vec::new();
    for (i, (kind, _)) in built.iter().enumerate() {
        match kind.as_str() {
            "phrase" => phrase_lat.push(latencies_us[i]),
            "or" => or_lat.push(latencies_us[i]),
            _ => term_lat.push(latencies_us[i]),
        }
    }
    let sortf = |v: &mut Vec<f64>| v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sortf(&mut term_lat);
    sortf(&mut phrase_lat);
    sortf(&mut or_lat);
    println!(
        "result split impl=tantivy term_p50={:.1} term_p95={:.1} ({} q) | phrase_p50={:.1} phrase_p95={:.1} ({} q) | or_p50={:.1} or_p95={:.1} ({} q)",
        percentile(&term_lat, 50.0),
        percentile(&term_lat, 95.0),
        term_lat.len(),
        percentile(&phrase_lat, 50.0),
        percentile(&phrase_lat, 95.0),
        phrase_lat.len(),
        percentile(&or_lat, 50.0),
        percentile(&or_lat, 95.0),
        or_lat.len(),
    );

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mode = if on_disk { "disk" } else { "mem" };
    println!(
        "result impl=tantivy mode={} run={} docs={} queries={} build_s={:.3} build_docs_per_s={:.0} \
         q_p50_us={:.1} q_p95_us={:.1} qps_1t={:.0} qps_nt={:.0} term_recall={:.4} phrase_recall={:.4} index_mb={:.1}",
        mode,
        args.run,
        docs.len(),
        built.len(),
        build_s,
        docs.len() as f64 / build_s,
        percentile(&latencies_us, 50.0),
        percentile(&latencies_us, 95.0),
        qps_1t,
        qps_nt,
        term_recall_v,
        phrase_recall_v,
        mem_bytes as f64 / 1.0e6,
    );
    println!(
        "{{\"impl\":\"tantivy\",\"mode\":\"{}\",\"run\":\"{}\",\"docs\":{},\"queries\":{},\"k\":{},\
         \"build_s\":{:.4},\"build_docs_per_s\":{:.1},\
         \"q_p50_us\":{:.2},\"q_p95_us\":{:.2},\"qps_1t\":{:.1},\"qps_nt\":{:.1},\
         \"term_recall_at_k\":{:.4},\"phrase_recall_at_k\":{:.4},\"or_recall_at_k\":{:.4},\"mem_bytes\":{}}}",
        mode,
        args.run,
        docs.len(),
        built.len(),
        args.k,
        build_s,
        docs.len() as f64 / build_s,
        percentile(&latencies_us, 50.0),
        percentile(&latencies_us, 95.0),
        qps_1t,
        qps_nt,
        term_recall_v,
        phrase_recall_v,
        or_recall_v,
        mem_bytes,
    );
}
