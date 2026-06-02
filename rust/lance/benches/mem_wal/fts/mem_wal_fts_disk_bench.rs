// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! On-disk Lance FTS bench — the on-disk arm of the FTS comparison (paired with
//! Lucene `FSDirectory` and Tantivy `MmapDirectory`).
//!
//! Unlike `mem_wal_fts_bench` (which queries the in-memory `FtsMemIndex`), this
//! bench exercises the *actual flushed memtable*: it writes the corpus through
//! the `ShardWriter`, force-seals so all rows flush into a single on-disk
//! generation (one Lance fragment + one standard `InvertedIndex`), reopens that
//! generation as a `Dataset`, and runs FTS queries through the normal scan path
//! (`scan().full_text_search()`). This is the honest on-disk number, DataFusion
//! planning overhead included.
//!
//! Reads the shared inputs produced by `mem_wal_fts_bench gen`
//! (corpus.txt / corpus_tok.txt / queries.txt / truth.txt) so the corpus,
//! queries and exact-BM25 truth are bit-identical to the Lucene/Tantivy runs.
//!
//! Run A (`--run a`): pre-tokenized `corpus_tok.txt` + whitespace tokenizer.
//! Run B (`--run b`): raw `corpus.txt` + the default Lance tokenizer.
//!
//! Emits one JSON line tagged `impl=lance_disk`, the same shape as the other
//! benches plus `fts_index_bytes` / `total_gen_bytes` index-size fields.
//!
//! Usage:
//!   mem_wal_fts_disk_bench --in-dir DIR --run a|b --k 10 --threads 64 \
//!       --data-dir /mnt/nvme/lance_fts [--no-positions]

#![recursion_limit = "256"]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::stream::{FuturesUnordered, StreamExt};
use lance::dataset::mem_wal::index::{FtsIndexConfig, MemIndexConfig};
use lance::dataset::mem_wal::write::{ShardWriter, ShardWriterConfig};
use lance::Dataset;
use lance_core::Result;
use lance_index::scalar::inverted::query::PhraseQuery;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::FullTextSearchQuery;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use uuid::Uuid;

const TEXT_COL: &str = "text";

struct Args {
    in_dir: PathBuf,
    run: char,
    k: usize,
    threads: usize,
    data_dir: PathBuf,
    with_position: bool,
}

fn read_lines(path: &FsPath) -> Vec<String> {
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

fn dir_bytes(dir: &FsPath) -> u64 {
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

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(TEXT_COL, DataType::Utf8, true),
    ]))
}

struct Q {
    kind: String,
    raw: String,
    tok: String,
}

/// Build the scan-level FTS query for one parsed query line.
fn make_query(q: &Q, run: char, k: usize) -> Result<FullTextSearchQuery> {
    let text = if run == 'a' { &q.tok } else { &q.raw };
    let fts = if q.kind == "phrase" {
        FullTextSearchQuery::new_query(PhraseQuery::new(text.clone()).into())
            .with_column(TEXT_COL.to_string())?
    } else {
        // term (1 token) and OR (multi-token) are both MatchQuery — the
        // multi-token form is a BM25-summed SHOULD/OR, matching the other benches.
        FullTextSearchQuery::new(text.clone()).with_column(TEXT_COL.to_string())?
    };
    Ok(fts.limit(Some(k as i64)))
}

/// Run one FTS query against the flushed dataset, returning the top-k doc ids.
async fn fts_topk(dataset: &Dataset, query: FullTextSearchQuery, k: usize) -> Result<Vec<usize>> {
    let mut scanner = dataset.scan();
    scanner.full_text_search(query)?;
    scanner.project(&["id"])?;
    scanner.limit(Some(k as i64), None)?;
    let batch = scanner.try_into_batch().await?;
    let ids = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .map(|a| (0..a.len()).map(|i| a.value(i) as usize).collect::<Vec<_>>())
        .unwrap_or_default();
    Ok(ids)
}

async fn build_and_open(
    args: &Args,
    docs: &[String],
) -> Result<(Arc<Dataset>, f64, u64, u64)> {
    // Fresh data dir per run so the single-generation invariant holds.
    let _ = fs::remove_dir_all(&args.data_dir);
    fs::create_dir_all(&args.data_dir)
        .map_err(|e| lance_core::Error::io(format!("create data dir: {e}")))?;
    let abs = fs::canonicalize(&args.data_dir)
        .map_err(|e| lance_core::Error::io(format!("canonicalize data dir: {e}")))?;
    let base_uri = format!("file://{}", abs.display());
    let (store, base_path): (Arc<ObjectStore>, Path) = ObjectStore::from_uri(&base_uri).await?;

    let params = if args.run == 'a' {
        InvertedIndexParams::default()
            .base_tokenizer("whitespace".to_string())
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .with_position(args.with_position)
    } else {
        InvertedIndexParams::default().with_position(args.with_position)
    };
    // field_id 1 = the "text" column (id is field 0).
    let index_configs = vec![MemIndexConfig::Fts(FtsIndexConfig::with_params(
        "text_fts".to_string(),
        1,
        TEXT_COL.to_string(),
        params,
    ))];

    let shard_id = Uuid::new_v4();
    // Thresholds sized to the whole corpus so the writer never auto-flushes
    // mid-run; we force-seal once at the end to land everything in a single
    // generation. They must stay finite: `max_memtable_batches` pre-allocates
    // the batch store and `is_batch_store_full()` flushes at that count, so we
    // give it the exact batch count plus headroom rather than a huge sentinel.
    let batch_size = 1000usize;
    let max_batches = docs.len() / batch_size + 64;
    let config = ShardWriterConfig::new(shard_id)
        .with_durable_write(false)
        .with_sync_indexed_write(true)
        .with_max_memtable_size(512 * 1024 * 1024 * 1024) // 512 GiB: never triggers
        .with_max_memtable_rows(docs.len().max(1))
        .with_max_memtable_batches(max_batches);

    let sch = schema();
    let writer =
        ShardWriter::open(store, base_path, base_uri.clone(), config, sch.clone(), index_configs)
            .await?;

    let build_start = Instant::now();
    let mut row = 0usize;
    while row < docs.len() {
        let end = (row + batch_size).min(docs.len());
        let ids: Vec<i64> = (row as i64..end as i64).collect();
        let texts: Vec<&str> = docs[row..end].iter().map(|s| s.as_str()).collect();
        let batch = RecordBatch::try_new(
            sch.clone(),
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(texts)) as ArrayRef,
            ],
        )
        .map_err(|e| lance_core::Error::io(format!("batch: {e}")))?;
        writer.put(vec![batch]).await?;
        row = end;
    }
    // Flush the memtable to a single on-disk generation (the "commit" analogue;
    // its FTS index build is counted in build time like Lucene/Tantivy commit).
    writer.force_seal_active().await?;
    writer.wait_for_flush_drain().await?;
    let build_s = build_start.elapsed().as_secs_f64();

    let manifest = writer
        .manifest()
        .await?
        .ok_or_else(|| lance_core::Error::io("no manifest after flush"))?;
    if manifest.flushed_generations.len() != 1 {
        return Err(lance_core::Error::io(format!(
            "expected exactly one flushed generation, got {}",
            manifest.flushed_generations.len()
        )));
    }
    let gen_rel = &manifest.flushed_generations[0].path;
    let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, gen_rel);
    let gen_fs = abs.join("_mem_wal").join(shard_id.to_string()).join(gen_rel);
    let total_gen_bytes = dir_bytes(&gen_fs);
    let fts_index_bytes = dir_bytes(&gen_fs.join("_indices"));

    let dataset = Arc::new(Dataset::open(&gen_uri).await?);
    writer.close().await?;
    Ok((dataset, build_s, fts_index_bytes, total_gen_bytes))
}

fn run(args: &Args) -> Result<()> {
    let corpus_file = if args.run == 'a' { "corpus_tok.txt" } else { "corpus.txt" };
    let docs = read_lines(&args.in_dir.join(corpus_file));
    let query_lines = read_lines(&args.in_dir.join("queries.txt"));
    let truth_lines = read_lines(&args.in_dir.join("truth.txt"));

    let nthreads = if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(nthreads)
        .enable_all()
        .build()
        .map_err(|e| lance_core::Error::io(format!("runtime: {e}")))?;

    let (dataset, build_s, fts_index_bytes, total_gen_bytes) =
        rt.block_on(build_and_open(args, &docs))?;

    // Parse queries (drop phrase under --no-positions).
    let mut queries: Vec<Q> = Vec::new();
    for l in &query_lines {
        let mut p = l.splitn(3, '\t');
        let (kind, raw, tok) = match (p.next(), p.next(), p.next()) {
            (Some(a), Some(b), Some(c)) => (a.to_string(), b.to_string(), c.to_string()),
            _ => continue,
        };
        if !args.with_position && kind == "phrase" {
            continue;
        }
        queries.push(Q { kind, raw, tok });
    }
    let truth: Vec<HashSet<usize>> = truth_lines
        .iter()
        .map(|l| l.split_whitespace().filter_map(|s| s.parse().ok()).collect())
        .collect();

    // Warm-up.
    rt.block_on(async {
        for q in &queries {
            let fq = make_query(q, args.run, args.k)?;
            let _ = fts_topk(&dataset, fq, args.k).await?;
        }
        Result::Ok(())
    })?;

    // Single-thread latency + top-k.
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut topk: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
    rt.block_on(async {
        for q in &queries {
            let fq = make_query(q, args.run, args.k)?;
            let t0 = Instant::now();
            let ids = fts_topk(&dataset, fq, args.k).await?;
            latencies_us.push(t0.elapsed().as_secs_f64() * 1.0e6);
            topk.push(ids);
        }
        Result::Ok(())
    })?;
    let st_s: f64 = latencies_us.iter().sum::<f64>() / 1.0e6;
    let qps_1t = queries.len() as f64 / st_s.max(1e-9);

    // Multi-thread QPS: fan out queries x reps as concurrent tasks.
    let reps = 4usize;
    let mt_start = Instant::now();
    let total = rt.block_on(async {
        let mut futs = FuturesUnordered::new();
        for _ in 0..reps {
            for q in &queries {
                let fq = make_query(q, args.run, args.k)?;
                let ds = dataset.clone();
                let k = args.k;
                futs.push(async move { fts_topk(&ds, fq, k).await.map(|v| v.len()) });
            }
        }
        let mut sum = 0usize;
        while let Some(r) = futs.next().await {
            sum += r?;
        }
        Result::Ok(sum)
    })?;
    let _ = total;
    let mt_s = mt_start.elapsed().as_secs_f64();
    let qps_nt = (queries.len() * reps) as f64 / mt_s.max(1e-9);

    // recall@k vs exact BM25 (Run A only).
    let (mut term_recall, mut term_n) = (0.0f64, 0usize);
    let (mut phrase_recall, mut phrase_n) = (0.0f64, 0usize);
    let (mut or_recall, mut or_n) = (0.0f64, 0usize);
    if args.run == 'a' {
        for (i, q) in queries.iter().enumerate() {
            let t = match truth.get(i) {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };
            let hit = topk[i].iter().filter(|d| t.contains(d)).count() as f64;
            let r = hit / args.k as f64;
            match q.kind.as_str() {
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
    let term_recall_v = if term_n > 0 { term_recall / term_n as f64 } else { f64::NAN };
    let phrase_recall_v = if phrase_n > 0 { phrase_recall / phrase_n as f64 } else { f64::NAN };
    let or_recall_v = if or_n > 0 { or_recall / or_n as f64 } else { f64::NAN };

    // Write top-k for the driver's mutual-overlap step.
    let topk_path = args.in_dir.join(format!("lance_disk_run{}_topk.txt", args.run));
    let mut tw = BufWriter::new(
        fs::File::create(&topk_path)
            .map_err(|e| lance_core::Error::io(format!("create topk: {e}")))?,
    );
    for ids in &topk {
        let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        writeln!(tw, "{}", s.join(" ")).ok();
    }
    tw.flush().ok();

    // Per-kind latency split.
    let mut term_lat = Vec::new();
    let mut phrase_lat = Vec::new();
    let mut or_lat = Vec::new();
    for (i, q) in queries.iter().enumerate() {
        match q.kind.as_str() {
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
        "result split impl=lance_disk term_p50={:.1} term_p95={:.1} ({} q) | phrase_p50={:.1} phrase_p95={:.1} ({} q) | or_p50={:.1} or_p95={:.1} ({} q)",
        percentile(&term_lat, 50.0), percentile(&term_lat, 95.0), term_lat.len(),
        percentile(&phrase_lat, 50.0), percentile(&phrase_lat, 95.0), phrase_lat.len(),
        percentile(&or_lat, 50.0), percentile(&or_lat, 95.0), or_lat.len(),
    );

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "result impl=lance_disk mode=disk run={} docs={} queries={} build_s={:.3} build_docs_per_s={:.0} \
         q_p50_us={:.1} q_p95_us={:.1} qps_1t={:.0} qps_nt={:.0} term_recall={:.4} phrase_recall={:.4} \
         fts_index_mb={:.1} total_gen_mb={:.1}",
        args.run, docs.len(), queries.len(), build_s, docs.len() as f64 / build_s,
        percentile(&latencies_us, 50.0), percentile(&latencies_us, 95.0),
        qps_1t, qps_nt, term_recall_v, phrase_recall_v,
        fts_index_bytes as f64 / 1.0e6, total_gen_bytes as f64 / 1.0e6,
    );
    println!(
        "{{\"impl\":\"lance_disk\",\"mode\":\"disk\",\"run\":\"{}\",\"docs\":{},\"queries\":{},\"k\":{},\
         \"build_s\":{:.4},\"build_docs_per_s\":{:.1},\
         \"q_p50_us\":{:.2},\"q_p95_us\":{:.2},\"qps_1t\":{:.1},\"qps_nt\":{:.1},\
         \"term_recall_at_k\":{:.4},\"phrase_recall_at_k\":{:.4},\"or_recall_at_k\":{:.4},\
         \"mem_bytes\":{},\"fts_index_bytes\":{},\"total_gen_bytes\":{}}}",
        args.run, docs.len(), queries.len(), args.k,
        build_s, docs.len() as f64 / build_s,
        percentile(&latencies_us, 50.0), percentile(&latencies_us, 95.0),
        qps_1t, qps_nt, term_recall_v, phrase_recall_v, or_recall_v,
        fts_index_bytes, fts_index_bytes, total_gen_bytes,
    );
    Ok(())
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let argv: Vec<&str> = argv.iter().map(|s| s.as_str()).filter(|s| *s != "--bench").collect();
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
        data_dir: get("--data-dir", "/tmp/lance_fts_disk").into(),
        with_position: !argv.contains(&"--no-positions"),
    };
    run(&args)
}
