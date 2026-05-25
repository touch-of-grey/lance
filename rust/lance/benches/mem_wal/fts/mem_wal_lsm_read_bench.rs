// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Standalone CLI benchmark for FTS read across LSM levels.
//!
//! Sibling of `mem_wal_vector_bench.rs` / `mem_wal_point_lookup_bench.rs`:
//! same `--phase prepare|search` shape, same `ShardWriter`-based ingestion
//! of flushed generations + an active memtable, same `--uri` cloud/local
//! detection, and the same JSON output contract. The payload is real
//! HuggingFace FineWeb `text` and the query path is
//! [`LsmFtsSearchPlanner`] over the base table + flushed generations +
//! active memtable.
//!
//! The "panel" for FTS is the scoring mode: each invocation runs the same
//! query set under both [`FtsScoringMode::Local`] and
//! [`FtsScoringMode::LocalWithGlobalRescore`] and reports per-mode latency
//! plus the top-k Jaccard between the two (how much rescoring moves the
//! ranking).
//!
//! Two phases, selected with `--phase`:
//!
//!   --phase prepare   Load FineWeb text, write the base dataset, create an
//!                     inverted (FTS) index, and initialize MemWAL with the
//!                     index maintained.
//!   --phase search    Ingest rows across LSM levels via ShardWriter, then run
//!                     the FTS query panel under both scoring modes.
//!
//! Example:
//!
//! ```bash
//! cargo bench -p lance --bench mem_wal_fts_read_bench -- \
//!   --phase prepare --uri /tmp/fts_read_bench \
//!   --base-rows 1000000 --cache-dir /tmp/fineweb-cache
//!
//! cargo bench -p lance --bench mem_wal_fts_read_bench -- \
//!   --phase search --uri /tmp/fts_read_bench \
//!   --base-rows 1000000 --max-memtable-rows 100000 \
//!   --queries 200 --k 10 --rescore-factor 10 \
//!   --cache-dir /tmp/fineweb-cache --output result.json
//! ```

#![recursion_limit = "256"]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::mem_wal::scanner::{
    FlushedMemTableCache, FtsScoringMode, LsmDataSourceCollector, LsmFtsSearchPlanner,
    LsmPointLookupPlanner, LsmVectorSearchPlanner, ShardSnapshot,
};
use lance::dataset::mem_wal::{DatasetMemWalExt, ShardWriterConfig};
use lance::dataset::{DEFAULT_METADATA_CACHE_SIZE, Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance::session::Session;
use lance_core::Result;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_io::object_store::ObjectStoreRegistry;
use lance_linalg::distance::DistanceType;
use lance_tokenizer::TokenStream;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use serde_json::json;
use uuid::Uuid;

const TEXT_COL: &str = "text";
const VECTOR_COL: &str = "vector";
const FTS_INDEX_NAME: &str = "text_fts";
const BTREE_INDEX_NAME: &str = "id_btree";
const VEC_INDEX_NAME: &str = "vec_ivfrq";
const HF_API_LISTING: &str =
    "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT";
const HF_FILE_BASE: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/";

// ----------------------------------------------------------------------
// Phase / Args
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prepare,
    Search,
}

impl Phase {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "prepare" => Ok(Self::Prepare),
            "search" => Ok(Self::Search),
            _ => Err(format!("unknown phase '{value}', expected prepare|search")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Prepare => "prepare",
            Self::Search => "search",
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    phase: Phase,
    uri: String,
    base_rows: usize,
    max_memtable_rows: usize,
    flushed_generations: usize,
    batch_rows: usize,
    queries: usize,
    k: usize,
    /// Top-k values to sweep within a single ingest. Defaults to `[k]`.
    k_values: Vec<usize>,
    rescore_factor: u32,
    vector_dim: usize,
    ivf_partitions: usize,
    rq_bits: u8,
    nprobes: usize,
    /// Shared index-cache size in GiB for the base + flushed-gen session.
    /// The default 6 GiB overflows once ~5 flushed FTS layers are in scope,
    /// thrashing the inverted-index postings; size it to hold the working set.
    index_cache_gb: usize,
    cache_dir: PathBuf,
    output: Option<PathBuf>,
    /// Directory + tag for per-k outputs: `<output_dir>/search_<tag>_k<K>.json`.
    output_dir: Option<PathBuf>,
    output_tag: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            phase: Phase::Search,
            uri: String::new(),
            base_rows: 1_000_000,
            max_memtable_rows: 100_000,
            flushed_generations: 2,
            batch_rows: 1_000,
            queries: 200,
            k: 10,
            k_values: Vec::new(),
            rescore_factor: 10,
            vector_dim: 128,
            ivf_partitions: 1024,
            rq_bits: 8,
            nprobes: 16,
            index_cache_gb: 32,
            cache_dir: std::env::temp_dir().join("mem_wal_fineweb_fts_cache"),
            output: None,
            output_dir: None,
            output_tag: None,
        }
    }
}

fn parse_val<T>(flag: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|e| lance_core::Error::invalid_input(format!("invalid {flag}: {value} ({e})")))
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    let mut has_phase = false;
    let mut has_uri = false;
    while let Some(flag) = iter.next() {
        if flag == "--bench" {
            continue;
        }
        let value = iter
            .next()
            .ok_or_else(|| lance_core::Error::invalid_input(format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--phase" => {
                args.phase = Phase::parse(&value).map_err(lance_core::Error::invalid_input)?;
                has_phase = true;
            }
            "--uri" => {
                args.uri = value;
                has_uri = true;
            }
            "--base-rows" => args.base_rows = parse_val(&flag, &value)?,
            "--max-memtable-rows" => args.max_memtable_rows = parse_val(&flag, &value)?,
            "--flushed-generations" => args.flushed_generations = parse_val(&flag, &value)?,
            "--batch-rows" => args.batch_rows = parse_val(&flag, &value)?,
            "--queries" => args.queries = parse_val(&flag, &value)?,
            "--k" => args.k = parse_val(&flag, &value)?,
            "--k-list" => {
                args.k_values = value
                    .split(',')
                    .map(|s| parse_val::<usize>(&flag, s.trim()))
                    .collect::<Result<Vec<_>>>()?;
            }
            "--rescore-factor" => args.rescore_factor = parse_val(&flag, &value)?,
            "--vector-dim" => args.vector_dim = parse_val(&flag, &value)?,
            "--ivf-partitions" => args.ivf_partitions = parse_val(&flag, &value)?,
            "--rq-bits" => args.rq_bits = parse_val(&flag, &value)?,
            "--nprobes" => args.nprobes = parse_val(&flag, &value)?,
            "--index-cache-gb" => args.index_cache_gb = parse_val(&flag, &value)?,
            "--cache-dir" => args.cache_dir = PathBuf::from(value),
            "--output" => args.output = Some(PathBuf::from(value)),
            "--output-dir" => args.output_dir = Some(PathBuf::from(value)),
            "--tag" => args.output_tag = Some(value),
            _ => {
                return Err(lance_core::Error::invalid_input(format!(
                    "unknown argument: {flag}"
                )));
            }
        }
    }
    if !has_phase {
        return Err(lance_core::Error::invalid_input(
            "--phase is required (prepare|search)",
        ));
    }
    if !has_uri {
        return Err(lance_core::Error::invalid_input("--uri is required"));
    }
    if args.batch_rows == 0 || args.base_rows == 0 || args.max_memtable_rows == 0 {
        return Err(lance_core::Error::invalid_input(
            "base-rows, max-memtable-rows, batch-rows must be > 0",
        ));
    }
    if args.k_values.is_empty() {
        args.k_values = vec![args.k];
    }
    Ok(args)
}

fn is_cloud_uri(uri: &str) -> bool {
    uri.starts_with("s3://") || uri.starts_with("gs://") || uri.starts_with("az://")
}

// ----------------------------------------------------------------------
// FineWeb loading (mirrors mem_wal_fineweb_fts.rs)
// ----------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
}

async fn list_shard_paths() -> Result<Vec<String>> {
    let entries: Vec<HfTreeEntry> = reqwest::get(HF_API_LISTING)
        .await
        .map_err(|e| lance_core::Error::io(format!("listing HTTP: {e}")))?
        .json()
        .await
        .map_err(|e| lance_core::Error::io(format!("listing JSON: {e}")))?;
    let mut shards: Vec<String> = entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.ends_with(".parquet"))
        .map(|e| e.path)
        .collect();
    shards.sort();
    Ok(shards)
}

async fn download_shard(rel_path: &str, dest: &std::path::Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    let url = format!("{HF_FILE_BASE}{rel_path}");
    let tmp = dest.with_extension("part");
    for attempt in 1..=5u32 {
        println!("downloading {rel_path} (attempt {attempt}/5) ...");
        let result: Result<bytes::Bytes> = async {
            let resp = reqwest::get(&url)
                .await
                .map_err(|e| lance_core::Error::io(format!("download HTTP: {e}")))?;
            if !resp.status().is_success() {
                return Err(lance_core::Error::io(format!(
                    "download {url} -> status {}",
                    resp.status()
                )));
            }
            resp.bytes()
                .await
                .map_err(|e| lance_core::Error::io(format!("read body: {e}")))
        }
        .await;
        match result {
            Ok(bytes) => {
                std::fs::write(&tmp, &bytes)
                    .map_err(|e| lance_core::Error::io(format!("write: {e}")))?;
                std::fs::rename(&tmp, dest)
                    .map_err(|e| lance_core::Error::io(format!("rename: {e}")))?;
                return Ok(());
            }
            Err(e) if attempt < 5 => {
                eprintln!("  attempt {attempt} failed: {e}; retrying");
                tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

async fn read_shard_text(
    path: &std::path::Path,
    out: &mut Vec<String>,
    max_rows: usize,
) -> Result<usize> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| lance_core::Error::io(format!("open parquet: {e}")))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|e| lance_core::Error::io(format!("parquet builder: {e}")))?;
    let mut stream = builder
        .build()
        .map_err(|e| lance_core::Error::io(format!("parquet stream: {e}")))?;
    let mut taken = 0usize;
    while taken < max_rows {
        let Some(rb) = stream
            .try_next()
            .await
            .map_err(|e| lance_core::Error::io(format!("parquet read: {e}")))?
        else {
            break;
        };
        let col = rb
            .column_by_name(TEXT_COL)
            .ok_or_else(|| lance_core::Error::io("text column missing".to_string()))?;
        let strs = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| lance_core::Error::io("text not StringArray".to_string()))?;
        for i in 0..strs.len() {
            if taken >= max_rows {
                break;
            }
            if strs.is_null(i) {
                continue;
            }
            out.push(strs.value(i).to_string());
            taken += 1;
        }
    }
    Ok(taken)
}

async fn load_corpus(needed_rows: usize, cache_dir: &std::path::Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| lance_core::Error::io(format!("mkdir cache: {e}")))?;
    let shards = list_shard_paths().await?;
    println!("fineweb sample/10BT: {} shards", shards.len());
    let mut buf: Vec<String> = Vec::with_capacity(needed_rows);
    for rel in &shards {
        if buf.len() >= needed_rows {
            break;
        }
        let name = rel.rsplit('/').next().unwrap_or(rel);
        let local = cache_dir.join(name);
        download_shard(rel, &local).await?;
        let want = needed_rows - buf.len();
        let got = read_shard_text(&local, &mut buf, want).await?;
        println!("  shard {name} -> {got} rows (cumulative {})", buf.len());
    }
    if buf.len() < needed_rows {
        return Err(lance_core::Error::io(format!(
            "fineweb yielded only {} rows, need {needed_rows}",
            buf.len()
        )));
    }
    Ok(buf)
}

// ----------------------------------------------------------------------
// Schema / batch helpers
// ----------------------------------------------------------------------

fn make_schema(vector_dim: usize) -> Arc<ArrowSchema> {
    let mut id_meta = HashMap::new();
    id_meta.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false).with_metadata(id_meta),
        Field::new(TEXT_COL, DataType::Utf8, true),
        Field::new(
            VECTOR_COL,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dim as i32,
            ),
            false,
        ),
    ]))
}

/// Deterministic pseudo-random unit-ish vector for `id`, clustered so IVF has
/// real structure (rows in the same cluster share a base direction).
fn gen_vector(id: i64, dim: usize) -> Vec<f32> {
    let cluster = (id as u64) % 4096;
    let mut state = (id as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    (0..dim)
        .map(|d| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            let base = ((cluster.wrapping_mul(1_103_515_245).wrapping_add(d as u64)) & 0xffff)
                as f32
                / 65_536.0;
            base + ((z & 0xffff) as f32 / 65_536.0) * 0.05
        })
        .collect()
}

fn make_batch(
    schema: Arc<ArrowSchema>,
    start_id: i64,
    texts: &[String],
    dim: usize,
) -> RecordBatch {
    let n = texts.len();
    let ids = Int64Array::from_iter_values(start_id..start_id + n as i64);
    let text = StringArray::from_iter_values(texts.iter().map(String::as_str));
    let mut vb = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for i in 0..n {
        let v = gen_vector(start_id + i as i64, dim);
        vb.values().append_slice(&v);
        vb.append(true);
    }
    RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(text), Arc::new(vb.finish())],
    )
    .unwrap()
}

fn wrap_query(values: &[f32], dim: usize) -> FixedSizeListArray {
    let mut b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    b.values().append_slice(values);
    b.append(true);
    b.finish()
}

// ----------------------------------------------------------------------
// Latency stats
// ----------------------------------------------------------------------

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((pct / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct LatencyStats {
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    mean_us: f64,
    qps: f64,
}

fn compute_stats(mut latencies_us: Vec<f64>) -> LatencyStats {
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = latencies_us.iter().sum::<f64>() / latencies_us.len().max(1) as f64;
    let total_s = latencies_us.iter().sum::<f64>() / 1_000_000.0;
    let qps = if total_s > 0.0 {
        latencies_us.len() as f64 / total_s
    } else {
        0.0
    };
    LatencyStats {
        p50_us: percentile(&latencies_us, 50.0) as u64,
        p95_us: percentile(&latencies_us, 95.0) as u64,
        p99_us: percentile(&latencies_us, 99.0) as u64,
        mean_us: mean,
        qps,
    }
}

// ----------------------------------------------------------------------
// Query set: mid-frequency single terms from the corpus
// ----------------------------------------------------------------------

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by", "is", "was",
    "are", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "but",
    "not", "no", "if", "then", "than", "so", "do", "does", "did", "have", "has", "had", "will",
    "would", "should", "could", "can", "may", "might", "must", "i", "you", "he", "she", "we",
    "they", "them", "his", "her", "their", "our", "us", "me", "my", "your", "him", "at", "from",
];

fn build_query_terms(sample: &[String], n: usize) -> Vec<String> {
    let mut tokenizer = InvertedIndexParams::default()
        .build()
        .expect("default tokenizer builds");
    let mut freq: HashMap<String, u64> = HashMap::new();
    for t in sample.iter().take(50_000) {
        let mut stream = tokenizer.token_stream_for_doc(t);
        while let Some(tok) = stream.next() {
            if tok.text.len() < 3 || tok.text.len() > 24 || STOPWORDS.contains(&tok.text.as_str()) {
                continue;
            }
            *freq.entry(tok.text.clone()).or_default() += 1;
        }
    }
    let mut by_freq: Vec<(String, u64)> = freq.into_iter().collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    // Skip the most-frequent tokens (near-ties in BM25), keep mid-frequency.
    let skip = (by_freq.len() / 4).min(300);
    by_freq
        .into_iter()
        .skip(skip)
        .map(|(t, _)| t)
        .take(n)
        .collect()
}

// ----------------------------------------------------------------------
// Prepare phase
// ----------------------------------------------------------------------

async fn run_prepare(args: &Args) -> Result<()> {
    let start = Instant::now();
    let corpus = load_corpus(args.base_rows, &args.cache_dir).await?;
    let schema = make_schema(args.vector_dim);

    let total_batches = corpus.len().div_ceil(args.batch_rows);
    let mut batches = Vec::with_capacity(total_batches);
    let mut lo = 0usize;
    while lo < corpus.len() {
        let hi = (lo + args.batch_rows).min(corpus.len());
        batches.push(Ok(make_batch(
            schema.clone(),
            lo as i64,
            &corpus[lo..hi],
            args.vector_dim,
        )));
        lo = hi;
    }
    let reader = RecordBatchIterator::new(batches.into_iter(), schema.clone());
    let write_start = Instant::now();
    // Pin the base table to Lance file format v2.2 (matching the flushed
    // generations) so the whole LSM read benchmark is v2.2 end to end.
    let write_params = WriteParams {
        data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &args.uri, Some(write_params)).await?;
    println!(
        "wrote {} base rows in {:.1}s",
        args.base_rows,
        write_start.elapsed().as_secs_f64()
    );

    let index_start = Instant::now();
    dataset
        .create_index(
            &[TEXT_COL],
            IndexType::Inverted,
            Some(FTS_INDEX_NAME.to_string()),
            &InvertedIndexParams::default(),
            true,
        )
        .await?;
    println!(
        "created FTS index in {:.1}s",
        index_start.elapsed().as_secs_f64()
    );

    let bt = Instant::now();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some(BTREE_INDEX_NAME.to_string()),
            &lance_index::scalar::ScalarIndexParams::default(),
            true,
        )
        .await?;
    println!("created BTree index in {:.1}s", bt.elapsed().as_secs_f64());

    let vt = Instant::now();
    let vparams =
        VectorIndexParams::ivf_rq(args.ivf_partitions, args.rq_bits, DistanceType::Cosine);
    dataset
        .create_index(
            &[VECTOR_COL],
            IndexType::IvfRq,
            Some(VEC_INDEX_NAME.to_string()),
            &vparams,
            true,
        )
        .await?;
    println!(
        "created IVF-RQ index in {:.1}s (partitions={}, bits={})",
        vt.elapsed().as_secs_f64(),
        args.ivf_partitions,
        args.rq_bits
    );

    dataset
        .initialize_mem_wal()
        .maintained_indexes([BTREE_INDEX_NAME, VEC_INDEX_NAME, FTS_INDEX_NAME])
        .execute()
        .await?;
    println!(
        "prepare complete in {:.1}s: uri={}",
        start.elapsed().as_secs_f64(),
        args.uri
    );
    Ok(())
}

// ----------------------------------------------------------------------
// Search phase
// ----------------------------------------------------------------------

/// Result row ids per query for one mode + the latency distribution.
struct ModeRun {
    top_ids: Vec<HashSet<i64>>,
    latencies_us: Vec<f64>,
}

async fn run_mode(
    planner: &LsmFtsSearchPlanner,
    mode: FtsScoringMode,
    queries: &[String],
    k: usize,
) -> Result<ModeRun> {
    let ctx = SessionContext::new();
    let mut top_ids = Vec::with_capacity(queries.len());
    let mut latencies_us = Vec::with_capacity(queries.len());
    for q in queries {
        let t0 = Instant::now();
        let plan = planner
            .plan_search(TEXT_COL, FullTextSearchQuery::new(q.clone()), k, None, mode)
            .await?;
        let stream = plan.execute(0, ctx.task_ctx())?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        latencies_us.push(t0.elapsed().as_micros() as f64);

        let mut ids: HashSet<i64> = HashSet::new();
        for b in &batches {
            if let Some(col) = b
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            {
                for i in 0..col.len() {
                    ids.insert(col.value(i));
                }
            }
        }
        top_ids.push(ids);
    }
    Ok(ModeRun {
        top_ids,
        latencies_us,
    })
}

fn mean_jaccard(a: &[HashSet<i64>], b: &[HashSet<i64>]) -> f64 {
    let pairs: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(x, y)| {
            if x.is_empty() && y.is_empty() {
                None
            } else {
                let inter = x.intersection(y).count() as f64;
                let union = x.union(y).count() as f64;
                Some(inter / union)
            }
        })
        .collect();
    if pairs.is_empty() {
        0.0
    } else {
        pairs.iter().sum::<f64>() / pairs.len() as f64
    }
}

async fn run_search(args: &Args) -> Result<Vec<(usize, serde_json::Value)>> {
    // Open the base with an explicitly sized index cache; the same session is
    // threaded into every flushed-gen open, so all sources share one cache.
    // The 6 GiB default overflows once ~5 flushed FTS layers are resident.
    let session = Arc::new(Session::new(
        args.index_cache_gb << 30,
        DEFAULT_METADATA_CACHE_SIZE,
        Arc::new(ObjectStoreRegistry::default()),
    ));
    println!("index cache: {} GiB", args.index_cache_gb);
    let dataset = Arc::new(
        DatasetBuilder::from_uri(&args.uri)
            .with_session(session)
            .load()
            .await?,
    );
    let arrow_schema: Arc<ArrowSchema> = Arc::new(ArrowSchema::from(dataset.schema()));
    let schema = make_schema(args.vector_dim);

    // Memtable text is drawn from FineWeb; row *ids* are assigned past the
    // base slice (via `id_base`) so they don't collide with base-table ids,
    // but the *text content* can reuse FineWeb rows freely — BM25 latency
    // doesn't depend on content novelty. Load just enough rows once,
    // covering both the memtable payload and the query-term sample, instead
    // of re-reading the whole base corpus from parquet.
    let active_rows = args.max_memtable_rows / 2;
    let total_memtable_rows = args.flushed_generations * args.max_memtable_rows + active_rows;
    let sample_rows = args.base_rows.min(50_000);
    let load_rows = total_memtable_rows.max(sample_rows);
    println!("loading {load_rows} FineWeb rows for memtable payload + query sample ...");
    let mt_corpus = load_corpus(load_rows, &args.cache_dir).await?;
    let mt_text = &mt_corpus[..total_memtable_rows];

    let shard_id = Uuid::new_v4();
    let row_bytes = 2048; // rough FineWeb text row size
    // The memtable flush trigger is `estimated_size >= max_memtable_size ||
    // batch_store_full`. FineWeb text rows vary in size, so a byte threshold
    // is an unreliable way to flush exactly one generation per
    // `max_memtable_rows`. Instead make the *batch-count* cap the trigger:
    // set `max_memtable_batches` to one generation's worth of batches so the
    // store fills (and flushes) precisely at each generation boundary,
    // independent of text length. Keep `max_memtable_size` high so it never
    // pre-empts the batch-count trigger.
    let batches_per_gen = (args.max_memtable_rows / args.batch_rows).max(1);
    let config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: false,
        sync_indexed_write: false,
        max_memtable_size: args.max_memtable_rows * row_bytes * 100,
        max_memtable_rows: args.max_memtable_rows,
        max_memtable_batches: batches_per_gen,
        max_unflushed_memtable_bytes: args.max_memtable_rows * row_bytes * 20,
        max_wal_flush_interval: Some(Duration::from_secs(60)),
        ..ShardWriterConfig::default()
    };
    let writer = dataset.mem_wal_writer(shard_id, config).await?;

    let flush_wait = if is_cloud_uri(&args.uri) {
        Duration::from_secs(5)
    } else {
        Duration::from_millis(500)
    };

    // Ingest flushed generations + 1 active (50% full).
    let mut gen_sizes: Vec<usize> = (0..args.flushed_generations)
        .map(|_| args.max_memtable_rows)
        .collect();
    gen_sizes.push(active_rows);

    let id_base = args.base_rows as i64;
    let mut cursor = 0usize;
    let ingest_start = Instant::now();
    for (gen_idx, &gen_rows) in gen_sizes.iter().enumerate() {
        let mut written = 0usize;
        while written < gen_rows {
            let chunk = args.batch_rows.min(gen_rows - written);
            let start = id_base + (cursor) as i64;
            let slice = &mt_text[cursor..cursor + chunk];
            let batch = make_batch(schema.clone(), start, slice, args.vector_dim);
            writer.put(vec![batch]).await?;
            cursor += chunk;
            written += chunk;
        }
        let is_flushed = gen_idx < args.flushed_generations;
        println!(
            "  gen {}: wrote {} rows ({})",
            gen_idx + 1,
            gen_rows,
            if is_flushed { "flushed" } else { "active" }
        );
        if is_flushed {
            tokio::time::sleep(flush_wait).await;
        }
    }
    // Wait for any triggered (sealed) memtable flushes to commit to the
    // manifest before we snapshot it — otherwise the flushed generations
    // race the read and may not all be visible yet.
    writer.wait_for_flush_drain().await?;
    println!(
        "ingested {} memtable rows in {:.1}s",
        cursor,
        ingest_start.elapsed().as_secs_f64()
    );

    let manifest = writer.manifest().await?;
    let in_memory_refs = writer.in_memory_memtable_refs().await?;
    let mut shard_snapshot = ShardSnapshot::new(shard_id);
    if let Some(ref m) = manifest {
        shard_snapshot = shard_snapshot.with_current_generation(m.current_generation);
        for fg in &m.flushed_generations {
            shard_snapshot = shard_snapshot.with_flushed_generation(fg.generation, fg.path.clone());
        }
    }
    let num_flushed = manifest
        .as_ref()
        .map(|m| m.flushed_generations.len())
        .unwrap_or(0);
    println!("manifest: {num_flushed} flushed generations");

    // Flushed generations carry the same maintained secondary indexes as
    // the active memtable: the flush handler builds them during flush
    // (lance #6901), so each generation already has the FTS index and
    // both scoring modes use the fast indexed path. No manual indexing
    // step is needed here. (The index-less flat fallback in the rescore
    // planner is still exercised by unit tests for the no-maintained-index
    // case.)

    let _ = in_memory_refs; // each panel rebuilds its own collector below.
    let pk_columns = vec!["id".to_string()];
    let session_ctx = SessionContext::new();
    let task_ctx = session_ctx.task_ctx();
    let id_base = args.base_rows as i64;
    let n_mt = total_memtable_rows.max(1) as i64;
    // Spread query keys across base table and the memtable id range so every
    // query touches base + flushed gens + active memtable.
    let pick_id = |i: usize| -> i64 {
        if i % 2 == 0 {
            ((i as i64) * 7919) % args.base_rows.max(1) as i64
        } else {
            id_base + (((i as i64) * 7919) % n_mt)
        }
    };
    // Rebuild a fresh snapshot per planner (cheap; from the manifest already read).
    macro_rules! snapshot {
        () => {{
            let mut s = ShardSnapshot::new(shard_id);
            if let Some(ref m) = manifest {
                s = s.with_current_generation(m.current_generation);
                for fg in &m.flushed_generations {
                    s = s.with_flushed_generation(fg.generation, fg.path.clone());
                }
            }
            s
        }};
    }

    // Shared across panels so repeated queries reuse opened flushed-generation
    // datasets + warm their index caches, matching a long-lived production
    // reader instead of cold-opening every flushed gen on each query.
    let session = dataset.session();
    let flushed_cache = Arc::new(FlushedMemTableCache::new(64));

    // Build the three planners once; they are k-independent, so a single
    // ingest can serve every top-k in `args.k_values`. All three share the
    // session + flushed cache so flushed-gen datasets open once and stay warm.
    let pl_planner = LsmPointLookupPlanner::new(
        LsmDataSourceCollector::new(dataset.clone(), vec![snapshot!()])
            .with_active_memtable(shard_id, writer.active_memtable_ref().await?),
        pk_columns.clone(),
        arrow_schema.clone(),
    )
    .with_session(session.clone())
    .with_flushed_cache(flushed_cache.clone());
    let vec_planner = LsmVectorSearchPlanner::new(
        LsmDataSourceCollector::new(dataset.clone(), vec![snapshot!()])
            .with_in_memory_memtables(shard_id, writer.in_memory_memtable_refs().await?),
        pk_columns.clone(),
        arrow_schema.clone(),
        VECTOR_COL.to_string(),
        DistanceType::Cosine,
    )
    .with_session(session.clone())
    .with_flushed_cache(flushed_cache.clone());
    let fts_planner = LsmFtsSearchPlanner::new(
        LsmDataSourceCollector::new(dataset.clone(), vec![snapshot!()])
            .with_in_memory_memtables(shard_id, writer.in_memory_memtable_refs().await?),
        pk_columns.clone(),
        arrow_schema.clone(),
    )
    .with_session(session.clone())
    .with_flushed_cache(flushed_cache.clone());
    let sample_end = mt_corpus
        .len()
        .min(sample_rows.max(total_memtable_rows.min(50_000)));
    let fts_queries = build_query_terms(&mt_corpus[..sample_end], args.queries);

    let panel = |s: &LatencyStats| {
        json!({
            "p50_us": s.p50_us, "p95_us": s.p95_us, "p99_us": s.p99_us,
            "mean_us": s.mean_us as u64, "qps": s.qps as u64,
        })
    };

    let mut results = Vec::with_capacity(args.k_values.len());
    for &k in &args.k_values {
        println!("===== k={k} =====");

        // ---- Panel 1: point lookup (btree across LSM; k-independent) ----
        println!("running point-lookup panel ({} queries) ...", args.queries);
        let mut pl_lat = Vec::with_capacity(args.queries);
        for i in 0..args.queries {
            let t0 = Instant::now();
            let plan = pl_planner
                .plan_lookup(&[ScalarValue::Int64(Some(pick_id(i)))], None)
                .await?;
            let _b: Vec<RecordBatch> = plan.execute(0, task_ctx.clone())?.try_collect().await?;
            pl_lat.push(t0.elapsed().as_micros() as f64);
        }
        let pl_stats = compute_stats(pl_lat);

        // ---- Panel 2: vector (IVF_RQ base + HNSW layers). overfetch_factor
        //      < 1.0 turns stale filtering off (allow stale rows). ----
        println!(
            "running vector panel ({} queries, k={k}, nprobes={}) ...",
            args.queries, args.nprobes
        );
        let mut vec_lat = Vec::with_capacity(args.queries);
        for i in 0..args.queries {
            let fsl = wrap_query(&gen_vector(pick_id(i), args.vector_dim), args.vector_dim);
            let t0 = Instant::now();
            let plan = vec_planner
                .plan_search(&fsl, k, args.nprobes, None, false, 0.0)
                .await?;
            let _b: Vec<RecordBatch> = plan.execute(0, task_ctx.clone())?.try_collect().await?;
            vec_lat.push(t0.elapsed().as_micros() as f64);
        }
        let vec_stats = compute_stats(vec_lat);

        // ---- Panel 3: FTS (Local mode) ----
        println!(
            "running FTS panel ({} queries, k={k}) ...",
            fts_queries.len()
        );
        let fts_run = run_mode(&fts_planner, FtsScoringMode::Local, &fts_queries, k).await?;
        let fts_stats = compute_stats(fts_run.latencies_us.clone());

        println!(
            "k={k}  point: p50={}us p99={}us | vector: p50={}us p99={}us | fts: p50={}us p99={}us",
            pl_stats.p50_us,
            pl_stats.p99_us,
            vec_stats.p50_us,
            vec_stats.p99_us,
            fts_stats.p50_us,
            fts_stats.p99_us,
        );

        results.push((
            k,
            json!({
                "bench": "mem_wal_lsm_read",
                "phase": "search",
                "uri_kind": if is_cloud_uri(&args.uri) { "cloud" } else { "local" },
                "base_rows": args.base_rows,
                "max_memtable_rows": args.max_memtable_rows,
                "flushed_generations": num_flushed,
                "active_rows": active_rows,
                "k": k,
                "vector_dim": args.vector_dim,
                "nprobes": args.nprobes,
                "queries": args.queries,
                "point_lookup": panel(&pl_stats),
                "vector": panel(&vec_stats),
                "fts": panel(&fts_stats),
            }),
        ));
    }

    // Keep writer alive so the active memtable stays reachable.
    std::mem::forget(writer);

    Ok(results)
}

// ----------------------------------------------------------------------
// Entrypoint
// ----------------------------------------------------------------------

async fn run(args: Args) -> Result<()> {
    println!(
        "bench=mem_wal_fts_read phase={} uri={} base_rows={} max_memtable_rows={} flushed_generations={} queries={} k={} rescore_factor={}",
        args.phase.as_str(),
        args.uri,
        args.base_rows,
        args.max_memtable_rows,
        args.flushed_generations,
        args.queries,
        args.k,
        args.rescore_factor,
    );

    match args.phase {
        Phase::Prepare => run_prepare(&args).await?,
        Phase::Search => {
            let results = run_search(&args).await?;
            for (k, result) in &results {
                let text = serde_json::to_string_pretty(result)
                    .map_err(|e| lance_core::Error::io(format!("serialize: {e}")))?;
                println!("{text}");
                // Per-k output: <output_dir>/search_<tag>_k<K>.json takes
                // precedence; otherwise fall back to a single --output file
                // (only meaningful for a single k).
                let out_path = match (&args.output_dir, &args.output_tag) {
                    (Some(dir), Some(tag)) => Some(dir.join(format!("search_{tag}_k{k}.json"))),
                    _ => args.output.clone(),
                };
                if let Some(path) = out_path {
                    if let Some(parent) = path.parent()
                        && !parent.as_os_str().is_empty()
                    {
                        std::fs::create_dir_all(parent).ok();
                    }
                    std::fs::write(&path, text.as_bytes()).map_err(|e| {
                        lance_core::Error::io(format!("write {}: {e}", path.display()))
                    })?;
                }
            }
        }
    }
    println!("=== DONE ===");
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| lance_core::Error::io(format!("build runtime: {e}")))?;
    runtime.block_on(run(args))
}
