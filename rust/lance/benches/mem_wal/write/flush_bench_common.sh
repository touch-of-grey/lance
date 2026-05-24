#!/usr/bin/env bash
# Shared helpers for the MemTable flush experiments (run_flush_cost.sh /
# run_flush_backpressure.sh). Source this; do not execute directly.

# Map a storage label to its dataset base URI. S3 Express is AZ-pinned: the
# bucket below lives in use1-az4 and is only fair when the instance is co-located.
storage_base_uri() {
    case "$1" in
        s3)        echo "s3://jack-devland-build/mem-wal-flush-bench" ;;
        s3express) echo "s3://jack-lancedb-devland--use1-az4--x-s3/mem-wal-flush-bench" ;;
        local)     echo "${LOCAL_BASE:-${TMPDIR:-/tmp}/mem-wal-flush-bench}" ;;
        *)         echo "unknown storage '$1' (expected s3|s3express|local)" >&2; exit 1 ;;
    esac
}

# Pick the write mode. With SYNC=1 the in-memory index is built inline with each
# put (so it is fully present when we snapshot index memory — used by the
# flush-cost experiment). With SYNC=0 (default) indexing is async, the realistic
# sustained-ingest path used by the backpressure experiment.
mode_for_combo() {
    local sync="${SYNC:-0}"
    if [ "$1" = "none" ]; then
        [ "$sync" = "1" ] && echo "sync_noidx" || echo "async_noidx"
    else
        [ "$sync" = "1" ] && echo "sync_idx" || echo "async_idx"
    fi
}

# Filesystem/S3-safe cell label: '+' in combos becomes '-'.
cell_label() {
    local storage="$1" combo="$2" size="$3"
    echo "${storage}_${combo//+/-}_${size}"
}

# Locate (building if needed) the backpressure bench binary.
find_bench_bin() {
    local bin
    bin="$(find target/release/deps -maxdepth 1 -type f -perm -111 \
        -name 'mem_wal_shard_writer_backpressure-*' ! -name '*.d' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    if [ -z "$bin" ]; then
        cargo bench -p lance --bench mem_wal_shard_writer_backpressure --no-run >&2
        bin="$(find target/release/deps -maxdepth 1 -type f -perm -111 \
            -name 'mem_wal_shard_writer_backpressure-*' ! -name '*.d' \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    fi
    echo "$bin"
}

# Copy result JSON + log to S3, then (if CLEANUP=1) delete the dataset itself so
# the sweep does not accumulate hundreds of GB. Only the dataset URI is removed,
# never the results prefix.
upload_and_cleanup() {
    local out="$1" log="$2" results_prefix="$3" dataset_uri="$4"
    [ -f "$out" ] && aws s3 cp "$out" "${results_prefix}.json" >/dev/null 2>&1
    aws s3 cp "$log" "${results_prefix}.log" >/dev/null 2>&1
    if [ "${CLEANUP:-1}" = "1" ] && [[ "$dataset_uri" == s3://* ]]; then
        # Trailing slash so a cell URI is never a string-prefix of another
        # (e.g. .../s3_btree_100000 vs .../s3_btree_1000000).
        aws s3 rm --recursive "${dataset_uri%/}/" >/dev/null 2>&1
    fi
}

summarize_flush_cost() {
    python3 - "$1" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
# flush_s   = persisted L0 write + persisted index build (the cee7d32 change)
# idx_upd_s = in-memory index build streamed during puts (HNSW/FTS/btree in RAM)
# drain_s   = total close() wall time (final WAL flush + freeze + flush)
# data_MB    = buffered row data + bloom (combo-independent)
# idxmem_MB  = in-memory maintained indexes (btree/HNSW/FTS), on top of data_MB
# gen_disk_MB= total on-disk size of the flushed generation (data + index files)
print(f"{'cell':40s} {'indexes':22s} {'rows':>9s} {'data_MB':>9s} {'idxmem_MB':>9s} "
      f"{'idx_upd_s':>9s} {'flush_s':>8s} {'flush_MB/s':>10s} {'gen_disk_MB':>11s} {'drain_s':>8s}")
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try: r = json.load(open(p))
    except Exception: continue
    name = os.path.basename(p)[:-5]
    rows = r.get("total_rows_written", 0)
    mt_mb = (r.get("final_memtable_bytes") or 0) / 1e6
    idxmem_mb = (r.get("index_memory_bytes") or 0) / 1e6
    gen_mb = (r.get("flushed_generation_bytes") or 0) / 1e6
    ws = r.get("write_stats") or {}
    flush_s = ws.get("memtable_flush_time_seconds", 0.0)
    rb = r.get("row_bytes", 0)
    flush_mbps = (rows * rb / 1e6 / flush_s) if flush_s > 0 else 0.0
    print(f"{name:40s} {r.get('indexes',''):22s} {rows:>9d} {mt_mb:>9.0f} {idxmem_mb:>9.0f} "
          f"{ws.get('index_update_time_seconds',0):>9.2f} "
          f"{flush_s:>8.2f} {flush_mbps:>10.1f} {gen_mb:>11.0f} "
          f"{r.get('elapsed_drain_seconds',0):>8.2f}")
PY
}

summarize_flush_backpressure() {
    python3 - "$1" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
print(f"{'cell':40s} {'indexes':22s} {'flush_rows':>10s} {'sust_rows/s':>11s} "
      f"{'sust_MB/s':>9s} {'flushes':>7s} {'avg_flush_ms':>12s} {'max_frozen':>10s} "
      f"{'max_unfl_MB':>11s} {'bp_cnt':>6s} {'bp_wait_ms':>10s} {'slow1s':>6s} {'p99_ms':>9s}")
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try: r = json.load(open(p))
    except Exception: continue
    name = os.path.basename(p)[:-5]
    ws = r.get("write_stats") or {}
    bp = r.get("backpressure") or {}
    flushes = ws.get("memtable_flush_count", 0)
    # Actual per-flush size (the memtable freezes on batch count, not the row
    # capacity), falling back to the configured batch trigger when 0 flushes.
    if flushes:
        flush_size = ws.get("memtable_flush_rows", 0) // flushes
    else:
        flush_size = r.get("max_memtable_batches", 0) * r.get("batch_rows", 0)
    print(f"{name:40s} {r.get('indexes',''):22s} "
          f"{flush_size:>10d} "
          f"{r.get('throughput_puts_rows_per_sec',0):>11.0f} "
          f"{r.get('throughput_puts_mb_per_sec',0):>9.1f} {flushes:>7d} "
          f"{r.get('avg_memtable_flush_ms',0):>12.1f} "
          f"{r.get('max_frozen_memtable_count',0):>10d} "
          f"{(r.get('max_unflushed_memtable_bytes_observed') or 0)/1e6:>11.1f} "
          f"{bp.get('count',0):>6d} {bp.get('total_wait_ms',0):>10d} "
          f"{r.get('slow_puts_1s',0):>6d} {r.get('p99_ms',0):>9.1f}")
PY
}
