#!/usr/bin/env bash
# Experiment 1 — MemTable FLUSH COST (post-cee7d32, which made flush rebuild the
# persisted secondary index). For each memtable size and index combination we
# grow ONE memtable to exactly N rows (memtable limits set huge so nothing
# flushes mid-run), then `close()` to trigger exactly one flush and measure:
#   - memtable_flush_s      : wall time of the single flush (incl. index build)
#   - final_memtable_mb     : how big the in-memory memtable grew before flush
#   - index_setup_s, etc.
#
# Sweeps {sizes} x {index combos} x {storage backends}.
#
# Usage:
#   rust/lance/benches/mem_wal/write/run_flush_cost.sh [run_id]
#
# Env knobs (with defaults):
#   SIZES      "100000 500000 1000000"   memtable row counts to flush
#   COMBOS     "none btree vector fts btree+vector btree+fts btree+vector+fts"
#   STORAGES   "s3 s3express"
#   BATCH_ROWS 1000
#   SEED_ROWS  5000        base-table rows (index training); separate from memtable
#   VECTOR_DIM 1024
#   THREADS    nproc
#   CLEANUP    1           delete each dataset from object storage after capture

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_ID="${1:-flushcost-$(date -u +%Y%m%dT%H%M%SZ)}"
SIZES="${SIZES:-100000 500000 1000000}"
COMBOS="${COMBOS:-none btree vector fts btree+vector btree+fts btree+vector+fts}"
STORAGES="${STORAGES:-s3 s3express}"
BATCH_ROWS="${BATCH_ROWS:-1000}"
SEED_ROWS="${SEED_ROWS:-5000}"
VECTOR_DIM="${VECTOR_DIM:-1024}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
CLEANUP="${CLEANUP:-1}"
# Hard per-cell wall-clock kill. A flush that panics (e.g. the HNSW index
# write at scale) deadlocks close(); without this the whole sweep would hang.
CELL_TIMEOUT="${CELL_TIMEOUT:-1800}"

# Memtable limits set huge so the ONLY flush is the one close() forces.
HUGE_BYTES=1099511627776   # 1 TiB
HUGE_ROWS=100000000        # 100M
HUGE_BATCHES=100000000

# shellcheck source=/dev/null
source "$SCRIPT_DIR/flush_bench_common.sh"

LOCAL_DIR="$REPO_ROOT/target/mem-wal-flush-results/${RUN_ID}"
mkdir -p "$LOCAL_DIR"

BIN="$(find_bench_bin)"
echo "exp=flush_cost bin=$BIN run_id=$RUN_ID threads=$THREADS"
echo "sizes=[$SIZES] combos=[$COMBOS] storages=[$STORAGES]"

for storage in $STORAGES; do
    base_uri="$(storage_base_uri "$storage")"
    for combo in $COMBOS; do
        mode="$(mode_for_combo "$combo")"
        for size in $SIZES; do
            calls=$(( size / BATCH_ROWS ))
            label="$(cell_label "$storage" "$combo" "$size")"
            out="$LOCAL_DIR/${label}.json"; log="$LOCAL_DIR/${label}.log"
            uri="$base_uri/$RUN_ID/${label}"
            echo ">>> $label (mode=$mode calls=$calls)"
            if [ -f "$out" ]; then echo "    already done"; continue; fi
            timeout "$CELL_TIMEOUT" "$BIN" --bench \
                --mode "$mode" --indexes "$combo" --schema-shape fineweb \
                --uri "$uri" \
                --seed-rows "$SEED_ROWS" --batch-rows "$BATCH_ROWS" --calls "$calls" \
                --vector-dim "$VECTOR_DIM" \
                --max-memtable-size "$HUGE_BYTES" \
                --max-memtable-rows "$HUGE_ROWS" \
                --max-memtable-batches "$HUGE_BATCHES" \
                --max-unflushed-memtable-bytes "$HUGE_BYTES" \
                --max-wal-buffer-size 52428800 \
                --max-wal-flush-interval-ms 0 \
                --sample-interval-ms 1000 \
                --threads "$THREADS" --tokio-threads "$THREADS" \
                --output "$out" > "$log" 2>&1
            rc=$?
            if [ "$rc" -eq 124 ]; then echo "    !!! TIMED OUT after ${CELL_TIMEOUT}s (likely flush panic/hang)"
            elif [ "$rc" -ne 0 ]; then echo "    !!! failed rc=$rc (see $log)"
            else echo "    ok"; fi
            upload_and_cleanup "$out" "$log" "$base_uri/$RUN_ID/results/${label}" "$uri"
        done
    done
done

echo ""
echo "=== flush-cost summary ==="
summarize_flush_cost "$LOCAL_DIR"
echo ""
echo "results: $LOCAL_DIR"
