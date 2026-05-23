#!/usr/bin/env bash
# Experiment 2 — sustained multi-flush throughput & MemTable-flush BACKPRESSURE.
# For each fixed flush size N and index combination, set max_memtable_rows=N so a
# flush fires every N rows, then ingest ~FLUSHES*N rows unpaced (skip-close) and
# observe whether the L0 flush backlog (frozen memtables) builds up faster than
# it drains. max_unflushed_memtable_bytes is set to a small multiple of one
# memtable so backpressure CAN engage; if flush keeps up it simply never does.
#
# Reports sustained rows/s, peak frozen-memtable backlog, backpressure wait time,
# and slow puts.
#
# Usage:
#   rust/lance/benches/mem_wal/write/run_flush_backpressure.sh [run_id]
#
# Env knobs (with defaults):
#   SIZES          "100000 500000 1000000"   per-flush memtable row counts
#   COMBOS         "none btree vector fts btree+vector btree+fts btree+vector+fts"
#   STORAGES       "s3 s3express"
#   FLUSHES        10        target number of flushes (calls = FLUSHES*N/BATCH_ROWS)
#   MAX_DURATION_S 600       per-cell wall-clock cap; whichever limit hits first
#   UNFLUSHED_MULT 3         backpressure budget = MULT * N * ROW_BYTES
#   BATCH_ROWS     1000
#   SEED_ROWS      5000
#   VECTOR_DIM     1024
#   ROW_BYTES      5760      fineweb row size used to size the backpressure budget
#   THREADS        nproc
#   CLEANUP        1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_ID="${1:-flushbp-$(date -u +%Y%m%dT%H%M%SZ)}"
SIZES="${SIZES:-100000 500000 1000000}"
COMBOS="${COMBOS:-none btree vector fts btree+vector btree+fts btree+vector+fts}"
STORAGES="${STORAGES:-s3 s3express}"
FLUSHES="${FLUSHES:-10}"
MAX_DURATION_S="${MAX_DURATION_S:-600}"
UNFLUSHED_MULT="${UNFLUSHED_MULT:-3}"
BATCH_ROWS="${BATCH_ROWS:-1000}"
SEED_ROWS="${SEED_ROWS:-5000}"
VECTOR_DIM="${VECTOR_DIM:-1024}"
ROW_BYTES="${ROW_BYTES:-5760}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
CLEANUP="${CLEANUP:-1}"
# Hard per-cell kill = duration cap + slack. A background flush that panics
# (e.g. HNSW index write at scale) stalls puts on backpressure forever, so the
# graceful --max-duration-s break can never fire; this guarantees progress.
CELL_TIMEOUT="${CELL_TIMEOUT:-$(( MAX_DURATION_S + 300 ))}"

HUGE_BYTES=1099511627776   # 1 TiB — size never triggers; row count N is the trigger

# shellcheck source=/dev/null
source "$SCRIPT_DIR/flush_bench_common.sh"

LOCAL_DIR="$REPO_ROOT/target/mem-wal-flush-results/${RUN_ID}"
mkdir -p "$LOCAL_DIR"

BIN="$(find_bench_bin)"
echo "exp=flush_backpressure bin=$BIN run_id=$RUN_ID threads=$THREADS"
echo "sizes=[$SIZES] combos=[$COMBOS] storages=[$STORAGES] flushes=$FLUSHES cap=${MAX_DURATION_S}s unflushed_mult=$UNFLUSHED_MULT"

for storage in $STORAGES; do
    base_uri="$(storage_base_uri "$storage")"
    for combo in $COMBOS; do
        mode="$(mode_for_combo "$combo")"
        for size in $SIZES; do
            calls=$(( FLUSHES * size / BATCH_ROWS ))
            unflushed=$(( UNFLUSHED_MULT * size * ROW_BYTES ))
            label="$(cell_label "$storage" "$combo" "$size")"
            out="$LOCAL_DIR/${label}.json"; log="$LOCAL_DIR/${label}.log"
            uri="$base_uri/$RUN_ID/${label}"
            echo ">>> $label (mode=$mode calls=$calls unflushed=$((unflushed/1000000))MB cap=${MAX_DURATION_S}s)"
            if [ -f "$out" ]; then echo "    already done"; continue; fi
            timeout "$CELL_TIMEOUT" "$BIN" --bench \
                --mode "$mode" --indexes "$combo" --schema-shape fineweb \
                --uri "$uri" \
                --seed-rows "$SEED_ROWS" --batch-rows "$BATCH_ROWS" --calls "$calls" \
                --vector-dim "$VECTOR_DIM" \
                --max-memtable-size "$HUGE_BYTES" \
                --max-memtable-rows "$size" \
                --max-unflushed-memtable-bytes "$unflushed" \
                --max-wal-buffer-size 52428800 \
                --max-wal-flush-interval-ms 0 \
                --sample-interval-ms 1000 \
                --max-duration-s "$MAX_DURATION_S" \
                --threads "$THREADS" --tokio-threads "$THREADS" \
                --skip-close \
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
echo "=== flush-backpressure summary ==="
summarize_flush_backpressure "$LOCAL_DIR"
echo ""
echo "results: $LOCAL_DIR"
