#!/usr/bin/env bash
# On-disk FTS comparison driver — Lance flushed memtable vs Apache Lucene
# (FSDirectory) vs Tantivy (MmapDirectory), all writing their index to local
# NVMe. The Lance side writes the corpus through the ShardWriter and flushes it
# into a single on-disk generation (one fragment + one InvertedIndex), then
# queries it through the normal Dataset scan path.
#
# Usage: run_fts_disk_compare.sh [run_id]
#
# Env:
#   SIZES        doc-count sweep (default "100000 500000 1000000")
#   K            top-k (default 10)
#   THREADS      query threads for the multi-thread QPS run
#   NVME_DIR     mount point for the local NVMe (default /mnt/nvme); all index
#                data, corpus and cache live here so the EBS root never fills
#   NVME_DEVICE  explicit block device to mount (default: largest unmounted nvme)
#   LUCENE_CP / LUCENE_DIR / JAVA_HOME   as in run_fts_mem_compare.sh
#   POSITIONS=0  index without positions on all sides (term-only, no phrase)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

NVME_DIR="${NVME_DIR:-/mnt/nvme}"

# ---- mount local NVMe (idempotent) ----
ensure_nvme() {
    if mountpoint -q "$NVME_DIR" 2>/dev/null; then
        echo "NVMe already mounted at $NVME_DIR"; return
    fi
    if [ -d "$NVME_DIR" ] && [ -w "$NVME_DIR" ] && [ -z "${NVME_DEVICE:-}" ]; then
        # Not a mountpoint but usable (e.g. plain dir on a fast disk) — accept.
        echo "WARNING: $NVME_DIR is not a mountpoint; using it as-is" >&2; return
    fi
    local dev="${NVME_DEVICE:-}"
    if [ -z "$dev" ]; then
        # Pick the largest whole NVMe disk that has no partitions and is not mounted.
        dev=$(lsblk -dpno NAME,TYPE,MOUNTPOINT | awk '$2=="disk" && $1 ~ /nvme/ && $3=="" {print $1}' \
              | while read -r d; do
                    # skip the root disk (has children / is mounted)
                    if ! lsblk -no MOUNTPOINT "$d" | grep -q '/'; then
                        echo "$(lsblk -bdno SIZE "$d") $d"
                    fi
                done | sort -nr | head -1 | awk '{print $2}')
    fi
    [ -n "$dev" ] || { echo "ERROR: no free NVMe device found; set NVME_DEVICE or NVME_DIR" >&2; exit 1; }
    echo "Formatting + mounting $dev at $NVME_DIR"
    sudo mkfs.ext4 -F -E nodiscard "$dev"
    sudo mkdir -p "$NVME_DIR"
    sudo mount "$dev" "$NVME_DIR"
    sudo chown "$(id -u):$(id -g)" "$NVME_DIR"
}
ensure_nvme

RUN_ID="${1:-fts-disk-compare-$(date -u +%Y%m%dT%H%M%SZ)}"
SIZES="${SIZES:-100000 500000 1000000}"
K="${K:-10}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
CACHE_DIR="${CACHE_DIR:-$NVME_DIR/lance-fineweb-cache}"
WORK="${WORK:-$NVME_DIR/fts_disk_compare/$RUN_ID}"
DATA_ROOT="$NVME_DIR/fts_disk_index"
RESULT_DIR="$REPO_ROOT/target/fts-compare-results/$RUN_ID"
mkdir -p "$WORK" "$RESULT_DIR" "$CACHE_DIR" "$DATA_ROOT"

# ---- locate JDK 25 ----
if [ -z "${JAVA_HOME:-}" ]; then
    for cand in /usr/lib/jvm/java-25-* /usr/lib/jvm/jdk-25* /opt/jdk-25* \
                "$HOME/.sdkman/candidates/java/25"*; do
        [ -x "$cand/bin/java" ] && JAVA_HOME="$cand" && break
    done
fi
if [ -z "${JAVA_HOME:-}" ] || [ ! -x "$JAVA_HOME/bin/java" ]; then
    echo "ERROR: JDK 25 not found; set JAVA_HOME" >&2; exit 1
fi
JAVA="$JAVA_HOME/bin/java"; JAVAC="$JAVA_HOME/bin/javac"
echo "JDK: $($JAVA -version 2>&1 | head -1)"

# ---- build Lucene classpath ----
if [ -z "${LUCENE_CP:-}" ]; then
    [ -n "${LUCENE_DIR:-}" ] || { echo "ERROR: set LUCENE_CP or LUCENE_DIR" >&2; exit 1; }
    echo "=== Building Lucene jars ($LUCENE_DIR) ==="
    ( cd "$LUCENE_DIR" && JAVA_HOME="$JAVA_HOME" ./gradlew -q \
        :lucene:core:jar :lucene:analysis:common:jar ) || { echo "ERROR: Lucene build failed" >&2; exit 1; }
    CORE_JAR="$(find "$LUCENE_DIR/lucene/core/build/libs" -name 'lucene-core-*.jar' | head -1)"
    ANALYSIS_JAR="$(find "$LUCENE_DIR/lucene/analysis/common/build/libs" -name 'lucene-analysis-common-*.jar' | head -1)"
    LUCENE_CP="$CORE_JAR:$ANALYSIS_JAR"
fi
echo "Lucene classpath: $LUCENE_CP"

# ---- build the Rust benches ----
build_bench() {  # $1 = bench name -> echoes freshest binary path
    rm -f "$REPO_ROOT"/target/release/deps/"$1"-*
    cargo bench -p lance --bench "$1" --no-run >/dev/null 2>&1 || {
        echo "ERROR: cargo build of $1 failed" >&2; cargo bench -p lance --bench "$1" --no-run; exit 1; }
    find "$REPO_ROOT/target/release/deps" -maxdepth 1 -type f -perm -111 \
        -name "$1-*" ! -name '*.d' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}
echo "=== Building Lance gen + Lance-disk + Tantivy benches ==="
GEN_BIN="$(build_bench mem_wal_fts_bench)";        echo "gen bench:        $GEN_BIN"
LANCE_DISK_BIN="$(build_bench mem_wal_fts_disk_bench)"; echo "lance-disk bench: $LANCE_DISK_BIN"
TANTIVY_BIN="$(build_bench tantivy_fts_bench)";    echo "tantivy bench:    $TANTIVY_BIN"

echo "=== Compiling Lucene FTS bench ==="
"$JAVAC" -cp "$LUCENE_CP" -d "$WORK" "$SCRIPT_DIR/LuceneFtsBench.java" \
    || { echo "ERROR: javac failed" >&2; exit 1; }

mutual_overlap() {  # $1=topk A  $2=topk B  $3=k
    python3 - "$1" "$2" "$3" <<'PY'
import sys
a, b, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    la = [set(l.split()) for l in open(a)]
    lb = [set(l.split()) for l in open(b)]
except FileNotFoundError:
    print("nan"); sys.exit()
n = min(len(la), len(lb))
if n == 0:
    print("nan"); sys.exit()
tot = sum(len(la[i] & lb[i]) / max(len(la[i] | lb[i]), 1) for i in range(n))
print(f"{tot / n:.4f}")
PY
}

FLAGS=""
if [ "${POSITIONS:-1}" = "0" ]; then FLAGS="--no-positions"; fi
echo "flags: '$FLAGS'   NVME_DIR: $NVME_DIR"
echo ""

for SIZE in $SIZES; do
    DIR="$WORK/n$SIZE"; mkdir -p "$DIR"
    echo "############ corpus size = $SIZE ############"
    echo "--- generating shared corpus + queries ---"
    "$GEN_BIN" --bench gen --docs "$SIZE" --out-dir "$DIR" \
        --cache-dir "$CACHE_DIR" --k "$K" > "$RESULT_DIR/gen_n$SIZE.log" 2>&1 || {
        echo "  !!! gen failed (see gen_n$SIZE.log)"; continue; }

    for RUN in a b; do
        echo "--- run $RUN: lance flushed memtable (on disk) ---"
        "$LANCE_DISK_BIN" --in-dir "$DIR" --run "$RUN" --k "$K" --threads "$THREADS" \
            --data-dir "$DATA_ROOT/lance_n${SIZE}_run${RUN}" $FLAGS \
            | tee "$RESULT_DIR/lance_disk_n${SIZE}_run${RUN}.txt" \
            | grep '^{' > "$RESULT_DIR/lance_disk_n${SIZE}_run${RUN}.json"
        echo "--- run $RUN: lucene (FSDirectory) ---"
        "$JAVA" -cp "$LUCENE_CP:$WORK" LuceneFtsBench --in-dir "$DIR" --run "$RUN" \
            --k "$K" --threads "$THREADS" --dir "$DATA_ROOT/lucene_n${SIZE}_run${RUN}" $FLAGS \
            | tee "$RESULT_DIR/lucene_n${SIZE}_run${RUN}.txt" \
            | grep '^{' > "$RESULT_DIR/lucene_n${SIZE}_run${RUN}.json"
        echo "--- run $RUN: tantivy (MmapDirectory) ---"
        "$TANTIVY_BIN" --in-dir "$DIR" --run "$RUN" --k "$K" --threads "$THREADS" \
            --dir "$DATA_ROOT/tantivy_n${SIZE}_run${RUN}" $FLAGS \
            | tee "$RESULT_DIR/tantivy_n${SIZE}_run${RUN}.txt" \
            | grep '^{' > "$RESULT_DIR/tantivy_n${SIZE}_run${RUN}.json"

        ov_ll="$(mutual_overlap "$DIR/lance_disk_run${RUN}_topk.txt" "$DIR/lucene_run${RUN}_topk.txt" "$K")"
        ov_lt="$(mutual_overlap "$DIR/lance_disk_run${RUN}_topk.txt" "$DIR/tantivy_run${RUN}_topk.txt" "$K")"
        ov_ut="$(mutual_overlap "$DIR/lucene_run${RUN}_topk.txt" "$DIR/tantivy_run${RUN}_topk.txt" "$K")"
        echo "    overlap run $RUN: lance<->lucene=$ov_ll lance<->tantivy=$ov_lt lucene<->tantivy=$ov_ut"
        echo "$ov_ll $ov_lt $ov_ut" > "$RESULT_DIR/overlap_n${SIZE}_run${RUN}.txt"
        # Free the index dirs before the next size to bound NVMe usage.
        rm -rf "$DATA_ROOT/lance_n${SIZE}_run${RUN}" "$DATA_ROOT/lucene_n${SIZE}_run${RUN}" \
               "$DATA_ROOT/tantivy_n${SIZE}_run${RUN}"
    done
    echo ""
done

echo "=== summary (on-disk; index_mb = inverted-index bytes, not data) ==="
python3 - "$RESULT_DIR" "$K" <<'PY'
import glob, json, os, sys
d, k = sys.argv[1], sys.argv[2]
print(f"{'size':>9} {'run':>4} {'impl':>11} {'build_dps':>11} {'q_p50_us':>10} "
      f"{'q_p95_us':>10} {'qps_1t':>9} {'qps_nt':>10} {'term_rec':>9} {'phr_rec':>9} "
      f"{'or_rec':>9} {'index_mb':>9}")
for p in sorted(glob.glob(os.path.join(d, "*_n*_run*.json"))):
    try: r = json.load(open(p))
    except Exception: continue
    # index size: Lance reports fts_index_bytes (index only); others report mem_bytes.
    idx = r.get('fts_index_bytes', r.get('mem_bytes', 0)) / 1e6
    print(f"{r['docs']:>9} {r['run']:>4} {r['impl']:>11} {r['build_docs_per_s']:>11.0f} "
          f"{r['q_p50_us']:>10.1f} {r['q_p95_us']:>10.1f} {r['qps_1t']:>9.0f} "
          f"{r['qps_nt']:>10.0f} {r['term_recall_at_k']:>9.3f} {r['phrase_recall_at_k']:>9.3f} "
          f"{r.get('or_recall_at_k', float('nan')):>9.3f} {idx:>9.1f}")
for p in sorted(glob.glob(os.path.join(d, "overlap_*.txt"))):
    name = os.path.basename(p)[:-4]
    print(f"  {name} (L<->Luc L<->Tan Luc<->Tan) = {open(p).read().strip()}")
PY
echo ""
echo "results: $RESULT_DIR"
