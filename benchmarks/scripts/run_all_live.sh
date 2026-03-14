#!/usr/bin/env bash
#
# bench/run_all.sh — Run all 12 live benchmark scenarios.
#
# Covers: 10k + 100k datasets, flat + usearch backends,
#         c=1 + c=10 concurrency, 80% + 40% encoding load.
#
# Must be run from the project root:
#   bash bench/run_all.sh
#
# Requires:
#   - ./target/release/meld built with --features usearch
#   - testdata/dataset_{a,b}_{10k,100k}.csv present
#   - python3 and curl on PATH

set -euo pipefail

BINARY="./target/release/meld"
PORT=18091
RESULTS="/tmp/bench_all_results.txt"
> "${RESULTS}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run() {
    local label="$1" config="$2" conc="$3" enc_pct="$4" a="$5" b="$6"

    echo ""
    echo "======== ${label} ========"

    # Start server, wait for health (up to 300s for flat 100k cold build)
    "${BINARY}" serve --config "${config}" --port "${PORT}" > /tmp/bs.log 2>&1 &
    local pid=$!
    local waited=0
    while ! curl -sf "http://localhost:${PORT}/api/v1/health" >/dev/null 2>&1; do
        sleep 1; waited=$((waited+1))
        if [ ${waited} -ge 300 ]; then
            echo "TIMEOUT"; tail -5 /tmp/bs.log
            kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null
            return 1
        fi
    done
    echo "  Server ready in ${waited}s"

    # Run benchmark
    local out
    out=$(python3 bench/live_concurrent_test.py \
        --no-serve --port "${PORT}" \
        --iterations 3000 --concurrency "${conc}" \
        --encoding-pct "${enc_pct}" \
        --a-path "${a}" --b-path "${b}" --seed 42 2>&1)

    # Extract summary lines
    local all_line throughput_line
    all_line=$(echo "${out}" | grep "^ALL" || echo "N/A")
    throughput_line=$(echo "${out}" | grep "req/s$" || echo "N/A")

    echo "  ${all_line}"
    echo "  ${throughput_line}"

    printf "%-40s | %s | %s\n" "${label}" "${all_line}" "${throughput_line}" \
        >> "${RESULTS}"

    kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null
    sleep 2
}

# Reset WAL and crossmap state between scenarios so each run starts clean.
reset_state() {
    rm -f bench/live_upserts_*Z.ndjson bench/live_upserts_100k_*Z.ndjson 2>/dev/null || true
    head -1 bench/crossmap_live.csv     > /tmp/cml.csv   && cp /tmp/cml.csv   bench/crossmap_live.csv
    head -1 bench/crossmap_live_100k.csv > /tmp/cml100.csv && cp /tmp/cml100.csv bench/crossmap_live_100k.csv
}

# ---------------------------------------------------------------------------
# Dataset and config paths
# ---------------------------------------------------------------------------

A10="testdata/dataset_a_10k.csv"
B10="testdata/dataset_b_10k.csv"
A100="testdata/dataset_a_100k.csv"
B100="testdata/dataset_b_100k.csv"

FLAT10="testdata/configs/bench_live.yaml"
FLAT100="testdata/configs/bench_live_100k_flat.yaml"
USEARCH10="testdata/configs/bench_live_usearch.yaml"
USEARCH100="testdata/configs/bench_live_100k.yaml"

# ---------------------------------------------------------------------------
# Benchmark run order
#
# The flat and usearch backends share the same cache path (keyed by field
# spec hash, not backend). Running usearch first, then wiping the .usearchdb
# directories lets flat load/build the .index files cleanly without needing
# its own separate cold build pass.
#
# 10k and 100k also share the same hash, so we wipe and rebuild when
# switching dataset sizes.
# ---------------------------------------------------------------------------

echo "Running all 12 live benchmark scenarios..."
echo "(~15 min total; first 100k run will rebuild caches if not warm)"
echo ""

# --- 100k usearch ---
reset_state
run "100k usearch c=1,  80% enc" "${USEARCH100}" 1  80 "${A100}" "${B100}"
reset_state
run "100k usearch c=10, 80% enc" "${USEARCH100}" 10 80 "${A100}" "${B100}"

# --- 100k flat (wipe usearchdb, keep .index for flat to use) ---
rm -rf bench/cache/*.usearchdb
reset_state
run "100k flat c=1,  80% enc"    "${FLAT100}"    1  80 "${A100}" "${B100}"
reset_state
run "100k flat c=10, 80% enc"    "${FLAT100}"    10 80 "${A100}" "${B100}"

# --- 10k usearch (wipe caches: different dataset size, same hash) ---
rm -rf bench/cache/*.index bench/cache/*.usearchdb \
       bench/cache/*.manifest bench/cache/*.texthash
reset_state
run "10k usearch c=1,  80% enc"  "${USEARCH10}"  1  80 "${A10}" "${B10}"
reset_state
run "10k usearch c=10, 80% enc"  "${USEARCH10}"  10 80 "${A10}" "${B10}"
reset_state
run "10k usearch c=1,  40% enc"  "${USEARCH10}"  1  40 "${A10}" "${B10}"
reset_state
run "10k usearch c=10, 40% enc"  "${USEARCH10}"  10 40 "${A10}" "${B10}"

# --- 10k flat (wipe usearchdb, keep .index) ---
rm -rf bench/cache/*.usearchdb
reset_state
run "10k flat c=1,  80% enc"     "${FLAT10}"     1  80 "${A10}" "${B10}"
reset_state
run "10k flat c=10, 80% enc"     "${FLAT10}"     10 80 "${A10}" "${B10}"
reset_state
run "10k flat c=1,  40% enc"     "${FLAT10}"     1  40 "${A10}" "${B10}"
reset_state
run "10k flat c=10, 40% enc"     "${FLAT10}"     10 40 "${A10}" "${B10}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================"
echo "  ALL RESULTS"
echo "========================================"
cat "${RESULTS}"
