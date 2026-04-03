#!/usr/bin/env bash
#
# Generate all synthetic datasets required by benchmarks/live/ benchmarks.
#
# Sizes needed:
#   10k   — dataset_{a,b}_10k.{csv,jsonl,parquet} + ground_truth_crossmap_10k.csv
#   100k  — dataset_{a,b}_100k.{csv,jsonl,parquet} + ground_truth_crossmap_100k.csv
#   1M    — dataset_{a,b}_1000k.{csv,jsonl,parquet} + ground_truth_crossmap_1000k.csv
#
# All output goes to benchmarks/data/.  The 10k datasets are committed in git
# so they already exist after a clone; this script regenerates them for
# completeness (same seed = identical output).
#
# No live benchmark generates its own data — all expect pre-existing CSVs.
#
# Prerequisites:
#   pip install faker pandas pyarrow
#
# Usage:
#   ./benchmarks/live/generate_data.sh              # all sizes
#   ./benchmarks/live/generate_data.sh 10k           # just 10k
#   ./benchmarks/live/generate_data.sh 10k 100k      # 10k and 100k
#   ./benchmarks/live/generate_data.sh 1M            # just 1M (slow)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$PROJECT_ROOT/benchmarks/data/generate.py"
OUT_DIR="$PROJECT_ROOT/benchmarks/data"

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERROR: Generator script not found at $GENERATOR"
    exit 1
fi

# Check Python dependencies
python3 -c "import faker, pandas, pyarrow" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies. Install with:"
    echo "  pip install faker pandas pyarrow"
    exit 1
}

# Default: generate all sizes
SIZES=("${@:-10k 100k 1M}")
if [[ $# -eq 0 ]]; then
    SIZES=(10k 100k 1M)
fi

generate() {
    local label="$1"
    local n="$2"

    echo ""
    echo "========================================"
    echo "  Generating $label datasets ($n records each)"
    echo "========================================"
    echo ""

    python3 "$GENERATOR" --out "$OUT_DIR" --size "$n" --addresses
    echo ""
    echo "Done: $label"
}

for size in "${SIZES[@]}"; do
    case "$size" in
        10k|10K)
            generate "10k" 10000
            ;;
        100k|100K)
            generate "100k" 100000
            ;;
        1m|1M)
            generate "1M" 1000000
            ;;
        *)
            echo "ERROR: Unknown size '$size'. Valid sizes: 10k, 100k, 1M"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo "  All requested datasets generated in:"
echo "  $OUT_DIR"
echo "========================================"
echo ""
ls -lh "$OUT_DIR"/dataset_*.csv "$OUT_DIR"/ground_truth_*.csv 2>/dev/null
