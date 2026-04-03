#!/usr/bin/env bash
#
# Generate the synthetic datasets required by benchmarks/accuracy/ benchmarks.
#
# Benchmarks that use pre-generated data from benchmarks/data/:
#   10kx10k_bm25          — dataset_{a,b}_10k + ground_truth_crossmap_10k
#   10kx10k_combined      — dataset_{a,b}_10k + ground_truth_crossmap_10k
#   10kx10k_embeddings    — dataset_{a,b}_10k + ground_truth_crossmap_10k
#   10kx10k_embeddings_bge — dataset_{a,b}_10k + ground_truth_crossmap_10k
#
# Excluded (generates its own data via run_test.py Phase 0):
#   10kx10k_exclusions    — calls generate.generate_with_seed() with n_exact=1000
#                           into its own data/ subdirectory
#
# Excluded (separate script):
#   science/              — has its own generate_data.sh
#
# Prerequisites:
#   pip install faker pandas pyarrow
#
# Usage:
#   ./benchmarks/accuracy/generate_data.sh

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

echo ""
echo "========================================"
echo "  Generating 10k datasets (10000 records each)"
echo "========================================"
echo ""
echo "  Used by: 10kx10k_bm25, 10kx10k_combined,"
echo "           10kx10k_embeddings, 10kx10k_embeddings_bge"
echo ""
echo "  NOTE: 10kx10k_exclusions is excluded — it generates"
echo "        its own data via run_test.py Phase 0."
echo ""

python3 "$GENERATOR" --out "$OUT_DIR" --size 10000 --addresses

echo ""
echo "========================================"
echo "  Done. Datasets generated in:"
echo "  $OUT_DIR"
echo "========================================"
echo ""
ls -lh "$OUT_DIR"/dataset_*_10k.csv "$OUT_DIR"/ground_truth_crossmap_10k.csv 2>/dev/null
