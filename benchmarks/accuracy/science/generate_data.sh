#!/usr/bin/env bash
#
# Generate the training-round B-side datasets for accuracy/science/ experiments.
#
# The science directory has its own data, separate from benchmarks/data/:
#   master/dataset_a.csv   — committed to git (seed 0, 10k records)
#   holdout/dataset_b.csv  — committed to git (seed 9999)
#   rounds/round_N/dataset_b.csv — GENERATED HERE (seeds 100..126, 27 rounds)
#
# This script wraps setup_datasets.py, which calls
# generate_b_from_master_1to1() from benchmarks/data/generate.py
# to produce one B dataset per training round against the fixed A master.
#
# Prerequisites:
#   pip install faker pandas pyarrow
#
# Usage:
#   ./benchmarks/accuracy/science/generate_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SETUP_SCRIPT="$SCRIPT_DIR/setup_datasets.py"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo "ERROR: setup_datasets.py not found at $SETUP_SCRIPT"
    exit 1
fi

# Check that the committed master/holdout files exist
if [[ ! -f "$SCRIPT_DIR/master/dataset_a.csv" ]]; then
    echo "ERROR: master/dataset_a.csv not found."
    echo "       This file should be committed in git. Try: git checkout -- benchmarks/accuracy/science/master/"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/holdout/dataset_b.csv" ]]; then
    echo "ERROR: holdout/dataset_b.csv not found."
    echo "       This file should be committed in git. Try: git checkout -- benchmarks/accuracy/science/holdout/"
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
echo "  Generating training-round B datasets"
echo "  (27 rounds, seeds 100..126, 10k each)"
echo "========================================"
echo ""
echo "  Master A: $SCRIPT_DIR/master/dataset_a.csv"
echo "  Holdout B: $SCRIPT_DIR/holdout/dataset_b.csv (committed, not regenerated)"
echo "  Output:    $SCRIPT_DIR/rounds/round_N/dataset_b.csv"
echo ""

python3 "$SETUP_SCRIPT"

echo ""
echo "========================================"
echo "  Done. Round datasets generated in:"
echo "  $SCRIPT_DIR/rounds/"
echo "========================================"
echo ""
ls -lh "$SCRIPT_DIR"/rounds/round_*/dataset_b.csv 2>/dev/null | head -5
TOTAL=$(ls "$SCRIPT_DIR"/rounds/round_*/dataset_b.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  ... ($TOTAL rounds total)"
