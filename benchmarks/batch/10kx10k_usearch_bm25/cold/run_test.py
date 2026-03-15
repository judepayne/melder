#!/usr/bin/env python3
"""Cold batch benchmark — 10k × 10k, usearch + BM25 re-ranking.

Adds method: bm25 as a 5% scoring term alongside the embedding field (which
drops from weight 0.55 to 0.50 to keep weights summing to 1.0). The pipeline
runs: ANN (ann_candidates: 40) → BM25 re-rank (bm25_candidates: 20) → full
scoring (top_n: 20).

Requires the bm25 feature:
    cargo build --release --features usearch,bm25

Compare throughput and auto-match count against:
    benchmarks/batch/10kx10k_usearch/cold/

Run from the project root:
    python3 benchmarks/batch/10kx10k_usearch_bm25/cold/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/10kx10k_usearch_bm25/cold"
BINARY_DEFAULT = "./target/release/meld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=BINARY_DEFAULT,
        help="Path to the meld binary (needs --features usearch,bm25)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch,bm25")
        sys.exit(1)

    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Cold batch run (usearch + BM25): {TEST_DIR} ===\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
