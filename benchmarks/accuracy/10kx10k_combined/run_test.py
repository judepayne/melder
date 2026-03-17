#!/usr/bin/env python3
"""Accuracy benchmark — 10k × 10k, BM25 + embeddings combined scoring (usearch).

First run is cold (encodes all records, builds HNSW index + BM25 index).
Subsequent runs are warm (loads cached embeddings + index).
After matching, evaluates accuracy against ground truth.

Requires: cargo build --release --features usearch,bm25

Run from the project root:
    python3 benchmarks/accuracy/10kx10k_combined/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/accuracy/10kx10k_combined"
BINARY_DEFAULT = "./target/release/meld"
EVAL_SCRIPT = "benchmarks/accuracy/eval.py"
GROUND_TRUTH = "benchmarks/data/ground_truth_crossmap_10k.csv"
DATASET_A = "benchmarks/data/dataset_a_10k.csv"
DATASET_B = "benchmarks/data/dataset_b_10k.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    parser.add_argument(
        "--cold", action="store_true", help="Force cold run (delete cache)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch,bm25")
        sys.exit(1)

    # Clean output and crossmap (always fresh for accuracy measurement)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    if args.cold:
        shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
        os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)

    warm = "cold" if args.cold or not os.path.exists(f"{TEST_DIR}/cache") else "warm"
    print(f"=== Accuracy: BM25 + Embeddings ({warm}) — {TEST_DIR} ===\n")

    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nBatch completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print("Batch run failed!")
        sys.exit(result.returncode)

    # Run evaluation
    print()
    subprocess.run(
        [
            sys.executable,
            EVAL_SCRIPT,
            "--results",
            f"{TEST_DIR}/output/results.csv",
            "--review",
            f"{TEST_DIR}/output/review.csv",
            "--unmatched",
            f"{TEST_DIR}/output/unmatched.csv",
            "--ground-truth",
            GROUND_TRUTH,
            "--dataset-a",
            DATASET_A,
            "--dataset-b",
            DATASET_B,
        ]
    )


if __name__ == "__main__":
    main()
