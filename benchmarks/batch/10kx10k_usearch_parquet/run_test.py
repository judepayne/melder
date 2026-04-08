#!/usr/bin/env python3
"""Batch benchmark — 10k × 10k, usearch backend, parquet input + output.

Validates parquet end-to-end: parquet in, parquet out.
Run from the project root:
    python3 benchmarks/batch/10kx10k_usearch_parquet/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/10kx10k_usearch_parquet"
BINARY_DEFAULT = "./target/release/meld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features parquet-format")
        sys.exit(1)

    # Check parquet data files exist
    for f in ["benchmarks/data/dataset_a_10k.parquet",
              "benchmarks/data/dataset_b_10k.parquet"]:
        if not os.path.exists(f):
            print(f"Data file not found: {f}")
            print("Generate with: python3 benchmarks/data/generate.py --size 10000")
            sys.exit(1)

    # Clean state
    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Parquet in+out batch run: {TEST_DIR} ===\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Verify parquet output files exist
    if result.returncode == 0:
        for f in ["relationships.parquet", "unmatched.parquet"]:
            path = os.path.join(TEST_DIR, "output", f)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  {f}: {size:,} bytes")
            else:
                print(f"  MISSING: {f}")
                result.returncode = 1

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
