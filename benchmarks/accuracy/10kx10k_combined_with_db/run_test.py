#!/usr/bin/env python3
"""Batch accuracy benchmark — 10k × 10k, BM25 + embeddings, CSV + SQLite DB output.

Clears cache and outputs, runs a full batch job producing both CSVs and SQLite DB.
Run from the project root:
    python3 benchmarks/accuracy/10kx10k_combined_with_db/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/accuracy/10kx10k_combined_with_db"
BINARY_DEFAULT = "./target/release/meld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release")
        sys.exit(1)

    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Batch run: {TEST_DIR} ===\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Report output files
    output_dir = f"{TEST_DIR}/output"
    for f in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path)
        if size > 1024 * 1024:
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {f}: {size / 1024:.1f} KB")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
