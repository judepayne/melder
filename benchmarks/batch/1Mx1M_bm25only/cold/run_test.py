#!/usr/bin/env python3
"""Cold batch benchmark — 1M × 1M, BM25-only (no embedding, no vector index).

No embedding fields are configured, so no ONNX model is loaded and no vector
index is built. BM25 serves as the sole candidate selection method, querying
the blocking index directly.

WARNING: At 1M records this will take a long time and significant memory
(~15-20 GB for the in-memory store + BM25 index + blocking index).

Requires the bm25 feature:
    cargo build --release --features bm25

Run from the project root:
    python3 benchmarks/batch/1Mx1M_bm25only/cold/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/1Mx1M_bm25only/cold"
BINARY_DEFAULT = "./target/release/meld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=BINARY_DEFAULT,
        help="Path to the meld binary (needs --features bm25)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N B records (for quick sanity checks)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features bm25")
        sys.exit(1)

    # Check dataset exists
    for ds in [
        "benchmarks/data/dataset_a_1000k.csv",
        "benchmarks/data/dataset_b_1000k.csv",
    ]:
        if not os.path.exists(ds):
            print(f"Dataset not found: {ds}")
            print("Generate with: python3 benchmarks/data/generate.py --size 1000000")
            sys.exit(1)

    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Cold batch run (1M × 1M, BM25-only, in-memory): {TEST_DIR} ===")
    print(
        "Note: no ONNX model — startup should be fast (data loading + BM25 index build)."
    )
    print("Warning: expect ~15-20 GB memory usage.\n")

    cmd = [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
