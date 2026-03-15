#!/usr/bin/env python3
"""Cold batch benchmark — 10k × 10k, BM25-only (no embedding, no vector index).

No embedding fields are configured, so no ONNX model is loaded and no vector
index is built. BM25 serves as the sole candidate selection method, querying
the blocking index directly. Startup should be near-instant even on a cold run.

The scoring equation mirrors the usearch baseline (same weights on fuzzy/exact)
but replaces the embedding term with BM25.

Requires the bm25 feature:
    cargo build --release --features bm25

Key things to compare against the usearch cold baseline
(benchmarks/batch/10kx10k_usearch/cold/):
  - Startup time: near-instant vs ~17s (no ONNX encoding)
  - Throughput: BM25 candidate selection is O(N) within a block but cheaper
    than ANN+scoring at this scale
  - Auto-match count: will differ — BM25 cannot find zero-token-overlap pairs
    that the embedding model handles (e.g. abbreviations, synonyms)

Run from the project root:
    python3 benchmarks/batch/10kx10k_bm25only/cold/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/10kx10k_bm25only/cold"
BINARY_DEFAULT = "./target/release/meld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=BINARY_DEFAULT,
        help="Path to the meld binary (needs --features bm25)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features bm25")
        sys.exit(1)

    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Cold batch run (BM25-only, no embedding): {TEST_DIR} ===")
    print("Note: no ONNX model — startup should be near-instant.\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
