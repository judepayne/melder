#!/usr/bin/env python3
"""Warm batch benchmark — 100k × 100k, usearch backend, mmap vector index.

Identical to the plain usearch 100k warm baseline except
performance.vector_index_mode: "mmap", which memory-maps the index file
instead of loading it fully into RAM. The OS pages graph nodes in on demand.

On first run the cache is empty so the index will be built (slow, ~3.5 min).
Run again afterwards for the true warm measurement, then compare against:
    benchmarks/batch/100kx100k_usearch/warm/

Expected trade-off:
  - Lower peak RAM (OS only keeps hot pages resident)
  - Slower throughput on first warm run (cold OS page cache, many page faults)
  - Throughput converges toward the load baseline once pages are cached by the OS

Run from the project root:
    python3 benchmarks/batch/100kx100k_usearch_mmap/warm/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/100kx100k_usearch_mmap/warm"
BINARY_DEFAULT = "./target/release/meld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    cache_dir = f"{TEST_DIR}/cache"
    cache_populated = os.path.exists(cache_dir) and bool(os.listdir(cache_dir))
    if not cache_populated:
        print("Note: cache is empty — this run will build the index (slow, ~3.5 min).")
        print("Run again afterwards for a true warm measurement.\n")

    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Warm batch run (mmap): {TEST_DIR} ===\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
