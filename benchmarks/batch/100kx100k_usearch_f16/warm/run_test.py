#!/usr/bin/env python3
"""Warm batch benchmark — 100k × 100k, usearch backend, f16 vector quantization.

Identical to the plain usearch 100k warm baseline except
performance.vector_quantization: f16, which halves the on-disk cache size
(~43% smaller) with negligible impact on throughput or match quality.

On first run the cache is empty so the index will be built (slow — ~3.5 min).
Run again afterwards for the true warm measurement, then compare:
  - Cache file sizes: du -sh benchmarks/batch/100kx100k_usearch_f16/warm/cache/
  - vs f32 baseline:  du -sh benchmarks/batch/100kx100k_usearch/warm/cache/
  - Throughput and wall time should be within noise of the f32 baseline.

Run from the project root:
    python3 benchmarks/batch/100kx100k_usearch_f16/warm/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/100kx100k_usearch_f16/warm"
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

    print(f"=== Warm batch run (f16): {TEST_DIR} ===\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")

    if cache_populated:
        # Report cache sizes for comparison
        import shutil as sh

        f16_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(cache_dir)
            for f in files
        )
        print(f"f16 cache size: {f16_size / 1024 / 1024:.1f} MB")
        f32_cache = "benchmarks/batch/100kx100k_usearch/warm/cache"
        if os.path.exists(f32_cache):
            f32_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(f32_cache)
                for f in files
            )
            print(f"f32 cache size: {f32_size / 1024 / 1024:.1f} MB  (baseline)")
            print(f"Reduction:      {(1 - f16_size / f32_size) * 100:.0f}%")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
