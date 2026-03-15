#!/usr/bin/env python3
"""Cold batch benchmark — 100k × 100k, usearch backend, quantized ONNX model.

Identical to the plain usearch 100k cold baseline except performance.quantized: true,
which switches to the INT8 ONNX model variant and roughly halves encoding time.
Compare index build time and total wall time against:
    benchmarks/batch/100kx100k_usearch/cold/

Run from the project root:
    python3 benchmarks/batch/100kx100k_usearch_quantized/cold/run_test.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/100kx100k_usearch_quantized/cold"
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

    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(f"=== Cold batch run (quantized): {TEST_DIR} ===")
    print(
        "Note: index build with quantized=true should take ~half the time of the plain baseline.\n"
    )
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
