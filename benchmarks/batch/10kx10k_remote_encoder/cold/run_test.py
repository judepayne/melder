#!/usr/bin/env python3
"""Cold batch benchmark — 10k × 10k with SubprocessEncoder (remote encoder).

Exercises every new config knob introduced in the RemoteEncoder design:

    embeddings.remote_encoder_cmd
    performance.encoder_pool_size         (required when remote is set)
    performance.encoder_call_timeout_ms
    XOR with embeddings.model             (config must NOT set a `model` key)

Uses tests/fixtures/stub_encoder.py as the "remote" encoder — deterministic
hash-based vectors with a simulated 5ms latency per call. This is not a
real remote service benchmark; it measures the end-to-end overhead of the
subprocess encoder path (IPC framing, pool dispatch, vector marshalling).

Run from the project root:

    cargo build --release
    python3 benchmarks/batch/10kx10k_remote_encoder/cold/run_test.py

Compared to benchmarks/batch/10kx10k_flat/cold (local ONNX) the wall time
here reflects stub latency + subprocess IPC overhead, NOT the cost of a
real embedding model. A production remote encoder will be dominated by
network round-trips to the central embedding service.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/10kx10k_remote_encoder/cold"
BINARY_DEFAULT = "./target/release/meld"


def check_python_available():
    """The stub requires python3 on PATH. Fail loudly if missing."""
    try:
        r = subprocess.run(
            ["python3", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Using {r.stdout.strip()} for stub encoder")
    except Exception as e:
        print(f"ERROR: python3 not available: {e}")
        print("The remote encoder stub requires Python 3.")
        sys.exit(1)


def cold_clean():
    """Wipe cache, output, and crossmap for a deterministic cold run."""
    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)


def assert_sanity(returncode: int):
    """Light sanity checks on the benchmark output."""
    if returncode != 0:
        print(f"meld exited non-zero ({returncode})")
        sys.exit(returncode)
    out_csv = f"{TEST_DIR}/output/relationships.csv"
    if not os.path.exists(out_csv):
        print(f"ERROR: expected output not written: {out_csv}")
        sys.exit(1)
    if os.path.getsize(out_csv) == 0:
        print(f"ERROR: output is empty: {out_csv}")
        sys.exit(1)
    print(f"Output written: {out_csv} ({os.path.getsize(out_csv)} bytes)")


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

    check_python_available()
    cold_clean()

    print(f"=== Cold batch run (remote encoder): {TEST_DIR} ===\n")
    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nTotal wall time: {elapsed:.1f}s")
    assert_sanity(result.returncode)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
