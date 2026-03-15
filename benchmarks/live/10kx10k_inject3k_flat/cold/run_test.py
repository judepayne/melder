#!/usr/bin/env python3
"""Cold live benchmark — 10k × 10k, flat backend, 3k events at c=10.

Clears cache, WAL and crossmap, starts the server, waits for it to be
ready, injects 3,000 events across 10 concurrent workers, then stops.
Run from the project root:
    python3 benchmarks/live/10kx10k_inject3k_flat/cold/run_test.py
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/live/10kx10k_inject3k_flat/cold"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8090
ITERATIONS = 3000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120  # seconds


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "listening on port" in f.read():
                    return True
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release")
        sys.exit(1)

    # Cold: clear everything
    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_bench_{os.getpid()}.log"
    print(f"=== Cold live run: {TEST_DIR} ===")
    print("Starting server...")

    server = subprocess.Popen(
        [
            args.binary,
            "serve",
            "--config",
            f"{TEST_DIR}/config.yaml",
            "--port",
            str(PORT),
        ],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(log_path, SERVER_READY_TIMEOUT):
        print(f"ERROR: Server did not start within {SERVER_READY_TIMEOUT}s.")
        print("Server log:")
        with open(log_path) as f:
            print(f.read())
        server.terminate()
        sys.exit(1)

    print("Server ready. Startup log:")
    with open(log_path) as f:
        for line in f:
            print(f"  {line}", end="")

    print(f"\nInjecting {ITERATIONS} events at concurrency {CONCURRENCY}...\n")
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/live_concurrent_test.py",
            "--no-serve",
            "--port",
            str(PORT),
            "--iterations",
            str(ITERATIONS),
            "--concurrency",
            str(CONCURRENCY),
            "--a-path",
            A_DATA,
            "--b-path",
            B_DATA,
        ]
    )

    server.terminate()
    server.wait()
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
