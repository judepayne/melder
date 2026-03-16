#!/usr/bin/env python3
"""Warm live benchmark — 1M × 1M, usearch backend, 10k events at c=10.

Preserves the cache. Clears WAL and crossmap for a clean injection.
On first run the cache is empty so the server builds the embedding index
from scratch — this will take a VERY long time (~30-60 min for 2M records).
On subsequent runs the server loads from cache and starts in seconds.

WARNING: Memory usage will be high (~8-10 GB for records + blocking index
+ vector index for 2M records).

Requires the usearch feature:
    cargo build --release --features usearch

Run from the project root:
    python3 benchmarks/live/1Mx1M_inject10k_usearch/warm/run_test.py
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/live/1Mx1M_inject10k_usearch/warm"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8090
ITERATIONS = 10000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_1000k.csv"
B_DATA = "benchmarks/data/dataset_b_1000k.csv"
# 1M records: cold index build can take 30-60 min; warm loads in seconds
SERVER_READY_TIMEOUT = 7200  # 2 hours for cold build


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "listening on port" in f.read():
                    return True
        time.sleep(5)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    # Check datasets exist
    for ds in [A_DATA, B_DATA]:
        if not os.path.exists(ds):
            print(f"Dataset not found: {ds}")
            print("Generate with: python3 benchmarks/data/generate.py --size 1000000")
            sys.exit(1)

    cache_dir = f"{TEST_DIR}/cache"
    is_cold = not (os.path.exists(cache_dir) and os.listdir(cache_dir))
    if is_cold:
        print("Note: cache is empty — this run will build the embedding index.")
        print("This will take a VERY long time (~30-60 min for 1M × 1M).")
        print("Run again afterwards for a true warm measurement.\n")
    else:
        print("Cache found — this should be a warm start.\n")

    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_bench_{os.getpid()}.log"
    print(
        f"=== {'Cold' if is_cold else 'Warm'} live run (1M × 1M, usearch): {TEST_DIR} ==="
    )
    print("Starting server...")

    startup_start = time.time()
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
        with open(log_path) as f:
            print(f.read()[-2000:])  # last 2K chars
        server.terminate()
        sys.exit(1)

    startup_elapsed = time.time() - startup_start
    print(f"Server ready. Startup time: {startup_elapsed:.1f}s")
    print("Startup log:")
    with open(log_path) as f:
        for line in f:
            print(f"  {line}", end="")

    print(
        f"\nInjecting {args.iterations} events at concurrency {args.concurrency}...\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/live_concurrent_test.py",
            "--no-serve",
            "--port",
            str(PORT),
            "--iterations",
            str(args.iterations),
            "--concurrency",
            str(args.concurrency),
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
