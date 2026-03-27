#!/usr/bin/env python3
"""Cold live benchmark — 10k × 10k, usearch backend, SQLite store.

Wipes the SQLite DB, embedding cache, WAL and crossmap, then starts the server
and injects 3,000 events at c=10. This is the DB-population run — the warm
restart test (../warm/run_test.py) reads this DB on its second pass.

The cold startup sequence is the same as the plain usearch baseline: load CSVs,
encode all records, build the vector index, then open the server. The DB is
populated during startup (records inserted into SQLite) and during the injection
(new records and crossmap pairs written through to SQLite).

Compare startup time and throughput against the memory-store baseline:
    benchmarks/live/10kx10k_inject3k_usearch/cold/

Run from the project root:
    python3 benchmarks/live/10kx10k_inject3k_usearch_sqlite/cold/run_test.py
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/live/10kx10k_inject3k_usearch_sqlite/cold"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8090
ITERATIONS = 3000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "server listening" in f.read():
                    return True
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    # Full cold wipe including the SQLite DB
    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    for f in [f"{TEST_DIR}/crossmap.csv", f"{TEST_DIR}/live.db"]:
        if os.path.exists(f):
            os.remove(f)

    log_path = f"/tmp/meld_bench_{os.getpid()}.log"
    print(f"=== Cold live run (SQLite): {TEST_DIR} ===")
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

    startup_start = time.time()
    if not wait_for_server(log_path, SERVER_READY_TIMEOUT):
        print(f"ERROR: Server did not start within {SERVER_READY_TIMEOUT}s.")
        with open(log_path) as f:
            print(f.read())
        server.terminate()
        sys.exit(1)
    startup_elapsed = time.time() - startup_start

    print("Server ready. Startup log:")
    with open(log_path) as f:
        for line in f:
            print(f"  {line}", end="")

    print(f"\nStartup time: {startup_elapsed:.1f}s")
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
