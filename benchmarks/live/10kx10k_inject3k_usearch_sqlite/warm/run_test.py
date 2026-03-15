#!/usr/bin/env python3
"""Warm live benchmark — 10k × 10k, usearch + SQLite store.

Two passes are needed:
  Pass 1 (first run):  Populates the SQLite DB from CSV + injection events.
                       Startup behaves like a cold run (~18s index build).
  Pass 2 (second run): Opens the existing DB directly — no CSV load, no WAL
                       replay, no re-encoding. Startup should be ~1-2s.

This test measures two things independently:
  1. Warm restart time (startup): should be near-instant vs ~1.7s for the
     WAL-based memory store warm baseline.
  2. Request throughput: compare against the memory store baseline
     (benchmarks/live/10kx10k_inject3k_usearch/warm/) to quantify the
     SQLite write-through cost per upsert.

The WAL and crossmap are cleared between runs (the DB is the durable state),
but the embedding cache and SQLite DB are preserved.

Run from the project root:
    python3 benchmarks/live/10kx10k_inject3k_usearch_sqlite/warm/run_test.py
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/live/10kx10k_inject3k_usearch_sqlite/warm"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8090
ITERATIONS = 10000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120


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
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    db_path = f"{TEST_DIR}/live.db"
    db_exists = os.path.exists(db_path)

    if not db_exists:
        print("Note: SQLite DB not found — this run will build it from scratch (slow).")
        print("Run again afterwards for the true warm measurement.\n")

    # Preserve DB and embedding cache; clear WAL, crossmap, and outputs only
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_bench_{os.getpid()}.log"
    print(f"=== Warm live run (SQLite): {TEST_DIR} ===")
    if db_exists:
        print(
            "SQLite DB found — server will open it directly (no CSV load or WAL replay)."
        )
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
