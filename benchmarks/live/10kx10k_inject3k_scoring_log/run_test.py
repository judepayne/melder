#!/usr/bin/env python3
"""Live output test — 10k x 10k with 3k injection, event-sourced output.

After injection, sends POST /admin/shutdown for a clean shutdown, then
uses `meld export` to build CSVs and SQLite DB from the match log (and
scoring log, if enabled in the config).

Run from the project root:
    python3 benchmarks/live/10kx10k_inject3k_scoring_log/run_test.py
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/live/10kx10k_inject3k_scoring_log"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8091
ITERATIONS = 3000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120
OUTPUT_DIR = f"{TEST_DIR}/output"


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "server listening" in f.read():
                    return True
        time.sleep(2)
    return False


def shutdown_server(port):
    """Send POST /admin/shutdown and wait for the process to exit."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/admin/shutdown"
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        resp = urllib.request.urlopen(req, timeout=30)
        print(f"  /admin/shutdown response: {resp.status} {resp.read().decode()}")
        return True
    except urllib.error.URLError as e:
        print(f"  /admin/shutdown failed: {e}")
        return False


def find_scoring_log():
    """Find the scoring log file if one was produced."""
    for pattern in [
        f"{OUTPUT_DIR}/*.scoring_log.ndjson.zst",
        f"{OUTPUT_DIR}/*.scoring_log.ndjson",
        f"{OUTPUT_DIR}/*.ndjson.zst",
    ]:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release")
        sys.exit(1)

    if not os.path.exists(A_DATA) or not os.path.exists(B_DATA):
        print("Test data not found. Generate with:")
        print("  python3 benchmarks/data/generate.py --a-size 10000 --b-size 10000")
        sys.exit(1)

    # Clean up from previous runs
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_scoring_log_test_{os.getpid()}.log"
    print(f"=== Live output test: {TEST_DIR} ===")
    print("Starting server (initial match pass enabled)...")

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
            print(f.read())
        server.terminate()
        sys.exit(1)

    print("Server ready. Startup log:")
    with open(log_path) as f:
        for line in f:
            print(f"  {line}", end="")

    print(f"\nInjecting {ITERATIONS} events at concurrency {CONCURRENCY}...\n")
    inject_result = subprocess.run(
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

    if inject_result.returncode != 0:
        print("ERROR: Injection failed.")
        server.terminate()
        server.wait()
        sys.exit(1)

    # --- Graceful shutdown via /admin/shutdown ---
    print("\nSending /admin/shutdown...")
    shutdown_server(PORT)

    try:
        server.wait(timeout=30)
        print(f"  Server exited with code {server.returncode}")
    except subprocess.TimeoutExpired:
        print("  WARNING: Server did not exit within 30s, terminating.")
        server.terminate()
        server.wait()

    # --- Check what was produced ---
    print("\n=== Checking log output ===")

    match_log_files = glob.glob(f"{TEST_DIR}/wal/*.ndjson")
    if not match_log_files:
        print("FAIL: No match log file found in wal/")
        sys.exit(1)
    ml_file = match_log_files[0]
    print(f"  Match log:   {ml_file} ({os.path.getsize(ml_file):,} bytes)")

    sl_file = find_scoring_log()
    has_scoring_log = sl_file is not None
    if has_scoring_log:
        print(f"  Scoring log: {sl_file} ({os.path.getsize(sl_file):,} bytes)")
    else:
        print("  Scoring log: not produced (scoring_log.enabled = false)")

    # --- Run meld export to build outputs ---
    print("\n=== Running meld export ===")
    export_result = subprocess.run(
        [
            args.binary,
            "export",
            "--config",
            f"{TEST_DIR}/config.yaml",
            "--out-dir",
            OUTPUT_DIR,
        ],
        capture_output=True,
        text=True,
    )
    print(export_result.stdout)
    if export_result.stderr:
        print(export_result.stderr)
    if export_result.returncode != 0:
        print(f"FAIL: meld export exited with code {export_result.returncode}")
        sys.exit(1)

    # --- Validate output files ---
    print("=== Validating output files ===")
    ok = True

    # relationships.csv (always expected)
    rel_csv = os.path.join(OUTPUT_DIR, "relationships.csv")
    if os.path.exists(rel_csv):
        with open(rel_csv) as f:
            rel_count = sum(1 for _ in f) - 1
        print(f"  relationships.csv: {rel_count} rows")
    else:
        print("  FAIL: relationships.csv not found")
        ok = False

    # unmatched.csv (always expected)
    unmatched_csv = os.path.join(OUTPUT_DIR, "unmatched.csv")
    if os.path.exists(unmatched_csv):
        with open(unmatched_csv) as f:
            unmatched_count = sum(1 for _ in f) - 1
        print(f"  unmatched.csv:     {unmatched_count} rows")
    else:
        print("  FAIL: unmatched.csv not found")
        ok = False

    # candidates.csv (only when scoring log enabled)
    cand_csv = os.path.join(OUTPUT_DIR, "candidates.csv")
    if os.path.exists(cand_csv):
        with open(cand_csv) as f:
            cand_count = sum(1 for _ in f) - 1
        print(f"  candidates.csv:    {cand_count} rows")
    elif has_scoring_log:
        print("  WARNING: candidates.csv not found despite scoring log being enabled")
    else:
        print("  candidates.csv:    not produced (scoring log disabled)")

    # SQLite DB
    db_path = f"{OUTPUT_DIR}/results.db"
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path)
        print(f"  results.db:        {db_size:,} bytes")

        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type"
            ).fetchall()
            for rtype, count in rows:
                print(f"    {rtype}: {count}")

            fs_count = conn.execute("SELECT COUNT(*) FROM field_scores").fetchone()[0]
            print(f"    field_scores rows: {fs_count}")
            if has_scoring_log and fs_count == 0:
                print("  WARNING: field_scores empty despite scoring log being enabled")
        finally:
            conn.close()
    else:
        print("  results.db:        not produced")

    print()
    if ok:
        print("PASS: Live output test completed successfully.")
        print(f"  Output files are in {OUTPUT_DIR}/ for inspection.")
    else:
        print("FAIL: Some required output files are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
