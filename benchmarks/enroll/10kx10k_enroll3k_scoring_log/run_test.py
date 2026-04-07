#!/usr/bin/env python3
"""Enroll output test — 10k pool with 3k enrollments, event-sourced output.

Pre-loads 10k records, then enrolls 3k from dataset_b (remapped to pool
field names). After enrollment, sends POST /admin/shutdown for a clean
shutdown, then uses `meld export` to build CSVs and SQLite DB from the
match log (and scoring log, if enabled in the config).

Run from the project root:
    python3 benchmarks/enroll/10kx10k_enroll3k_scoring_log/run_test.py
"""

import argparse
import csv
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

TEST_DIR = "benchmarks/enroll/10kx10k_enroll3k_scoring_log"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8092
ITERATIONS = 3000
CONCURRENCY = 10
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120
OUTPUT_DIR = f"{TEST_DIR}/output"

# Map dataset_b field names to pool (dataset_a) field names.
FIELD_MAP = {
    "counterparty_id": "entity_id",
    "counterparty_name": "legal_name",
    "domicile": "country_code",
    "lei_code": "lei",
}


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


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
    url = f"http://127.0.0.1:{port}/admin/shutdown"
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        resp = urllib.request.urlopen(req, timeout=30)
        print(f"  /admin/shutdown response: {resp.status} {resp.read().decode()}")
        return True
    except urllib.error.URLError as e:
        print(f"  /admin/shutdown failed: {e}")
        return False


def post_json(base_url, path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return elapsed_ms, r.status, body
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return elapsed_ms, exc.code, body
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return elapsed_ms, 0, {"error": str(exc)}


def remap_record(record):
    """Remap dataset_b fields to pool (dataset_a) field names."""
    return {FIELD_MAP.get(k, k): v for k, v in record.items()}


def load_b_records(path, limit):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            records.append(remap_record(row))
            if len(records) >= limit:
                break
    return records


def worker(base_url, work_queue, results, results_lock):
    while True:
        item = work_queue.get()
        if item is None:
            work_queue.task_done()
            break
        elapsed, code, body = post_json(base_url, "/api/v1/enroll", {"record": item})
        edges = len(body.get("edges", [])) if isinstance(body, dict) else 0
        with results_lock:
            results.append((elapsed, code, edges))
        work_queue.task_done()


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

    if not os.path.exists(B_DATA):
        print("Test data not found. Generate with:")
        print("  python3 benchmarks/data/generate.py --a-size 10000 --b-size 10000")
        sys.exit(1)

    # Clean up from previous runs
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)

    log_path = f"/tmp/meld_enroll_scoring_log_test_{os.getpid()}.log"
    print(f"=== Enroll output test: {TEST_DIR} ===")
    print("Starting enroll server...")

    server = subprocess.Popen(
        [
            args.binary,
            "enroll",
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

    # Load B records, remap to pool field names
    records = load_b_records(B_DATA, ITERATIONS)
    print(f"\nEnrolling {len(records)} records at concurrency {CONCURRENCY}...\n")

    # Build work queue
    wq = queue.Queue()
    for rec in records:
        wq.put(rec)
    for _ in range(CONCURRENCY):
        wq.put(None)

    results = []
    results_lock = threading.Lock()

    t0 = time.perf_counter()
    threads = []
    for _ in range(CONCURRENCY):
        t = threading.Thread(
            target=worker,
            args=(f"http://127.0.0.1:{PORT}", wq, results, results_lock),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    wall_time = time.perf_counter() - t0

    # Print enrollment stats
    latencies = [r[0] for r in results]
    codes = [r[1] for r in results]
    edge_counts = [r[2] for r in results]
    ok_count = sum(1 for c in codes if c == 200)
    err_count = len(codes) - ok_count
    total_edges = sum(edge_counts)
    avg_edges = total_edges / len(edge_counts) if edge_counts else 0

    print(f"Enrollment complete:")
    print(f"  Records:    {ok_count}  (errors: {err_count})")
    print(f"  Edges:      {total_edges} (avg {avg_edges:.1f}/record)")
    print(f"  Throughput: {ok_count / wall_time:.0f} enrollments/sec")
    print(f"  Latency:    p50={percentile(latencies, 50):.1f}ms  "
          f"p95={percentile(latencies, 95):.1f}ms  "
          f"max={max(latencies):.1f}ms")

    if err_count > 0:
        print("ERROR: Some enrollments failed.")
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

    rel_csv = os.path.join(OUTPUT_DIR, "relationships.csv")
    if os.path.exists(rel_csv):
        with open(rel_csv) as f:
            rel_count = sum(1 for _ in f) - 1
        print(f"  relationships.csv: {rel_count} rows")
    else:
        print("  FAIL: relationships.csv not found")
        ok = False

    unmatched_csv = os.path.join(OUTPUT_DIR, "unmatched.csv")
    if os.path.exists(unmatched_csv):
        with open(unmatched_csv) as f:
            unmatched_count = sum(1 for _ in f) - 1
        print(f"  unmatched.csv:     {unmatched_count} rows")
    else:
        print("  FAIL: unmatched.csv not found")
        ok = False

    cand_csv = os.path.join(OUTPUT_DIR, "candidates.csv")
    if os.path.exists(cand_csv):
        with open(cand_csv) as f:
            cand_count = sum(1 for _ in f) - 1
        print(f"  candidates.csv:    {cand_count} rows")
    elif has_scoring_log:
        print("  WARNING: candidates.csv not found despite scoring log being enabled")
    else:
        print("  candidates.csv:    not produced (scoring log disabled)")

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
        print("PASS: Enroll output test completed successfully.")
        print(f"  Output files are in {OUTPUT_DIR}/ for inspection.")
    else:
        print("FAIL: Some required output files are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
