#!/usr/bin/env python3
"""Warm enroll benchmark — 10k pool, usearch, 3k enrollments at c=10.

Pre-loads 10k records from dataset_a into the pool at startup (no edges).
Then enrolls 3k records from dataset_b (remapped to pool field names) and
measures throughput and latency.

Preserves the embedding cache. Clears WAL for a clean run.
On first run the cache is empty so the server builds the index (slow).
Run again afterwards for a true warm measurement.

Run from the project root:
    python3 benchmarks/enroll/10kx10k_enroll3k_usearch/warm/run_test.py
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

TEST_DIR = "benchmarks/enroll/10kx10k_enroll3k_usearch/warm"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8091
ITERATIONS = 3000
CONCURRENCY = 10
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120

# Map dataset_b field names to pool (dataset_a) field names.
FIELD_MAP = {
    "counterparty_id": "entity_id",
    "counterparty_name": "legal_name",
    "domicile": "country_code",
    "lei_code": "lei",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
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


def post_json(base_url: str, path: str, payload: dict) -> tuple:
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


def remap_record(record: dict) -> dict:
    """Remap dataset_b fields to pool (dataset_a) field names."""
    out = {}
    for k, v in record.items():
        out[FIELD_MAP.get(k, k)] = v
    return out


def load_b_records(path: str, limit: int) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            records.append(remap_record(row))
            if len(records) >= limit:
                break
    return records


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker(base_url, work_queue, results, results_lock):
    while True:
        item = work_queue.get()
        if item is None:
            work_queue.task_done()
            break
        record = item
        elapsed, code, body = post_json(base_url, "/api/v1/enroll", {"record": record})
        edges = len(body.get("edges", [])) if isinstance(body, dict) else 0
        with results_lock:
            results.append((elapsed, code, edges))
        work_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    cache_dir = f"{TEST_DIR}/cache"
    if not (os.path.exists(cache_dir) and os.listdir(cache_dir)):
        print("Note: cache is empty — this run will build the index (slow).")
        print("Run again afterwards for a true warm measurement.\n")

    # Clean up WAL (keep cache for warm start)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)

    log_path = f"/tmp/meld_enroll_bench_{os.getpid()}.log"
    print(f"=== Warm enroll run: {TEST_DIR} ===")
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
    records = load_b_records(B_DATA, args.iterations)
    print(f"\nEnrolling {len(records)} records at concurrency {args.concurrency}...\n")

    # Build work queue
    wq = queue.Queue()
    for rec in records:
        wq.put(rec)
    for _ in range(args.concurrency):
        wq.put(None)  # poison pills

    results = []
    results_lock = threading.Lock()

    # Launch workers
    t0 = time.perf_counter()
    threads = []
    for _ in range(args.concurrency):
        t = threading.Thread(
            target=worker,
            args=(f"http://localhost:{PORT}", wq, results, results_lock),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    wall_time = time.perf_counter() - t0

    # Collect results
    latencies = [r[0] for r in results]
    codes = [r[1] for r in results]
    edge_counts = [r[2] for r in results]
    ok_count = sum(1 for c in codes if c == 200)
    err_count = len(codes) - ok_count
    total_edges = sum(edge_counts)
    avg_edges = total_edges / len(edge_counts) if edge_counts else 0

    print(f"Enroll benchmark complete:")
    print(f"  Records enrolled: {ok_count}")
    print(f"  Errors:           {err_count}")
    print(f"  Total edges:      {total_edges} (avg {avg_edges:.1f} per record)")
    print(f"  Wall time:        {wall_time:.1f}s")
    print(f"  Throughput:       {ok_count / wall_time:.0f} enrollments/sec")
    print()
    print(f"  Latency (ms):")
    print(f"    p50:  {percentile(latencies, 50):.1f}")
    print(f"    p90:  {percentile(latencies, 90):.1f}")
    print(f"    p95:  {percentile(latencies, 95):.1f}")
    print(f"    p99:  {percentile(latencies, 99):.1f}")
    print(f"    max:  {max(latencies):.1f}")

    # Shutdown
    server.terminate()
    server.wait()

    print(f"\nTotal wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
