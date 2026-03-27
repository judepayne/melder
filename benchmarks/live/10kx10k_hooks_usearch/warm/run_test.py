#!/usr/bin/env python3
"""Warm hooks benchmark — 10k × 10k, usearch, 200 B-side adds.

Pre-loads 10k A + 10k B records from datasets at startup. Then injects
200 new B-side records via the API and observes hook events printed by
the hook script.

Preserves the embedding cache. Clears WAL and crossmap for a clean run.
On first run the cache is empty so the server builds the index (slow).
Run again afterwards for a true warm measurement.

Run from the project root:
    python3 benchmarks/live/10kx10k_hooks_usearch/warm/run_test.py
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

TEST_DIR = "benchmarks/live/10kx10k_hooks_usearch/warm"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8092
ITERATIONS = 200
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120

COUNTRY_CODES = ["GB", "DE", "FR", "NL", "US", "CH", "ES", "IT", "SE", "NO"]


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "server listening" in f.read():
                    return True
        time.sleep(2)
    return False


def post_json(base_url, path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            return r.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body
    except Exception as exc:
        return 0, {"error": str(exc)}


def make_new_b(counter):
    """Generate a new B-side record."""
    cc = COUNTRY_CODES[counter % len(COUNTRY_CODES)]
    return {
        "counterparty_id": f"CP-HOOK-{counter:05d}",
        "counterparty_name": f"Hook Test Corp {counter} {cc}",
        "domicile": cc,
        "lei_code": f"HOOK{counter:013d}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    cache_dir = f"{TEST_DIR}/cache"
    if not (os.path.exists(cache_dir) and os.listdir(cache_dir)):
        print("Note: cache is empty — this run will build the index (slow).")
        print("Run again afterwards for a true warm measurement.\n")

    # Clean up WAL and crossmap (keep cache for warm start)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_hooks_bench_{os.getpid()}.log"
    print(f"=== Warm hooks benchmark: {TEST_DIR} ===")
    print("Starting server with hooks enabled...")

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

    print(f"\nInjecting {args.iterations} B-side records...\n")

    t0 = time.perf_counter()
    results = {"auto": 0, "review": 0, "no_match": 0, "error": 0}

    for i in range(args.iterations):
        record = make_new_b(i)
        code, body = post_json(
            f"http://localhost:{PORT}",
            "/api/v1/b/add",
            {"record": record},
        )
        if code == 200:
            status = body.get("classification", body.get("status", "unknown"))
            if status in results:
                results[status] += 1
            else:
                results[status] = 1
        else:
            results["error"] += 1

        # Progress every 50
        if (i + 1) % 50 == 0:
            print(f"  injected {i + 1}/{args.iterations}...")

    wall_time = time.perf_counter() - t0

    print(f"\nInjection complete in {wall_time:.1f}s")
    print(f"  Throughput: {args.iterations / wall_time:.0f} records/sec")
    print(f"  Results: {results}")

    # Give hook script a moment to flush
    time.sleep(1)

    # Print hook output from the server log (hook stdout is inherited
    # and goes to the same log file as server stderr)
    print("\n--- Hook output (from server log) ---")
    with open(log_path) as f:
        for line in f:
            if "[hook]" in line or "Hook summary" in line:
                print(f"  {line}", end="")

    print("\nShutting down server...")
    server.terminate()
    server.wait()

    # The hook script prints its summary on EOF — check the log again
    time.sleep(0.5)
    with open(log_path) as f:
        for line in f:
            if (
                "Hook summary" in line
                or "on_confirm" in line
                or "on_review" in line
                or "on_nomatch" in line
            ):
                if "[hook]" in line or "---" in line:
                    print(f"  {line}", end="")

    print(f"\nTotal wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
