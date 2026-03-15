#!/usr/bin/env python3
"""SQLite cache size sweep benchmark.

Measures warm live-mode throughput at different SQLite page cache sizes
and compares against the in-memory baseline.

Each test:
  1. Ensures the SQLite DB is populated (cold run if needed)
  2. Runs a warm benchmark with 2k injections at c=10
  3. Waits 3 minutes between runs for thermal cooling

Usage (from project root):
    python3 benchmarks/live/sqlite_cache_sweep.py
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import yaml

BINARY = "./target/release/meld"
PORT = 8090
ITERATIONS = 2000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120
COOL_DOWN_SECS = 180  # 3 minutes between runs

# Cache sizes to test (MB)
SQLITE_CACHE_SIZES = [2, 8, 32, 64, 256]

# Base directory for this benchmark
BENCH_DIR = "benchmarks/live/sqlite_cache_sweep"

# Template config (shared between all SQLite runs)
CONFIG_TEMPLATE = """\
job:
  name: sqlite_cache_sweep_{label}
  description: "SQLite cache sweep — {label}"

datasets:
  a:
    path: {a_data}
    id_field: entity_id
    format: csv
  b:
    path: {b_data}
    id_field: counterparty_id
    format: csv

cross_map:
  backend: local
  path: {work_dir}/crossmap.csv
  a_id_field: entity_id
  b_id_field: counterparty_id

embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: {cache_dir}
  b_cache_dir: {cache_dir}

blocking:
  enabled: true
  operator: and
  fields:
    - field_a: country_code
      field_b: domicile

match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.55
  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy
    scorer: partial_ratio
    weight: 0.20
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05

thresholds:
  auto_match: 0.85
  review_floor: 0.60

vector_backend: usearch
top_n: 5
ann_candidates: 5

performance:
  encoder_pool_size: 4
  vector_index_mode: "load"

live:
  upsert_log: {work_dir}/wal/events.ndjson
{live_extra}

output:
  results_path:   {work_dir}/output/results.csv
  review_path:    {work_dir}/output/review.csv
  unmatched_path: {work_dir}/output/unmatched.csv
"""


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "listening on port" in f.read():
                    return True
        time.sleep(2)
    return False


def parse_results(output: str) -> dict:
    """Extract throughput, p50, p95 from the concurrent test output."""
    result = {}

    # Look for the "ALL" summary line
    # ALL         2000      5.6     18.2     25.3     45.1     0  (ms)
    m = re.search(
        r"ALL\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+",
        output,
    )
    if m:
        result["p50"] = float(m.group(1))
        result["p95"] = float(m.group(2))
        result["p99"] = float(m.group(3))
        result["max"] = float(m.group(4))

    # Look for total throughput: "Total: 2,000 requests in 1.84s  →  1,087.0 req/s"
    m = re.search(r"([\d,.]+)\s+req/s", output)
    if m:
        result["rps"] = float(m.group(1).replace(",", ""))

    return result


def run_single_test(label, config_path, work_dir, is_cold=False):
    """Run a single benchmark test. Returns parsed results dict."""

    # Clean WAL, crossmap, output (but preserve DB and cache)
    shutil.rmtree(f"{work_dir}/output", ignore_errors=True)
    os.makedirs(f"{work_dir}/output", exist_ok=True)
    for f in glob.glob(f"{work_dir}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{work_dir}/wal", exist_ok=True)
    crossmap = f"{work_dir}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_bench_sweep_{os.getpid()}.log"

    kind = "cold" if is_cold else "warm"
    print(f"\n{'=' * 60}")
    print(f"  {label} ({kind})")
    print(f"{'=' * 60}")
    print("Starting server...")

    server = subprocess.Popen(
        [BINARY, "serve", "--config", config_path, "--port", str(PORT)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    startup_start = time.time()
    if not wait_for_server(log_path, SERVER_READY_TIMEOUT):
        print(f"ERROR: Server did not start within {SERVER_READY_TIMEOUT}s.")
        with open(log_path) as f:
            print(f.read())
        server.terminate()
        return None
    startup_elapsed = time.time() - startup_start

    print(f"Server ready. Startup: {startup_elapsed:.1f}s")

    if is_cold:
        print(f"(cold run — building DB, not measuring throughput)")

    print(f"Injecting {ITERATIONS} events at c={CONCURRENCY}...\n")
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
        ],
        capture_output=True,
        text=True,
    )

    server.terminate()
    server.wait()

    output = result.stdout + result.stderr
    print(output)

    parsed = parse_results(output)
    parsed["startup"] = startup_elapsed
    parsed["label"] = label
    return parsed


def ensure_dirs(work_dir):
    for sub in ["cache", "output", "wal"]:
        os.makedirs(f"{work_dir}/{sub}", exist_ok=True)


def main():
    if not os.path.exists(BINARY):
        print(f"Binary not found: {BINARY}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    os.makedirs(BENCH_DIR, exist_ok=True)

    results = []

    # ── 1. In-memory baseline (warm) ─────────────────────────────────

    mem_dir = f"{BENCH_DIR}/memory"
    ensure_dirs(mem_dir)

    # Check if embedding cache exists (need a cold run first)
    cache_files = glob.glob(f"{mem_dir}/cache/*.idx")
    if not cache_files:
        print("\n** In-memory: no cache found, running cold pass first **")
        config_path = f"{mem_dir}/config.yaml"
        config = CONFIG_TEMPLATE.format(
            label="memory",
            a_data=A_DATA,
            b_data=B_DATA,
            work_dir=mem_dir,
            cache_dir=f"{mem_dir}/cache",
            live_extra="",
        )
        with open(config_path, "w") as f:
            f.write(config)
        run_single_test("In-Memory", config_path, mem_dir, is_cold=True)
        print(f"\nCooling down {COOL_DOWN_SECS}s...")
        time.sleep(COOL_DOWN_SECS)

    print("\n** In-memory baseline (warm) **")
    config_path = f"{mem_dir}/config.yaml"
    config = CONFIG_TEMPLATE.format(
        label="memory",
        a_data=A_DATA,
        b_data=B_DATA,
        work_dir=mem_dir,
        cache_dir=f"{mem_dir}/cache",
        live_extra="",
    )
    with open(config_path, "w") as f:
        f.write(config)
    r = run_single_test("In-Memory", config_path, mem_dir)
    if r:
        results.append(r)

    # ── 2. SQLite cache size sweep ────────────────────────────────────

    # All SQLite runs share the same embedding cache (built once)
    sqlite_cache_dir = f"{BENCH_DIR}/sqlite_shared_cache"
    os.makedirs(sqlite_cache_dir, exist_ok=True)

    # Check if we need a cold run to build the embedding cache
    sqlite_cache_files = glob.glob(f"{sqlite_cache_dir}/*.idx")
    need_cold = len(sqlite_cache_files) == 0

    if need_cold:
        # Do one cold run with default 64MB to build the DB + cache
        cold_dir = f"{BENCH_DIR}/sqlite_cold_setup"
        ensure_dirs(cold_dir)
        config_path = f"{cold_dir}/config.yaml"
        config = CONFIG_TEMPLATE.format(
            label="sqlite_cold_setup",
            a_data=A_DATA,
            b_data=B_DATA,
            work_dir=cold_dir,
            cache_dir=sqlite_cache_dir,
            live_extra=f"  db_path: {cold_dir}/live.db",
        )
        with open(config_path, "w") as f:
            f.write(config)

        print("\n** SQLite: building embedding cache (cold run) **")
        run_single_test("SQLite cold setup", config_path, cold_dir, is_cold=True)
        print(f"\nCooling down {COOL_DOWN_SECS}s...")
        time.sleep(COOL_DOWN_SECS)

    for cache_mb in SQLITE_CACHE_SIZES:
        print(f"\nCooling down {COOL_DOWN_SECS}s before SQLite {cache_mb}MB test...")
        time.sleep(COOL_DOWN_SECS)

        work_dir = f"{BENCH_DIR}/sqlite_{cache_mb}mb"
        ensure_dirs(work_dir)

        # Remove old DB so each run starts from a cold DB build
        # (we want to measure warm throughput, so we do a cold pass then a warm pass)
        db_path = f"{work_dir}/live.db"

        if not os.path.exists(db_path):
            # Need a cold run to populate the DB
            config_path = f"{work_dir}/config.yaml"
            config = CONFIG_TEMPLATE.format(
                label=f"sqlite_{cache_mb}mb",
                a_data=A_DATA,
                b_data=B_DATA,
                work_dir=work_dir,
                cache_dir=sqlite_cache_dir,
                live_extra=f"  db_path: {db_path}\n  sqlite_cache_mb: {cache_mb}",
            )
            with open(config_path, "w") as f:
                f.write(config)

            print(f"\n** SQLite {cache_mb}MB: cold run to populate DB **")
            run_single_test(f"SQLite {cache_mb}MB", config_path, work_dir, is_cold=True)
            print(f"\nCooling down {COOL_DOWN_SECS}s...")
            time.sleep(COOL_DOWN_SECS)

        # Warm run
        config_path = f"{work_dir}/config.yaml"
        config = CONFIG_TEMPLATE.format(
            label=f"sqlite_{cache_mb}mb",
            a_data=A_DATA,
            b_data=B_DATA,
            work_dir=work_dir,
            cache_dir=sqlite_cache_dir,
            live_extra=f"  db_path: {db_path}\n  sqlite_cache_mb: {cache_mb}",
        )
        with open(config_path, "w") as f:
            f.write(config)

        print(f"\n** SQLite {cache_mb}MB: warm run **")
        r = run_single_test(f"SQLite {cache_mb}MB", config_path, work_dir)
        if r:
            results.append(r)

    # ── 3. Summary ────────────────────────────────────────────────────

    print("\n")
    print("=" * 72)
    print("  SQLite Cache Size Sweep — Summary")
    print("=" * 72)
    print(f"  Dataset: 10k x 10k, {ITERATIONS} injections, c={CONCURRENCY}")
    print(f"  Cooling: {COOL_DOWN_SECS}s between runs")
    print("-" * 72)
    print(
        f"  {'Config':<20} {'Startup':>8} {'req/s':>8} {'p50':>8} {'p95':>8} {'p99':>8}"
    )
    print("-" * 72)

    baseline_rps = None
    for r in results:
        rps = r.get("rps", 0)
        if baseline_rps is None:
            baseline_rps = rps
        pct = f"({rps / baseline_rps * 100:.0f}%)" if baseline_rps else ""
        print(
            f"  {r['label']:<20} {r.get('startup', 0):>7.1f}s {rps:>7.0f} {pct:>5}  "
            f"{r.get('p50', 0):>7.1f} {r.get('p95', 0):>7.1f} {r.get('p99', 0):>7.1f}"
        )

    print("=" * 72)

    if baseline_rps and len(results) > 1:
        best_sqlite = max(results[1:], key=lambda r: r.get("rps", 0))
        gap = (1 - best_sqlite["rps"] / baseline_rps) * 100
        print(
            f"\n  Best SQLite: {best_sqlite['label']} at {best_sqlite.get('rps', 0):.0f} req/s"
        )
        print(f"  In-Memory baseline: {baseline_rps:.0f} req/s")
        print(f"  Gap: {gap:.1f}% slower than in-memory")


if __name__ == "__main__":
    main()
