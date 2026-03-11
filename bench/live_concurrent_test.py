#!/usr/bin/env python3
"""bench/live_concurrent_test.py — Concurrent live-mode throughput and latency test.

Fires --iterations requests across --concurrency parallel workers to exercise
opportunistic batching (the coordinator goroutine coalesces concurrent upserts
into a single BatchUpsertAndQuery call). Run --concurrency 1 to get the
sequential baseline; increase concurrency to measure batching gains.

Operation mix (same as live_stress_test.py):
  30%  POST /api/v1/a/add   new A record
  30%  POST /api/v1/b/add   new B record
  10%  POST /api/v1/a/add   update A record — embedding field changed
  10%  POST /api/v1/a/add   update A record — non-embedding field only
  10%  POST /api/v1/b/add   update B record — embedding field changed
  10%  POST /api/v1/b/add   update B record — non-embedding field only

Usage:
  python bench/live_concurrent_test.py [options]

  --config      Path to YAML config for `match serve`    (default: bench/bench_live.yaml)
  --port        TCP port to use for the server            (default: 8090)
  --iterations  Total number of API calls to make        (default: 3000)
  --concurrency Number of parallel workers               (default: 10)
  --binary      Path to the `match` binary               (default: ./match)
  --a-path      Path to dataset A CSV                    (default: testdata/dataset_a_10k.csv)
  --b-path      Path to dataset B CSV                    (default: testdata/dataset_b_10k.csv)
  --a-id        A ID field name                          (default: entity_id)
  --b-id        B ID field name                          (default: counterparty_id)
  --seed          Random seed for reproducibility          (default: 42)
  --no-serve      Skip starting the server (must already be running on --port)
  --encoding-pct  Percentage of requests that require ONNX encoding (default: 80).
                  Split evenly across 4 encoding op types (new_a, new_b,
                  upd_a_emb, upd_b_emb); remainder split across 2 non-encoding
                  types (upd_a_field, upd_b_field).
"""

import argparse
import csv
import json
import os
import queue
import random
import signal
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


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


def wait_for_health(base_url: str, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = urllib.request.urlopen(f"{base_url}/api/v1/health", timeout=2)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def post_json(base_url: str, path: str, payload: dict) -> tuple[float, int, dict]:
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


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Record factories
# ---------------------------------------------------------------------------

COUNTRY_CODES = ["GB", "DE", "FR", "NL", "US", "CH", "ES", "IT", "SE", "NO"]


def make_new_a(counter: int, a_id: str) -> dict:
    cc = COUNTRY_CODES[counter % len(COUNTRY_CODES)]
    return {
        a_id: f"ENT-CON-{counter:07d}",
        "legal_name": f"Concurrent Corp {counter} {cc}",
        "short_name": f"ConCorp{counter}",
        "country_code": cc,
        "lei": f"CONC{counter:013d}",
    }


def make_new_b(counter: int, b_id: str) -> dict:
    cc = COUNTRY_CODES[counter % len(COUNTRY_CODES)]
    return {
        b_id: f"CP-CON-{counter:07d}",
        "counterparty_name": f"Concurrent Party {counter} {cc}",
        "domicile": cc,
        "lei_code": f"CONCB{counter:012d}",
    }


def mutate_a_emb(record: dict) -> dict:
    r = dict(record)
    r["legal_name"] = r.get("legal_name", "") + " (upd)"
    return r


def mutate_a_field(record: dict) -> dict:
    r = dict(record)
    r["lei"] = r.get("lei", "") + "-R"
    return r


def mutate_b_emb(record: dict) -> dict:
    r = dict(record)
    r["counterparty_name"] = r.get("counterparty_name", "") + " (upd)"
    return r


def mutate_b_field(record: dict) -> dict:
    r = dict(record)
    r["lei_code"] = r.get("lei_code", "") + "-R"
    return r


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker(
    base_url: str,
    work_queue: "queue.Queue[tuple | None]",
    results: list,
    results_lock: threading.Lock,
) -> None:
    """Pull (op, record, path) tuples from work_queue and fire POST requests."""
    while True:
        item = work_queue.get()
        if item is None:
            work_queue.task_done()
            break
        op, path, payload = item
        elapsed, code, body = post_json(base_url, path, payload)
        with results_lock:
            results.append((op, elapsed, code, body))
        work_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent live-mode throughput and latency test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="bench/bench_live.yaml")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Number of parallel HTTP workers",
    )
    parser.add_argument("--binary", default="./match")
    parser.add_argument("--a-path", default="testdata/dataset_a_10k.csv")
    parser.add_argument("--b-path", default="testdata/dataset_b_10k.csv")
    parser.add_argument("--a-id", default="entity_id")
    parser.add_argument("--b-id", default="counterparty_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoding-pct",
        type=int,
        default=80,
        help="Percentage of requests requiring ONNX encoding (default: 80)",
    )
    parser.add_argument(
        "--no-serve", action="store_true", help="Skip starting the server"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    base_url = f"http://localhost:{args.port}"

    # ------------------------------------------------------------------
    # Build binary if needed
    # ------------------------------------------------------------------
    if not args.no_serve and not Path(args.binary).exists():
        print(f"Binary not found at {args.binary!r} — building...")
        env = {**os.environ, "GONOSUMDB": "*", "GOFLAGS": "-mod=mod"}
        subprocess.run(
            ["go", "build", "-o", args.binary, "./cmd/match"],
            check=True,
            env=env,
        )
        print("Build done.")

    # ------------------------------------------------------------------
    # Start server
    # ------------------------------------------------------------------
    proc = None
    if not args.no_serve:
        cmd = [args.binary, "serve", "--config", args.config, "--port", str(args.port)]
        print(f"Starting: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        # ------------------------------------------------------------------
        # Wait for health
        # ------------------------------------------------------------------
        print("Waiting for server to be ready...", end=" ", flush=True)
        if not wait_for_health(base_url):
            print("TIMEOUT — server did not become ready within 120s")
            sys.exit(1)
        print("ready.")

        # ------------------------------------------------------------------
        # Load datasets
        # ------------------------------------------------------------------
        a_records = load_csv(args.a_path)
        b_records = load_csv(args.b_path)
        print(f"Loaded {len(a_records):,} A records and {len(b_records):,} B records.")

        a_by_id = {r[args.a_id]: r for r in a_records}
        b_by_id = {r[args.b_id]: r for r in b_records}
        existing_a_ids = list(a_by_id.keys())
        existing_b_ids = list(b_by_id.keys())

        # ------------------------------------------------------------------
        # Build operation sequence
        # ------------------------------------------------------------------
        n = args.iterations
        enc = args.encoding_pct / 100.0
        each_enc = enc / 4.0  # split evenly across 4 encoding op types
        each_non = (1.0 - enc) / 2.0  # split evenly across 2 non-encoding types
        op_counts = {
            "new_a": int(n * each_enc),
            "new_b": int(n * each_enc),
            "upd_a_emb": int(n * each_enc),
            "upd_b_emb": int(n * each_enc),
            "upd_a_field": int(n * each_non),
            "upd_b_field": int(n * each_non),
        }
        ops = []
        for op, count in op_counts.items():
            ops.extend([op] * count)
        while len(ops) < n:
            ops.append(random.choice(list(op_counts.keys())))
        random.shuffle(ops)

        # Pre-build payloads (so payload construction doesn't skew timing).
        new_a_ctr = 0
        new_b_ctr = 0
        work_items: list[tuple[str, str, dict]] = []
        for op in ops:
            if op == "new_a":
                new_a_ctr += 1
                rec = make_new_a(new_a_ctr, args.a_id)
                work_items.append((op, "/api/v1/a/add", {"record": rec}))
            elif op == "new_b":
                new_b_ctr += 1
                rec = make_new_b(new_b_ctr, args.b_id)
                work_items.append((op, "/api/v1/b/add", {"record": rec}))
            elif op == "upd_a_emb":
                rec = mutate_a_emb(a_by_id[random.choice(existing_a_ids)])
                work_items.append((op, "/api/v1/a/add", {"record": rec}))
            elif op == "upd_a_field":
                rec = mutate_a_field(a_by_id[random.choice(existing_a_ids)])
                work_items.append((op, "/api/v1/a/add", {"record": rec}))
            elif op == "upd_b_emb":
                rec = mutate_b_emb(b_by_id[random.choice(existing_b_ids)])
                work_items.append((op, "/api/v1/b/add", {"record": rec}))
            else:
                rec = mutate_b_field(b_by_id[random.choice(existing_b_ids)])
                work_items.append((op, "/api/v1/b/add", {"record": rec}))

        # ------------------------------------------------------------------
        # Run with thread pool
        # ------------------------------------------------------------------
        c = args.concurrency
        print(
            f"\nRunning {n:,} iterations with concurrency={c}  "
            f"(new_a={op_counts['new_a']}, new_b={op_counts['new_b']}, "
            f"upd_a_emb={op_counts['upd_a_emb']}, upd_a_field={op_counts['upd_a_field']}, "
            f"upd_b_emb={op_counts['upd_b_emb']}, upd_b_field={op_counts['upd_b_field']})\n"
        )

        results: list[tuple[str, float, int, dict]] = []
        results_lock = threading.Lock()
        work_queue: queue.Queue = queue.Queue(maxsize=c * 4)

        # Start workers
        workers = []
        for _ in range(c):
            t = threading.Thread(
                target=worker,
                args=(base_url, work_queue, results, results_lock),
                daemon=True,
            )
            t.start()
            workers.append(t)

        # Feed work and track progress
        t_wall_start = time.perf_counter()
        progress_lock = threading.Lock()
        submitted = [0]

        def progress_printer():
            last_reported = 0
            while True:
                time.sleep(1.0)
                with results_lock:
                    done = len(results)
                if done >= n:
                    break
                elapsed = time.perf_counter() - t_wall_start
                rps = done / elapsed if elapsed > 0 else 0
                if done != last_reported:
                    print(f"  {done:>5}/{n}  ({rps:5.0f} req/s so far)")
                    last_reported = done

        progress_thread = threading.Thread(target=progress_printer, daemon=True)
        progress_thread.start()

        for item in work_items:
            work_queue.put(item)

        # Signal workers to stop
        for _ in range(c):
            work_queue.put(None)

        work_queue.join()
        t_wall_total = time.perf_counter() - t_wall_start
        total_rps = n / t_wall_total

        progress_thread.join(timeout=2)

        # ------------------------------------------------------------------
        # Aggregate results
        # ------------------------------------------------------------------
        latencies: dict[str, list[float]] = defaultdict(list)
        statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        old_mappings: dict[str, int] = defaultdict(int)
        errors: dict[str, int] = defaultdict(int)

        for op, elapsed, code, body in results:
            latencies[op].append(elapsed)
            status = body.get("status", f"http_{code}")
            statuses[op][status] += 1
            if body.get("old_mapping"):
                old_mappings[op] += 1
            if code not in (200, 201) or "error" in body:
                errors[op] += 1

        # ------------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------------
        W = 80
        print()
        print("=" * W)
        print(f"Concurrency: {c} workers")
        print("=" * W)
        print(
            f"{'Op':<14} {'N':>5}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'err':>4}  Statuses"
        )
        print("-" * W)

        all_latencies: list[float] = []
        for op in [
            "new_a",
            "new_b",
            "upd_a_emb",
            "upd_a_field",
            "upd_b_emb",
            "upd_b_field",
        ]:
            lats = latencies[op]
            if not lats:
                continue
            all_latencies.extend(lats)
            p50 = percentile(lats, 50)
            p95 = percentile(lats, 95)
            p99 = percentile(lats, 99)
            mx = max(lats)
            err = errors[op]
            st = "  ".join(f"{k}={v}" for k, v in sorted(statuses[op].items()))
            om = old_mappings[op]
            om_str = f"  old_mapping={om}" if om else ""
            print(
                f"{op:<14} {len(lats):>5}  {p50:>7.1f}  {p95:>7.1f}  {p99:>7.1f}  {mx:>7.1f}  {err:>4}  {st}{om_str}"
            )

        print("-" * W)
        if all_latencies:
            print(
                f"{'ALL':<14} {len(all_latencies):>5}  "
                f"{percentile(all_latencies, 50):>7.1f}  "
                f"{percentile(all_latencies, 95):>7.1f}  "
                f"{percentile(all_latencies, 99):>7.1f}  "
                f"{max(all_latencies):>7.1f}  "
                f"{sum(errors.values()):>4}  (ms)"
            )
        print("=" * W)
        print(
            f"\nTotal: {n:,} requests in {t_wall_total:.2f}s  →  {total_rps:.1f} req/s\n"
        )

    finally:
        if proc is not None:
            print("Stopping server (SIGTERM)...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
