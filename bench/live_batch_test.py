#!/usr/bin/env python3
"""bench/live_batch_test.py — Batch endpoint throughput and latency benchmark.

Compares single-record endpoints vs batch endpoints by running the same
workload through both and printing side-by-side results.

Operation mix (total --records spread across operations):
  30%  A-side add (new records)
  30%  B-side add (new records)
  20%  A-side add (updates to existing records — embedding field changed)
  20%  B-side add (updates to existing records — embedding field changed)

The workload is executed twice:
  1. Single-record mode: one POST per record via /api/v1/{a,b}/add
  2. Batch mode: records grouped into --batch-size chunks via /api/v1/{a,b}/add-batch

At the end, a short match-batch and remove-batch test is also run.

Usage:
  python bench/live_batch_test.py [options]

  --batch-size    Records per batch request         (default: 50)
  --records       Total records to process           (default: 3000)
  --config        Path to YAML config                (default: testdata/configs/bench_live.yaml)
  --port          TCP port for the server             (default: 8090)
  --binary        Path to the meld binary             (default: ./target/release/meld)
  --a-path        Path to dataset A csv               (default: testdata/dataset_a_10k.csv)
  --b-path        Path to dataset B csv               (default: testdata/dataset_b_10k.csv)
  --a-id          A ID field name                     (default: entity_id)
  --b-id          B ID field name                     (default: counterparty_id)
  --seed          Random seed for reproducibility     (default: 42)
  --no-serve      Skip starting the server            (assume already running)
  --batch-only    Skip single-record baseline         (only run batch mode)
"""

import argparse
import csv
import json
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of data (0-100)."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def wait_for_health(base_url: str, timeout: float = 120.0) -> bool:
    """Poll /api/v1/health until it returns 200 or timeout expires."""
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
    """POST JSON; return (elapsed_ms, http_status, body_dict)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
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


def make_new_a(counter: int, a_id: str, prefix: str = "BATCH") -> dict:
    cc = COUNTRY_CODES[counter % len(COUNTRY_CODES)]
    return {
        a_id: f"ENT-{prefix}-{counter:07d}",
        "legal_name": f"{prefix} Corp {counter} {cc}",
        "short_name": f"{prefix}Corp{counter}",
        "country_code": cc,
        "lei": f"{prefix}{counter:012d}",
    }


def make_new_b(counter: int, b_id: str, prefix: str = "BATCH") -> dict:
    cc = COUNTRY_CODES[counter % len(COUNTRY_CODES)]
    return {
        b_id: f"CP-{prefix}-{counter:07d}",
        "counterparty_name": f"{prefix} Party {counter} {cc}",
        "domicile": cc,
        "lei_code": f"{prefix}B{counter:011d}",
    }


def mutate_a(record: dict) -> dict:
    r = dict(record)
    r["legal_name"] = r.get("legal_name", "") + " (rev)"
    return r


def mutate_b(record: dict) -> dict:
    r = dict(record)
    r["counterparty_name"] = r.get("counterparty_name", "") + " (rev)"
    return r


# ---------------------------------------------------------------------------
# Build work items
# ---------------------------------------------------------------------------


def build_work_items(
    n: int,
    a_records: list[dict],
    b_records: list[dict],
    a_id: str,
    b_id: str,
    prefix: str = "BATCH",
    seed: int = 42,
) -> list[tuple[str, str, dict]]:
    """Build a list of (op_name, api_path, single_record) tuples."""
    rng = random.Random(seed)
    a_by_id = {r[a_id]: r for r in a_records}
    b_by_id = {r[b_id]: r for r in b_records}
    existing_a_ids = list(a_by_id.keys())
    existing_b_ids = list(b_by_id.keys())

    op_counts = {
        "new_a": int(n * 0.30),
        "new_b": int(n * 0.30),
        "upd_a": int(n * 0.20),
        "upd_b": int(n * 0.20),
    }
    ops = []
    for op, count in op_counts.items():
        ops.extend([op] * count)
    while len(ops) < n:
        ops.append(rng.choice(list(op_counts.keys())))
    rng.shuffle(ops)

    new_a_ctr = 0
    new_b_ctr = 0
    items: list[tuple[str, str, dict]] = []
    for op in ops:
        if op == "new_a":
            new_a_ctr += 1
            rec = make_new_a(new_a_ctr, a_id, prefix)
            items.append((op, "/api/v1/a/add", rec))
        elif op == "new_b":
            new_b_ctr += 1
            rec = make_new_b(new_b_ctr, b_id, prefix)
            items.append((op, "/api/v1/b/add", rec))
        elif op == "upd_a":
            aid = rng.choice(existing_a_ids)
            rec = mutate_a(a_by_id[aid])
            items.append((op, "/api/v1/a/add", rec))
        else:  # upd_b
            bid = rng.choice(existing_b_ids)
            rec = mutate_b(b_by_id[bid])
            items.append((op, "/api/v1/b/add", rec))
    return items


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_single(
    base_url: str, items: list[tuple[str, str, dict]]
) -> tuple[float, list[float], int]:
    """Run items one-by-one via single-record endpoints.

    Returns (wall_seconds, per_record_latencies_ms, error_count).
    """
    latencies: list[float] = []
    errors = 0
    n = len(items)

    t0 = time.perf_counter()
    for i, (op, path, rec) in enumerate(items):
        elapsed, code, body = post_json(base_url, path, {"record": rec})
        latencies.append(elapsed)
        if code not in (200, 201):
            errors += 1
        if (i + 1) % 500 == 0:
            wall = time.perf_counter() - t0
            rps = (i + 1) / wall if wall > 0 else 0
            print(f"    {i + 1:>5}/{n}  ({rps:5.0f} rec/s)")

    wall = time.perf_counter() - t0
    return wall, latencies, errors


def run_batch(
    base_url: str,
    items: list[tuple[str, str, dict]],
    batch_size: int,
) -> tuple[float, list[float], list[float], int, int]:
    """Run items via batch endpoints.

    Groups items by side (a/b) into batches of batch_size.
    Returns (wall_seconds, per_batch_latencies_ms, per_record_latencies_ms,
             error_count, batch_count).
    """
    # Separate items by side, preserving order within each side
    a_items = [(op, rec) for op, path, rec in items if "/a/" in path]
    b_items = [(op, rec) for op, path, rec in items if "/b/" in path]

    # Build batches: alternate A and B batches to interleave work
    batches: list[tuple[str, list[dict]]] = []  # (path, records)
    a_idx = 0
    b_idx = 0
    while a_idx < len(a_items) or b_idx < len(b_items):
        if a_idx < len(a_items):
            chunk = [rec for _, rec in a_items[a_idx : a_idx + batch_size]]
            batches.append(("/api/v1/a/add-batch", chunk))
            a_idx += batch_size
        if b_idx < len(b_items):
            chunk = [rec for _, rec in b_items[b_idx : b_idx + batch_size]]
            batches.append(("/api/v1/b/add-batch", chunk))
            b_idx += batch_size

    batch_latencies: list[float] = []
    record_latencies: list[float] = []
    errors = 0
    total_records = 0

    t0 = time.perf_counter()
    for i, (path, records) in enumerate(batches):
        elapsed, code, body = post_json(base_url, path, {"records": records})
        batch_latencies.append(elapsed)
        n_rec = len(records)
        total_records += n_rec
        per_rec = elapsed / n_rec if n_rec > 0 else elapsed
        record_latencies.extend([per_rec] * n_rec)

        if code not in (200, 201):
            errors += 1
        # Count per-record errors within the batch response
        if isinstance(body, dict) and "results" in body:
            for r in body["results"]:
                if r.get("status") == "error":
                    errors += 1

        if (i + 1) % 20 == 0 or i == len(batches) - 1:
            wall = time.perf_counter() - t0
            rps = total_records / wall if wall > 0 else 0
            print(
                f"    batch {i + 1:>4}/{len(batches)}  ({rps:5.0f} rec/s,  {total_records} records)"
            )

    wall = time.perf_counter() - t0
    return wall, batch_latencies, record_latencies, errors, len(batches)


def run_match_batch_test(
    base_url: str,
    a_records: list[dict],
    a_id: str,
    batch_size: int,
    count: int = 100,
) -> None:
    """Quick match-batch benchmark: send count A records through match-batch."""
    recs = a_records[:count]
    batches = [recs[i : i + batch_size] for i in range(0, len(recs), batch_size)]

    latencies: list[float] = []
    t0 = time.perf_counter()
    for batch in batches:
        elapsed, code, body = post_json(
            base_url, "/api/v1/a/match-batch", {"records": batch}
        )
        latencies.append(elapsed)
    wall = time.perf_counter() - t0

    total_rec = sum(len(b) for b in batches)
    rps = total_rec / wall if wall > 0 else 0

    print(
        f"\n  match-batch: {len(batches)} batches, {total_rec} records in {wall:.2f}s"
    )
    print(f"    throughput: {rps:.0f} rec/s")
    print(
        f"    per-batch latency: p50={percentile(latencies, 50):.1f}ms  "
        f"p95={percentile(latencies, 95):.1f}ms  "
        f"p99={percentile(latencies, 99):.1f}ms"
    )
    per_rec = [l / batch_size for l in latencies]
    print(
        f"    per-record latency: p50={percentile(per_rec, 50):.2f}ms  "
        f"p95={percentile(per_rec, 95):.2f}ms"
    )


def run_remove_batch_test(
    base_url: str,
    ids: list[str],
    side: str,
    batch_size: int,
) -> None:
    """Quick remove-batch benchmark."""
    path = f"/api/v1/{side}/remove-batch"
    batches = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    latencies: list[float] = []
    removed = 0
    not_found = 0
    t0 = time.perf_counter()
    for batch in batches:
        elapsed, code, body = post_json(base_url, path, {"ids": batch})
        latencies.append(elapsed)
        if isinstance(body, dict) and "results" in body:
            for r in body["results"]:
                if r.get("status") == "removed":
                    removed += 1
                elif r.get("status") == "not_found":
                    not_found += 1
    wall = time.perf_counter() - t0

    total = sum(len(b) for b in batches)
    rps = total / wall if wall > 0 else 0

    print(
        f"\n  remove-batch ({side}): {len(batches)} batches, {total} IDs in {wall:.2f}s"
    )
    print(
        f"    throughput: {rps:.0f} rec/s  (removed={removed}, not_found={not_found})"
    )
    if latencies:
        print(
            f"    per-batch latency: p50={percentile(latencies, 50):.1f}ms  "
            f"p95={percentile(latencies, 95):.1f}ms"
        )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    label: str,
    wall: float,
    n_records: int,
    latencies: list[float],
    errors: int,
    batch_count: int | None = None,
    batch_latencies: list[float] | None = None,
) -> None:
    rps = n_records / wall if wall > 0 else 0
    print(f"\n  {label}")
    print(f"    records:     {n_records:,}")
    if batch_count is not None:
        print(f"    batches:     {batch_count}")
    print(f"    wall time:   {wall:.2f}s")
    print(f"    throughput:  {rps:.0f} rec/s")
    print(f"    errors:      {errors}")
    print(
        f"    per-record latency (ms):  "
        f"p50={percentile(latencies, 50):.2f}  "
        f"p95={percentile(latencies, 95):.2f}  "
        f"p99={percentile(latencies, 99):.2f}  "
        f"max={max(latencies):.2f}"
    )
    if batch_latencies:
        print(
            f"    per-batch latency  (ms):  "
            f"p50={percentile(batch_latencies, 50):.1f}  "
            f"p95={percentile(batch_latencies, 95):.1f}  "
            f"p99={percentile(batch_latencies, 99):.1f}  "
            f"max={max(batch_latencies):.1f}"
        )


def print_comparison(
    single_wall: float,
    single_n: int,
    batch_wall: float,
    batch_n: int,
    batch_size: int,
) -> None:
    s_rps = single_n / single_wall if single_wall > 0 else 0
    b_rps = batch_n / batch_wall if batch_wall > 0 else 0
    speedup = b_rps / s_rps if s_rps > 0 else 0

    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'Mode':<20} {'Records':>8} {'Wall (s)':>10} {'rec/s':>10}")
    print(f"  {'-' * 58}")
    print(f"  {'Single-record':<20} {single_n:>8,} {single_wall:>10.2f} {s_rps:>10.0f}")
    print(
        f"  {'Batch (size=' + str(batch_size) + ')':<20} {batch_n:>8,} {batch_wall:>10.2f} {b_rps:>10.0f}"
    )
    print(f"  {'-' * 58}")
    print(f"  Speedup: {speedup:.2f}x")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch endpoint throughput and latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Records per batch request"
    )
    parser.add_argument(
        "--records", type=int, default=3000, help="Total records to process"
    )
    parser.add_argument("--config", default="testdata/configs/bench_live.yaml")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--binary", default="./target/release/meld")
    parser.add_argument("--a-path", default="testdata/dataset_a_10k.csv")
    parser.add_argument("--b-path", default="testdata/dataset_b_10k.csv")
    parser.add_argument("--a-id", default="entity_id")
    parser.add_argument("--b-id", default="counterparty_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument(
        "--batch-only", action="store_true", help="Skip single-record baseline"
    )
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    a_records = load_csv(args.a_path)
    b_records = load_csv(args.b_path)
    print(f"Loaded {len(a_records):,} A records and {len(b_records):,} B records.")

    # Build work items (same seed for both runs so they produce the same records)
    items = build_work_items(
        args.records,
        a_records,
        b_records,
        args.a_id,
        args.b_id,
        prefix="SINGLE",
        seed=args.seed,
    )
    batch_items = build_work_items(
        args.records,
        a_records,
        b_records,
        args.a_id,
        args.b_id,
        prefix="BTCH",
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Helper to start/stop server
    # ------------------------------------------------------------------
    def start_server() -> subprocess.Popen | None:
        if args.no_serve:
            return None
        cmd = [args.binary, "serve", "--config", args.config, "--port", str(args.port)]
        print(f"\nStarting: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("Waiting for server to be ready...", end=" ", flush=True)
        if not wait_for_health(base_url, timeout=120.0):
            print("TIMEOUT")
            proc.terminate()
            proc.wait()
            sys.exit(1)
        print("ready.")
        return proc

    def stop_server(proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        print("Stopping server...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # ==================================================================
    # Phase 1: Single-record baseline
    # ==================================================================
    single_wall = 0.0
    single_latencies: list[float] = []
    single_errors = 0

    if not args.batch_only:
        print("\n" + "=" * 70)
        print(f"  PHASE 1: Single-record mode ({args.records:,} records)")
        print("=" * 70)
        proc = start_server()
        try:
            single_wall, single_latencies, single_errors = run_single(base_url, items)
            print_summary(
                "Single-record results",
                single_wall,
                args.records,
                single_latencies,
                single_errors,
            )
        finally:
            stop_server(proc)

    # ==================================================================
    # Phase 2: Batch mode
    # ==================================================================
    print("\n" + "=" * 70)
    print(
        f"  PHASE 2: Batch mode (batch_size={args.batch_size}, {args.records:,} records)"
    )
    print("=" * 70)
    proc = start_server()
    try:
        batch_wall, batch_lats, batch_rec_lats, batch_errors, batch_count = run_batch(
            base_url,
            batch_items,
            args.batch_size,
        )
        print_summary(
            f"Batch results (size={args.batch_size})",
            batch_wall,
            args.records,
            batch_rec_lats,
            batch_errors,
            batch_count=batch_count,
            batch_latencies=batch_lats,
        )

        # ----------------------------------------------------------
        # Phase 3: match-batch + remove-batch quick test
        # ----------------------------------------------------------
        print("\n" + "-" * 70)
        print("  Additional: match-batch and remove-batch tests")
        print("-" * 70)

        run_match_batch_test(
            base_url,
            a_records,
            args.a_id,
            args.batch_size,
            count=200,
        )

        # Remove the batch-added A records
        a_remove_ids = [
            f"ENT-BTCH-{i:07d}" for i in range(1, min(201, args.records // 3 + 1))
        ]
        run_remove_batch_test(base_url, a_remove_ids, "a", args.batch_size)

    finally:
        stop_server(proc)

    # ==================================================================
    # Comparison
    # ==================================================================
    if not args.batch_only and single_latencies:
        print_comparison(
            single_wall,
            args.records,
            batch_wall,
            args.records,
            args.batch_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
