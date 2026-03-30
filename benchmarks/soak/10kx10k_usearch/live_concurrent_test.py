#!/usr/bin/env python3
"""Concurrent live-mode throughput and latency test (soak-patched).

Local copy of benchmarks/scripts/live_concurrent_test.py with two additions:

  --extra-ids-a FILE   Newline-delimited file of additional A-side IDs to
                       include in the update pool (growing pool for soak tests).
  --extra-ids-b FILE   Same for B-side IDs.

When provided, records for these IDs are synthesised on the fly (using the
same field schema as make_new_a / make_new_b) so they can be mutated by the
update operations, giving the soak test a growing upsert pool.

Operation mix (configurable via --encoding-pct):
  Encoding ops (split 4 ways): new_a, new_b, upd_a_emb, upd_b_emb
  Non-encoding ops (split 2 ways): upd_a_field, upd_b_field
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


def load_id_file(path: str) -> list[str]:
    """Load newline-delimited IDs from a file. Returns empty list if missing."""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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


def synth_a_record(id_str: str, a_id: str) -> dict:
    """Synthesise a minimal A record for an extra ID (for update operations)."""
    cc = COUNTRY_CODES[hash(id_str) % len(COUNTRY_CODES)]
    return {
        a_id: id_str,
        "legal_name": f"Soak Entity {id_str}",
        "short_name": f"Soak{id_str[-6:]}",
        "country_code": cc,
        "lei": f"SOAK{id_str[-12:]}",
    }


def synth_b_record(id_str: str, b_id: str) -> dict:
    """Synthesise a minimal B record for an extra ID (for update operations)."""
    cc = COUNTRY_CODES[hash(id_str) % len(COUNTRY_CODES)]
    return {
        b_id: id_str,
        "counterparty_name": f"Soak Counterparty {id_str}",
        "domicile": cc,
        "lei_code": f"SOAKB{id_str[-11:]}",
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
        description="Concurrent live-mode throughput and latency test (soak-patched)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="benchmarks/soak/10kx10k_usearch/config.yaml"
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Number of parallel HTTP workers",
    )
    parser.add_argument("--binary", default="./target/release/meld")
    parser.add_argument("--a-path", default="benchmarks/data/dataset_a_10k.csv")
    parser.add_argument("--b-path", default="benchmarks/data/dataset_b_10k.csv")
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
    # --- Soak-specific: extra ID files for growing update pool ---
    parser.add_argument(
        "--extra-ids-a",
        default="",
        help="Newline-delimited file of extra A-side IDs for the update pool",
    )
    parser.add_argument(
        "--extra-ids-b",
        default="",
        help="Newline-delimited file of extra B-side IDs for the update pool",
    )
    # --- Counter offset so new IDs don't collide across bursts ---
    parser.add_argument(
        "--a-counter-offset",
        type=int,
        default=0,
        help="Starting counter for new A record IDs (avoids cross-burst collisions)",
    )
    parser.add_argument(
        "--b-counter-offset",
        type=int,
        default=0,
        help="Starting counter for new B record IDs (avoids cross-burst collisions)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    base_url = f"http://localhost:{args.port}"

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
            print("TIMEOUT -- server did not become ready within 120s")
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

        # Merge extra IDs into the update pool (soak: growing pool).
        extra_a_ids = load_id_file(args.extra_ids_a)
        for eid in extra_a_ids:
            if eid not in a_by_id:
                a_by_id[eid] = synth_a_record(eid, args.a_id)

        extra_b_ids = load_id_file(args.extra_ids_b)
        for eid in extra_b_ids:
            if eid not in b_by_id:
                b_by_id[eid] = synth_b_record(eid, args.b_id)

        existing_a_ids = list(a_by_id.keys())
        existing_b_ids = list(b_by_id.keys())
        print(
            f"Update pool: {len(existing_a_ids):,} A IDs, "
            f"{len(existing_b_ids):,} B IDs "
            f"(+{len(extra_a_ids):,} extra A, +{len(extra_b_ids):,} extra B)"
        )

        # ------------------------------------------------------------------
        # Build operation sequence
        # ------------------------------------------------------------------
        n = args.iterations
        enc = args.encoding_pct / 100.0
        each_enc = enc / 4.0
        each_non = (1.0 - enc) / 2.0
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

        # Pre-build payloads.
        new_a_ctr = args.a_counter_offset
        new_b_ctr = args.b_counter_offset
        new_a_ids_created: list[str] = []
        new_b_ids_created: list[str] = []
        work_items: list[tuple[str, str, dict]] = []

        for op in ops:
            if op == "new_a":
                new_a_ctr += 1
                rec = make_new_a(new_a_ctr, args.a_id)
                new_a_ids_created.append(rec[args.a_id])
                work_items.append((op, "/api/v1/a/add", {"record": rec}))
            elif op == "new_b":
                new_b_ctr += 1
                rec = make_new_b(new_b_ctr, args.b_id)
                new_b_ids_created.append(rec[args.b_id])
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

        workers = []
        for _ in range(c):
            t = threading.Thread(
                target=worker,
                args=(base_url, work_queue, results, results_lock),
                daemon=True,
            )
            t.start()
            workers.append(t)

        t_wall_start = time.perf_counter()

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
            f"\nTotal: {n:,} requests in {t_wall_total:.2f}s  ->  {total_rps:.1f} req/s\n"
        )

        # ------------------------------------------------------------------
        # Write newly-created IDs to stdout markers so the soak controller
        # can parse them and grow the pool.
        # ------------------------------------------------------------------
        if new_a_ids_created:
            print(f"__NEW_A_IDS__:{','.join(new_a_ids_created)}")
        if new_b_ids_created:
            print(f"__NEW_B_IDS__:{','.join(new_b_ids_created)}")
        # Also emit counters so next burst can continue from the right offset.
        print(f"__A_COUNTER__:{new_a_ctr}")
        print(f"__B_COUNTER__:{new_b_ctr}")
        print(f"__THROUGHPUT__:{total_rps:.2f}")
        print(f"__WALL_SECS__:{t_wall_total:.2f}")
        print(f"__ERRORS__:{sum(errors.values())}")

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
