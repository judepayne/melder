#!/usr/bin/env python3
"""Enroll accuracy regression test — single-pool entity resolution.

Validates that enroll mode correctly:
  1. Loads a pre-existing pool (5k records)
  2. Returns correct edges when enrolling new records
  3. Handles record removal
  4. Returns correct edges for post-removal enrollments

Phases:
  0. Generate fixed datasets (if not present)
  1. Start enroll server (pre-loads 5k pool)
  2. Verify pool loaded (count = 5000)
  3. Enroll 1,000 records, collect all edges
  4. Remove 50 records, verify they're gone
  5. Enroll 50 more records, collect edges
  6. Validate all collected edges against expected output
  7. Report pass/fail

Expected output files are created on first run and stored in expected/.
Subsequent runs validate that results are identical.

Run from project root:
    python3 benchmarks/accuracy/enroll_5k_inject1k/run_test.py
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

TEST_DIR = "benchmarks/accuracy/enroll_5k_inject1k"
DATA_DIR = f"{TEST_DIR}/data"
EXPECTED_DIR = f"{TEST_DIR}/expected"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8093
BASE_URL = f"http://localhost:{PORT}"
SERVER_READY_TIMEOUT = 180


def post_json(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


def get_json(path):
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


def tail_log(log_path, stop_event):
    pos = 0
    while not stop_event.is_set():
        try:
            with open(log_path) as f:
                f.seek(pos)
                new = f.read()
                if new:
                    for line in new.splitlines():
                        print(f"  [meld] {line}", flush=True)
                    pos += len(new)
        except FileNotFoundError:
            pass
        time.sleep(0.5)


def wait_for_server(log_path, timeout):
    stop_event = threading.Event()
    tailer = threading.Thread(target=tail_log, args=(log_path, stop_event), daemon=True)
    tailer.start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "server listening" in f.read():
                    stop_event.set()
                    tailer.join(timeout=2)
                    return True
        time.sleep(1)
    stop_event.set()
    tailer.join(timeout=2)
    return False


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_edges_csv(path, all_edges):
    """Save edges as CSV: enrolled_id, matched_id, score."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["enrolled_id", "matched_id", "score"])
        for enrolled_id, matched_id, score in sorted(all_edges):
            writer.writerow([enrolled_id, matched_id, f"{score:.4f}"])


def load_edges_csv(path):
    """Load edges from CSV as set of (enrolled_id, matched_id) tuples."""
    edges = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.add((row["enrolled_id"], row["matched_id"]))
    return edges


def validate_edges(phase_name, actual_edges, expected_path):
    """Compare actual edge pairs against expected. Create expected on first run."""
    # actual_edges is list of (enrolled_id, matched_id, score) tuples
    actual_pairs = set((e[0], e[1]) for e in actual_edges)

    if not os.path.exists(expected_path):
        save_edges_csv(expected_path, actual_edges)
        print(
            f"  [{phase_name}] Created expected file: {expected_path} ({len(actual_pairs)} edge pairs)"
        )
        return True

    expected_pairs = load_edges_csv(expected_path)

    if actual_pairs == expected_pairs:
        print(f"  [{phase_name}] PASS — {len(actual_pairs)} edge pairs match expected")
        return True

    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs
    print(f"  [{phase_name}] FAIL — edges differ from expected")
    print(f"    Expected: {len(expected_pairs)}, Actual: {len(actual_pairs)}")
    print(f"    Missing: {len(missing)}, Extra: {len(extra)}")
    if missing:
        for p in sorted(missing)[:10]:
            print(f"      - {p[0]} -> {p[1]}")
    if extra:
        for p in sorted(extra)[:10]:
            print(f"      + {p[0]} -> {p[1]}")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--update-expected", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        sys.exit(1)

    # Phase 0: Generate data
    data_exists = os.path.exists(f"{DATA_DIR}/pool_5k.csv")
    if not data_exists or args.regenerate:
        print("=== Phase 0: Generating data ===")
        result = subprocess.run(
            [sys.executable, f"{TEST_DIR}/generate_data.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Data generation failed:\n{result.stderr}")
            sys.exit(1)
        print(result.stdout)
    else:
        print("=== Phase 0: Using existing data ===")

    # Clean runtime artifacts
    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)

    if args.update_expected:
        shutil.rmtree(EXPECTED_DIR, ignore_errors=True)
    os.makedirs(EXPECTED_DIR, exist_ok=True)

    # Phase 1: Start server
    print("\n=== Phase 1: Starting enroll server ===")
    log_path = f"/tmp/meld_enroll_accuracy_{os.getpid()}.log"

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

    print("Server ready.\n")

    passed = True

    try:
        # Phase 2: Verify pool loaded
        print("=== Phase 2: Verify pool ===")
        status, body = get_json("/api/v1/enroll/count")
        pool_count = body.get("count", 0)
        print(f"  Pool count: {pool_count}")
        if pool_count != 5000:
            print(f"  FAIL — Expected 5000, got {pool_count}")
            passed = False
        else:
            print("  PASS — Pool loaded correctly")

        # Phase 3: Enroll 1,000 records
        print("\n=== Phase 3: Enrolling 1,000 records ===")
        enroll_records = load_csv(f"{DATA_DIR}/enroll_events.csv")
        enroll_edges = []
        errors = 0
        t0 = time.time()

        for i, rec in enumerate(enroll_records):
            status, body = post_json("/api/v1/enroll", {"record": rec})
            if status not in (200, 201):
                errors += 1
                if errors <= 5:
                    print(f"  ERROR on record {i}: status={status}, body={body}")
            else:
                rec_id = body.get("id", "")
                for edge in body.get("edges", []):
                    score = edge.get("score", 0.0)
                    matched_id = edge.get("id", "")
                    enroll_edges.append((rec_id, matched_id, score))

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(
                    f"  Enrolled {i + 1}/{len(enroll_records)} ({rate:.0f} rec/s, {len(enroll_edges)} edges so far)"
                )

        elapsed = time.time() - t0
        print(
            f"  Done: {len(enroll_records)} records in {elapsed:.1f}s, {errors} errors, {len(enroll_edges)} total edges"
        )

        if not validate_edges(
            "enroll_edges", enroll_edges, f"{EXPECTED_DIR}/enroll_edges.csv"
        ):
            passed = False

        # Phase 4: Remove 50 records
        print("\n=== Phase 4: Removing 50 records ===")
        remove_ids = [
            row["entity_id"] for row in load_csv(f"{DATA_DIR}/remove_ids.csv")
        ]
        remove_errors = 0

        for rid in remove_ids:
            status, body = post_json("/api/v1/enroll/remove", {"id": rid})
            if status not in (200, 201):
                remove_errors += 1
                if remove_errors <= 5:
                    print(f"  ERROR removing {rid}: status={status}, body={body}")

        # Verify removals
        removed_ok = 0
        for rid in remove_ids:
            status, body = get_json(f"/api/v1/enroll/query?id={rid}")
            if status == 404 or body.get("record") is None:
                removed_ok += 1

        print(
            f"  Removed: {len(remove_ids)}, Verified gone: {removed_ok}, Errors: {remove_errors}"
        )
        if removed_ok == len(remove_ids):
            print("  PASS — All removals verified")
        else:
            print(
                f"  FAIL — {len(remove_ids) - removed_ok} records still in pool after removal"
            )
            passed = False

        # Verify count decreased
        status, body = get_json("/api/v1/enroll/count")
        expected_count = 5000 + len(enroll_records) - len(remove_ids)
        actual_count = body.get("count", 0)
        print(f"  Pool count: {actual_count} (expected ~{expected_count})")

        # Phase 5: Enroll 50 more records after removals
        print("\n=== Phase 5: Post-removal enrollment (50 records) ===")
        post_remove_records = load_csv(f"{DATA_DIR}/post_remove_events.csv")
        post_remove_edges = []

        for rec in post_remove_records:
            status, body = post_json("/api/v1/enroll", {"record": rec})
            if status in (200, 201):
                rec_id = body.get("id", "")
                for edge in body.get("edges", []):
                    post_remove_edges.append(
                        (rec_id, edge.get("id", ""), edge.get("score", 0.0))
                    )

        print(
            f"  Enrolled: {len(post_remove_records)}, Edges: {len(post_remove_edges)}"
        )

        if not validate_edges(
            "post_remove_edges",
            post_remove_edges,
            f"{EXPECTED_DIR}/post_remove_edges.csv",
        ):
            passed = False

        # Phase 6: Verify removed records don't appear in new edges
        print("\n=== Phase 6: Verify no edges to removed records ===")
        removed_set = set(remove_ids)
        bad_edges = [(e, m) for e, m, _ in post_remove_edges if m in removed_set]
        if bad_edges:
            print(f"  FAIL — {len(bad_edges)} edges point to removed records:")
            for e, m in bad_edges[:5]:
                print(f"    {e} -> {m}")
            passed = False
        else:
            print("  PASS — No edges to removed records")

    finally:
        print("\nStopping server...")
        server.terminate()
        server.wait(timeout=15)

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
