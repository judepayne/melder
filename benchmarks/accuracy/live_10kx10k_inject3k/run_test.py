#!/usr/bin/env python3
"""Live accuracy regression test — asymmetric field names, initial match + injection.

Validates that scoring works correctly with asymmetric field names
(field_a != field_b) in live mode, where both A and B sides can be
query or candidate.

Phases:
  0. Generate fixed datasets (if not present)
  1. Start server with initial match pass (skip_initial_match: false)
  2. Wait for server ready (initial match encodes + scores all 10k B records)
  3. Dump crossmap after initial match — validate against expected
  4. Inject 3,000 B records via API (sequential for determinism)
  5. Dump final crossmap — validate against expected
  6. Validate review queue is non-empty
  7. Report pass/fail

Expected output files are created on first run and stored in expected/.
Subsequent runs validate that results are identical.

Run from project root:
    python3 benchmarks/accuracy/live_10kx10k_inject3k/run_test.py
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

TEST_DIR = "benchmarks/accuracy/live_10kx10k_inject3k"
DATA_DIR = f"{TEST_DIR}/data"
EXPECTED_DIR = f"{TEST_DIR}/expected"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8092  # unique port to avoid conflicts
BASE_URL = f"http://localhost:{PORT}"
SERVER_READY_TIMEOUT = 300  # initial match on CI can be slow


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def post_json(path, payload):
    """POST JSON to the server, return (status, body)."""
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
    """GET JSON from the server, return (status, body)."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


def fetch_all_crossmap_pairs():
    """Paginate through all crossmap pairs."""
    pairs = []
    cursor = None
    while True:
        url = "/api/v1/crossmap/pairs?limit=1000"
        if cursor:
            url += f"&cursor={cursor}"
        status, body = get_json(url)
        if status != 200:
            print(f"  ERROR: crossmap pairs returned {status}: {body}")
            break
        page = body.get("pairs", [])
        pairs.extend((p["a_id"], p["b_id"]) for p in page)
        cursor = body.get("next_cursor")
        if not cursor:
            break
    return sorted(pairs)


def fetch_review_count():
    """Get review queue size."""
    status, body = get_json("/api/v1/review/list?limit=1")
    if status != 200:
        return -1
    return body.get("total", 0)


def tail_log(log_path, stop_event):
    """Tail the log file to stdout."""
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
    """Wait for 'server listening' in the log file."""
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


def load_pairs_csv(path):
    """Load a CSV of (a_id, b_id) pairs."""
    pairs = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.add((row["entity_id"], row["counterparty_id"]))
    return pairs


def save_pairs_csv(path, pairs):
    """Save a sorted list of (a_id, b_id) pairs to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id", "counterparty_id"])
        for a_id, b_id in sorted(pairs):
            writer.writerow([a_id, b_id])


def load_inject_records(path):
    """Load injection records from CSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_pairs(phase_name, actual_pairs, expected_path, ground_truth):
    """Compare actual pairs against expected. Create expected on first run."""
    actual_set = set(actual_pairs)

    if not os.path.exists(expected_path):
        # First run — save as expected
        save_pairs_csv(expected_path, actual_pairs)
        print(
            f"  [{phase_name}] Created expected file: {expected_path} ({len(actual_set)} pairs)"
        )

        # Still validate against ground truth
        true_positives = actual_set & ground_truth
        false_positives = actual_set - ground_truth
        recall = len(true_positives) / len(ground_truth) * 100 if ground_truth else 0
        print(
            f"  [{phase_name}] Recall: {len(true_positives)}/{len(ground_truth)} ({recall:.1f}%)"
        )
        print(f"  [{phase_name}] False positives: {len(false_positives)}")
        return True

    # Subsequent run — compare against expected
    expected_set = load_pairs_csv(expected_path)

    if actual_set == expected_set:
        print(f"  [{phase_name}] PASS — {len(actual_set)} pairs match expected")
        return True

    # Detailed diff
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    print(f"  [{phase_name}] FAIL — pairs differ from expected")
    print(f"    Expected: {len(expected_set)}, Actual: {len(actual_set)}")
    print(f"    Missing (in expected, not in actual): {len(missing)}")
    print(f"    Extra (in actual, not in expected): {len(extra)}")
    if missing:
        for p in sorted(missing)[:10]:
            print(f"      - {p[0]} <-> {p[1]}")
        if len(missing) > 10:
            print(f"      ... and {len(missing) - 10} more")
    if extra:
        for p in sorted(extra)[:10]:
            print(f"      + {p[0]} <-> {p[1]}")
        if len(extra) > 10:
            print(f"      ... and {len(extra) - 10} more")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate datasets even if they exist",
    )
    parser.add_argument(
        "--update-expected",
        action="store_true",
        help="Update expected output files from this run",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release")
        sys.exit(1)

    # Phase 0: Generate data if needed
    data_exists = os.path.exists(f"{DATA_DIR}/dataset_a_10k.csv")
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

    # Load ground truth
    ground_truth = load_pairs_csv(f"{DATA_DIR}/ground_truth.csv")
    print(f"Ground truth: {len(ground_truth)} known pairs")

    # Clean runtime artifacts
    shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/cache", exist_ok=True)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    if args.update_expected:
        shutil.rmtree(EXPECTED_DIR, ignore_errors=True)
    os.makedirs(EXPECTED_DIR, exist_ok=True)

    # Phase 1: Start server (with initial match pass)
    print("\n=== Phase 1: Starting server (initial match pass) ===")
    log_path = f"/tmp/meld_accuracy_live_{os.getpid()}.log"

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

    print("Server ready (initial match pass complete).\n")

    passed = True

    try:
        # Phase 2: Validate initial match crossmap
        print("=== Phase 2: Validate initial match ===")
        initial_pairs = fetch_all_crossmap_pairs()
        print(f"  Crossmap after initial match: {len(initial_pairs)} pairs")
        review_count = fetch_review_count()
        print(f"  Review queue: {review_count} entries")

        if not validate_pairs(
            "initial_match",
            initial_pairs,
            f"{EXPECTED_DIR}/initial_crossmap.csv",
            ground_truth,
        ):
            passed = False

        # Phase 3: Inject 3,000 B records
        print("\n=== Phase 3: Injecting 3,000 B records ===")
        inject_records = load_inject_records(f"{DATA_DIR}/inject_events.csv")
        errors = 0
        t0 = time.time()
        for i, rec in enumerate(inject_records):
            status, body = post_json("/api/v1/b/add", {"record": rec})
            if status not in (200, 201):
                errors += 1
                if errors <= 5:
                    print(f"  ERROR on record {i}: status={status}, body={body}")
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Injected {i + 1}/{len(inject_records)} ({rate:.0f} rec/s)")

        elapsed = time.time() - t0
        print(
            f"  Done: {len(inject_records)} records in {elapsed:.1f}s, {errors} errors"
        )

        if errors > 0:
            print(f"  WARNING: {errors} injection errors")

        # Phase 4: Validate final crossmap
        print("\n=== Phase 4: Validate final state ===")
        final_pairs = fetch_all_crossmap_pairs()
        print(f"  Crossmap after injection: {len(final_pairs)} pairs")
        final_review = fetch_review_count()
        print(f"  Review queue: {final_review} entries")

        if not validate_pairs(
            "final_crossmap",
            final_pairs,
            f"{EXPECTED_DIR}/final_crossmap.csv",
            ground_truth,
        ):
            passed = False

        # Phase 5: Validate review queue is non-empty
        print("\n=== Phase 5: Review queue check ===")
        if final_review > 0:
            print(f"  PASS — Review queue has {final_review} entries")
        else:
            print("  FAIL — Review queue is empty (expected some ambiguous matches)")
            passed = False

        # Phase 6: Validate initial match contributed matches
        print("\n=== Phase 6: Initial match contribution check ===")
        initial_set = set(map(tuple, initial_pairs))
        if len(initial_set) > 0:
            print(f"  PASS — Initial match pass produced {len(initial_set)} pairs")
        else:
            print("  FAIL — Initial match pass produced 0 pairs (broken)")
            passed = False

        # Check that injection added new matches
        final_set = set(map(tuple, final_pairs))
        new_matches = final_set - initial_set
        print(f"  Injection added {len(new_matches)} new matches")
        if len(new_matches) == 0:
            print("  FAIL — Injection produced 0 new matches (broken)")
            passed = False

    finally:
        print("\nStopping server...")
        server.terminate()
        server.wait(timeout=15)

    # Summary
    print("\n" + "=" * 60)
    if passed:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("RESULT: SOME CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
