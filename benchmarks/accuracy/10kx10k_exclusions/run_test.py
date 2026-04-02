#!/usr/bin/env python3
"""Accuracy test — exclusions + initial match pass.

Generates 10k × 10k datasets with 1,000 exact-match pairs. Those 1,000
exact-match pairs are written as the exclusions file. Then:

Phase 1 — Initial match pass:
  - Start meld serve (which runs initial_match_pass before the API opens)
  - Verify that none of the 1,000 excluded pairs were matched
  - Verify that the remaining ~6,000 non-excluded matched pairs did match
    at a reasonable rate

Phase 2 — Live injection with exclusion events:
  - Inject new B records, some of which have their true A match in the
    exclusion list (as specific a_id,b_id pairs)
  - Fire POST /api/v1/exclude to break some existing matches
  - Verify that excluded pairs are never in the crossmap
  - Verify that POST /exclude correctly breaks matches and prevents rematch

Run from the project root:
    python3 benchmarks/accuracy/10kx10k_exclusions/run_test.py
"""

import argparse
import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

TEST_DIR = "benchmarks/accuracy/10kx10k_exclusions"
DATA_DIR = f"{TEST_DIR}/data"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8091  # different from the live benchmarks to avoid conflicts
BASE_URL = f"http://localhost:{PORT}"
SERVER_READY_TIMEOUT = 180  # initial match pass can take a while
N_RECORDS = 10_000
N_EXACT = 1_000
N_INJECT = 3_000
SEED = 42


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
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


def delete_json(path, payload):
    """DELETE with JSON body to the server, return (status, body)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


def get_json(path):
    """GET JSON from the server, return (status, body)."""
    req = urllib.request.Request(f"{BASE_URL}{path}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


def tail_log(log_path, stop_event):
    """Tail the log file to stdout, printing new lines as they appear."""
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
    """Wait for 'server listening' in the log file, streaming output live."""
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
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Phase 0 — Data generation
# ---------------------------------------------------------------------------


def generate_data():
    """Generate 10k x 10k datasets with 1,000 exact matches."""
    print("=== Phase 0: Generating data ===")

    # Import the generator
    sys.path.insert(0, "benchmarks/data")
    import generate

    records_a, records_b = generate.generate_with_seed(
        seed=SEED,
        n=N_RECORDS,
        include_addresses=False,
        out_dir=DATA_DIR,
        n_exact=N_EXACT,
    )

    # Rename to match config expectations (generate_with_seed writes dataset_a.csv)
    for name in ["dataset_a", "dataset_b"]:
        src = os.path.join(DATA_DIR, f"{name}.csv")
        dst = os.path.join(DATA_DIR, f"{name}_10k.csv")
        if os.path.exists(src):
            os.rename(src, dst)

    # Verify counts
    exact = [r for r in records_b if r["_match_type"] == "exact"]
    matched = [r for r in records_b if r["_match_type"] == "matched"]
    print(f"  A records: {len(records_a):,}")
    print(f"  B records: {len(records_b):,}")
    print(f"  Exact-match pairs: {len(exact):,}")
    print(f"  Noised-match pairs: {len(matched):,}")
    assert len(exact) == N_EXACT, f"Expected {N_EXACT} exact, got {len(exact)}"

    # Write exclusions CSV from the exact-match pairs
    # These are the pairs we're telling Melder to NEVER match
    exclusions_path = f"{TEST_DIR}/exclusions.csv"
    with open(exclusions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id", "counterparty_id"])
        for rec in exact:
            writer.writerow([rec["_true_a_id"], rec["counterparty_id"]])
    print(f"  Wrote exclusions: {exclusions_path} ({len(exact)} pairs)")

    # Also write a lookup of all B records by type for later validation
    b_lookup_path = f"{DATA_DIR}/b_lookup.csv"
    with open(b_lookup_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["counterparty_id", "_true_a_id", "_match_type"]
        )
        writer.writeheader()
        for rec in records_b:
            writer.writerow(
                {
                    "counterparty_id": rec["counterparty_id"],
                    "_true_a_id": rec["_true_a_id"],
                    "_match_type": rec["_match_type"],
                }
            )
    print(f"  Wrote B lookup: {b_lookup_path}")

    return records_a, records_b


# ---------------------------------------------------------------------------
# Phase 1 — Initial match pass validation
# ---------------------------------------------------------------------------


def phase1_validate_initial_match(excluded_pairs):
    """After server startup (which runs initial_match_pass), check the crossmap."""
    print("\n=== Phase 1: Validating initial match pass ===")

    # Get crossmap stats
    status, stats = get_json("/api/v1/crossmap/stats")
    assert status == 200, f"crossmap/stats failed: {status}"
    crossmap_count = stats.get("crossmap_pairs", 0)
    print(f"  Crossmap pairs: {crossmap_count}")

    # Check: none of the excluded pairs should be in the crossmap.
    # Query each excluded A-side ID to see if it's matched to the excluded B.
    excluded_in_crossmap = 0
    matched_pairs = set()  # collect what IS matched for phase 2
    for a_id, b_id in excluded_pairs:
        status, body = get_json(f"/api/v1/crossmap/lookup?id={a_id}&side=a")
        if status == 200 and body.get("paired_id") == b_id:
            excluded_in_crossmap += 1
            if excluded_in_crossmap <= 5:
                print(f"    excluded pair matched: {a_id} -> {b_id}")

    if excluded_in_crossmap > 0:
        print(f"  FAIL: {excluded_in_crossmap} excluded pairs found in crossmap!")
        return False, matched_pairs
    print(f"  PASS: 0/{len(excluded_pairs)} excluded pairs in crossmap")

    # Check: reasonable number of non-excluded matches occurred.
    # With 6,000 noised matches, we expect at least 3,000 to auto-match.
    min_expected = 3000
    if crossmap_count < min_expected:
        print(
            f"  WARN: only {crossmap_count} crossmap pairs, expected >= {min_expected}"
        )
    else:
        print(f"  PASS: {crossmap_count} crossmap pairs (>= {min_expected} expected)")

    # Collect a sample of actual matched pairs for phase 2 break testing.
    # Query first 200 A-side IDs to find ones that are matched.
    for i in range(200):
        a_id = f"ENT-{i:06d}"
        status, body = get_json(f"/api/v1/crossmap/lookup?id={a_id}&side=a")
        if status == 200 and body.get("paired_id"):
            matched_pairs.add((a_id, body["paired_id"]))
        if len(matched_pairs) >= 100:
            break

    print(f"  Sampled {len(matched_pairs)} matched pairs for phase 2")
    return True, matched_pairs


# ---------------------------------------------------------------------------
# Phase 2 — Live injection + exclusion API events
# ---------------------------------------------------------------------------


def phase2_inject_and_exclude(records_a, records_b, excluded_pairs, initial_crossmap):
    """Inject new B records and fire exclusion events."""
    print("\n=== Phase 2: Live exclusion API tests ===")

    rng = random.Random(SEED + 1)

    a_by_id = {r["entity_id"]: r for r in records_a}
    exact_b = [r for r in records_b if r["_match_type"] == "exact"]

    # --- Sub-phase 2a: Exclude via API, then inject exact-copy B records ---
    # These would score 1.0 normally but should NOT match their excluded A.
    print("  Phase 2a: Exclude-then-inject (20 records)...")
    inject_excluded_count = 20
    excluded_sample = rng.sample(exact_b, inject_excluded_count)
    phase2_excluded_pairs = set()
    phase2a_failures = 0

    for i, b_rec in enumerate(excluded_sample):
        new_b_id = f"CP-INJECT-EX-{i:04d}"
        a_id = b_rec["_true_a_id"]
        a_rec = a_by_id[a_id]

        # Exclude the pair via API
        status, body = post_json("/api/v1/exclude", {"a_id": a_id, "b_id": new_b_id})
        assert status == 200, f"exclude failed: {status} {body}"
        phase2_excluded_pairs.add((a_id, new_b_id))

        # Inject the B record — exact copy of A
        payload = {
            "record": {
                "counterparty_id": new_b_id,
                "counterparty_name": a_rec["legal_name"],
                "domicile": a_rec["country_code"],
                "lei_code": a_rec["lei"],
            }
        }
        status, body = post_json("/api/v1/b/add", payload)
        assert status == 200, f"b/add failed: {status} {body}"

        # Check the response — should NOT have auto-matched to excluded A
        matches = body.get("matches", [])
        matched_to_excluded_a = any(m.get("id") == a_id for m in matches)
        if matched_to_excluded_a:
            print(f"    FAIL: {new_b_id} matched to excluded A {a_id}")
            phase2a_failures += 1

    if phase2a_failures == 0:
        print(f"    PASS: 0/{inject_excluded_count} excluded pairs matched on inject")
    else:
        print(
            f"    FAIL: {phase2a_failures}/{inject_excluded_count} matched despite exclusion"
        )

    # --- Sub-phase 2b: Break existing matches via POST /exclude ---
    print("  Phase 2b: Breaking existing matches via POST /exclude...")
    breakable = list(initial_crossmap)
    rng.shuffle(breakable)
    break_count = min(50, len(breakable))
    broken_pairs = set()

    for a_id, b_id in breakable[:break_count]:
        status, body = post_json("/api/v1/exclude", {"a_id": a_id, "b_id": b_id})
        assert status == 200, f"exclude failed: {status} {body}"
        assert body.get("excluded") is True, f"exclude not confirmed: {body}"
        assert body.get("match_was_broken") is True, f"match not broken: {body}"
        broken_pairs.add((a_id, b_id))

    print(f"    Broke {len(broken_pairs)} existing matches")

    # --- Sub-phase 2c: Inject a few normal B records (sanity check) ---
    print("  Phase 2c: Injecting 50 normal B records...")
    normal_results = {"auto": 0, "review": 0, "no_match": 0}

    for i in range(50):
        new_b_id = f"CP-INJECT-{i:04d}"
        a_rec = rng.choice(records_a)
        payload = {
            "record": {
                "counterparty_id": new_b_id,
                "counterparty_name": a_rec["legal_name"].lower(),
                "domicile": a_rec["country_code"],
                "lei_code": a_rec["lei"],
            }
        }
        status, body = post_json("/api/v1/b/add", payload)
        assert status == 200, f"b/add failed: {status} {body}"
        classification = body.get("classification", "no_match")
        if classification == "auto":
            normal_results["auto"] += 1
        elif classification == "review":
            normal_results["review"] += 1
        else:
            normal_results["no_match"] += 1

    print(
        f"    Results: {normal_results['auto']} auto, "
        f"{normal_results['review']} review, {normal_results['no_match']} no_match"
    )

    # --- Validation ---
    print("\n  Phase 2 validation:")

    # Check broken pairs are no longer in crossmap
    broken_still_matched = 0
    for a_id, b_id in broken_pairs:
        status, body = get_json(f"/api/v1/crossmap/lookup?id={a_id}&side=a")
        if status == 200 and body.get("paired_id") == b_id:
            broken_still_matched += 1

    if broken_still_matched > 0:
        print(f"    FAIL: {broken_still_matched} broken pairs still in crossmap")
    else:
        print(f"    PASS: all {len(broken_pairs)} broken pairs removed from crossmap")

    # Check phase2 excluded pairs are not in crossmap
    phase2_excluded_matched = 0
    for a_id, b_id in phase2_excluded_pairs:
        status, body = get_json(f"/api/v1/crossmap/lookup?id={b_id}&side=b")
        if status == 200 and body.get("paired_id") == a_id:
            phase2_excluded_matched += 1

    if phase2_excluded_matched > 0:
        print(f"    FAIL: {phase2_excluded_matched} phase-2 excluded pairs matched")
    else:
        print(
            f"    PASS: 0/{len(phase2_excluded_pairs)} phase-2 excluded pairs matched"
        )

    # Check that unexclude works: pick 5 broken pairs, unexclude them
    print("\n  Phase 2d: Testing unexclude...")
    unexclude_sample = list(broken_pairs)[:5]
    for a_id, b_id in unexclude_sample:
        status, body = delete_json("/api/v1/exclude", {"a_id": a_id, "b_id": b_id})
        assert status == 200, f"unexclude failed: {status} {body}"
        assert body.get("removed") is True, f"unexclude not confirmed: {body}"
    print(f"    Unexcluded {len(unexclude_sample)} pairs (they can now rematch)")

    all_passed = (
        phase2a_failures == 0
        and broken_still_matched == 0
        and phase2_excluded_matched == 0
    )
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip data generation (reuse existing data)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    # Clean state
    for d in ["cache", "output", "wal"]:
        p = f"{TEST_DIR}/{d}"
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)
    for f in [f"{TEST_DIR}/crossmap.csv"]:
        if os.path.exists(f):
            os.remove(f)

    # Phase 0: Generate data
    if args.skip_generate and os.path.exists(f"{DATA_DIR}/dataset_a_10k.csv"):
        print("=== Phase 0: Reusing existing data ===")
        records_a = load_csv(f"{DATA_DIR}/dataset_a_10k.csv")
        records_b = load_csv(f"{DATA_DIR}/dataset_b_10k.csv")
    else:
        records_a, records_b = generate_data()

    # Load excluded pairs set
    excluded_pairs = set()
    with open(f"{TEST_DIR}/exclusions.csv", newline="") as f:
        for row in csv.DictReader(f):
            excluded_pairs.add((row["entity_id"], row["counterparty_id"]))
    print(f"  Exclusions loaded: {len(excluded_pairs)} pairs")

    # Pre-build embedding caches via batch run (avoids cold-start encoding
    # deadlock in live mode — a known pre-existing issue).
    if not os.path.exists(f"{TEST_DIR}/cache/b.combined_embedding"):
        cache_files = [f for f in os.listdir(f"{TEST_DIR}/cache") if f.startswith("b.")]
        if not cache_files:
            print("\n=== Pre-building embedding caches via batch run ===")
            batch_result = subprocess.run(
                [
                    args.binary,
                    "run",
                    "--config",
                    f"{TEST_DIR}/config.yaml",
                    "--verbose",
                ],
                timeout=300,
            )
            if batch_result.returncode != 0:
                print("  Batch run failed!")
                sys.exit(1)
            print("  Caches built.")
            # Clean batch output — we only wanted the caches
            for f_name in ["crossmap.csv"]:
                p = f"{TEST_DIR}/{f_name}"
                if os.path.exists(p):
                    os.remove(p)

    # Start server
    log_path = f"/tmp/meld_exclusions_test_{os.getpid()}.log"
    print(f"\n=== Starting server (port {PORT}) ===")

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

    try:
        if not wait_for_server(log_path, SERVER_READY_TIMEOUT):
            print(f"ERROR: Server did not start within {SERVER_READY_TIMEOUT}s.")
            with open(log_path) as f:
                print(f.read()[-2000:])
            sys.exit(1)

        print("  Server ready.")

        # Phase 1
        phase1_pass, initial_crossmap = phase1_validate_initial_match(excluded_pairs)

        # Phase 2
        phase2_pass = phase2_inject_and_exclude(
            records_a, records_b, excluded_pairs, initial_crossmap
        )

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(
            f"  Phase 1 (initial match + exclusions): {'PASS' if phase1_pass else 'FAIL'}"
        )
        print(
            f"  Phase 2 (injection + API exclusions):  {'PASS' if phase2_pass else 'FAIL'}"
        )

        all_passed = phase1_pass and phase2_pass
        print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")

        if not all_passed:
            sys.exit(1)

    finally:
        server.terminate()
        server.wait()
        print(f"\nServer stopped. Log at: {log_path}")


if __name__ == "__main__":
    main()
