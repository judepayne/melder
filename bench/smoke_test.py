#!/usr/bin/env python3
"""bench/smoke_test.py — Minimal live-mode smoke test.

Starts the server (stderr visible), waits for health, sends 10 upsert
requests (5 A-side, 5 B-side), prints each response + latency, then stops.

Usage:
    python bench/smoke_test.py
    python bench/smoke_test.py --no-serve   # if server already running on 8090
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error


BASE_URL = None  # set in main()


def wait_for_health(timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = urllib.request.urlopen(f"{BASE_URL}/api/v1/health", timeout=2)
            if r.status == 200:
                return json.loads(r.read())
        except Exception:
            pass
        time.sleep(0.25)
    return None


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
            ms = (time.perf_counter() - t0) * 1000
            return ms, r.status, body
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"raw": str(e)}
        return ms, e.code, body
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return ms, 0, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--config", default="testdata/configs/bench_live.yaml")
    parser.add_argument("--binary", default="./target/release/meld")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://localhost:{args.port}"

    proc = None
    if not args.no_serve:
        cmd = [args.binary, "serve", "--config", args.config, "--port", str(args.port)]
        print(f"Starting: {' '.join(cmd)}")
        t_start = time.monotonic()
        proc = subprocess.Popen(cmd)  # stderr/stdout visible in terminal

    try:
        print("Waiting for health...", end=" ", flush=True)
        t_health_start = time.monotonic()
        health = wait_for_health(timeout=60)
        if health is None:
            print("FAILED — server not ready within 30s")
            sys.exit(1)
        elapsed = time.monotonic() - (t_start if not args.no_serve else t_health_start)
        print(f"ready in {elapsed:.2f}s")
        print(f"  health: {health}\n")

        # 5 A-side adds
        print("--- A-side upserts ---")
        for i in range(1, 6):
            rec = {
                "entity_id": f"SMOKE-A-{i:03d}",
                "legal_name": f"Smoke Corp {i} GB",
                "short_name": f"SmokeCorp{i}",
                "country_code": "GB",
                "lei": f"SMOKE{i:015d}",
            }
            ms, status, body = post("/api/v1/a/add", {"record": rec})
            ok = "OK" if status == 200 else f"ERROR {status}"
            matches = body.get("matches", [])
            detail = body if status != 200 else f"matches={len(matches)}"
            print(f"  A{i}: {ok}  {ms:.1f}ms  {detail}")

        print()

        # 5 B-side adds
        print("--- B-side upserts ---")
        for i in range(1, 6):
            rec = {
                "counterparty_id": f"SMOKE-B-{i:03d}",
                "counterparty_name": f"Smoke Party {i} GB",
                "domicile": "GB",
                "lei_code": f"SMOKEB{i:014d}",
            }
            ms, status, body = post("/api/v1/b/add", {"record": rec})
            ok = "OK" if status == 200 else f"ERROR {status}"
            matches = body.get("matches", [])
            detail = body if status != 200 else f"matches={len(matches)}"
            print(f"  B{i}: {ok}  {ms:.1f}ms  {detail}")

        print("\nDone.")

    finally:
        if proc:
            print("\nStopping server...")
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
