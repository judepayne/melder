#!/usr/bin/env python3
"""Soak test controller -- long-running live-mode stress test.

Starts a melder server (cold, 10k x 10k, usearch), then repeatedly injects
random-sized bursts of records with random sleep intervals between them.
Each burst is a mix of ~50% new records and ~50% upserts of existing records.
The upsert pool grows across bursts so later bursts update records created
by earlier ones.

After each burst the script collects diagnostics:
  - Burst performance (size, throughput, errors)
  - Server health (records_a, records_b, crossmap_entries)
  - Process memory (RSS via psutil or ps)
  - Disk usage (cache + WAL + crossmap)

All metrics are appended to soak_log.csv for post-run analysis.

Usage (from project root):
    python3 benchmarks/soak/10kx10k_usearch/run_test.py
    python3 benchmarks/soak/10kx10k_usearch/run_test.py --duration 0.5  # 30 min
    python3 benchmarks/soak/10kx10k_usearch/run_test.py --duration 24   # 24 hours
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from random import Random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DIR = "benchmarks/soak/10kx10k_usearch"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8090
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120

EXTRA_IDS_A_FILE = os.path.join(TEST_DIR, "extra_ids_a.txt")
EXTRA_IDS_B_FILE = os.path.join(TEST_DIR, "extra_ids_b.txt")
SOAK_LOG_FILE = os.path.join(TEST_DIR, "soak_log.csv")

SOAK_LOG_COLUMNS = [
    "timestamp",
    "burst_num",
    "burst_size",
    "duration_secs",
    "throughput_rps",
    "errors",
    "records_a",
    "records_b",
    "crossmap_entries",
    "healthy",
    "memory_rss_mb",
    "disk_usage_mb",
    "total_injected",
    "pool_a_size",
    "pool_b_size",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def wait_for_server(base_url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = urllib.request.urlopen(f"{base_url}/api/v1/health", timeout=2)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def get_health(base_url: str) -> dict:
    """Fetch /api/v1/health and return the parsed JSON, or an error dict."""
    try:
        r = urllib.request.urlopen(f"{base_url}/api/v1/health", timeout=5)
        return json.loads(r.read())
    except Exception as e:
        return {
            "status": f"error: {e}",
            "records_a": -1,
            "records_b": -1,
            "crossmap_entries": -1,
        }


def get_process_rss_mb(pid: int) -> float:
    """Get the RSS of a process in MB. Tries psutil first, falls back to ps."""
    try:
        import psutil

        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: parse `ps` output (macOS / Linux).
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            timeout=5,
        ).strip()
        return int(out) / 1024  # ps reports in KB
    except Exception:
        return -1.0


def get_disk_usage_mb(test_dir: str) -> float:
    """Sum the size of cache/, wal/, and crossmap.csv in the test directory."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(os.path.join(test_dir, "cache")):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _dirnames, filenames in os.walk(os.path.join(test_dir, "wal")):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    crossmap = os.path.join(test_dir, "crossmap.csv")
    if os.path.exists(crossmap):
        total += os.path.getsize(crossmap)
    return total / (1024 * 1024)


def append_ids_to_file(filepath: str, ids: list[str]) -> None:
    """Append IDs to a newline-delimited file."""
    if not ids:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        for id_str in ids:
            f.write(id_str + "\n")


def parse_marker(output: str, marker: str) -> str:
    """Parse a __MARKER__:value line from subprocess output."""
    for line in output.splitlines():
        if line.startswith(marker):
            return line[len(marker) :]
    return ""


def clean_test_dir(test_dir: str) -> None:
    """Remove runtime artifacts for a clean cold start."""
    shutil.rmtree(os.path.join(test_dir, "cache"), ignore_errors=True)
    os.makedirs(os.path.join(test_dir, "cache"), exist_ok=True)
    shutil.rmtree(os.path.join(test_dir, "output"), ignore_errors=True)
    os.makedirs(os.path.join(test_dir, "output"), exist_ok=True)
    for f in glob.glob(os.path.join(test_dir, "wal", "*.ndjson")):
        os.remove(f)
    os.makedirs(os.path.join(test_dir, "wal"), exist_ok=True)
    crossmap = os.path.join(test_dir, "crossmap.csv")
    if os.path.exists(crossmap):
        os.remove(crossmap)
    # Clean extra-ID files and soak log from previous runs.
    for f in [EXTRA_IDS_A_FILE, EXTRA_IDS_B_FILE, SOAK_LOG_FILE]:
        if os.path.exists(f):
            os.remove(f)


def init_soak_log() -> None:
    """Write the CSV header for the soak log."""
    with open(SOAK_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOAK_LOG_COLUMNS)


def append_soak_log(row: dict) -> None:
    """Append a single row to the soak log CSV."""
    with open(SOAK_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in SOAK_LOG_COLUMNS])


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soak test controller -- long-running live-mode stress test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Total run time in hours",
    )
    parser.add_argument(
        "--min-burst",
        type=int,
        default=3000,
        help="Minimum injection burst size",
    )
    parser.add_argument(
        "--max-burst",
        type=int,
        default=50000,
        help="Maximum injection burst size",
    )
    parser.add_argument(
        "--min-sleep",
        type=int,
        default=60,
        help="Minimum sleep between bursts (seconds)",
    )
    parser.add_argument(
        "--max-sleep",
        type=int,
        default=600,
        help="Maximum sleep between bursts (seconds)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="HTTP worker threads for injection",
    )
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    rng = Random(args.seed)
    base_url = f"http://localhost:{args.port}"
    duration_secs = args.duration * 3600
    inject_script = os.path.join(TEST_DIR, "live_concurrent_test.py")

    # ------------------------------------------------------------------
    # Clean and start
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  MELDER SOAK TEST")
    print("=" * 72)
    print(f"  Duration:       {args.duration}h ({format_duration(duration_secs)})")
    print(f"  Burst range:    {args.min_burst:,} - {args.max_burst:,} records")
    print(f"  Sleep range:    {args.min_sleep}s - {args.max_sleep}s")
    print(f"  Concurrency:    {args.concurrency} workers")
    print(f"  Port:           {args.port}")
    print(f"  Binary:         {args.binary}")
    print(f"  Seed:           {args.seed}")
    print(f"  Log file:       {SOAK_LOG_FILE}")
    print("=" * 72)

    clean_test_dir(TEST_DIR)
    init_soak_log()

    log_path = f"/tmp/meld_soak_{os.getpid()}.log"
    print("\nStarting server (cold)...")

    server = subprocess.Popen(
        [
            args.binary,
            "serve",
            "--config",
            os.path.join(TEST_DIR, "config.yaml"),
            "--port",
            str(args.port),
        ],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    try:
        if not wait_for_server(base_url, SERVER_READY_TIMEOUT):
            print(f"ERROR: Server did not start within {SERVER_READY_TIMEOUT}s.")
            with open(log_path) as f:
                print(f.read())
            server.terminate()
            sys.exit(1)

        print("Server ready.\n")

        # ------------------------------------------------------------------
        # Soak loop
        # ------------------------------------------------------------------
        t_start = time.monotonic()
        burst_num = 0
        total_injected = 0
        a_counter_offset = 0
        b_counter_offset = 0
        peak_memory_mb = 0.0
        peak_disk_mb = 0.0

        while True:
            elapsed = time.monotonic() - t_start
            remaining = duration_secs - elapsed
            if remaining <= 0:
                break

            burst_num += 1
            burst_size = rng.randint(args.min_burst, args.max_burst)

            print("-" * 72)
            print(
                f"BURST {burst_num}  |  "
                f"size={burst_size:,}  |  "
                f"elapsed={format_duration(elapsed)}  |  "
                f"remaining={format_duration(remaining)}"
            )
            print("-" * 72)

            # Run the inject script.
            # Use --encoding-pct 50 to get ~50% new records, ~50% upserts.
            # Vary the seed per burst so each burst gets a different shuffle.
            burst_seed = args.seed + burst_num
            cmd = [
                sys.executable,
                inject_script,
                "--no-serve",
                "--port",
                str(args.port),
                "--iterations",
                str(burst_size),
                "--concurrency",
                str(args.concurrency),
                "--a-path",
                A_DATA,
                "--b-path",
                B_DATA,
                "--encoding-pct",
                "50",
                "--seed",
                str(burst_seed),
                "--extra-ids-a",
                EXTRA_IDS_A_FILE,
                "--extra-ids-b",
                EXTRA_IDS_B_FILE,
                "--a-counter-offset",
                str(a_counter_offset),
                "--b-counter-offset",
                str(b_counter_offset),
            ]

            t_burst_start = time.monotonic()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(3600, burst_size * 2),  # generous timeout
            )
            t_burst_secs = time.monotonic() - t_burst_start

            # Print inject script output (filtered: skip progress lines).
            for line in result.stdout.splitlines():
                if line.startswith("  ") and "/" in line and "req/s" in line:
                    continue  # skip progress lines
                if line.startswith("__"):
                    continue  # skip markers
                print(f"  {line}")

            if result.returncode != 0:
                print(f"  WARNING: inject script exited with code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.splitlines()[-5:]:
                        print(f"  stderr: {line}")

            # Parse markers from output.
            throughput = float(parse_marker(result.stdout, "__THROUGHPUT__:") or "0")
            wall_secs = float(
                parse_marker(result.stdout, "__WALL_SECS__:") or str(t_burst_secs)
            )
            error_count = int(parse_marker(result.stdout, "__ERRORS__:") or "0")

            new_a_ctr = parse_marker(result.stdout, "__A_COUNTER__:")
            new_b_ctr = parse_marker(result.stdout, "__B_COUNTER__:")
            if new_a_ctr:
                a_counter_offset = int(new_a_ctr)
            if new_b_ctr:
                b_counter_offset = int(new_b_ctr)

            # Grow the ID pool files with newly created IDs.
            new_a_ids_str = parse_marker(result.stdout, "__NEW_A_IDS__:")
            new_b_ids_str = parse_marker(result.stdout, "__NEW_B_IDS__:")
            new_a_ids = new_a_ids_str.split(",") if new_a_ids_str else []
            new_b_ids = new_b_ids_str.split(",") if new_b_ids_str else []
            append_ids_to_file(EXTRA_IDS_A_FILE, new_a_ids)
            append_ids_to_file(EXTRA_IDS_B_FILE, new_b_ids)

            total_injected += burst_size

            # Count pool sizes.
            pool_a_size = 10000 + a_counter_offset  # base 10k + newly created
            pool_b_size = 10000 + b_counter_offset

            # ------------------------------------------------------------------
            # Diagnostics
            # ------------------------------------------------------------------
            health = get_health(base_url)
            healthy = health.get("status") == "ok"
            records_a = health.get("records_a", -1)
            records_b = health.get("records_b", -1)
            crossmap_entries = health.get("crossmap_entries", -1)

            memory_mb = get_process_rss_mb(server.pid)
            disk_mb = get_disk_usage_mb(TEST_DIR)

            peak_memory_mb = max(peak_memory_mb, memory_mb)
            peak_disk_mb = max(peak_disk_mb, disk_mb)

            print()
            print(f"  Burst performance:")
            print(f"    Records injected:  {burst_size:,}")
            print(f"    Duration:          {wall_secs:.1f}s")
            print(f"    Throughput:        {throughput:.1f} req/s")
            print(f"    Errors:            {error_count}")
            print()
            print(f"  Server state:")
            print(f"    Healthy:           {'YES' if healthy else 'NO'}")
            print(f"    Records A:         {records_a:,}")
            print(f"    Records B:         {records_b:,}")
            print(f"    Crossmap entries:  {crossmap_entries:,}")
            print()
            print(f"  Resources:")
            print(f"    Memory (RSS):      {memory_mb:.1f} MB")
            print(f"    Disk usage:        {disk_mb:.1f} MB")
            print()
            print(f"  Cumulative:")
            print(f"    Total injected:    {total_injected:,}")
            print(f"    Update pool:       {pool_a_size:,} A / {pool_b_size:,} B")
            print(f"    Peak memory:       {peak_memory_mb:.1f} MB")
            print(f"    Peak disk:         {peak_disk_mb:.1f} MB")

            # Append to soak log.
            append_soak_log(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "burst_num": burst_num,
                    "burst_size": burst_size,
                    "duration_secs": f"{wall_secs:.2f}",
                    "throughput_rps": f"{throughput:.2f}",
                    "errors": error_count,
                    "records_a": records_a,
                    "records_b": records_b,
                    "crossmap_entries": crossmap_entries,
                    "healthy": "true" if healthy else "false",
                    "memory_rss_mb": f"{memory_mb:.1f}",
                    "disk_usage_mb": f"{disk_mb:.1f}",
                    "total_injected": total_injected,
                    "pool_a_size": pool_a_size,
                    "pool_b_size": pool_b_size,
                }
            )

            # Check if server is still alive.
            if server.poll() is not None:
                print("\nERROR: Server process has died!")
                print(f"Exit code: {server.returncode}")
                with open(log_path) as f:
                    tail = f.readlines()[-20:]
                    for line in tail:
                        print(f"  {line}", end="")
                sys.exit(1)

            # Sleep before next burst (if time remains).
            remaining = duration_secs - (time.monotonic() - t_start)
            if remaining <= 0:
                break

            sleep_secs = rng.randint(args.min_sleep, args.max_sleep)
            sleep_secs = min(sleep_secs, int(remaining))
            if sleep_secs > 0:
                print(f"\n  Sleeping {sleep_secs}s until next burst...\n")
                time.sleep(sleep_secs)

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        total_elapsed = time.monotonic() - t_start
        final_health = get_health(base_url)
        final_memory = get_process_rss_mb(server.pid)
        final_disk = get_disk_usage_mb(TEST_DIR)

        print()
        print("=" * 72)
        print("  SOAK TEST COMPLETE")
        print("=" * 72)
        print(f"  Total elapsed:       {format_duration(total_elapsed)}")
        print(f"  Total bursts:        {burst_num}")
        print(f"  Total injected:      {total_injected:,}")
        print(f"  Final records A:     {final_health.get('records_a', '?'):,}")
        print(f"  Final records B:     {final_health.get('records_b', '?'):,}")
        print(f"  Final crossmap:      {final_health.get('crossmap_entries', '?'):,}")
        print(f"  Final memory (RSS):  {final_memory:.1f} MB")
        print(f"  Final disk usage:    {final_disk:.1f} MB")
        print(f"  Peak memory:         {peak_memory_mb:.1f} MB")
        print(f"  Peak disk:           {peak_disk_mb:.1f} MB")
        print(f"  Log file:            {SOAK_LOG_FILE}")
        print("=" * 72)

    finally:
        print("\nStopping server (SIGTERM)...")
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
