#!/usr/bin/env python3
"""Run all live benchmarks in sequence and print a summary table.

Runs cold tests first, then warm tests. Because cold tests build the cache,
the immediately following warm test only needs one run (not two) to measure
true warm performance.

All tests use port 8090 and run sequentially — never in parallel.

Run from the project root:
    python3 benchmarks/live/run_all_tests.py
"""

import argparse
import os
import re
import sys
import subprocess
import time

BINARY_DEFAULT = "./target/release/meld"

# (label, script path)
TESTS = [
    ("flat 10k cold", "benchmarks/live/10kx10k_inject3k_flat/cold/run_test.py"),
    ("flat 10k warm", "benchmarks/live/10kx10k_inject3k_flat/warm/run_test.py"),
    ("usearch 10k cold", "benchmarks/live/10kx10k_inject3k_usearch/cold/run_test.py"),
    ("usearch 10k warm", "benchmarks/live/10kx10k_inject3k_usearch/warm/run_test.py"),
    (
        "usearch 100k cold",
        "benchmarks/live/100kx100k_inject10k_usearch/cold/run_test.py",
    ),
    (
        "usearch 100k warm",
        "benchmarks/live/100kx100k_inject10k_usearch/warm/run_test.py",
    ),
    (
        "sqlite 10k cold",
        "benchmarks/live/10kx10k_inject3k_usearch_sqlite/cold/run_test.py",
    ),
    (
        "sqlite 10k warm",
        "benchmarks/live/10kx10k_inject3k_usearch_sqlite/warm/run_test.py",
    ),
]


def run_and_capture(cmd):
    """Run cmd, stream output to terminal in real time, and return (returncode, output)."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def parse_metrics(output):
    """Extract key metrics from a live run_test.py output."""
    metrics = {}

    # "Total: 10,000 requests in 7.55s  →  1325.1 req/s"
    m = re.search(r"→\s+([\d.]+) req/s", output)
    if m:
        metrics["throughput"] = float(m.group(1))

    # ALL summary row: "ALL   3000   7.2   21.2   36.3   ..."
    m = re.search(r"^ALL\s+\d+\s+([\d.]+)\s+([\d.]+)", output, re.MULTILINE)
    if m:
        metrics["p50"] = float(m.group(1))
        metrics["p95"] = float(m.group(2))

    # Startup time: "Live state loaded in 208.9s" or "Built blocking indices in 7195.3ms"
    m = re.search(r"Live state loaded in ([\d.]+)s", output)
    if m:
        metrics["startup_s"] = float(m.group(1))

    return metrics


def fmt_throughput(v):
    return f"{v:,.0f} req/s" if v else "—"


def fmt_latency(v):
    return f"{v:.1f}ms" if v else "—"


def fmt_startup(v):
    if v is None:
        return "—"
    if v >= 60:
        return f"~{int(v // 60)}m {int(v % 60)}s"
    return f"~{v:.1f}s"


def fmt_elapsed(v):
    if v >= 60:
        return f"{int(v // 60)}m {int(v % 60)}s"
    return f"{v:.0f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    results = []
    failed = []

    for label, script in TESTS:
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  {label}")
        print(f"{sep}\n")

        start = time.time()
        rc, output = run_and_capture([sys.executable, script, "--binary", args.binary])
        elapsed = time.time() - start

        metrics = parse_metrics(output)
        metrics["elapsed"] = elapsed
        results.append((label, metrics))

        if rc != 0:
            failed.append(label)
            print(f"\n[FAILED] {label} exited with code {rc}")

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"  {'Test':<22}  {'Throughput':>12}  {'p50':>7}  {'p95':>7}  {'Startup':>10}"
    )
    print(f"  {'-' * 22}  {'-' * 12}  {'-' * 7}  {'-' * 7}  {'-' * 10}")
    for label, m in results:
        status = " [FAILED]" if label in failed else ""
        print(
            f"  {label:<22}  {fmt_throughput(m.get('throughput')):>12}"
            f"  {fmt_latency(m.get('p50')):>7}"
            f"  {fmt_latency(m.get('p95')):>7}"
            f"  {fmt_startup(m.get('startup_s')):>10}"
            f"{status}"
        )
    print()

    if failed:
        print(f"FAILED tests: {', '.join(failed)}")
        sys.exit(1)

    total = sum(m["elapsed"] for _, m in results)
    mins, secs = divmod(int(total), 60)
    print(f"All {len(TESTS)} tests passed. Total time: {mins}m {secs}s")


if __name__ == "__main__":
    main()
