#!/usr/bin/env python3
"""Run all batch benchmarks in sequence and print a summary table.

Runs cold tests first, then warm tests. Because cold tests build the cache,
the immediately following warm test only needs one run (not two) to measure
true warm performance.

BM25 tests require a binary built with --features bm25. Pass --bm25-binary
to use a separate binary for those tests, otherwise they are skipped when
the default binary lacks the feature.

Run from the project root:
    python3 benchmarks/batch/run_all_tests.py
    python3 benchmarks/batch/run_all_tests.py --bm25-binary ./target/release/meld-bm25
"""

import argparse
import os
import re
import sys
import subprocess
import time

BINARY_DEFAULT = "./target/release/meld"

# (label, script path, requires_bm25)
TESTS = [
    ("flat 10k cold", "benchmarks/batch/10kx10k_flat/cold/run_test.py", False),
    ("flat 10k warm", "benchmarks/batch/10kx10k_flat/warm/run_test.py", False),
    ("usearch 10k cold", "benchmarks/batch/10kx10k_usearch/cold/run_test.py", False),
    ("usearch 10k warm", "benchmarks/batch/10kx10k_usearch/warm/run_test.py", False),
    ("usearch 100k cold", "benchmarks/batch/100kx100k_usearch/cold/run_test.py", False),
    ("usearch 100k warm", "benchmarks/batch/100kx100k_usearch/warm/run_test.py", False),
    (
        "quantized 100k cold",
        "benchmarks/batch/100kx100k_usearch_quantized/cold/run_test.py",
        False,
    ),
    ("f16 100k warm", "benchmarks/batch/100kx100k_usearch_f16/warm/run_test.py", False),
    (
        "mmap 100k warm",
        "benchmarks/batch/100kx100k_usearch_mmap/warm/run_test.py",
        False,
    ),
    (
        "usearch+bm25 10k cold",
        "benchmarks/batch/10kx10k_usearch_bm25/cold/run_test.py",
        True,
    ),
    ("bm25-only 10k cold", "benchmarks/batch/10kx10k_bm25only/cold/run_test.py", True),
    (
        "remote-encoder 10k cold",
        "benchmarks/batch/10kx10k_remote_encoder/cold/run_test.py",
        False,
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
    """Extract key metrics from a batch run_test.py output."""
    metrics = {}

    m = re.search(r"Throughput:\s+([\d,]+) records/sec", output)
    if m:
        metrics["throughput"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Total wall time:\s+([\d.]+)s", output)
    if m:
        metrics["wall_time"] = float(m.group(1))

    # Cold: "Built blocking indices in 16927.9ms" covers full index build
    m = re.search(r"Built blocking indices in ([\d.]+)ms", output)
    if m:
        metrics["index_build_s"] = float(m.group(1)) / 1000.0

    # Warm: sum of A + B cache load times
    loads = re.findall(r"all fresh in ([\d.]+)ms", output)
    if loads:
        metrics["cache_load_ms"] = sum(float(x) for x in loads)

    return metrics


def fmt_throughput(v):
    return f"{v:,} rec/s" if v else "—"


def fmt_wall(v):
    if v is None:
        return "—"
    if v >= 60:
        return f"{int(v // 60)}m {int(v % 60)}s"
    return f"{v:.1f}s"


def fmt_build(v):
    if v is None:
        return "—"
    if v >= 60:
        return f"~{int(v // 60)}m {int(v % 60)}s"
    return f"~{v:.0f}s"


def fmt_cache(v):
    return f"~{v:.0f}ms" if v else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    parser.add_argument(
        "--bm25-binary",
        default=None,
        help="Path to a binary built with --features bm25 (required for BM25 tests)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    results = []
    failed = []
    skipped = []

    for label, script, requires_bm25 in TESTS:
        if requires_bm25:
            binary = args.bm25_binary or args.binary
        else:
            binary = args.binary

        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  {label}")
        print(f"{sep}\n")

        start = time.time()
        rc, output = run_and_capture([sys.executable, script, "--binary", binary])
        elapsed = time.time() - start

        metrics = parse_metrics(output)
        metrics["elapsed"] = elapsed

        if rc != 0 and requires_bm25 and "Binary not found" in output:
            print(f"  [SKIPPED] {label} — pass --bm25-binary to run BM25 tests")
            skipped.append(label)
            metrics["skipped"] = True

        results.append((label, metrics))

        if rc != 0 and label not in skipped:
            failed.append(label)
            print(f"\n[FAILED] {label} exited with code {rc}")

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"  {'Test':<22}  {'Throughput':>14}  {'Wall time':>10}  {'Index build':>12}  {'Cache load':>10}"
    )
    print(f"  {'-' * 22}  {'-' * 14}  {'-' * 10}  {'-' * 12}  {'-' * 10}")
    for label, m in results:
        if m.get("skipped"):
            status = " [SKIPPED]"
        elif label in failed:
            status = " [FAILED]"
        else:
            status = ""
        print(
            f"  {label:<22}  {fmt_throughput(m.get('throughput')):>14}"
            f"  {fmt_wall(m.get('wall_time')):>10}"
            f"  {fmt_build(m.get('index_build_s')):>12}"
            f"  {fmt_cache(m.get('cache_load_ms')):>10}"
            f"{status}"
        )
    print()

    if skipped:
        print(f"Skipped (no BM25 binary): {', '.join(skipped)}")
    if failed:
        print(f"FAILED tests: {', '.join(failed)}")
        sys.exit(1)

    total = sum(m["elapsed"] for _, m in results if not m.get("skipped"))
    ran = len(TESTS) - len(skipped)
    print(f"{ran}/{len(TESTS)} tests ran. Total time: {fmt_wall(total)}")


if __name__ == "__main__":
    main()
