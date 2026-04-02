#!/usr/bin/env python3
"""Warm batch benchmark — 1M × 1M, usearch backend, production scoring config.

Preserves the cache; clears outputs only. On first run the cache is empty
so the index will be built (very slow at 1M scale, ~30+ min). On subsequent
runs the cache is reused and only the scoring phase is timed.

WARNING: Expect ~30-40 GB memory usage for the in-memory store + embedding
index + BM25 index + blocking index at 1M records.

Requires the usearch feature:
    cargo build --release --features usearch

Run from the project root:
    python3 benchmarks/batch/1Mx1M_usearch/warm/run_test.py
"""

import argparse
import io
import os
import re
import selectors
import shutil
import subprocess
import sys
import time

TEST_DIR = "benchmarks/batch/1Mx1M_usearch/warm"
BINARY_DEFAULT = "./target/release/meld"


def run_with_tee(cmd):
    """Run a command, streaming stdout/stderr to the terminal in real-time
    while also capturing both for later parsing."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

    captured_out = io.BytesIO()
    captured_err = io.BytesIO()

    while sel.get_map():
        for key, _ in sel.select():
            data = key.fileobj.read1(8192)
            if not data:
                sel.unregister(key.fileobj)
                continue
            if key.data == "stdout":
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                captured_out.write(data)
            else:
                sys.stderr.buffer.write(data)
                sys.stderr.buffer.flush()
                captured_err.write(data)

    proc.wait()
    sel.close()

    return (
        proc.returncode,
        captured_out.getvalue().decode("utf-8", errors="replace"),
        captured_err.getvalue().decode("utf-8", errors="replace"),
    )


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_timings(stdout: str, stderr: str, wall_time: float):
    """Parse embedding and scoring timings from meld output."""
    timings = {}
    clean_err = strip_ansi(stderr)

    # Embedding build time from tracing output (stderr).
    # Format: combined embedding index built side="A" ... elapsed_s="123.4"
    emb_total = 0.0
    for m in re.finditer(
        r'combined embedding index built.*?elapsed_s="?(\d+\.?\d*)"?', clean_err
    ):
        emb_total += float(m.group(1))
    if emb_total > 0:
        timings["embedding"] = emb_total

    # Index load time from tracing output (stderr).
    # Format: loaded combined index ... elapsed_ms="46.0"
    load_total = 0.0
    for m in re.finditer(
        r'loaded combined index.*?elapsed_ms="?(\d+\.?\d*)"?', clean_err
    ):
        load_total += float(m.group(1)) / 1000.0
    if load_total > 0:
        timings["index_load"] = load_total

    # Scoring time from stdout summary.
    m = re.search(r"Scoring time:\s*(\d+\.?\d*)s", stdout)
    if m:
        timings["scoring"] = float(m.group(1))

    # Index build (BM25 + synonym) from stdout summary.
    m = re.search(r"Index build:\s*(\d+\.?\d*)s", stdout)
    if m:
        timings["bm25_synonym_build"] = float(m.group(1))

    # Total elapsed (run_batch) from stdout summary.
    m = re.search(r"Total elapsed:\s*(\d+\.?\d*)s", stdout)
    if m:
        timings["batch_total"] = float(m.group(1))

    # Throughput from stdout summary.
    m = re.search(r"Throughput:\s*(\d+\.?\d*)\s*records/sec", stdout)
    if m:
        timings["throughput"] = float(m.group(1))

    timings["wall_time"] = wall_time
    return timings


def print_timing_summary(timings: dict):
    """Print a breakdown of where time was spent."""
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN")
    print("=" * 60)

    emb = timings.get("embedding", 0)
    load = timings.get("index_load", 0)
    bm25 = timings.get("bm25_synonym_build", 0)
    scoring = timings.get("scoring", 0)
    wall = timings.get("wall_time", 0)

    if emb > 0:
        print(f"  Embedding encode:    {emb:8.1f}s")
    if load > 0:
        print(f"  Index load (cache):  {load:8.1f}s")
    if bm25 > 0:
        print(f"  BM25/synonym build:  {bm25:8.1f}s")
    if scoring > 0:
        print(f"  Scoring:             {scoring:8.1f}s")

    accounted = emb + load + bm25 + scoring
    other = wall - accounted
    if other > 1.0:
        print(f"  Other (data load):   {other:8.1f}s")

    print(f"  {'─' * 28}")
    print(f"  Total wall time:     {wall:8.1f}s")

    if timings.get("throughput"):
        print(f"\n  Scoring throughput:  {timings['throughput']:,.0f} records/sec")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N B records (for quick sanity checks)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    # Check datasets exist
    for ds in [
        "benchmarks/data/dataset_a_1000k.csv",
        "benchmarks/data/dataset_b_1000k.csv",
    ]:
        if not os.path.exists(ds):
            print(f"Dataset not found: {ds}")
            print("Generate with: python3 benchmarks/data/generate.py --size 1000000")
            sys.exit(1)

    cache_dir = f"{TEST_DIR}/cache"
    cache_populated = os.path.exists(cache_dir) and bool(os.listdir(cache_dir))
    if not cache_populated:
        print("Note: cache is empty — this run will build the index (~30+ min).")
        print("Run again afterwards for a true warm measurement.\n")

    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    print(
        f"=== Warm batch run (1M x 1M, usearch, production config): {TEST_DIR} ===\n",
        flush=True,
    )

    cmd = [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    start = time.time()
    returncode, stdout, stderr = run_with_tee(cmd)
    elapsed = time.time() - start

    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Parse and display timing breakdown
    timings = parse_timings(stdout, stderr, elapsed)
    print_timing_summary(timings)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
