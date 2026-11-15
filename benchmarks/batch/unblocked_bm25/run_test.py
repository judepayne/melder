#!/usr/bin/env python3
"""Reproducible cold, unblocked, BM25-only batch comparison (stdlib only).

Example (from any directory):
  python3 benchmarks/batch/unblocked_bm25/run_test.py \
    --baseline /tmp/meld-baseline --candidate ./target/release/meld \
    --pool-sizes 10000 100000 --queries 1000 --repeats 3 --threads 4 \
    --output /tmp/unblocked-bm25.json

Omit --candidate for baseline validation only. This never builds binaries.
A has pool-size rows; B has --queries rows, NOT pool-size rows. Each run
uses a fresh crossmap/output directory, with no blocking, exact fields,
common IDs, embedding scoring fields, or persistent indices. The CLI may still
initialize its default encoder at startup; this is outside CLI scoring time.
Selective queries contain a unique token; common queries also contain two
tokens present in every A row.
Every tenth B row has only an absent token, exercising unmatched outputs.
The remaining queries have distinct best matches, avoiding tie/claim races.
All CSV columns/rows (including rounded scores) are compared order-independently
across repeats and binaries; mismatches cause exit 1. Scoring duration is
estimated as query count / CLI throughput (rounded to whole records/sec),
avoiding the CLI duration's 0.1s rounding for fast runs. Both original rounded
durations are retained. Wall includes startup, data/index construction,
scoring, persistence and exports. CLI total minus rounded scoring estimates
batch setup/index-build time; wall minus scoring is broader overhead,
NOT an index-only measurement.
Artifacts/logs remain in the reported /tmp directory for diagnosis. Each
subprocess has a timeout; failures are saved to the JSON and cause exit 1.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def make_data(directory, size, queries, workload):
    shared = " common shared" if workload == "common" else ""
    a_path, b_path = directory / "a.csv", directory / "b.csv"
    with a_path.open("w", newline="") as source:
        writer = csv.writer(source)
        writer.writerow(["a_id", "text_a"])
        for index in range(size):
            writer.writerow([f"a{index:09d}", f"unique{index:09d}{shared}"])
    with b_path.open("w", newline="") as source:
        writer = csv.writer(source)
        writer.writerow(["b_id", "text_b"])
        for index in range(queries):
            # Evenly sample the whole A pool, not just its first queries rows.
            target = index * size // queries
            text = (f"absent{index:09d}" if index % 10 == 0
                    else f"unique{target:09d}{shared}")
            writer.writerow([f"b{index:09d}", text])
    return a_path, b_path


def canonical_outputs(output, queries):
    tables = {}
    seen = set()
    for name in ("relationships", "unmatched"):
        path = output / f"{name}.csv"
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "b_id" not in reader.fieldnames:
                raise ValueError(f"invalid output header: {path}")
            rows = [tuple(sorted(row.items())) for row in reader]
        tables[name] = sorted(rows)
        seen.update(dict(row)["b_id"] for row in rows)
    expected = {f"b{index:09d}" for index in range(queries)}
    if seen != expected:
        raise ValueError("outputs do not account for exactly all B query IDs")
    return tables


def run(binary, directory, a_path, b_path, threads, timeout, queries):
    directory.mkdir()
    config = {
        "job": {"name": "unblocked_bm25"},
        "datasets": {
            "a": {"path": str(a_path), "id_field": "a_id", "format": "csv"},
            "b": {"path": str(b_path), "id_field": "b_id", "format": "csv"},
        },
        "cross_map": {"backend": "local", "path": str(directory / "crossmap.csv"),
                      "a_id_field": "a_id", "b_id_field": "b_id"},
        # Required schema section; there are no embedding scoring fields.
        "embeddings": {"model": "all-MiniLM-L6-v2",
                       "a_cache_dir": str(directory / "unused-embedding-cache")},
        "blocking": {"enabled": False},
        "match_fields": [{"method": "bm25", "weight": 1.0,
                          "fields": [{"field_a": "text_a", "field_b": "text_b"}]}],
        "thresholds": {"auto_match": 0.8, "review_floor": 0.5},
        "output": {"csv_dir_path": str(directory / "output")},
        "top_n": 1,
        "bm25_candidates": 10,
    }
    config_path = directory / "config.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    command = [str(binary), "run", "--config", str(config_path)]
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads), NO_COLOR="1")
    start = time.perf_counter()
    record = {"command": command, "directory": str(directory)}
    try:
        with (directory / "cli.log").open("w") as log:
            process = subprocess.run(command, cwd=directory, env=env,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout, check=False)
        record["wall_seconds"] = time.perf_counter() - start
        record["returncode"] = process.returncode
        if process.returncode:
            raise ValueError(f"subprocess exited {process.returncode}")
        text = (directory / "cli.log").read_text()
        for label, key in (("Scoring time", "scoring_seconds"),
                           ("Total elapsed", "cli_total_seconds")):
            match = re.search(rf"{label}:\s*([0-9.]+)s", text)
            if not match:
                raise ValueError(f"missing CLI timing: {label}")
            record[key] = float(match.group(1))
        record["scoring_seconds_rounded"] = record["scoring_seconds"]
        rate_match = re.search(r"Throughput:\s*([0-9.]+) records/sec", text)
        if not rate_match or float(rate_match.group(1)) <= 0:
            raise ValueError("missing or zero CLI scoring throughput")
        record["reported_records_per_second"] = float(rate_match.group(1))
        record["scoring_seconds"] = queries / record["reported_records_per_second"]
        record["batch_index_seconds_rounded"] = max(
            0, record["cli_total_seconds"] - record["scoring_seconds_rounded"])
        record["wall_non_scoring_seconds"] = (
            record["wall_seconds"] - record["scoring_seconds"])
        tables = canonical_outputs(directory / "output", queries)
        record["output_counts"] = {key: len(rows) for key, rows in tables.items()}
        record["output_sha256"] = hashlib.sha256(
            json.dumps(tables, sort_keys=True).encode()).hexdigest()
        return record, tables
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        record["wall_seconds"] = time.perf_counter() - start
        record["error"] = str(error)
        return record, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--pool-sizes", nargs="+", type=positive, default=[10000, 100000])
    parser.add_argument("--queries", type=positive, default=1000)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--timeout", type=positive, default=600,
                        help="per subprocess timeout in seconds (default: 600)")
    parser.add_argument("--workloads", nargs="+", choices=["selective", "common"],
                        default=["selective", "common"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.queries > min(args.pool_sizes):
        parser.error("--queries must not exceed the smallest pool (distinct best matches)")
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        parser.error("--output must be outside the repository")
    binaries = {"baseline": args.baseline.resolve()}
    if args.candidate:
        binaries["candidate"] = args.candidate.resolve()
    for binary in binaries.values():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"not an executable file: {binary}")
    artifacts = Path(tempfile.mkdtemp(prefix="unblocked-bm25-", dir="/tmp"))
    report = {
        "artifacts": str(artifacts), "python": sys.version, "platform": platform.platform(),
        "threads": args.threads, "queries_b": args.queries, "repeats": args.repeats,
        "timeout_seconds": args.timeout, "cli_timing_resolution_seconds": 0.1,
        "scoring_seconds_method": "queries / whole-records-per-second CLI throughput",
        "binaries": {label: {"path": str(path), "sha256": digest(path)}
                     for label, path in binaries.items()},
        "cases": [], "ok": True,
    }
    for size in args.pool_sizes:
        for workload in args.workloads:
            directory = artifacts / f"{size}-{workload}"
            directory.mkdir()
            a_path, b_path = make_data(directory, size, args.queries, workload)
            case = {"pool_a": size, "queries_b": args.queries, "workload": workload,
                    "dataset_sha256": {"a": digest(a_path), "b": digest(b_path)},
                    "runs": [], "medians": {}, "semantics_equal": True}
            reference = None
            # Alternate order to reduce systematic thermal/cache/order bias.
            for repeat in range(args.repeats):
                order = list(binaries)
                if repeat % 2:
                    order.reverse()
                for label in order:
                    record, tables = run(binaries[label], directory / f"{label}-{repeat}",
                                         a_path, b_path, args.threads, args.timeout, args.queries)
                    record.update(binary=label, repeat=repeat)
                    if tables is None:
                        report["ok"] = False
                        case["semantics_equal"] = False
                    elif reference is None:
                        reference = tables
                    elif tables != reference:
                        record["error"] = "relationship/unmatched CSV differs from first baseline run"
                        case["semantics_equal"] = False
                        report["ok"] = False
                    case["runs"].append(record)
                    print(f"A={size} B={args.queries} {workload} {label} #{repeat + 1}: "
                          f"scoring={record.get('scoring_seconds')}s "
                          f"wall={record['wall_seconds']:.3f}s "
                          f"{record.get('error', 'OK')}", flush=True)
            for label in binaries:
                runs = [run for run in case["runs"] if run["binary"] == label]
                if any("error" in run for run in runs):
                    continue
                case["medians"][label] = {
                    key: statistics.median(run[key] for run in runs)
                    for key in ("scoring_seconds", "scoring_seconds_rounded", "wall_seconds", "cli_total_seconds",
                                "batch_index_seconds_rounded", "wall_non_scoring_seconds")}
            if all(label in case["medians"] for label in ("baseline", "candidate")):
                before, after = (case["medians"][label]["scoring_seconds"]
                                 for label in ("baseline", "candidate"))
                case["scoring_speedup"] = before / after if before > 0 and after > 0 else None
            report["cases"].append(case)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {output}\nArtifacts: {artifacts}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
