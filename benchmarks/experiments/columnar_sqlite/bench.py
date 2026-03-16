#!/usr/bin/env python3
"""
Experiment: JSON blob vs columnar storage in SQLite.

Measures the two hot-path operations during batch scoring:
  1. get_many: fetch ~500 records by ID, deserialize to dict
  2. blocking_query: look up candidate IDs by blocking key

Tests three approaches:
  A. JSON blob (current): SELECT id, record_json WHERE id IN (...)
  B. json_extract: SELECT id, json_extract(...) WHERE id IN (...)
  C. Columnar: SELECT id, field1, field2, ... WHERE id IN (...)

Uses the same 10K dataset as the batch benchmarks.
"""

import sqlite3
import json
import csv
import time
import random
import os
import statistics

DATASET = "benchmarks/data/dataset_a_10k.csv"
DB_PATH = "/tmp/columnar_experiment.db"
N_QUERIES = 2000  # number of get_many calls
CANDIDATES = 500  # IDs per get_many call
SCORING_FIELDS = ["short_name", "country_code", "lei"]  # fields needed for scoring
REPEAT = 3


def load_csv(path):
    """Load CSV into list of (id, dict) pairs."""
    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["entity_id"]
            records.append((rid, dict(row)))
    return records


def setup_json_db(conn, records):
    """Create JSON blob table and insert records."""
    conn.execute("DROP TABLE IF EXISTS json_records")
    conn.execute("""
        CREATE TABLE json_records (
            id TEXT PRIMARY KEY,
            record_json TEXT NOT NULL
        )
    """)
    conn.execute("BEGIN")
    for rid, rec in records:
        conn.execute(
            "INSERT INTO json_records (id, record_json) VALUES (?, ?)",
            (rid, json.dumps(rec)),
        )
    conn.execute("COMMIT")


def setup_columnar_db(conn, records):
    """Create columnar table with one column per field."""
    if not records:
        return
    fields = list(records[0][1].keys())

    conn.execute("DROP TABLE IF EXISTS col_records")
    cols = ", ".join(f'"{f}" TEXT' for f in fields if f != "entity_id")
    conn.execute(f"""
        CREATE TABLE col_records (
            id TEXT PRIMARY KEY,
            {cols}
        )
    """)

    placeholders = ", ".join(["?"] * len(fields))
    col_names = ", ".join(
        f'"{f}"' for f in ["id"] + [f for f in fields if f != "entity_id"]
    )

    conn.execute("BEGIN")
    for rid, rec in records:
        vals = [rid] + [rec.get(f, "") for f in fields if f != "entity_id"]
        conn.execute(
            f"INSERT INTO col_records ({col_names}) VALUES ({placeholders})", vals
        )
    conn.execute("COMMIT")


def bench_get_many_json(conn, id_batches):
    """Benchmark: fetch + json.loads for each batch."""
    total_records = 0
    start = time.perf_counter()
    for ids in id_batches:
        placeholders = ",".join(["?"] * len(ids))
        cur = conn.execute(
            f"SELECT id, record_json FROM json_records WHERE id IN ({placeholders})",
            ids,
        )
        for row in cur:
            rec = json.loads(row[1])
            total_records += 1
    elapsed = time.perf_counter() - start
    return elapsed, total_records


def bench_get_many_json_extract(conn, id_batches, fields):
    """Benchmark: json_extract for specific fields."""
    field_exprs = ", ".join(
        f"json_extract(record_json, '$.{f}') AS \"{f}\"" for f in fields
    )
    total_records = 0
    start = time.perf_counter()
    for ids in id_batches:
        placeholders = ",".join(["?"] * len(ids))
        cur = conn.execute(
            f"SELECT id, {field_exprs} FROM json_records WHERE id IN ({placeholders})",
            ids,
        )
        for row in cur:
            rec = {fields[i]: row[i + 1] for i in range(len(fields))}
            total_records += 1
    elapsed = time.perf_counter() - start
    return elapsed, total_records


def bench_get_many_columnar(conn, id_batches, fields):
    """Benchmark: direct column reads."""
    col_list = ", ".join(f'"{f}"' for f in fields)
    total_records = 0
    start = time.perf_counter()
    for ids in id_batches:
        placeholders = ",".join(["?"] * len(ids))
        cur = conn.execute(
            f"SELECT id, {col_list} FROM col_records WHERE id IN ({placeholders})", ids
        )
        for row in cur:
            rec = {fields[i]: row[i + 1] for i in range(len(fields))}
            total_records += 1
    elapsed = time.perf_counter() - start
    return elapsed, total_records


def bench_get_many_columnar_all(conn, id_batches, all_fields):
    """Benchmark: columnar read of ALL fields (for output_mapping)."""
    col_list = ", ".join(f'"{f}"' for f in all_fields)
    total_records = 0
    start = time.perf_counter()
    for ids in id_batches:
        placeholders = ",".join(["?"] * len(ids))
        cur = conn.execute(
            f"SELECT id, {col_list} FROM col_records WHERE id IN ({placeholders})", ids
        )
        for row in cur:
            rec = {all_fields[i]: row[i + 1] for i in range(len(all_fields))}
            total_records += 1
    elapsed = time.perf_counter() - start
    return elapsed, total_records


def main():
    print(f"Loading {DATASET}...")
    records = load_csv(DATASET)
    all_ids = [r[0] for r in records]
    all_fields = [f for f in records[0][1].keys() if f != "entity_id"]
    print(f"  {len(records)} records, {len(all_fields)} fields per record")
    print(f"  Scoring fields: {SCORING_FIELDS}")
    print(f"  All fields: {all_fields}")
    print()

    # Generate random ID batches (simulating blocking buckets)
    random.seed(42)
    id_batches = []
    for _ in range(N_QUERIES):
        batch = random.sample(all_ids, min(CANDIDATES, len(all_ids)))
        id_batches.append(batch)

    total_lookups = N_QUERIES * CANDIDATES
    print(
        f"Benchmark: {N_QUERIES} queries × {CANDIDATES} candidates = {total_lookups:,} record lookups"
    )
    print(f"Repeat: {REPEAT} runs each")
    print()

    # Setup
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = -131072")  # 128MB
    conn.execute("PRAGMA synchronous = NORMAL")

    print("Setting up JSON table...")
    setup_json_db(conn, records)
    print("Setting up columnar table...")
    setup_columnar_db(conn, records)
    print()

    # Warm the cache
    conn.execute("SELECT count(*) FROM json_records").fetchone()
    conn.execute("SELECT count(*) FROM col_records").fetchone()

    # Run benchmarks
    results = {}

    for label, bench_fn, args in [
        ("A. JSON blob (full deserialize)", bench_get_many_json, (conn, id_batches)),
        (
            "B. json_extract (3 fields)",
            bench_get_many_json_extract,
            (conn, id_batches, SCORING_FIELDS),
        ),
        (
            "C. Columnar (3 scoring fields)",
            bench_get_many_columnar,
            (conn, id_batches, SCORING_FIELDS),
        ),
        (
            "D. Columnar (all fields)",
            bench_get_many_columnar_all,
            (conn, id_batches, all_fields),
        ),
    ]:
        times = []
        for r in range(REPEAT):
            elapsed, count = bench_fn(*args)
            times.append(elapsed)

        avg = statistics.mean(times)
        rate = total_lookups / avg
        per_lookup_us = avg / total_lookups * 1_000_000
        results[label] = (avg, rate, per_lookup_us)
        print(f"{label}")
        print(
            f"  {avg:.3f}s avg  |  {rate:,.0f} lookups/s  |  {per_lookup_us:.2f} μs/lookup"
        )
        print()

    # Summary
    json_time = results["A. JSON blob (full deserialize)"][0]
    print("=" * 60)
    print("Summary (speedup vs JSON blob):")
    for label, (avg, rate, per_us) in results.items():
        speedup = json_time / avg
        print(f"  {label}: {speedup:.2f}x")

    # Cleanup
    conn.close()
    os.remove(DB_PATH)


if __name__ == "__main__":
    main()
