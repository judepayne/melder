← [Back to Index](./) | [Configuration](configuration.md) | [Live Mode](live-mode.md)

# Batch Mode

> For a hands-on walkthrough, see the
> [batch worked example](../examples/batch/README.md).

Batch mode processes an entire B dataset against a pre-indexed A pool in
a single pass. Run it with:

```bash
meld run --config config.yaml
```

Batch mode works unchanged whether embeddings come from a local ONNX
model (`embeddings.model`) or a user-supplied subprocess calling your
organisation's central embedding service
(`embeddings.remote_encoder_cmd`). See [Remote Encoder](remote-encoder.md)
for the subprocess path. A worked batch example using the remote encoder
stub lives at `benchmarks/batch/10kx10k_remote_encoder/cold/`.

## Command flags

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--dry-run` | | Validate config, load data, print what would be processed, then exit. No matching or output files. |
| `--limit` | | Process only the first N B records. Useful for quick sanity checks on large datasets. |
| `--verbose` | `-v` | Print job metadata, dataset paths, and threshold values at startup. |

Use `--dry-run` to validate the config and print what would be processed
without actually running. Use `--limit N` to process only the first N
B records (useful for quick sanity checks on large datasets).

## Output files

Batch mode is event-sourced: during scoring, events are written to a
match log on disk. At the end of the run, the build pipeline reads the
match log (and the optional scoring log) to produce outputs. This means
a crashed run leaves a recoverable match log, and outputs can be rebuilt
with `meld export` without re-scoring.

Output is written to any combination of three independent sinks:
`output.csv_dir_path` (directory for CSV files),
`output.parquet_dir_path` (directory for Parquet files), and
`output.db_path` (SQLite database). At least one must be set. Setting
more than one is allowed — all enabled sinks read from the same match
log, so enabling Parquet alongside CSV adds no extra scoring work.

### CSV outputs

| File | Contents |
|------|----------|
| `relationships.csv` | All confirmed matches and review-band pairs. Columns: `b_id`, `a_id`, `score`, `relationship_type` (`match` or `review`), `reason`, `rank`, plus configured A-side fields. Replaces the former `results.csv` + `review.csv` split. |
| `unmatched.csv` | B records with no match above `review_floor`. Columns: `b_id`, B-side fields, `best_score`, `best_a_id`. The `best_a_id` column is new — it shows the closest A-side record even when below threshold. |
| `candidates.csv` | Produced only when the scoring log is enabled. Ranks 2..N for every scored query record. Columns: `b_id`, `rank`, `a_id`, `score`. |

### Parquet outputs

When `output.parquet_dir_path` is set, the build pipeline writes
`relationships.parquet`, `unmatched.parquet`, and (when the scoring log
is enabled) `candidates.parquet` alongside any CSV output. Contents
match the CSV files row-for-row. Schemas: record field columns are
written as `Utf8` (the internal "everything is string" dataset model),
while engine-generated columns are typed — `score` as `Float64`,
`rank` as `UInt8`. Writes are atomic (`*.parquet.tmp` then rename).

Requires the `parquet-format` feature at build time (see
[Building](building.md)). If the binary was built without the feature,
starting a run with `parquet_dir_path` set returns an error.

### SQLite output database

When `output.db_path` is set, the build pipeline writes a SQLite
database with tables `a_records`, `b_records`, `relationships`,
`field_scores` (populated when the scoring log is on), and `metadata`.
Eleven analytical views are included (e.g. `confirmed_matches`,
`review_queue`, `unmatched_a`, `unmatched_b`, `relationship_detail`).
See the output data design for the full schema.

### Match log

The match log is written during scoring to a file in the output
directory. By default it is kept after the build for debugging and
rebuilds. Set `output.cleanup_match_log: true` to delete it after a
successful build.

The cross-map csv is also updated: every auto-matched pair is added so
that re-running the job skips already-resolved records.

## Stdout summary

A summary is printed to stdout at the end:

```
Batch matching complete:
  Total B records: 1000
  Skipped (crossmap): 0
  Auto-matched: 712
  Review:       138
  No match:     150
  Index build:  14.2s
  Scoring time: 8.3s
  Total elapsed:22.5s
  Throughput:   120 records/sec

Output files:
  relationships: output/relationships.csv (850 rows)
  unmatched:     output/unmatched.csv (150 rows)
```

## SQLite batch mode (large datasets)

For datasets that exceed available RAM (e.g. 55M A records would need
~100 GB in memory), set `batch.db_path` to store records in SQLite
instead:

```yaml
batch:
  db_path: batch.db
```

The database is created fresh each run and deleted on completion — the
source CSV files, crossmap.csv, and exclusions.csv (if configured) remain
the only persistent state.
Records are stored in columnar format (one column per field, no JSON
serialization) for fast scoring. Data is loaded via streaming — only one
10K-record chunk is in memory at a time, regardless of dataset size.

### Memory footprint

The memory footprint is approximately: `sqlite_cache_mb + pool_size ×
pool_worker_cache_mb + BM25 index + blocking index` (typically 10-12 GB
for a 55M-record dataset).

### Configuration

```yaml
batch:
  db_path: batch.db                 # optional — triggers SQLite batch mode.
                                    #   The file is created fresh and deleted after the run.
                                    #   Default: not set (in-memory storage).
  sqlite_read_pool_size: 8          # optional — read connection pool size.
                                    #   Default: num_cpus (matches Rayon parallelism).
  sqlite_pool_worker_cache_mb: 128  # optional — page cache per read connection in MB.
                                    #   Total read cache = pool_size × this value.
                                    #   Default: 128.
  sqlite_cache_mb: 64               # optional — write connection page cache in MB.
                                    #   Default: 64.
```

### When to use SQLite batch mode

- Datasets larger than ~50% of available RAM (to avoid swap pressure)
- Any dataset where you want predictable, bounded memory usage

### When to use in-memory batch mode (the default)

- Datasets that fit comfortably in RAM
- Maximum scoring throughput is needed (in-memory is ~1.6x faster)

**Note:** SQLite batch mode currently supports BM25 + fuzzy + exact
scoring methods. Embedding-based scoring requires the in-memory path
(embedding indices are held in RAM regardless of storage backend).

## Data Formats

The melder reads CSV (default), JSONL/NDJSON, and Parquet files. The
format is inferred from the file extension or can be set explicitly in
the config:

```yaml
datasets:
  a:
    path: data.jsonl
    format: jsonl   # csv, jsonl, ndjson, parquet
```

Parquet support (both input and output) requires a feature flag at
build time:

```bash
cargo build --release --features parquet-format
```

All input column types (string, integer, float, boolean) are converted
to strings internally to provide a uniform interface across formats.
Snappy-compressed Parquet files are supported on input.

Parquet output is controlled by `output.parquet_dir_path` and is
independent of the input format — you can read CSV and write Parquet,
or vice versa. See [Output files](#output-files) above.
