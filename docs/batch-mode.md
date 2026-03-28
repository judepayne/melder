← [Back to Index](./) | [Configuration](configuration.md) | [Live Mode](live-mode.md)

# Batch Mode

> For a hands-on walkthrough, see the
> [batch worked example](../examples/batch/README.md).

Batch mode processes an entire B dataset against a pre-indexed A pool in
a single pass. Run it with:

```bash
meld run --config config.yaml
```

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

When the job completes, the melder writes three output csvs (paths
configured in the `output` section):

| File | Contents |
|------|----------|
| `results.csv` | Confirmed matches — pairs that scored at or above `auto_match`. Columns: `a_id`, `b_id`, `score`, `classification`, plus a per-field score column for each match field. |
| `review.csv` | Borderline pairs — scored between `review_floor` and `auto_match`. Same columns as results. These need a human decision. |
| `unmatched.csv` | B records that had no match above `review_floor`. Contains all original B-side fields so you can inspect what failed. |

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
  results:   output/results.csv (712 rows)
  review:    output/review.csv (138 rows)
  unmatched: output/unmatched.csv (150 rows)
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
source CSV files and crossmap.csv remain the only persistent state.
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

Parquet support requires a feature flag at build time:

```bash
cargo build --release --features parquet-format
```

All column types (string, integer, float, boolean) are converted to
strings internally to provide a uniform interface across formats.
Snappy-compressed Parquet files are supported.
