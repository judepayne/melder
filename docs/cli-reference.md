← [Back to Index](./) | [Configuration](configuration.md) | [API Reference](api-reference.md)

# CLI Reference

All commands accept `--log-format json` for structured log output.
Logs go to stderr; command output goes to stdout.

## `meld validate`

Parse and validate a config file without loading data or running
anything. Catches missing fields, invalid method names, bad threshold
values, and malformed blocking rules. Use this to check a config before
committing to a long batch run.

```bash
meld validate --config config.yaml
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |

## `meld run`

Run batch matching: load both datasets, score every B record against the
A-side pool, and write three output csvs (results, review, unmatched).
The cross-map is updated with auto-matched pairs so that re-running
skips already-resolved records.

```bash
meld run --config config.yaml
meld run --config config.yaml --dry-run
meld run --config config.yaml --limit 500 --verbose
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--dry-run` | | Validate config, load data, print what would be processed, then exit. No matching or output files. |
| `--limit` | | Process only the first N B records. Useful for quick sanity checks on large datasets. |
| `--verbose` | `-v` | Print job metadata, dataset paths, and threshold values at startup. |

## `meld serve`

Start the live-mode HTTP server. Datasets are loaded (into memory or
SQLite, depending on whether `live.db_path` is set), embedding and
blocking indices are built, and the write-ahead log is replayed for
crash recovery. Once ready, the server accepts requests on the
configured port. See [API Reference](api-reference.md) for endpoint
details.

```bash
meld serve --config config.yaml --port 8090
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--port` | `-p` | TCP port to listen on (default: 8080) |

## `meld enroll`

Start the enroll-mode HTTP server for single-pool entity resolution.
Records are enrolled into one growing pool and scored against everything
already there. Designed for graph-based ER workflows and deduplication.
Uses a simplified config format with `field:` instead of
`field_a:`/`field_b:`. See [Enroll Mode](enroll-mode.md) for full
details.

```bash
meld enroll --config enroll_config.yaml --port 8090
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to enroll-mode YAML config file (required) |
| `--port` | `-p` | TCP port to listen on (default: 8080) |

## `meld tune`

Run the full batch pipeline without writing any output files, then print
a diagnostic report: score distribution histogram, per-field statistics
(min/max/mean/median/stddev), threshold analysis showing how the current
thresholds split your records, and suggested threshold values based on
percentiles.

```bash
meld tune --config config.yaml
meld tune --config config.yaml --verbose
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--verbose` | `-v` | Show current threshold values at startup. |

See [Accuracy & Tuning](accuracy-and-tuning.md) for a detailed guide on
interpreting the tune output, a worked example with the benchmark
dataset, and the recommended weight-tuning workflow.

## `meld cache build`

Pre-build embedding index caches for one or both sides. Encodes all
records through the ONNX model and writes the resulting vectors to disk
so that subsequent `meld run` or `meld serve` invocations start
instantly instead of re-encoding. This is especially useful when the
same dataset is matched repeatedly with different configs or thresholds.

```bash
meld cache build --config config.yaml
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |

## `meld cache status`

Show the status of each cache file: whether it exists, its size on
disk, and the number of records it contains (for index files).

```bash
meld cache status --config config.yaml
```

## `meld cache clear`

Delete stale cache files. The default behaviour is *smart*: it computes
the cache filename that the current config expects (derived from a hash
of the embedding field names, order, and weights) and deletes only files
that do **not** match — i.e. files left over from a previous config that
are now unreachable. The current valid cache is left untouched.

Use `--all` to delete everything regardless.

```bash
# Smart clear: delete stale files only (safe to run before any rebuild)
meld cache clear --config config.yaml

# Full wipe: delete all cache files including the current valid ones
meld cache clear --config config.yaml --all
```

| Flag | Description |
|------|-------------|
| `--all` | Delete all cache files, including the current valid cache. Forces a cold rebuild on the next run. |

**When to use `--all`**: after changing the embedding model, or when
you want to reclaim disk space and are happy to re-encode from scratch.

**When the smart default is enough**: after changing field weights,
adding a new match field, or renaming fields. These all change the spec
hash, so the old cache files become unreachable automatically — the
smart clear finds and removes them without touching anything current.

## `meld review list`

Print the review queue as a formatted table. The review csv is produced
by `meld run` and contains borderline pairs that scored between
`review_floor` and `auto_match`. This command reads that file and
displays it with aligned columns for easy scanning.

```bash
meld review list --config config.yaml
```

## `meld review import`

Import human decisions on review pairs. The decisions file is a csv with
columns `a_id`, `b_id`, and `decision` (either `accept` or `reject`).
Accepted pairs are added to the cross-map. Both accepted and rejected
pairs are removed from the review csv, shrinking the queue.

```bash
meld review import --config config.yaml --file decisions.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--file` | `-f` | Path to decisions csv (required) |

## `meld crossmap stats`

Show cross-map statistics: total matched pairs, and coverage as a
percentage of both A and B datasets. Loads the datasets to compute
totals.

```bash
meld crossmap stats --config config.yaml
```

## `meld crossmap export`

Export the cross-map to a csv file at a specified path. Useful for
backing up the current state or transferring matches to another system.

```bash
meld crossmap export --config config.yaml --out matches.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--out` | `-o` | Output file path (required) |

## `meld crossmap import`

Import match pairs from a csv file into the cross-map. The csv must have
columns matching the configured `a_id_field` and `b_id_field`. Pairs are
merged with any existing cross-map entries — duplicates are ignored.

```bash
meld crossmap import --config config.yaml --file pairs.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--file` | `-f` | Input csv file path (required) |
