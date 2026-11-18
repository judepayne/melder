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

## Greedy assignment

By default (`batch.assignment.strategy: greedy`), batch matching uses parallel
greedy claims: each B query tries its scored
candidates in ranked order, atomically claiming a free A candidate at or above
the auto-match threshold. Occupied candidates are skipped, not displaced. There
is no post-pass reassignment or assignment solver in this path. Parallel
claim order and approximate candidate generation can still produce different
whole-run assignments between runs.

## Choosing pairs with the graph optimizer

The optimizer changes **which pairs are accepted**, not how Melder scores them.
For example, suppose B1 scores 0.99 with A1 and 0.98 with A2; B2 scores 0.97
with A1 and 0.85 with A2. Taking the strongest pair first gives 1.84 in total.
Choosing B1–A2 and B2–A1 gives 1.95. Each record still has at most one partner,
and records may remain unmatched. The objective is total similarity score,
not maximum pair count or verified real-world identity accuracy.

Enable it in YAML; this is not a new CLI subcommand:

```yaml
batch:
  assignment:
    strategy: graph
    mode: approximate   # optional; default for graph
```

Protected/preloaded links and common-ID/exact prematches stay fixed. Melder
considers only automatically eligible candidates on the existing shortlists,
using the shared scoring pipeline. It does not widen `top_n`, refill slots
occupied by protected endpoints, reconsider excluded pairs, or turn a list
demoted for ambiguity into an automatic match. Threshold and whole-list
score-gap eligibility checks happen **before** scores are rounded to nine
decimal places for the assignment objective. Selected output scores retain
the original scoring metadata.

### Approximate or exact?

**Start with approximate.** Melder divides candidates into independent groups
(connected components): records belong to the same group if you can follow a
chain of candidate pairs between them. No pair crosses a group boundary. Group
size counts **distinct participating records on both sides together**, not
edges or records in the whole batch.

- Through **512 records inclusive**, solve exactly.
- At **513 and above**, try two deterministic answers: take the highest-scoring
  available pairs first, and separately find a matching with the most pairs.
  Choose the answer with the higher total represented score for that group;
  on a tie keep score-first.

A million-record batch can therefore finish entirely exact. The cutoff is
fixed, not a tuning setting, and no score-loss tolerance is promised.

**Choose `mode: exact` if the highest possible total similarity score matters
more than runtime.** Every group is solved exactly, with no heuristic fallback.
The guarantee is over the eligible retained pairs and their nine-decimal
represented scores. It does not mean the shortlist contains every possible
match or that the selected pairs are the correct real-world identities.
Failure to finish is an error, not permission to publish a partial result as
optimal. Large connected groups can be expensive; see
[Performance](performance.md#batch-graph-optimizer).

**Repeatability:** the same Melder version, mode, retained pairs and represented
weights produce the same assignments regardless of worker scheduling or edge
order. Different retrieval results can change the graph; this is not a promise
of repeatable ANN retrieval. Hardware speed and elapsed time never determine
which method runs. With `--limit`, graph mode sorts B IDs before applying the
limit; greedy retains its existing input ordering.

Unlike [live deferred acceptance](live-mode.md#live-assignment-policy), this
is a batch-wide choice after scoring, not a sequence of reciprocal proposals
or incumbent displacements. Live and enroll behavior is unchanged.

### Reading the assignment summary

The successful completion summary reports the **effective** strategy/mode,
exact and approximate group counts, largest group size, and objective guarantee.
These illustrative numbers are not new measurements:

```text
Batch assignment: graph (approximate)
Groups: 51,695 exact · 0 approximate · largest: 76 records
Result: optimal over all retained eligible pairs.
```

```text
Batch assignment: graph (approximate)
Groups: 38 exact · 2 approximate · largest: 1,420 records
Total score: at least 98.7% of the best possible over retained eligible pairs.
```

Approximate mode can still be entirely exact, as the first example shows.
For a mixed run, the bound combines the exact objective of each exact group
with an upper bound for each fast group: the smaller of the sum of each A's
best score and the sum of each B's best score. Achieved total divided by the
combined upper bound certifies a lower bound on the fraction of optimal score.
Displayed lower bounds are rounded **down**, never up. A loose bound is not
measured score loss or identity accuracy. If the upper bound is zero, Melder
reports no positive eligible score instead of dividing by zero. The completion
summary is emitted only after successful publication.

A query that loses an eligible candidate to assignment competition may remain
in review with reason `assignment_conflict`, even above `auto_match`. That
review is not an accepted link. Scoring-log candidates are evidence of what was
scored, not final assignments; graph-selected matches use source
`graph_assignment` in the match log.

## Output files

Batch mode is event-sourced: during scoring, events are written to a
match log on disk. At the end of the run, the build pipeline reads the
match log (and the optional scoring log) to produce outputs. This means
a crashed greedy run leaves a recoverable match log, and outputs can be rebuilt
with `meld export` without re-scoring. The SQLite record-storage batch path
(`batch.db_path`) does not produce a match log or scoring log; its results are
built directly, so this log-based recovery does not apply to that path.

For graph runs that write a match log, `batch_assignment_started` and
`batch_assignment_completed` frame the assignment publication. **No partial
export:** if a started graph run lacks its completion marker, output building
and `meld export` reject it rather than treating a prefix as final or optimal.
To rebuild a completed graph run, set `live.match_log_path` in a copy of the job config to the **single timestamped graph log** printed by `meld run`, then run `meld export --config <copy> --out-dir <directory>`. Do not pass the untimestamped base if there are multiple runs, and do not use this log for live startup. Do not delete the start marker to force export. A solver/publication/logging
failure is an error: Melder does not save final CrossMap/CSV outputs or print a
successful assignment summary for that failed run. Protected links are not
removed to roll back new automatic selections. Hooks, if configured, remain
best-effort notifications, not a substitute for completed logs and final
CrossMap membership.

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
Eleven analytical views are included: `confirmed_matches`,
`scored_matches`, `asserted_matches`, `review_queue`, `near_misses`,
`runner_ups`, `unmatched_a`, `unmatched_b`, `broken_matches`, `summary`,
and `relationship_detail`. The authoritative schema is in
`src/output/db.rs`; view definitions are in `src/output/views.sql`.

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
for a 55M-record dataset). This estimate describes the existing greedy path.
Opting into graph assignment retains the candidate graph and materializes records
for the shared CSV, Parquet and SQLite output builder. Do not assume the same
bounded memory footprint for graph assignment.

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
