← [Back to Index](./) | [Configuration](configuration.md) | [Live Mode](live-mode.md)

# Batch Mode

Use batch mode to match an incoming dataset **B** against a reference pool
**A** in one run. A and B have different roles; each record can have at most
one accepted partner. Start with the small
[worked example](../examples/batch/README.md), or run your own config:

```bash
meld validate --config config.yaml
meld run --config config.yaml --dry-run
meld run --config config.yaml
```

`validate` checks configuration; `--dry-run` also loads the data without
matching or writing result files. For an in-memory run configured with
`output.csv_dir_path`, inspect:

- **`relationships.csv`**: accepted links (`relationship_type=match`) and
  proposed reviews (`relationship_type=review`). Only matches are accepted.
- **`unmatched.csv`**: B records without a match or review outcome.
- **`candidates.csv`**: additional ranked candidates, only when the optional
  scoring log is enabled. A candidate's rank is not acceptance.
- **The configured crossmap CSV**: persisted accepted pairs, reused on later runs.

Thresholds determine automatic eligibility and review status; assignment then
chooses among eligible competing pairs. A score above `auto_match` does **not**
guarantee acceptance, and a selected pair need not be a query's rank-one candidate.

## Why graph assignment is the default

The [live startup pass](live-mode.md#initial-matching-pass) uses this optimizer by default too, but reads `live.startup.assignment`, not `batch.assignment`.

Both strategies use the same scoring rules. They differ in how they choose
one-to-one matches from the eligible candidates.

**Greedy claims matches as it goes.**

1. For each B record, try its eligible A candidates from highest to lowest score.
2. Claim the first A that's still free. If none is free, don't accept a match
   for that B record.
3. Keep each claim. A later B record can't displace an earlier match, even if
   swapping partners would improve the total score.

**Graph chooses competing matches together.**

1. Score the candidates for each B record and collect the pairs eligible for
   automatic matching. Leave already confirmed matches in place.
2. Group competing pairs. If two B records could both match the same A, put
   those choices in the same group. Follow any other shared candidates until
   the group is complete. Choices in one group don't affect another.
3. Look for the combination of one-to-one matches with the highest total score
   in each group. The default finds the best combination in small groups and
   uses a faster approximation for large groups. Some records may not get an
   accepted match.

For example, suppose all four pairs qualify. B1 scores **0.99** with A1 and
**0.98** with A2; B2 scores **0.97** with A1 and **0.85** with A2. If B1 claims
first, greedy picks B1–A1 and B2–A2 for **1.84** total. If B2 claims first,
greedy picks B2–A1 and B1–A2 for **1.95**. Graph picks the **1.95** combination
regardless of which B record is scored first.

**Greedy is order-dependent; graph assignment is deterministic.** With greedy,
changing which B record claims first can change the result. Even with the same
input order, parallel workers can claim in a different order. With graph, you
can shuffle the B records and get the same matches as long as the retained
candidates and scores are unchanged and you use the same Melder version and
assignment mode.

Graph is the default because it can achieve a higher **total match score across
A and B** without depending on claim order. A higher score doesn't guarantee more
matches or more correct identities. Graph takes extra work to collect and solve
competing pairs. In our [1M × 1M benchmark](performance.md#batch-graph-optimizer),
it was about **2% slower** end to end than greedy and found more correct pairs.
See [graph settings and limits](#choosing-pairs-with-the-graph-optimizer) below.

## Command flags

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required). |
| `--dry-run` | | Validate config, load data, print what would be processed, then exit without matching or result files. |
| `--limit` | | Process only N B records for a smoke test. Graph sorts B IDs before limiting; greedy uses its existing input ordering. Not a representative sample. |
| `--verbose` | `-v` | Print job metadata, dataset paths and thresholds at startup. |

## Output files

### Sinks and scoring evidence

For either input-storage backend and assignment strategy, enable any combination of `output.csv_dir_path`,
`output.parquet_dir_path`, and `output.db_path`; at least one sink is required.
They share the same output build, so additional sinks do not re-score records.
`scoring_log.enabled: true` adds candidate/field-score evidence; it is not
required for accepted links.

**SQLite input storage is different from SQLite output.** With `batch.db_path`,
both graph and explicit greedy honor all configured sinks, including Parquet-only
and SQLite-output-only runs, without writing extra CSV files in the working directory.
They have no match log or scoring log. Graph uses the shared builder for every
sink; greedy preserves its legacy CSV writer while using the shared builder for
Parquet and SQLite outputs. See [SQLite batch mode](#sqlite-batch-mode-large-datasets)
for memory and scoring restrictions.

### CSV outputs

The shared output builder writes:

| File | Contents |
|------|----------|
| `relationships.csv` | Confirmed matches and reviews, including assignment conflicts. Columns: configured B/A ID names, `score`, `relationship_type` (`match` or `review`), `reason`, `rank`, plus configured A-side fields. |
| `unmatched.csv` | B records with neither an accepted match nor a review outcome. Columns: configured B ID, B-side fields, `best_score`, `best_a_id` (when available). Direct SQLite batch output does not retain the best candidate ID. |
| `candidates.csv` | Only with a scoring log: ranks 2..N for scored queries, with configured B/A ID names, `rank`, `score`. These are ranking evidence, not final assignments. |

With `scoring_log.enabled: true`, `field_scores.csv` explains scored pairs without changing `relationships.csv`. Its columns are `a_id,b_id,field_a,field_b,method,score,weight`, matching the SQLite `field_scores` table. These are literal column names even when the input ID columns have different names. The score is the raw field score, not a weighted contribution. See [scoring](scoring.md) for how weights and synonym bonuses combine.

For example, to inspect the evidence for A123 and B456 in DuckDB:

```sql
SELECT field_a, field_b, method, score, weight
FROM read_csv('field_scores.csv', types = {'a_id': 'VARCHAR', 'b_id': 'VARCHAR'})
WHERE a_id = 'A123' AND b_id = 'B456';
```

Log-based batches include field scores for every logged shortlist candidate, including rank one. Direct SQLite batch has no scoring log: enabling the option exports the selected/review field scores it retains, just as in its output database. An enabled run with no detail writes a schema-only file. Use a fresh output directory: disabling the option does not delete files from earlier runs.

A scored candidate need not be a confirmed match. Check `relationships.csv` for the final decision. An asserted pair that bypassed scoring has no per-field evidence.

SQLite input with explicit greedy preserves its legacy CSV schema and result rows
(`a_id`, `b_id`, `classification` and field-score columns for relationships).
Its Parquet/SQLite sinks use the shared schema above and final CrossMap membership,
including protected pairs; they are not a literal conversion of the legacy CSV.

### Parquet outputs

`output.parquet_dir_path` enables `relationships.parquet`, `unmatched.parquet`
and, with batch scoring evidence enabled, `field_scores.parquet`. Log-based builds can also include `candidates.parquet`. The field-score file uses strings for identifiers, field names and method, and `Float64` for score and weight. For an Apache Iceberg workflow, ingest it into a compatible table using your catalog and table engine; `meld run` writes a Parquet file, not an Iceberg table or snapshot.

Contents match the shared CSV
outputs row-for-row. Record fields are `Utf8`; engine columns include `Float64`
scores and `UInt8` ranks. Writes use a temporary file followed by rename.

Requires the `parquet-format` build feature; otherwise the configuration is
rejected. See [Building](building.md).

### SQLite output database

`output.db_path` enables tables `a_records`, `b_records`, `relationships`,
`field_scores`, and `metadata`. In log-based builds, field-score evidence
requires the scoring log; direct SQLite batch builds retain selected/review
field scores without a scoring log.

Eleven analytical views are included: `confirmed_matches`, `scored_matches`,
`asserted_matches`, `review_queue`, `near_misses`, `runner_ups`, `unmatched_a`,
`unmatched_b`, `broken_matches`, `summary`, and `relationship_detail`.
See the authoritative [schema](../src/output/db.rs) and
[view definitions](../src/output/views.sql).

### Match log and rebuilding outputs

The in-memory path writes events to a match log, then builds outputs from that
log and the optional scoring log. A crashed greedy run leaves a log that can
be rebuilt with `meld export` without re-scoring; it is not necessarily a
completed batch. SQLite input storage produces neither log, so this recovery
path does not apply.

Graph logs use `batch_assignment_started` and `batch_assignment_completed`
markers around publication. **No partial export:** output building and
`meld export` reject a started graph run without its completion marker.

To rebuild a completed graph run:

1. Copy the job config and set `live.match_log_path` to the **single timestamped
   graph log** printed by `meld run`.
2. Keep that run's scoring log beside it if candidate/field-score evidence is
   needed. Without the companion, export rebuilds accepted links but does not
   borrow scores from another run.
3. Export into a fresh directory:

   ```bash
   meld export --config <config-copy.yaml> --out-dir <fresh-directory>
   ```

A missing, empty, damaged or ambiguous graph log is an error, not an empty
batch. Do not pass the untimestamped base when multiple runs exist, use a
batch graph log for live startup, or delete the start marker to force export.
Solver/publication/logging failure does not save final CrossMap/CSV outputs
or print a successful assignment summary. Rollback preserves protected links.
Hooks are best-effort notifications, not substitutes for completed logs or
final CrossMap membership.

Match logs are retained after output building. `output.cleanup_match_log` is
**not implemented**; do not use it to request deletion. Remove logs manually
only when no longer needed for audit or export. The configured crossmap CSV
persists accepted pairs so later runs skip resolved B records.

## Choosing pairs with the graph optimizer

Graph/approximate is the default with either in-memory or SQLite input storage.
Omitting `batch`, `assignment`, or `strategy` retains it. To spell it out
(there is no assignment CLI override):

```yaml
batch:
  assignment:
    strategy: graph
    mode: approximate
```

`batch.assignment.mode: exact` alone is also valid: omitted strategy means graph.
The optimizer changes accepted pairs, not scoring. It uses retained,
automatically eligible candidates from the shared pipeline; it does not widen
`top_n`, refill slots occupied by protected endpoints, reconsider excluded pairs,
or promote a list demoted for ambiguity. Threshold and whole-list score-gap
checks happen **before** rounding weights to nine decimal places for the
objective. Output scores retain their original metadata.

Protected/preloaded links and common-ID/exact prematches remain fixed. Existing
protection or exclusion CSVs must contain configured ID columns and nonempty
IDs. Protected links must be one-to-one; conflicting partners cause an error
on both storage paths.

### Approximate or exact?

**Start with approximate.** Candidates are split into independent connected
groups: records share a group if a chain of candidate pairs connects them.
Size counts distinct participating **A+B records**, not edges or whole-batch size.

- Through **512 records inclusive**, solve exactly.
- At **513 and above**, try deterministic score-first and maximum-pair-count
  matchings; choose the higher total represented score, keeping score-first
  on ties.

A large batch can therefore be entirely exact. This cutoff is fixed, not a
config setting, and no score-loss tolerance is promised.

Use **`mode: exact`** when maximizing represented similarity matters more than
runtime. Every group is solved exactly, with no heuristic fallback. This is a
guarantee over retained eligible pairs and nine-decimal weights, not every
possible pair or correct real-world identities. Large connected groups can be
expensive; failure is an error, not permission to publish a partial optimum.
See [Performance](performance.md#batch-graph-optimizer).

**Repeatability:** the same version, mode, retained pairs and represented weights
produce the same assignments regardless of scheduling or edge order. This does
not promise repeatable ANN retrieval or identical whole runs. Hardware speed and
elapsed time never select the method. Unlike
[live deferred acceptance](live-mode.md#live-assignment-policy), this is a
batch-wide choice, not reciprocal proposals or incumbent displacements.

### Reading the assignment summary

The successful completion summary includes effective strategy/mode, exact and
approximate group counts, largest group, retained edges, fixed pairs, solver
time and objective guarantee. These excerpts are **illustrative**, not measurements:

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

Approximate mode can still be entirely exact. For mixed runs, the bound combines
exact group objectives with an upper bound for each fast group: the smaller of
the sums of each A's best score and each B's best score. Achieved total divided
by the combined upper bound certifies a lower bound on the fraction of optimal
score. Displayed bounds round **down**; they are not measured score loss or
identity accuracy. A zero upper bound reports no positive eligible score.
The summary appears only after successful publication.

A query losing assignment competition may remain in review with reason
`assignment_conflict`, even above `auto_match`. Graph-selected matches use
source `graph_assignment` in the match log.

### Explicit greedy assignment

Set `batch.assignment.strategy: greedy` and omit `mode`; any explicit mode with
greedy is rejected. Each B query tries scored candidates in rank order,
atomically claiming a free eligible A. Occupied candidates are skipped, not
displaced. There is no optimizer graph or post-pass reassignment. Parallel
claim order and approximate retrieval can produce different whole-run results.

## SQLite batch mode (large datasets)

To reduce input-record storage in RAM, set `batch.db_path`. Test on a small
representative subset first; this is not bounded-memory execution or a
recommendation for datasets exceeding your machine's capacity.

```yaml
batch:
  db_path: batch.db
  sqlite_read_pool_size: 8          # default: num_cpus
  sqlite_pool_worker_cache_mb: 128  # default: 128 MB per read connection
  sqlite_cache_mb: 64               # default: 64 MB write cache
```

Choose an unused path for `batch.db_path`. Melder refuses an existing database or SQLite sidecar, even for a dry run; it does not clear old files for you. If a run was interrupted, choose a new path or inspect the leftovers first. The temporary database is deleted on successful completion.
Source datasets and configured crossmap/exclusion files remain; configured
output sinks are separate from this temporary database. Records use one column
per field, not JSON. Loading streams 10,000-record chunks; that limits the
loader's chunk size, not the rest of the run's memory.

### Memory footprint

Budget for SQLite page caches (`sqlite_cache_mb` plus
`sqlite_read_pool_size × sqlite_pool_worker_cache_mb`), BM25/blocking indices,
result collections and other allocations. Default graph assignment also keeps
retained candidates and connected components in memory and materializes records
for output assembly. **SQLite does not bound graph/output memory**; page-cache
settings are not a total RAM cap. Explicit greedy avoids the optimizer graph,
but does not make total memory constant.

Use SQLite when measured record-storage savings justify disk/lookup overhead
and you can budget separately for indices, graph and output assembly. Prefer
the in-memory path when data fits comfortably and you want log-based outputs.
SQLite input currently supports BM25, fuzzy and exact scoring, not embeddings.

## Encoder choices

The in-memory path supports local ONNX models (`embeddings.model`) and a
subprocess calling a central embedding service (`embeddings.remote_encoder_cmd`).
See [Remote Encoder](remote-encoder.md).

## Data Formats

Inputs can be CSV (default), JSONL/NDJSON or Parquet. Format is inferred from
extension or set explicitly:

```yaml
datasets:
  a:
    path: data.jsonl
    format: jsonl   # csv, jsonl, ndjson, parquet
```

Parquet input and output require:

```bash
cargo build --release --features parquet-format
```

Input string, integer, float and boolean columns become strings internally.
Snappy-compressed Parquet input is supported. Output format is independent of
input format; CSV input can produce Parquet output and vice versa.
