# Batch Mode Worked Example

Match 25 incoming counterparties (**B**, `incoming.csv`) against 10 reference
entities (**A**, `reference.csv`). Run the commands below from the project root
(the directory containing `Cargo.toml`). Batch is asymmetric: A is the indexed
reference pool; B is the query side.

## 1. Build and fix the teaching config

```bash
cargo build --release
./target/release/meld validate -c examples/batch/config.yaml
```

The supplied config deliberately has weights summing to **1.05**. In
`examples/batch/config.yaml`, change the last match field's weight:

```yaml
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05     # supplied value: 0.10
```

The four weights now sum to 1.00 (0.55 + 0.20 + 0.20 + 0.05). Validate and
check the inputs before matching:

```bash
./target/release/meld validate -c examples/batch/config.yaml
./target/release/meld run -c examples/batch/config.yaml --dry-run
```

Validation checks configuration, not match quality. The dry run also loads
the data, but does not match or write result files.

## 2. Run and inspect the results

```bash
./target/release/meld run -c examples/batch/config.yaml --verbose
ls examples/batch/output/
head -5 examples/batch/output/relationships.csv
head -5 examples/batch/output/unmatched.csv
head -5 examples/batch/output/crossmap.csv
```

The first embedding run may need network access to download the configured
`all-MiniLM-L6-v2` model. Later runs can reuse its downloaded files and vector
caches. Do not expect particular pair counts or scores from this tutorial:
they are not verified reference results.

| Output | How to read it |
|--------|----------------|
| `relationships.csv` | `relationship_type=match` means an accepted link; `review` means a proposal requiring a decision, **not** an accepted link. Includes configured ID columns (`counterparty_id`, `entity_id`), `score`, `reason`, `rank`, and A-side fields. |
| `unmatched.csv` | B records without a match or review outcome, with B fields and best-candidate evidence when available. It does not prove that no real-world match exists. |
| `crossmap.csv` | Accepted one-to-one pairs, using the configured `entity_id,counterparty_id` columns. Later runs reuse these links and skip resolved B records. |

There is no separate `results.csv` or `review.csv`. The supplied config does
not enable the optional scoring log, so it does not produce `candidates.csv`
or per-field score columns in `relationships.csv`. To collect additional
candidate evidence, add this top-level section before a fresh run:

```yaml
scoring_log:
  enabled: true
```

`candidates.csv` lists ranks 2..N from scoring. **Rank is not acceptance:** a
rank-two candidate may be selected, while a rank-one candidate may lose to a
competing query. Use `relationships.csv` and the final crossmap to identify
accepted links. See [batch outputs](../../docs/batch-mode.md#output-files) for
logs, optional Parquet/SQLite output and rebuild instructions.

## 3. Understand why graph assignment is the default

Several B records can compete for the same A entity. Melder scores candidates
first, then jointly chooses eligible one-to-one links with **graph assignment**
(default mode: `approximate`). It optimizes total represented similarity, not
pair count or known identity accuracy.

For a hypothetical eligible set:

| Pair | Score |
|------|-------|
| B1–A1 | 0.99 |
| B1–A2 | 0.98 |
| B2–A1 | 0.97 |
| B2–A2 | 0.85 |

Greedy can claim B1–A1 followed by B2–A2 (total **1.84**). Graph can instead
select B1–A2 and B2–A1 (total **1.95**). These are illustrative scores, not
results from the supplied CSVs. This joint handling of competition is why
graph is the default; see [measured evidence and solver details](../../docs/batch-mode.md#why-graph-assignment-is-the-default).

The example config omits `batch`, which already selects graph/approximate.
To compare explicit greedy on a **fresh crossmap**, add:

```yaml
batch:
  assignment:
    strategy: greedy
```

Greedy claims available partners locally without reconsidering occupied A
records. Do not specify `mode` with greedy. To return to the default, remove
this section. Neither strategy guarantees correct real-world identities.

### Thresholds are eligibility gates, not assignment decisions

The supplied thresholds are `auto_match: 0.85` and `review_floor: 0.60`.

- A candidate must meet automatic eligibility checks to enter graph assignment;
  exceeding 0.85 alone does **not** guarantee a link.
- Review-band scores are proposals, not accepted relationships. An otherwise
  eligible query can also remain in review with `reason=assignment_conflict`.
- A record without an accepted link or review outcome appears in `unmatched.csv`.

The pipeline maps differently named fields, uses country blocking, retrieves
embedding candidates, and scores all four configured fields. Graph assignment
does not change those scores or search beyond the retained candidates. See
[scoring](../../docs/scoring.md) for the full eligibility rules.

## 4. Try a clean threshold comparison

Preserved crossmap links would remain fixed rather than being reconsidered.
For a comparison, first keep any outputs you need, then change **both** paths
in your config to a new, unused directory:

```yaml
cross_map:
  backend: local
  path: examples/batch/output_trial/crossmap.csv
  a_id_field: entity_id
  b_id_field: counterparty_id

output:
  csv_dir_path: examples/batch/output_trial/
```

Lower `auto_match` to `0.75` and `review_floor` to `0.50`, then run again.
This changes eligibility and review coverage; it does not promise more correct
matches or fewer reviews. Compare accepted pairs and reasons, not just counts.
Restore the original paths and thresholds when finished. Do not delete a
production crossmap to tune thresholds. For labelled evaluation, see
[Accuracy & Tuning](../../docs/accuracy-and-tuning.md).

## 5. Inspect or clear vector caches

```bash
ls -lh examples/batch/cache/
```

Caches include vector indices and manifest/text-hash sidecars. Compatible
caches can avoid re-encoding unchanged text; changes to embedding fields,
weights or other index settings can invalidate them. Cache reuse is separate
from crossmap reuse: a warm cache does not imply records are reconsidered.

To remove this example's current and stale vector caches:

```bash
./target/release/meld cache clear -c examples/batch/config.yaml --all
```

Without `--all`, only stale caches are cleared. This does not reset accepted
links or remove outputs/logs. Keep logs you may need for audit or export;
automatic match-log cleanup is not implemented. See
[Vector Caching](../../docs/vector-caching.md) and the full
[configuration reference](../../docs/configuration.md).
