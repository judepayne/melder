# Local reassignment: productionization notes

Status: draft handoff notes, captured for Jude on 2026-11-17. This is not an
approved implementation plan or a claim that the experimental feature is ready
for production.

**Subsequent design decision:** Jude selected optional deferred acceptance for
live mode and optional global graph optimization for batch mode, initially off.
The original bounded-repair productionization proposal below is historical;
`.pi/designs/active.md` records the current discussion. Algorithm details, the
batch optimization objective and exact YAML setting names are not finalized.
Existing measurements describe bounded repair, not the newly selected algorithms.

## Current state and agreed direction

- Working branch: `experiment/bounded-local-reassignment`, based on main commit
  `b9a83a41d8c238569f915531420877a81ab2b0d7`.
- Trial changes remain uncommitted and unmerged. Do not commit, merge or push
  without Jude's instruction.
- Batch mode matches B queries against the A reference pool. Live is symmetric.
- Ordinary successful matching keeps its existing path. A query without an
  automatic match may take an occupied candidate if its eligible automatic
  incumbent can move to a qualifying free alternative. Replace one pair with
  two atomically; do not leave the displaced query queued for later rematching.
- Repair considers **all candidates in the existing `top_n` shortlists** of the
  incoming query and incumbent. The earlier hardcoded three-entry limit was
  removed. There is no separate repair-width parameter.
- `ann_candidates` controls retrieval breadth; `top_n` truncates the fully scored
  results. Already-matched candidates can consume shortlist slots. Raising
  `top_n` cannot recover candidates that were never retrieved.
- No recursive displacement, global optimizer, score-only swaps or canonical
  rearrangement of already-complete assignments is implemented. Enroll is unchanged.
- Shared scoring, atomic CrossMap replacement, eligibility/protection guards,
  reassignment WAL events, replay/export support and regression tests already
  exist. Production work should harden these, not create a second scoring path.

## Agreed configuration direction

Productionization should make the feature optional, **initially defaulting off**:

```yaml
local_reassignment:
  enabled: true
```

This switch is **not implemented yet**. It belongs in YAML as matching policy,
not under `performance` and not initially as a CLI option. This follows the
existing optional-feature style, such as `exact_prefilter.enabled`.

- One configuration source makes saved jobs reproducible and avoids override
  precedence rules.
- Use the same setting for batch/live; leave enroll unchanged.
- Keep the existing candidate limits; do not add a separate repair-width knob.
- Off must skip repair work and associated bookkeeping, not merely suppress the
  final replacement. Preserve ordinary matching semantics and avoid unnecessary
  hint caching, eligibility tracking or live locking when disabled.
- Log the effective setting at startup and expose it in validation/configuration
  diagnostics where appropriate.
- Turning off prevents future repairs; it does not undo saved assignments.
- Existing jobs must not silently change matching policy on upgrade.

## Production release requirements

### 1. Acceptance policy and accuracy

The current objective increases accepted-pair count by replacing one qualifying
pair with two. It is **not an identity-accuracy guarantee**. More accepted pairs,
more repeatable assignments and more correct identities are different outcomes.

Before production, decide:

- Whether the current cardinality-first policy is acceptable for the intended
  business use and false-positive/false-negative costs.
- Whether very confident existing automatic matches need additional protection.
- Whether existing thresholds and score-gap rules provide sufficient safeguards.
- Which evidence is required before enabling on representative datasets.

Do not infer accuracy from a higher score sum: adding a second accepted edge can
increase total score while harming an existing correct match. Do not add policy
knobs or choose new margins without evidence and an explicit design decision.

Manual, canonical/common-ID, preloaded and restored assignments must remain
protected. Threshold, score-gap, blocking and exclusion rules must not be bypassed.
The effect of startup/restoration on automatic-pair eligibility needs clear user
semantics; enabling the feature must not imply reoptimizing all historical links.

### 2. Observability and audit

Make repairs explainable through existing structured logs, summaries, hooks and
persistent audit events:

- Old pair and score, both replacement pairs and scores, candidate ranks, trigger
  and decision reason.
- Attempts, successful repairs and skip reasons: protected incumbent, no qualifying
  free alternative, stale records/vectors or live contention.
- Effective configuration and the distinction between candidate lists and accepted
  assignments. Live callers need visibility that another record's link changed.

Prefer existing logging/interfaces over introducing a monitoring stack solely
for this feature.

### 3. Persistence, failures and version compatibility

Atomic one-to-two replacement, append-error rollback and replay tests already
exist. Extend validation with crash/fault injection at replacement, WAL append,
flush, restart and output/compaction boundaries, on memory and SQLite backends.
Preserve each backend's source-of-truth and buffered-durability contracts.

Define upgrade/downgrade behaviour explicitly. Older binaries may not understand
`crossmap_reassign` events. Turning the feature off does not remove those events
or make an old binary able to read them. Decide format/version compatibility,
clear startup errors and any supported checkpoint/export/conversion or rollback
procedure before release.

### 4. Operational documentation and rollout

Document:

- What can move, what stays protected, when a repair is attempted and which
  existing candidate/threshold settings limit it.
- Why the feature does not guarantee deterministic complete assignments.
- Why switching off does not reverse saved changes.
- How restored assignments and eligibility behave across restarts.
- Live best-effort behaviour: repair skips a busy mutation window rather than
  waiting; this can affect repair coverage under load.
- Costs of larger shortlists and how to inspect whether the feature actually helps.

Start opt-in. Consider a batch-first rollout if live validation is not ready,
with explicit support/validation semantics rather than silently pretending live
repair is enabled. Default-on requires a later evidence-based decision.

## Further studies needed

### Highest priority: representative data with trustworthy truth

Use genuinely varied records and repairable conflicts, not just repeated
calibrated templates. Quantify correct matches, wrong known-positive assignments,
known-negative acceptance and unresolved records separately from ambiguous cases.

The ordinary-data study showed a small adverse accuracy signal and no clear gain
in correct pairs. Determine whether this is an acceptable trade-off, not just
whether matching counts or repeatability improve.

### Current implementation at 100k and larger

The available 100k timings describe the **older three-entry-capped trial**, not
the current full-`top_n` implementation. Future measurements should cover current
runtime, memory/hint-cache growth, crowded blocks and larger candidate settings.

Study worst-case candidate/alternative search and live rescoring costs as `top_n`
grows. Do not reintroduce an independent width cap without revisiting the agreed
configuration decision. These are proposed future studies, not authorization to
rerun benchmarks automatically.

### Live workloads that actually trigger repairs

Earlier live performance workloads triggered **zero repairs**. They establish
neither repair-path latency nor matching-quality benefits.

Build representative A- and B-side update workloads with actual repairs, and
measure tail latency, throughput, contention/skips and repair coverage. Exercise
record updates/removal, stale vector snapshots, bulk upserts, exclusions,
confirmation/protection, flush and restart under concurrency.

### Broader difficult cases

Cover missing/corrupted identifiers, different naming patterns, score gaps,
exclusions, bad blocking, larger duplicate clusters and conflicts requiring more
than one relocation. Distinguish unsupported chains from defects; supporting
recursive/global optimization is not automatically part of productionization.

### Actual candidate-retrieval failures

The custom five-A blocks guarantee exposure of all five candidates. They test
shortlist retention/repair, **not ANN recall in large crowded pools**.

Build cases where useful alternatives are outside a small retrieval budget.
Separate retrieval breadth from shortlist width: for example increase retrieval
while holding `top_n` fixed, then vary `top_n` with retrieval fixed. Verify actual
ranks and scores through the shared real scoring pipeline before interpreting
results.

## Evidence available for the next session

### Ordinary 10k dataset: larger stability study

Report: `benchmarks/assignment_stability/results/2026-11-17-study/report.md`.

- 100 shuffled runs per binary plus 50 identical-input repeats per binary:
  300 measured processes, same settings and frozen cache bytes.
- `ann_candidates: 20`, `top_n: 20` for both binaries.
- Main versus experiment mean changed B between runs: roughly **463 versus 456
  out of 10,000**.
- Relative variation reduction: **1.44%** for shuffles, 95% interval 1.11–1.77%;
  **1.50%** for identical-input repeats, interval 0.96–2.03%.
- Both still produced a different complete mapping on every run. This is small
  conditional stability improvement, not determinism.
- Roughly four repairs per 10,000-query run. No clear correct-pair gain; shuffled
  runs showed about 0.93 extra wrong known-positive assignments per run, an
  adverse secondary signal.
- Paired wall overhead about 0.95% / 1.22% for shuffled / fixed workloads.
- Whole-paired-run bootstrap, not independent resampling of pairwise distances.
  Ambiguous records are not automatically treated as proven identity errors.

### Constructed five-candidate competition

Report: `benchmarks/assignment_competition/results/report.md`.

- 100 five-A/five-B groups appended to original 10k background; 10,500 per side.
- Twenty groups per family: free escape rank 2, 3 or 5, fully ambiguous and
  one-to-two-unrepairable controls.
- Real ONNX/pipeline calibration verified qualifying edges, constrained queries
  and ranks; combined-data gate passed before measurements.
- `ann_candidates: 100`; `top_n: 1, 2, 3, 5`; twenty shared B permutations per
  combination, 160 measured processes.
- At `top_n: 5`, mean fixture-correct queries per 100:

  | Free escape rank | Main | Experiment |
  |---|---:|---:|
  | Second | 80.0 | 100 |
  | Third | 72.5 | 100 |
  | Fifth | 66.9 | 100 |

- Experimental repair-family pairings were exactly identical in all twenty
  shuffles once the required alternative was shortlisted. Main had twenty
  different mappings per cohort. This concerns actual pairs, not just counts.
- 311 actual fifth-ranked repairs; all custom repairs replaced one fixture-wrong
  pair with two fixture-correct pairs.
- Ambiguous controls remained variable and unrepairable controls had no legal
  augmentation; wider lists sometimes increased known-negative acceptance.
- Twenty matching-text clones per family, concatenated aliases and colliding
  synthetic identifiers: an assignment-mechanism diagnostic, not independent
  real-world accuracy evidence or ANN-retrieval evidence.
- Paired wall overhead roughly 0.5–1.2%; at width five about 1.2%.

### Earlier 100k batch timings: historical only

Report: `benchmarks/results/2026-11-17-m3-local-reassignment-trial.md`.

- Main median 13.73s; older capped trial median 15.40s.
- About 8% slower by median paired ratio, or 12% comparing separate medians.
- Do not present these as measurements of the current full-shortlist version.

### Validation and artifact caveats

- Current full-shortlist Rust source previously passed 513 library tests plus
  13 integration tests and one doctest with all features, and focused checks.
- Subsequent benchmark work changed Python/data/docs only; measured Rust source
  and binary identities were verified unchanged.
- Competition checks: 26 Python tests plus 28 shared tests; all 160 WALs
  independently folded to exact final CSV mappings. Final deep review clean.
- Larger stability study also retained independent WAL/inference verification
  and source/protocol snapshots; review acceptable with reproducibility limits
  explicitly documented.
- Durable compressed measurements and retained scripts support statistical
  reanalysis. Exact execution conditions/original WAL replay also need the
  original pinned binaries, frozen ANN cache, model weights and raw files;
  some are temporary/ignored artifacts. Hashes alone cannot recreate a graph.
- Preserve unrelated `.memory/`, `scripts/count_files_by_folder.py` and `target 2/`.

## Recommended next step

Take these notes through **Designer** to decide the production acceptance,
persistence/versioning, observability and rollout contracts. Then use **Planner**
for an implementation-ready plan. Keep the release opt-in until representative
quality, current large-scale performance and real live-repair behaviour have been
validated. Do not expand scope into a global optimizer implicitly.
