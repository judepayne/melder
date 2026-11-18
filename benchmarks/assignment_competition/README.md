# Calibrated five-candidate assignment competition

This separate experiment tests whether wider scored shortlists unlock useful
local repairs. It compares the same pinned main and full-`top_n` experimental
binaries as `../assignment_stability/`; it changes no production Rust code or
original benchmark records. Read the [completed experiment report](results/report.md)
for all 160 runs and the verified fifth-ranked repairs.

## Design

Append 100 groups, each with five A reference records and five B queries, to the
existing 10k × 10k synthetic company background. Twenty groups exercise each
family:

| Family | Intended qualifying options | What it tests |
|---|---|---|
| `rank2` | One flexible B accepts all five A records; four constrained B records accept only their own A. The free escape is the flexible B's second-ranked candidate. | Repair unlocked at `top_n >= 2` |
| `rank3` | Same, with the free escape ranked third | Repair unlocked at `top_n >= 3` |
| `rank5` | Same, with the free escape ranked fifth | Repair unlocked at `top_n = 5` |
| `ambiguous` | All five A records are equally plausible for all five B records; every bijection is acceptable | More candidates alone can increase coverage without determining identity |
| `unrepairable` | Four constrained positives plus a similar but known-unmatched flexible B; the fifth A is unrelated and cannot qualify | No legal one-to-two repair; extra options must not bypass thresholds |

For repair families, the constrained records identify four entities and the
remaining entity is the flexible query's unique complete one-to-one solution.
We do not impose an arbitrary unique identity oracle on fully ambiguous groups.

The benchmark uses realistic-style company names/aliases and identifiers, but is
an intentionally constructed assignment diagnostic, not a representative estimate
of real-world business accuracy. The calibrated fixture uses concatenated company
aliases for flexible queries and deliberately duplicated/colliding LEI values;
those identifiers must not be interpreted as trustworthy unique legal-entity IDs.
Twenty exact matching-text clones per family scale the workload. They are not
100 independent name families or independent model-accuracy observations.

## Scoring and isolation

Use the original weighted scorers: name embedding 0.55, short-name partial ratio
0.20, country exact 0.20 and LEI exact 0.05; auto threshold 0.85 and review floor
0.60. Common-ID and exact-prefilter shortcuts are disabled.

An added `competition_group` blocking column has value `background` on every
original record and a private group ID on custom records. AND-blocking by country
and group preserves the original background country pools while preventing
cross-group interference. No original matching-field values are changed. This
is a controlled shortlist-width test: each custom block has five A records, so
ANN retrieval at 100 should expose all five. It is not a test of finding a hidden
neighbour in a large crowded ANN block.

**Calibration is mandatory.** Run the actual pinned binary, ONNX model and shared
scoring pipeline with uncompressed scoring logs. Check all custom query/candidate
scores, qualifying edges and declared escape ranks before measuring. The combined
10,500-record dataset is checked again during cache preparation. No invented
vectors, surrogate scorer, or assumed text-similarity geometry is accepted.

## Measurement plan

- Fix `ann_candidates: 100`; compare `top_n: 1, 2, 3, 5` on both binaries.
- Use 20 seeded B permutations shared across all eight combinations: 160 measured
  subprocesses, fresh crossmaps/outputs and identical prepared cache bytes.
- Rotate shortlist-setting order and alternate binary order to reduce drift.
  Execute sequentially with eight Rayon threads; preparation/calibration are
  outside performance measurements.
- Preserve complete normalized assignments, matching counts, known/allowed
  identity quality, local moves and replacement ranks, wall time and peak RSS.
- Report the three repair families, ambiguity control, unrepairable control and
  original background separately. Do not let background variation hide a clear
  result on the designed groups.
- CSV order is not guaranteed processing order: fresh-process HashMap iteration
  and concurrent claims can still vary. These are first-pass descriptive results,
  not general determinism or population-accuracy claims.

## Running

Run from the project root, using separately pinned main/experimental binaries:

```sh
python3 -B -m unittest discover -s benchmarks/assignment_competition -p 'test_*.py'
python3 -B benchmarks/assignment_competition/generate_data.py \
  --output benchmarks/assignment_competition/data
python3 -B benchmarks/assignment_competition/calibrate.py \
  --baseline "$BASE" --output /tmp/new-custom-probe
python3 -B benchmarks/assignment_competition/run_test.py \
  --baseline-binary "$BASE" --experimental-binary "$EXP" \
  --runs 20 --seed 20261117 --rayon-threads 8 --output /tmp/new-competition-run
```

Use fresh output directories. Dataset generation refuses to overwrite an existing
`data/`; reuse a matching generated dataset or supply `--data` to the runner for a
new one. The custom-only probe is retained as calibration evidence, while the
runner independently audits every custom query in the combined dataset before
starting measured runs. If the combined gate fails, it stops without measuring.
Measured runs disable scoring logs and verify all results against their WALs.

Reanalyze saved normalized assignments without rerunning meld:

```sh
python3 -B benchmarks/assignment_competition/analyze_results.py \
  /tmp/new-competition-run/measurements.json --output /tmp/new-competition-analysis
```

The full report separates valid ambiguous assignments from proven known-positive
and known-negative errors. It also records which replacement ranks were actually
used. Rebuilding an ANN cache may change background candidate retrieval; pin the
prepared cache and binaries to repeat identical execution conditions. Process
randomness still means exact assignment outcomes are not promised.

All generated data and raw artifacts remain separate from the original benchmark
folders. Fixture specifications, real-score calibration evidence and experiment
results are retained here. No commit or merge is part of this experiment.
