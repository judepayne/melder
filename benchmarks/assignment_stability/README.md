# Shuffled-B assignment stability — 10k × 10k

This is a separate batch benchmark for comparing `main` with the experimental
full-`top_n` local-reassignment implementation. B records query the fixed A pool.
It does not modify the original benchmarks, their data, or production Rust code.

## First-pass dataset

The runner copies these existing files byte-for-byte into this folder's ignored
`data/` directory and rejects conflicting existing copies:

- `benchmarks/data/dataset_a_10k.csv`: 10,000 reference entities.
- `benchmarks/data/dataset_b_10k.csv`: 10,000 query records: 7,000 labelled
  `matched`, 2,000 labelled `ambiguous`, and 1,000 labelled `unmatched`.
- `benchmarks/data/ground_truth_crossmap_10k.csv`: 7,000 strict-ID positive pairs.

Names, countries, identifiers, missing values and existing ambiguous examples
are retained. No synthetic records are added in this first pass. Ground-truth
metadata columns are not matching fields.

**Quality caveat:** the strict oracle does not describe every acceptable identity
assignment for the 2,000 ambiguous records. A pair absent from that oracle is a
strict-oracle false positive, not necessarily a proven real-world identity error.
The first-pass report also separates matched/ambiguous/unmatched cohorts.

## Comparison controls

- Both binaries receive exactly the same ten seeded B permutations. A stays fixed;
  B record contents and IDs stay fixed, and only CSV row order changes.
- Scoring matches the existing 10k usearch warm benchmark: country blocking,
  name embeddings 0.55, short-name partial-ratio 0.20, country exact 0.20,
  LEI exact 0.05; auto threshold 0.85, review floor 0.60, `top_n: 20`,
  `ann_candidates: 20`, four encoder threads.
- In the experiment, repair examines the full shortlist. Main uses its normal
  claim loop without reassignment. No parameter sweep is performed.
- Prepare one baseline cache, then give every measured run a fresh copy of those
  same bytes, an empty crossmap, and separate output/log paths. Preparation is
  excluded from performance summaries. Model weights use the existing cache.
- Run sequentially, alternating baseline/experiment order between pairs, with
  eight Rayon threads. Pin binary, input, permutation and cache hashes.
- Normalize final assignments by IDs, including unmatched B records. Compare
  complete mappings, per-record variation, pairwise differences and stable pairs;
  equal match counts alone are not evidence of repeatability.

**Interpretation:** this measures stability across shuffled inputs and fresh
processes, not isolated identical-input determinism. Melder loads B into unordered
maps, and parallel claim scheduling can also vary. CSV row order is not a promise
of processing order, and observed variation cannot be attributed to shuffling
alone. Frozen ANN cache bytes remove index rebuilding as another moving part.

## Running

Build and pin the current experimental binary before running; do not accidentally
reuse the earlier three-entry trial binary. Supply a binary built from the exact
`main` revision separately; there is no need to switch or reset the working tree.
The recorded first-pass binary/source identities are in `binary_provenance.json`.

```sh
cargo build --release --features usearch --locked
python3 -m unittest discover -s benchmarks/assignment_stability -p 'test_*.py'
python3 benchmarks/assignment_stability/run_test.py \
  --baseline-binary /path/to/pinned/main/meld \
  --experimental-binary /path/to/pinned/experimental/meld \
  --runs 10 --seed 20261117 --rayon-threads 8 \
  --output /tmp/melder-assignment-stability-new-run
```

Reproduce the independent WAL folding, cohort diagnostics and within-run repair
analysis from retained raw results with the permanent verification script:

```sh
python3 -B benchmarks/assignment_stability/verify_results.py \
  --run-root /tmp/melder-assignment-stability-new-run \
  --output /tmp/melder-assignment-stability-verified --prefix first-pass
```

The verifier checks the first-pass identities in `binary_provenance.json` by
default. For newly built binaries/source snapshots, supply their matching metadata
with `--provenance /path/to/binary_provenance.json`. The raw files must remain
available or be regenerated; compressed measurements retain the original numbers.

Use a new output directory; the runner refuses to overwrite a previous result.
Prefer a local temporary directory for raw measurements rather than a cloud-
synced directory. The runner retains raw logs, configs, shuffled inputs, active
outputs, crossmaps, WAL and structured comparisons. Only safe explicit environment
overrides are recorded; the full inherited process environment is never saved.

## Larger shuffled and fixed-order study

The [larger-study report](results/2026-11-17-study/report.md) records the completed
300-run comparison and confidence intervals. Its
[prespecified protocol](results/2026-11-17-study/protocol.json) defines
100 shuffled pairs plus 50 fixed-order pairs (300 measured subprocesses), with
identical binaries, settings and first-pass seed-cache bytes. The earlier ten-run
sample is not pooled into this study. Set `BASE`, `EXP`, `SEED_CACHE` and `WORK`
to the pinned binaries, prepared cache and a fresh local temporary root:

```sh
python3 -B benchmarks/assignment_stability/run_test.py \
  --baseline-binary "$BASE" --experimental-binary "$EXP" \
  --seed-cache "$SEED_CACHE" --rayon-threads 8 \
  --runs 100 --input-order shuffle --seed 20261117 --output "$WORK/shuffled"
python3 -B benchmarks/assignment_stability/run_test.py \
  --baseline-binary "$BASE" --experimental-binary "$EXP" \
  --seed-cache "$SEED_CACHE" --rayon-threads 8 \
  --runs 50 --input-order fixed --output "$WORK/fixed"
python3 -B benchmarks/assignment_stability/analyze_study.py \
  --shuffled "$WORK/shuffled" --fixed "$WORK/fixed" \
  --bootstrap 5000 --seed 20261118 --output "$WORK/inference"
```

Fixed mode uses the same original B file bytes **and path** in every process;
there are no fabricated permutations. `--seed-cache` never writes to its source.
Each completed process saves an immutable `run_result.json`; lightweight
`checkpoint.json` references all completed outcomes, including after failure.
Full comparisons are calculated once at successful campaign completion rather
than repeatedly during measurements.

The primary metric is the **mean fraction of B assignments differing between
two runs**, not the count of records ever seen to vary. Inference resamples whole
paired runs 5,000 times and reports 95% basic bootstrap intervals. Within-run
pairwise distances share observations and are not independent samples. Each mode
answers a separate question; intervals are conditional on this corpus, frozen
index and thread count, and do not establish general determinism or familywise
coverage. Positive baseline-minus-experiment differences mean better stability.
Quality checks separate the ambiguous cohort from known positives/negatives.

Generated local data and large artifacts are ignored. Read the
[first-pass report](results/2026-11-17-first-pass.md): the unmodified data already
produced ten distinct outcomes per binary. Compressed complete measurements and
independent validation are saved alongside it. Original benchmark files and
earlier trial evidence remain untouched.
