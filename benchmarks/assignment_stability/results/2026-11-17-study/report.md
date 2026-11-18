# Larger assignment-stability study: 100 shuffles and 50 fixed-order repeats

**Conclusion:** the experiment has a small, statistically supported stability
advantage on this fixed 10k corpus/cache/thread setup. It changes about **6.7–6.9
fewer B assignments per 10,000** between runs, a **1.4–1.5% relative reduction**
in variation. This is not a determinism fix, and there is no clear gain in
correct matches. A small adverse quality signal remains.

## Primary results

The primary metric was declared before execution: the mean number/fraction of B
automatic assignments differing between two runs of the same binary, including
changes between an A ID and no automatic assignment (`None`). It does not measure
review-list or score-byte repeatability. Positive main-minus-experiment differences
mean better experimental stability, not more correct matches.

| Campaign | Runs per binary | Main mean changed B | Experiment mean changed B | Relative reduction | 95% basic interval for relative reduction |
|---|---:|---:|---:|---:|---:|
| Shuffled B | 100 | 462.38 | 455.71 | **1.44%** | **1.11% to 1.77%** |
| Identical B bytes/path | 50 | 462.81 | 455.86 | **1.50%** | **0.96% to 2.03%** |

The shuffled absolute reduction is 6.67 changed B records, with interval
5.22–8.26, or **0.0667 percentage points** of all B records, interval
0.0522–0.0826 points. The fixed-order reduction is 6.95 records, interval
4.61–9.52, or **0.0695 percentage points**, interval 0.0461–0.0952 points.
Both separate intervals exclude zero. The estimate is smaller than the roughly
3% reduction suggested by the earlier ten-run pairwise-median comparison; this
study uses the prespecified mean rather than selecting a favorable statistic.

### Still not deterministic

| Descriptive measure | Shuffled main / experiment | Fixed main / experiment |
|---|---:|---:|
| Distinct complete assignment outcomes | 100 / 100 | 50 / 50 |
| B records ever varying | 924 / 910 | 924 / 910 |
| Always-identical confirmed pairs | 4,692 / 4,706 | 4,692 / 4,706 |
| Always without an automatic match | 4,384 / 4,384 | 4,384 / 4,384 |

Every fixed-order repeat produced a different complete assignment in both
versions. Unordered-map iteration and scheduling can vary even without CSV
reordering. The experiment is **slightly more stable, not deterministic**.
Ever-variable counts are descriptive only; their dependence on sample size is
why they were not selected as the primary endpoint.

## Method and controls

- **300 measured fresh processes:** 100 shuffled pairs, then 50 fixed-order pairs.
  Both campaigns completed without discarding or repeating any measured run.
  Earlier first-pass outcomes were not pooled into this sample.
- Same unchanged synthetic company corpus: 10,000 A reference records and 10,000
  B queries. Known positives: 7,000; ambiguous: 2,000; known unmatched: 1,000.
- Same scoring configuration, country blocking, thresholds, `top_n: 20`,
  `ann_candidates: 20`, eight Rayon threads, Apple M3/eight cores/16 GiB on AC.
- Same pinned main and full-`top_n` experimental binaries as the first pass:
  main `b9a83a41d8c238569f915531420877a81ab2b0d7`; binary SHA256
  `5aaac6df9ae2095be1f13093863b61987bcef511687cb61ea86b6f74d2abae81`.
  Experiment is the uncommitted full-shortlist tree, binary SHA256
  `fc395c5100ed5fc4640c763ba4116047e89d66f7837bd34c3e43d118fb8271fe`.
- Every process cloned the **same first-pass prepared cache bytes**, used a fresh
  crossmap and separate outputs. No cold preparation or encoding was needed.
  Initial vector/index bytes were identical across both campaigns and binaries.
- The shuffled campaign used 100 distinct seeded permutations, shared exactly
  within each pair. Fixed mode used the same original copied B bytes **and path**
  for every process. A stayed fixed. Hashes and record contents were checked.
- Sequential execution only; alternating main/experiment and experiment/main pair
  order. No concurrent builds, tests, reviews or inference jobs during measurement.
  Cloning, integrity checks and analysis were outside subprocess timing.
- Python benchmark/statistical source and the protocol were frozen and archived
  before the first measurement. Production Rust source stayed unchanged.

### Inference

For each binary, compute the mean Hamming distance over all unordered distinct
run pairs: 4,950 pairs at n=100 or 1,225 at n=50, normalized by 10,000 B records.
Those distance pairs share runs and **are not independent observations**.

Use 5,000 bootstrap replicates, resampling **whole paired run indices** with
replacement. Baseline and experimental indices are sampled together, preserving
input/time-pair covariance. Duplicate sampled indices have zero self-distance;
the denominator remains the number of pairs of sampled positions, `n*(n-1)/2`.
The two-sided 95% basic interval is `[2*estimate - q97.5, 2*estimate - q2.5]`,
with linearly interpolated quantiles. Relative intervals resample the ratio too,
not just divide an absolute interval by the observed baseline. Seeds were
20261118 and 20261119. There were no undefined relative replicates.

The two modes answer separate prespecified questions; they are not pooled and
these are per-mode intervals, not a joint/familywise 95% coverage claim. Inference
assumes fresh process pairs are approximately independent/exchangeable within
mode. It is conditional on this corpus, frozen ANN graph, binaries and thread
count—not proof of improvement on other datasets, graph builds or thread counts.

## Match-quality checks

Quality effects below are **experiment minus main means per 10k run**. Their
intervals are secondary, unadjusted uncertainty summaries, not confirmatory
multiplicity-adjusted claims. The oracle excludes ambiguous B records; do not
interpret every non-oracle assignment as a proven identity error.

| Quality endpoint | Shuffled effect [95% interval] | Fixed-order effect [95% interval] |
|---|---:|---:|
| Correct strict-oracle pairs | -0.02 [-2.57, +2.64] | +0.48 [-3.62, +4.30] |
| Known-positive B assigned the wrong A | **+0.93 [+0.30, +1.57]** | +0.90 [-0.02, +1.84] |
| Known-unmatched B automatically matched | +0.08 [-0.53, +0.69] | -0.96 [-1.94, +0.02] |
| Ambiguous B automatically matched | +2.76 [+0.13, +5.18] | +3.70 [0.00, +7.62] |
| Total automatic matches | +3.75 [+3.47, +4.03] | +4.12 [+3.62, +4.62] |
| Strict precision, percentage points | -0.0636 [-0.1135, -0.0119] | -0.0602 [-0.1386, +0.0145] |

There is **no clear overall increase in correct pairs**. The shuffled sample has
about one extra incorrect known-positive assignment per run, an adverse secondary
signal. Most extra accepted pairs involve the ambiguous cohort. The modest
stability benefit is therefore not a demonstrated accuracy improvement.

### Within-run repair effects are a different comparison

The experiment performed **603 repairs**: 401 shuffled and 202 fixed. Against
its own pre-repair greedy state, those moves added net 255/115 oracle true pairs
and 146/87 non-oracle pairs across the two campaigns. They displaced 96/56 old
oracle true pairs. These are aggregated move effects across repeated runs, not
counts of unique entities. Five experimental runs had a negative net true-pair
change from repair (four shuffled, one fixed).

Strict precision increased in 30/100 shuffled runs and 7/50 fixed runs; it fell
in the remaining runs. Within-run gains do not imply gains over a separate main
run, because the ordinary greedy assignments also differ between processes.

All chosen replacement ranks remained at most three. The implementation searched
full shortlists, but this corpus still does not stress useful fourth/fifth-ranked
escape paths; dedicated unit tests cover those, and a targeted benchmark would
be needed to measure their benefit/cost.

## Performance: descriptive

| Measure | Shuffled main / experiment | Fixed main / experiment |
|---|---:|---:|
| End-to-end wall median | 0.7000 / 0.7053 s | 0.6823 / 0.6903 s |
| Median paired experiment/main wall ratio | **1.0095×** | **1.0122×** |
| Peak process RSS median | 659.21 / 616.38 MiB | 659.09 / 616.63 MiB |

Observed paired wall overhead was about **0.95% / 1.22%**. These are short warm
runs, with all outliers retained; no formal performance inference is claimed.
Warm refers to vectors/indexes, not OS/binary page caches. Shuffled B may require
source-fingerprint/text-hash checking even with no encoding, so cross-mode wall
differences are not isolated effects of CSV ordering. RSS is macOS `time -l`'s
per-process maximum, not aggregate process memory.

## Validation and statistical reanalysis

Partner Review was **acceptable**, with no material correctness findings. Its
one Low reproducibility finding is addressed by distinguishing the durable
statistical evidence from temporary execution artifacts below.

- All 300 measured processes exited successfully; no skipped B records. Active
  match/review/unmatched outputs reconciled with the full B universe and summary.
- Every crossmap was bijective. The retained independent verifier folded all
  **300 WALs to the exact final CSV mappings**.
- Original data, measured Rust source and both pinned binaries stayed unchanged.
  B identities, within-pair bytes, fixed path/bytes and initial cache manifests
  were verified. Only source-fingerprint manifest files changed after warm loads;
  vector/index bytes did not change.
- **28 Python unit tests passed** before execution. No Rust source changed in
  this packet, so no new Cargo test run was needed.
- Independent real-data inference checks used per-record assignment frequencies,
  not pairwise matrices, to reproduce both point estimates. They reproduced
  40 weighted bootstrap values and all four primary effect/relative confidence
  intervals with `statistics.quantiles` rather than the analyzer's quantile helper.
- All runs, bootstrap replicates, cohort diagnostics and adverse outcomes are
  retained. No optional stopping, result-based exclusions or metric switching.

### What remains reproducible after temporary cleanup?

**Statistical reanalysis is durable.** The two compressed measurement archives
contain all normalized run assignments and provenance; the inference JSON also
retains every bootstrap replicate. Decompress each archive into a fresh directory
as `measurements.json`, then run the retained `analyze_study.py` with those shuffled
and fixed directories, 5,000 replicates and seed 20261118. The original tracked
10k metadata/oracle and retained source snapshot allow the reported statistics
to be recalculated without the temporary cache, binaries or raw logs. This
archive-only reanalysis was exercised successfully: both complete mode results
and all bootstrap replicates matched the original inference exactly.

**Rerunning the same execution conditions is not a fully durable bundle.** The
frozen ANN/vector cache, pinned binaries, raw logs and WALs currently remain at
the temporary paths in `execution_identity.json`. They are not included in the
durable archives. Their hashes identify them but cannot reconstruct them after
cleanup. Preserve those artifacts separately to repeat the same frozen-graph
experiment or independently refold its original WALs. Rebuilding the cache may
produce a different ANN graph; even with the same artifacts, process/scheduling
randomness means exact assignment outcomes are not promised.

### Files

- [Prespecified protocol](protocol.json).
- [Execution identities and cache/source hashes](execution_identity.json).
- [Exact premeasurement benchmark/statistical source snapshot](source_snapshot.tar.gz).
- [Inference JSON with all bootstrap replicates](analysis.json) and [generated summary](inference.md).
- [Shuffled complete measurements](shuffled-measurements.json.gz) and [WAL/cohort validation](shuffled-validation.json).
- [Fixed complete measurements](fixed-measurements.json.gz) and [WAL/cohort validation](fixed-validation.json).
- [Independent inference-check script](validate_inference.py) and [results](independent_checks.json).
- [Reusable benchmark instructions](../../README.md).
- Raw run directories and launcher logs:
  `/var/folders/g6/zfm6_wzj20g4dq7x42bv4_l40000gn/T/melder-assignment-study-rb6kcyl_/`.

**Recommendation:** regard this as evidence of a small conditional stability
benefit, not deterministic matching or a clear accuracy win. Keep the experiment
unmerged pending an explicit decision about the quality trade-off and broader
workloads. No commit or merge was performed.
