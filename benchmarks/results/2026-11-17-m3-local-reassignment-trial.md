# Bounded local reassignment trial — Apple M3

**Status: completed experimental trial, uncommitted.** The implementation passed
correctness validation and review, but **did not solve determinism**. Local
reassignment increased accepted pair count and usually recall while reducing
strict-ID precision. Performance was mixed. Do not merge this as a determinism
fix or treat higher match counts as an accuracy improvement.

## Follow-up: full `top_n` shortlists

After the measurements below, Jude requested removal of the separate hardcoded
three-entry repair limit. The current working tree considers every candidate
in the existing `top_n` shortlist for both the incoming query and each eligible
incumbent, including batch hint caching and the live conflict precheck. No new
configuration parameter or extra candidate retrieval was added; protection,
one-incumbent atomic replacement and no-recursion rules are unchanged.

**The performance, quality and determinism results below describe the pinned
three-entry snapshot, not this follow-up.** Their hashes and raw evidence remain
unchanged. Performance and determinism have not been remeasured for the wider
shortlist implementation. The wider search can increase CPU and hint-cache use
when `top_n` exceeds three.

### Full-shortlist follow-up validation

- `cargo test --locked`: **508 library + 13 integration + 1 doctest passed**.
- `cargo test --locked --all-features`: **513 library + 13 integration + 1 doctest passed**.
- `cargo test --locked --no-default-features --lib local`: **30 focused tests passed**.
- `cargo clippy --locked --all-features -- -D warnings`, formatting and diff
  whitespace checks passed.
- Partner Reviewer completed the full follow-up review: **clean**, no findings.
- Planner regressions cover useful conflicts/alternatives at ranks four and
  five, respecting every supplied shortlist prefix, choosing a better repair
  beyond rank three, and continuing to reject recursive displacement.
- Batch regression varies `top_n` from one through five: the fourth-ranked free
  alternative permits repair only when actually shortlisted.
- Live regression covers 16 combinations: occupied incoming target at rank four
  or free incumbent alternative at rank four, both A/B directions, memory/SQLite,
  and `top_n` three/four. It verifies real scoring order, manual protection,
  final crossmap consistency and one atomic WAL reassignment.

Cargo reported incremental-cache cleanup warnings for missing old object files;
both complete test suites still passed, and strict Clippy passed without warnings.
Logs: `/tmp/melder-full-shortlist-{default,all-features,no-default,clippy}.log`.

## Scope

- Branch: `experiment/bounded-local-reassignment`.
- Parent commit: `b9a83a41d8c238569f915531420877a81ab2b0d7`.
- Machine: Apple M3, 8 physical/logical cores, 16 GiB RAM, macOS 26.5.1,
  AC power, low-power mode off.
- Baseline: [pre-change report](2026-11-17-m3-before-local-reassignment.md)
  and its adjacent JSON.
- No global optimization or full edge graph is maintained. Enroll is unchanged.
- No trial commit, merge or push was performed.

## What the measured three-entry snapshot does

Ordinary successful automatic claims keep their fast path. A query still lacking
an automatic match can be matched to an occupied candidate if that candidate's
ordinary automatic incumbent can move to a free qualifying alternative. This
changes one accepted pair into two, inspecting at most three incoming candidates
and three alternatives per incumbent. Among available augmentations, prefer the
highest sum of replacement scores, then deterministic ID order for exact ties.

Batch does one serial pass after its ordinary parallel pass and reuses bounded
cached scores from immutable records. Live re-scores stored records and only
repairs during an immediately available exclusive mutation window; contention
causes a skip, not a wait. Bulk live upserts may try pending queries after their
records are installed. Reassignment is symmetric for A/B live requests.

Manual/canonical/preloaded/restored pairs are protected. Newly scored automatic
pairs in this run/session can be moved. The CrossMap remains authoritative;
optional eligibility hints cannot establish a pair. Both new pairs must satisfy
threshold, score-gap, blocking and exclusion requirements. A memory/SQLite
compare-and-replace installs both pairs atomically. One new WAL frame records
the operation, with output/replay/compaction support and append-error rollback.

**What is not promised:** score-only swaps between already successful queries,
recursive reassignment chains, global optimality, a guaranteed accuracy increase,
or deterministic final assignments. Approximate candidate selection and the
ordinary parallel claim order remain. These limitations must be tested and
reported rather than inferred from match counts.

## Evaluation performed

1. Run focused planner, memory/SQLite atomicity, live symmetry/protection,
   stale-state, bulk-upsert, WAL/restart, export, and final batch metadata tests.
2. Run default/all-features tests, focused no-default tests, formatting and Clippy.
3. Run the existing 10k/100k usearch batch and live benchmarks using unchanged
   data/configs/scripts. After the preliminary pass, measure four final-source
   candidate cold runs and 24 interleaved warm baseline/candidate runs, with
   identical prepared cache bytes. Check live errors and final active outputs.
4. Compare strict-ID ground-truth precision/recall as a diagnostic, including
   within-run greedy-before/local-after pair differences. More accepted matches
   alone are not evidence of better identity matching.
5. Repeat the earlier duplicate-record experiment for **both** baseline and
   candidate: 20 runs with two Rayon threads, plus five single-thread controls.
   Inputs are S1/S2 and T1/T2, all `John Smith`, DOB `12-05-1995`; BM25 name
   weight 0.8 plus exact DOB weight 0.2; no blocking/common-ID/exact-prefilter
   shortcuts; auto threshold 0.8, review floor 0.5, `top_n: 2`.
6. Complete the bounded Partner Reviewer workflow and validate material fixes.

## Determinism — measured result

**The trial does not achieve deterministic matching.** The original two-identical-
records-per-side test was repeated on this M3 for both binaries, with a fresh
crossmap/output directory every run. Every run produced two bijective matches
with score 1.0, and there were **zero local reassignments** in this case.

| Binary | Rayon threads | Runs | T1→S1, T2→S2 | T1→S2, T2→S1 | Distinct outcomes |
|---|---:|---:|---:|---:|---:|
| Baseline | 2 | 20 | 8 | 12 | 2 |
| Trial | 2 | 20 | 11 | 9 | 2 |
| Baseline control | 1 | 5 | 3 | 2 | 2 |
| Trial control | 1 | 5 | 1 | 4 | 2 |

These frequencies are not a quality ranking: neither identical-record pairing
has stronger evidence. The repeatability requirement fails because more than
one outcome is observed for the same inputs. Both query records already obtain
an automatic match, so the bounded augmentation step does not run. Unordered
B-record iteration and parallel first-successful claims remain; deterministic
candidate ranking alone does not determine the complete assignment. The single-
threaded control also demonstrates that scheduling is not the only source.

- Full 50-run evidence, scores, commands, per-run pairings, binary hashes and
  artifact paths: [`2026-11-17-m3-local-reassignment-determinism.json`](2026-11-17-m3-local-reassignment-determinism.json).
- Baseline binary SHA256: `5aaac6df9ae2095be1f13093863b61987bcef511687cb61ea86b6f74d2abae81`.
- Measured trial binary SHA256: `c5707da8dd0049e190b2d2ce05d5700295c58454dc13d02ad5f316e07d89ef06`.
- Reproduction script: `/tmp/melder-local-determinism.EM6RiH/reproduce.py`.

## Preliminary performance and quality

The first 20-case performance pass used an earlier trial snapshot
(`caae1f3a2bd41a95720889abd61af935cd0e819907dbf6ea4afe27c3cd2b05a2`),
before final resolved-query review cleanup and score-gap/namespace hardening.
Those results are retained for provenance, **not the primary final-source
comparison**:

- [Preliminary per-run measurements](2026-11-17-m3-local-reassignment-preliminary-performance.json).
- [Preliminary comparison and strict-ID truth diagnostics](2026-11-17-m3-local-reassignment-preliminary-comparison.json).

That pass found more automatic batch links but slightly lower precision, and
nontrivial overhead in some cases. It recorded no live reassignment events in
the concurrent workload. These findings motivated an interleaved, same-session
final warm A/B measurement using identical prepared cache snapshots per pair,
rather than attributing all time-of-day differences to the code change.

## Correctness validation and review

All checks below passed on the final-source candidate before the final timing
pass. The release build uses `--features usearch --locked`.

| Check | Result |
|---|---|
| `cargo test --locked` | 505 library + 13 integration + 1 doctest passed |
| `cargo test --all-features` | 510 library + 13 integration + 1 doctest passed |
| `cargo test --no-default-features --lib local` | 27 focused tests passed |
| `cargo test --no-default-features --lib remove_query_reviews` | 3 tests passed |
| `cargo test --no-default-features --lib reassign` | 6 tests passed |
| `cargo clippy --all-features` | Passed |
| Formatting and diff whitespace checks | Passed |
| Partner Reviewer, completed full implementation diff | **Clean**, no findings |

Coverage includes memory/SQLite atomic compare-and-replace, failed/stale plans,
log-append rollback, symmetric live moves on both backends, manual/canonical/
preloaded protection, delayed auto-registration after manual confirmation,
record updates/removal, stale vector hashes, busy live gates, bulk upserts,
unchanged enroll multi-edge behavior, threshold/gap/exclusion bounds, no recursive
chains, WAL truncation/compaction/replay against a newer overlapping CSV snapshot,
and agreement between final batch statistics and exported active outcomes.
Unrelated pending reviews and overlapping A/B identifier namespaces are retained.

The review verdict concerns code correctness only. **It does not establish that
the experiment meets the business goals or is ready to merge.** The observed
lack of determinism remains a negative result, not a passing feature test.

## Final performance and quality

### Controlled comparison

The primary final-source comparison contains **28 sequential runs**: four
candidate cold runs and three paired warm repetitions for each of four workloads
(24 warm runs). Within each workload the pair order was baseline/trial,
trial/baseline, baseline/trial. Each warm run began in a fresh working directory
cloned from the same case-specific baseline-prepared cache; every cache file's
SHA256 was checked before execution. Cache preparation was outside timing.

Measurements ran from 11:48:28 to 11:59:36 UTC on 2026-11-17. No builds, tests or
other benchmark runs overlapped. The benchmark and determinism trial binaries
have the identical SHA256 recorded above. The parent independently checked all
90 source-file hashes against the measured snapshot, retained raw-log hashes,
run checks and paired wall-time calculations. An independent parent WAL fold
matched all 28 final bijective CSV crossmaps, and independent active-output
checks matched all 14 batch runs' automatic/review/unmatched statistics.

#### Warm end-to-end wall time

All ratios below are **trial / baseline**. Less than 1 means faster for time.
The paired ratio is the median of three per-repetition ratios, not the ratio
of the independently computed medians.

| Workload | Baseline median s | Trial median s | Median paired ratio | Ratio of medians |
|---|---:|---:|---:|---:|
| Batch 10k × 10k | 0.971 | 0.887 | 1.067× | 0.914× |
| Batch 100k × 100k | 13.730 | 15.401 | 1.079× | 1.122× |
| Live 10k/side, 3k requests | 5.408 | 4.501 | 1.010× | 0.832× |
| Live 100k/side, 10k requests | 18.964 | 18.037 | 0.869× | 0.951× |

The 100k batch case shows an overhead signal: about **8% higher paired wall
time**, or 12% comparing the separate medians. The smaller cases vary enough
that paired ratios and ratios of medians can point in opposite directions.
With only three pairs, these are descriptive measurements, not evidence of
statistical significance or a reliable live speedup.

#### Throughput, request latency and memory

Throughput ratios above 1 mean faster; latency ratios below 1 mean faster.
Batch throughput is meld's reported scoring throughput, not full startup/output
wall time. Peak RSS is macOS `time -l`'s per-process maximum, not summed process
memory.

| Workload | Throughput baseline → trial median | Paired throughput ratio | p95 ms baseline → trial median | Paired p95 ratio | RSS MiB baseline → trial median |
|---|---:|---:|---:|---:|---:|
| Batch 10k | 26,628 → 29,392 records/s | 0.965× | — | — | 659.2 → 624.0 |
| Batch 100k | 9,125 → 8,313 records/s | 0.970× | — | — | 2,195.5 → 2,329.4 |
| Live 10k | 1,490.3 → 1,533.1 requests/s | 0.885× | 21.9 → 21.9 | 1.244× | 682.9 → 681.6 |
| Live 100k | 1,020.6 → 1,023.7 requests/s | 1.030× | 27.4 → 27.5 | 0.967× | 1,666.6 → 1,660.6 |

The live 10k paired results include roughly **11.5% lower throughput and 24.4%
higher p95**, despite nearly unchanged separate medians. The 100k batch memory
increase is about **134 MiB / 6.1%**. These regressions should not be hidden by
selecting the more favorable statistic. Live 100k request throughput is close
on separate medians; its lower end-to-end wall time also includes startup and
cache-refresh variability, not just request processing.

#### Candidate cold runs

| Workload | End-to-end wall s | Peak RSS MiB |
|---|---:|---:|
| Batch 10k | 15.412 | 725.7 |
| Batch 100k | 160.775 | 2,188.8 |
| Live 10k | 21.580 | 759.1 |
| Live 100k | 204.616 | 1,726.4 |

Cold means a fresh vector/index cache, not a cold OS page cache or model download.
These are single candidate observations. The earlier baseline cold measurements
remain in the baseline report, but time-of-day differences prevent treating
that cross-session comparison as an isolated code effect.

### Assignment quality: count/score improvements are not precision improvements

The supplied strict-ID ground truth has 7,000 expected pairs at 10k and 70,000
at 100k. The following compares each trial run's own greedy assignments with its
final assignments, isolating the local step from cross-run greedy variation.
TP and FP mean true and false pairs under that ground truth.

| Workload / run | Local moves | TP before → after | FP before → after | Precision before → after | Recall before → after | Moves losing an old true pair |
|---|---:|---:|---:|---:|---:|---:|
| Batch 10k cold | 2 | 4,513 → 4,513 | 662 → 664 | 87.2077% → 87.1740% | 64.4714% → 64.4714% | 1 |
| Batch 10k warm 1 | 5 | 4,510 → 4,514 | 661 → 662 | 87.2172% → 87.2102% | 64.4286% → 64.4857% | 1 |
| Batch 10k warm 2 | 4 | 4,533 → 4,536 | 639 → 640 | 87.6450% → 87.6352% | 64.7571% → 64.8000% | 1 |
| Batch 10k warm 3 | 5 | 4,519 → 4,520 | 654 → 658 | 87.3574% → 87.2924% | 64.5571% → 64.5714% | 2 |
| Batch 100k cold | 191 | 46,056 → 46,104 | 11,910 → 12,053 | 79.4535% → 79.2751% | 65.7943% → 65.8629% | 62 |
| Batch 100k warm 1 | 196 | 46,022 → 46,083 | 11,930 → 12,065 | 79.4140% → 79.2512% | 65.7457% → 65.8329% | 57 |
| Batch 100k warm 2 | 205 | 46,100 → 46,160 | 11,844 → 11,989 | 79.5596% → 79.3823% | 65.8571% → 65.9429% | 67 |
| Batch 100k warm 3 | 185 | 45,967 → 46,018 | 11,993 → 12,127 | 79.3081% → 79.1435% | 65.6671% → 65.7400% | 59 |

Precision decreased in **all eight final-source candidate batch runs**. In the
100k warm runs, local repair added a net 51–61 true pairs but 134–145 false pairs;
precision fell about 0.16–0.18 percentage points. Each run had 57–67 moves that
lost an old true pair and had a negative net true-pair effect.

Pair count and total accepted score increased in every run. At 100k warm, score
sum increased by 167.335 / 174.971 / 157.159 respectively. Replacing one pair
with two qualifying pairs is therefore an improvement to the bounded planner's
objective, **not a guarantee of better entity identification**.

Cross-binary comparison is additionally affected by greedy nondeterminism. For
example, warm 100k repetition 3 ended with 46,018 true pairs in the trial versus
46,090 in its paired baseline, despite gaining 51 true pairs within the trial.
Full paired-baseline precision, recall and scores are retained in the comparison
JSON rather than conflating these two comparisons.

### Live coverage and validation

All **91,000 live requests** across these measurements had zero recorded
HTTP/body errors. There were **zero live local reassignment events**. Final
mapping counts agreed within each baseline/trial warm pair, but injected IDs
have no supplied ground truth. These runs measure overhead and observed safety,
**not live matching-quality improvement or reassignment-path performance**.
Focused tests, rather than these workloads, exercise actual live repairs.

The baseline-prepared warm live caches include previously injected vector IDs.
They are partial-refresh workloads: exactly **577/574 A/B records** were refreshed
at 10k and **1,970/1,972** at 100k in every warm run. They are not pristine
cache-only startups. Identical seed bytes preserve that behavior across binaries.

Across all 28 runs, every wrapper exited successfully and all recorded checks
passed. Batch active match + review + unmatched counts equaled the B dataset
size, with no skipped queries; across runs, 770,000 B records were processed.
Active automatic pairs agreed with the bijective crossmap and reconstructed
WAL state. Historical broken rows were retained but excluded from active totals.
Official datasets/configs/scripts/model files, prepared baseline caches and
measured source remained unchanged.

### Evidence and reproduction

- [Final complete per-run measurements, checks, hashes and move diagnostics](2026-11-17-m3-local-reassignment-final-performance.json).
- [Final paired comparisons and strict-ID quality summaries](2026-11-17-m3-local-reassignment-final-comparison.json).
- [Determinism evidence](2026-11-17-m3-local-reassignment-determinism.json).
- Raw timing/logs, CSV and WAL outputs: `/tmp/melder_final_xzn2CBhe/results/`.
- Final orchestration: `/tmp/melder_final_xzn2CBhe/run_final.py`; original
  official benchmark scripts/configs were used unchanged. JSON records commands,
  resource hashes and prepared-cache manifests. Temporary per-run working roots
  were removed only after outputs were retained; raw artifacts are not committed.
- Only safe matching/encoding-related environment overrides are retained in
  repository JSON, not the full process environment.
- Initial harness setup had a helper filename mismatch; it was corrected before
  executing any measured wrapper. No measured run was discarded or repeated.

## Conclusion

The bounded trial is implemented and tested, with atomicity/protection/output
safety preserved in validation. It demonstrates that a small local augmentation
can rescue extra automatic assignments without retaining a full graph.

It **does not satisfy deterministic matching**, and its count/score objective
trades lower precision for a small recall increase on these fixtures. Performance
is mixed, with measured batch 100k and live 10k regressions. The live benchmark
provides no evidence of a matching-quality gain because it never performed a
repair.

**Recommendation:** keep this branch experimental and unmerged. A separate
follow-up would need explicit repeatability semantics for already complete
assignments, an accuracy objective that does not equate extra accepted pairs
with better identity resolution, and workloads that actually trigger live
repairs. Do not expand this trial into a global optimizer or quietly claim
those remaining goals are solved.

