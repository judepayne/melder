# First pass: shuffled-B assignment stability, 10k × 10k

**Result: the existing dataset is already useful. No records were added.**
Both main and the experiment produced ten distinct complete automatic-assignment
outcomes across ten B permutations. Variation affects hundreds of records,
including records with known positive ground truth, not just ambiguous examples.

## Dataset and controls

This uses separate byte-identical copies of the existing synthetic company
benchmark: 10,000 A reference records and 10,000 B query records. B contains
7,000 labelled matched records, 2,000 ambiguous records and 1,000 unmatched
records. The strict-ID oracle contains only the 7,000 matched positive pairs.
No matching configuration uses the truth metadata columns.

- Main commit: `b9a83a41d8c238569f915531420877a81ab2b0d7`.
- Experimental branch: `experiment/bounded-local-reassignment`, uncommitted
  **full-top_n** implementation, not the older three-entry trial binary.
- Main binary SHA256: `5aaac6df9ae2095be1f13093863b61987bcef511687cb61ea86b6f74d2abae81`.
- Experimental binary SHA256: `fc395c5100ed5fc4640c763ba4116047e89d66f7837bd34c3e43d118fb8271fe`.
- Machine: Apple M3, eight cores, 16 GiB RAM, AC power; eight Rayon threads.
- Same weighted scoring, country blocking, thresholds and candidate settings as
  the existing 10k usearch warm benchmark: `top_n: 20`, `ann_candidates: 20`.
- Ten seeded B permutations, seeds 20261117 through 20261126. A is fixed; all
  record contents and IDs are preserved. Both binaries receive identical
  permutation bytes within each pair.
- Twenty measured sequential runs; pair order alternates main/experiment and
  experiment/main. One baseline cache-preparation run is excluded from summaries.
- Every measured run starts with an empty crossmap, separate outputs and the
  same prepared vector/index cache bytes. Warm logs confirm both caches loaded
  with no encoding or index rebuilding. Only source-fingerprint manifest files
  changed after loading shuffled B; vector/index bytes did not change.
- Cache cloning, hashing and report analysis are outside subprocess timing.
  No concurrent builds/tests/reviews ran during measurements.

This is **shuffled-input assignment stability**, not isolated same-input
repeatability: unordered map iteration and parallel claim scheduling also vary
between processes, so observed changes cannot be attributed to CSV shuffling
alone. CSV row order is not guaranteed processing order. Assignment hashes sort
IDs and include unmatched B records; output-row ordering cannot create a false
assignment difference. Review-list or score-byte repeatability is not measured.

## Assignment variation

| Measure across ten runs | Main | Experiment |
|---|---:|---:|
| Distinct complete assignment outcomes | 10 | 10 |
| B records with more than one assignment, including unmatched | 923 | 909 |
| Always-identical confirmed pairs | 4,693 | 4,707 |
| Always without an automatic match | 4,384 | 4,384 |
| Pairwise changed B: minimum | 388 | 408 |
| Pairwise changed B: median | 465 | 451 |
| Pairwise changed B: maximum | 510 | 501 |
| Automatic match-count range | 5,170–5,174 | 5,176–5,178 |

The experiment has 14 fewer varying B records (about 1.5%) and 14 fewer changed
B records at the pairwise median (about 3%). These are descriptive differences
from one ten-run sample, **not evidence that reassignment causes a reliable
repeatability improvement**. Both versions remain clearly nondeterministic.
The 45 pairwise comparisons per binary are not independent replications.

### Which records vary?

| B cohort | Records | Varying on main | Varying on experiment |
|---|---:|---:|---:|
| Known matched positives | 7,000 | 467 | 458 |
| Ambiguous | 2,000 | 431 | 426 |
| Known unmatched | 1,000 | 25 | 25 |

## Repair coverage and quality

The experiment performed **39 repairs**, with per-run counts
`5, 4, 6, 4, 4, 5, 4, 0, 3, 4`. Main performed none. The workload therefore
exercises actual reassignment, unlike the earlier live benchmark.

All chosen replacement ranks were within the first three: 37 moves used ranks
(1, 2), one used (2, 2), and one used (2, 3). The implementation searched the full
shortlists, but this corpus **does not demonstrate useful fourth/fifth-ranked
escapes or stress their search cost**. Those remain a separate targeted-fixture
follow-up; existing unit regressions cover those paths.

| Median diagnostic | Main | Experiment |
|---|---:|---:|
| Automatic matches | 5,172 | 5,176 |
| Strict-oracle true pairs | 4,509 | 4,512.5 |
| Strict-oracle precision | 87.1978% | 87.1728% |
| Strict-oracle recall | 64.4143% | 64.4643% |
| Incorrect A assignment among known-positive B records | 11 | 11.5 |
| Ambiguous B records automatically matched | 620 | 626.5 |
| Known-unmatched B records automatically matched | 26.5 | 27 |

**Oracle caveat:** pairs involving ambiguous B records are absent from the
positive oracle and count as non-truth pairs in strict metrics. They are not all
proven identity errors. The known-unmatched and known-positive cohorts above
provide separate, less ambiguous diagnostics.

Within the experiment's own greedy-before/repair-after runs, the 39 repairs
produced a net **29 additional oracle true pairs and 10 additional non-oracle
pairs**. Eight moves displaced an old oracle true pair. One run's four repairs
reduced its true-pair count by two. Strict precision improved in four runs,
decreased in five and was unchanged in the run with no repairs. These within-run
changes are distinct from comparing independently varying main/experimental runs.
Extra accepted pairs are still not an unconditional accuracy improvement.

## Performance: descriptive first-pass measurements

| Measure | Main | Experiment |
|---|---:|---:|
| Warm end-to-end wall median | 0.801 s | 0.796 s |
| Wall minimum–maximum | 0.764–0.857 s | 0.766–1.415 s |
| Peak process RSS median | 658.0 MiB | 616.3 MiB |

The median paired experiment/main wall ratio is **1.010×** (about 1% slower);
the ratio of independent medians is about **0.995×**. The paired ratio range is
0.931–1.765×. The first experimental run was the 1.415-second outlier; it was
not excluded or repeated. Warm means vector/index caches, not cold OS/binary
page caches. These short runs do not establish a significant speed difference.
Peak RSS is macOS `time -l`'s per-process maximum, not aggregate process memory.

### All measured pairs

| Shuffle | Main wall s | Experiment wall s | Main matches | Experiment matches | Main true pairs | Experiment true pairs | Repairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.802 | 1.415 | 5,170 | 5,176 | 4,509 | 4,519 | 5 |
| 2 | 0.857 | 0.798 | 5,171 | 5,176 | 4,537 | 4,533 | 4 |
| 3 | 0.803 | 0.864 | 5,171 | 5,176 | 4,492 | 4,509 | 6 |
| 4 | 0.800 | 0.795 | 5,172 | 5,178 | 4,523 | 4,501 | 4 |
| 5 | 0.768 | 0.779 | 5,172 | 5,177 | 4,517 | 4,516 | 4 |
| 6 | 0.811 | 0.828 | 5,174 | 5,176 | 4,499 | 4,508 | 5 |
| 7 | 0.780 | 0.784 | 5,173 | 5,176 | 4,530 | 4,508 | 4 |
| 8 | 0.789 | 0.792 | 5,174 | 5,176 | 4,507 | 4,521 | 0 |
| 9 | 0.803 | 0.843 | 5,170 | 5,176 | 4,506 | 4,502 | 3 |
| 10 | 0.764 | 0.766 | 5,172 | 5,176 | 4,509 | 4,525 | 4 |

## Validation and evidence

All twenty measured subprocesses exited successfully. Every active match/review/
unmatched category reconciled with the batch summary and complete B universe;
no B records were skipped. Crossmaps were bijective. The parent independently
folded every measured WAL and obtained the exact final CSV crossmap in all
20 runs. Original data and measured Rust source hashes stayed unchanged, as did
both pinned binaries. All ten permutations were distinct and preserved records;
every pair used identical B bytes and identical prepared cache bytes.

Partner Review was **acceptable**, with no material findings. Its one Low
reproducibility finding was addressed by retaining the independent analysis
script in the benchmark folder with documented arguments.

Nine focused stdlib Python unit tests passed. The release build passed. No Rust
source was changed for this benchmark, so the earlier complete Cargo validation
of the full-shortlist implementation remains applicable.

- [Complete measurements, normalized per-run assignments, pairwise comparisons,
  cohort diagnostics and verification](2026-11-17-first-pass-measurements.json.gz)
  (gzip JSON, about 1.1 MB compressed).
- [Independent validation and cohort/move details](2026-11-17-first-pass-validation.json).
- [Binary/source/input provenance](../binary_provenance.json).
- [Runner and reproduction instructions](../README.md).
- Raw results: `/var/folders/g6/zfm6_wzj20g4dq7x42bv4_l40000gn/T/melder-assignment-stability-t393kh8g/results/`.
- [Permanent independent analysis script](../verify_results.py), with CLI
  arguments documented in the README. Its SHA256 is retained in the validation
  JSON; it was rerun successfully against all twenty measured outputs.

**Conclusion:** keep this unmodified 10k pair as a realistic first-pass stability
benchmark. It already produces substantial assignment variation and triggers
repairs. No dataset augmentation is needed to establish that. A later targeted
set would be needed to benchmark late-ranked escape cases specifically.
No commit or merge was performed.
