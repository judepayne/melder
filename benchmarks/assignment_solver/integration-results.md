# Phase 2B: warm-cache end-to-end CLI check

One fresh process per row on Apple M3/AC, four Rayon workers. The pinned optimized binary ran `meld run` against unchanged USearch warm 10k/100k fixtures with isolated caches and output directories, no starting CrossMap, and `top_n: 20`. Cache hashes were unchanged. Scoring log and CSV output were included. These are smoke-run timings, not a distribution or service-level target. Full pair-set hashes, run summaries and binary SHA-256 are in [integration-measurements.json](integration-measurements.json).

| Records per side | Strategy | Full wall | Solver only | Accepted pairs | In fixture truth | Not in fixture truth |
|---|---|---:|---:|---:|---:|---:|
| 10k | Greedy | 2.505 s | — | 5,170 | 4,504 | 666 |
| 10k | Graph (approximate) | 1.537 s | 1.897 ms | 5,178 | 4,628 | 550 |
| 10k | Graph (exact) | 1.926 s | 2.084 ms | 5,178 | 4,628 | 550 |
| 100k | Greedy | 106.019 s | — | 57,957 | 46,079 | 11,878 |
| 100k | Graph (approximate) | 30.308 s | 27.008 ms | 58,286 | 48,428 | 9,858 |
| 100k | Graph (exact) | 26.952 s | 31.414 ms | 58,286 | 48,428 | 9,858 |

All saved CrossMaps agreed with the accepted pairs in the generated relationships files. Graph modes gave identical actual pair sets: no component crossed 512 (largest: 8/76, retained edges: 5,808/86,476). The 100k graph-path wall reduction here is predominantly output replay (about 85 s after WAL read to CSV completion in greedy vs about 7 s in graph). It is not a solver speedup or evidence that approximate beats exact for this workload. Builds/checks during some runs and single sampling mean the elapsed ratios are indicative only. Truth comparisons use the synthetic fixture labels (7,000/70,000 known pairs), not a claim of general identity accuracy. Greedy may select different pairs in other runs; timing and truth results are for the particular runs shown.

## Release validation

After review remediation, the final release binary (`3f45fa50dcf8b71b4dc00690425751970641d2dc59f9fc09f9da192a34e0e6da`) reran the 10k and 100k approximate-mode jobs. Both reproduced the full selected pair sets above, and CSV matches again agreed with the saved CrossMap. The timing table is from the earlier integration binary, not this repeat.

Default tests passed (600 total), all-features tests passed (605), and flat-only tests passed (564), including nine real CLI cases per configuration. The CLI cases cover memory and SQLite storage, CSV/SQLite/feature-gated Parquet outputs, input and worker permutations, protected links, exclusions, whole-list score-gap gating, incomplete-log rejection and 512/513 dispatch. Both CrossMap backends passed publication failure tests, including failures writing and syncing the completion marker. Checked scoring tests cover a failed partial index scan and A-vector/ANN failures. Formatting, strict all-target Clippy, all six build targets, and closure review passed. Local final test/build logs and rerun artifacts: `/tmp/melder-b2-closure.9xRNu0`.

Reproduce from a private warm capture prepared by `capture.py --unseeded`: `python3 -B benchmarks/assignment_solver/run_integration.py --binary /path/to/release/meld --capture /path/to/capture --out /tmp/new-batch-graph-runs --repeats 1`. Prior [frozen-graph comparisons](approx-results.md) isolate the solver and measure the very different giant-component trade-off.
