# B1: parallel components and equal-gain batches

Both speedups work on the workloads they target. **Neither resolves the
100k random connected cases.** No production code, defaults or candidate graphs
were changed. The sequential prototype remains the reference.

| Records per side / graph | Sequential SSP | Four workers | Batched, one worker | Both |
| --- | ---: | ---: | ---: | ---: |
| 10k, independent groups, degree 5 | 0.651 s | 0.195 s | 0.664 s | 0.197 s |
| 10k, independent groups, degree 20 | 1.192 s | 0.366 s | 1.179 s | 0.362 s |
| 100k, independent groups, degree 5 | 6.748 s | 1.903 s | 6.878 s | 1.932 s |
| 100k, independent groups, degree 20 | 12.132 s | 3.611 s | 12.247 s | 3.723 s |
| 10k, one connected graph, degree 5 | 11.045 s | 11.191 s | 11.355 s | 11.162 s |
| 10k, one connected graph, degree 20 | 19.043 s | 18.940 s | 19.057 s | 19.453 s |
| 100k, one connected graph, degree 5 or 20 | >25 s | >25 s | >25 s | >25 s |
| 10k, equal-gain chain | 5.026 s | 5.027 s | 0.0037 s | 0.0037 s |
| 100k, equal-gain chain | >25 s | >25 s | 0.0455 s | 0.0451 s |

Times are elapsed **solver** times on completed cases. `>25 s` denotes a
process cutoff including graph generation, not measured solver completion time.
One fresh process per case, one seed, Apple M3 / 16 GiB / AC, same pinned release
binary and unchanged graph generator. Four controls were rotated by fixture. The
first campaign's total budget cut one case short and skipped six; those seven
cases then got the full 25 seconds using the same binary. The initial attempts
are retained in [raw measurements](optimization-measurements.json). The final
matrix has 50 completed cases and 10 process timeouts. Full local artifacts:
`/tmp/melder-b1-parallel-matrix` and `/tmp/melder-b1-parallel-followup`. See the
[README](README.md) for a reproduction command with a larger campaign budget.

- **Parallel components** gave 3.36–3.55× at 100k with disconnected groups.
  Peak process RSS rose from 58.5 to 62.5 MiB (degree 5) and 197.5 to 208.4 MiB
  (degree 20). A single group cannot use additional component workers.
- **Batching** turned 10,001 searches into two in the 10k chain. At 100k it added
  99,999 pairs after the first search; the second search terminates. On the
  completed broad-score graphs it found **zero** additional paths to batch. These
  synthetic weights are not evidence for how often ties occur in production.

## Determinism and exactness

Both single-threaded algorithms passed the existing independent exhaustive and
10,000-case seeded oracle suites against the same nine-decimal objective. Tests
shuffle inputs, reassign payload handles and vary worker counts (1, 2, 8 on
larger fixtures). Each algorithm returned the same pairs across its own test
variants. In the benchmark all completed treatments for a graph agreed on
objective, and changing workers within a policy left the ordered payload digest
unchanged. Batching may select a **different** tied optimum from the reference;
it does not introduce scheduling-dependent tie decisions. The post-review equal-gain regression
asserts that batching actually adds multiple pairs per search. Debug/release
Rust suites (16 tests each), four Python checks, formatting and strict Clippy passed.

## Decision

Keep parallel execution as a useful prototype option. Batching can be very
valuable when many gains are exactly equal, but is not a general fix for a large
component with varied scores. Measure group sizes and score structure on actual
retained candidate graphs before choosing the production solver. The connected
100k case and G6 approval are still open; batch integration has not begun.
