# B1 batch solver validation and scaling

**G4 and G5 passed the prototype tests. G6 is blocked.** This implementation
solves the tested graphs exactly, but all three connected 100k-per-side cases
exceeded the process limit. Do not promote it to the production batch strategy
without an explicit performance/design decision.

## Correctness

Debug and release suites passed. The independent enumerator allows each B to
stay unmatched or choose one unused A and maximizes integer weight. Coverage
includes all 19,683 ternary 3×3 graphs (plus smaller and other alphabets),
10,000 seeded rectangular graphs with dimensions 0–7, and 24 medium disconnected
cases checked component by component. The harness verifies input membership,
bijection, weight total, and repeatability of actual pairs with shuffled input
and changed payload handles. Worked cases include the cardinality trap, zero
gain and a path that reverses an earlier choice. Analytic chains, stars, dense
and sparse connected graphs exercise graph structure separately from the oracle.
Debug builds check feasible reduced costs and reverse links at every step;
release additionally checks them at extraction. Precision, overflow, malformed
inputs, limits and deliberately corrupted residual links have focused tests.

## Scaling

One fresh release process per case, Apple M3 (8 logical cores, 16 GiB, AC power,
macOS 26.5.1), Rust 1.94.0. `N` is **per side**. Weights and graph generation use
fixed recipe `sparse-v1` and seed `0x5eedb1`. The skewed workload uses connected
groups from 1 to 1,024 records per side; giant and chain each have one connected
component. Degree is requested per B before clamping to group size. The chain
uses diagonal .75 and successor .5, so its optimum is the full diagonal. This
is independent of Melder scoring and does not estimate production accuracy.

| N | Shape | Degree | Edges | Solver seconds | Peak MiB | Result |
|---:|---|---:|---:|---:|---:|---|
| 1,000 | skewed | 5 | 4,992 | 0.062 | 3.3 | complete |
| 1,000 | skewed | 20 | 19,853 | 0.069 | 6.2 | complete |
| 1,000 | giant | 5 | 5,000 | 0.077 | 3.5 | complete |
| 1,000 | giant | 20 | 20,000 | 0.135 | 7.3 | complete |
| 1,000 | chain | — | 1,999 | 0.037 | 2.7 | complete |
| 10,000 | skewed | 5 | 49,936 | 0.610 | 8.6 | complete |
| 10,000 | skewed | 20 | 198,824 | 1.096 | 24.9 | complete |
| 10,000 | giant | 5 | 50,000 | 10.632 | 18.1 | complete |
| 10,000 | giant | 20 | 200,000 | 18.248 | 56.1 | complete |
| 10,000 | chain | — | 19,999 | 4.850 | 11.3 | complete |
| 100,000 | skewed | 5 | 499,408 | 6.342 | 55.5 | complete |
| 100,000 | skewed | 20 | 1,989,038 | 11.486 | 194.3 | complete |
| 100,000 | giant | 5 | 500,000 | — | 165.6 | killed at 25 s |
| 100,000 | giant | 20 | 2,000,000 | — | 540.5 | killed at 25 s |
| 100,000 | chain | — | 199,999 | — | 87.4 | killed at 25 s |

Peak RSS includes the benchmark's graph generator and validations. For timed-out
runs it is a lower bound on completed-run peak RSS. Timeouts include generation
and never produce a successful matching result. The 25-second per-process
limit is a measurement bound, **not** an approved performance target. The
completed 10k giant runs spend over 99% of solver time on augmentations, not on
construction or output. Small disconnected groups scale much better than one
giant component despite comparable total edge counts.

The [raw measurements](measurements.json) carry compiler, binary and source
hashes, graph checksums, wall and CPU times, phase timings, and statuses.
Full local artifacts, with the exact release executable and a source snapshot,
are in `/tmp/melder-b1-assignment-final`. See [README](README.md) for commands.
These are not matched graphs for the previous library assessment; no direct
speed comparison is claimed.

## Recommendation

Keep this implementation as a correctness reference. Before integration,
measure component sizes on representative retained candidate graphs and, if
large connected components are in scope, prototype a more scalable exact method
against the same oracle and workloads. Do not silently return greedy or partial
results when large components time out.
