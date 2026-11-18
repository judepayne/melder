# Deterministic approximate assignment experiment

**Recommendation:** keep the exact solver for small independent groups; offer bounded-score approximation for large connected groups only as an explicit choice. Exact solved the entire captured 100k graph in 24 ms. By contrast, on the synthetic 100k giant it hit the 25-second process cutoff and hybrid returned in 130 ms of solver time. This is not a production batch-mode change.

## Matched comparisons

Median wall time **inside the solver call** (three fresh release processes, four workers, rotated mode order, differing shuffle seeds, Apple M3). All strategies pay for graph preparation; timing excludes loading, scoring, writing and verifying pairs. A fourth trial uses one worker and another shuffle for repeatability, not in the timing median. Exact is the updated exact component-parallel, batched solver on the same binary/graph, not the old single-threaded baseline.

| Graph (records per side) | Exact | Score-first | Count-first | Best per group | Hybrid | Hybrid score / optimum |
|---|---:|---:|---:|---:|---:|---:|
| Melder capture 10k | 2.76 ms | 2.21 ms | 2.23 ms | 2.19 ms | 2.47 ms | 100% (same pairs) |
| Melder capture 100k | 23.88 ms | 16.32 ms | 17.48 ms | 18.30 ms | 21.83 ms | 100% (same pairs) |
| Synthetic giant 10k, degree 5 | 11.404 s | 7.15 ms | 8.43 ms | 9.78 ms | 9.73 ms | 92.28% measured |
| Synthetic giant 100k, degree 5 | timeout | 93.38 ms | 113.05 ms | 129.95 ms | 129.66 ms | **at least** 83.27%; optimum unknown |

- Hybrid used the exact solver for all 5,087 / 51,695 nonempty groups in the Melder captures: largest groups had just 8 / 76 active endpoints. The small timing differences between hybrid and exact there are not evidence of an algorithmic speedup. Forcing Best on the 100k graph gave a 1.31× speedup (5.58 ms saved) and 99.985% of optimal score, but **518 exact pairs were replaced**, at the same total count of 58,286. This is similarity, not identity accuracy.
- Giant 10k: hybrid is about 1,172× faster with 8,850 pairs instead of 9,745. Of the exact pairs, 4,227 are missing and there are 3,332 new ones. The graph uses broad weights, not the captured 0.85 automatic threshold. Count-first returns 10,000 pairs but scores lower than score-first. Equal cardinality and high summed score are distinct objectives.
- Giant 100k: 500,000 edges; hybrid returns 88,215 pairs. The **whole process**, including input generation and output, took a median 0.152 s versus an exact process killed at 25 s: an observed lower bound of 164× on process wall time, not a measured exact solver speedup. Hybrid peak RSS was 97.8 MiB (10k giant: 11.1 MiB vs exact 18.5 MiB). Full per-method RSS, objectives, row/column upper bounds, counts and pair differences are in [the measurements](approx-measurements.json).

## What is actually guaranteed

Score-first sorts by descending integer weight then canonical B and A. Count-first uses iterative Hopcroft–Karp to maximize the number of pairs, with deterministic neighbor ordering. Best picks the higher summed weight **per group**, favoring score-first on a tie. Hybrid chooses exact at or below **256 active vertices (A plus B)**, Best above, fixed before timing. No clock or worker scheduling enters any assignment decision. A timed-out exact solve is a failure, never a partial assignment.

For approximate groups, sum each B's best possible weight and separately each A's; the smaller sum is an upper bound on any bijective matching. Add these bounds across groups, using the actual optimum for exact groups. Achieved score divided by this upper bound is a conservative certificate, not an estimate of accuracy. All completed comparisons obeyed `score <= exact <= bound` wherever exact was available. Every completed method produced the same ordered pair file across the tested input permutations and worker counts. Tests independently enumerate small graphs to check objective bounds, maximum cardinality and determinism, including 10,000 seeded graphs. These guarantees assume identical **retained graphs**, not repeatable ANN retrieval.

## Capture and reproduce

The ordinary warm benchmark crossmaps already protect so many endpoints that there are **zero** eligible free edges at either scale (see [protected capture](capture/README.md)). Those are not used for speed claims. For a useful fresh-batch comparison we kept both original datasets, warm cache bytes and original retrieval/scoring settings, but supplied an **empty private starting crossmap**. We verified pre-claim scoring logs against terminal query events; this is not a reconstruction from greedy winning pairs. At 10k/100k the fresh graphs contained 5,808/86,476 eligible edges. These are benchmark datasets, not independent evidence of production quality. Recapturing ANN candidates could give a different graph; use the supplied graph archives to reproduce the solver comparison.

```sh
# Or regenerate captures with capture.py --unseeded --output <new dir> --binary /path/to/meld
mkdir /tmp/approx-graphs
for s in 10k 100k; do gzip -dc benchmarks/assignment_solver/capture/unseeded/$s.graph.gz > /tmp/approx-graphs/$s.graph; done
python3 -B benchmarks/assignment_solver/compare_methods.py --capture /tmp/approx-graphs --output /tmp/new-compare-results
```

Raw trials (including actual pair files) locally: `/tmp/melder-approx-comparison`. Capture inputs, score logs and ID mappings: `/tmp/melder-approx-unseeded`. No changes to Melder's operational matching or dataset files. No auction implemented or needed for this comparison.
