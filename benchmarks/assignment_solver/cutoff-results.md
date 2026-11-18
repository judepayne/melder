# Where is an approximate switch worth it?

Single connected balanced components, measured independently of the hybrid switch. Compare exact (batched) against Best (better of score-first and count-first). Both see identical eligible edges; neither strategy uses a clock to choose a solution. Size is total active endpoints, A+B. Solver wall time only, excluding graph generation, process startup, pair validation and I/O. Apple M3 on AC; a single worker since there is only one component.

Eight sizes × two degrees (5 and 20 edges per B) × five seeds × three score profiles = 240 graphs. Three paired executions per graph with shuffled inputs and alternating treatment order: 1,440 successful solves. Each pair set's input membership, bijection, recomputed integer objective and same-method repeatability were checked. Score loss is `100 * (exact - best) / exact`, not loss in match count or identity accuracy.

For the competing **0.85–1.00** scores, below are medians per group size over both degrees and all seeds; speedups are medians of paired ratios (not ratios of the time columns). Worst loss is the largest observed across ten graphs, not a guaranteed bound.

| A+B | Exact ms | Fast ms | Speedup | Score lost | Worst loss |
|---:|---:|---:|---:|---:|---:|
|64|0.151|0.049|3.3×|0.99%|1.48%|
|128|0.500|0.089|6.1×|1.07%|1.50%|
|256|1.785|0.172|12.1×|1.08%|1.58%|
|512|7.225|0.346|24.2×|1.26%|1.84%|
|768|16.196|0.543|33.9×|1.20%|1.53%|
|1,024|28.539|0.715|45.9×|1.10%|1.50%|
|2,048|122.266|1.584|92.9×|1.20%|1.49%|
|4,096|534.277|3.335|190.2×|1.21%|1.52%|

**What changes the result?** On the same topologies, planting a superior diagonal (0.97–1.00 vs 0.85–0.95 alternatives) yielded zero loss at every size. With broad random scores over (0,1], the median loss at 256/512/1,024 was 6.89%/7.40%/7.25%. This is a stress test, not an eligible graph for a 0.85 threshold. Density matters for time: at 512 vertices exact took 5.34 ms with degree 5 and 9.18 ms with degree 20; at 1,024 it took 21.32 ms and 37.26 ms. The threshold should express how much exact time is worth spending, not an assumption that larger groups have lower quality loss. Hardware and the number of such components per batch affect total wall time. Prior ordinary 100k capture had no component larger than 76 vertices; none of the proposed 256+ limits changes that outcome.

**Suggested discussion point (not applied):** keep exact through 512 endpoints, switch above that in approximate mode. This buys exactness through the region where exact took about 5–9 ms per group on this host. If a further 20–37 ms per group is acceptable to preserve all score through 1,024, choose 1,024 instead. There was no mathematical crossover and no universal optimum.

[Raw timings, integer objectives, graph and pair hashes](cutoff-measurements.json.gz). To reproduce: build the standalone `compare` binary in release mode and run `python3 -B benchmarks/assignment_solver/sweep_cutoff.py --binary /path/to/compare --out /tmp/new-cutoff-study`. No solver or configuration change.
