# Live assignment: existing warm benchmarks (2026-11-18)

Three sequential repetitions per workload and mode on Apple M3, 16 GiB RAM,
AC power. The original live wrappers and client ran unchanged in isolated
workspaces, with seed 42 and 10 concurrent clients. Every run started with an
empty CrossMap, fresh log and output, and a verified copy of the same prepared
index bytes for its workload. The only YAML change was `live.assignment.strategy:
deferred_acceptance` in enabled runs. Each mode ran first once per workload.

Main is the pinned `b9a83a4` release. Both new-branch modes used the same
post-review-fix release binary. All 18 runs completed; the client reported zero
HTTP/body errors for 117,000 requests. All final CrossMap CSVs matched the WAL
fold. The 10k/100k prepared caches refreshed 577/574 and 1,970/1,972 records on
side A/B at startup, respectively, in every mode. These were warm-cache runs,
not cold-cache or SQLite runs.

| Workload | Mode | Requests/s | Injection time | p50 | p95 | p99 | Server load time | Full wrapper time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10k × 10k, 3k requests | Main | 1,601 | 1.87 s | 2.7 ms | 20.7 ms | 36.7 ms | 1.3 s | 4.41 s |
| | New branch, off | 1,592 | 1.89 s | 2.6 ms | 20.2 ms | 31.7 ms | 1.3 s | 4.41 s |
| | New branch, on | 785 | 3.82 s | 11.8 ms | 23.6 ms | 33.8 ms | 1.4 s | 6.52 s |
| 100k × 100k, 10k requests | Main | 1,289 | 7.76 s | 4.2 ms | 22.3 ms | 31.4 ms | 5.4 s | 16.34 s |
| | New branch, off | 1,263 | 7.92 s | 4.6 ms | 22.5 ms | 33.5 ms | 5.4 s | 16.07 s |
| | New branch, on | 574 | 17.42 s | 16.1 ms | 30.3 ms | 49.3 ms | 5.5 s | 26.17 s |

Entries are medians of the three run statistics, not statistics over pooled
requests. Enabled versus main, the median paired request-throughput ratios were
0.476 (10k) and 0.424 (100k). For disabled versus main they were 1.030 and 1.010;
the sample is small, and the disabled comparisons do not show a clear change.
Enabled was slower in every repetition. Startup and memory measurements do not
show a similarly large change. Full wrapper times also include readiness polling,
client setup and shutdown.

Main and disabled returned identical *final pair sets* in all runs. Enabled made
one displacement per 3,000-request run and six per 10,000-request run, ending with
734 rather than 733 pairs at 10k and 3,272 rather than 3,270 pairs at 100k. These
are counts, not evidence of additional correct matches. The enabled path also
verifies reciprocal scores for ordinary proposals and serializes mutations;
these are likely sources of the throughput cost, but this benchmark does not
separate their contributions.

The raw logs, frozen identities and orchestration script are under
`/tmp/melder-live-before-after.AhM33r/`. The durable
[`official-warm-measurements.json`](official-warm-measurements.json) records run
statistics and hashes. Original benchmark inputs, scripts and caches were
unchanged.
