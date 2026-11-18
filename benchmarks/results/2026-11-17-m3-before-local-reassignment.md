# Baseline before local match reassignment — Apple M3

**Status: complete.** All 20 benchmark wrapper runs finished successfully; measurements and validation are recorded below.

## Purpose and source

Record existing batch/live usearch benchmark performance before any local
reassignment change. No matching algorithm, benchmark configuration, or benchmark
script is changed for this baseline. No implementation branch is created.

- Date: 2026-11-17 (UTC).
- Branch: `main`.
- Commit: `b9a83a41d8c238569f915531420877a81ab2b0d7`
  (`Optimize unblocked batch BM25 candidate searches`).
- Tracked working tree was clean before measurements. Existing untracked
  `.memory/`, `scripts/count_files_by_folder.py`, and `target 2/` were left alone.
- Build: `cargo build --release --features usearch --locked` (successful).
- Executable: native arm64 release build, pinned to a separate copy for the runs.

## Machine and environment

| Item | Value |
|---|---|
| CPU | Apple M3 |
| Hardware model | Mac15,13 |
| Physical / logical cores | 8 / 8 |
| RAM | 16 GiB (17,179,869,184 bytes) |
| Operating system | macOS 26.5.1, build 25F80 |
| Power | AC power; low-power mode disabled |
| Rust / Cargo | 1.94.0 / 1.94.0 |
| Python | 3.13.3 |
| Thread / inference overrides | No `RAYON_*`, `ORT_*`, `TOKENIZERS_*`, `OMP_*`, `MKL_*`, or `VECLIB_*` overrides were set |

This is a different machine from the earlier Apple M1 Ultra BM25-only timing
record. Those timings are not used as this baseline.

## Workloads and procedure

Use the existing `run_test.py` and `config.yaml` pairs under:

- `benchmarks/batch/10kx10k_usearch/{cold,warm}/`
- `benchmarks/batch/100kx100k_usearch/{cold,warm}/`
- `benchmarks/live/10kx10k_inject3k_usearch/{cold,warm}/`
- `benchmarks/live/100kx100k_inject10k_usearch/{cold,warm}/`

Run sequentially in an isolated temporary working directory preserving the
original relative layout. Original benchmark outputs, WALs, crossmaps and caches
in the checkout are not cleared. The four existing CSV inputs are reused without
regeneration. Model files are reused where available.

For each mode/size: one cold run; one warm-directory cache preparation run; three
measured warm repetitions. Cold and warm wrappers use separate cache directories,
so the preparation run must not be labelled a measured warm run. Each wrapper
clears its own output and crossmap; live wrappers also clear their WAL.

Invocation from the isolated working directory:

```sh
python3 benchmarks/<mode>/<case>/<cold-or-warm>/run_test.py \
  --binary /absolute/path/to/pinned/meld
```

### Shared scoring configuration

- Model: `all-MiniLM-L6-v2`.
- Vector backend: `usearch`, index mode `load`.
- AND blocking: A `country_code` against B `domicile`.
- Weights: embedding 0.55; fuzzy partial-ratio 0.20; exact country 0.20;
  exact LEI 0.05.
- Thresholds: auto-match 0.85; review floor 0.60.
- Batch: `top_n: 20`, `ann_candidates: 20`, encoder pool 4.
- Live: `top_n: 5`, `ann_candidates: 5`, encoder pool 4 (10k) or 6 (100k).
- Live: `skip_initial_match: true`; this measures incremental matching, not
  an initial assignment of every preloaded record.

### Live workload

10k A + 10k B starts with 20,000 records and injects **3,000 total requests**.
100k A + 100k B starts with 200,000 records and injects **10,000 total requests**.
Both use 10 HTTP workers, seed 42, and the helper's default 80% encoding mix.

The helper's actual allocation is about 20% each new A, new B, embedding-update A
and embedding-update B, and 10% each field-only update A/B. This differs from its
introductory comment. Exact printed operation counts will be recorded below.

## Metrics and limitations

- Batch CLI scoring/elapsed times are rounded to 0.1s. Throughput is the CLI's
  scoring-only records/second, not whole-process throughput.
- External wall time includes startup, data/index preparation, output generation,
  and shutdown. Live throughput excludes those phases.
- Live latency includes HTTP request/response and response parsing, but excludes
  time waiting in the client's work queue.
- Live throughput counts attempted requests, including failures. Error columns
  must therefore be checked separately; exit status zero alone is insufficient.
- Cold means cleared benchmark vector cache, not a cleared OS page cache or a
  forced model download.
- These are performance benchmarks, not ground-truth accuracy tests. Assignment
  results can vary with candidate generation and concurrent claims.
- Warm status must be checked against cache load/encoding messages, not inferred
  from a nonempty cache directory.

## Results

Completed **20 sequential runs**, 2026-11-17T10:17:07.110839+00:00 through 2026-11-17T10:31:27.795746+00:00 UTC. All wrappers exited zero; all recorded validation checks passed.

### Summary

Warm values below are **independent medians of three repetitions**. They do not necessarily describe one particular run; latency percentiles are medians of per-run percentiles, not a pooled request distribution.

| Workload | Cold wall s | Warm wall s | Warm records/s (batch) or requests/s (live) | Warm p95 ms | Warm max RSS MiB |
| --- | --- | --- | --- | --- | --- |
| batch 10k | 14.251 | 0.786 | 37,478.0 | — | 659.0 |
| batch 100k | 144.215 | 13.101 | 9,859.0 | — | 2,186.9 |
| live 10k | 20.730 | 4.416 | 1,805.6 | 17.8 | 682.4 |
| live 100k | 171.320 | 15.837 | 1,389.7 | 19.3 | 1,658.4 |

Max RSS is the macOS `/usr/bin/time -l` value for the wrapper and waited-for descendants; it is **not simultaneous aggregate memory**. Cold has one measured run, not a median.

### Batch: all timings and resource measurements

| Workload | Phase | External wall s | A+B logged index build s | CLI scoring s | CLI engine elapsed s | Scoring records/s | User CPU s | System CPU s | Max RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch 10k | cold | 14.251 | 12.3 | 0.3 | 0.3 | 36,729 | 97.81 | 1.10 | 712.1 |
| batch 10k | preparation | 13.933 | 13.1 | 0.3 | 0.3 | 39,843 | 100.85 | 1.05 | 843.2 |
| batch 10k | warm 1 | 0.845 | 0.0 | 0.3 | 0.3 | 37,478 | 2.18 | 0.14 | 659.5 |
| batch 10k | warm 2 | 0.786 | 0.0 | 0.3 | 0.3 | 37,206 | 2.18 | 0.14 | 659.0 |
| batch 10k | warm 3 | 0.746 | 0.0 | 0.3 | 0.3 | 38,521 | 2.20 | 0.13 | 658.8 |
| batch 100k | cold | 144.215 | 132.0 | 9.0 | 9.3 | 11,153 | 1,085.81 | 11.25 | 2,444.3 |
| batch 100k | preparation | 172.973 | 159.0 | 10.3 | 10.6 | 9,748 | 1,116.19 | 12.70 | 2,273.0 |
| batch 100k | warm 1 | 11.636 | 0.0 | 8.8 | 9.2 | 11,381 | 68.83 | 1.12 | 2,186.9 |
| batch 100k | warm 2 | 13.862 | 0.0 | 11.2 | 11.5 | 8,949 | 64.98 | 1.06 | 2,189.1 |
| batch 100k | warm 3 | 13.101 | 0.0 | 10.1 | 10.6 | 9,859 | 67.29 | 1.38 | 2,138.7 |

The CLI engine elapsed timer starts after most loading/encoding done by the command layer; it is **not total cold-run duration**. The external wall column is the end-to-end measurement. Index-build sums are rounded log values; warm runs load indices instead of building them.

### Batch: all outcomes and output checks

| Workload | Phase | Total B | Skipped | Auto | Review | No match | Relationship rows | Unmatched rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch 10k | cold | 10000 | 0 | 5173 | 4667 | 160 | 9840 | 160 |
| batch 10k | preparation | 10000 | 0 | 5172 | 4669 | 159 | 9841 | 159 |
| batch 10k | warm 1 | 10000 | 0 | 5175 | 4667 | 158 | 9842 | 158 |
| batch 10k | warm 2 | 10000 | 0 | 5173 | 4672 | 155 | 9845 | 155 |
| batch 10k | warm 3 | 10000 | 0 | 5175 | 4664 | 161 | 9839 | 161 |
| batch 100k | cold | 100000 | 0 | 57967 | 29309 | 12724 | 87276 | 12724 |
| batch 100k | preparation | 100000 | 0 | 57952 | 29324 | 12724 | 87276 | 12724 |
| batch 100k | warm 1 | 100000 | 0 | 57952 | 29323 | 12725 | 87275 | 12725 |
| batch 100k | warm 2 | 100000 | 0 | 57955 | 29323 | 12722 | 87278 | 12722 |
| batch 100k | warm 3 | 100000 | 0 | 57954 | 29323 | 12723 | 87277 | 12723 |

Every B record was processed; crossmaps started empty. Outcome totals and CSV row totals agree. Small changes in auto/review/no-match counts between runs are retained rather than averaged away: this baseline does not impose deterministic whole-batch assignment.

### Live: all aggregate measurements

| Workload | Phase | External wall s | Logged state load s | Requests | Injection wall s | Requests/s | p50 ms | p95 ms | p99 ms | Max ms | Errors | Old mappings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| live 10k | cold | 20.730 | 14.4 | 3000 | 3.03 | 990.2 | 8.4 | 23.1 | 40.6 | 90.3 | 0 | 71 |
| live 10k | preparation | 17.491 | 13.3 | 3000 | 2.69 | 1,115.5 | 8.2 | 19.3 | 27.8 | 45.4 | 0 | 71 |
| live 10k | warm 1 | 4.437 | 1.2 | 3000 | 1.66 | 1,805.6 | 1.5 | 17.9 | 26.5 | 58.4 | 0 | 71 |
| live 10k | warm 2 | 4.407 | 1.2 | 3000 | 1.69 | 1,770.9 | 2.0 | 17.8 | 25.8 | 73.4 | 0 | 71 |
| live 10k | warm 3 | 4.416 | 1.2 | 3000 | 1.65 | 1,821.1 | 1.6 | 17.5 | 28.4 | 68.0 | 0 | 71 |
| live 100k | cold | 171.320 | 156.1 | 10000 | 12.55 | 796.8 | 11.4 | 26.7 | 36.4 | 91.9 | 0 | 91 |
| live 100k | preparation | 173.417 | 157.7 | 10000 | 11.29 | 885.7 | 10.5 | 22.9 | 29.7 | 84.3 | 0 | 91 |
| live 100k | warm 1 | 14.850 | 5.0 | 10000 | 7.04 | 1,421.2 | 4.4 | 18.9 | 25.5 | 76.3 | 0 | 91 |
| live 100k | warm 2 | 15.837 | 4.8 | 10000 | 7.20 | 1,389.7 | 4.5 | 19.3 | 26.7 | 80.5 | 0 | 91 |
| live 100k | warm 3 | 17.205 | 5.3 | 10000 | 8.76 | 1,141.0 | 5.6 | 23.6 | 34.1 | 91.6 | 0 | 91 |

Logged state-load time is an internal elapsed field, not end-to-end server startup: it excludes some encoder initialization. Wrapper readiness polling adds up to two seconds (10k) or three seconds (100k). Whole-wrapper wall additionally includes injection, shutdown, exports and cache saving. `Old mappings` counts response metadata for broken prior links during updates; it is not a ground-truth match-quality metric.

### Live: per-operation latency, errors, statuses and counts

| Workload | Phase | Operation | N | p50 ms | p95 ms | p99 ms | Max ms | Errors | Statuses | Old mappings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| live 10k | cold | new_a | 601 | 8.9 | 25.4 | 46.1 | 90.3 | 0 | added=601 | 0 |
| live 10k | cold | new_b | 600 | 9.5 | 25.1 | 39.6 | 85.6 | 0 | added=600 | 0 |
| live 10k | cold | upd_a_emb | 600 | 9.6 | 25.5 | 40.6 | 53.0 | 0 | updated=600 | 25 |
| live 10k | cold | upd_a_field | 299 | 1.5 | 11.2 | 13.7 | 41.5 | 0 | updated=299 | 17 |
| live 10k | cold | upd_b_emb | 600 | 10.5 | 25.0 | 40.6 | 49.5 | 0 | updated=600 | 16 |
| live 10k | cold | upd_b_field | 300 | 1.5 | 11.0 | 12.0 | 14.5 | 0 | updated=300 | 13 |
| live 10k | preparation | new_a | 601 | 8.7 | 18.7 | 26.5 | 39.3 | 0 | added=601 | 0 |
| live 10k | preparation | new_b | 600 | 8.5 | 19.9 | 27.1 | 35.2 | 0 | added=600 | 0 |
| live 10k | preparation | upd_a_emb | 600 | 9.9 | 21.1 | 30.7 | 45.4 | 0 | updated=600 | 25 |
| live 10k | preparation | upd_a_field | 299 | 1.3 | 5.0 | 11.6 | 14.8 | 0 | updated=299 | 17 |
| live 10k | preparation | upd_b_emb | 600 | 10.1 | 21.6 | 31.3 | 44.3 | 0 | updated=600 | 16 |
| live 10k | preparation | upd_b_field | 300 | 1.4 | 6.9 | 11.9 | 22.3 | 0 | updated=300 | 13 |
| live 10k | warm 1 | new_a | 601 | 0.9 | 2.4 | 4.9 | 11.6 | 0 | added=601 | 0 |
| live 10k | warm 1 | new_b | 600 | 1.0 | 2.5 | 8.0 | 11.6 | 0 | added=600 | 0 |
| live 10k | warm 1 | upd_a_emb | 600 | 10.4 | 21.8 | 31.2 | 41.2 | 0 | updated=600 | 25 |
| live 10k | warm 1 | upd_a_field | 299 | 1.1 | 4.0 | 13.1 | 21.1 | 0 | updated=299 | 17 |
| live 10k | warm 1 | upd_b_emb | 600 | 10.6 | 25.0 | 31.1 | 58.4 | 0 | updated=600 | 16 |
| live 10k | warm 1 | upd_b_field | 300 | 1.1 | 6.7 | 11.8 | 17.8 | 0 | updated=300 | 13 |
| live 10k | warm 2 | new_a | 601 | 1.0 | 3.6 | 7.6 | 19.3 | 0 | added=601 | 0 |
| live 10k | warm 2 | new_b | 600 | 1.0 | 3.4 | 7.7 | 12.2 | 0 | added=600 | 0 |
| live 10k | warm 2 | upd_a_emb | 600 | 10.2 | 23.1 | 30.5 | 73.4 | 0 | updated=600 | 25 |
| live 10k | warm 2 | upd_a_field | 299 | 1.2 | 4.3 | 12.5 | 22.2 | 0 | updated=299 | 17 |
| live 10k | warm 2 | upd_b_emb | 600 | 10.5 | 22.1 | 30.3 | 39.2 | 0 | updated=600 | 16 |
| live 10k | warm 2 | upd_b_field | 300 | 1.2 | 4.3 | 20.0 | 41.0 | 0 | updated=300 | 13 |
| live 10k | warm 3 | new_a | 601 | 1.0 | 2.3 | 4.2 | 7.9 | 0 | added=601 | 0 |
| live 10k | warm 3 | new_b | 600 | 1.0 | 2.1 | 4.0 | 11.5 | 0 | added=600 | 0 |
| live 10k | warm 3 | upd_a_emb | 600 | 10.4 | 23.2 | 33.2 | 68.0 | 0 | updated=600 | 25 |
| live 10k | warm 3 | upd_a_field | 299 | 1.2 | 3.8 | 11.5 | 17.2 | 0 | updated=299 | 17 |
| live 10k | warm 3 | upd_b_emb | 600 | 10.5 | 23.4 | 37.3 | 41.7 | 0 | updated=600 | 16 |
| live 10k | warm 3 | upd_b_field | 300 | 1.1 | 2.8 | 12.1 | 19.2 | 0 | updated=300 | 13 |
| live 100k | cold | new_a | 2001 | 12.1 | 26.4 | 36.9 | 89.6 | 0 | added=2001 | 0 |
| live 100k | cold | new_b | 2000 | 12.2 | 26.6 | 35.2 | 91.9 | 0 | added=2000 | 0 |
| live 100k | cold | upd_a_emb | 2000 | 14.0 | 29.5 | 38.5 | 80.2 | 0 | updated=2000 | 31 |
| live 100k | cold | upd_a_field | 999 | 2.9 | 12.6 | 22.3 | 74.5 | 0 | updated=999 | 13 |
| live 100k | cold | upd_b_emb | 2000 | 13.9 | 29.4 | 40.3 | 81.0 | 0 | updated=2000 | 36 |
| live 100k | cold | upd_b_field | 1000 | 2.9 | 12.7 | 17.6 | 33.3 | 0 | updated=1000 | 11 |
| live 100k | preparation | new_a | 2001 | 11.0 | 22.8 | 30.7 | 84.3 | 0 | added=2001 | 0 |
| live 100k | preparation | new_b | 2000 | 11.3 | 23.7 | 30.2 | 79.2 | 0 | added=2000 | 0 |
| live 100k | preparation | upd_a_emb | 2000 | 12.5 | 23.9 | 29.7 | 80.5 | 0 | updated=2000 | 31 |
| live 100k | preparation | upd_a_field | 999 | 2.9 | 9.1 | 14.5 | 28.6 | 0 | updated=999 | 13 |
| live 100k | preparation | upd_b_emb | 2000 | 12.8 | 24.6 | 31.3 | 70.2 | 0 | updated=2000 | 36 |
| live 100k | preparation | upd_b_field | 1000 | 3.0 | 8.4 | 14.8 | 70.3 | 0 | updated=1000 | 11 |
| live 100k | warm 1 | new_a | 2001 | 2.6 | 6.1 | 9.5 | 68.2 | 0 | added=2001 | 0 |
| live 100k | warm 1 | new_b | 2000 | 2.6 | 5.9 | 9.6 | 73.1 | 0 | added=2000 | 0 |
| live 100k | warm 1 | upd_a_emb | 2000 | 11.7 | 22.9 | 28.9 | 76.2 | 0 | updated=2000 | 31 |
| live 100k | warm 1 | upd_a_field | 999 | 3.0 | 7.1 | 11.7 | 17.9 | 0 | updated=999 | 13 |
| live 100k | warm 1 | upd_b_emb | 2000 | 11.4 | 22.9 | 28.0 | 76.3 | 0 | updated=2000 | 36 |
| live 100k | warm 1 | upd_b_field | 1000 | 3.0 | 7.0 | 12.1 | 67.8 | 0 | updated=1000 | 11 |
| live 100k | warm 2 | new_a | 2001 | 2.7 | 6.2 | 11.3 | 23.4 | 0 | added=2001 | 0 |
| live 100k | warm 2 | new_b | 2000 | 2.7 | 6.1 | 9.8 | 68.0 | 0 | added=2000 | 0 |
| live 100k | warm 2 | upd_a_emb | 2000 | 11.6 | 23.6 | 29.9 | 80.5 | 0 | updated=2000 | 31 |
| live 100k | warm 2 | upd_a_field | 999 | 3.2 | 6.9 | 12.9 | 28.3 | 0 | updated=999 | 13 |
| live 100k | warm 2 | upd_b_emb | 2000 | 11.6 | 23.4 | 32.4 | 75.7 | 0 | updated=2000 | 36 |
| live 100k | warm 2 | upd_b_field | 1000 | 3.1 | 7.5 | 11.8 | 24.9 | 0 | updated=1000 | 11 |
| live 100k | warm 3 | new_a | 2001 | 2.8 | 11.5 | 16.1 | 84.9 | 0 | added=2001 | 0 |
| live 100k | warm 3 | new_b | 2000 | 2.8 | 11.9 | 17.3 | 42.4 | 0 | added=2000 | 0 |
| live 100k | warm 3 | upd_a_emb | 2000 | 14.1 | 30.0 | 43.6 | 91.6 | 0 | updated=2000 | 31 |
| live 100k | warm 3 | upd_a_field | 999 | 3.1 | 12.5 | 20.1 | 28.0 | 0 | updated=999 | 13 |
| live 100k | warm 3 | upd_b_emb | 2000 | 14.0 | 28.6 | 39.6 | 84.6 | 0 | updated=2000 | 36 |
| live 100k | warm 3 | upd_b_field | 1000 | 3.2 | 13.0 | 18.0 | 42.9 | 0 | updated=1000 | 11 |

Each live run used 10 workers and seed 42. At 10k the actual counts are new A 601, new B 600, embedding-update A/B 600 each, field-only update A 299 and B 300. At 100k they are 2,001 / 2,000 / 2,000 / 2,000 / 999 / 1,000 respectively. Integer rounding and seeded remainder allocation explain the departure from exact percentages. All **65,000 live requests** across the ten runs completed with zero reported HTTP/body errors.

### Live resources and cache behaviour

| Workload | Phase | User CPU s | System CPU s | Max RSS MiB | Cache classification | Refreshed records |
| --- | --- | --- | --- | --- | --- | --- |
| live 10k | cold | 121.85 | 4.52 | 757.0 | cold build | full build |
| live 10k | preparation | 113.40 | 5.04 | 758.4 | cold build | full build |
| live 10k | warm 1 | 13.75 | 2.85 | 684.4 | partial cache refresh after prior injections; retained extra vectors, not clean immutable warm | A=577, B=574 |
| live 10k | warm 2 | 14.20 | 2.81 | 682.4 | partial cache refresh after prior injections; retained extra vectors, not clean immutable warm | A=577, B=574 |
| live 10k | warm 3 | 14.09 | 2.88 | 681.1 | partial cache refresh after prior injections; retained extra vectors, not clean immutable warm | A=577, B=574 |
| live 100k | cold | 1,162.63 | 27.57 | 1,550.7 | cold build | full build |
| live 100k | preparation | 1,231.93 | 27.39 | 1,760.4 | cold build | full build |
| live 100k | warm 1 | 57.58 | 11.09 | 1,668.9 | partial cache refresh after prior injections; retained extra vectors, not clean immutable warm | A=1970, B=1972 |
| live 100k | warm 2 | 59.30 | 11.36 | 1,658.4 | partial cache refresh after prior injections; retained extra vectors, not clean immutable warm | A=1970, B=1972 |
| live 100k | warm 3 | 55.71 | 9.81 | 1,623.0 | partial cache refresh after prior injections; retained extra vectors, not clean immutable warm | A=1970, B=1972 |

**The official live warm wrappers do not create pristine, immutable warm starts.** Previous injection updates are saved to vector caches; the next run reloads the original CSVs and refreshes changed vectors. Measured warm runs refreshed A/B **577/574** records at 10k and **1,970/1,972** at 100k. Injected extra IDs also remain in the caches (about 10,601/10,600 and 102,001/102,000 vectors). No full re-encoding occurred. Identical injected IDs are reused between runs, so the nominal 80% encoding operation mix is not a guarantee that 80% of warm requests execute fresh inference. Preserve this procedure for comparisons, or explicitly establish a separate pristine-cache baseline; do not conflate the two.

### Inputs and provenance

Pinned binary SHA256: `5aaac6df9ae2095be1f13093863b61987bcef511687cb61ea86b6f74d2abae81`.

| Input CSV | Data rows | SHA256 |
| --- | --- | --- |
| benchmarks/data/dataset_a_10k.csv | 10000 | `aa181b8f7a83767c47216621fdd5dd3cb1e14e00655d3da20245a348397ef650` |
| benchmarks/data/dataset_a_100k.csv | 100000 | `958b0608d77edfd6a86ff4914e26620102492452b9a1573db2a4c309d63def4a` |
| benchmarks/data/dataset_b_10k.csv | 10000 | `32746d52a7754a17354c077ce2f812fa5a681d603e27a9eec3be15d12fe1b9eb` |
| benchmarks/data/dataset_b_100k.csv | 100000 | `708dc389b0613cdcc3c24c9730f1c84dbf49680df8c20d8a507cd29f6ca809fa` |

Copies of all 17 official configuration/script/helper resources were hash-verified unchanged against the checkout. Individual resource hashes, model-file hashes, exact commands, UTC run timestamps, cache evidence, raw-log hashes, all checks and all parsed measurements are retained in the adjacent machine-readable file.

### Files, reproduction and stopping point

- Machine-readable results: [`2026-11-17-m3-before-local-reassignment.json`](2026-11-17-m3-before-local-reassignment.json). Unrelated environment variables are omitted.
- Isolated run directory: `/tmp/melder_pre_baseline_5kezwp_6`.
- Full raw stdout/stderr, live server logs (including shutdown), `/usr/bin/time -l` output, crossmaps and CSV outputs: `/tmp/melder_pre_baseline_5kezwp_6/results/`. Each JSON run records its exact `logs_dir`.
- Temporary orchestration and parser: `/tmp/melder_pre_baseline_5kezwp_6/run_all.py` and `/tmp/melder_pre_baseline_5kezwp_6/analyze.py`.
- Each of the eight listed official wrapper commands was run unchanged with `--binary /private/tmp/melder_pre_baseline_5kezwp_6/meld`, from the sandbox. The JSON records the complete command for every run.
- Rerun from a **fresh sandbox with the same input hashes**: cold once, separate warm preparation once, then three warm repetitions per workload. Do not reuse this final mutated live cache as a fresh preparation state.
- Initial sandbox orchestration mistook TCP TIME_WAIT for an occupied port after the first live run. Nine checks launched no wrappers; the orchestration listener check was corrected, and all remaining runs completed. No unrelated process was terminated and no official script was edited.
- The release build and benchmarks are the validation performed here. This task made no source changes and did not rerun the unit-test suite.
- No branch, commit, local reassignment implementation or global optimization was started. Stop after recording this baseline.
