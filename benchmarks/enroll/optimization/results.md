# Enrollment preparation optimization — M3 MacBook Air

Baseline `3feb027` versus the experimental `experiment/enroll-performance` candidate, default-feature release builds. Apple M3, MiniLM, USearch, encoder pool four, 10k seed records. Requests use 1,000 evenly sourced fixture B records (the first 1,000 of the deterministic shuffled fixture), remapped into enrollment fields; entries 49 and 99 repeat earlier IDs. All record text, seed bytes, configs and response evidence are retained. Fresh seed indexes copied for every run, warm model/OS caches. Server readiness and shutdown excluded from timing. Only encoding/search/publication plus HTTP overhead measured. No matching/assignment configuration changes.

| Workload | Baseline rec/s | Candidate rec/s | Ratio |
|---|---:|---:|---:|
| Individual requests, one client | 479 | 488 | 1.02× |
| Individual requests, ten clients | 558 | 1,032 | 1.85× |
| Bulk requests, one client, 50 records/request | 583 | 964 | 1.65× |

Medians of three pairs per workload, alternating baseline/candidate execution order; all timed server sessions were serial and preceded by at least a 120-second idle pause after the previous session. Three initial baseline-only screening runs are retained but excluded from these medians. Benchmark scratch: `/tmp/melder-enroll-opt.KxQ6uH/`. Initial cache preparation is outside the measured results. Timing runs were performed on AC power.

## Correctness checks and limitations

All 21 runs completed 1,000 enrollment results successfully. Across paired runs, all records returned the same **sets of edge IDs**, including duplicate IDs. Individual-request results were byte-value-equivalent as parsed JSON at both client counts. Bulk inference caused small floating-point differences (maximum absolute score difference `2.980232238769531e-7`) in 442/1,000 results, and **three records returned the same edges in a different order** due to numerically near-tied scores. These differences repeated in all three trials; they must not be described as exact ordered-response equivalence. No observed edge crossed an inclusion boundary in this fixture; another fixture could contain scores arbitrarily close to a threshold, so this is not proof of universal bitwise equivalence.

The ten-client comparison is a scheduling-dependent online workload: the pool evolves as incoming enrollments are committed. Equal edge sets here are evidence for this fixture, not a promise of order independence. A near-2× concurrent speedup and substantial bulk speedup are supported; a material sequential speedup is not.

`results.json` includes every result, hardware health snapshots and edge audit; `manifest.json` records seed/input/binary hashes. `measure.py` and `analyze.py` preserve the exact driver and analysis. `evidence.tar.gz` contains the 21 configs, server logs and response/latency data. Binary files and seed cache remain private scratch, not repository artifacts.
