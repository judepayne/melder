# M1 Ultra Mac Studio — documentation benchmark rerun sheet

**Status: M1 Ultra batch reruns and artifact checks complete (runs started 2026-11-19).** Live benchmarks will be rerun separately on the M3 MacBook Air; no Studio live measurements are included. Abt–Buy and the two-hour soak were not run. Published figures are transcribed from the documentation at `c57156d`. Paths starting with `benchmarks/` or `docs/` are relative to the repository root.

## Studio run record

- Machine: Apple M1 Ultra Mac Studio (`Mac13,2`), 20 logical CPUs, 64 GiB RAM, macOS 26.6.2 (`25G83`), AC power.
- Candidate: `3e3dc2f566710d892237bd0c68b6c3d72a145531`. Pinned `--locked --release` default binary SHA-256: `06ca792ba9cdaf9563ff1a7fada1008d41d814954e920268a5d464e4f30b4e61`; `--features gpu-encode` binary SHA-256: `ac5d87ba06f74e4873fe6eccccd50e450936e0e6fdfd07afbad994911de9849d`.
- Run artifacts on the Studio: `/tmp/melder-m1-rerun.uvKgZ3`. Workers share an exclusive measurement lock to avoid benchmarking against each other. Workload-specific settings, baseline provenance, and comparisons are recorded with the results.

<!-- studio-results-start -->
## New Studio measurements and comparisons

The following run reports are copied from the raw-artifact directory above. They are separate from the published baseline tables below. A pending row is not a result. Do not publish a speedup unless the workloads and timing boundaries match.

| Workload | Runner state |
|---|---|
| #1 Standard batch | finished |
| #2–3 Blocked and unblocked BM25 | finished |
| #4–5 1M CPU/CoreML build and sweep | finished |
| #6–9 Live and bulk API | Awaiting M3 MacBook Air rerun; see worksheet below |
| Abt–Buy | Cancelled before any run |
| #10 Two-hour soak | Excluded; not run |

### M1 Ultra standard batch rerun (#1)

**Execution state:** standard_complete_abt_pending; 40/40 ordinary jobs completed. Empty tables are not measurements. Runner PID: 53267. Abt–Buy was cancelled before any run.

Candidate and fixture provenance: `standard/manifest.json`. Exact configs, input hashes, initial cache hashes, process stdout/stderr, `/usr/bin/time -l` (including peak RSS), output CSVs and accepted pair sets are preserved under `standard/`. All measured jobs exclusively hold the shared Python `fcntl.flock` on `measurement.lock`. No repository fixture/output/source/doc modifications.

#### Conditions and baseline comparability

- Pinned default-feature release binary `$D/meld` (built by parent), MiniLM-L6-v2, encoder pool 4; no GPU. Retained local 10k/100k synthetic CSV fixtures, country blocking and original field weights/thresholds/top_n/ANN settings. Environment overrides and SHA-256 hashes are in the manifest.
- `default` omits assignment selection (current graph/approximate); `greedy` explicitly selects strategy greedy without mode. Fresh crossmap and fresh outputs per repetition. Cold removes only this private case/policy cache; warm is a new process reusing that same cold-created cache. OS filesystem/model caches are not flushed: cold means absent vector-index cache, not cold OS cache.
- Planned sampling: 10k three cold and three warm repeats per backend/policy; 100k one expensive cold and three warm repeats. Actual completed n is reported in the measured table. Flat 100k is a private derivative of the preserved 100k usearch config with only backend and private paths/policy changed. There is no original named flat-100k harness.
- Published opening table: M3-attributed, historical greedy/default and unpinned original binary/fixture hashes. User identifies Studio as intended older machine, but documentation attribution conflicts. No old binary was rebuilt; no historical same-machine or code-speedup claim is justified. Compare new default versus greedy only within these retained fixtures/settings; scoring throughput excludes setup/output and must not be equated with full process wall time.
- `/usr/bin/time` wraps each real meld process; wall excludes lock waiting and pre-run hashing/cleanup. Actual output row counts are read from the completed CSVs, not inferred from scores. CLI scoring throughput is retained at the precision printed by the binary. See per-job metric-lines.txt for index/cache/setup timing evidence; do not relabel the CLI graph “Setup and assignment” duration as index build time.

#### Measured medians

| Case | Policy | Cache | n | Process wall s | Scoring rec/s | Accepted rows | Review rows | Unmatched rows | Success |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 10kx10k-flat | default | cold | 3 | 11.507 | 1883.0 | [5179] | [4721] | [100] | True |
| 10kx10k-flat | default | warm | 3 | 6.510 | 1678.0 | [5179] | [4721] | [100] | True |
| 10kx10k-flat | greedy | cold | 3 | 11.258 | 1913.0 | [5171, 5174] | [4674, 4677, 4681] | [148, 149, 152] | True |
| 10kx10k-flat | greedy | warm | 3 | 5.740 | 1909.0 | [5172, 5175] | [4677, 4680, 4682] | [143, 148] | True |
| 10kx10k-usearch | default | cold | 3 | 6.470 | 45041.0 | [5178] | [4709, 4710] | [112, 113] | True |
| 10kx10k-usearch | default | warm | 3 | 0.787 | 45370.0 | [5178] | [4709] | [113] | True |
| 10kx10k-usearch | greedy | cold | 3 | 6.672 | 60602.0 | [5172, 5173, 5174] | [4665, 4667, 4674] | [154, 160, 161] | True |
| 10kx10k-usearch | greedy | warm | 3 | 0.680 | 65648.0 | [5172, 5173] | [4665, 4666, 4668] | [160, 161, 163] | True |
| 100kx100k-flat | default | cold | 1 | 538.145 | 208.0 | [58430] | [29008] | [12562] | True |
| 100kx100k-flat | default | warm | 3 | 553.018 | 182.0 | [58430] | [29008] | [12562] | True |
| 100kx100k-flat | greedy | cold | 1 | 546.448 | 204.0 | [58101] | [29331] | [12568] | True |
| 100kx100k-flat | greedy | warm | 3 | 496.894 | 203.0 | [58092, 58105, 58128] | [29304, 29327, 29340] | [12568] | True |
| 100kx100k-usearch | default | cold | 1 | 71.741 | 31882.0 | [58282] | [29009] | [12709] | True |
| 100kx100k-usearch | default | warm | 3 | 6.471 | 31094.0 | [58282] | [29009] | [12709] | True |
| 100kx100k-usearch | greedy | cold | 1 | 72.358 | 29804.0 | [57946] | [29335] | [12719] | True |
| 100kx100k-usearch | greedy | warm | 3 | 6.134 | 31295.0 | [57931, 57948, 57951] | [29330, 29335, 29352] | [12717, 12719] | True |

#### Within-candidate policy controls

Wall ratio is graph/default ÷ greedy (>1 = graph slower); this is not an old-code speedup.
- 100kx100k-flat cold: wall ratio 0.985×; accepted [58430] vs [58101]; reviews [29008] vs [29331].
- 100kx100k-flat warm: wall ratio 1.113×; accepted [58430] vs [58092, 58105, 58128]; reviews [29008] vs [29304, 29327, 29340].
- 100kx100k-usearch cold: wall ratio 0.991×; accepted [58282] vs [57946]; reviews [29009] vs [29335].
- 100kx100k-usearch warm: wall ratio 1.055×; accepted [58282] vs [57931, 57948, 57951]; reviews [29009] vs [29330, 29335, 29352].
- 10kx10k-flat cold: wall ratio 1.022×; accepted [5179] vs [5171, 5174]; reviews [4721] vs [4674, 4677, 4681].
- 10kx10k-flat warm: wall ratio 1.134×; accepted [5179] vs [5172, 5175]; reviews [4721] vs [4677, 4680, 4682].
- 10kx10k-usearch cold: wall ratio 0.970×; accepted [5178] vs [5172, 5173, 5174]; reviews [4709, 4710] vs [4665, 4667, 4674].
- 10kx10k-usearch warm: wall ratio 1.157×; accepted [5178] vs [5172, 5173]; reviews [4709] vs [4665, 4666, 4668].

#### Flat 10k warm: controlled local `main` comparison

Three runs per revision on the same M1 Ultra, using the same verified 10k CSV inputs and byte-identical warm caches, with a fresh CrossMap per run. Local `main` (`b9a83a4`) uses greedy; the branch (`3e3dc2f`) was set explicitly to greedy. Runs were interleaved under the shared lock, with the original branch restored afterward.

| Revision | Scoring rec/s, individual runs | Median scoring rec/s | Median process wall |
|---|---:|---:|---:|
| Local `main` | 1,690 / 1,910 / 1,699 | 1,699 | 6.464 s |
| Assignment branch, greedy | 1,897 / 1,884 / 1,736 | 1,884 | 5.960 s |

The observed ranges overlap; this does not establish an 11% speedup. It does show that local `main` also falls well short of the published 5,507 rec/s. The follow-up does not reproduce the original published workload environment. Raw logs, config and cache hashes, and per-run results: `/tmp/melder-m1-rerun.uvKgZ3/flat-main-compare-ohl2xf4r`.

#### Flat 10k warm: one M3 MacBook Air check

On 2026-11-20, one warm greedy run on the M3 Air (macOS 26.5.1, AC power) scored **5,597 records/s** and finished in **2.240 s**. Both 10k indexes loaded from cache. The fixture hashes and branch release binary SHA-256 matched the Studio run. Rayon used its default thread count on each machine. The M3 result is close to the published **5,507 records/s** and **2.2 s**, and roughly 3× the Studio branch median of **1,884 records/s** from the controlled greedy comparison above. This is one measurement, not a hardware-only attribution or a new published baseline. M3 logs: `/tmp/melder-m3-flat.1XnlBe/`.

#### USearch 10k warm: one M3 MacBook Air check

On 2026-11-20, one warm greedy run on the M3 Air scored **37,773 records/s** and finished in **0.790 s**. Both 10k indexes loaded from cache and the run accepted 5,171 pairs. The release binary and input hashes match the Studio runs. The published figures were **33,738 records/s** and **0.3 s**; the Studio greedy medians were **65,648 records/s** and **0.680 s**. This is one spot check, not a new baseline. M3 logs and result JSON: `/var/folders/g6/zfm6_wzj20g4dq7x42bv4_l40000gn/T/melder-m3-usearch.j8euvjl2/`.

#### Historical reference (not like-for-like speedup)

| Metric | Flat 10k | Usearch 10k | Flat 100k | Usearch 100k |
|---|---:|---:|---:|---:|
| Published index build | ~17s | ~17s | ~3m | ~3m32s |
| Published cache load | ~47ms | ~78ms | ~650ms | ~235ms A + ~261ms B |
| Published scoring rec/s | 5,507 | 33,738 | not reported | 10,539 |
| Published warm wall | 2.2s | 0.3s | not reported | 9.5s |

Historical cold “3m32s” appears as both 100k usearch index-build and wall; retain as published, do not substitute timing categories. New measured values above are independently labeled M1 results. Quantized/unquantized is outside this assigned ordinary-table run and remains unmeasured here.

#### Abt–Buy quality

Cancelled before any run.

#### Execution status

Report generated 2026-11-19 19:35:00 +0000; completed jobs: 40. Consult `standard/status.json`, `standard/runner.log` and `standard/runner.exit` for exact remaining/running status. Ordinary target: 40 measured jobs (24 10k +16 100k); Abt–Buy not included unless explicitly added. Raw per-run machine-readable results: `standard/results.json`; grouped medians: `standard/summary.json`.

### M1 Ultra BM25 rerun — worksheet #2 and #3

Status: **0**. Measured 2026-11-19; originals retained in `bm25/original-*`. No shared worksheet/docs changed. No soak or commits.

#### Conditions
Apple M1 Ultra Mac Studio, 20 cores (16P+4E), 64 GB; macOS 26.6.2. Current revision `3e3dc2f566710d892237bd0c68b6c3d72a145531`, default-feature locked release; binary `$D/meld` SHA-256 `06ca792ba9cdaf9563ff1a7fada1008d41d814954e920268a5d464e4f30b4e61`. `$D=/tmp/melder-m1-rerun.uvKgZ3`. `build.exit=0` checked first. All measurements/data preparation serialized under Python `fcntl.flock(LOCK_EX)` on `$D/measurement.lock`; separate output/crossmap/cache/scratch DB for every process. No input regeneration for blocked tests. No persistent embedding indices; local MiniLM encoder can initialize but no embedding scoring fields. Model cache was already present; cold here means private matching artifacts, **not** a cold OS page/model cache.
Blocked tests inherit Rayon automatic 20-thread machine default (no overrides captured); configs retain encoder pool 4 for 10k memory, 1 for SQLite and 1M. Unblocked explicitly uses four Rayon threads. Power settings and full hashes are in `bm25/metadata.json` and `inputs-*.json`.

#### #2 Blocked BM25
Graph is the genuinely omitted assignment setting (current graph/approximate default), not a pretend historical policy. Greedy is explicit and has no mode. Original configs retained; only paths and greedy selection changed. 10k uses 3 fresh processes/strategy (alternating order), 1M uses one full process/strategy, no B limit. Medians shown for timings/RSS, counts shown as observed ranges.

Scoring phase reconstructed as B count / integer CLI records/s; rounded CLI phase also retained. Graph scoring excludes solver/publication; process wall includes initialization, data/index construction, scoring, solver, persistence and CSV export. `/usr/bin/time -l` peak RSS is bytes converted to MiB, not system RAM or estimated footprint.

| Case | Strategy | n | Scoring s (CLI / from rate) | Rate B/s | Wall s | Solver s | Peak RSS MiB | Accepted | Review | Unmatched |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k-memory | graph | 3 | 0.1 / 0.128586 | 77,769 | 0.760 | 0.010686 | 599.2 | 6092 | 3144 | 764 |
| 10k-memory | greedy | 3 | 0.1 / 0.128612 | 77,753 | 0.626 | — | 598.8 | 6079–6084 | 2751–2779 | 1137–1168 |
| 10k-sqlite | graph | 3 | 4.3 / 4.284490 | 2,334 | 4.617 | 0.012278 | 148.0 | 6092 | 3143 | 765 |
| 10k-sqlite | greedy | 3 | 6.5 / 6.510417 | 1,536 | 6.777 | — | 75.2 | 6079–6080 | 2761–2781 | 1140–1160 |
| 1M-memory | graph | 1 | 1070.8 / 1070.663812 | 934 | 1109.440 | 0.424790 | 17397.7 | 657119 | 197523 | 145358 |
| 1M-memory | greedy | 1 | 1080.0 / 1079.913607 | 926 | 1112.387 | — | 19964.4 | 645954 | 196776 | 157270 |

1M graph actual effective summary: Encoder ready (dim=384), took 0.1s; Loaded dataset A: 1000000 records in 1.4s; State loaded in 2.4s (A: 1000000 records); Loaded 1000000 B records in 1.7s;   Batch assignment: graph (approximate);   Groups: 461757 exact · 3 approximate · largest: 617 records;   Retained edges: 1298579 · fixed pairs: 0;   Total score: at least 99.9% of the best possible over retained eligible pairs.;   Setup and assignment: 14.2s

SQLite bulk-load logged progress rates (all sides/repeats): 162,870–181,639 records/s. The 0.1s per-side completion display is coarse; do not derive precise rates from it.

##### Preserved old figures and interpretation
| Published metric | Old | Interpretation |
|---|---:|---|
| 10k memory scoring | 49,337 B/s | Hardware attribution uncertain; not a controlled old-code speedup. |
| 10k SQLite scoring | 2,099 B/s | Same caveat; current settings retain automatic 20 readers. |
| 10k accepted memory / SQLite | 7,824 / 7,931 | Compare accepted links, not all relationship rows (which include review). Current counts differ substantially. |
| 10k RSS memory / SQLite | ~2 GB / ~1.2 GB | Historical method/units unknown; new process RSS is explicitly MiB. |
| SQLite bulk load | 170–194K records/s | New actual logged progress rates above. |
| 1M memory scoring | 1,062 B/s | Historical one-million case used top_n=5, bm25_candidates=10, encoder pool=1, same as isolated new config. |
| 1M projected full run | ~16 min | Preserve as historical projection; new wall is a measured complete run. |
No confirmed historical input hashes or old blocked binary were supplied; input reuse means existing local CSVs, not proof of identical historical datasets. Pipeline/assignment/engine changes and different acceptance counts preclude a causal speedup claim. Graph approximates only above 512 endpoints/group; actual effective exactness/group summaries are retained in enriched JSON and raw logs. Greedy has scheduling-dependent accepted counts; medians do not assert deterministic membership. Relationships include review records, so they are not CrossMap cardinality. All short-case selected CSV links were audited against final CrossMap. No 55M-memory claim tested.

#### #3 Unblocked original greedy harness

Resumed #3 only; terminal exit **0**. #2 was not rerun. See `bm25-resume3.log` and `bm25/unblocked.exit`.

Pinned baseline `877f261` SHA-256 `896cb2f7bf67cf6a6b7d105d0b21d8fa1e657226950af889b2ff9b4dc81b0ec2` matches the retained historical report; no rebuild. Candidate SHA-256 `06ca792ba9cdaf9563ff1a7fada1008d41d814954e920268a5d464e4f30b4e61` matches metadata. Original copied harness verified byte-for-byte against the actual repository harness, then its `__file__` anchored to that source to correct the outside-repository guard. No matching/config/comparison semantics changed.

Explicit greedy; 1,000 B queries, 10k/100k A pools, selective/common workloads, four Rayon threads, three repeats per binary in alternating order. Exclusive `measurement.lock` held for all preparation and all 24 subprocesses. External `/usr/bin/time -l` adds RSS only. Cached inputs verified by SHA-256 before reuse. Cold private matching artifacts, warm OS/model caches.

Full order-independent CSV audit: **True**. Each successful run has 900 relationships and 100 unmatched. Scoring reconstructed as 1,000 / integer CLI B/s; wall includes startup and export. Medians below; historical candidate is not the current candidate.

| A / workload | Old scoring baseline → candidate s | New scoring baseline → current s | New wall baseline → current s | Speedup | CSV equal |
|---|---:|---:|---:|---:|---|
| 10,000 / selective | 0.321 → 0.011 | 0.308 → 0.004 | 0.517 → 0.188 | 74.05× | True |
| 10,000 / common | 0.957 → 0.887 | 1.006 → 0.962 | 1.168 → 1.111 | 1.05× | True |
| 100,000 / selective | 3.968 → 0.014 | 4.202 → 0.006 | 4.724 → 0.568 | 693.40× | True |
| 100,000 / common | 13.333 → 8.065 | 11.905 → 7.937 | 12.479 → 8.561 | 1.50× | True |

Individual timings/RSS, counts and hashes: `bm25/unblocked.json`. Logs/outputs: `bm25/unblocked-artifacts` (original JSON paths relocated per `artifact-location.json`). Input audit: `cached-input-audit.json`; resume provenance: `resume3-provenance.json`. Same-machine/baseline/input comparison is not a causal claim about historical environmental drift.

#### Artifacts
`bm25/blocked.json`, `blocked-enriched.json`: individual blocked results and phase/acceptance audit; `unblocked.json`: original harness report; `metadata.json`: provenance/settings; `original-*.yaml`, `original_worksheet.md`, `original_unblocked_run_test.py`: preserved originals. Per-run `cli.log`, configs, CrossMaps, outputs and graph event logs remain in `bm25/`. Driver `$D/bm25_driver.py`, renderer `$D/bm25_report.py`, launch log `$D/bm25-launch.log`, exit `$D/bm25/run.exit`. No #10 soak; no commits.

### Worksheet #4–5: full-scale cold encoding comparison and GPU sweep

Status is live in `encoding/status.json`; runner PID and per-case PID are recorded there. All cases acquire `measurement.lock` using `fcntl.flock(LOCK_EX)` independently; the lock is held through termination/resource collection and released between cases. No concurrent benchmark workloads are launched by this runner.

#### Scope and conventions

Pinned parent binaries `meld` (CPU) / `meld-gpu` (GPU), production 1M fixture per side, MiniLM, two embedding fields (name .30/address .20), BM25 .50/synonym .20, blocked USearch; inputs unchanged. Exact hashes/counts/environment are in `encoding/manifest.json`. Configs are private and retained per case. CPU uses explicit batch 64 (documented CPU default; historical CPU batch unspecified); GPU settings explicitly select GPU. All builds are cold: every attempt uses fresh private A/B cache directories (required by configuration validation), verified empty before launch. No preexisting vector caches are loaded. A cache save occurs between A build completion and B build start; it is excluded from BOTH logged build intervals. B is intentionally stopped immediately after its build-complete log, so its save may be absent or partial. Save/startup time belongs only to wall-to-stop, NEVER to combined build throughput. No fixture or repository config is edited. Shared GPU pool12/batch256 measurement is used once for both tables.

**This is encode + combined-vector construction + USearch indexing, not isolated ONNX encoding or whole matching wall time.** Current `src/vectordb/mod.rs` and historical GPU-introduction revision `c963a91` both start `build_elapsed` before index creation and end after all `encode_and_upsert` workers; `records_per_sec = side records / build_elapsed`. The denominator is records, not texts: 1,000,000 A + 1,000,000 B records and 4,000,000 field-text encodes total. Aggregate rec/s below is 2,000,000 / (A build seconds + B build seconds), using logged elapsed rounded to 0.1s. Index work is inseparable in these pinned binaries; do not call it inference-only throughput.

The published sequential/parallel CPU projections (~79/~45 minutes) imply roughly one million records at their reported rates, while GPU ~18 minutes implies roughly two million. The old table is inconsistent; historical values are not rewritten. New summed build elapsed is measured, **not a projection**. No measured encoding share of end-to-end wall or full-job speedup is available.

Runner deliberately terminates the process immediately after the log confirms **B combined index built with 1,000,000 vectors**, after requiring the same for A. It does not time scoring, graph assignment or exports. The CLI loads/encodes all B before applying `--limit`; `--limit 0` is a defense-in-depth scoring limit, not a smaller encoding workload. Process wall-to-stop includes startup, A data/BM25/synonym/blocking loading, model/session initialization, both build phases and termination; see per-case JSON. Signal termination is intentional, not a completed batch run. No accepted/unmatched or end-to-end results are claimed.

An actual 64-record-per-side MiniLM sanity run executes under the same exclusive measurement lock before the suite, with explicit ORT_DYLIB_PATH pointing to the Homebrew runtime documented in docs/building.md. Its raw log and result remain in encoding/sanity-*/; the suite is blocked unless both sides encode 64 vectors, the process exits successfully, and an explicit same-dylib C API probe successfully appends CoreML and creates a MiniLM session. Merely finding a dylib or requesting GPU is insufficient. Initial failed artifacts are preserved in encoding/initial-failure-*/.

GPU provider validation uses an explicit C API probe linked to the SAME Homebrew ORT dylib, requiring CoreML in GetAvailableProviders, successful CoreML append (MLProgram/CPUAndGPU), and successful MiniLM session creation, followed by pinned meld-gpu encoding both 64-record sides. The pinned binaries omit ort's optional tracing feature, so ORT registration logs are absent even with RUST_LOG=ort=info; their absence alone is not evidence of fallback. Per-case provider labels explicitly distinguish same-runtime sanity evidence from direct registration logs. The probe plus actual encoding—not just Melder's requested-GPU message—is required. CoreML is MLProgram/CPUAndGPU, allowing CPU operation fallback; this does not prove every operator ran on GPU. Logs preserve provider registration/fallback messages. If provider registration cannot be verified, the row is explicitly unverified, not a confirmed CoreML result.

USearch dynamically doubles per-block reserved capacity (minimum 64); no fixed dataset-capacity knob exists in config. Full count and final vector count are checked rather than shrinking the input. Memory/disk are sampled during each case (`resources.jsonl`), peak process RSS is sampled (30-second interval; not an exact OS high-water mark and excludes auxiliary CoreML processes). Safety stops at 50 GiB process RSS or <25 GiB free scratch disk; each case has an eight-hour bound. No sample or safety-aborted case is published as a 1M result. No OS file-cache flush is claimed (cold refers to vector indexes).

#### Results (one repetition per setting; pending means unmeasured)

| Device | Pool | Batch | Status | Combined build records/s | A+B build seconds | Peak process RSS GiB |
|---|---:|---:|---|---:|---:|---:|
| gpu | 12 | 256 | complete_encode_and_index | 735.08 | 2720.8 | 15.75 |
| cpu | 1 | 64 | complete_encode_and_index | 435.14 | 4596.2 | 10.46 |
| cpu | 8 | 64 | complete_encode_and_index | 920.09 | 2173.7 | 11.52 |
| gpu | 16 | 128 | complete_encode_and_index | 665.23 | 3006.5 | 16.98 |
| gpu | 8 | 256 | complete_encode_and_index | 827.20 | 2417.8 | 20.83 |
| gpu | 12 | 128 | complete_encode_and_index | 758.27 | 2637.6 | 15.58 |
| gpu | 4 | 256 | complete_encode_and_index | 754.12 | 2652.1 | 16.02 |
| gpu | 12 | 512 | complete_encode_and_index | 610.02 | 3278.6 | 23.25 |
| gpu | 16 | 512 | complete_encode_and_index | 255.19 | 7837.2 | 28.54 |

Current runner state: `{"at": "2026-11-19T19:33:18.973316+00:00", "cases": 9, "pid": 55511, "state": "complete"}`.

Raw per-case `raw.log`, `events.jsonl`, `resources.jsonl`, `config.yaml`, `result.json`, and `exit.json` remain under `encoding/`. Overall completion is `encoding/runner.exit`; absent means the queue has not completed.

#### Tuning finalists: small samples versus full-scale cold builds

On 2026-11-20 the M1 Ultra screened 20 configurations (10 CPU, 10 CoreML) on 16k records per side. The two fastest per device advanced to 64k per side, with two runs per finalist in opposite orders. The best median per device determined the two full runs; these are the best settings tested, not a global optimum. All runs used the original fixture or evenly spaced samples, fresh caches, identical MiniLM fields, and serial execution after a two-minute sync pause. Throughput is records across both sides divided by summed encoding + USearch build times, excluding saving, scoring, matching and export.

| Selected configuration | 64k per side (median rec/s) | 1M per side (rec/s) | 1M per side (A+B build) |
|---|---:|---:|---:|
| CPU, standard release, 8 slots, batch 128, default threads | 1,561 | 931 | 35 min 47.5 s |
| CoreML, experimental GPU-enabled build, 12 slots, batch 128, 2 CPU threads per session | 1,887 | 905 | 36 min 50.2 s |

CoreML led by 21% in throughput at 64k per side. At full scale it took 62.7 s longer (~2.9%); one full run per device does not establish a repeatable difference that small. GPU utilization was nonzero in 717/724 system-wide samples during the full CoreML build (21.1% mean, 49% peak), versus 0.1% mean on CPU. Mean process CPU use during the builds was ~610% for CoreML versus ~1,385% for CPU (100% = one core). The thread cap requires the experimental fastembed patch, not a shipping YAML setting. Only the two finalists were run at 1M in this sweep; do not label the smaller tuning rows as 1M results.

All 30 per-run results are saved in [results.json](m1-encoder-tuning/results.json); `key` is `[device, build, encoder slots, batch size, CPU threads per session]` (`stock` = standard release, `cap` = experimental GPU-enabled build, `0` = runtime default). The [evidence archive](m1-encoder-tuning/run-evidence.tar.gz) contains the driver, manifest and hashes, selection, and all 30 configs, CLI logs and utilization traces. Original Studio directory: `/tmp/melder-tune.0A0KvJ/`.

#### ONNX Runtime package swap (M1 Ultra, 16k records per side)

Same experimental Melder binary, MiniLM inputs, and fresh USearch indexes; two verified runs per cell, reversing package order. CPU used 8 slots and batch 64; CoreML used 12 slots and batch 128. Only the loaded library changed within each row. These are mean records/s for encoding plus index construction, not inference-only or matching throughput.

| Device and CPU threads per session | 1.23.2 (PyPI arm64 wheel dylib) | 1.31.1 (Homebrew dylib) |
|---|---:|---:|
| CPU, 20 threads | 398 | 168 |
| CPU, 2 threads | 1,620 | 1,492 |
| CoreML, 2 threads | 2,148 | 2,119 |

The older package was 2.37× faster on CPU at 20 threads but only 1.09× faster at 2. Reducing threads improved both packages. CoreML differed by ~1.3%; system-wide GPU utilization averaged 46–48% during those verified runs. Different package build options prevent attributing differences to version alone. The extracted 1.23.2 library is not the standard release's bundled archive. Verified the actual loaded library path with dyld output or `lsof`; excluded one extra GPU trial without this evidence. See [individual runs](m1-encoder-tuning/ort-swap.json) (only rows with `version` and `repeat` are included above). Studio logs, library hashes and drivers: `/tmp/melder-ort-swap.I3bfeb/`. No new 1M run.

#### CPU thread cap: eight encoder slots on the Studio

2026-11-20. In an isolated copy of `3e3dc2f`, the GPU-enabled binary was built against a private fastembed patch that passes the test thread count to ONNX Runtime's `with_intra_threads`. All six runs used that same binary in CPU mode with Homebrew ORT 1.31.1, the same 16k records per side, eight encoder slots, batch size 64 and fresh USearch caches. Each log confirmed the requested setting at all eight session constructors and full A/B builds. Runs were serial after a two-minute sync wait; the second pass reversed the order. Times are the sum of encoding and index construction on both sides, excluding cache saves and matching.

| CPU threads per session | First build | Second build | Mean build |
|---|---:|---:|---:|
| 20 (unchanged default) | 204.0 s | 202.9 s | 203.5 s |
| 1 | 38.9 s | 38.6 s | 38.8 s |
| 2 | 21.4 s | 21.6 s | 21.5 s |

Changing only the per-session thread count from 20 to 2 made the build **9.5× faster**, close to the unmodified standard CPU build's 20.4 s on the same sample. Excessive per-session parallelism explains almost all the observed gap on this machine and workload; this does not establish a universal two-thread optimum or change the published 1M results. No product code or defaults changed. Remote driver, private patch, logs and `results.json`: `/tmp/melder-cap.M2zC25/`.

#### CPU scaling check: one versus eight encoder slots on the Studio

On 2026-11-20 both builds ran on CPU with batch size 64, the same MiniLM model and the same 16k records per side as the matched-device test below. Each cell shows both trials, in seconds for A+B encoding and USearch index construction (not matching). Trials were serial, with fresh caches and reversed run order in the second pass. The Studio waited two minutes after setup before the first run.

| CPU execution path | One slot | Eight slots |
|---|---:|---:|
| Standard release | 49.1 / 49.1 s | 20.5 / 20.3 s |
| GPU-enabled release in CPU mode, Homebrew ONNX Runtime 1.31.1 | 56.0 / 57.8 s | 205.3 / 206.4 s |

Relative to its own one-slot time, the standard build was 2.4× faster with eight slots; the Homebrew-runtime build took 3.6× longer. The gap between builds was 1.16× at one slot and 10.1× at eight. This isolates poor multi-session scaling in this setup, not the underlying cause. Competition between the sessions’ internal CPU threads is a hypothesis; thread counts and scheduler waits were not measured. Raw configs, driver, phase logs and results on the Studio: `/tmp/melder-cpu-slots.7LSY05/`.

#### Follow-up: matched CPU and CoreML settings on the Studio

2026-11-20, same M1 Ultra and revision. Used the same `gpu-encode` release binary and Homebrew ONNX Runtime 1.31.1 for both devices, with eight encoder slots and batch size 256. Both modes used `all-MiniLM-L6-v2`, the same scoring config, and the same evenly spaced 16,000 records from each side of the 1M fixture. Each run built both USearch indexes from empty caches. Only the A+B encode-and-build phases were timed; no matching queries ran. Three runs per device, alternating order by pair, with an exclusive lock and verified non-overlapping build windows.

| Device | A+B build times (s) | Median build time | Median throughput | GPU device utilization during build |
|---|---:|---:|---:|---|
| CPU, GPU-enabled build | 135.1 / 134.5 / 135.7 | 135.1 s | 237 rec/s | 0% in all 423 samples |
| CoreML, GPU-enabled build | 26.4 / 26.0 / 25.9 | 26.0 s | 1,231 rec/s | 34.5–35.5% median by run; active in 80/81 samples |

The CoreML build was **5.2× faster** than CPU mode in the same binary. `ioreg` samples system-wide GPU device utilization, not per-process or per-operator placement. GPU utilization was 0% in all pre-run samples for these six trials.

Two single-run CPU controls used batch size 64 to match the full-scale CPU row, with every other workload setting unchanged:

| CPU build | A+B build time | Throughput |
|---|---:|---:|
| GPU-enabled binary, Homebrew runtime | 200.5 s | 160 rec/s |
| Standard release binary | 21.2 s | 1,509 rec/s |

CPU execution was **9.5× faster** in the standard build than CPU mode in the GPU-enabled build at the same settings. It also beat the matched CoreML runs, consistent with the full-scale ranking of the standard CPU build vs. CoreML. The controls do not isolate which build/runtime difference caused the CPU gap. The controlled same-binary result demonstrates a CoreML benefit for that runtime and workload; it is not a speedup over the standard CPU release or a full-job speedup. Each CPU control ran once. No published figures changed.

Studio artifacts: `/tmp/melder-m1-coreml-check.1_60zope/` (`manifest.json`, `results.json`, `cpu-controls.json`, per-run configs/logs and GPU samples). The SSH command timed out after the fourth result, but the fifth trial completed and left its report; only the missing sixth trial was resumed.

### Final artifact checks

- Pass: 40 successful ordinary batch runs; no Abt–Buy run.
- Pass: 14 successful blocked BM25 runs.
- Pass: unblocked BM25 CSV equivalence.
- Pass: 9 encoding configurations completed at full scale.
- Pass: no overlaps in timestamped benchmark and setup logs.

Timestamp audit covered 128 benchmark and setup logs. Details: `worksheet-validation.json`.

<!-- studio-results-end -->

## Run order and comparison rules

1. The Studio batch runs are complete. Rerun the live and bulk-request cases on the M3 MacBook Air using the worksheet below. Do not run the two-hour soak. Reuse results for pages that quote the same workload.
2. Record hardware/OS: ______; date/power mode: ______; baseline commit/binary SHA: ______; candidate commit/binary SHA: ______; build features/ONNX provider: ______; input, config and initial cache hashes: ______; log directory: ______. For GPU rows record the actual provider, not just a GPU request. For live runs record clients, total requests, mutation mix, server workers and errors.
3. **Batch policy matters.** The published ordinary batch figures predate graph as default. For each applicable job run the current default (`graph`/`approximate`); use explicit `greedy` on the same starting files as the policy control. Preserve old figures; if a speedup over *old code* matters and a condition or machine is uncertain, run a pinned old revision on the Studio as well. An old/new-default comparison measures the whole product change, not just solver speed. Live still defaults to greedy. Abt–Buy quality and Linux CI are separate pre-merge checks, not published Studio performance baselines.
4. Use private job directories: several wrappers clear caches, crossmaps, WALs and outputs. Cold means start without an index cache; warm means start a new process with a prepared cache. Reset mutable records, crossmaps and outputs to the intended starting state for **each** repetition. Match model, feature flags, thread count and power settings. Keep input hashes and raw per-run results. Prefer three comparable runs and a median for short cases; state actual repetition counts for expensive runs.
5. **Report like for like.** Keep encoder records/s, scoring records/s, requests/s, process wall time, and projected time in separate cells. Record accepted/review/unmatched counts and errors alongside timings, particularly when assignments change. Only call an improvement where the conditions support it. If results are unchanged, worse, or not comparable, record that here rather than silently replacing the old table.

**Machine attribution:** Jude identifies the Studio as the intended original comparison machine. The site itself mixes machines: several sections explicitly say M1 Ultra, the [Performance overview](../../docs/performance.md#overview) names the M3 Air, and some rows have no per-table hardware label. The inventory keeps these distinctions. A confirmed new Studio result can be published as such even when an old figure cannot support a same-machine speedup. Explicitly labeled M3 assignment studies stay intact.

## Batch, GPU encoding and BM25 rerun worksheet

Scope: `docs/performance.md` → **Batch mode performance**, **GPU-accelerated batch encoding** (both child headings), and **BM25-only batch mode** (both child headings). Excludes **Batch graph optimizer** and its M3 evidence, plus live/soak/remote-encoder measurements. This is a source-inspection worksheet, not a benchmark report: **no benchmarks, builds, or data generation were run**. Empty new-result cells are intentional; `—` is a published missing measurement, not zero.

### Provenance and comparison rules

- **M3-attributed:** the overview says “Benchmarked on Apple M3 MacBook Air, `all-MiniLM-L6-v2` model, `encoder_pool_size: 4`.” This attributes the opening batch table to M3, not M1 Ultra.
- **Explicit M1 Ultra:** GPU headings/sweep specify 20 cores, 64 GPU cores, 64 GB; the unblocked BM25 comparison specifies Apple M1 Ultra (20 cores, 64 GB).
- **Uncertain:** blocked BM25 tables have no local hardware declaration. The overview implies M3, but `benchmarks/results/2026-11-17-m3-before-local-reassignment.md` explicitly distinguishes an earlier M1 Ultra BM25-only timing record without identifying these particular table cells. Do not silently relabel these tables as confirmed M1 or confirmed M3.
- **Historical projections:** GPU table encoding times and blocked 1M “Projected full run” are projections, not measured full-job elapsed times. Retain the published figures verbatim; do not substitute arithmetic-derived estimates or treat them as rerun targets already verified.
- Record rerun commit, exact binary/features, OS/hardware/RAM, Rayon/inference environment overrides, model identity, dataset hashes, actual cache state, assignment strategy, repetitions, wall time and peak RSS. Keep historical comparison settings separate from current-default measurements.

### `docs/performance.md` → Batch mode performance

Published conditions: both backends use `country_code` blocking; 10k x 10k means 10,000 records per side, **zero initial crossmappings**. Cold builds encode/save indices; subsequent runs reuse cached indices. M3-attributed, not an existing M1 baseline.

| Published metric | Published flat 10k x 10k | New M1 flat 10k | Published usearch 10k x 10k | New M1 usearch 10k | Published flat 100k x 100k | New M1 flat 100k | Published usearch 100k x 100k | New M1 usearch 100k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Index build time (no cache) | ~17s | 5.5 s | ~17s | 5.6 s | ~3m | 51.1 s | ~3m 32s | 64.4 s |
| Index load time (cached) | ~47ms | 35.5 ms | ~78ms | 45.6 ms | ~650ms | 360.8 ms | ~235ms (A) + ~261ms (B) | 447.1 ms |
| Scoring throughput | 5,507 rec/s | 1,678 rec/s | **33,738 rec/s** | 45,370 rec/s | — | 182 rec/s | **10,539 rec/s** | 31,094 rec/s |
| Wall time (cold) | — | 11.507 s | — | 6.470 s | — | 538.145 s | 3m 32s | 71.741 s |
| Wall time (warm) | 2.2s | 6.510 s | 0.3s | 0.787 s | — | 553.018 s | **9.5s** | 6.471 s |

New figures are M1 Ultra with the graph default. Index times sum the logged A and B phases for each run, then take the median where repeated. Build includes encoding and index construction, not cache saving. Scoring throughput is from the warm runs. All 10k and warm 100k values are medians of three runs; cold 100k values are single runs. Index timings were verified against the original Studio `standard/` logs over SSH.

Related published claim (not a table measurement): `performance.quantized: true` “roughly double[s] encoding speed.” New M1 quantized/unquantized comparison: __________.

#### Harness/config mapping and preparation

Commands below are **future reproduction instructions**, from project root, not commands executed for this worksheet.

```bash
cargo build --release --features usearch
# If data is absent; generation needs faker, pandas, pyarrow:
python3 -m pip install faker pandas pyarrow
bash benchmarks/batch/generate_data.sh 10k 100k

python3 benchmarks/batch/10kx10k_flat/cold/run_test.py --binary ./target/release/meld
python3 benchmarks/batch/10kx10k_flat/warm/run_test.py --binary ./target/release/meld
python3 benchmarks/batch/10kx10k_usearch/cold/run_test.py --binary ./target/release/meld
python3 benchmarks/batch/10kx10k_usearch/warm/run_test.py --binary ./target/release/meld
python3 benchmarks/batch/100kx100k_usearch/cold/run_test.py --binary ./target/release/meld
python3 benchmarks/batch/100kx100k_usearch/warm/run_test.py --binary ./target/release/meld
```

- Each listed directory has `config.yaml` and `run_test.py`; use those exact adjacent configs. Inputs are `benchmarks/data/dataset_{a,b}_10k.csv` and `dataset_{a,b}_100k.csv`. IDs are `entity_id` / `counterparty_id`; blocking is A `country_code` against B `domicile`.
- Shared basic embedding case: `all-MiniLM-L6-v2`; embedding legal-name weight 0.55, fuzzy short-name `partial_ratio` 0.20, country exact 0.20, LEI exact 0.05; auto/review thresholds 0.85/0.60; `encoder_pool_size: 4`, `top_n: 20`, `ann_candidates: 20`. Flat/usearch differs by vector backend.
- **Missing harness:** no `benchmarks/batch/100kx100k_flat/` cold/warm config or script was found. Its published build/load figures cannot be reproduced by an existing named flat-100k harness. Prepare an isolated 100k flat derivative with explicit input/cache/crossmap/output paths before filling those cells; do not confuse usearch with flat or invent a command to a nonexistent script.
- Cold wrappers delete their own cache/output/crossmap. Warm wrappers preserve their own cache, clear output/crossmap, and require a preparation run if empty. **Cold and warm configs use different cache directories**: running cold then warm does not by itself warm the warm directory. Run each warm script once for preparation, then again for measurement (or document a verified cache copy). This contradicts the full-suite doc/script claim that cold automatically prepares the following warm run.
- `python3 benchmarks/batch/run_all_tests.py --binary ./target/release/meld` exists, but is broader than this table, omits flat 100k, SQLite BM25 and 1M cases, and includes quantized/f16/mmap/mixed/remote cases. Its timing regexes expect historical log strings; preserve raw logs and verify extracted metrics rather than trusting empty summary cells.
- Extra relevant case: `benchmarks/batch/100kx100k_usearch_quantized/cold/{config.yaml,run_test.py}` for the quantization claim. f16 and mmap warm variants also exist but are not columns of the published opening table.
- Pure-Rust flat build is available with `cargo build --release --no-default-features`; record this feature change if used rather than assuming parity with the default-feature binary.

### `docs/performance.md` → GPU-accelerated batch encoding

Published context: at **1M x 1M**, production scoring encoding accounts for **~72%** of total wall time. New measured encoding share: __________. Parallel ONNX sessions via Rayon apply to CPU and GPU. This paragraph's percentage has no independent local machine declaration; the following measurements explicitly identify M1 Ultra.

#### Encoding throughput (M1 Ultra, 20 cores, 64 GPU cores, 64 GB)

Published workload: **1M x 1M cold batch**, `all-MiniLM-L6-v2`, production scoring `(name_emb 0.30, addr_emb 0.20, BM25 0.50, synonym 0.20)`.

| Configuration | Published encoding rate | New M1 encoding rate | Published vs baseline | New vs baseline | Published projected 1M x 1M encoding time | New projection (label as projected) | New measured encoding elapsed |
|---|---:|---:|:---:|---:|---:|---:|---:|
| Sequential CPU (pool=1) | 210 rec/s | | 1.0x | | ~79 min | | |
| Parallel CPU (pool=8) | 369 rec/s | | 1.8x | | ~45 min | | |
| **GPU CoreML (pool=12, batch=256)** | **1,828 rec/s** | | **8.7x** | | **~18 min** | | |

Do not reconcile the published time/rate arithmetic by editing historical values. The table does not specify a consistent record/text/side denominator or explain all projected durations. Capture total records per side, embedding fields/texts encoded, cache misses, batching, encoding phase boundaries and denominator on rerun.

#### Optimal GPU settings by pool size

Explicitly “Sweep results at 1M scale on M1 Ultra.”

| pool | batch | Published rec/s | New M1 rec/s | Published Notes | New measured encoding elapsed / memory / provider notes |
|-----:|------:|------:|------:|---|---|
| 12 | 256 | **1,828** | | Best overall | |
| 16 | 128 | 1,718 | | | |
| 8 | 256 | 1,677 | | | |
| 12 | 128 | 1,590 | | | |
| 4 | 256 | 1,152 | | GPU underutilised | |
| 12 | 512 | 765 | | GPU memory pressure | |
| 16 | 512 | 473 | | GPU memory pressure | |

Published tuning rule: `encoder_pool_size` ≈ **60%** of CPU cores, `encoder_batch_size: 256`, avoid pool × batch products above **~3,000**. New M1 conclusion (not assumed): __________.

#### GPU reproduction: existing base, missing sweep harness

```bash
cargo build --release --features usearch,gpu-encode
brew install onnxruntime
# Only for a nonstandard installation:
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
bash benchmarks/batch/generate_data.sh 1M
python3 benchmarks/batch/1Mx1M_usearch/cold/run_test.py --binary ./target/release/meld
```

- Base config: `benchmarks/batch/1Mx1M_usearch/cold/config.yaml`; wrapper supports `--binary` and optional `--limit N`. `--limit` restricts B scoring, not a verified full-scale encoding/sweep equivalent; label partial runs explicitly. Warm counterpart: `benchmarks/batch/1Mx1M_usearch/warm/{config.yaml,run_test.py}` (not the cold GPU workload).
- Base cold config matches the four production weights above, blocks country/domicile, uses `top_n: 5`, `ann_candidates: 50`, auto/review 0.88/0.60. **It currently has pool 4, batch 256 and commented-out `#encoder_device: gpu`: the command above as shipped is CPU, not the published GPU configuration.**
- No dedicated GPU pool/batch sweep or CPU pool=1/8 benchmark harness/config was found under `benchmarks/`. Future isolated config variants must explicitly set `performance.encoder_device: cpu` or `gpu`, pool=1/8/12/etc., and batch size for every row; CPU historical batch sizes are unspecified. Existing wrapper has no device/pool/batch override flags and hardcodes its config path. Use isolated working-layout copies or invoke `meld run --config /path/to/variant.yaml --verbose` after resetting that variant's private cache/crossmap/output. Do not pass invented sweep flags to the wrapper.
- Cold wrapper parses embedding-build and scoring phases and prints wall time, but does not implement the published rec/s denominator, repetitions, sweep, or peak-RSS collection. Preserve raw A/B encode timings and separately measure RSS. Wrapper warns of ~30–40 GB memory at 1M; this is a harness warning, not a new measurement.
- `docs/building.md` → **GPU encoding**, **macOS (Apple Silicon)**, **Configuration** documents the release feature build, Homebrew runtime discovery and explicit dylib override; sample config is GPU/pool 12/batch 256. Confirm the selected CoreML provider and any fallback/error in logs; CoreML may dispatch across CPU/GPU/Neural Engine, not exclusively GPU cores. GPU is batch-only. Linux CUDA setup is not an M1 reproduction route.

### `docs/performance.md` → BM25-only batch mode

Published general setup: BM25 + fuzzy + exact, country-code blocking; WAND skips noncompetitive documents in large eligible pools. Unblocked optimization removes per-query eligibility-set construction; small pools stay exhaustive, large pools use WAND; no new config switch beyond `blocking.enabled`.

#### Unblocked batch BM25 comparison — explicit M1 Ultra

Published: against commit **`877f261`**, Apple M1 Ultra (**20 cores, 64 GB**), default-feature release builds, **four Rayon threads**, **median of three runs**. Every case has **1,000 B queries**, not a square workload; selective queries have unique tokens, common queries also have two tokens in every A record. Both binaries produced identical CSVs every run: **900 matches**, **100 unmatched queries**.

| A pool | Query type | Published scoring before → after | New scoring before → after | Published wall before → after | New wall before → after | New matched / unmatched / CSV equivalence |
|---|---|---:|---:|---:|---:|---|
| 10,000 | Selective | 0.321s → 0.011s | | 0.511s → 0.508s | | |
| 10,000 | Common terms | 0.957s → 0.887s | | 1.124s → 1.066s | | |
| 100,000 | Selective | 3.968s → 0.014s | | 4.509s → 0.569s | | |
| 100,000 | Common terms | 13.333s → 8.065s | | 13.957s → 8.660s | | |

Scoring times are reconstructed from rounded CLI records/second; duration display rounds to **0.1s**. Wall includes startup/index construction/exports. Synthetic bookkeeping isolation is not a production speedup guarantee; common terms still pay posting-list copying/search costs.

```bash
# Prepare separate pinned default-feature release binaries first; baseline is 877f261.
# Harness does not build either binary. Preserve the baseline outside target/release.
python3 benchmarks/batch/unblocked_bm25/run_test.py \
  --baseline /path/to/baseline/meld --candidate ./target/release/meld \
  --pool-sizes 10000 100000 --queries 1000 --repeats 3 --threads 4 \
  --output /tmp/unblocked-bm25-comparison.json
```

`benchmarks/batch/unblocked_bm25/run_test.py` creates its own synthetic inputs, private run configs/crossmaps/outputs/logs under temporary storage; no adjacent checked-in `config.yaml` or large-data generation is needed. Generated config explicitly selects **greedy**, `blocking.enabled: false`, BM25 weight 1.0 on `text_a`/`text_b`, auto/review 0.8/0.5, `top_n: 1`, `bm25_candidates: 10`; no embedding match fields, exact/common-ID shortcut, or persistent indices. Script sets `RAYON_NUM_THREADS` from `--threads`, compares all output columns/rows order-independently, and retains per-run logs/JSON. `--candidate` can be omitted for baseline-only validation; `--timeout` defaults to 600 seconds per subprocess. Pin and record the new candidate commit: this is not an unspecified baseline-versus-latest historical claim.

#### Blocked batch benchmarks — uncertain hardware attribution

Published **10k × 10k**:

| Metric | Published In-memory | New M1 In-memory | Published SQLite | New M1 SQLite |
|---|---:|---:|---:|---:|
| Bulk load | — | | 170–194K rec/s | |
| Scoring throughput | **49,337 rec/s** | | 2,099 rec/s | |
| Auto-matched | 7,824 | | 7,931 | |
| Peak memory | ~2 GB | | ~1.2 GB | |

Published **1M × 1M (in-memory, `bm25_candidates: 10`)**:

| Metric | Published Value | New M1 result |
|---|---:|---:|
| Scoring throughput | **1,062 rec/s** | |
| Projected full run | ~16 minutes | |
| Measured full-run wall time (not published) | — | |

Other published scale/memory assertion (not a table measurement): **at 55M scale**, in-memory **~15-20 GB**, SQLite **~10-12 GB**. New in-memory peak RSS: __________; new SQLite peak RSS: __________. No matching 55M harness/config or measurement method was found; keep this unverified scale claim separate from the 1M wrapper's ~15–20 GB warning.

```bash
cargo build --release
python3 benchmarks/batch/10kx10k_bm25only/cold/run_test.py --binary ./target/release/meld
python3 benchmarks/batch/1Mx1M_bm25only/cold/run_test.py --binary ./target/release/meld
# SQLite has a config but NO run_test.py; prepare clean private artifacts first:
./target/release/meld run \
  --config benchmarks/batch/10kx10k_bm25only_sqlite/cold/config.yaml --verbose
```

- In-memory configs are adjacent to the two listed scripts. SQLite config: `benchmarks/batch/10kx10k_bm25only_sqlite/cold/config.yaml`, with `batch.db_path` pointing to that case's disposable `batch.db`. Manually invoked SQLite does **not** automatically clear its existing crossmap/output/database; use an isolated layout and fresh scratch paths for every repetition. Never delete user artifacts to prepare a run.
- 10k configs: BM25 0.55, fuzzy short-name partial_ratio 0.20, country exact 0.20, LEI exact 0.05, country/domicile blocking, thresholds 0.85/0.60, `top_n: 20`, `bm25_candidates: 20`. Pool sizes differ (in-memory 4, SQLite 1); no embedding match fields. 1M config uses the same weights, explicit BM25 name fields (`legal_name` and `short_name` against `counterparty_name`), pool 1, `top_n: 5`, `bm25_candidates: 10`. This is not an otherwise identical 10k-versus-1M config.
- 1M wrapper takes `--limit N`; a limited score sample is not a measured full run and any extrapolation belongs in the projection cell. Generate missing `dataset_{a,b}_1000k.csv` with `bash benchmarks/batch/generate_data.sh 1M` (`--addresses` is included by that script). Cold wrappers clear private cache/output/crossmap; add external RSS capture and retain bulk-load logs for the published memory/load rows.
- **Stale feature instructions:** BM25 wrappers and suite refer to `cargo build --release --features bm25` / `--bm25-binary`; `Cargo.toml` has **no `bm25` feature**. BM25 is built in: use `cargo build --release` (or record another legitimate feature set). `--bm25-binary` remains a suite binary-path option, not evidence that the Cargo feature exists.
- **Stale engine explanation:** the blocked-results prose says the in-memory fast path “queries Tantivy”; current project architecture uses `SimpleBm25`/WAND, not Tantivy. Preserve published figures as historical rather than asserting a current-engine explanation.
- **Changed assignment default:** basic batch, GPU-base and blocked BM25 configs omit assignment selection. Current code defaults to graph/approximate; historical throughput/auto-match counts are not guaranteed to be comparable under this changed default. For historical greedy comparisons use explicitly labelled private configs with `batch.assignment.strategy: greedy` and no `mode`; record a separate current-default result if desired. This worksheet does not import or rerun the M3 graph benchmarks. Unblocked comparison already pins greedy.

### Repeated documentation and coverage notes

- Repository Markdown searches for these distinctive published numbers/table labels found **no duplicate batch/GPU metric tables in other docs**. `docs/performance.md` remains the source of all metric cells above.
- `docs/configuration.md` → **Performance field reference** repeats the GPU tuning guidance (~60%, batch 256, product ~3,000), defines device/pool/batch settings, and lists default batch sizes **64 CPU / 256 GPU**. These defaults do not establish the historical CPU table's actual batching.
- `docs/building.md` → **GPU encoding** repeats GPU setup and pool 12/batch 256 sample configuration, not measured throughput tables. Its **Benchmark data** section and `benchmarks/batch/generate_data.sh` distinguish committed 10k data from larger generated data. Generator prep requires third-party packages even though benchmark runners themselves use Python's standard library.
- `docs/performance.md` → **Benchmarking** publishes **45–60 minutes** for all batch tests and **~3.5 minutes each** for 100k cold tests encoding **200,000 records**. These are historical planning estimates, not measured new M1 results. New batch-suite elapsed: __________; new 100k cold elapsed: __________. The full-suite cache-preparation claim is not valid for the separate configured cold/warm cache paths noted above.

## M3 MacBook Air live rerun worksheet

Live results are pending an M3 MacBook Air rerun. The published figures, runner paths, and destinations below are retained for comparison; the new result cells are intentionally blank. Record the OS, revision, build features, model and cache identity, and run count for the M3 session. The soak remains historical only.

### Live mode performance — destination table and adjacent prose

Destination: `docs/performance.md` → **Live mode performance**, first table (10k × 10k).

Completed on the M3 MacBook Air (AC), default release `3e3dc2f`, greedy. Seven workloads × three repetitions, 450,000 total requests and zero HTTP/application errors. Seed 42, ten clients, 80% encoding; warm starts copy pristine seed indexes (and seed DB for SQLite). Medians below exclude startup except readiness rows. From shutdown to next launch the shortest gap was 120.006 s across 25 server sessions including four cache preparations. Flat runs logged missing-vector warnings; these are speed measurements, not an accuracy audit. Driver, timings, raw requests, configs and logs: `/tmp/melder-air-live-tables.b8atT0/`.

| Case | Metric | Published old | New M3 Air |
|---|---|---:|---:|
| flat c=10 cold | Throughput | 843 req/s | 740 req/s |
| usearch c=10 cold | Throughput | 1,045 req/s | 1,093 req/s |
| flat c=10 warm | Throughput | 1,113 req/s | 731 req/s |
| usearch c=10 warm | Throughput | 1,558 req/s | 1,166 req/s |
| flat c=10 cold | p50 | 8.4ms | 10.5 ms |
| usearch c=10 cold | p50 | 5.5ms | 8.5 ms |
| flat c=10 warm | p50 | 7.2ms | 10.0 ms |
| usearch c=10 warm | p50 | 3.5ms | 7.8 ms |
| flat c=10 cold | p95 | 30.4ms | 33.7 ms |
| usearch c=10 cold | p95 | 29.0ms | 20.3 ms |
| flat c=10 warm | p95 | 21.2ms | 35.9 ms |
| usearch c=10 warm | p95 | 25.6ms | 18.7 ms |

| Destination prose claim | Published old | New M3 Air |
|---|---:|---:|
| Warm startup | ~1.7s | 0.36 s flat / 0.33 s USearch |
| Cold startup | ~18s | 12.6 s flat / 13.4 s USearch |
| 100k × 100k, usearch warm, c=10, 10k events, 80% encoding: throughput | 1,325 req/s | 991 req/s |
| Same: p50 | 6.0ms | 9.3 ms |
| Same: p95 | 19.0ms | 20.4 ms |

Verified 10k wrappers (each has a sibling `config.yaml` used verbatim):

```text
benchmarks/live/10kx10k_inject3k_flat/cold/run_test.py
benchmarks/live/10kx10k_inject3k_usearch/cold/run_test.py
benchmarks/live/10kx10k_inject3k_flat/warm/run_test.py
benchmarks/live/10kx10k_inject3k_usearch/warm/run_test.py
```

Run individually as `python3 <wrapper> --binary ./target/release/meld`. Each wrapper accepts `--binary`, not `--iterations`, `--concurrency` or `--encoding-pct`. It starts `<binary> serve --config <sibling config.yaml> --port 8090`, then calls:

```bash
python3 benchmarks/scripts/live_concurrent_test.py --no-serve --port 8090 \
  --iterations 3000 --concurrency 10 \
  --a-path benchmarks/data/dataset_a_10k.csv \
  --b-path benchmarks/data/dataset_b_10k.csv
```

Default helper seed is 42 and encoding percentage 80. **3000 is the TOTAL number of requests across ten workers**, not 3000 per client. The performance page says 30k total for c=10; this conflicts with the checked-in wrappers/helper. Do not silently claim a 30k rerun. Either benchmark the verified 3000-total harness and correct the description, or explicitly use `--iterations 30000` in a separately controlled helper run. No c=1 measurements appear in the published table.

Cold wrappers delete cache/output/WAL/crossmap before startup. Warm wrappers preserve cache but clear output/WAL/crossmap; first pass with absent cache is not a warm measurement. Preserve pristine cache bytes separately for matched repetitions, since serving can mutate caches. Startup and injection throughput are separate measurements; readiness polling is not a precise server startup measurement.

10k live configs: `all-MiniLM-L6-v2`, pool 4, vector index `load`, AND country_code↔domicile blocking, name embedding 0.55 + short-name `partial_ratio` fuzzy 0.20 + country exact 0.20 + LEI exact 0.05; auto/review thresholds 0.85/0.60; `top_n: 5`, `ann_candidates: 5`, `skip_initial_match: true`. Flat/usearch changes the backend, not this scoring mix.

100k prose maps to `benchmarks/live/100kx100k_inject10k_usearch/warm/run_test.py` and its sibling `config.yaml`. Same helper arguments except `--iterations 10000` and dataset paths `benchmarks/data/dataset_a_100k.csv`, `benchmarks/data/dataset_b_100k.csv`; c=10, port 8090. **Current 100k config uses encoder_pool_size: 6**, not the overview's 4. Use/report the actual config; do not substitute the separate `_usearch_bm25` workload.

### SQLite live mode — destination table and claims

Destination: `docs/performance.md` → **SQLite live mode**, entire table and following comparison prose. Historical machine: M3 MacBook Air by page overview.

| Metric | Published old in-memory | New M3 Air in-memory | Published old SQLite | New M3 Air SQLite |
|---|---:|---:|---:|---:|
| Throughput | 1,698 req/s | 1,214 req/s | 1,395 req/s | 903 req/s |
| p50 | 3.4ms | 7.9 ms | 6.4ms | 10.3 ms |
| p95 | 23.9ms | 18.3 ms | 13.6ms | 19.6 ms |
| p99 | 37.3ms | 25.5 ms | 20.3ms | 24.8 ms |
| Throughput penalty, SQLite vs memory | — | reference | ~18% slower | 25.6% fewer req/s |
| Restart/startup comparison | ~18s in-memory cold | 0.36 s warm (13.4 s cold USearch) | ~0.4s SQLite warm | 0.53 s warm |

Verified SQLite warm wrapper: `benchmarks/live/10kx10k_inject3k_usearch_sqlite/warm/run_test.py`; config: same directory `config.yaml`. Despite `inject3k` in the path, **ITERATIONS=10000**, c=10; helper call is the 10k-data command above with `--iterations 10000`. Config matches ordinary 10k usearch scoring/pool=4 and adds `live.db_path: benchmarks/live/10kx10k_inject3k_usearch_sqlite/warm/live.db`, `sqlite_cache_mb: 64`. Wrapper preserves **both DB and embedding cache**, clears WAL/crossmap/output. First pass creates/populates DB and injects events; next pass measures restart on a DB already changed by the first pass. Distinguish that durable restart experiment from an equal-starting-state storage-throughput comparison. The wrapper's 2-second readiness polling cannot directly resolve a ~0.4s startup claim; capture server-side load timings or an appropriately precise external timer.

There is **no exact 10k-events ordinary in-memory warm wrapper** at `10kx10k_inject3k_usearch/warm`: it runs 3000. For a matched 10k-event comparison, use that memory config with a controlled fresh-state server and the helper's explicit `--iterations 10000 --concurrency 10 --encoding-pct 80 --seed 42` plus the same 10k CSV paths. Document/reset state and preserve caches explicitly; direct helper startup performs no wrapper cleanup. Do not substitute the 100k-pool wrapper. SQLite cold wrapper exists at `benchmarks/live/10kx10k_inject3k_usearch_sqlite/cold/run_test.py` but uses 3000 requests and is not the published warm comparison.

### Low-encoding-ratio performance — destination table and prose

Destination: `docs/performance.md` → **Low-encoding-ratio performance**, entire 40% table; inherited historical M3 attribution.

| Metric (40% encoding) | Published old flat c=10 | New M3 Air flat | Published old usearch c=10 | New M3 Air usearch |
|---|---:|---:|---:|---:|
| Throughput | 890 req/s | | 2,474 req/s | |
| p50 | 10.1ms | | 2.4ms | |
| p95 | 21.0ms | | 11.1ms | |
| Non-encoding completion claim (prose, not per-backend table value) | under 1ms | | under 1ms | |

**No dedicated 40%-encoding wrapper/config was found.** The ordinary wrappers do not forward encoding percentage. Verified helper supports `--encoding-pct 40`. A reproducible proposed run (not evidence of the historical run's unspecified cache state/request count) uses each ordinary 10k warm config, with separately prepared/reset state:

```bash
python3 benchmarks/scripts/live_concurrent_test.py --binary ./target/release/meld \
  --config benchmarks/live/10kx10k_inject3k_flat/warm/config.yaml \
  --port 8090 --iterations 3000 --concurrency 10 --encoding-pct 40 --seed 42 \
  --a-path benchmarks/data/dataset_a_10k.csv --b-path benchmarks/data/dataset_b_10k.csv
```

Repeat with `benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml`. Or start a controlled server and replace binary/config arguments with `--no-serve`. Helper splits encoding traffic equally across four operation types, remainder equally across two non-encoding types: at 40%, 10% each new A/new B/embedding update A/embedding update B, 30% each non-embedding update A/B. At 80%, this is 20% each encoding type and 10% each non-encoding type; the helper's older descriptive 30/30/20/20 text does not reflect that implementation. Prebuild the Rust binary: helper's missing-binary auto-build branch incorrectly invokes `go build`.

### Soak — historical reference only (excluded from reruns)

Destination: `docs/performance.md` → **Soak test — stability under sustained load** → **Results summary**, **Dashboard**, and associated narrative/image `docs/img/soak_dashboard.png`. Old hardware is explicitly **M1 Ultra (20 cores, 64 GB RAM)**, not M3. The published image was inspected; its title/counts and all four chart annotations match the inventory below (including crossmap annotation `mean 4314 (range 210)`). Repository copy: `benchmarks/soak/10kx10k_usearch/soak_dashboard.png`.

| Destination/metric | Exact published old | New M1 Ultra |
|---|---|---|
| Summary + dashboard title: duration | 2 hours (image: 2h run) | |
| Summary + dashboard title: bursts | 71 | |
| Summary + dashboard title: injections | 1,849,542 | |
| Summary + dashboard title: final records A+B | 482,378 (241k per side, up from 10k base) | |
| Summary + dashboard title: errors | 0 | |
| Summary + throughput chart annotation | ~383 req/s | |
| Summary: memory growth | 714 → 2,035 MB (~3.0 KB per new record) | |
| Summary: disk growth | 54 → 485 MB (~1.0 KB per new record) | |
| Summary + crossmap chart annotation | mean 4,314 (range 210) | |
| Throughput narrative: initial burst | 577 req/s | |
| Throughput narrative: observed total-pool growth | 31k → 482k records (15× increase) | |
| Memory chart annotation | ~3.0 KB/record | |
| Memory narrative: RSS dip position | around 130k records | |
| Disk chart annotation | ~1.0 KB/record | |
| Crossmap narrative: confirmed base pairs | ~4,300 | |
| Crossmap chart legend/narrative: moving window | 10 bursts (image: Moving avg (w=10)) | |
| Crossmap narrative: moving-average stability | within 1.5% of mean across full 2-hour run | |

Published setup: base 10k × 10k; usearch; `all-MiniLM-L6-v2`; embedding 0.55 + address BM25 0.40 + exact LEI 0.05; AND country_code/domicile blocking; ten HTTP workers; pool 4; ~25% new/~75% upserts; sleep 10–60 seconds. New synthetic records have no true opposite-side counterpart; auto-match threshold 0.85. These are setup values, not new measurements.

Verified entry/config/helper:

```text
benchmarks/soak/10kx10k_usearch/run_test.py
benchmarks/soak/10kx10k_usearch/config.yaml
benchmarks/soak/10kx10k_usearch/live_concurrent_test.py  (LOCAL helper, not benchmarks/scripts/)
```

Exact explicit controller invocation matching published duration and current defaults:

```bash
python3 benchmarks/soak/10kx10k_usearch/run_test.py --binary ./target/release/meld \
  --duration 2 --min-burst 3000 --max-burst 50000 \
  --min-sleep 10 --max-sleep 60 --concurrency 10 --port 8090 --seed 42
```

Default duration without the flag is **12 hours**, not two. Controller deletes cache/output/WAL/crossmap, extra-ID files and previous soak CSV; starts cold using its sibling config. For each randomized burst it calls the local helper with:

```text
--no-serve --port 8090 --iterations <burst_size> --concurrency 10
--a-path benchmarks/data/dataset_a_10k.csv --b-path benchmarks/data/dataset_b_10k.csv
--encoding-pct 50 --seed <42 + burst_number>
--extra-ids-a benchmarks/soak/10kx10k_usearch/extra_ids_a.txt
--extra-ids-b benchmarks/soak/10kx10k_usearch/extra_ids_b.txt
--a-counter-offset <cumulative new-A count> --b-counter-offset <cumulative new-B count>
```

Implemented 50%-encoding split makes ~25% new records total, ~75% upserts; controller comments claiming 50% new are stale. ID offsets and extra-ID lists grow across bursts. Duration is checked between bursts; a final burst can overrun the requested two hours. Preserve actual elapsed time, not just requested duration.

Artifacts: `benchmarks/soak/10kx10k_usearch/soak_log.csv`, `/tmp/meld_soak_<pid>.log`, cache/WAL/crossmap and helper stdout. RSS and disk values labelled MB in the script are calculated using **1024² bytes** (MiB); preserve exact old labels above, but state the rerun unit definition. Disk accounting sums cache files + WAL files + crossmap CSV, not total filesystem consumption. Capture throughput per burst, errors, health records/crossmap and resource readings from CSV; recompute the moving average, slopes and ranges. Do not infer scoring accuracy from stable CrossMap counts alone, or repeat the historical OS-compression explanation as verified causality without evidence. No dashboard-generation wrapper was verified; publication needs an explicitly recorded plotting step and refreshed image, not stale annotations.

### Deferred acceptance — separate M3-only historical evidence

Destination: `docs/performance.md` → **Deferred acceptance performance**, entire table. **Do not relabel these as M1 Ultra measurements or roll them into ordinary live baselines.** Supporting source: `benchmarks/assignment_live/official-warm-report.md` (2026-11-18), `official-warm-measurements.json`. Three repetitions per workload/mode on Apple M3, 16 GiB RAM, AC power; in-memory USearch, ten clients, seed 42, identical prepared cache bytes; median of per-run statistics, not pooled requests. Request time excludes startup/shutdown.

| Workload | Mode | Published old throughput | New M3 Air | Published old request run time | New M3 Air | Published old p95 | New M3 Air |
|---|---|---:|---:|---:|---:|---:|---:|
| 10k × 10k, 3,000 requests | Main (greedy) | 1,601 req/s | | 1.87 s | | 20.7 ms | |
| Same | New build, off | 1,592 req/s | | 1.89 s | | 20.2 ms | |
| Same | New build, on | 785 req/s | | 3.82 s | | 23.6 ms | |
| 100k × 100k, 10,000 requests | Main (greedy) | 1,289 req/s | | 7.76 s | | 22.3 ms | |
| Same | New build, off | 1,263 req/s | | 7.92 s | | 22.5 ms | |
| Same | New build, on | 574 req/s | | 17.42 s | | 30.3 ms | |

| Destination prose | Published old M3 evidence | New M3 Air |
|---|---|---|
| Requests/error result | All 117,000 requests completed without errors | |
| Off-mode final pairings | Matched main in every run | |
| Enabled displacements/run, 10k / 100k | one / six | |
| Startup comparison | essentially unchanged | |

The historical comparison reused the ordinary 10k/100k USearch warm wrappers unchanged in isolated workspaces; enabled YAML alone added `live.assignment.strategy: deferred_acceptance`. Main was pinned `b9a83a4`; both new modes shared one post-review-fix release binary. Raw orchestration/logs were recorded under `/tmp/melder-live-before-after.AhM33r/`, not a verified durable checked-in reproduction wrapper for this three-mode comparison. Existing ordinary wrappers identify workloads but do not themselves reproduce pinned historical builds, cache copies, rotations and repetitions. A new before/after rerun requires that separate controlled orchestration; do not present the unrelated assignment helper workload as the exact source of these numbers. Enabled mutation serialization and reciprocal verification were not separately cost-attributed; displacement counts are not correctness gains.

## Documentation-site mirrors and measurements outside `docs/performance.md`

Inventory only; no benchmarks run and no repository files changed. Paths and line numbers refer to the inspected checkout. `—` below is an intentionally unfilled **new M1 Ultra result**, not zero or an inferred measurement. This section is intended for insertion into `benchmarks/results/m1-ultra-rerun.md`.

### Attribution and update rules

- No prose page outside `docs/performance.md` explicitly labels a timing/throughput table **M1 Ultra**. Do not convert unlabelled numbers or generic “Apple Silicon” expectations into historical M1 measurements.
- The exception at asset level is `docs/img/soak_dashboard.png`: its host performance section explicitly identifies the measured M1 Ultra. It must be tracked even though its only current embed is on the performance page.
- `docs/performance.md` begins with **Apple M3 MacBook Air, MiniLM, pool=4**. Its ordinary batch/live results and their mirrors are not M1 baselines. Explicit M1 subsections do not retroactively change that attribution.
- Fine-tuning research has **linked M1 provenance**, not hardware attribution in the public pages: `benchmarks/accuracy/science/AGENTS.md:36` states experiments run on a remote M1 Ultra. Keep this distinct from a per-run hardware manifest and from general accuracy examples.
- Prefer one authoritative measured table on the performance page, with mirrors linking to it. A new M1 column/result must remain distinct from the old M3 or unattributed figure. Historical comparisons requiring an old binary/config cannot be reconstructed from one current-binary run.
- Benchmark paths marked “candidate” identify relevant existing fixtures, not proven raw sources for the published number.

### Landing page, layouts, and image assets

| Exact location | Existing figures / content | Attribution and proposed treatment | New M1 result |
|---|---|---|---|
| `docs/index.html:104–239`, hero; **Two Modes**, **Quick Start**, **Scoring Methods**, **Documentation**, **Learn the Codebase** | No benchmark cards, throughput/latency table, M1 comparison, or accuracy percentage. Quick Start says “Build and run in under a minute.” Scoring table is descriptive, not measured. | Do not invent a landing-page benchmark replacement. The under-a-minute statement is an unmeasured onboarding promise, not benchmark evidence; qualify/remove separately if necessary. Performance card already links to `performance`. | Not applicable |
| `docs/_layouts/default.html`; `docs/_config.yml` | Presentation/navigation, no benchmark result table. | No measured mirror to update. | Not applicable |
| `docs/img/melder.svg` (landing logo) | Branding, not a benchmark chart. | No benchmark replacement. | Not applicable |
| `docs/img/soak_dashboard.png`, **Soak Test Dashboard — 10k × 10k usearch, 2h run**; embedded by performance **Soak test → Dashboard** | Title: **71 bursts; 1,849,542 injections; 482,378 stored records; 0 errors**. Panels: **Throughput vs Pool Size**, annotation **~383 req/s**; **Memory (RSS) vs Pool Size**, **~3.0 KB/record**; **Disk Usage vs Pool Size**, **~1.0 KB/record**; **Crossmap Stability (Accuracy Proxy)**, **mean 4314 (range 210)**, moving average **w=10**. | Explicit measured M1 via host **Test setup** (20 cores, 64 GB). Regenerate all four plots/title/annotations from a new soak CSV, not merely the summary prose. Retain the old image as historical evidence if not rerun. “Accuracy proxy” is membership-count stability, not labelled precision/recall. | New image/artifact: —; summary: cross-reference rerun report's canonical soak table |

Soak fixture/artifacts: `benchmarks/soak/10kx10k_usearch/run_test.py`, local `live_concurrent_test.py`, `config.yaml`, and generated `soak_log.csv`. The corresponding old prose summary also contains **2 h; 714→2,035 MB; 54→485 MB; initial 577 req/s; pool 31k→482k (~15×); 10-burst average within 1.5% of mean**. Those are the same run, not independent evidence. No other benchmark image was found in `docs/img/`.

### Public-page quantitative mirrors and standalone claims

| Exact page / section / table | Existing figures | Benchmark path / evidence | Attribution; new result or canonical cross-reference |
|---|---|---|---|
| `docs/live-mode.md:36–44`, **Storage backends**, opening paragraph | SQLite **~18%** lower throughput: **~1,395 vs ~1,698 req/s**, 10k, **c=10**; qualitative better SQLite p95/p99. | `benchmarks/live/10kx10k_inject3k_usearch_sqlite/warm/run_test.py` and in-memory `benchmarks/live/10kx10k_inject3k_usearch/warm/run_test.py`; preserve the documented **10k-event** comparison rather than assuming runner defaults are identical. | Exact mirror of performance **Live mode performance → SQLite live mode** table (M3 overview, not M1). New M3 Air values: use canonical SQLite-live comparison table; recompute percentage and tail-latency claim from it. |
| `docs/api-reference.md:289–304`, batch endpoints **Throughput**, table **Batch size / Throughput (rec/s) / Speedup vs single** | Batch **1: 221 / 0.9×; 10: 331 / 1.4×; 50: 445 / 1.8×; 100: 319 / 1.3×; 500: 325 / 1.3×**. Narrative: batch 50 sweet spot, latency **<200 ms**. | `benchmarks/scripts/live_batch_test.py`; caller example in performance **Manual test scripts**, using `benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml`. | Dataset stated as 10k×10k; hardware not attributed. Standalone table, not duplicated in performance. New M3 Air rec/s/speedup for each row: **—**; batch-50 latency: **—**. Measure the separate single-request baseline (batch=1 row is not the speedup denominator: it says 0.9×), and clarify whether “<200 ms” is batch or per-record latency before repeating. |
| `docs/remote-encoder.md:29–30`, **No, if**; `:87–89`, **How it works** | Local **~1,300 rec/s** versus remote **~50–200 rec/s**; second occurrence uses **requests/s / req/s**. | Local candidate: `benchmarks/live/100kx100k_inject10k_usearch/warm/`. Remote example explicitly linked later at `:366`: `benchmarks/batch/10kx10k_remote_encoder/cold/run_test.py`. No raw source provided for the “typical corporate service” range. | Unattributed local illustration plus generic remote-service estimate, not an M1-vs-remote controlled comparison. Cross-reference canonical local live table only if workload matches; otherwise new matched local/remote measurements **— / —**. Keep requests versus records units explicit; do not infer remote performance from local subprocess overhead. |
| `docs/vector-caching.md:14–16`, **Overview** | Encoding **10,000 records ~8 s**. | Candidate `benchmarks/batch/10kx10k_usearch/cold/run_test.py`; no exact raw source cited. | Unattributed single-side encoding example; do not equate to 20k two-side build/wall time. New M1 encode time **—**, or link canonical encoding table with model/side count. |
| `docs/vector-caching.md:46–49`, **Config options** | Cached index load **~170 ms for 100k records**. | Candidate `benchmarks/batch/100kx100k_usearch/warm/run_test.py`. | Unattributed; not the current performance table's **235 ms A + 261 ms B**. New M1 A/B load: cross-reference canonical batch table, specifying whether timing includes disk load, reconstruction, or full startup. |
| `docs/vector-caching.md:153–161`, **meld cache status**, sample output | **1 index file, 52.3 MB** at `benchmarks/batch/100kx100k_usearch/warm/cache`. | Same cache directory; sample CLI output, not a timing benchmark. | Unattributed illustrative footprint. New M1 cache size **—** only if refreshing the sample; do not call disk footprint a hardware speedup. |
| `docs/building.md:102`, **Feature flags**, `usearch` row | **Up to 5× faster at scale** vs flat. | Candidate paired flat/usearch fixtures under `benchmarks/batch/10kx10k_{flat,usearch}/{cold,warm}/`; 100k flat fixture is not currently present. | Unattributed generic upper-bound claim. Prefer link to performance **Batch mode performance** with measured workload-specific ratio; new ratio **—**. Do not conflate scoring throughput and wall time. |
| `docs/scoring.md:69–72`, **Embedding** | ONNX **~1–3 ms per text**. | Candidate cold encode/live fixtures; no exact benchmark source cited. | Unattributed explanatory estimate. New per-text M1 inference latency **—**, or canonical encoding table; batch throughput reciprocal is not single-call latency. |
| `docs/accuracy-and-tuning.md:382–409`, **Blocking and match field interaction** | Blocking removes **95%+** of pairs; measured **10× at 100k**; disabling blocking takes scoring **~8,500→~800 rec/s**. | Candidate `benchmarks/batch/100kx100k_usearch/{cold,warm}/` plus controlled config with blocking disabled; no retained paired raw source cited. `benchmarks/batch/unblocked_bm25/run_test.py` is a different BM25-only workload, not proof of these old embedding numbers. | Unattributed speed comparison, not explicitly M1. New matched blocked/unblocked scoring **— / —**, or cross-reference a canonical controlled comparison if added. Keep blocking's recall trade-off. |
| `docs/configuration.md:208–210`, **blocking**; `:743`, **Example config → What's happening** | Eliminates **95%+ / ~95%** of candidate pairs. | Example explicitly uses `benchmarks/batch/10kx10k_usearch/cold/config.yaml`. | Dataset-dependent illustration, not an M1 measurement. New fraction **—** only with candidate counts; otherwise qualify as illustrative. |
| `docs/configuration.md:608`, **performance**, `quantized` table row | INT8 encoding **~2× faster**, “negligible quality loss.” | Candidate `benchmarks/batch/100kx100k_usearch_quantized/cold/run_test.py`, compared with `100kx100k_usearch/cold/` under identical model/settings. | Unattributed/model-dependent; performance's opening tip also says roughly double. Cross-reference canonical quantized encoding comparison; new ratio **—**. Do not extend quality claim to Arctic-xs: linked science experiment 14 reports quality degradation (below). |
| `docs/configuration.md:555–568`, **What these settings usually save**; `:646`, **Performance field reference**, f16 row | Vector f16/bf16 footprint **40–50%** smaller; f16 table says **~43%**, negligible recall loss. | Candidate `benchmarks/batch/100kx100k_usearch_f16/warm/run_test.py` versus normal warm cache. | Explicit rules of thumb / unattributed footprint and quality claims, not M1 speed measurements. New disk/RSS/recall deltas **—** if measured; avoid making them guaranteed savings. |
| `docs/configuration.md:661–665`, **Performance field reference**, GPU tuning callout | Pool **~60% of CPU cores**, batch **256**, pool×batch above **~3,000** degrades throughput. | Candidate `benchmarks/batch/1Mx1M_usearch/cold/run_test.py` with controlled GPU pool/batch sweep. | Unattributed generalization that mirrors the M1-specific performance **GPU encoding → Pool size and batch size** sweep. Prefer link to that measured hardware/model-specific table rather than universal prescription; new recommended setting/ceiling: canonical sweep. |

Configuration memory estimates such as **~50–100 MB per encoder slot** (`:625`) and SQLite-cache arithmetic are configuration guidance, not labelled measured M1 results. Likewise remote startup “8 slots × 3 seconds → ~3 seconds” is a parallelism example, and handshake/shutdown timeouts are protocol settings, not benchmark observations.

### Interactive HTML course (`docs/codebase-course.html`)

These are visible numerical claims, not CSS animation timings. None identifies M1 Ultra; “Apple Silicon” is not an exact machine label.

| Exact section / location | Existing figures | Source fixture / proposed treatment | New M1 result |
|---|---|---|---|
| **Embeddings in Melder**, `:483–485` | CPU encoding **~5 ms/text**, **~80%** of request latency; other work **<2 ms combined**. | No exact raw source. Candidate live warm instrumentation. Conflicts in granularity/model with scoring's 1–3 ms estimate; link canonical measured latency decomposition if available, otherwise qualify. | Encode **—**; fraction **—**; rest **—** |
| **The Scoring Pipeline → One Pipeline to Rule Them All**, `:599` | Blocking removes **95%+** of candidates. | Same illustrative blocking claim as configuration; not independently measured hardware evidence. | Cross-reference blocking comparison |
| **The Text-Hash Skip**, `:740–741` | Skip **~4 ms** ONNX call; **~20% throughput gain** for updates to non-embedding fields. | Candidate `benchmarks/live/10kx10k_inject3k_usearch/warm/run_test.py` with controlled encoding ratios. Current performance **Low-encoding-ratio performance** compares workload mixes, not an optimization-disabled counterfactual. Do not use it as proof of the 20% gain. | Skip latency **—**; controlled gain **—** |
| **SimpleBm25: Lock-Free with WAND**, callout **Performance Win: 3.2x Throughput**, `:752–753` | Tantivy **461→1,460 req/s**, **3.2×**, ~40 dependencies removed. | No raw comparison path cited. Candidate live fixture cannot recreate retired Tantivy baseline without historical binary/config. Preserve as historical/unattributed or remove/link a historical report, not replace denominator with current performance. | Matched historical comparison **—** |
| Same section, `:770`; quiz `:976`; explanatory JS answer `:1180` | WAND processes **1–10%** / skips **90–99%** of documents or posting entries. | Candidate `benchmarks/batch/unblocked_bm25/run_test.py` with instrumentation. Algorithmic upper-bound correctness is separate from dataset-dependent pruning rate; documents and postings are not interchangeable. | Observed pruning **—** |
| **Scoring Log: Explainability**, `:890` | zstd **~8×**, **46 MB→5.6 MB** at 10k scale. | Candidates `benchmarks/live/10kx10k_inject3k_scoring_log/`, `benchmarks/enroll/10kx10k_enroll3k_scoring_log/`; no exact raw source cited. Size/compression comparison, not M1 runtime attribution. | Matched raw/compressed size **— / —**, ratio **—** |
| **See It In Action → Batch Mode: Process Everything At Once**, step 1, `:1027` | First embedding-cache build **~17 s**, second cached run **~0.3 s**. | Explicit `benchmarks/batch/10kx10k_usearch/warm/run_test.py`; paired cold fixture exists. Mirrors performance batch **index build time** and **warm wall time**, which are different metrics; do not call 0.3 s an index load time. | Cross-reference canonical cold-build / warm-wall table |
| Same section, step 3, `:1029` | All batch benchmarks **~45–60 min on Apple Silicon**. | `benchmarks/batch/run_all_tests.py`. Mirrors performance running-tests budget; suite membership/caches affect total. Generic estimate, not measured M1 suite duration. | Full-suite elapsed **—** only if actually measured |
| **See It In Action → Live Mode: Real-Time HTTP Matching**, step 2, `:1039` | **3,000 records**, **10 workers**, expected **~1,500 req/s on Apple Silicon**. | Explicit `benchmarks/live/10kx10k_inject3k_usearch/warm/run_test.py`. Approximate mirror of performance live 10k warm usearch **1,558 req/s** (under M3 overview). Confirm event-count conventions: performance describes 3k per client/30k total in its main table. | Link matching canonical live workload, not a differently sized request run |
| **What the Benchmarks Measure**, `:1091–1092` | Production target **1,000+ req/s** at 10k/side; p50 **<10 ms**. | Targets, not observed measurements. | No replacement result; label as targets |

### Accuracy and comparisons: do not silently relabel as measured M1

1. **No explicit public-site M1 accuracy/comparison table was found outside performance.** `docs/batch-mode.md:43–47`, **Why graph assignment is the default**, explicitly says **M3 MacBook Air**: **498,162 correct fixture pairs vs 456,571**, **63m09s vs 61m45s**, one run/mode, 1M×1M blocked BM25. This is a duplicate of performance **Batch graph optimizer**, not an M1 comparison. Fixture: `benchmarks/batch/1Mx1M_bm25only/cold/`; optimizer evidence: `benchmarks/assignment_solver/`. New M1 results, if collected, belong in a separate canonical graph comparison table (**—**), not in place of the M3 attribution. Batch mode's score example **1.84 vs 1.95** and **98.7%** bound sample are expressly examples, not measurements.

2. **Linked M1 research provenance merits a caveat, not automatic inclusion of every accuracy example.** `docs/fine-tuning.md:38–45`, **Production Result**, table **Config / Overlap / Recall**, states **Arctic-embed-xs R22 + 50% BM25 + synonym 0.20: 0.0003 / 100%**. It links `benchmarks/accuracy/science/experiments.md`; the science directory's instructions explicitly identify remote M1 Ultra execution. The page itself has no machine/per-run manifest. New fixed-model/fixed-holdout overlap/recall: **— / —** only if accuracy evaluation is in rerun scope. No retraining implied.

3. `docs/accuracy-and-tuning.md:723–730`, **Worked example → The progression**, table **Configuration / Overlap / Recall / Review queue / Review FPs**, reports:

   | Existing configuration | Overlap | Recall | Review queue | Review FPs |
   |---|---:|---:|---:|---:|
   | Off-the-shelf embeddings | 0.165 | 99.4% | 4,316 | 2,958 |
   | Fine-tuned (LoRA, 17 rounds) | 0.046 | 98.5% | 1,652 | 84 |
   | + BM25 at 20% | 0.005 | 99.2% | 1,662 | 2 |
   | + Synonym at 0.20, tuned thresholds | 0.003 | 100.0% | 221 | 4 |

   Corresponding narrative/score-distribution text appears in **Starting point**, **Fine-tuning**, **Adding BM25**, and **Adding synonym matching** (`:526–721`): **72% overlap drop, 0.165→0.046; ~89% drop, 0.046→0.005; 100% combined recall; 221 reviews (2.2%)**. Reference at `:810–814` points to science experiments. Treat as historical synthetic-data/model experiments with linked M1 instructions, not an exact per-row M1 hardware certification or a throughput comparison. The opening `meld tune` sample, other sample ground-truth output and model-size comparisons have no explicit M1 attribution. Preserve unless separately reproducing exact checkpoints/data/thresholds. Do not force the progression's **0.003** and Production Result's **0.0003** into one “corrected” figure: they describe different BM25 weights/configs.

4. Important linked accuracy qualification: science **Experiment 14: INT8 quantization of Arctic-embed-xs R22**, final **Configuration / Overlap / Combined recall / Review FPs / Model size** table, reports fp32 **0.0000 / 99.98% / 0 / 86 MB** versus INT8 synonym=0.20 **0.0132 / 100.00% / 26 / 22 MB**, and synonym=0.35 the same latter metrics. It explicitly rejects INT8 for that production configuration. This is linked historical research, not an additional site table, and contradicts interpreting the generic “negligible quality loss” phrase as universally proven. Keep any new M1 quantization speed result separate from quality; matching quality rerun cells remain **—**.

5. `docs/local-reassignment-productionization.md`, **Evidence available for the next session** (`:183–243`), is historical bounded-repair research, not M1 evidence or the current live assignment policy. It cites `benchmarks/assignment_stability/results/2026-11-17-study/report.md` (**463 vs 456 changed B; 1.44%/1.50% variation reduction; ~0.95%/1.22% overhead**), `benchmarks/assignment_competition/results/report.md` (**rank 2/3/5: 80.0/72.5/66.9 vs 100 fixture-correct; ~0.5–1.2% overhead**), and explicitly M3 `benchmarks/results/2026-11-17-m3-local-reassignment-trial.md` (**13.73 vs 15.40 s; ~8% paired / ~12% separate-median slowdown**). Preserve historical attribution; no M1 substitution.

6. `docs/batch-solver-library-assessment.md:85–100`, **Bounded performance measurements**, is explicitly **not a Melder benchmark**: third-party `netsimplex` vs `mcmf`, arm64 macOS, no M1 model specified. Table **Size / Edges / netsimplex solve / mcmf call**: **1k×1k/5k: 60.6/6.27 ms; 10k×10k/50k: 6.75 s/170 ms; 1k chain/1,999: 12.1/1.80 ms; 10k chain/19,999: 1.19 s/25.2 ms**. Sources are pinned external clones and temporary workspace `/tmp/melder-solver-assessment.zZoPlN/validation/`, not a repository benchmark runner. Different timing boundaries and known mcmf defects are documented. Exclude from Melder M1 rerun replacement.

7. `docs/building.md:293–299`, benchmark-data notes, links **Abt-Buy**, a third-party Leipzig product dataset (**1,081×1,092**), and `benchmarks/accuracy/abt_buy/`. This is dataset provenance/size, not a published M1 runtime or accuracy result. No new blank measurement needed.

### Completion checklist for the eventual publication

- Fill blanks only from identified run artifacts with binary/config/model/hardware and timing boundaries recorded; leave unmeasured cases explicitly unmeasured.
- Link exact duplicates to canonical performance/rerun tables; update HTML prose, API batch table and soak image as applicable, not just Markdown performance tables.
- Keep M3 history, hardware-unattributed estimates, third-party solver tests, targets, illustrative CLI output and linked M1 research in their separate categories.
- No additional measured M1 table was found in the remaining site pages (`api` outside throughput, `cli-reference`, `enroll-mode`, `hooks`, or the remaining scoring/live/configuration material). No external-engine M1 head-to-head comparison was found on the site.
