# melder — Build Plan

Detailed implementation plan for the melder Rust rewrite. Each task is scoped
to 1-3 hours. Tasks within a phase are sequential unless marked `[parallel]`.
Every task has a verification step — do not move on until it passes.

**Reference paths:**
- Go project: `/Users/jude/Library/CloudStorage/Dropbox/Projects/match/`
- Rust project: `/Users/jude/Library/CloudStorage/Dropbox/Projects/melder/`

**Key design doc:** `DESIGN.md` in this directory (1614 lines, 21 sections).

---

## Key Invariants (check at every phase boundary)

1. **Symmetry (live mode only):** Every feature for A exists identically for B.
   Batch mode is asymmetrical — A is the reference pool, B records are queries.
2. **Config compatibility:** Go YAML configs work unchanged (ignoring `sidecar`).
3. **API compatibility:** Same endpoints, same request/response JSON shapes.
4. **Scoring compatibility:** Same classification logic, same weight handling.
   Small float differences in embedding scores are acceptable.
5. **No `Ref` across `.await`:** DashMap guards are never held across async
   boundaries.
6. **Tests pass:** `cargo test` passes at every commit.

---

## Phases 0–5: Complete (48/48 tasks)

All foundational work is done. The Rust rewrite is feature-complete and
passes all stress tests.

| Phase | Description | Tasks | Status |
|---|---|---|---|
| 0 | Scaffold + test data | 2 | Done |
| 1 | Config + models + scoring | 11 | Done |
| 2 | Encoder + index + cache | 6 | Done |
| 3 | Matching engine + batch | 10 | Done |
| 4 | Live server | 11 | Done |
| 5 | Polish + CLI + benchmark | 8 | Done |

**Phase 0** — Cargo project initialised, test data and configs copied from Go project.

**Phase 1** — Error types, core models, config schema/loader/validation, CSV
data loader, exact + fuzzy scorers (with golden test data from Python
rapidfuzz), CLI skeleton with `validate` command. 11 tasks.

**Phase 2** — Encoder pool (fastembed/ONNX), embedding scorer (cosine
similarity), flat vector index (`VecIndex` with upsert/remove/search/
search_filtered), index cache serialization, state loader, `cache` CLI
commands. 6 tasks.

**Phase 3** — Composite scorer, blocking filter (linear scan + BlockingIndex),
CrossMap (local CSV backend), candidate generation, match engine,
batch engine with Rayon parallelism, batch output writer, `run` and `tune`
CLI commands. 10 tasks.

**Phase 4** — Async dependencies (tokio/axum/dashmap), WAL (upsert log),
live MatchState with DashMap + RwLock, session (upsert/try-match/CrossMap
management), CrossMap background flusher, HTTP handlers + server, `serve`
CLI command, stress test validation. 11 tasks.

**Phase 5** — `review` and `crossmap` CLI commands, JSONL + Parquet data
loaders, tracing + structured logging, graceful shutdown hardening, error
handling audit, final benchmarks. 8 tasks.

---

## Phase 6: Scaling to 1M+ Records

Three pillars: usearch integration, encoding improvements, and candidate
selection improvements.

### Pillar 1: usearch Integration (Vector Search)

Replace the flat brute-force VecIndex with a usearch-backed HNSW index.
This is the foundation — everything else builds on it.

#### 6.1 — Add usearch dependency

- [ ] Add `usearch` crate to Cargo.toml. Verify it compiles on macOS
  (Apple Silicon). Run a minimal smoke test: create index, add 100 vectors,
  search, remove, save, load.

**Verify:** `cargo build` succeeds. Smoke test passes.

#### 6.2 — Design UsearchIndex adapter

- [ ] Create `src/index/usearch.rs` implementing the same interface as the
  current `VecIndex`:
  - `new(dim, capacity)` — create index with inner product metric, f32 dtype
  - `upsert(id, vec)` — add or replace a vector. Maintains bidirectional
    `String <-> u64` key mapping (`HashMap<String, u64>` + `Vec<String>`)
  - `remove(id)` — soft-remove: delete from key mapping but leave vector in
    usearch (cheap). Optionally call usearch `remove()` to mark as tombstone.
  - `search(query, k)` — search, map u64 keys back to String IDs
  - `search_filtered(query, k, allowed_ids)` — use usearch's
    `filtered_search()` with a closure predicate:
    `|key| allowed_u64s.contains(&key)`
  - `get(id)` — retrieve vector by string ID (via key mapping + usearch get)
  - `contains(id)` — check key mapping
  - `len()` — key mapping length (not usearch size, to avoid #697)

**Verify:** Unit tests for all interface methods. Parity with VecIndex
behaviour.

#### 6.3 — Implement persistence

- [ ] Replace the current `.index` binary cache format:
  - `save(path)` — save usearch index to file + save key mapping as sidecar
    (e.g. `path.keys` as bincode or msgpack)
  - `load(path)` — load usearch index + key mapping sidecar
  - `view(path)` — memory-mapped mode for large indexes (usearch native
    feature)
  - Staleness check: store a record count + hash in the sidecar header

**Verify:** Save index with 1K vectors, load back, all vectors and IDs
identical. Staleness check reports stale after adding a record.

#### 6.4 — Keep vectors on remove

- [ ] Change the remove semantics:
  - `remove_record()` deletes from record store, blocking index, and
    crossmap — but NOT from the vector index
  - The vector stays in usearch as an orphan
  - Re-adding a record checks if the embedding text matches an existing
    orphan — if so, reuse the vector (skip encoding)
  - Add periodic compaction: save/reload cycle that cleans up orphans
    (can run on a timer or on explicit API call)

**Verify:** Remove a record, re-add with same text — encoding is skipped.
Compaction removes orphans. Vector count before/after compaction differs.

#### 6.5 — Wire into MatchState

- [ ] Replace `VecIndex` with `UsearchIndex` in `LiveSideState` and
  `MatchState`. Update `src/state/state.rs` and `src/state/live.rs`. The
  DashMap records + UsearchIndex vectors should be the only storage.

**Verify:** All existing tests pass with the new index backend. Live mode
startup loads usearch index from cache.

#### 6.6 — Wire into batch engine

- [ ] Update `src/batch/engine.rs` to use the new index. The batch engine's
  Rayon parallel scoring should work without changes since usearch is
  concurrent-read safe.

**Verify:** `meld run` produces same output as before (within float
tolerance). Batch throughput is not regressed.

#### 6.7 — Wire into pipeline

- [ ] Update `src/matching/pipeline.rs`. The `score_pool()` function calls
  `index.get(id)` to retrieve vectors for cosine similarity — this must
  work with UsearchIndex.

**Verify:** Pipeline scoring produces same results as with flat index.

#### 6.8 — Update candidates for embedding method

- [ ] When `candidates.method: embedding`, use
  `UsearchIndex.search_filtered()` instead of the current flat scan. This
  should be a near drop-in replacement since the interface is the same.

**Verify:** Embedding-based candidate selection returns same top-K results
(order may differ for tied scores).

#### 6.9 — Tests

- [ ] Unit tests for UsearchIndex (create, upsert, search, filtered_search,
  remove, save/load, orphan reuse). Integration tests: run the full pipeline
  with usearch at 1K and 10K. All existing 116 tests must still pass.

**Verify:** `cargo test` passes. Integration tests at 1K and 10K pass.

#### 6.10 — Benchmark at 1M

- [ ] Generate 1M x 1M synthetic datasets. Run batch mode, measure
  throughput. Target: >1000 rec/s in batch mode with warm caches (vs
  306 rec/s at 100K with flat index). Run live mode, measure latency.
  Target: <5ms p50 at 1M.

**Verify:** Benchmark results recorded. Targets met or root cause of
shortfall identified.

---

### Pillar 2: Encoding Improvements

Reduce the cost of the ONNX forward pass, which is 65% of live-mode
wall time.

#### 6.11 — Request coalescing for live mode

- [ ] Add an encoding coordinator that collects live-mode requests arriving
  within a configurable window (default 2ms) and submits them as a single
  ONNX batch:
  - Requests enqueue their text + a oneshot channel into a bounded queue
  - A background task drains the queue every N ms (or when batch reaches
    size M)
  - Single ONNX encode call for the batch
  - Results fanned out via the oneshot channels
  - The `_with_vec` refactor already exists — the coordinator produces
    vectors, hands them to the existing scoring path
  - Config: `performance.coalesce_window_ms` (default: 2),
    `performance.coalesce_max_batch` (default: 32)

**Verify:** Under concurrent load, batch sizes > 1 observed in logs.
Throughput improves vs non-coalesced baseline.

#### 6.12 — Vector LRU cache

- [ ] Add a shared LRU cache keyed on embedding text hash (SHA-256 or
  xxhash of the concatenated embedding string):
  - Before calling ONNX, check the cache
  - On cache hit, return the cached vector (skip encoding entirely)
  - On cache miss, encode and insert into cache
  - Config: `performance.vector_cache_size` (default: 10000 entries)
  - The session already checks if the embedding text changed for a single
    record — this extends it across all records

**Verify:** Repeated upserts with same text hit the cache (log message).
Cache hit rate reported in metrics.

#### 6.13 — GPU / CoreML execution provider

- [ ] Add a config option `performance.encoder_backend`:
  - `cpu` (default) — current behaviour
  - `coreml` — Apple Silicon Neural Engine via CoreML execution provider
  - `cuda` — NVIDIA GPU via CUDA execution provider
  - At init time, pass the selected provider to fastembed/ort
  - When using GPU, pool_size should default to 1 (GPU sessions parallelise
    internally)

**Verify:** `coreml` backend initialises on Apple Silicon. Encoding
produces same vectors (within float tolerance). Throughput improves.

#### 6.14 — INT8 quantised model support

- [ ] Add config option `embeddings.quantized: true`:
  - When enabled, load the quantised ONNX graph variant of the model
  - fastembed supports this natively for some models
  - Expected: ~2x faster encoding on CPU with negligible quality loss for
    entity matching

**Verify:** Quantised model loads and produces vectors. Quality check:
cosine similarity between quantised and full-precision vectors > 0.95
for test pairs.

---

### Pillar 3: Candidate Selection Improvements

Make the candidate stage fast enough to handle 100K+ blocked records.

#### 6.15 — Parallel candidate scoring

- [ ] When `candidates.method: fuzzy`, use Rayon `par_iter` inside
  `select_candidates()` for the wratio scoring loop:
  - Each candidate score is independent — trivially parallelisable
  - At 100K candidates on 8 cores: ~200ms -> ~25ms
  - Small code change: `.iter()` -> `.par_iter()` + collect

**Verify:** Candidate scoring is parallelised (observable via CPU
utilisation). Latency reduced proportionally to core count.

#### 6.16 — Embedding-based candidates with HNSW

- [ ] When `candidates.method: embedding` and usearch is the backend,
  candidate selection becomes a single `search_filtered()` call:
  - Generate embedding for the candidate field text
  - Search the opposite side's usearch index with the blocking filter as
    predicate
  - Return top N by embedding similarity
  - This replaces N wratio comparisons with a single O(log N) HNSW search
  - Expected: <0.1ms for candidate selection at any scale

**Verify:** Candidate selection completes in <1ms at 100K scale. Results
are valid candidates.

#### 6.17 — Hybrid candidates

- [ ] New `candidates.method: hybrid`:
  - First pass: embedding search to grab top 100 candidates (sub-millisecond
    with HNSW)
  - Second pass: wratio on those 100 to re-rank
  - Return top N after re-ranking
  - Gets the recall of embedding search with the precision of
    character-level scoring

**Verify:** Hybrid method produces better precision than embedding-only
at similar latency. Recall matches or exceeds fuzzy-only.

#### 6.18 — Adaptive candidate method

- [ ] When `candidates.method: auto` (or as an internal optimisation):
  - If the blocked candidate pool is <= 1000 records, use fuzzy (wratio is
    fast enough)
  - If the blocked pool is > 1000, switch to embedding candidates
  - Log the switch so the user can see it in tracing output

**Verify:** Auto method selects fuzzy for small pools and embedding for
large pools. Tracing output shows the selected method.

---

### Supporting Work

#### 6.19 — Generate 1M synthetic datasets

- [ ] Create `testdata/generate_dataset.py` (or extend existing) to produce
  `dataset_a_1000000.csv` and `dataset_b_1000000.csv`. Same schema as
  existing synthetic data. Add a bench config
  `testdata/configs/bench1000000x1000000.yaml`.

**Verify:** Datasets generated with correct row counts and schema.
Config validates.

#### 6.20 — Update bench scripts for 1M

- [ ] Update `live_stress_test.py`, `live_batch_test.py` to support
  1M-scale testing. May need to increase timeouts and adjust progress
  reporting.

**Verify:** Bench scripts run against 1M data without timeout errors.

#### 6.21 — Update documentation

- [ ] Update DESIGN.md, README.md, FUTURE.md with usearch architecture,
  new config options, and 1M benchmark results.

**Verify:** Documentation reflects the current architecture and config
options.

#### 6.22 — Update FUTURE.md

- [ ] Move completed items from the scaling roadmap into a "completed"
  section. Update the performance roadmap table with actual 1M results.

**Verify:** FUTURE.md is current and accurate.

---

### Sequencing

The work should be done in this order:

1. Tasks 6.1–6.3 (usearch basic integration + persistence) — foundation
2. Tasks 6.4–6.8 (wire into pipeline, keep-vectors-on-remove) — full
   integration
3. Task 6.9 (tests) — validate correctness
4. Task 6.15 (parallel candidates) — quick win, independent of usearch
5. Tasks 6.16–6.18 (embedding/hybrid/adaptive candidates) — requires
   usearch
6. Task 6.19 (1M datasets) — needed for benchmarking
7. Task 6.10 (1M benchmark) — validate the whole stack
8. Tasks 6.11–6.14 (encoding improvements) — optimisation layer, can be
   done incrementally
9. Tasks 6.20–6.22 (docs + bench updates) — final polish

---

## Task Summary

| Phase | Description | Tasks | Status |
|---|---|---|---|
| 0 | Scaffold + test data | 2 | Done |
| 1 | Config + models + scoring | 11 | Done |
| 2 | Encoder + index + cache | 6 | Done |
| 3 | Matching engine + batch | 10 | Done |
| 4 | Live server | 11 | Done |
| 5 | Polish + CLI + benchmark | 8 | Done |
| 6 | Scaling to 1M+ records | 22 | Pending |
| **Total** | | **70** | |

---

## Critical Path

```
Phase 0–5 (done) → Phase 6
                      ├─ Pillar 1: usearch (6.1–6.10)
                      ├─ Pillar 3: candidates (6.15–6.18, after 6.5)
                      ├─ Pillar 2: encoding (6.11–6.14, independent)
                      └─ Supporting (6.19–6.22)
```

The single riskiest task is **6.1–6.2 (usearch integration)** — it's the
first time we integrate the `usearch` crate and verify HNSW search works
correctly as a drop-in replacement for the flat index. If the crate has
API or build issues on Apple Silicon, we need to find alternatives early.

The second riskiest is **6.5 (wire into MatchState)** — swapping the index
backend in the live path touches concurrency-sensitive code. Budget extra
time for debugging.
