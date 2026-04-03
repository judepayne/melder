---
type: task
module: general
status: active
tags: [todo, backlog, progress]
related_code: []
---

# Project Status

Last updated: 2026-04-02 (Exclusions system, initial match pass, rayon deadlock fix, accuracy test, competitor analysis)

## Completed

*(Earlier work through 2026-03-27: core engine, all scoring methods, flat/usearch backends, CSV/JSONL/Parquet, combined vector index, three-layer cache invalidation, CrossMap, WAL, encoding coordinator, text-hash skip, CLI subcommands, batch API endpoints, graceful shutdown, benchmarks, Windows support, Live API enhancements, WAL restart consistency, vault docs, BM25/Tantivy phase 1, SQLite Phase 2 Steps 1–5 — RecordStore + CrossMapOps traits + SQLite implementations + live startup wiring + review queue write-through, memory-mapped vector index, backend abstraction cleanup, export CLI, batch timing fix, SQLite benchmarks, BM25 commit batching, bm25_fields config, SQLite batch mode, SQLite connection pool, columnar SQLite storage.)*
- [x] **BM25 query sanitization + analytical self-score** — Sanitized BM25 query text (lowercase, remove special chars) to prevent Tantivy parse errors. Implemented analytical self-score computation (max possible BM25 score for a query) using token count + average IDF, cached by text hash. Enables true BM25 normalization without per-query insert/commit overhead. See [[decisions/key_decisions#Exact Prefilter Pre-Blocking Exact Match Confirmation]].
- [x] **Exact prefilter feature** — Pre-blocking exact match confirmation phase. Checks whether all configured field pairs match exactly (AND semantics). If they all match, auto-confirms at score 1.0 immediately. Runs before blocking, recovering cross-block matches (e.g. matching LEI but wrong country code). On 10k×10k dataset: 4,211 exact matches confirmed in ~17ms, raising combined ceiling from 6,675 to 6,863 (+188 recovered pairs). See [[decisions/key_decisions#Exact Prefilter Pre-Blocking Exact Match Confirmation]].
- [x] **Score gap / confidence margin (`min_score_gap`)** — Implemented in `src/config/schema.rs`, `src/config/loader.rs`, `src/matching/pipeline.rs`. Allows filtering candidate pairs by minimum score gap between top and second-best match, reducing ambiguous review queue entries.
- [x] **SIMD-accelerated dot product (`--features simd`)** — Consolidated 3 duplicate dot product implementations (`flat.rs`, `embedding.rs`, `pipeline.rs`, `candidates.rs`) into a single `dot_product_f32()` in `src/scoring/embedding.rs`. Added `simsimd` as optional dependency behind `simd` feature flag. When enabled, dispatches to SimSIMD's hardware-accelerated inner product (NEON / SVE / AVX2 / AVX-512). Without the feature, falls back to the iterator loop LLVM auto-vectorizes. Primary beneficiaries: Windows users on the flat backend (no usearch).
- [x] **Published fine-tuned Arctic-embed-xs to HuggingFace** — Uploaded experiment 9 R22 model to `themelder/arctic-embed-xs-entity-resolution` with full model card (YAML front matter, benchmark results, training details, usage examples). Added `hf-hub` dependency and HuggingFace Hub download path to `EncoderPool::new()` — model names containing `/` that aren't local paths are downloaded automatically. Commented-out model reference added to all 30 batch/live benchmark configs.
- [x] **Enroll endpoint for single-pool entity resolution** — New `mode: enroll` with 5 HTTP endpoints: `POST /api/v1/enroll`, `POST /api/v1/enroll-batch`, `POST /api/v1/enroll/remove`, `GET /api/v1/enroll/query`, `GET /api/v1/enroll/count`. New `EnrollConfig` serde schema in `src/config/enroll_schema.rs`. New `Mode` enum (Match/Enroll) in `src/config/schema.rs`. `load_enroll_config()` in `src/config/loader.rs`. `Session::enroll()` and `Session::enroll_batch()` methods. Self-match exclusion in `pipeline.rs` when query_side == pool_side. `blocking_query()` gains `pool_side` parameter for same-side blocking. `LiveMatchState::load_enroll()` for single-pool startup (A-side only, no crossmap). Conditional router: enroll endpoints only mounted in enroll mode. 6 new tests for enroll config parsing. 382 tests pass, zero clippy warnings. See [[Enroll Endpoint Design]].
- [x] **Structured logging** — Converted ~70 `eprintln!` calls in live-path files to `tracing` (info!/warn!). `--log-format json` now produces clean structured JSON for all server output including startup, encoding, WAL, shutdown. CLI-only commands (run, tune, etc.) kept as `eprintln!`. Prerequisite for pipeline hooks.
- [x] **HuggingFace Hub model download** — Added `hf-hub` dependency. `EncoderPool::new()` auto-downloads models with `/` in the name from HuggingFace Hub (e.g. `themelder/arctic-embed-xs-entity-resolution`). Local paths detected by heuristic (absolute, `./`, `../`, `.onnx` suffix, or resolves on disk).
- [x] **Pipeline hooks** — Single long-running subprocess receiving NDJSON events on stdin. Config: `hooks: { command: "python hook.py" }` in both match and enroll mode configs. 4 event types: `on_confirm`, `on_review`, `on_nomatch`, `on_break`. `HookEvent` enum in `src/hooks/mod.rs` with custom JSON serialization. Hook writer task in `src/hooks/writer.rs` with subprocess lifecycle, exponential backoff respawn (1s-60s), disable after 5 consecutive failures. Non-blocking: scoring thread uses `try_send` on mpsc channel (~10ns), dedicated writer task handles pipe I/O. 6 injection points in `src/session/mod.rs` (2× on_confirm auto, 1× on_confirm manual, 1× on_review, 1× on_nomatch, 1× on_break). Platform dispatch: `sh -c` on Unix, `cmd /C` on Windows. Validation: empty command rejected. 392 tests pass, zero clippy warnings. Full docs page at `docs/hooks.md` with example Python script. Design spec at `HOOK_DESIGN.md`.
- [x] **`expansion_search` config knob** — Added `performance.expansion_search` (HNSW ef parameter) as a configurable field for the usearch backend. Default 0 (usearch default). Threaded through config → vectordb → usearch_backend. Documented in docs/configuration.md.
- [x] **Tracing spans** — Added lightweight info_span tracing at hot-path boundaries: upsert_record, encode_combined, onnx_encode, bm25_upsert, bm25_commit, blocking_query, score_pool, ann_candidates, bm25_candidates, full_scoring, claim_loop. Near-zero cost when not collecting.
- [x] **New benchmark** — Added benchmarks/live/10kx10k_inject50k_usearch/ with resource monitoring (CPU/GPU/memory via psutil and ioreg). Uses Arctic-embed-xs + 50% BM25 scoring config.
- [x] **SimpleBm25 — Tantivy replacement** — Replaced the Tantivy-backed BM25 index (`src/bm25/index.rs`, 1,226 lines) with a custom DashMap-based scorer (`src/bm25/simple.rs`, ~1,000 lines). Removed `tantivy = "0.22"` from Cargo.toml (~40 transitive dependencies eliminated). `LiveSideState.bm25_index` changed from `Option<RwLock<BM25Index>>` to `Option<SimpleBm25>` — no external locks needed. All RwLock write/read lock acquisition, `commit_if_ready`, `commit_if_dirty`, and `Bm25Ctx` usage removed from `session/mod.rs`. `score_pool()` signature now accepts pre-computed BM25 and synonym candidates. `bm25_commit_batch_size` config deprecated (instant write visibility). Results: default live throughput 3.2× (461→1,460 req/s); batch +6.6%; startup 1.7s faster. 394 tests pass, zero clippy warnings. See [[decisions/key_decisions#Replace Tantivy BM25 Index with Custom DashMap Based SimpleBm25]].
- [x] **WAND BM25 implementation** — Implemented Block-Max WAND (Weak AND) scoring in `src/bm25/simple.rs` to replace the old inverted index path. Key innovations: (1) Compact doc IDs via `CompactIdMap` reduce posting entry size from ~28 bytes to 8 bytes, saving ~2.2GB at 4.5M scale. (2) `BlockedPostingList` with blocks of ~128 entries and precomputed `max_tf` per block. (3) Block-Max WAND scorer uses per-block upper bounds to skip documents whose cumulative score can't beat Kth-best. (4) Simplified API: `score_blocked()` no longer takes `query_record`/`query_side`. (5) Two-path strategy: exhaustive (B ≤ 5,000) and WAND (B > 5,000). Results: 100k×100k live benchmark 1,070 req/s (+3.4%), p50=6.7ms, p95=24.1ms, BM25 build=309ms (4.7× faster). 396 tests pass, zero clippy warnings. See [[decisions/key_decisions#WAND BM25 Implementation]].
- [x] **OR blocking removed** — `blocking.operator` now only accepts `"and"`. `"or"` is rejected at validation with a clear error message. Removed `BlockingOperator::Or` enum, `or_indices` field, and all OR branches from `matching/blocking.rs`. Removed OR SQL path in `store/sqlite.rs`. Removed `blocking_hash_changes_on_operator` test from `vectordb/manifest.rs`. Updated `docs/configuration.md` and `docs/accuracy-and-tuning.md`. Rationale: OR blocking created overlapping blocks incompatible with per-block candidate generation (both ANN and BM25). Never used in production configs. See [[decisions/key_decisions#OR Blocking Removed]].
- [x] **CI/CD pipeline (GitHub Actions)** — `.github/workflows/ci.yml` (test --all-features, fmt, clippy) and `.github/workflows/release.yml` (macOS ARM, Linux x86_64, Windows MSVC). Triggers: CI on every push/PR, release on `v*.*.*` tags.
- [x] **Pipeline parallelization** — Encode, store+blocking, BM25, and synonym branches run concurrently via `rayon::scope` in `upsert_record`. Hides BM25/store/synonym work under encoding latency.
- [x] **CI performance regression tests** — Added `perf` job to CI workflow: generates 10k data, builds release binary, runs batch (10kx10k_usearch) and live (10kx10k_inject3k_usearch) benchmarks, parses throughput, fails if below 70% of baseline (batch: 9000 rec/s, live: 375 req/s). Baseline on GitHub ubuntu-latest: batch 12,984 rec/s, live 535 req/s.
- [x] **Single-artifact deployment (builtin-model)** — `--features builtin-model` compiles an ONNX embedding model into the binary via `include_bytes!()`. `build.rs` downloads from HuggingFace Hub or copies from local path (controlled by `MELDER_BUILTIN_MODEL` env var, defaults to `themelder/arctic-embed-xs-entity-resolution`). Config: `model: builtin`. Binary size 37 MB → 125 MB. 397 tests pass, zero clippy warnings.

- [x] **Exclusions (Known Non-Matches)** — Stateful system for known non-matching pairs. New `exclusions:` config section with `path`, `a_id_field`, `b_id_field`. New `Exclusions` struct in `src/matching/exclusions.rs` (RwLock<HashSet>). WAL events: `Exclude`, `Unexclude`. API endpoints: `POST /api/v1/exclude`, `DELETE /api/v1/exclude` (both match and enroll modes). Hook event: `on_exclude` with `match_was_broken` field. Session methods: `exclude()`, `unexclude()`. Pipeline filter after candidate union, before scoring. Batch mode: loads from CSV (read-only). Live mode: loads from CSV + WAL replay, flushes on shutdown. If an excluded pair is currently matched in CrossMap, the match is broken first. 417 tests pass, zero clippy warnings.

- [x] **Initial match pass at live startup** — After datasets are loaded and all indices built, `meld serve` now runs an initial matching pass that scores all unmatched B records against the A pool before the HTTP server starts listening. Uses the same scoring pipeline as live upserts (blocking, BM25, ANN, synonym, exclusions). Skips encoding — records already encoded and in vector index from startup load. Crossmap claims written to WAL (crash-safe). Review-band matches added to review queue. Hook events (on_confirm) fire normally. Progress logged every 1000 records. Skips records already in crossmap (from CSV or WAL replay). No-op if either side empty or all B records already matched. Configurable via `live.skip_initial_match: true` to suppress. Implementation: `Session::initial_match_pass()` in `src/session/mod.rs`, called from `src/cli/serve.rs` after session creation, before `start_server()`. Reuses scoring pipeline rather than calling `run_batch()` because live mode already has pre-built BM25/synonym indices that `run_batch()` would duplicate.

- [x] **Rayon/ONNX encoding deadlock fix** — Root cause: GPU encoding commit changed `encode_and_upsert` from sequential `for` loop to `par_chunks` on rayon's global thread pool. With fewer encoder slots than rayon workers, blocked workers starved ONNX's internal rayon tasks. Fix: dedicated rayon thread pool in `encode_and_upsert` sized to `encoder_pool.pool_size()`, preventing global pool starvation. This was a pre-existing bug since the GPU encoding commit (c963a91).

- [x] **Generator enhancement: --exact-matches flag** — Added `--exact-matches N` flag to `benchmarks/data/generate.py`. First N matched B records are exact copies of their A counterpart (no noise). Marked as `_match_type: "exact"` for test identification. Backwards compatible (defaults to 0).

- [x] **Accuracy test: 10kx10k_exclusions** — New benchmark in `benchmarks/accuracy/10kx10k_exclusions/`. Phase 1: validates initial match pass respects exclusions (0/1000 excluded pairs matched). Phase 2: validates live API exclusions (exclude-then-inject, break existing matches, unexclude). Streams embedding progress and server logs live to terminal.

- [x] **Competitor analysis documents** — Created `Competitor_analysis.md` (positioning vs Quantexa and ES/OpenSearch) and `vs_Elastic_OpenSearch.md` (detailed technical comparison). Key sections: core distribution problem, in-process advantage, cost comparison, capability matrix, Melder design philosophy.

- [x] **Benchmark data regeneration scripts** — Per-directory `generate_data.sh` scripts for `batch/`, `live/`, and `accuracy/`. Each calls `benchmarks/data/generate.py` with the sizes its benchmarks need (10k/100k/1M for batch and live, 10k for accuracy). `accuracy/10kx10k_exclusions` excluded (generates its own data). `accuracy/science/` excluded (research journal, not reproducible benchmarks). User docs added to `docs/building.md`.

## In Progress

*(Nothing currently in progress.)*

---

**Benchmark Results Updated: 2026-03-14**
All 12 benchmarks completed (3 batch + 3 live configurations). Results updated in [[benchmarks_and_experiments]]:
- Batch: flat 10k, usearch 10k, usearch 100k
- Live: flat 10k (c=10), usearch 10k (c=10), usearch 100k (c=10, 10k events)
Measured on Apple M3 MacBook Air with `all-MiniLM-L6-v2` and `encoder_pool_size: 4`.

## Backlog (Ranked)

Items are ranked by assessed usefulness.

---

**1. External vector database.** *(Vector index) -- maybe*
Qdrant / Milvus / Weaviate over gRPC. The `VectorDB` trait surface is already
the right abstraction -- a client implementation would be a drop-in replacement.
Buys: no staleness problem, durable persistence, multiple melder instances sharing
one index. Costs: network round-trip (~1-5ms) per search, external service
dependency. Only worth it at 1M+ records or when horizontal scaling is needed.

**2. Persistent edge store for enroll mode.** *(Enroll)*
Enroll mode currently returns edges in the HTTP response and discards them — no
persistent record of discovered relationships. Add a persistent edge store
(in-memory + WAL or SQLite-backed) that maintains bidirectional edges: when
record Y is enrolled and matches X, both Y→X and X→Y are recorded. Since
scoring is symmetric (`score_pair(a,b) == score_pair(b,a)`), this requires no
re-scoring — just symmetric bookkeeping after the single scoring pass. Enables:
query an enrolled record's current neighbours, survive restarts without losing
relationships, hook notifications when existing records gain new edges.
Prerequisite for enroll mode being a real entity resolution service rather than
a one-shot dedup check.

**3. GPU encoding on Linux CI / CUDA.** *(Infrastructure)*
The `gpu-encode` feature currently works on macOS (CoreML) but is untested on
Linux (CUDA path). CI runs on Ubuntu and cannot enable `gpu-encode` because
there is no `libonnxruntime.so` installed. Investigate: (a) adding ORT to the
CI runner (apt package or download), (b) whether the CUDA execution provider
works without a GPU (CPU fallback), (c) whether a lighter ORT build
(CPU-only dynamic) would let us test the `ort-load-dynamic` codepath without
needing a full CUDA stack. Goal: CI tests all features including `gpu-encode`.

---

*10M+ regime: stateless scoring workers + stateful coordination layer (crossmap,
WAL, shard routing). Batch mode becomes a Spark-style shuffle job. Not worth
designing until the need is real. See `splink` for prior art.*


