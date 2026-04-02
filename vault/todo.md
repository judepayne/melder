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

- [x] Core matching engine (batch + live modes)
- [x] Scoring pipeline: exact, fuzzy (ratio, partial_ratio, token_sort, wratio), embedding, numeric (equality-only)
- [x] Vector backends: flat (brute-force) and usearch (HNSW ANN, feature-gated)
- [x] Data formats: CSV, JSONL, Parquet (feature-gated)
- [x] Combined vector index with sqrt(weight) scaling -- see [[Constitution#4 Combined Vector Weighted Cosine Identity]]
- [x] Three-layer cache invalidation (spec hash + manifest + text hash) -- see [[Key Decisions#Three-Layer Cache Invalidation]]
- [x] CrossMap with atomic 1:1 claiming -- see [[Constitution#3 CrossMap Bijection 1 1 Under One Lock]]
- [x] Blocking: single and multi-field, AND/OR modes, indexed for live
- [x] Common-ID pre-matching
- [x] WAL with replay, compaction, per-run timestamped files
- [x] Encoding coordinator (optional batched ONNX inference) -- see [[Key Decisions#Encoding Coordinator Batched ONNX Inference]]
- [x] Text-hash skip optimization -- see [[Key Decisions#Text-Hash Skip Optimization]]
- [x] Vector quantization: f32, f16, bf16 (usearch backend)
- [x] CLI: run, serve, validate, tune, cache (build/status/clear), review (list/import), crossmap (stats/export/import)
- [x] Batch API endpoints: add-batch, match-batch, remove-batch (max 1000 per request)
- [x] Graceful shutdown with state persistence
- [x] Benchmark suite (smoke test, stress test, concurrent test, batch endpoint test)
- [x] README, TUNE.md, worked examples (batch + live)
- [x] Windows compatibility: cross-platform `rename_replacing`, `PathBuf`-based cache paths, `.gitattributes`, platform-agnostic tests, README Windows section
- [x] Live API enhancements: 5 new read-only endpoints (`crossmap/pairs`, `crossmap/stats`, `a/unmatched`, `b/unmatched`, `review/list`) with pagination, review queue backed by `DashMap` on `LiveMatchState`, drained on confirm/break/re-upsert/remove, WAL replay populates queue on startup
- [x] WAL replay restart consistency: (a) blocking index now updated during WAL replay for both upserts and removes, (b) vector index cache preserved across restarts via `skip_deletes` in `build_or_load_combined_index` — WAL-added vectors are retained in cache instead of being deleted then re-encoded, (c) `contains()` guard skips re-encoding when vector already in index. End-to-end restart test added (`tests/restart/test_restart.sh`): 23 assertions covering record survival, crossmap persistence, stats consistency, blocking index rebuild, and vector cache reuse.
- [x] Vault cleanup: deleted `vault/performance/` (rogue dir), `vault/LIVE_PERFORMANCE.md` (empty), `vault/Untitled.canvas` (artefact); created `vault/benchmarks/Benchmarks.md` with updated numbers and fixed broken link; added YAML frontmatter to all 9 vault `.md` files per doc-agent spec.
- [x] Vault graph linkage: all 8 content nodes now have ≥2 incoming WikiLinks; added cross-links between CONSTITUTION, Business Logic Flow, Module Map, Key Decisions, Discarded Ideas, Benchmarks, Use Cases, and Performance Baselines to form a fully connected Obsidian graph.
- [x] Vault expanded with 4 new architecture notes: Config Reference (full YAML schema), API Reference (all 30+ HTTP routes with request/response shapes), Scoring Algorithm (composite formula, per-field methods, classification), State & Persistence (13-step startup sequence, WAL event types and replay, compaction, CrossMap flush). All 13 vault notes fully interconnected (minimum 2 incoming links per node).
- [x] README: updated blocking section to document multi-field AND/OR capability with config examples.
- [x] Vault: added `Scaling to Millions.md` (scaling architecture roadmap) and `Dual Process LLM Architecture.md` (speculative cognitive architecture for LLM persistent memory) to `vault/ideas/`.
- [x] BM25 scoring and candidate filtering (Phase 1 of Scaling to Millions): Tantivy-backed BM25 index as both a first-class scoring method (`method: bm25`) and sequential candidate filter. Feature-gated behind `--features bm25`. Supports batch and live modes, with real-time index updates on upsert/remove. Normalised BM25 scores via self-score ratio. Sequential pipeline: blocking → ANN → BM25 re-rank → full scoring. New config fields: `ann_candidates`, `bm25_candidates`. 32 new tests (13 index, 7 scorer, 10 validation, 2 scoring). See [[Scaling to Millions]].
- [x] **Phase 2 Step 1: RecordStore trait + MemoryStore** — Introduced `RecordStore` trait abstraction (18 methods across 4 categories: records, blocking, unmatched, common_id) with `MemoryStore` as the DashMap-backed implementation. Both batch and live modes now go through the trait. Pipeline functions take `&dyn RecordStore` instead of `&DashMap`. LiveSideState simplified from 6 fields to 2. Pure refactor — zero behavior change. Enables SqliteStore implementation in Step 3. See [[Key Decisions#RecordStore trait for storage abstraction]].
- [x] **Phase 2 Step 2: CrossMapOps trait + MemoryCrossMap** — Introduced `CrossMapOps` trait abstraction (13 runtime methods: add, remove, claim, break, get, has, pairs, stats, export, import, and 3 internal helpers) to decouple CrossMap operations from the concrete `RwLock<CrossMapInner>` implementation. Persistence methods (load/save) remain as inherent methods on `MemoryCrossMap` since they differ between backends. Both batch and live modes now take `&dyn CrossMapOps`. Pure refactor — zero behavior change. Enables SqliteCrossMap implementation in Step 3. See [[Key Decisions#CrossMapOps trait for crossmap abstraction]].
- [x] **Phase 2 Step 3: SqliteStore + SqliteCrossMap** — Implemented SQLite-backed storage and crossmap backends. Added `rusqlite` dependency (always compiled, not feature-gated). SqliteStore implements RecordStore trait (records as JSON, blocking keys with composite index, AND/OR query modes). SqliteCrossMap implements CrossMapOps trait (bijection via UNIQUE constraints, DELETE-then-INSERT for add, check-both-sides for claim). Factory function `open_sqlite()` creates both from shared `Arc<Mutex<Connection>>` with WAL mode, 64MB cache. 25 new tests (13 store + 12 crossmap). See [[Key Decisions#SqliteStore SqliteCrossMap for Durable Live Mode]].
- [x] **Phase 2 Step 4: Wire SqliteStore into meld serve startup** — LiveMatchState generalized to use `Arc<dyn RecordStore>` + `Box<dyn CrossMapOps>` with trait-object dynamic dispatch. Two startup paths: memory-backed (default, backward compatible) and SQLite-backed (`live.db_path` config). SQLite cold start loads CSV into SqliteStore; warm start opens existing DB. CrossMap flush is no-op for SQLite. Added `as_any()` to CrossMapOps trait for safe downcasting. Session and pipeline code unchanged. See [[Key Decisions#Wiring SqliteStore into Live Startup]].
- [x] **Phase 2 Step 5: Review queue SQLite write-through + WAL replay skip** — Review queue mutations now write through to SQLite's `reviews` table via three new methods on LiveMatchState: `insert_review()`, `drain_reviews_for_id()`, `drain_reviews_for_pair()`. Session uses these instead of direct DashMap access. SQLite warm start loads reviews from DB. `open_sqlite()` returns shared connection for write-through. WAL replay skipped for SQLite path (established in Step 4). Completes Phase 2: SQLite Record Store. See [[Key Decisions#Review Queue SQLite Write-Through WAL Replay Skip]].
- [x] **Phase 3: Memory-Mapped Vector Index** — Added `performance.vector_index_mode: "load" | "mmap"` config option. When `"mmap"`, the usearch backend calls `index.view()` instead of `index.load()`, memory-mapping the HNSW graph so the OS manages paging. Lower peak RAM for extreme-scale batch jobs; read-only so not suitable for `meld serve` (warning added). Default is `"load"` — zero behaviour change. 6 new tests. See [[Scaling to Millions]] and [[Key Decisions#Memory-Mapped Vector Index]].
- [x] **Backend abstraction cleanup** — Eliminated all runtime backend awareness from `LiveMatchState`. Added `flush()` to `CrossMapOps` trait (replaces `as_any()` + downcast), added 4 review persistence methods to `RecordStore` trait (replaces `sqlite_conn` checks), removed `uses_sqlite`/`sqlite_conn`/`as_any()` fields. Extracted `load_datasets()`, `replay_wal()`, `finish()` helpers to reduce duplication between memory and SQLite startup paths. Zero behavior change — all 332 tests pass. See [[Key Decisions#Backend Abstraction Cleanup]].
- [x] **`meld export` CLI command** — New subcommand that exports live-mode state (records + crossmap) to CSV files. Supports both SQLite and memory backends. 3 tests: `sqlite_export_round_trip`, `memory_export_with_wal`, `empty_state_writes_headers_only`. See `src/cli/export.rs`.
- [x] **Batch `rec/s` calculation fix** — `BatchStats` now tracks `scoring_start` and `scoring_elapsed_secs` separately. Progress reporting and summary use scoring-only elapsed time instead of total wall time.
- [x] **README improvements** — Updated config YAML annotations (optional fields only), batch summary example with new timing format, rewrote Live mode section covering both backends.
- [x] **SQLite live-mode benchmarks** — Measured SQLite vs in-memory at 10k scale: ~25% lower throughput (1,183 vs 1,616 req/s warm), better p95 latency (18.0ms vs 20.7ms), dramatically faster warm start (0.5s vs 1.9s). Results added to [[Performance Baselines]].
- [x] **BM25 index commit batching in live mode** — Buffered BM25 writes with a `dirty` flag. `upsert()` and `remove()` mark `dirty = true` without committing. New `commit_if_dirty()` method commits + reloads + clears cache only when dirty. Session calls `commit_if_dirty()` on opposite-side index before querying. Naturally batches commits by side transition (A,A,A,B commits A once when B arrives). 2x throughput improvement: 256 → 512 req/s on 10k×10k with usearch+BM25. See [[Key Decisions#BM25 Index Commit Batching in Live Mode]].
- [x] **Explicit `bm25_fields` config section** — Added optional top-level `bm25_fields` config section so users can specify which text fields to index for BM25 independently of match_fields. When omitted, falls back to deriving from fuzzy/embedding fields (backward compatible). Eliminates ghost zero-weight match_fields in BM25-only configs. See [[Key Decisions#Explicit bm25_fields Config Section]].
- [x] **SQLite-backed batch mode** — Wired `SqliteStore` + `SqliteCrossMap` into `meld run` (batch mode). Added `batch.db_path` config option. When set, batch mode uses streaming data loaders (stream_csv, stream_jsonl, stream_parquet) with chunked callback-based reading, SqliteStore.bulk_load() with single-transaction inserts and deferred index creation, and get_many() for batched candidate lookups. DB is created fresh each run and deleted on completion. Memory footprint drops to ~10-12GB (cache + BM25 index + blocking index). Benchmark: 1420 rec/s at 10K scale, expected ~100 minutes for 4.5M B records at 55M scale. See [[Key Decisions#SQLite-Backed Batch Mode]].
- [x] **SQLite connection pool** — Replaced single `Arc<Mutex<Connection>>` with writer + N read-only connections (round-robin try_lock). 7% live SQLite improvement (1183 → 1389 req/s). New config: `sqlite_read_pool_size` (default 4), `sqlite_pool_worker_cache_mb` (default 128). See [[Key Decisions#SQLite Connection Pool Writer N Readers]].
- [x] **Columnar SQLite storage** — Replaced JSON blob `record_json` column with one column per field. Dynamic schema from config required_fields. Eliminated all JSON serialization/deserialization in scoring hot path. 90% batch improvement (748 → 1420 rec/s). BM25 index build 2x faster. See [[Key Decisions#Columnar SQLite Storage]].
- [x] **BM25 query sanitization + analytical self-score** — Sanitized BM25 query text (lowercase, remove special chars) to prevent Tantivy parse errors. Implemented analytical self-score computation (max possible BM25 score for a query) using token count + average IDF, cached by text hash. Enables true BM25 normalization without per-query insert/commit overhead. See [[Key Decisions#Exact Prefilter Pre-Blocking Exact Match Confirmation]].
- [x] **Exact prefilter feature** — Pre-blocking exact match confirmation phase. Checks whether all configured field pairs match exactly (AND semantics). If they all match, auto-confirms at score 1.0 immediately. Runs before blocking, recovering cross-block matches (e.g. matching LEI but wrong country code). On 10k×10k dataset: 4,211 exact matches confirmed in ~17ms, raising combined ceiling from 6,675 to 6,863 (+188 recovered pairs). See [[Key Decisions#Exact Prefilter Pre-Blocking Exact Match Confirmation]].
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
- [x] **SimpleBm25 — Tantivy replacement** — Replaced the Tantivy-backed BM25 index (`src/bm25/index.rs`, 1,226 lines) with a custom DashMap-based scorer (`src/bm25/simple.rs`, ~1,000 lines). Removed `tantivy = "0.22"` from Cargo.toml (~40 transitive dependencies eliminated). `LiveSideState.bm25_index` changed from `Option<RwLock<BM25Index>>` to `Option<SimpleBm25>` — no external locks needed. All RwLock write/read lock acquisition, `commit_if_ready`, `commit_if_dirty`, and `Bm25Ctx` usage removed from `session/mod.rs`. `score_pool()` signature now accepts pre-computed BM25 and synonym candidates. `bm25_commit_batch_size` config deprecated (instant write visibility). Results: default live throughput 3.2× (461→1,460 req/s); batch +6.6%; startup 1.7s faster. 394 tests pass, zero clippy warnings. See [[Key Decisions#Replace Tantivy BM25 Index with Custom DashMap Based SimpleBm25]].
- [x] **WAND BM25 implementation** — Implemented Block-Max WAND (Weak AND) scoring in `src/bm25/simple.rs` to replace the old inverted index path. Key innovations: (1) Compact doc IDs via `CompactIdMap` reduce posting entry size from ~28 bytes to 8 bytes, saving ~2.2GB at 4.5M scale. (2) `BlockedPostingList` with blocks of ~128 entries and precomputed `max_tf` per block. (3) Block-Max WAND scorer uses per-block upper bounds to skip documents whose cumulative score can't beat Kth-best. (4) Simplified API: `score_blocked()` no longer takes `query_record`/`query_side`. (5) Two-path strategy: exhaustive (B ≤ 5,000) and WAND (B > 5,000). Results: 100k×100k live benchmark 1,070 req/s (+3.4%), p50=6.7ms, p95=24.1ms, BM25 build=309ms (4.7× faster). 396 tests pass, zero clippy warnings. See [[Key Decisions#WAND BM25 Implementation]].
- [x] **OR blocking removed** — `blocking.operator` now only accepts `"and"`. `"or"` is rejected at validation with a clear error message. Removed `BlockingOperator::Or` enum, `or_indices` field, and all OR branches from `matching/blocking.rs`. Removed OR SQL path in `store/sqlite.rs`. Removed `blocking_hash_changes_on_operator` test from `vectordb/manifest.rs`. Updated `docs/configuration.md` and `docs/accuracy-and-tuning.md`. Rationale: OR blocking created overlapping blocks incompatible with per-block candidate generation (both ANN and BM25). Never used in production configs. See [[Key Decisions#OR Blocking Removed]].
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

## In Progress

*(Nothing currently in progress.)*

---

**Benchmark Results Updated: 2026-03-14**
All 12 benchmarks completed (3 batch + 3 live configurations). Results updated in [[Performance Baselines]]:
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

**3. Benchmark data regeneration script.** *(Developer Experience)*
The `benchmarks/data/` directory is gitignored (datasets are too large to commit). A user who clones the repo has no data to run benchmarks. Add a `benchmarks/data/generate_all.sh` (or similar) script that calls `generate.py` with all required sizes (1K for tests, 10K, 100K, 1M) and validates the output. Should be documented in the README under a "Running Benchmarks" section or similar.

**4. GPU encoding on Linux CI / CUDA.** *(Infrastructure)*
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


