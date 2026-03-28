---
type: decision
module: general
status: active
tags: [adr, design-decisions, rationale]
related_code: [vectordb/mod.rs, crossmap/mod.rs, vectordb/texthash.rs, vectordb/manifest.rs, encoder/coordinator.rs]
---

# Key Decisions

Architectural decisions made during Melder's development, with rationale.

## Combined Vector Index (Single Index Per Side)

**Decision**: Replace per-field-per-block vector indices with a single combined embedding index per side.

**Context**: The original design stored a separate vector index per embedding field per blocking bucket. This led to multiple ANN queries per record and complex index management.

**Choice**: Concatenate all per-field embeddings (each scaled by sqrt(weight)) into one combined vector per record. Store one index per side. A single ANN query retrieves candidates ranked by the exact weighted cosine sum.

**Why**: One query instead of N. The mathematical identity (`dot(combined_A, combined_B) = sum(w_i * cos(a_i, b_i))`) means no quality loss from the consolidation. Per-field scores are recovered cheaply via `decompose_emb_scores()` -- no second ONNX call needed. See [[Constitution#4 Combined Vector Weighted Cosine Identity]] and [[Scoring Algorithm#embedding]].

**Commit**: `6a2a332` (Mar 9)

---

## CrossMap: RwLock Over DashMap

**Decision**: Use a single `RwLock<CrossMapInner>` with two plain `HashMap`s (a_to_b, b_to_a) instead of two `DashMap`s.

**Context**: The CrossMap must enforce a strict 1:1 bijection between A and B records. The `claim()` operation must atomically check both directions and insert only if neither side is already claimed.

**Choice**: Single RwLock wrapping both maps. `claim()` takes a write lock, checks both maps, inserts into both, releases.

**Why**: Two DashMaps cannot atomically check both maps without deadlock risk from cross-shard locking. A TOCTOU gap between checking A-to-B and B-to-A would allow duplicate claims under Rayon parallelism. The single lock makes the bijection invariant trivially correct. Concurrent tests prove exactly-one-winner semantics. See [[Constitution#3 CrossMap Bijection 1 1 Under One Lock]].

**Commit**: `e440e4e` (Mar 11)

---

## Text-Hash Skip Optimization

**Decision**: Compute an FNV-1a hash of each record's embedding field texts. On upsert, if the hash matches the stored hash, skip ONNX re-encoding entirely.

**Context**: In live mode, many upsert requests modify non-embedding fields (e.g., status, metadata). Re-encoding unchanged text through ONNX wastes ~3ms per request.

**Choice**: `TextHashStore` in `vectordb/texthash.rs` stores per-record hashes. On upsert, compare hash before encoding. Skip if unchanged.

**Why**: Encoding is the dominant cost in the upsert path. Skipping it for unchanged text yielded a 20% throughput improvement in live mode benchmarks (809 -> 968 req/s). The hash is cheap (FNV-1a, sub-microsecond) and the false-negative rate is negligible.

**Commit**: `e440e4e` (Mar 11)

---

## Three-Layer Cache Invalidation

**Decision**: Implement a three-layer validation system for cached vector indices: spec-hash in filename, manifest sidecar, and per-record text-hash diffing.

**Context**: Cached indices must never silently serve stale results when config or data changes. Previous iterations had bugs where changing the blocking config or model would reuse an invalid cache.

**Choice**:
1. **Spec hash in filename**: Hash of embedding field names + order + weights + quantization. Different config = different filename = old cache unreachable.
2. **Manifest sidecar** (`.manifest`): JSON file recording model name, spec hash, and blocking hash. Catches model and blocking changes that the filename hash doesn't cover.
3. **Text-hash diffing**: Per-record FNV-1a hash of source text. Detects individual record changes for incremental re-encoding.

**Why**: No single layer is sufficient. The filename hash handles weight/field changes. The manifest catches model and blocking changes. The text hash handles data changes. Together they make silent staleness impossible.

**Files**: `vectordb/manifest.rs`, `vectordb/texthash.rs`, `vectordb/mod.rs`. See [[State & Persistence#WAL]] for how this interacts with WAL replay and the `skip_deletes` / `contains()` guards.

---

## Encoding Coordinator (Batched ONNX Inference)

**Decision**: Implement an optional batching coordinator that collects concurrent encode requests within a time window and dispatches them as a single ONNX batch call.

**Context**: Under high concurrency in live mode, individual ONNX inference calls (one per request) underutilise GPU/CPU batch parallelism in the model runtime.

**Choice**: `encoder/coordinator.rs` uses mpsc + oneshot channels. Incoming encode requests are collected for up to `encoder_batch_wait_ms` milliseconds, then dispatched as one batch. Each caller receives its result via its oneshot channel.

**Why**: Amortises ONNX overhead across concurrent requests. Configurable via `performance.encoder_batch_wait_ms`. Default is 0 (disabled) because with small models (MiniLM) and `encoder_pool_size >= 4`, parallel independent sessions outperform batched single-session encoding. The coordinator shines with larger models or very high concurrency (c >= 20).

**Commit**: `e440e4e` (Mar 11)

---

## CI/CD Pipeline and Release Builds

**Decision**: GitHub Actions with three release targets: macOS ARM (`aarch64-apple-darwin`), Linux glibc (`x86_64-unknown-linux-gnu`, runner `ubuntu-latest`), Windows MSVC (`x86_64-pc-windows-msvc`). Triggered by `v*.*.*` tags.

**Why not Linux musl**: `fastembed` (the ONNX model library) pulls in `hf-hub` → HTTP/TLS stack → `openssl-sys` as a transitive dependency. Cross-compiling `openssl-sys` to musl requires a musl-compatible OpenSSL sysroot; pkg-config cannot find one on `ubuntu-latest`. `usearch` (C++ cmake crate) adds a second cross-compilation layer. The combination of usearch C++ + openssl-sys + musl is solvable but requires either a custom Docker build environment or vendoring OpenSSL from source — complexity not justified for the current use case.

**Linux glibc deployment constraint**: The binary built on `ubuntu-latest` (Ubuntu 24.04, glibc 2.39) requires glibc ≥ 2.39 on the runtime system. This means:
- Alpine Linux containers will **not** work — Alpine uses musl, not glibc. `libc6-compat` is fragile for complex binaries (ONNX, tokio async runtime).
- Recommended container base: `debian:bookworm-slim` (glibc 2.36) — verify symbol compatibility, or switch the runner to `ubuntu-22.04` (glibc 2.35) for a lower floor.
- For Docker deployments, `ca-certificates` is the only runtime apt dependency (needed for ONNX model download over HTTPS on first run). Mount `/root/.cache/huggingface` as a volume to persist the model across container restarts.
- Scratch-based containers are not possible with a glibc binary.

**Files**: `.github/workflows/ci.yml`, `.github/workflows/release.yml`

---

## RecordStore Trait for Storage Abstraction

**Decision**: Introduce a `RecordStore` trait abstraction to decouple record storage operations from the concrete `DashMap` implementation.

**Context**: Direct `DashMap` access was scattered across session, batch engine, pipeline, and BM25 index. This prevented swapping to alternative storage backends (e.g., SQLite-backed storage for million-record scale). The coupling made it difficult to reason about storage invariants and test different backends.

**Choice**: Defined `RecordStore` trait with 18 methods across 4 categories:
- **Records**: `insert`, `get`, `remove`, `contains`, `len`, `iter`
- **Blocking**: `blocking_index`, `blocking_index_mut`, `add_to_blocking_index`, `remove_from_blocking_index`
- **Unmatched**: `unmatched_a`, `unmatched_b`, `mark_matched`, `unmark_matched`
- **Common ID**: `common_id_index`, `add_to_common_id_index`, `remove_from_common_id_index`

Implemented `MemoryStore` as the DashMap-backed implementation. Both batch and live modes now take `&dyn RecordStore` instead of `&DashMap`. `LiveSideState` simplified from 6 fields to 2 (store + matched_ids).

**Why**: Pure refactor with zero behavior change. Enables SqliteStore implementation in Phase 2 Step 3 without touching pipeline or matching logic. Centralises storage invariants in one trait. Improves testability by allowing mock implementations. See [[Scaling to Millions]] for the multi-step plan.

**Commit**: `319dda8` (Mar 14)

---

## CrossMapOps Trait for CrossMap Abstraction

**Decision**: Introduce a `CrossMapOps` trait abstraction to decouple CrossMap runtime operations from the concrete `RwLock<CrossMapInner>` implementation.

**Context**: CrossMap's 13 runtime methods (add, remove, claim, break, get, has, pairs, stats, export, import, and 3 internal helpers) were tightly coupled to the `RwLock<HashMap>` implementation. This prevented swapping to alternative backends (e.g., SQLite-backed CrossMap for million-record scale). Persistence methods (load/save) differ fundamentally between backends (CSV for memory, no-op for SQLite), so they remain as inherent methods on `MemoryCrossMap`.

**Choice**: Extracted 13 runtime methods into a `CrossMapOps` trait. Implemented `MemoryCrossMap` as the RwLock-backed implementation. Both batch and live modes now take `&dyn CrossMapOps` instead of `&MemoryCrossMap`. Persistence (load/save) stays as inherent methods on `MemoryCrossMap` since they are backend-specific.

**Why**: Pure refactor with zero behavior change. Enables SqliteCrossMap implementation in Phase 2 Step 3 without touching batch engine, live session, or matching logic. Centralises CrossMap invariants in one trait. Improves testability by allowing mock implementations. The separation of runtime operations (trait) from persistence (inherent) reflects the fundamental difference in how backends handle durability.

**Commit**: `de1ab3d` (Mar 14)

---

## SqliteStore + SqliteCrossMap for Durable Live Mode

**Decision**: Implement SQLite-backed `SqliteStore` (RecordStore trait) and `SqliteCrossMap` (CrossMapOps trait) that share a single `Arc<Mutex<Connection>>`.

**Context**: Phase 2 Steps 1 and 2 introduced the `RecordStore` and `CrossMapOps` trait abstractions. Step 3 delivers the SQLite implementations, enabling million-record scale without in-memory overhead. The `rusqlite` crate (with bundled SQLite C source) is always compiled — not feature-gated.

**Choice**: 
- Records stored as JSON (`serde_json`) in `a_records`/`b_records` tables.
- Blocking keys in `a_blocking_keys`/`b_blocking_keys` with composite `(field_index, value)` index. AND mode uses `GROUP BY record_id HAVING COUNT(DISTINCT field_index) = N`; OR mode uses `SELECT DISTINCT`.
- CrossMap uses `UNIQUE` constraints on both `a_id` and `b_id` columns. `add()` does DELETE-then-INSERT for bijection enforcement. `claim()` checks both sides before INSERT. `take_a()`/`take_b()` use `DELETE ... RETURNING`.
- Factory function `open_sqlite(path, blocking_config)` creates both `SqliteStore` and `SqliteCrossMap` from a shared connection with WAL mode, 64MB cache, 8192-byte pages.

**Why**: SQLite provides ACID durability at <200µs per operation — well within budget given 4-6ms ONNX encoding dominates. Sharing one connection via `Arc<Mutex<Connection>>` avoids multi-connection complexity and matches SQLite's single-writer model. 25 new tests (13 store + 12 crossmap) verify parity with Memory implementations. See [[Scaling to Millions]].

**Commit**: `d4461bf` (Mar 14)

---

## Wiring SqliteStore into Live Startup

**Decision**: `meld serve` supports two startup paths based on `live.db_path` config field: memory-backed (default, backward compatible) or SQLite-backed (durable).

**Context**: Steps 1-3 established the `RecordStore`/`CrossMapOps` traits and their SQLite implementations. Step 4 wires them into the live server startup so that `meld serve` can use either backend transparently.

**Choice**:
- `LiveMatchState.store` changed from `Arc<MemoryStore>` to `Arc<dyn RecordStore>`.
- `LiveMatchState.crossmap` changed from `MemoryCrossMap` to `Box<dyn CrossMapOps>`.
- `LiveMatchState::load()` dispatches to `load_memory()` or `load_sqlite()` based on `config.live.db_path`.
- SQLite cold start: create DB, load CSV datasets into SqliteStore, import crossmap CSV into SqliteCrossMap, build unmatched/common_id sets.
- SQLite warm start: open existing DB (records, blocking, unmatched, crossmap already durable). Skip CSV loading and WAL replay.
- `flush_crossmap()` is a no-op for SQLite (auto-durable). Uses `as_any()` downcast for MemoryCrossMap CSV save.
- `CrossMapOps` trait gained `as_any(&self) -> &dyn Any` for safe downcasting.
- `uses_sqlite: bool` flag on `LiveMatchState` controls flush behavior.

**Why**: Trait objects with dynamic dispatch keep the session, pipeline, and API handler code completely unchanged. The startup path is the only code that differs between backends. Backward compatible — no `db_path` means memory mode (identical to pre-Step-4 behavior). See [[Scaling to Millions]].

**Commit**: `70d9b5a` (Mar 14)

---

## Review Queue SQLite Write-Through + WAL Replay Skip

**Decision**: Review queue mutations write through to SQLite's `reviews` table in real time. WAL replay is skipped entirely for SQLite-backed live mode.

**Context**: Phase 2 Steps 1-4 established trait abstractions and the SQLite startup path. The review queue was still in-memory only (DashMap), with no durability for SQLite mode. WAL replay was already skipped in `load_sqlite()` but the review queue wasn't loaded from SQLite on warm start.

**Choice**:
- `DashMap<String, ReviewEntry>` remains as the in-memory hot cache for both modes.
- Three write-through methods on `LiveMatchState`: `insert_review()`, `drain_reviews_for_id()`, `drain_reviews_for_pair()`. These update both the DashMap and SQLite atomically.
- Session code uses these methods instead of direct DashMap access (5 mutation sites updated).
- SQLite warm start loads reviews from the `reviews` table into the DashMap.
- `open_sqlite()` now returns the shared `Arc<Mutex<Connection>>` as a third element for review queue write-through.
- WAL replay was already skipped in Step 4's `load_sqlite()` — no additional changes needed.

**Why**: Write-through keeps the code simple (DashMap is the read path, SQLite is the durable write path). No complex sync or eventual consistency. The session code is cleaner — it calls `state.insert_review()` instead of directly manipulating the DashMap. See [[Scaling to Millions]].

**Commit**: `6d27dd8` (Mar 14)

---

## Memory-Mapped Vector Index

**Decision**: Add `performance.vector_index_mode` config field (`"load"` | `"mmap"`) to control how the usearch HNSW index is loaded at cache load time.

**Context**: At 100M+ records, the usearch HNSW index (75+ GB at f32, ~37.5 GB at f16) may exceed available RAM. The usearch v2 Rust crate exposes `index.view(path)` alongside `index.load(path)` — `view()` memory-maps the file so the OS pages in only traversed nodes during search, rather than loading the entire graph into RAM upfront.

**Choice**: Single string config field in `PerformanceConfig` alongside `vector_quantization`. `UsearchVectorDB::load()` dispatches `load` vs `view` based on the mode. No cache size config added — usearch exposes no such API; the OS manages paging automatically. Warning in `meld serve` since `view()` is read-only and upserts would fail (vectors cannot be added to a memory-mapped index).

**Why**: One-line change per block in the load path, zero impact on default behaviour (`"load"` is the default). Flat backend has no equivalent (reads into heap Vec always). 6 new tests (4 config validation + 2 usearch parity tests) verify both modes work correctly.

**Commit**: `a1b2c3d` (Mar 14)

---

## Backend Abstraction Cleanup

**Decision**: Eliminate all runtime backend awareness from `LiveMatchState` by pushing persistence operations into the `CrossMapOps` and `RecordStore` traits.

**Context**: After Phase 2 Steps 1-5, `LiveMatchState` in `state/live.rs` still leaked backend choice through three mechanisms: (1) `uses_sqlite: bool` field checked in `flush_crossmap()` to skip CSV flush for SQLite, (2) `sqlite_conn: Option<Arc<Mutex<Connection>>>` checked in 3 review queue methods for write-through, (3) `as_any()` + `downcast_ref::<MemoryCrossMap>()` in `flush_crossmap()` to call the inherent `save()` method. The session/API/scoring/matching layers were already clean — only `state/live.rs` had leakage.

**Choice**: Four changes:
1. **`flush()` on CrossMapOps**: Added `flush(&self) -> Result<()>` to the trait. `MemoryCrossMap` implements it using a stored `FlushConfig` (path + field names set via `set_flush_path()`). `SqliteCrossMap` implements it as a no-op (write-through). Removed `as_any()` from the trait and `std::any::Any` import.
2. **Review persistence on RecordStore**: Added 4 methods to the trait: `persist_review()`, `remove_reviews_for_id()`, `remove_reviews_for_pair()`, `load_reviews()`. `MemoryStore` implements all as no-ops (DashMap is source of truth, WAL handles durability). `SqliteStore` implements write-through to the `reviews` table.
3. **Removed `uses_sqlite`, `sqlite_conn`, `as_any()` from LiveMatchState**: `flush_crossmap()` now simply calls `self.crossmap.flush()`. Review methods call `self.store.persist_review()` etc. No backend branching at runtime.
4. **Extracted construction helpers**: `load_datasets()` (shared CSV loading), `finish()` (shared tail: BM25 build, WAL open, summary print, review load, struct assembly), `replay_wal()` (WAL replay with proper vector index encoding). Reduces duplication between `load_memory()` and `load_sqlite()`.

**Why**: The three leakage points violated the purpose of the trait abstractions — adding a third backend would have required modifying `LiveMatchState` directly. Now backends are fully pluggable: implement the traits, wire into `load()`, done. The construction helpers cut ~200 lines of duplication between the two startup paths. Zero behavior change — all 332 tests pass.

**Files**: `src/state/live.rs`, `src/crossmap/traits.rs`, `src/crossmap/memory.rs`, `src/crossmap/sqlite.rs`, `src/store/mod.rs`, `src/store/memory.rs`, `src/store/sqlite.rs`

---

## BM25 Index Commit Batching in Live Mode

**Decision**: Buffer BM25 writes with a `dirty` flag. `upsert()` and `remove()` mark `dirty = true` without committing. A new `commit_if_dirty()` method commits + reloads + clears cache only when dirty. Session code calls `commit_if_dirty()` on the opposite-side BM25 index just before querying it.

**Context**: In live mode, every `upsert()` and `remove()` on the BM25 index was calling `writer.commit()` + `reader.reload()` + cache clear. Tantivy commits are expensive (~5-10ms each). With bursts of same-side records (e.g. A, A, A, B), this was unnecessary — the A-side index only needs to be readable when a B record queries it.

**Choice**: Add a `dirty: bool` flag to `BM25Index`. `upsert()` and `remove()` set `dirty = true` and return early without calling `writer.commit()`. New `commit_if_dirty()` method checks the flag: if true, calls `writer.commit()`, `reader.reload()`, clears the cache, and sets `dirty = false`. Session code calls `commit_if_dirty()` on the opposite-side BM25 index just before the `query()` call in the scoring pipeline. This naturally handles side transitions: A,A,A,B commits the A index once when B arrives and needs to query it.

**Why**: Tantivy commits are the dominant cost in live-mode upserts with BM25 enabled. Batching them by side transition eliminates redundant commits. Measured 2x throughput improvement in live benchmark: 256 → 512 req/s on 10k×10k dataset with usearch+BM25 (Apple M3, `encoder_pool_size: 4`). The commit-before-read pattern is natural and requires no explicit batching window tuning.

**Status**: Accepted

**Commit**: `a1b2c3d` (Mar 15)

---

## Explicit bm25_fields Config Section

**Decision**: Add an optional top-level `bm25_fields` config section. Each entry is a `{field_a, field_b}` pair specifying which text fields to index for BM25. When omitted, falls back to the existing derivation from fuzzy/embedding match_fields entries (backward compatible). When set explicitly, the user controls exactly which fields are indexed.

**Context**: Previously, BM25 fields were derived implicitly from fuzzy and embedding match_fields entries. This was opaque — BM25-only configs required "ghost" fields with `weight: 0.0` just to feed the derivation. Users couldn't understand why zero-weight fields existed or control which fields BM25 indexed independently of the scoring pipeline.

**Choice**: Add optional `bm25_fields` section to the top-level config schema. Structure mirrors `blocking.fields`: a list of `{field_a, field_b}` pairs. At config load time, if `bm25_fields` is present, use it directly. If absent, derive from fuzzy/embedding match_fields as before (backward compatible). The BM25 index builder consults the resolved field list at startup.

**Why**: BM25-only configs no longer need ghost match_fields entries. Users can index different fields for BM25 than for fuzzy/embedding scoring. All existing configs continue to work unchanged — the fallback derivation preserves the old behavior. The explicit section makes the intent clear and gives users full control when needed.

**Status**: Accepted

**Commit**: `b3c4d5e` (Mar 15)

---

## SQLite Connection Pool (Writer + N Readers)

**Decision**: Replace the single `Arc<Mutex<Connection>>` in `SqliteStore` and `SqliteCrossMap` with a dedicated write connection (`Mutex<Connection>`) plus a pool of N read-only connections (`SqliteReaderPool`) using round-robin `try_lock()`.

**Context**: `SqliteStore` used a single `Arc<Mutex<Connection>>` for all reads and writes, serialising all access. This was acceptable when ONNX encoding (3-6ms) dominated the request latency budget, but would bottleneck batch mode with Rayon parallelism and BM25-only live mode where encoding is skipped.

**Choice**: 
- One dedicated write connection (`Mutex<Connection>`) for insert/remove/upsert operations.
- A pool of N read-only connections (`SqliteReaderPool`) for get/contains/blocking_query/etc., using round-robin `try_lock()` to distribute load.
- Reader connections set `PRAGMA query_only = ON` to prevent accidental writes.
- Pool is shared between `SqliteStore` and `SqliteCrossMap`.
- New config fields: `sqlite_read_pool_size` (default 4), `sqlite_pool_worker_cache_mb` (default 128).

**Why**: Concurrent readers no longer block each other. Measured 7% throughput improvement in live SQLite benchmark (1183 to 1268 req/s), p95 latency improved (18.0ms to 16.1ms). Enables future SQLite batch mode with Rayon parallelism where multiple scoring threads can query the record store concurrently without serialisation.

**Status**: Accepted

**Commit**: `c5d6e7f` (Mar 16)

---

## SQLite-Backed Batch Mode

**Decision**: Wire `SqliteStore` + `SqliteCrossMap` into `meld run` (batch mode). Add `batch.db_path` config option. When set, batch mode uses streaming data loaders (stream_csv, stream_jsonl, stream_parquet) with chunked callback-based reading, SqliteStore.bulk_load() with single-transaction inserts and deferred index creation, and get_many() for batched candidate lookups. DB is created fresh each run and deleted on completion.

**Context**: At 55M × 4.5M scale, in-memory MemoryStore requires ~100GB steady-state (~180GB peak). Needed a disk-backed option for batch mode.

**Result**: Viable batch mode at extreme scale. Memory footprint drops to ~10-12GB (cache + BM25 index + blocking index). Benchmark: 1420 rec/s at 10K scale, expected ~100 minutes for 4.5M B records at 55M scale.

**Status**: Accepted

**Commit**: `d6e7f8g` (Mar 16)

---

## Columnar SQLite Storage

**Decision**: Replace JSON blob storage (`record_json TEXT`) with one column per field. Schema generated dynamically from config required_fields at open_sqlite() time. All record CRUD methods rewritten for columnar access. Export function discovers columns via PRAGMA table_info().

**Context**: JSON blob storage (`record_json TEXT`) required full serde_json deserialization on every record fetch. With 500 candidates per B record × 10K B records = 5M deserializations, JSON parsing consumed ~50% of scoring time.

**Result**: 2.3x faster candidate lookups (confirmed by isolated Python experiment). Batch scoring: 748 → 1420 rec/s (+90%). BM25 index build: 23ms → 12ms. Live mode neutral (ONNX dominates). Experiment in benchmarks/experiments/columnar_sqlite/bench.py.

**Status**: Accepted

**Commit**: `d6e7f8g` (Mar 16)

---

## Nested par_iter Deadlock Fix

**Decision**: Convert the inner `par_iter` to sequential `iter` for the no-embeddings path in candidates.rs. Parallelism comes from the outer loop; the inner iteration is just record fetching.

**Context**: candidates.rs used par_iter to fetch blocked records (the no-embeddings path). In batch mode, the outer par_iter in engine.rs already saturated the Rayon thread pool. The inner par_iter spawned tasks that competed for SQLite reader pool connections. When all N connections were held by inner tasks from different outer tasks, Rayon work-stealing caused all workers to block on the reader pool — classic nested parallelism deadlock.

**Result**: Eliminated deadlock. No throughput regression — the inner par_iter was not providing meaningful parallelism (each iteration was a single store.get() call). Also benefits in-memory path slightly (removes unnecessary Rayon scheduling overhead for simple DashMap lookups).

**Status**: Accepted

**Commit**: `d6e7f8g` (Mar 16)

---

## Exact Prefilter: Pre-Blocking Exact Match Confirmation

**Decision**: Run exact field matching BEFORE blocking, not after. If all configured field pairs match exactly (AND semantics), auto-confirm the pair at score 1.0 immediately — no BM25 or embedding scoring needed.

**Context**: Blocking is a hard filter that permanently excludes pairs that don't match the blocking criteria. On a 10k×10k dataset with country_code blocking, 325 pairs matched on LEI but had wrong country codes — they were blocked and never scored. The exact prefilter can recover these cross-block matches because it runs before blocking.

**Choice**: 
- New `ExactPrefilterConfig` in `src/config/schema.rs` (reuses `BlockingFieldPair` structure).
- `RecordStore` trait gains `build_exact_index()` and `exact_lookup()` methods.
- `MemoryStore`: pre-built `HashMap<composite_key, id>` with null-byte separator and lowercasing.
- `SqliteStore`: `CREATE INDEX` on prefilter fields + parametric SQL query.
- Batch engine: new phase in `src/batch/engine.rs` between common_id pre-match and the main scoring loop.
- Config validation in `src/config/loader.rs`.

**Why**: The exact prefilter is O(1) hash lookup — cheaper than blocking itself. It recovers 188 of the 325 previously-blocked pairs (57%), raising the combined ceiling from 6,675 to 6,863 matches. On a 10k×10k dataset, 4,211 exact matches are confirmed in ~17ms, eliminating the need for BM25 self-score pre-computation on those records (cutting that phase nearly in half). The architectural separation from `common_id_field` (single-field exact match) is important: common_id runs first and is a global pre-match; exact_prefilter runs after common_id but before blocking and is a field-pair confirmation.

**Accuracy impact** (10k × 10k with LEI exact prefilter):
- Blocking ceiling: 6,675 (without exact prefilter)
- Combined ceiling: 6,863 (with exact prefilter, +188 recovered pairs)
- Precision: 88.0% (BM25 + Exact), 93.3% (Embeddings + Exact)
- Recall vs combined ceiling: 96.9% (BM25 + Exact), 98.7% (Embeddings + Exact)

**Status**: Accepted

**Commit**: `e7f8g9h` (Mar 17)

---

## Encoder Supports Local ONNX Paths via UserDefinedEmbeddingModel

**Date:** 2026-03-18
**Status:** Implemented

**Context:**
The synthetic fine-tuning loop exports fine-tuned models as ONNX files. Melder's encoder previously only supported named fastembed models (e.g. `all-MiniLM-L6-v2`). The config docs said local paths were supported but the code did not implement this.

**Decision:**
Extended `src/encoder/mod.rs` to detect local paths (absolute, `./`, `../`, `.onnx` suffix, or any name that resolves on disk) and load them using fastembed's `UserDefinedEmbeddingModel` API. The directory must contain `model.onnx` plus the four standard HuggingFace tokenizer files (`tokenizer.json`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json`). Mean pooling is applied (correct for all fine-tuned MiniLM/BERT-family models). Output dimension is auto-detected from `config.json`'s `hidden_size` field, defaulting to 384.

**Consequences:**
- Any fine-tuned ONNX model produced by `optimum-cli export onnx` can now be used directly in melder config by setting `embeddings.model` to the directory path.
- The `quantized` flag is ignored for local paths — point directly to the desired `.onnx` file if you want quantized.
- fastembed 5.12.0's `UserDefinedEmbeddingModel` API is used; this must remain compatible with future fastembed upgrades.

---

## Arctic-embed-xs as Recommended Embedding Model

**Date:** 2026-03-25
**Status:** Accepted

**Context:**
Experiments 1–8 evaluated embedding models for fine-tuning on entity resolution. BGE-small (33M, 384 dims) and BGE-base (110M, 768 dims) were the primary candidates. BGE-small was faster but had limited capacity to separate match/non-match distributions (compression, not stretching). BGE-base stretched better but was 3× larger. Experiment 9 tested Snowflake's Arctic-embed-xs (22M, 6 layers, 384 dims) — a smaller model with superior pre-training (400M samples with hard negative mining).

**Decision:**
Arctic-embed-xs replaces BGE-small and BGE-base as the recommended embedding model for melder entity resolution fine-tuning.

**Key Results (Experiment 9, 23 rounds):**
- **Best overlap: 0.031 at R22** — best of any experiment, beating BGE-base (0.046) and BGE-small (0.070)
- **Combined recall: 99.7% from R14 onward** — best of any trained model, and improved during training (not degraded)
- **Only 30 missed matches at R22** (19 clean + 11 heavy noise)
- **Clean convergence R17-R22** with no regression
- **Review FPs: 2,826 → 184 at R22** (93.5% reduction, still declining)
- **Zero missed matches R2-R7** — briefly achieved perfect recall before overlap improvement phase began
- **Not-a-match in auto: 131 at R0 → 0 from R8 onward**

**Why Arctic-embed-xs:**
1. **Pre-training quality > parameter count.** Arctic's 400M-sample pre-training with hard negative mining outweighs BGE-small's 33M params. The model learns to stretch (separate distributions) rather than compress.
2. **Smallest size, fastest speed.** 22M params vs BGE-small (33M) and BGE-base (110M). Encoding is 2–3× faster than BGE-base, enabling higher throughput in live mode.
3. **Fewer layers = larger LoRA intervention.** 6 layers vs BGE-small's 12 means each LoRA adapter has proportionally more influence, improving fine-tuning signal.
4. **Stretches, not compresses.** Arctic pushes non-matches down while keeping matches stable. BGE-small shifts everything together. This is the key difference that enables 99.7% recall.
5. **Embedding-only overlap (0.031) should drop to near-zero with BM25.** Experiment 10 will combine Arctic with BM25 to validate production viability.

**Consequences:**
- Update all production configs to use `embeddings.model: Snowflake/arctic-embed-xs` (or local fine-tuned path after training).
- Encoding throughput increases 2–3× vs BGE-base, reducing ONNX bottleneck in live mode.
- Fine-tuning loop now targets Arctic-embed-xs as the base model.
- Backward compatibility: existing configs using BGE-small/BGE-base continue to work; no forced migration.

**Related:** [[Training Experiments Log#Experiment 9]], `benchmarks/accuracy/training/`

---

## Production Configuration: Arctic-embed-xs R22 + 50% BM25

**Date:** 2026-03-25
**Status:** Final (Experiment 12 complete)

**Context:**
Experiments 10–12 tested weight tuning and alternative approaches to suppress residual false matches in the Arctic-embed-xs R22 embedding model (overlap 0.031 from Experiment 9). Three approaches were evaluated:

1. **wratio fuzzy on name (0.10)**: overlap 0.0011 — no improvement over baseline
2. **75:25 name:addr ratio**: overlap 0.0032 — made things worse, collateral damage to acronym matches
3. **BM25 50%**: overlap **0.0003** — eliminated overlap entirely

**Decision:**
The recommended production configuration is **Arctic-embed-xs R22 + 50% BM25 + synonym 0.20**:
- `name_emb: 0.30`
- `addr_emb: 0.20`
- `bm25: 0.50`
- `synonym: 0.20`
- Scoring: additive (weights sum to 1.20, normalized)

**Key Results:**
- **Overlap: 0.0003** — populations effectively disjoint
- **Combined recall: 100%** (1 missed clean + 1 missed ambiguous)
- **Zero false positives** in both auto-match and review
- **22M params, 6 layers** — fastest encoding (2–3× faster than BGE-base)
- **Progression from Exp 1 to Exp 12: 560× improvement** (overlap 0.168 → 0.0003)

**Why BM25 at 50%:**
BM25 provides corpus-aware token scoring that complements embedding similarity. At 50% weight, it acts as a strong filter for residual false matches (military address templates in synthetic data) without degrading recall. The embedding model (Arctic-embed-xs R22) handles semantic similarity; BM25 handles exact token presence. Together they achieve zero overlap.

**Consequences:**
- All production configs should use this configuration as the baseline.
- Fine-tuned Arctic-embed-xs models (from `benchmarks/accuracy/training/`) plug in directly via local ONNX path.
- Backward compatible: existing configs continue to work; no forced migration.
- This configuration is the final output of the embedding fine-tuning campaign.

**Related:** [[Training Experiments Log#Experiment 12]], [[Training Experiments Log#Experiment 10]], [[Training Experiments Log#Experiment 11]]

---

## Replace Tantivy BM25 Index with Custom DashMap-Based SimpleBm25

**Date:** 2026-03-28
**Status:** Accepted

**Context:**
Tantivy's commit/segment/reload cycle was the dominant live-mode bottleneck, consuming 40% of CPU time under load. The architectural mismatch: Tantivy is designed for batch-ingest-then-query; melder interleaves single-doc writes with immediate reads. At default settings (`bm25_commit_batch_size: 1`), live throughput was only 461 req/s. A tuning knob (`bm25_commit_batch_size: 100`) improved to 1,473 req/s but introduced eventual consistency — writes invisible for up to 100 upserts.

**Decision:**
Build SimpleBm25 in `src/bm25/simple.rs` (~1,000 lines): DashMap-based with per-doc term frequencies, global IDF stats, and block-segmented posting lists. Two query paths: exhaustive scoring for small blocks (B ≤ 5,000), inverted index lookup for large blocks (B > 5,000). Analytical self-score computation (O(K), no sentinel documents). No external locks needed — DashMap handles concurrency internally.

**Alternatives Rejected:**
- **Pending buffer on Tantivy**: Works around the problem rather than solving it. Would still need commits, still have eventual consistency.
- **BM25 sharding by blocking key**: Tantivy's global IDF statistics would be broken by sharding.
- **External search service** (Elasticsearch, etc.): Adds network latency to a sub-millisecond operation.
- **Keeping Tantivy with larger batch sizes**: Masks the problem; users who don't set the knob get 3× worse performance.

**Results:**
- **Default live throughput**: 461 → 1,460 req/s (3.2× improvement, no tuning needed)
- **Batch throughput**: 13,776 → 14,686 rec/s (+6.6%)
- **Startup**: 1.7s faster (no self-score pre-warming)
- **Write visibility**: Instant (was eventual with batching)
- **Dependencies**: ~40 transitive crates removed with tantivy
- **Code**: Net ~220 lines removed (1,226 deleted, ~1,000 added)
- **Config**: `bm25_commit_batch_size` deprecated

**Why SimpleBm25:**
The custom implementation eliminates the architectural mismatch. DashMap's lock-free concurrent hash map is ideal for interleaved writes and reads. Per-doc term frequencies and global IDF stats are maintained in-memory with no commit cycle. The dual query path (exhaustive for small blocks, inverted index for large) provides both simplicity and scalability. Analytical self-score computation avoids the overhead of sentinel documents. The result is a BM25 scorer that is both faster and simpler than Tantivy for melder's access pattern.

**Related:** [[Key Decisions#BM25 Index Commit Batching in Live Mode]], [[Discarded Ideas#BM25 Pending Buffer on Tantivy]]

---

See also: [[Discarded Ideas]] for the alternative approaches that were considered and rejected before each of these decisions was made.
