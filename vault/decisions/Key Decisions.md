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

**Why**: One-line change per block in the load path, zero impact on default behaviour (`"load"` is the default). Flat backend has no equivalent (reads into heap Vec always). Phase 4 (`memory budget auto-configuration`) builds on this by auto-selecting mmap when RAM pressure demands it. 6 new tests (4 config validation + 2 usearch parity tests) verify both modes work correctly.

**Commit**: `a1b2c3d` (Mar 14)

---

## Memory Budget Auto-Configuration

**Decision**: Add `memory_budget: auto | "24GB"` top-level config key that auto-selects SQLite and/or mmap based on estimated footprint and available RAM.

**Context**: At 20M+ records, users shouldn't need to manually configure SQLite vs memory and load vs mmap. The budget calculator estimates footprint (500B/record + dim×2B/vector for f16) and applies a 70/30 split: 70% of budget reserved for vector index (HNSW random access has higher per-miss cost than B-tree record lookups), 30% for the record store (SQLite page cache).

**Choice**:
- New module `src/budget.rs` with four functions:
  - `parse_budget(s)` — parses `"auto"` (→ 80% of available RAM via sysinfo) or size strings like `"24GB"`, `"512MB"`
  - `available_ram()` — uses sysinfo crate to detect system RAM
  - `estimate_record_count(cache_dir, data_path, format)` — reads from CacheManifest if available (fast), else line-counts the data file (CSV/JSONL), else 0 (Parquet — can't count without loading)
  - `decide(budget_bytes, record_count, embedding_dim, n_embedding_fields, use_f16)` — returns `BudgetDecision { use_sqlite, use_mmap, sqlite_cache_bytes }`
- `MatchState.store` generalized from `Arc<MemoryStore>` to `Arc<dyn RecordStore>` (trait object)
- `MatchState._batch_sqlite_dir: Option<tempfile::TempDir>` — keeps temp SQLite alive for batch run lifetime, auto-cleaned on drop
- `open_sqlite()` gains `cache_kb: Option<u64>` parameter for dynamic cache sizing
- `load_state()` (batch mode): computes budget decision, auto-selects mmap, may create temp SQLite store
- `LiveMatchState::load()` (live mode): computes budget decision, may auto-set `live.db_path` and/or `vector_index_mode`
- Explicit settings (`live.db_path`, `performance.vector_index_mode`) take precedence over budget decisions (warn-only at validation time)
- New dependencies: `sysinfo = "0.30"`, `tempfile = "3"` (moved from dev-deps to regular deps)

**Why**: Eliminates manual tuning at scale. The 70/30 split reflects the asymmetric cost of cache misses: HNSW random-access traversal pays a higher per-miss penalty than B-tree record lookups. When count is 0 (Parquet format), no auto-configuration is applied (safe fallback to current behavior). Batch mode uses temp SQLite (no persistent DB needed). Live mode auto-generates `live.db_path` as `"{a_cache_dir}/{job_name}.db"` when budget triggers SQLite. 16 new tests (11 in budget.rs, 5 in loader.rs) verify estimation accuracy and decision correctness.

**Commit**: `d5e6f7g` (Mar 14)

---

See also: [[Discarded Ideas]] for the alternative approaches that were considered and rejected before each of these decisions was made.
