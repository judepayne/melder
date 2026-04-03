---
type: decision
module: general
status: active
tags: [adr, implementation, history]
related_code: [src/state/live.rs, src/crossmap/, src/store/, src/bm25/simple.rs, src/matching/pipeline.rs]
---

# Implementation Decisions Log

Detailed implementation ADRs moved from [[decisions/key_decisions]] for context efficiency. These record *how* features were built, not *what* the current architecture is. Useful for debugging or understanding why code is structured a specific way.

---

## SqliteStore + SqliteCrossMap for Durable Live Mode

**Commit**: `d4461bf` (Mar 14)

Records stored as JSON (`serde_json`) in `a_records`/`b_records` tables. Blocking keys in `a_blocking_keys`/`b_blocking_keys` with composite `(field_index, value)` index. AND mode uses `GROUP BY record_id HAVING COUNT(DISTINCT field_index) = N`. CrossMap uses `UNIQUE` constraints on both `a_id` and `b_id` columns. `add()` does DELETE-then-INSERT for bijection enforcement. `claim()` checks both sides before INSERT. Factory function `open_sqlite(path, blocking_config)` creates both `SqliteStore` and `SqliteCrossMap` from a shared connection with WAL mode, 64MB cache, 8192-byte pages. 25 new tests.

---

## Wiring SqliteStore into Live Startup

**Commit**: `70d9b5a` (Mar 14)

`LiveMatchState.store` changed from `Arc<MemoryStore>` to `Arc<dyn RecordStore>`. `LiveMatchState.crossmap` changed from `MemoryCrossMap` to `Box<dyn CrossMapOps>`. `LiveMatchState::load()` dispatches to `load_memory()` or `load_sqlite()` based on `config.live.db_path`. SQLite cold start: create DB, load CSV, import crossmap CSV. SQLite warm start: open existing DB (no WAL replay). `flush_crossmap()` is a no-op for SQLite via `CrossMapOps::flush()` trait method.

---

## Review Queue SQLite Write-Through + WAL Replay Skip

**Commit**: `6d27dd8` (Mar 14)

`DashMap<String, ReviewEntry>` remains as in-memory hot cache. Three write-through methods on `LiveMatchState`: `insert_review()`, `drain_reviews_for_id()`, `drain_reviews_for_pair()`. Session code calls these instead of direct DashMap access. SQLite warm start loads reviews from `reviews` table into DashMap. WAL replay skipped for SQLite path — SQLite state is already durable.

---

## Backend Abstraction Cleanup

**Commit**: Mar 14. See [[decisions/key_decisions#Backend Abstraction Cleanup]] for the summary.

Detailed changes:
1. `flush()` on `CrossMapOps` trait — `MemoryCrossMap` writes CSV; `SqliteCrossMap` is no-op.
2. 4 review persistence methods on `RecordStore` trait — `MemoryStore` no-ops; `SqliteStore` writes through.
3. Removed `uses_sqlite`, `sqlite_conn`, `as_any()` from `LiveMatchState`.
4. Extracted `load_datasets()`, `replay_wal()`, `finish()` construction helpers shared by both startup paths.

Files: `src/state/live.rs`, `src/crossmap/traits.rs`, `src/crossmap/memory.rs`, `src/crossmap/sqlite.rs`, `src/store/mod.rs`, `src/store/memory.rs`, `src/store/sqlite.rs`.

---

## BM25 Index Commit Batching in Live Mode

**Status**: Superseded by SimpleBm25 (which has no commit cycle at all).

Previous Tantivy implementation buffered writes with a `dirty` flag. `commit_if_dirty()` committed only when the opposite side queried. Measured 2× throughput improvement (256→512 req/s). Superseded when Tantivy was replaced — SimpleBm25 has instant write visibility and no commit overhead.

See [[ideas/discarded_ideas#BM25 Pending Buffer on Tantivy]].

---

## Columnar SQLite Storage

**Commit**: `d6e7f8g` (Mar 16)

Replaced JSON blob `record_json TEXT` with one column per field. Schema generated dynamically from config `required_fields` at `open_sqlite()` time. All CRUD methods rewritten for columnar access. Export uses `PRAGMA table_info()` to discover columns.

**Result**: 2.3× faster candidate lookups. Batch scoring: 748→1420 rec/s (+90%). BM25 index build: 23ms→12ms.

---

## SQLite-Backed Batch Mode

**Commit**: `d6e7f8g` (Mar 16). Added `batch.db_path` config option.

When set: streaming data loaders (stream_csv, stream_jsonl, stream_parquet) with chunked callback-based reading; `SqliteStore.bulk_load()` with single-transaction inserts and deferred index creation; `get_many()` for batched candidate lookups. DB created fresh each run, deleted on completion. Memory footprint drops to ~10-12GB vs ~100GB in-memory. Benchmark: 1420 rec/s at 10K scale.

---

## SQLite Connection Pool (Writer + N Readers)

**Commit**: `c5d6e7f` (Mar 16). New config: `sqlite_read_pool_size` (default 4), `sqlite_pool_worker_cache_mb` (default 128).

One dedicated write connection (`Mutex<Connection>`) + N read-only connections (`SqliteReaderPool`) with round-robin `try_lock()`. Reader connections set `PRAGMA query_only = ON`. Pool shared between `SqliteStore` and `SqliteCrossMap`. **Why separate connections**: `rusqlite::Connection` is `Send` but not `Sync` — concurrent calls on the same handle unsafe. **Why not RwLock**: Each connection must have exclusive access via Mutex; concurrency comes from multiple connections.

**Result**: 7% live SQLite improvement (1183→1268 req/s), p95: 18.0ms→16.1ms.

---

## Nested par_iter Deadlock Fix

**Commit**: `d6e7f8g` (Mar 16).

`candidates.rs` used `par_iter` to fetch blocked records (no-embeddings path). In batch mode, outer `par_iter` in `engine.rs` already saturated Rayon thread pool. Inner `par_iter` competed for SQLite reader pool connections — all N connections held by inner tasks from different outer tasks caused classic nested parallelism deadlock.

**Fix**: Convert inner `par_iter` to sequential `iter` for the no-embeddings path. Parallelism comes from the outer loop; inner iteration is just record fetching. No throughput regression.

---

## Cursor-Based Pagination

**Date:** 2026-03-29. **Breaking change for API clients using offset/limit.**

`crossmap_pairs`, `unmatched_records`, `review_list` endpoints replaced offset/limit with cursor-based pagination. Old implementation cloned + sorted ALL entries on every request — O(N) allocation + O(N log N) sort per page. Cursor is an opaque base64-encoded string. Request: omit cursor for first page, pass `next_cursor` from response for subsequent pages. Response: `next_cursor` (null on last page).

---

## WAND Scoring Uses BinaryHeap

**Date:** 2026-03-29.

WAND scorer initial implementation used `Vec<OrdScore>` sorted on every insertion — O(K log K) per insert. Replaced with `std::collections::BinaryHeap<Reverse<OrdScore>>` for O(log K) per insert. Custom `OrdScore` wrapper for total ordering on f64. At scale (thousands of candidates), reduces WAND scoring time by 10-20%.

---

## Batch Pre-Match Uses claim() Not add()

**Date:** 2026-03-29.

Common ID pre-match and exact prefilter phases used `crossmap.add()` (bypasses bijection enforcement). Under Rayon parallelism, two B records with the same common ID could both overwrite the same A mapping; `has_a()` guard had a TOCTOU race. Fixed by replacing all `add()` calls in pre-match phases with `claim()` (atomic check-and-insert). Removed `has_a()` guard from exact prefilter (redundant with `claim()`).

---

## confirm_match Breaks Existing Pairs Before Inserting

**Date:** 2026-03-29.

Manual `confirm_match` used `crossmap.add()` which silently overwrote existing mappings, orphaning old partners. Fix: before inserting, check if `a_id` or `b_id` already claimed. If claimed, call `take_a(a_id)`/`take_b(b_id)` to get old partner, mark old partner as unmatched, then `claim(a_id, b_id)`.

---

## WAL Compact Holds Writer Lock for Entire Operation

**Date:** 2026-03-29.

Old `compact()` flushed buffer, read events, wrote temp file, renamed, then re-acquired writer lock. Any concurrent `append()` between flush and lock re-acquisition would write to the old file handle, silently losing events.

**Fix**: Hold `Mutex<File>` writer lock for the entire compaction (flush → read → write temp → rename → swap writer). Compaction is infrequent; lock contention is minimal.

---

## CompactIdMap Uses DashMap entry() for Atomic ID Assignment

**Date:** 2026-03-29.

`CompactIdMap::get_or_insert()` used get()-then-insert() pattern. Under concurrent BM25 upserts, two threads could both miss the get() check and allocate different compact IDs — second insert overwrote first in `str_to_u32`, causing silent data loss in posting lists.

**Fix**: Replace with `DashMap::entry().or_insert_with()` for atomic get-or-create semantics.

---

## BlockingIndex Partial-Key Filter

**Date:** 2026-03-29.

`BlockingIndex::query()` in live mode returned ALL pool records when any single blocking field was empty — defeating blocking. Batch-mode `passes_blocking()` correctly skips empty constraints and filters on remaining fields.

**Fix**: `BlockingIndex::query()` now skips empty blocking field values; only non-empty fields are used as constraints. Matches batch-mode semantics exactly.

---

## SessionError Split into Typed Variants

**Date:** 2026-03-29.

`SessionError::MissingField` was overloaded for 5+ semantically different errors; all mapped to HTTP 400. Split into: `MissingField` → 400, `NotFound` → 404, `BatchValidation` → 422, `Encoder` → 500. Added `status_code()` method and `run_blocking_session` handler helper. HTTP handlers now call `e.status_code()`.

---

## RecordStore Trait Returns Result<T, StoreError>

**Date:** 2026-03-29.

All 26 `RecordStore` trait methods previously returned bare types; `SqliteStore` wrapped all SQL in `.expect()` (35 total). Changed all methods to `Result<T, StoreError>`. `MemoryStore` wraps returns in `Ok()`. `SqliteStore` replaces `.expect()` with `?`. New `StoreError` enum with `From<rusqlite::Error>`. Eliminates all potential server panics from SQL errors (disk full, WAL corruption, etc.).

---

## A1: Parameter Object for score_pool()

**Date:** 2026-03-30.

`score_pool()` had 14 parameters. Introduced `ScoringQuery` (id, record, side, combined_vec) and `ScoringPool` (store, side, combined_index, blocked_ids, bm25_candidate_ids, bm25_scores_map, synonym_candidate_ids, synonym_dictionary) structs. New signature: `score_pool(query: &ScoringQuery, pool: &ScoringPool, config: &Config, ann_candidates: usize, top_n: usize)`. Removed `#[allow(clippy::too_many_arguments)]`.

---

## A3: Load-Time Rejection of Unknown Scorer/Method Names

**Date:** 2026-03-30.

Invalid scorer/method names were previously accepted at config load time and rejected at runtime with a `warn!` fallback. Fixed with Rust enums + serde deserialization: `MatchMethod` enum (`Exact`, `Fuzzy`, `Embedding`, `Numeric`, `Bm25`, `Synonym`) and `FuzzyScorer` enum (`Wratio`, `PartialRatio`, `TokenSort`, `Ratio`) with `#[serde(rename_all = "snake_case")]`. `FuzzyScorer::TokenSort` has `#[serde(alias = "token_sort_ratio")]` for backward compat. Removed `VALID_METHODS`, `VALID_SCORERS` constants and `require_one_of` calls.

---

## A4: Lock Ordering Documentation for UsearchVectorDB

**Date:** 2026-03-30.

Added `## Lock ordering` to `src/vectordb/usearch_backend.rs` module doc comment. Canonical order: (1) block_router RwLock, (2) blocks DashMap, (3) blocks[i] RwLock, (4) record_block RwLock, (5) text_hashes DashMap. `next_key` AtomicU64 is lock-free, can be accessed at any point.

---

## A5: DashMap TOCTOU Race Fix in BM25 decrement_stats()

**Date:** 2026-03-30.

Two TOCTOU races in `SimpleBm25::decrement_stats()`:
1. `doc_freq.get_mut()` then separate `doc_freq.remove()` — could delete a newly-inserted entry
2. `postings.get_mut()`, drop reference, `postings.remove()` — same issue

**Fix**: Replace both with `DashMap::remove_if_mut()` for atomic check-and-remove.

---

## A6: Batch Error Handling Consistency

**Date:** 2026-03-30.

Standardized error handling across all batch endpoints:
- `enroll_batch`: changed fail-fast (`?`) to per-item error collection; added empty/MAX_BATCH_SIZE guards
- `enroll()` WAL append: changed `let _ =` to `if let Err(e) { warn!() }`
- `remove_batch`: distinguish `NotFound` (→ "not_found") from actual errors (→ "error: ...")
- HTTP handlers: 7 hardcoded `StatusCode::BAD_REQUEST` changed to `e.status_code()`

---

## Dedicated Rayon Pool for Encoding

**Date:** 2026-04-02.

GPU encoding commit changed `encode_and_upsert` to `par_chunks` on Rayon global pool. Deadlock: with fewer encoder slots than Rayon workers, blocked workers starved ONNX's internal Rayon tasks.

**Fix**: `rayon::ThreadPoolBuilder` in `encode_and_upsert` with `num_threads = encoder_pool.pool_size()`. Scoped pool used for parallel encoding chunks. Global pool remains available for batch scoring.

---

See also: [[decisions/key_decisions]] for current architectural principles and major design choices.
