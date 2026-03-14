# Phase 2: SQLite Record Store — Detailed Plan

Reference: `PHASED_PLAN_HIGH_LEVEL.md` (Phase 2), `vault/ideas/Scaling to Millions.md`, `SCALING_TO_MILLIONS_SQLITE_STEP.md`

## Design Decisions (Agreed)

1. **Separate smaller traits**, not a god trait. `RecordStore` handles records + blocking + unmatched + common_id. `CrossMap` stays as its own abstraction with Memory/SQLite variants. Review queue stays separate.

2. **Both batch and live go through the trait** (via `MemoryStore` for batch, `SqliteStore` for live). Ensures the trait is fully exercised and enables future SQLite-for-batch at extreme scale.

3. **Pipeline functions take `&dyn RecordStore`** for record access. Cleanest separation of concerns. Dynamic dispatch overhead negligible (5-100 record lookups per query vs 4-6ms ONNX encoding).

4. **WAL stays for audit in live mode, but replay is skipped.** SQLite is the source of truth for live mode. WAL remains append-only event log. Batch mode keeps WAL+replay as-is.

5. **`rusqlite` is always compiled** (not feature-gated). ~5s build cost for SQLite C amalgamation.

## Implementation Steps

Ordered for incremental testability. Each step produces a compilable, testable state. Steps 1-2 are pure refactors (no new dependencies, no new functionality). Steps 3-5 add SQLite.

---

### Step 1: RecordStore trait + MemoryStore (pure refactor)

**Goal:** Define the `RecordStore` trait and wrap current DashMap-based storage behind `MemoryStore`. Refactor all callers. All existing tests pass identically — zero behaviour change.

**New files:**
- `src/store/mod.rs` — trait definition + `pub mod memory;`
- `src/store/memory.rs` — `MemoryStore` implementation

**Modified files:**
- `src/lib.rs` — add `pub mod store;`
- `src/state/live.rs` — `LiveSideState` uses `RecordStore` instead of raw DashMaps
- `src/state/state.rs` — `MatchState` uses `RecordStore` (batch mode)
- `src/session/mod.rs` — all record/blocking/unmatched/common_id access goes through trait
- `src/matching/pipeline.rs` — `pool_records: &DashMap<String, Record>` becomes `pool_store: &dyn RecordStore` + `pool_side: Side`
- `src/matching/candidates.rs` — same signature change
- `src/batch/engine.rs` — uses `MemoryStore` for A-side records
- `src/bm25/index.rs` — `BM25Index::build()` takes `&dyn RecordStore` instead of `&DashMap`
- `src/cli/run.rs`, `src/cli/tune.rs`, `src/cli/cache.rs` — adapt to new `MatchState` shape

#### 1a. RecordStore trait design

```rust
// src/store/mod.rs

pub mod memory;

use crate::models::{Record, Side};

/// Abstract record storage with blocking, unmatched tracking, and common ID index.
///
/// Two implementations: `MemoryStore` (DashMap-backed, used by batch mode)
/// and `SqliteStore` (rusqlite-backed, used by live mode).
pub trait RecordStore: Send + Sync {
    // --- Records ---

    /// Get a record by ID on the given side. Returns None if not found.
    fn get(&self, side: Side, id: &str) -> Option<Record>;

    /// Insert or replace a record. Returns the previous record if it existed.
    fn insert(&self, side: Side, id: &str, record: &Record) -> Option<Record>;

    /// Remove a record by ID. Returns the removed record if it existed.
    fn remove(&self, side: Side, id: &str) -> Option<Record>;

    /// Check if a record exists.
    fn contains(&self, side: Side, id: &str) -> bool;

    /// Count of records on the given side.
    fn len(&self, side: Side) -> usize;

    /// Collect all record IDs on the given side.
    fn ids(&self, side: Side) -> Vec<String>;

    // --- Blocking ---

    /// Insert a record's blocking keys into the index.
    fn blocking_insert(&self, side: Side, id: &str, record: &Record);

    /// Remove a record's blocking keys from the index.
    fn blocking_remove(&self, side: Side, id: &str, record: &Record);

    /// Query the blocking index: return candidate IDs from the given side
    /// that share blocking key values with the query record.
    fn blocking_query(&self, record: &Record, query_side: Side) -> Vec<String>;

    // --- Unmatched ---

    /// Mark a record as unmatched.
    fn mark_unmatched(&self, side: Side, id: &str);

    /// Mark a record as matched (remove from unmatched set).
    fn mark_matched(&self, side: Side, id: &str);

    /// Check if a record is in the unmatched set.
    fn is_unmatched(&self, side: Side, id: &str) -> bool;

    /// Count of unmatched records on the given side.
    fn unmatched_count(&self, side: Side) -> usize;

    /// Collect all unmatched IDs on the given side (sorted).
    fn unmatched_ids(&self, side: Side) -> Vec<String>;

    // --- Common ID index ---

    /// Insert or replace a common_id → record_id mapping.
    fn common_id_insert(&self, side: Side, common_id: &str, record_id: &str);

    /// Look up a record_id by common_id value.
    fn common_id_lookup(&self, side: Side, common_id: &str) -> Option<String>;

    /// Remove a common_id mapping.
    fn common_id_remove(&self, side: Side, common_id: &str);
}
```

**Key design notes:**

- The trait is per-store (one store holds both sides), not per-side. This matches SQLite's model (one DB file with `a_records` and `b_records` tables) and simplifies the interface.
- `blocking_query()` takes `record` + `query_side` — the implementation knows the blocking config (stored at construction time).
- `insert()` returns the old record (needed for blocking index removal before overwrite).
- The trait does NOT include iteration over records with values (no `iter() -> (id, record)` pairs). The pipeline only needs `get()` by ID. If bulk iteration is needed (e.g. for BM25 index building), we use `ids()` + `get()`.

#### 1b. MemoryStore implementation

```rust
// src/store/memory.rs

/// In-memory record store backed by DashMap, DashSet, and BlockingIndex.
///
/// Used by both batch mode (`meld run`) and as the default for live mode
/// before SQLite migration.
pub struct MemoryStore {
    a_records: DashMap<String, Record>,
    b_records: DashMap<String, Record>,
    a_blocking: RwLock<BlockingIndex>,
    b_blocking: RwLock<BlockingIndex>,
    a_unmatched: DashSet<String>,
    b_unmatched: DashSet<String>,
    a_common_ids: DashMap<String, String>,
    b_common_ids: DashMap<String, String>,
    blocking_config: BlockingConfig,
}
```

`MemoryStore` wraps the exact same data structures currently spread across `LiveSideState`, `LiveMatchState`, and `MatchState`. The implementation is a thin delegation layer — each trait method maps to the same DashMap/RwLock operation currently inlined in `session/mod.rs`.

**Constructor:** `MemoryStore::new(blocking_config: &BlockingConfig) -> Self`
**Batch helper:** `MemoryStore::from_records(records_a: HashMap<String, Record>, records_b: HashMap<String, Record>, blocking_config: &BlockingConfig) -> Self` — loads records and builds blocking indices (replaces the current init code in `state.rs` and `live.rs`).

#### 1c. Pipeline signature changes

Current:
```rust
// pipeline.rs
pub fn score_pool(
    ...
    pool_records: &DashMap<String, Record>,
    ...
)

// candidates.rs
pub fn select_candidates(
    ...
    pool_records: &DashMap<String, Record>,
    ...
)
```

New:
```rust
// pipeline.rs
pub fn score_pool(
    ...
    pool_store: &dyn RecordStore,
    pool_side: Side,  // which side of the store to read from
    ...
)

// candidates.rs
pub fn select_candidates(
    ...
    pool_store: &dyn RecordStore,
    pool_side: Side,
    ...
)
```

**Candidate struct change:** `Candidate.record` is currently populated by cloning from the DashMap during candidate selection. With the trait, `select_candidates()` calls `pool_store.get(pool_side, &id)` instead. Same semantics, different source.

#### 1d. BM25Index::build() signature change

Current: `pub fn build(records: &DashMap<String, Record>, fields: &[(String, String)], side: Side) -> Result<Self, anyhow::Error>`

New: `pub fn build(store: &dyn RecordStore, side: Side, fields: &[(String, String)]) -> Result<Self, anyhow::Error>`

The implementation changes from `records.iter()` to `store.ids(side)` + `store.get(side, &id)`.

#### 1e. LiveSideState simplification

Current `LiveSideState` has 6 fields (7 with bm25). After Step 1:

```rust
pub struct LiveSideState {
    pub combined_index: Option<Box<dyn VectorDB>>,
    #[cfg(feature = "bm25")]
    pub bm25_index: Option<Mutex<BM25Index>>,
}
```

Records, blocking, unmatched, common_id all move into the shared `RecordStore`. The store is on `LiveMatchState`:

```rust
pub struct LiveMatchState {
    pub config: Config,
    pub store: Arc<dyn RecordStore>,   // MemoryStore initially, SqliteStore later
    pub a: LiveSideState,             // just combined_index + bm25_index
    pub b: LiveSideState,
    pub crossmap: CrossMap,           // unchanged until Step 2
    pub encoder_pool: Arc<EncoderPool>,
    pub coordinator: Option<EncoderCoordinator>,
    pub wal: UpsertLog,
    pub crossmap_dirty: AtomicBool,
    pub review_queue: DashMap<String, ReviewEntry>,
}
```

Similarly, `MatchState` (batch mode) gets `store: Arc<dyn RecordStore>` replacing `records_a`/`records_b`.

#### 1f. Session refactor

Every `this_side.records.get(...)`, `this_side.blocking_index.write()...`, `this_side.unmatched.insert(...)`, `this_side.common_id_index.insert(...)` in `session/mod.rs` becomes `self.state.store.get(side, id)`, `self.state.store.blocking_insert(side, id, &record)`, `self.state.store.mark_unmatched(side, id)`, `self.state.store.common_id_insert(side, cid, id)`.

The `side()` and `opposite_side()` helpers on `LiveMatchState` still work — they return `&LiveSideState` which now only has `combined_index` and `bm25_index`.

#### 1g. Testing

- All existing tests pass unchanged (MemoryStore wraps the same data structures).
- New unit tests in `src/store/memory.rs` via `macro_rules!` test suite:
  - `insert_and_get`, `insert_replaces`, `remove_returns_old`, `contains`, `len`, `ids`
  - `blocking_insert_query_remove` (AND and OR modes)
  - `unmatched_lifecycle`
  - `common_id_lifecycle`
- The macro test suite will be reused for SqliteStore in Step 3.

**Compile check:** `cargo test --all-features` passes. `cargo clippy --all-features` clean.

---

### Step 2: CrossMap trait + MemoryCrossMap (pure refactor)

**Goal:** Extract the CrossMap's public API into a trait. Wrap the current `RwLock<CrossMapInner>` implementation as `MemoryCrossMap`. Refactor all callers.

**New files:**
- `src/crossmap/traits.rs` — `CrossMapOps` trait definition
- `src/crossmap/memory.rs` — `MemoryCrossMap` (renamed from current `CrossMap`)

**Modified files:**
- `src/crossmap/mod.rs` — re-exports trait + memory impl
- `src/state/live.rs` — `crossmap: Arc<dyn CrossMapOps>` (or `Box<dyn CrossMapOps>`)
- `src/session/mod.rs` — all `self.state.crossmap.*` calls go through trait
- `src/batch/engine.rs` — `crossmap: &dyn CrossMapOps`
- `src/cli/run.rs`, `src/cli/crossmap.rs`, `src/cli/review.rs` — adapt to trait

#### 2a. CrossMapOps trait

```rust
pub trait CrossMapOps: Send + Sync {
    fn add(&self, a_id: &str, b_id: &str);
    fn remove(&self, a_id: &str, b_id: &str);
    fn remove_if_exact(&self, a_id: &str, b_id: &str) -> bool;
    fn take_a(&self, a_id: &str) -> Option<String>;
    fn take_b(&self, b_id: &str) -> Option<String>;
    fn claim(&self, a_id: &str, b_id: &str) -> bool;
    fn get_b(&self, a_id: &str) -> Option<String>;
    fn get_a(&self, b_id: &str) -> Option<String>;
    fn has_a(&self, a_id: &str) -> bool;
    fn has_b(&self, b_id: &str) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn pairs(&self) -> Vec<(String, String)>;
}
```

**Persistence stays out of the trait.** `load()` and `save()` are constructor/serialization concerns, not runtime operations. `MemoryCrossMap` keeps its `load()`/`save()` as inherent methods. `SqliteCrossMap` won't need them (SQLite is durable).

#### 2b. MemoryCrossMap

Identical to the current `CrossMap` struct — just renamed. All existing tests move to `src/crossmap/memory.rs`.

**Compile check:** `cargo test --all-features` passes.

---

### Step 3: SqliteStore + SqliteCrossMap implementation

**Goal:** Add `rusqlite` dependency. Implement `SqliteStore` and `SqliteCrossMap`. Both pass the same test suites as their Memory counterparts.

**New files:**
- `src/store/sqlite.rs` — `SqliteStore` implementation
- `src/crossmap/sqlite.rs` — `SqliteCrossMap` implementation

**Modified files:**
- `Cargo.toml` — add `rusqlite = { version = "0.32", features = ["bundled"] }`
- `src/store/mod.rs` — add `pub mod sqlite;`
- `src/crossmap/mod.rs` — add `pub mod sqlite;`

#### 3a. SQLite schema

```sql
CREATE TABLE a_records (
    id          TEXT PRIMARY KEY,
    record_json TEXT NOT NULL
);
CREATE TABLE b_records (
    id          TEXT PRIMARY KEY,
    record_json TEXT NOT NULL
);

CREATE TABLE a_blocking_keys (
    record_id   TEXT    NOT NULL,
    field_index INTEGER NOT NULL,
    value       TEXT    NOT NULL
);
CREATE INDEX idx_a_blocking ON a_blocking_keys(field_index, value);

CREATE TABLE b_blocking_keys (
    record_id   TEXT    NOT NULL,
    field_index INTEGER NOT NULL,
    value       TEXT    NOT NULL
);
CREATE INDEX idx_b_blocking ON b_blocking_keys(field_index, value);

CREATE TABLE a_unmatched (id TEXT PRIMARY KEY);
CREATE TABLE b_unmatched (id TEXT PRIMARY KEY);

CREATE TABLE a_common_ids (
    common_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL
);
CREATE TABLE b_common_ids (
    common_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL
);

CREATE TABLE crossmap (
    a_id TEXT NOT NULL UNIQUE,
    b_id TEXT NOT NULL UNIQUE
);

CREATE TABLE reviews (
    key          TEXT PRIMARY KEY,
    id           TEXT NOT NULL,
    side         TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    score        REAL NOT NULL
);
```

#### 3b. SqliteStore internals

```rust
pub struct SqliteStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
    blocking_config: BlockingConfig,
}
```

**Key implementation notes:**

- `Mutex<Connection>` — SQLite is single-writer. The `Mutex` serializes writes. Reads also go through the same connection in WAL mode.
- `PRAGMA journal_mode = WAL` — enables concurrent reads during writes.
- `PRAGMA cache_size = -65536` — 64MB page cache.
- `PRAGMA page_size = 8192` — better B-tree efficiency for ~500-byte records.
- `PRAGMA foreign_keys = OFF` — no FK enforcement overhead.
- Records stored as JSON (`serde_json::to_string()` / `serde_json::from_str()`).
- `blocking_query()` uses parameterized SQL (see `SCALING_TO_MILLIONS_SQLITE_STEP.md` for AND/OR query patterns).
- Each trait method is one SQL statement within an implicit auto-commit transaction. No explicit transaction batching in the trait — each operation is atomic individually.

#### 3c. SqliteCrossMap

```rust
pub struct SqliteCrossMap {
    conn: Arc<Mutex<rusqlite::Connection>>,  // shared with SqliteStore
}
```

**SqliteStore and SqliteCrossMap share `Arc<Mutex<Connection>>`** via a factory function:

```rust
pub fn open_sqlite(path: &Path, blocking_config: &BlockingConfig)
    -> Result<(SqliteStore, SqliteCrossMap), ...>
{
    let conn = Connection::open(path)?;
    // CREATE TABLE IF NOT EXISTS ...
    // PRAGMAs ...
    let conn = Arc::new(Mutex::new(conn));
    Ok((
        SqliteStore { conn: Arc::clone(&conn), blocking_config: ... },
        SqliteCrossMap { conn },
    ))
}
```

#### 3d. Testing

- Run the `macro_rules!` test suite from Step 1 against `SqliteStore` (uses `tempfile::tempdir()` for the DB path).
- Run the CrossMap test suite from Step 2 against `SqliteCrossMap`.
- Additional SQLite-specific tests: persistence across close/reopen, WAL mode verification.

**Compile check:** `cargo test --all-features` passes (both Memory and SQLite tests).

---

### Step 4: Wire SqliteStore into `meld serve`

**Goal:** `meld serve` uses `SqliteStore` + `SqliteCrossMap`. First-run migration loads CSV datasets into SQLite. Subsequent startups skip data loading — SQLite is durable.

**Modified files:**
- `src/state/live.rs` — `LiveMatchState::load()` refactored for two startup paths
- `src/cli/serve.rs` — passes SQLite path to `LiveMatchState::load()`
- `src/config/schema.rs` — add `live.db_path: Option<String>` config field (default: `"{job.name}.db"`)

#### 4a. New startup sequence for live mode

```
1. Determine SQLite DB path from config (default: "{job.name}.db")
2. Check if DB file exists:
   a. EXISTS → "warm start": open DB, tables already populated
   b. NOT EXISTS → "cold start": create DB, load from CSV, populate tables
3. Init encoder pool
4. Build/load A-side combined embedding index from vector cache
5. Build/load B-side combined embedding index from vector cache
6. Reconcile: for records in SQLite but missing from vector index, re-encode.
   For vector entries whose record no longer exists in SQLite, prune.
7. Build BM25 indices from SqliteStore (if configured)
8. Open WAL (append-only, no replay)
9. Log summary
```

**Cold start (first run):**
- Create tables
- Load A and B datasets from CSV/JSONL/Parquet (using existing `data::load_dataset()`)
- Insert all records into `a_records` / `b_records` via `SqliteStore::insert()`
- Build blocking keys via `SqliteStore::blocking_insert()`
- Build unmatched sets (all records start as unmatched)
- Build common_id_index via `SqliteStore::common_id_insert()`
- Load existing crossmap from CSV → insert into `crossmap` table via `SqliteCrossMap::add()`
- Build embedding indices (as today)

**Warm start (subsequent runs):**
- Open existing DB — records, blocking, unmatched, common_id, crossmap, reviews all already populated
- Build embedding indices from vector cache (as today)
- Reconcile vectors vs SQLite records
- Skip WAL replay (SQLite is source of truth)

#### 4b. Config change

```yaml
live:
  db_path: "my_job.db"  # optional, defaults to "{job.name}.db"
```

#### 4c. Crossmap flush

With `SqliteCrossMap`, crossmap mutations are immediately durable. The background flush task and `crossmap_dirty` flag become no-ops for SQLite. `LiveMatchState::flush_crossmap()` checks the crossmap type — no-op for SQLite, CSV flush for Memory.

#### 4d. Testing

- Start `meld serve` with no DB → cold start, verify records in SQLite.
- Stop server, restart → warm start, verify records persist.
- Upsert via API, restart → verify upserted record persists.
- Remove via API, restart → verify removal persists.
- Crossmap confirm → restart → verify crossmap persists.

**Compile check:** `cargo test --all-features` passes.

---

### Step 5: Review queue in SQLite + WAL replay cleanup

**Goal:** Move review queue durability to SQLite for live mode. Remove WAL replay from live startup. WAL remains append-only for audit.

**Modified files:**
- `src/state/live.rs` — remove WAL replay logic for SQLite path
- `src/session/mod.rs` — review queue mutations write through to SQLite

#### 5a. Review queue approach

Keep `DashMap<String, ReviewEntry>` on `LiveMatchState` as the in-memory hot cache. For SqliteStore: load from `reviews` table on startup, write through on every mutation. For MemoryStore: rebuilt from WAL replay (as today).

#### 5b. WAL replay removal for live mode

In `LiveMatchState::load()`, when the store is `SqliteStore`:
- Skip the WAL replay loop entirely
- Skip the post-replay unmatched set rebuild
- Skip the review queue rebuild from WAL (load from SQLite instead)
- Still open the WAL for append-only writes

For `MemoryStore` (batch mode), WAL replay continues as-is.

#### 5c. Testing

- Live mode: upsert records, create review entries, restart → reviews survive.
- Live mode: WAL file not replayed (SQLite is source of truth).
- Batch mode: WAL replay still works as before.

---

## Cross-Cutting Concerns

### Error handling

- `RecordStore` mutating methods return `Result<T, StoreError>` for mutating operations.
- Read methods (`get`, `contains`, `len`, `ids`) return plain types (SQLite read failures are exceptional).

### Thread safety

- `MemoryStore`: DashMap (sharded locks) + RwLock (blocking). Same as today.
- `SqliteStore`: `Mutex<Connection>` serializes all access. Not a bottleneck — ONNX encoding at 4-6ms dominates.

### Migration from existing deployments

- First startup after Phase 2: cold start (no `.db` file). Loads from CSV, builds SQLite.
- WAL events from before migration are NOT replayed.
- Users should shut down cleanly before upgrading (flushes crossmap CSV).

### Performance expectations

| Operation | MemoryStore | SqliteStore (64MB cache) |
|---|---|---|
| `get()` | ~50ns | ~5-50µs |
| `blocking_query()` | ~1µs | ~10-100µs |
| `insert()` | ~1µs | ~50-200µs |
| `crossmap.claim()` | ~100ns | ~20-50µs |

All well within budget (ONNX encoding = 4-6ms per request).

---

## File Inventory

### New files (6)
1. `src/store/mod.rs` — `RecordStore` trait
2. `src/store/memory.rs` — `MemoryStore`
3. `src/store/sqlite.rs` — `SqliteStore`
4. `src/crossmap/traits.rs` — `CrossMapOps` trait
5. `src/crossmap/memory.rs` — `MemoryCrossMap`
6. `src/crossmap/sqlite.rs` — `SqliteCrossMap`

### Modified files (~17)
1. `Cargo.toml` — add `rusqlite`
2. `src/lib.rs` — add `pub mod store;`
3. `src/state/live.rs` — `LiveSideState` simplified, `LiveMatchState` refactored
4. `src/state/state.rs` — `MatchState` uses `RecordStore`
5. `src/session/mod.rs` — all data access through traits
6. `src/matching/pipeline.rs` — signature change (`&dyn RecordStore`)
7. `src/matching/candidates.rs` — signature change
8. `src/batch/engine.rs` — uses `MemoryStore`
9. `src/bm25/index.rs` — `build()` takes `&dyn RecordStore`
10. `src/crossmap/mod.rs` — re-exports trait + impls
11. `src/cli/run.rs` — adapt to new `MatchState`
12. `src/cli/serve.rs` — create `SqliteStore`
13. `src/cli/tune.rs` — adapt
14. `src/cli/cache.rs` — adapt
15. `src/cli/crossmap.rs` — adapt to `CrossMapOps`
16. `src/cli/review.rs` — adapt to `CrossMapOps`
17. `src/config/schema.rs` — add `live.db_path`

---

## Notes for Future Sessions (Anti-Compaction)

### Critical context that must survive compaction:

1. **Step ordering matters.** Steps 1-2 are pure refactors with zero new deps. Step 3 adds rusqlite. Step 4 wires it in. Step 5 cleans up WAL. Don't jump ahead.

2. **The `pool_records` parameter in `pipeline.rs:101` and `candidates.rs:61` is the most invasive signature change.** It ripples to 3 callers: `batch/engine.rs`, `session/mod.rs` (2 call sites: upsert_record_inner and try_match_inner). The new signature adds `pool_side: Side` alongside `pool_store: &dyn RecordStore`.

3. **`LiveSideState` drops from 6 fields to 2** (combined_index + bm25_index). Records, blocking, unmatched, common_id all move to the shared `RecordStore` on `LiveMatchState`.

4. **`MatchState` in `state/state.rs` also needs the same treatment** — `records_a`/`records_b` (DashMap) replaced by `store: Arc<dyn RecordStore>` (MemoryStore).

5. **CrossMap persistence: `MemoryCrossMap` keeps CSV load/save as inherent methods. `SqliteCrossMap` is auto-durable.** The `flush_crossmap()` mechanism becomes a no-op for SQLite.

6. **WAL replay stays for MemoryStore (batch), removed for SqliteStore (live).** WAL append-only writes stay in both modes for audit.

7. **Review queue stays as `DashMap` on `LiveMatchState`** — loaded from SQLite `reviews` table on startup for SqliteStore, rebuilt from WAL for MemoryStore.

8. **SqliteStore and SqliteCrossMap share `Arc<Mutex<Connection>>`** via a factory function `open_sqlite()`.

9. **The blocking index in SQLite uses parameterized queries** with `(field_index, value)` composite index. AND mode uses `GROUP BY record_id HAVING COUNT(DISTINCT field_index) = N`. OR mode uses `SELECT DISTINCT record_id ... OR ...`. See `SCALING_TO_MILLIONS_SQLITE_STEP.md`.

10. **`rusqlite` features: `["bundled"]`** — compiles SQLite from C source, no system dependency.

11. **The `RecordStore` trait is per-store (holds both sides), not per-side.** Methods take a `Side` parameter. This matches SQLite's model (one DB, two sets of tables).

12. **`blocking_query()` on the trait takes `(record, query_side)`.** The implementation knows the blocking config from construction. For MemoryStore, it delegates to `BlockingIndex::query()`. For SqliteStore, it builds the SQL query from the config's field pairs.

13. **`insert()` returns `Option<Record>`** (the previous record, if any). Needed by session's re-upsert path to remove old blocking keys before inserting new ones. With SQLite, this is `SELECT ... THEN INSERT OR REPLACE` in one transaction.

14. **All record iteration in the pipeline is by ID.** `select_candidates()` and `score_pool()` iterate `blocked_ids` (a `Vec<String>`), then call `store.get(side, id)` for each candidate. No bulk `iter()` over all records is needed in the hot path.

15. **The `MemoryStore::from_records()` constructor** replaces the scattered init code in `state/state.rs` (lines 89-164) and `state/live.rs` (lines 145-241, 199-255, 385-407). It loads HashMap records into DashMaps, builds blocking indices, builds unmatched sets, and builds common_id_index — all in one place.
