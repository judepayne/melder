# Scaling to Millions: The SQLite Step

An operational robustness improvement for live mode (`meld serve`). This document describes the concrete code changes needed to move the melder's live-mode state from in-memory data structures to SQLite, while preserving correctness and keeping the binary self-contained. Batch mode (`meld run`) is unaffected — it always uses in-memory storage for maximum speed. See also the [scaling architecture](vault/ideas/Scaling%20to%20Millions.md).

## Why SQLite

The melder is a single binary with no external dependencies. SQLite preserves that property — it links directly into the binary via `rusqlite`, no daemon, no sidecar. Its B-tree indices handle the melder's workload (indexed equality lookups for blocking, keyed record retrieval) in microseconds. Its page cache (`PRAGMA cache_size`) keeps hot index pages in process memory after first access, so repeated blocking lookups across millions of records are effectively at in-memory speed. WAL journal mode allows concurrent reads (important for Rayon-parallelised batch scoring). It scales comfortably to 50M+ rows and compiles everywhere without issue — unlike RocksDB, which pulls in a C++ build dependency with known MSVC pain on Windows.

## What Changes

In live mode, six in-memory data structures move to SQLite. The VectorDB stays in memory (or on disk via usearch's own mmap) — it's already persisted through the cache layer and is not the bottleneck. Batch mode continues using the current in-memory structures unchanged.

| Current structure | Location | SQLite replacement |
|---|---|---|
| `DashMap<String, Record>` (records) | `LiveSideState.records` | `records` table |
| `BlockingIndex` (AND/OR hash maps) | `LiveSideState.blocking_index` | `blocking_keys` table with indices |
| `DashSet<String>` (unmatched IDs) | `LiveSideState.unmatched` | `unmatched` table |
| `DashMap<String, String>` (common ID → record ID) | `LiveSideState.common_id_index` | `common_ids` table |
| `CrossMap` (RwLock over two HashMaps) | `LiveMatchState.crossmap` | `crossmap` table with unique constraints |
| `DashMap<String, ReviewEntry>` (review queue) | `LiveMatchState.review_queue` | `reviews` table |

## SQLite Schema

```sql
-- Per-side tables (one set for A, one for B, parameterised by prefix)

CREATE TABLE a_records (
    id          TEXT PRIMARY KEY,
    record_json TEXT NOT NULL          -- full record as JSON object
);

CREATE TABLE b_records (
    id          TEXT PRIMARY KEY,
    record_json TEXT NOT NULL
);

-- Blocking keys: one row per (record_id, field_index, value)
-- Supports both AND and OR modes via query pattern

CREATE TABLE a_blocking_keys (
    record_id   TEXT    NOT NULL REFERENCES a_records(id) ON DELETE CASCADE,
    field_index INTEGER NOT NULL,      -- index into config.blocking.fields
    value       TEXT    NOT NULL        -- lowercased, trimmed field value
);
CREATE INDEX idx_a_blocking ON a_blocking_keys(field_index, value);

CREATE TABLE b_blocking_keys (
    record_id   TEXT    NOT NULL REFERENCES b_records(id) ON DELETE CASCADE,
    field_index INTEGER NOT NULL,
    value       TEXT    NOT NULL
);
CREATE INDEX idx_b_blocking ON b_blocking_keys(field_index, value);

-- Unmatched sets

CREATE TABLE a_unmatched (
    id TEXT PRIMARY KEY REFERENCES a_records(id) ON DELETE CASCADE
);

CREATE TABLE b_unmatched (
    id TEXT PRIMARY KEY REFERENCES b_records(id) ON DELETE CASCADE
);

-- Common ID index (only populated when common_id_field is configured)

CREATE TABLE a_common_ids (
    common_id   TEXT PRIMARY KEY,      -- the common ID field value
    record_id   TEXT NOT NULL REFERENCES a_records(id) ON DELETE CASCADE
);

CREATE TABLE b_common_ids (
    common_id   TEXT PRIMARY KEY,
    record_id   TEXT NOT NULL REFERENCES b_records(id) ON DELETE CASCADE
);

-- CrossMap: bijection enforced by UNIQUE constraints on both columns

CREATE TABLE crossmap (
    a_id TEXT NOT NULL UNIQUE,
    b_id TEXT NOT NULL UNIQUE
);

-- Review queue

CREATE TABLE reviews (
    key          TEXT PRIMARY KEY,     -- "{side}:{id}:{candidate_id}"
    id           TEXT NOT NULL,
    side         TEXT NOT NULL,        -- "a" or "b"
    candidate_id TEXT NOT NULL,
    score        REAL NOT NULL
);
```

### Blocking queries

The current `BlockingIndex` uses two in-memory strategies:

- **AND mode**: composite key `Vec<String>` → `HashSet<String>`. A query builds the composite key and does a single hash lookup.
- **OR mode**: per-field `HashMap<String, HashSet<String>>` maps. A query looks up each field and unions the results.

With SQLite, both reduce to single indexed queries:

```sql
-- OR mode: any field matches
SELECT DISTINCT record_id FROM b_blocking_keys
WHERE (field_index = 0 AND value = ?1)
   OR (field_index = 1 AND value = ?2);

-- AND mode: all fields must match
SELECT record_id FROM b_blocking_keys
WHERE (field_index = 0 AND value = ?1)
   OR (field_index = 1 AND value = ?2)
GROUP BY record_id
HAVING COUNT(DISTINCT field_index) = 2;
```

Both are driven by the `(field_index, value)` index, which keeps them fast even at millions of rows.

### CrossMap bijection

The current `CrossMap` enforces 1:1 via a single `RwLock` over two `HashMap`s. With SQLite, the `UNIQUE` constraints on both `a_id` and `b_id` enforce the bijection at the database level. The `claim()` operation becomes:

```sql
INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?2);
-- fails with UNIQUE constraint violation if either a_id or b_id is already mapped
```

The `take_a()` / `take_b()` operations become:

```sql
-- take_a: atomically read and delete by a_id
DELETE FROM crossmap WHERE a_id = ?1 RETURNING b_id;

-- take_b: atomically read and delete by b_id
DELETE FROM crossmap WHERE b_id = ?1 RETURNING a_id;
```

Both are atomic within a single SQL statement — no TOCTOU window.

## What the WAL Becomes

Currently the WAL serves two purposes:
1. **Recovery**: replay all events on startup to reconstruct in-memory state.
2. **Event log**: audit trail of what happened.

With SQLite as the source of truth, purpose (1) disappears — SQLite is durable, and the state survives restarts without replay. The WAL becomes a pure event log: append-only, useful for auditing and debugging, but not replayed on startup.

The WAL event types stay the same. The `append()` calls stay in place. The `replay()` call in `LiveMatchState::load()` is removed.

## New Startup Sequence

Current startup (13 steps in `LiveMatchState::load()`):

1. Init encoder pool
2. Load A dataset from CSV/JSONL
3. Load B dataset from CSV/JSONL
4. Build/load A-side combined embedding index
5. Build/load B-side combined embedding index
6. Build BlockingIndex for A
7. Build BlockingIndex for B
8. Load CrossMap from CSV
9. Build unmatched sets
10. Open WAL, replay events
11. Rebuild unmatched sets after replay
12. Build common_id_index
13. Log summary

New startup:

1. **Open SQLite database** (create tables if first run)
2. Init encoder pool
3. Load/build A-side combined embedding index from vector cache
4. Load/build B-side combined embedding index from vector cache
5. **Reconcile**: for each side, compare records in SQLite against vectors in the embedding index. Re-encode any records present in SQLite but missing from the index. Prune any index entries whose record ID no longer exists in SQLite.
6. Open WAL (append-only, no replay)
7. Log summary

Steps 2-5 (load dataset from files), 6-7 (build blocking), 8 (load crossmap), 9-11 (unmatched + WAL replay), and 12 (common_id_index) all collapse — the data is already in SQLite.

### First-run migration

On first startup of `meld serve`, if the SQLite database doesn't exist:

1. Create tables
2. Load A and B datasets from CSV/JSONL files (as today)
3. Insert all records into `a_records` / `b_records`
4. Build blocking keys, unmatched sets, common_id_index in SQLite
5. Load crossmap from CSV, insert into `crossmap` table
6. Build embedding indices (as today)

Subsequent startups skip steps 2-5 — the data is already in SQLite.

## New Upsert Sequence

Current upsert in `Session::upsert_record()` touches 6+ data structures in a specific order:

1. Text-hash check → encode combined vector → upsert into VectorDB
2. Extract ID, check if existing record
3. If existing: break crossmap pair, remove from blocking index
4. WAL append
5. Insert/replace record in DashMap
6. Add to unmatched DashSet
7. Update blocking index
8. Update common_id_index, check for common ID match
9. Pipeline: blocking → candidate selection → full scoring
10. Claim loop (crossmap)

New upsert — steps 2-8 become a single SQLite transaction:

```
BEGIN IMMEDIATE;
  -- 2. Check existing
  SELECT record_json FROM a_records WHERE id = ?;

  -- 3. If existing: break crossmap pair
  DELETE FROM crossmap WHERE a_id = ? RETURNING b_id;
  INSERT INTO a_unmatched (id) VALUES (?) ON CONFLICT DO NOTHING;
  INSERT INTO b_unmatched (id) VALUES (?) ON CONFLICT DO NOTHING;

  -- 3. Remove old blocking keys
  DELETE FROM a_blocking_keys WHERE record_id = ?;

  -- 4. Insert/replace record
  INSERT OR REPLACE INTO a_records (id, record_json) VALUES (?, ?);

  -- 5. Add to unmatched
  INSERT INTO a_unmatched (id) VALUES (?) ON CONFLICT DO NOTHING;

  -- 6. Insert new blocking keys
  INSERT INTO a_blocking_keys (record_id, field_index, value) VALUES (?, ?, ?);

  -- 7. Update common_id_index
  INSERT OR REPLACE INTO a_common_ids (common_id, record_id) VALUES (?, ?);
  -- Check opposite side for common ID match
  SELECT record_id FROM b_common_ids WHERE common_id = ?;
COMMIT;
```

The vector upsert (step 1) stays outside the transaction — it's an in-memory operation on the VectorDB. The WAL append (event log) also stays outside.

Steps 9-10 (scoring + claim) are unchanged — they read from SQLite instead of DashMaps, but the logic is identical.

## New Remove Sequence

Same pattern: a single SQLite transaction replaces the multi-structure dance.

```
BEGIN IMMEDIATE;
  -- Read record for blocking key extraction
  SELECT record_json FROM a_records WHERE id = ?;

  -- Break crossmap pair if matched
  DELETE FROM crossmap WHERE a_id = ? RETURNING b_id;

  -- Remove from blocking keys (CASCADE would handle this, but explicit is clearer)
  DELETE FROM a_blocking_keys WHERE record_id = ?;

  -- Remove from common_id_index
  DELETE FROM a_common_ids WHERE record_id = ?;

  -- Remove from unmatched
  DELETE FROM a_unmatched WHERE id = ?;

  -- Remove from records
  DELETE FROM a_records WHERE id = ?;
COMMIT;
```

Vector index removal and WAL append stay outside the transaction.

## Code Changes

### `LiveSideState` simplifies from 5 fields to 2

```rust
// Before
pub struct LiveSideState {
    pub records: DashMap<String, Record>,
    pub combined_index: Option<Box<dyn VectorDB>>,
    pub blocking_index: RwLock<BlockingIndex>,
    pub unmatched: DashSet<String>,
    pub common_id_index: DashMap<String, String>,
}

// After (live mode with SQLite)
pub struct LiveSideState {
    pub combined_index: Option<Box<dyn VectorDB>>,
    pub db: Arc<RecordStore>,  // SQLite-backed
}
```

### `LiveMatchState` simplifies

```rust
// Before
pub struct LiveMatchState {
    pub config: Config,
    pub a: LiveSideState,
    pub b: LiveSideState,
    pub crossmap: CrossMap,
    pub encoder_pool: Arc<EncoderPool>,
    pub coordinator: Option<EncoderCoordinator>,
    pub wal: UpsertLog,
    pub crossmap_dirty: AtomicBool,
    pub review_queue: DashMap<String, ReviewEntry>,
}

// After (live mode with SQLite)
pub struct LiveMatchState {
    pub config: Config,
    pub a: LiveSideState,
    pub b: LiveSideState,
    pub encoder_pool: Arc<EncoderPool>,
    pub coordinator: Option<EncoderCoordinator>,
    pub wal: UpsertLog,
    // crossmap, crossmap_dirty, review_queue all move into RecordStore
}
```

### New `RecordStore` module

A new module `src/store/` with a trait and two implementations:

```rust
/// Abstract record storage, blocking, crossmap, and review operations.
pub trait RecordStore: Send + Sync {
    // Records
    fn get_record(&self, side: Side, id: &str) -> Option<Record>;
    fn insert_record(&self, side: Side, id: &str, record: &Record);
    fn remove_record(&self, side: Side, id: &str) -> Option<Record>;
    fn contains_record(&self, side: Side, id: &str) -> bool;
    fn record_count(&self, side: Side) -> usize;

    // Blocking
    fn blocking_query(&self, record: &Record, query_side: Side) -> Vec<String>;
    fn blocking_insert(&self, side: Side, id: &str, record: &Record);
    fn blocking_remove(&self, side: Side, id: &str, record: &Record);

    // Unmatched
    fn is_unmatched(&self, side: Side, id: &str) -> bool;
    fn mark_unmatched(&self, side: Side, id: &str);
    fn mark_matched(&self, side: Side, id: &str);
    fn unmatched_ids(&self, side: Side) -> Vec<String>;
    fn unmatched_count(&self, side: Side) -> usize;

    // Common ID
    fn common_id_insert(&self, side: Side, common_id: &str, record_id: &str);
    fn common_id_lookup(&self, side: Side, common_id: &str) -> Option<String>;
    fn common_id_remove(&self, side: Side, common_id: &str);

    // CrossMap
    fn crossmap_add(&self, a_id: &str, b_id: &str);
    fn crossmap_remove(&self, a_id: &str, b_id: &str);
    fn crossmap_claim(&self, a_id: &str, b_id: &str) -> bool;
    fn crossmap_take_a(&self, a_id: &str) -> Option<String>;
    fn crossmap_take_b(&self, b_id: &str) -> Option<String>;
    fn crossmap_get_b(&self, a_id: &str) -> Option<String>;
    fn crossmap_get_a(&self, b_id: &str) -> Option<String>;
    fn crossmap_has_a(&self, a_id: &str) -> bool;
    fn crossmap_has_b(&self, b_id: &str) -> bool;
    fn crossmap_len(&self) -> usize;
    fn crossmap_pairs(&self) -> Vec<(String, String)>;

    // Reviews
    fn review_insert(&self, key: &str, entry: &ReviewEntry);
    fn review_remove_by_id(&self, id: &str);
    fn review_remove_by_pair(&self, a_id: &str, b_id: &str);
    fn review_list(&self) -> Vec<ReviewEntry>;
    fn review_count(&self) -> usize;
}
```

Two implementations:
- `MemoryStore` — wraps the current DashMap/HashSet/RwLock structures. Used by `meld run` (batch mode).
- `SqliteStore` — wraps a `rusqlite::Connection` behind a `Mutex`. Used by `meld serve` (live mode).

### Files that change

| File | Change |
|---|---|
| `src/store/mod.rs` | **New.** `RecordStore` trait, `pub mod memory; pub mod sqlite;` |
| `src/store/memory.rs` | **New.** `MemoryStore` impl wrapping current data structures |
| `src/store/sqlite.rs` | **New.** `SqliteStore` impl using `rusqlite` |
| `src/state/live.rs` | `LiveSideState` drops to 2 fields. `LiveMatchState` drops crossmap/review_queue/crossmap_dirty. `load()` refactored for two startup paths. |
| `src/session/mod.rs` | All record/blocking/crossmap/unmatched/review operations go through `RecordStore` trait methods instead of direct DashMap/RwLock access. |
| `src/matching/blocking.rs` | `BlockingIndex` stays for `MemoryStore` (batch mode); `SqliteStore` (live mode) does blocking via SQL queries. `apply_blocking()` (batch mode) unchanged. |
| `src/crossmap/mod.rs` | `CrossMap` stays for `MemoryStore` (batch mode); `SqliteStore` (live mode) does crossmap via SQL. |
| `src/state/upsert_log.rs` | `replay()` called in batch mode only (MemoryStore). Live mode (SqliteStore) skips replay — SQLite is durable. |
| `src/api/handlers.rs` | Minor — access patterns change from `state.a.records.get()` to `state.store.get_record()`. |
| `src/batch/engine.rs` | Same access pattern changes for batch mode record iteration. Uses `MemoryStore` — no SQLite involvement. |
| `src/lib.rs` | Add `pub mod store;` |
| `Cargo.toml` | Add `rusqlite` as a dependency (always compiled in — ~5s build cost for the SQLite C amalgamation). |

## Performance Considerations

**Page cache.** SQLite's default page cache is 2 MB. For the melder's workload (repeated blocking lookups across the same index pages), setting `PRAGMA cache_size = -64000` (64 MB) ensures the working set stays in memory. The actual records are only fetched when needed for scoring — the blocking query returns just IDs.

**WAL mode.** `PRAGMA journal_mode = WAL` enables concurrent reads while a write transaction is in progress. This matters for live mode when concurrent HTTP requests read records and blocking keys while another request's upsert transaction is in progress.

**Transaction batching.** Each upsert is one transaction (the sequence described above). SQLite handles ~50,000 transactions per second on modern hardware. The melder's live-mode throughput ceiling is ~1,000 upserts/sec (dominated by ONNX encoding at ~4-6ms per request), so SQLite's transaction throughput is not the bottleneck.

**Record serialisation.** Records are stored as JSON in a TEXT column. This is simple but means every record read requires a `serde_json::from_str()` deserialisation. For the live-mode scoring pipeline (which reads 5-100 candidate records per query), this is negligible — ONNX encoding at 4-6ms per request dominates.

## Migration Path

1. **Implement `MemoryStore`** wrapping the current data structures. Refactor `Session` and `LiveMatchState` to go through the `RecordStore` trait. Both `meld run` and `meld serve` use `MemoryStore`. All existing tests pass — behaviour is identical.

2. **Implement `SqliteStore`** and wire it into `cli/serve.rs`. Batch mode (`cli/run.rs`) continues using `MemoryStore` — no changes needed. Test with the existing benchmark suite's live-mode tests.

3. **Shadow mode** (optional): run both stores in parallel in live mode during development, asserting they produce identical results for every operation.

4. **Remove WAL replay from live startup** — SQLite is the source of truth. WAL becomes append-only event log. Batch mode retains WAL replay for `MemoryStore`.

## What This Document Doesn't Cover

- **BM25 candidate selection** — independent of storage. See [Scaling to Millions](vault/ideas/Scaling%20to%20Millions.md).
- **Batch mode** — always uses `MemoryStore`. No SQLite involvement, no performance change.
- **Distributed matching** — beyond scope. See the 10M+ note in the backlog.
