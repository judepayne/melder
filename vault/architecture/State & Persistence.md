---
type: architecture
module: state
status: active
tags: [state, persistence, wal, restart, cache, crossmap]
related_code: [src/state/live.rs, src/state/upsert_log.rs, src/crossmap/mod.rs, src/vectordb/mod.rs]
---

# State & Persistence

How Melder persists state to disk, survives restarts, and replays history. Relevant to live mode only — batch mode is stateless. See [[Business Logic Flow#Live Mode]] for where these mechanisms fit in the upsert flow.

---

## Files Written to Disk

| File | Format | Written when | Notes |
|---|---|---|---|
| `crossmap.csv` (config: `cross_map.path`) | CSV with configured column headers | Periodically (default every 5s when dirty) + graceful shutdown | Atomic write via temp file + rename |
| `{a_cache_dir}/combined_a_{spec_hash}.idx` | Backend-specific binary | At graceful shutdown (and initial build) | Spec hash encodes field names, order, weights, quantization |
| `{b_cache_dir}/combined_b_{spec_hash}.idx` | Same | Same | Only if `b_cache_dir` configured |
| `{wal_base}_{timestamp}.ndjson` | Newline-delimited JSON | Continuously — each server run creates a new timestamped file | Append-only; BufWriter, not fsynced per-write |

**Old cache files are never automatically cleaned up.** Changing config (field names, weights, quantization) produces a new spec hash and a new filename; the old file stays on disk and must be removed manually if disk space matters.

---

## WAL

### File Naming

Given `live.upsert_log: "state/events.wal"`, each server run creates a new file:
```
state/events_20260311T184207Z.wal
```
The timestamp is the server startup time in compact ISO-8601 UTC (colons/hyphens stripped). Each run gets its own file, so historical events are preserved across restarts.

On replay, **all files** matching `{stem}*{ext}` in the same directory are found and sorted lexicographically (which equals chronological order given the timestamp format). Legacy non-timestamped files (from before this naming scheme) are replayed first for backward compatibility.

---

### WAL Event Types (NDJSON)

Each line is a JSON object with a `ts` field (ISO-8601 UTC, second resolution) and a `type` discriminant.

#### `upsert_record`
```json
{ "ts": "2026-03-11T18:42:07Z", "type": "upsert_record", "side": "a", "record": { "id": "ENT-1", "name": "Acme Ltd" } }
```
Emitted on every `/add` call. Side is `"a"` or `"b"`.

#### `remove_record`
```json
{ "ts": "...", "type": "remove_record", "side": "a", "id": "ENT-1" }
```

#### `crossmap_confirm`
```json
{ "ts": "...", "type": "crossmap_confirm", "a_id": "ENT-1", "b_id": "CP-42", "score": 0.91 }
```
`score` is absent for manual confirms (via `/crossmap/confirm`) and old WAL files — handled gracefully.

#### `review_match`
```json
{ "ts": "...", "type": "review_match", "id": "ENT-1", "side": "a", "candidate_id": "CP-5", "score": 0.72 }
```
Emitted when a result lands in the review band. No state change on replay — used only to reconstruct the review queue.

#### `crossmap_break`
```json
{ "ts": "...", "type": "crossmap_break", "a_id": "ENT-1", "b_id": "CP-42" }
```

#### `exclude`
```json
{ "ts": "...", "type": "exclude", "a_id": "ENT-1", "b_id": "CP-42" }
```
Emitted when a pair is added to the exclusions set via `POST /api/v1/exclude`.

#### `unexclude`
```json
{ "ts": "...", "type": "unexclude", "a_id": "ENT-1", "b_id": "CP-42" }
```
Emitted when a pair is removed from the exclusions set via `DELETE /api/v1/exclude`.

---

### WAL Replay Mechanics

- Truncated last lines (crash mid-write) are tolerated — malformed JSON lines are **skipped with a warning**, not fatal.
- `ts` is ignored during replay.
- `upsert_record` → update DashMap + BlockingIndex; re-encode vector **only if the ID is not already in the cached index** (`idx.contains(id)`). This `contains()` guard is what makes WAL replay fast — vectors loaded from the cache are not re-encoded.
- `crossmap_confirm` → `crossmap.add()` (unconditional; replay is authoritative, not competitive like `claim()`).
- `crossmap_break` → `crossmap.remove()`.
- `remove_record` → remove from DashMap, BlockingIndex, combined index; break any associated CrossMap pair.
- `exclude` → add pair to exclusions set.
- `unexclude` → remove pair from exclusions set.
- `review_match` → no live state change; used only for review queue reconstruction at the end of startup.

---

### WAL Compaction

Compaction operates on the **current run's WAL file only** (not all historical files). Call `UpsertLog::compact()` to trigger.

Algorithm:
1. Flush BufWriter to OS
2. Read all events from the current file
3. **`upsert_record`:** last-write-wins per `(side, id)` key — 100 upserts of the same record collapse to 1
4. **`remove_record`:** supersedes any prior `upsert_record` for the same key; only the final remove is kept
5. **`crossmap_confirm`, `crossmap_break`, `review_match`:** all kept in order (no deduplication)
6. Write compacted events to `{wal_path}.wal.tmp`
7. Atomic rename (Unix: `fs::rename`; Windows: remove-then-rename)
8. Reopen file for appending (old handle is stale after rename)

Compaction is not called automatically — invoke it manually when WAL file size becomes a concern.

---

## CrossMap Persistence

The CrossMap is flushed to disk by a background task every `crossmap_flush_secs` seconds (default: 5) whenever the `crossmap_dirty` flag is set. The flag is set on every successful `claim()`, manual confirm, or break.

Flushing now goes through the `CrossMapOps::flush()` trait method:
- `MemoryCrossMap::flush()` writes to CSV via temp file + atomic rename (same as before).
- `SqliteCrossMap::flush()` is a no-op — writes are already durable (write-through to the `crossmap` table).

**Load:** missing file → empty CrossMap (not an error, normal on first run). Header-only or missing expected columns → empty CrossMap (silent, not fatal).

**Distinction between `add` and `claim` in the CrossMap:**
- `claim(a_id, b_id)` — used during live scoring. Checks both directions are vacant, inserts only if both free, returns `false` if either taken. Competitive.
- `add(a_id, b_id)` — used during WAL replay. Unconditional insert in both directions. Authoritative.

This distinction ensures that during WAL replay, already-confirmed pairs are always restored even if a competing pair was also recorded.

---

## Startup Sequence

`state/live.rs::LiveMatchState::load()` — 13 steps executed synchronously before the HTTP server starts accepting requests.

| Step | What happens |
|---|---|
| 1 | Init encoder pool — create `encoder_pool_size` ONNX sessions |
| 2 | Load A dataset — read CSV/JSONL/Parquet into `HashMap<id, Record>` |
| 3 | Load B dataset — same |
| 4 | Build/load A combined embedding index — `skip_deletes: true` (WAL replay may re-add records; retain cache entries until confirmed) |
| 5 | Build/load B combined embedding index — same; only if embedding fields configured |
| 6 | Build A BlockingIndex — iterate all A records |
| 7 | Build B BlockingIndex — iterate all B records |
| 8 | Load CrossMap from `crossmap.csv` — silent empty CrossMap if file absent |
| 9 | Load exclusions from CSV — silent empty set if file absent or not configured |
| 10 | Build unmatched sets — A and B record IDs not present in CrossMap |
| 11 | Open WAL + replay all WAL files — chronological order; `contains()` guard skips re-encoding cached vectors; `exclude`/`unexclude` events update exclusions set |
| 12 | Rebuild unmatched sets — CrossMap may have changed during WAL replay |
| 13 | Build `common_id_index` — reverse index `common_id_value → record_id` per side (only if `common_id_field` configured) |
| 14 | Rebuild review queue — re-hydrate `review_match` WAL events; drain entries superseded by confirms, breaks, or removes |

**After startup (separate step):** `init_coordinator()` — spawns the background encoding coordinator task. Must be called from within a tokio runtime after `Arc<LiveMatchState>` is built. Only active if `encoder_batch_wait_ms > 0`.

The `skip_deletes: true` flag at steps 4–5 is the key to fast restarts: vectors that were encoded during previous runs and saved to the cache file are retained in-place. WAL replay then adds only the delta (new records since last shutdown), with the `contains()` guard skipping re-encoding for anything already indexed.

### Startup Path Selection: Memory vs. SQLite

`LiveMatchState::load()` now dispatches to `load_memory()` or `load_sqlite()` based on `config.live.db_path`. Common logic is extracted into helper functions:
- `load_datasets()` — shared CSV/JSONL/Parquet loading for both paths
- `replay_wal()` — WAL replay with proper vector index encoding (encodes vectors for WAL-added records not in cache)
- `finish()` — shared tail: BM25 build, WAL open, summary print, review load from store, struct assembly

**Memory path** (`load_memory()`): Follows the 13-step sequence above. WAL replay is active and reconstructs state from the log.

**SQLite path** (`load_sqlite()`): When `live.db_path` is set, startup follows the SQLite path: open existing DB (warm start) or create + populate from CSV (cold start). WAL replay is skipped — SQLite state is already durable. Common logic (dataset loading, BM25 build, WAL open, review load) is shared via extracted helpers.

---

## LiveSideState Structure

Each side (`a` and `b`) is a `LiveSideState`:

| Field | Type | Notes |
|---|---|---|
| `store` | `Arc<dyn RecordStore>` | Trait object encapsulating records, blocking index, unmatched set, and common_id_index. Implementations: `MemoryStore` (in-memory), `SqliteStore` (SQLite-backed). |
| `combined_index` | `Option<Box<dyn VectorDB>>` | None if no embedding fields configured |

`LiveSideState` has been simplified to use a `RecordStore` trait object instead of individual `DashMap`/`RwLock`/`DashSet` fields. The store abstraction encapsulates:
- Records (`DashMap<String, Record>` in memory, `records` table in SQLite)
- Blocking index (`RwLock<BlockingIndex>`)
- Unmatched set (`DashSet<String>`)
- Common ID index (`DashMap<String, String>` reverse lookup)

---

## Review Queue

`LiveMatchState.review_queue: DashMap<String, ReviewEntry>`

Keyed by `"{side}:{id}:{candidate_id}"` (e.g. `"a:ENT-1:CP-2"`). Holds `{ id, side, candidate_id, score }`.

Entries are **drained** (via `retain`) when:
- `crossmap_confirm` fires for either party → they are now confirmed, no longer in review
- `crossmap_break` fires for either party
- A `remove_record` fires for either party — any review mentioning that ID is purged
- A record is re-upserted — its stale reviews are cleared before re-scoring

Review queue mutations now go through `RecordStore` trait methods:
- `persist_review(side, id, candidate_id, score)` — add or update a review entry
- `remove_reviews_for_id(side, id)` — purge all reviews mentioning an ID
- `remove_reviews_for_pair(a_id, b_id)` — purge reviews for a specific pair
- `load_reviews()` — load all reviews from persistent storage

Implementations:
- `MemoryStore` — these are no-ops (reviews live only in the in-memory `review_queue`)
- `SqliteStore` — writes through to the `reviews` table (read-through on load)

At startup, reviews are loaded via `store.load_reviews()` instead of raw SQL queries, then reconstructed from `review_match` WAL events at step 13, and filtered against the current CrossMap state.

---

## Graceful Shutdown

`api/server.rs` listens for `SIGINT` (Ctrl-C) and `SIGTERM` simultaneously via `tokio::select!`. On Unix both signals are handled natively; on non-Unix platforms only Ctrl-C is supported. On signal receipt:
1. Combined index cache files are written to disk
2. CrossMap is flushed to disk

A dirty WAL BufWriter is also flushed. Any in-flight requests complete normally before shutdown.

---

See also: [[Config Reference#live]] and [[Config Reference#cross_map]] for the relevant config fields, [[Key Decisions#Three-Layer Cache Invalidation]] for how cache validity is checked, [[Business Logic Flow#Live Mode meld serve]] for where persistence fits in the upsert flow.
