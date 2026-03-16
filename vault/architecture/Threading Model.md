---
type: architecture
module: runtime
status: active
tags: [threading, concurrency, synchronisation, runtime]
related_code: [src/main.rs, src/cli/run.rs, src/cli/serve.rs, src/batch/engine.rs, src/session/mod.rs, src/encoder/mod.rs, src/encoder/coordinator.rs]
---

# Threading Model

How melder uses threads, async tasks, and synchronisation primitives across its two operating modes.

## Batch Mode (`meld run`)

<img src="img/batch_threading.svg" width="900" />

### Thread pools

| Pool | Type | Size | Role |
|---|---|---|---|
| Main thread | OS thread | 1 | CLI, data loading, index building, output writing |
| Rayon global | Work-stealing | CPU cores | Parallel scoring (Phase 4 only) |
| Encoder pool | `Mutex<TextEmbedding>` slots | `encoder_pool_size` (default 1) | ONNX inference during index building |

**No tokio runtime exists in batch mode.** `main()` is synchronous; there is no `Runtime::new()`, no `block_on()`, no `spawn_blocking`. The encoding coordinator (mpsc/oneshot channels) is live-mode only.

### Phases

| Phase | Thread | Work |
|---|---|---|
| 1 | Main | Load config, parse A CSV, build `MemoryStore`, build A embedding index |
| 2 | Main | Load CrossMap from CSV |
| 3 | Main | Parse B CSV, build B embedding index, build BM25 index (A-side), common-ID pre-match |
| 4 | **Rayon** | `work_ids.par_iter()` -- score every B record against A candidates |
| 5 | Main | `collect()` results, sort, write output CSVs, save CrossMap |

Encoding (Phases 1 + 3) completes entirely before scoring begins -- the encoder pool is never accessed during Phase 4.

### Phase 4 lock profile (per Rayon worker)

Each worker processes one B record. Locks are acquired and released one at a time -- **no nesting**.

| # | Operation | Lock | Type | Duration |
|---|---|---|---|---|
| 1 | `store.get(B, id)` | DashMap shard | read | ns |
| 2 | `crossmap.has_b(id)` | `RwLock<Inner>` | read | ns |
| 3 | `blocking_query(rec)` | `RwLock<BlockingIndex>` | read | us |
| 4 | `candidate_select()` | `RwLock<VecIndex>` | read | us-ms |
| 5 | `score_pair()` | -- | pure CPU | us |
| 6 | `bm25.score()` | `Mutex<BM25Index>` | **exclusive** | us |
| 7 | `crossmap.claim(a,b)` | `RwLock<Inner>` | **write** | ns |

Steps 1-5 are read-only on pre-built structures -- fully parallel. Step 6 (BM25) is the only serialisation point: all Rayon workers queue on the single `Mutex<BM25Index>`. Step 7 briefly takes a write lock but holds it for nanoseconds (two HashMap lookups + two inserts).

---

## Live Mode (`meld serve`)

<img src="img/live_threading.svg" width="1200" />

### Thread pools and tasks

| Pool / Task | Type | Size | Role |
|---|---|---|---|
| Tokio workers | OS threads (async) | CPU cores | HTTP accept, route, response serialisation |
| Tokio blocking pool | OS threads (on-demand) | unbounded | `upsert_record_inner()`, scoring, store mutations |
| Coordinator | Tokio task (green thread) | 1 | Batches concurrent encode requests for ONNX efficiency |
| CrossMap flusher | Tokio task | 1 | Writes dirty CrossMap to disk every `crossmap_flush_secs` |
| WAL flusher | Tokio task | 1 | Calls `BufWriter::flush()` every 1s |

The tokio runtime is created manually via `Runtime::new()` in `cmd_serve()`. The main thread blocks on `rt.block_on()` for the server lifetime.

### Request lifecycle (upsert)

```
HTTP request
  |
  v
Tokio worker (async) --- deserialise JSON, extract route
  |
  |  spawn_blocking
  v
Blocking thread (sync) --- upsert_record_inner()
  |
  |  mpsc send --> Coordinator task (async)
  |                    |  spawn_blocking
  |                    v
  |               Encoder Pool (blocking thread, Mutex<TextEmbedding>)
  |                    |
  |  oneshot reply <---+
  |
  |  ... store, score, claim (see lock table below) ...
  |
  |  result
  v
Tokio worker (async) --- serialise JSON, send response
```

When the coordinator is disabled (`encoder_batch_wait_ms = 0`), the blocking thread calls `EncoderPool::encode()` directly, acquiring a `Mutex<TextEmbedding>` slot without the channel round-trip.

### Upsert lock acquisition order

Every lock is acquired and released before the next -- **no nesting, no deadlock risk**.

| # | Operation | Lock | Type | Duration |
|---|---|---|---|---|
| 1 | Text-hash skip check | `RwLock<TextHashStore>` | read | ns |
| 2 | ONNX encode | `Mutex<TextEmbedding>` | exclusive | **3-6ms** |
| 3 | Store combined vector | `RwLock<VecIndex>` | write | us |
| 4 | Update text hash | `RwLock<TextHashStore>` | write | ns |
| 5 | Check record exists | DashMap shard | read | ns |
| 6 | Break old crossmap pair | `RwLock<CrossMap::Inner>` | write | ns |
| 7 | Remove old blocking keys | `RwLock<BlockingIndex>` | write | us |
| 8 | WAL append (break) | `Mutex<UpsertLog>` | exclusive | us |
| 9 | Insert record | DashMap shard | write | ns |
| 10 | Insert blocking keys | `RwLock<BlockingIndex>` | write | us |
| 11 | WAL append (upsert) | `Mutex<UpsertLog>` | exclusive | us |
| 12 | BM25 upsert (this side) | `Mutex<BM25Index>` | exclusive | us |
| 13 | BM25 commit (opp side) | `Mutex<BM25Index>` | exclusive | us |
| 14 | Blocking query (opp side) | `RwLock<BlockingIndex>` | read | us |
| 15 | ANN search (opp index) | `RwLock<VecIndex>` | read | us-ms |
| 16 | Fetch candidate records | DashMap shards | read | ns x N |
| 17 | `score_pair()` | -- | pure CPU | us |
| 18 | `crossmap.claim(a,b)` | `RwLock<CrossMap::Inner>` | write | ns |
| 19 | Mark matched | DashSet shards | write | ns |
| 20 | WAL append (confirm) | `Mutex<UpsertLog>` | exclusive | us |

Step 2 (ONNX encoding) dominates at 3-6ms. All other locks combined total < 0.5ms.

### Background task contention

| Task | Lock acquired | Blocks | Frequency |
|---|---|---|---|
| CrossMap flusher | `RwLock<CrossMap::Inner>` read | write ops (claim, add, break) | every 5s |
| WAL flusher | `Mutex<UpsertLog>` | WAL appends | every 1s |

Both hold their locks for sub-millisecond durations under normal load.

---

## Synchronisation primitives summary

| Primitive | What it guards | Why this choice |
|---|---|---|
| `DashMap` | Record stores (A/B records) | Shard-level locking; high read concurrency with negligible write contention |
| `RwLock` | Blocking indices, vector indices, CrossMap inner | Many concurrent readers, infrequent writers. CrossMap uses a single RwLock (not two DashMaps) to prevent TOCTOU and cross-shard deadlock -- see [[CONSTITUTION#3 CrossMap Bijection 1 1 Under One Lock]] |
| `std::sync::Mutex` | Encoder pool slots, BM25 indices, WAL writer, SqliteStore connection | Exclusive access required. Encoder slots use try_lock round-robin for fairness |
| `DashSet` | Unmatched ID sets | Same as DashMap (shard-level) |
| `AtomicBool` / `AtomicU64` | CrossMap dirty flag, upsert counter | Lock-free; no contention |
| mpsc + oneshot channels | Encoding coordinator | Decouples handler tasks from ONNX inference; enables batching without shared mutable state |

### SqliteStore: Writer + Reader Pool

When using the SQLite backend, a single write connection (`Arc<Mutex<Connection>>`) handles all mutations, and a pool of N read-only connections (`SqliteReaderPool`) serves concurrent reads via round-robin `try_lock()`. The pool is shared between `SqliteStore` and `SqliteCrossMap`.

**Why separate connections (not RwLock):** `rusqlite::Connection` is `Send` but not `Sync` — the underlying C API is not thread-safe for concurrent calls on the same handle, even for reads. Each connection must have exclusive access via `Mutex`. Concurrency comes from having multiple connections, not from sharing one.

**Reader pool sizing:** Default 4 for live mode (sufficient with ONNX staggering arrivals). Default `num_cpus` for batch mode (matches Rayon thread pool). Configurable via `sqlite_read_pool_size`.

**Nested par_iter hazard:** The candidate selection path in `candidates.rs` originally used `par_iter` to fetch blocked records. With SQLite, this created nested parallelism (outer `par_iter` in `engine.rs` + inner `par_iter` in `candidates.rs`), causing deadlock when all reader pool connections were held by inner tasks. Fixed by converting the inner `par_iter` to sequential `iter` for the no-embeddings path.

See [[Key Decisions#SQLite Connection Pool Writer N Readers]].

---

*See also: [[Business Logic Flow]], [[State & Persistence]], [[Module Map]]*
