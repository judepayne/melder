← [Back to Index](./) | [Configuration](configuration.md) | [API Reference](api-reference.md)

# Live Mode

> For a hands-on walkthrough, see the
> [live worked example](../examples/live/README.md).

Live mode starts an HTTP server that matches records on the fly as they
arrive. It supports two storage backends:

- **In-memory** (default) — records and crossmap held in RAM, persisted
  via a write-ahead log (WAL) and crossmap CSV.
- **SQLite** (set `live.db_path`) — records, crossmap, and review queue
  stored in a SQLite database. Durable by default, instant warm restarts.

Both modes use the same scoring pipeline, so a match score means the
same thing regardless of which backend produced it.

## Starting the server

```bash
meld serve --config config.yaml --port 8090
```

Once ready, the server prints:

```
meld serve listening on port 8090
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--port` | `-p` | TCP port to listen on (default: 8080) |

## Storage backends

SQLite throughput is ~18% lower than in-memory at the same scale
(~1,395 vs ~1,698 req/s at 10k, c=10). The gap comes from B-tree
traversal and per-connection page cache overhead vs DashMap hash lookups.
Tail latencies (p95, p99) are actually better with SQLite — the
connection pool smooths out contention spikes. Records are stored in
columnar format (one column per field) for fast access with no JSON
serialization overhead.

SQLite uses a writer + reader pool architecture:
`sqlite_read_pool_size` (default 4) read-only connections serve
concurrent reads, while a single write connection handles all mutations.
`sqlite_pool_worker_cache_mb` (default 128) controls the page cache per
read connection.

## Startup sequences

What happens at launch depends on the storage backend:

### In-memory (no `db_path`)

1. Dataset files (CSV/JSONL/Parquet) are loaded as the base record set
2. Embedding index caches are loaded from disk (if present and valid)
3. Blocking indices are built from the dataset records
4. Crossmap CSV is loaded
5. Exclusions CSV is loaded (if configured)
6. All WAL files are replayed in chronological order:
   - Records are inserted/removed from the in-memory store
   - Blocking indices are updated for each replayed record
   - Crossmap confirms/breaks are applied
   - Exclusions are applied/removed
   - Embedding vectors already in the cached index are skipped
     (no ONNX re-encoding)
7. Unmatched sets and common-ID indices are rebuilt from the final state
8. Review queue is populated from unresolved `ReviewMatch` WAL events
9. A new timestamped WAL file is opened for the current run
10. **Initial matching pass** — all unmatched B records are scored against
    the A pool using the full scoring pipeline (blocking, BM25, ANN,
    synonym). Auto-matches are claimed in the crossmap and persisted via
    WAL. Review-band matches are added to the review queue. This ensures
    pre-loaded datasets are fully matched before the API starts listening.
    Set `live.skip_initial_match: true` to skip this step and start the
    API immediately.

### SQLite — cold start (DB file does not exist)

1. A new SQLite database is created
2. Dataset files (CSV/JSONL/Parquet) are loaded and inserted into SQLite
3. If a crossmap CSV exists at `cross_map.path`, its pairs are imported
   into SQLite (one-time migration — the CSV is not updated afterwards)
4. Embedding indices are built or loaded from cache
5. Blocking indices are built
6. A new WAL file is opened

### SQLite — warm start (DB file exists)

1. SQLite database is opened directly — records, crossmap, and reviews
   are already there
2. Embedding index caches are loaded from disk
3. Blocking indices are rebuilt from the SQLite records
4. A new WAL file is opened

No CSV loading. No WAL replay. Restarts are fast.

## Logging

All log output goes to stderr, so it can be redirected independently of
any stdout output. Control the level with `RUST_LOG`:

```bash
# Default — info-level messages (startup, shutdown, errors)
meld serve --config config.yaml --port 8090

# Debug — includes per-request timing, encode/search/score spans
RUST_LOG=melder=debug meld serve --config config.yaml --port 8090

# JSON structured logs (for piping to a log aggregator)
meld serve --config config.yaml --port 8090 --log-format json

# Run in background and tail the log
meld serve --config config.yaml --port 8090 2>serve.log &
tail -f serve.log
```

## Pipeline hooks

For event notifications (match confirmed, review queued, no match,
match broken), see [Hooks](hooks.md). Hooks run a single long-lived
subprocess that receives events as JSON on stdin — zero impact on
scoring throughput.

## Write-ahead log (WAL)

Every record addition and cross-map change is appended to the WAL file
(configured via `live.upsert_log`, e.g. `wal.ndjson`). This is a
newline-delimited JSON file — one event per line.

In in-memory mode, the WAL is essential for crash recovery: if the
server is killed, the next startup replays these events to restore
state. In SQLite mode, the WAL is still written as a redundant safety
net but is not needed for recovery.

On clean shutdown the WAL is compacted (duplicate entries collapsed) and
can be inspected with:

```bash
# See recent WAL entries
tail -20 wal.ndjson

# Count events by type
jq -r .type wal.ndjson | sort | uniq -c
```

Each server run creates a new timestamped WAL file (e.g.
`wal_20260312T143207Z.ndjson`). On startup, all WAL files matching the
configured base path are discovered and replayed in lexicographic
(chronological) order. Each run's WAL is compacted at shutdown. Old WAL
files accumulate across runs; delete them manually if disk space is a
concern (only the most recent compacted file is needed for full
recovery).

## Cross-map persistence

How confirmed matches are persisted depends on the storage backend:

- *In-memory:* The crossmap is held in RAM and flushed to the crossmap
  CSV periodically (every `crossmap_flush_secs`, default 5 seconds) and
  on shutdown. The CSV is the durable record of which pairs have been
  matched.
- *SQLite:* Every confirm/break is written to the database immediately.
  The crossmap CSV is never updated. Use the `/crossmap/pairs` API
  endpoint or query the `crossmap` table in the SQLite DB directly to
  export pairs.

## Shutdown

Send Ctrl-C or SIGTERM. The melder will stop accepting new connections,
drain in-flight requests, flush and compact the WAL, save the cross-map
(in-memory mode) or no-op (SQLite mode), and persist index caches. No
data is lost.

## Persistence and restart

Live mode is designed to survive restarts. The full state — records
added via the API, confirmed crossmap pairs, and embedding vectors — is
persisted to disk and restored on the next startup.

### In-memory mode (default)

#### What is persisted

| Component | Mechanism | When |
|-----------|-----------|------|
| Record mutations (add, remove) | Write-ahead log (WAL) | Every API call |
| Crossmap confirmations/breaks | WAL + crossmap CSV | API call + periodic flush |
| Embedding vectors | Index cache (`.usearchdb` or `.index`) | Shutdown |
| Review queue | WAL (`ReviewMatch` events) | Every API call |

#### Shutdown sequence

1. WAL is flushed and compacted (deduplicates per record ID, last-write-wins)
2. Crossmap CSV is flushed to disk
3. Combined embedding index caches are saved (includes all API-added vectors)

#### Startup sequence

1. Dataset files (CSV/JSONL/Parquet) are loaded as the base record set
2. Embedding index caches are loaded from disk (if present and valid)
3. Blocking indices are built from the dataset records
4. Crossmap CSV is loaded
5. All WAL files are replayed in chronological order:
   - Records are inserted/removed from the in-memory store
   - Blocking indices are updated for each replayed record
   - Crossmap confirms/breaks are applied
   - Embedding vectors already in the cached index are skipped
     (no ONNX re-encoding)
6. Unmatched sets and common-ID indices are rebuilt from the final state
7. Review queue is populated from unresolved `ReviewMatch` WAL events
8. A new timestamped WAL file is opened for the current run

#### What this means in practice

- Records added via `/a/add` or `/b/add` survive restarts. They are
  replayed from the WAL and their embedding vectors are loaded from the
  index cache — no re-encoding required.
- Confirmed crossmap pairs survive via both the crossmap CSV and WAL
  replay (belt and suspenders).
- Blocking works correctly for WAL-replayed records. A new record added
  after restart will find WAL-replayed records on the opposite side as
  match candidates.
- The review queue is rebuilt from WAL events, minus any pairs that were
  subsequently confirmed or broken.
- The base dataset files are never modified. The WAL captures the delta.

#### WAL files

Each server run creates a new timestamped WAL file (e.g.
`wal_20260312T143207Z.ndjson`). On startup, all WAL files matching the
configured base path are discovered and replayed in lexicographic
(chronological) order. Each run's WAL is compacted at shutdown. Old WAL
files accumulate across runs; delete them manually if disk space is a
concern (only the most recent compacted file is needed for full
recovery).

### SQLite mode (`live.db_path` set)

#### What is persisted

| Component | Mechanism | When |
|-----------|-----------|------|
| Records | SQLite `records` table | Immediately on every add/remove |
| Crossmap pairs | SQLite `crossmap` table | Immediately on every confirm/break |
| Review queue | SQLite `reviews` table | Immediately on every review-band match |
| Embedding vectors | Index cache (`.usearchdb` or `.index`) | Shutdown |
| WAL | Same as in-memory mode | Every API call (redundant safety net) |

#### Shutdown sequence

1. WAL is flushed and compacted
2. Combined embedding index caches are saved
3. (No crossmap CSV flush — SQLite is already durable)

#### Warm startup (DB exists)

1. SQLite database is opened directly — records, crossmap, and reviews
   are already there
2. Embedding index caches are loaded from disk
3. Blocking indices are rebuilt from the SQLite records
4. A new WAL file is opened

No CSV loading. No WAL replay. Restarts are fast.

#### Cold startup (no DB file)

1. A new SQLite database is created
2. Dataset files (CSV/JSONL/Parquet) are loaded and inserted into SQLite
3. If a crossmap CSV exists at `cross_map.path`, its pairs are imported
   into SQLite (one-time migration — the CSV is not updated afterwards)
4. Embedding indices are built or loaded from cache
5. Blocking indices are built
6. A new WAL file is opened

### Migration from in-memory to SQLite

Add `live.db_path` to your config and restart. The first startup is a
cold start — datasets are loaded from CSV, the crossmap CSV is imported
into the database, and the WAL is written as a redundant log. From the
second startup onwards, the database is the sole source of truth and
restarts are instant. The crossmap CSV is never written to again.
