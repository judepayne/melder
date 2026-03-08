# Live Mode Worked Example

This example starts a melder server with 10 records on each side
(`set_1.csv` and `set_2.csv`), then walks you through the full
lifecycle: querying, adding records, confirming and breaking matches,
removing records, monitoring the WAL, and restarting with state
recovery.

In live mode, both sides are **fully symmetric** -- adding a record to
A searches B for matches, and adding a record to B searches A. Both
sides support the same operations (add, remove, query, match). The file
names `set_1.csv` and `set_2.csv` reflect this: neither side is
privileged. This is different from batch mode, where A is the
pre-indexed reference set and B records are scored against it one by
one.

## Prerequisites

Build melder from the project root:

```bash
cargo build --release
```

All commands below assume you are in the **project root** directory
(the one containing `Cargo.toml`). You will need two terminal windows
-- one for the server, one for curl commands.

---

## Step 1: Start the server

In **Terminal 1**:

```bash
./target/release/meld serve -c examples/live/config.yaml --port 9090
```

You will see startup output like:

```
Initializing encoder pool (model=all-MiniLM-L6-v2, pool_size=2)...
Encoder ready (dim=384), took 0.2s
Loaded dataset A: 10 records in 0.0s
Building A index (10 records)...
...
Loaded dataset B: 10 records in 0.0s
Building B index (10 records)...
...
Live state loaded in 1.2s (A: 10 records/10 unmatched, B: 10 records/10 unmatched, crossmap: 0 pairs)
Listening on 0.0.0.0:9090
```

Let's break that down:

- **Encoder pool** -- the ONNX model is loaded into memory. Two
  parallel sessions (`encoder_pool_size: 2`) for concurrent encoding.
- **A index / B index** -- every record's text is encoded into a
  384-dimensional vector. On first run this is done from scratch; on
  subsequent runs the `.index` cache files are loaded in milliseconds.
- **10 unmatched** on each side -- no crossmap exists yet, so all
  records start as unmatched.
- **Listening** -- the HTTP server is ready for requests.

> **The server is now running.** Leave Terminal 1 open. All the
> following commands are run in **Terminal 2**.

---

## Step 2: Health and status

Check the server is alive:

```bash
curl -s http://localhost:9090/api/v1/health | jq .
```

```json
{
  "status": "ok"
}
```

Get detailed status:

```bash
curl -s http://localhost:9090/api/v1/status | jq .
```

```json
{
  "a_records": 10,
  "b_records": 10,
  "a_unmatched": 10,
  "b_unmatched": 10,
  "crossmap_pairs": 0,
  "uptime_secs": 5
}
```

> **Everything is unmatched.** The server loaded the data but has not
> performed any matching yet. In live mode, matching happens when you
> add or update a record -- not at startup.

---

## Step 3: Query an existing record

Look up an entity by ID:

```bash
curl -s 'http://localhost:9090/api/v1/a/query?id=ENT-001' | jq .
```

```json
{
  "id": "ENT-001",
  "side": "a",
  "record": {
    "entity_id": "ENT-001",
    "legal_name": "Acme Corporation",
    "short_name": "Acme Corp",
    "country_code": "US",
    "lei": "5493001KJTIIGC8Y1R12"
  },
  "crossmap": {
    "status": "unmatched"
  }
}
```

And a counterparty:

```bash
curl -s 'http://localhost:9090/api/v1/b/query?id=CP-001' | jq .
```

> **Query is read-only.** It shows you the record and whether it is
> currently matched or unmatched. It never modifies anything.

---

## Step 4: Add (upsert) a B record to trigger matching

When you "add" a record that already exists, melder updates it and
re-runs the scoring pipeline. This is how matching happens in live
mode.

Let's upsert CP-001 to trigger matching against the A side:

```bash
curl -s -X POST http://localhost:9090/api/v1/b/add \
  -H 'Content-Type: application/json' \
  -d '{
    "counterparty_id": "CP-001",
    "counterparty_name": "ACME Corp.",
    "domicile": "US",
    "lei_code": "5493001KJTIIGC8Y1R12"
  }' | jq .
```

The response shows the top matches from the A side:

```json
{
  "status": "updated",
  "id": "CP-001",
  "side": "b",
  "matches": [
    {
      "matched_id": "ENT-001",
      "score": 0.95,
      "classification": "auto",
      "field_scores": [...]
    }
  ]
}
```

> **What just happened?** Melder encoded "ACME Corp." into a vector,
> searched the A-side index for similar vectors (filtered to US records
> by blocking), scored each candidate using all four match fields, and
> returned the results. Because the top score exceeded 0.85 (the
> `auto_match` threshold), the pair was **automatically confirmed** and
> added to the crossmap.

Check the status to confirm:

```bash
curl -s http://localhost:9090/api/v1/status | jq .
```

You should see `crossmap_pairs: 1` and one fewer unmatched on each side.

Query CP-001 again to see its crossmap status:

```bash
curl -s 'http://localhost:9090/api/v1/b/query?id=CP-001' | jq .
```

The `crossmap` section now shows `"status": "matched"` with the paired
A record.

---

## Step 5: Match-only (read-only scoring)

Sometimes you want to see what would match without storing the record.
Use the `/match` endpoint instead of `/add`:

```bash
curl -s -X POST http://localhost:9090/api/v1/b/match \
  -H 'Content-Type: application/json' \
  -d '{
    "counterparty_id": "CP-NEW",
    "counterparty_name": "Sakura Holdings Ltd",
    "domicile": "JP",
    "lei_code": ""
  }' | jq .
```

This returns candidate matches but does not store CP-NEW or update the
crossmap. The record is discarded after scoring.

> **Use `/match` for what-if queries.** It's useful for testing how a
> record would score before committing it to the system.

---

## Step 6: Confirm a match manually

Let's manually confirm a match. First, upsert CP-004 (Sakura Financial
Grp) to trigger scoring:

```bash
curl -s -X POST http://localhost:9090/api/v1/b/add \
  -H 'Content-Type: application/json' \
  -d '{
    "counterparty_id": "CP-004",
    "counterparty_name": "Sakura Financial Grp",
    "domicile": "JP",
    "lei_code": ""
  }' | jq .
```

If the score falls below 0.85, it won't be auto-confirmed. You can
confirm it manually:

```bash
curl -s -X POST http://localhost:9090/api/v1/crossmap/confirm \
  -H 'Content-Type: application/json' \
  -d '{
    "a_id": "ENT-003",
    "b_id": "CP-004"
  }' | jq .
```

```json
{
  "status": "confirmed",
  "a_id": "ENT-003",
  "b_id": "CP-004"
}
```

Check status -- `crossmap_pairs` should have increased.

> **Confirming a match removes both records from the unmatched pool.**
> They will not appear as candidates in future scoring. This is how
> review decisions are applied in production.

---

## Step 7: Break a match

Changed your mind? Break the pair:

```bash
curl -s -X POST http://localhost:9090/api/v1/crossmap/break \
  -H 'Content-Type: application/json' \
  -d '{
    "a_id": "ENT-003",
    "b_id": "CP-004"
  }' | jq .
```

```json
{
  "status": "broken",
  "a_id": "ENT-003",
  "b_id": "CP-004"
}
```

Both records are returned to the unmatched pool and become eligible
for future matching.

> **Breaking a match does not delete the records.** It only removes
> the crossmap pairing. Both records remain in the system.

---

## Step 8: Add a new A record

Matching works symmetrically. Adding an A record searches the B side:

```bash
curl -s -X POST http://localhost:9090/api/v1/a/add \
  -H 'Content-Type: application/json' \
  -d '{
    "entity_id": "ENT-011",
    "legal_name": "Osaka Electronics Corporation",
    "short_name": "Osaka Electronics",
    "country_code": "JP",
    "lei": ""
  }' | jq .
```

This searches for matching counterparties in Japan. If you earlier
upserted CP-004 and it is unmatched, it may appear as a candidate.

> **Symmetry.** Everything that works for B (`/b/add`, `/b/query`,
> `/b/remove`, `/b/match`) works identically for A. Both sides are
> first-class.

---

## Step 9: Remove a record

Remove a record from all indices:

```bash
curl -s -X POST http://localhost:9090/api/v1/a/remove \
  -H 'Content-Type: application/json' \
  -d '{"id": "ENT-011"}' | jq .
```

```json
{
  "status": "removed",
  "id": "ENT-011",
  "side": "a"
}
```

If ENT-011 was matched, the pairing is broken and the opposite-side
record returns to the unmatched pool. The record is removed from the
vector index, blocking index, and records store.

> **Removal is permanent for the current session.** The record is
> gone from all in-memory structures. However, if the record exists
> in the original csv file, it will be reloaded on the next server
> restart (unless a `RemoveRecord` WAL event overrides it during
> replay).

---

## Step 10: Look up a crossmap entry

Check whether a specific record is matched:

```bash
curl -s 'http://localhost:9090/api/v1/crossmap/lookup?id=CP-001&side=b' | jq .
```

---

## Step 11: Monitor the WAL

The write-ahead log (WAL) records every mutation: record additions,
crossmap confirmations, crossmap breaks, and record removals. It is
written to disk with a rolling ~1-second delay.

Open a **third terminal** and tail the WAL:

```bash
tail -f examples/live/output/wal.ndjson
```

Now go back to Terminal 2 and add another record:

```bash
curl -s -X POST http://localhost:9090/api/v1/b/add \
  -H 'Content-Type: application/json' \
  -d '{
    "counterparty_id": "CP-006",
    "counterparty_name": "Nordic Energy Sys AB",
    "domicile": "SE",
    "lei_code": ""
  }' | jq .
```

Within a second, you should see a new line appear in the WAL tail:

```json
{"UpsertRecord":{"side":"b","record":{"counterparty_id":"CP-006","counterparty_name":"Nordic Energy Sys AB","domicile":"SE","lei_code":""}}}
```

If the record auto-matched, you will also see a `CrossMapConfirm`
event.

> **The WAL is the crash-recovery mechanism.** If the server crashes,
> these events are replayed on restart to reconstruct any changes
> that happened after the initial csv load. The WAL is compacted on
> clean shutdown (duplicate events for the same record are collapsed
> to keep the file small).

---

## Step 12: Shutdown and restart

Go to Terminal 1 and press **Ctrl-C** to stop the server. You will
see shutdown output:

```
Shutting down...
Flushing WAL...
Compacting WAL...
Saving crossmap (N pairs)...
Saving index caches...
Done.
```

What happens on shutdown:

1. **WAL flush** -- any buffered events are written to disk.
2. **WAL compaction** -- the WAL is deduplicated (if you upserted the
   same record 5 times, only the last version is kept).
3. **Crossmap save** -- the current crossmap is written to
   `examples/live/output/crossmap.csv`.
4. **Index caches** -- both A and B vector indices are saved to
   `examples/live/cache/`.

Now restart:

```bash
./target/release/meld serve -c examples/live/config.yaml --port 9090
```

Watch the startup log. You should see:

```
Loaded A index from cache: 10 vecs in 0.1ms
Loaded B index from cache: 10 vecs in 0.1ms
...
Loaded crossmap: N pairs
Replaying M WAL events...
WAL replay complete
```

What happens on restart:

1. **Cache load** -- vector indices are loaded from the `.index` files
   saved at shutdown. No ONNX encoding needed.
2. **Crossmap load** -- confirmed pairs are restored.
3. **WAL replay** -- any events that happened after the last csv load
   are replayed. Records added via the API are re-inserted, crossmap
   changes are re-applied.

Check status to confirm everything is restored:

```bash
curl -s http://localhost:9090/api/v1/status | jq .
```

The record counts, unmatched counts, and crossmap pairs should match
what you had before shutdown.

> **Persistence model.** The csv files are the baseline. The crossmap
> csv and WAL capture everything that changed on top of that baseline.
> Together they fully reconstruct the server state.

---

## Step 13: Clean up

Stop the server (Ctrl-C), then remove all generated files:

```bash
rm -f examples/live/cache/*.index
rm -f examples/live/output/*.csv
rm -f examples/live/output/*.ndjson
```

---

## Summary of endpoints used

| Method | Path | What it does |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/status` | Record counts, unmatched counts, crossmap size, uptime |
| GET | `/api/v1/a/query?id=X` | Look up an A record and its crossmap status |
| GET | `/api/v1/b/query?id=X` | Look up a B record and its crossmap status |
| POST | `/api/v1/a/add` | Add/update an A record, return matches from B |
| POST | `/api/v1/b/add` | Add/update a B record, return matches from A |
| POST | `/api/v1/a/match` | Score an A record against B (read-only, not stored) |
| POST | `/api/v1/b/match` | Score a B record against A (read-only, not stored) |
| POST | `/api/v1/a/remove` | Remove an A record from all indices |
| POST | `/api/v1/b/remove` | Remove a B record from all indices |
| POST | `/api/v1/crossmap/confirm` | Confirm a match (add to crossmap) |
| POST | `/api/v1/crossmap/break` | Break a confirmed match |
| GET | `/api/v1/crossmap/lookup?id=X&side=a` | Check if a record is matched |

For full API documentation, see the main [README](../../README.md).
