← [Back to Index](./) | [Configuration](configuration.md) | [Live Mode](live-mode.md)

# Enroll Mode

Enroll mode runs a single-pool entity resolution server. Instead of
matching records between two distinct datasets (A vs B), records are
**enrolled** into one growing pool and scored against everything already
there. The result is a list of scored edges — ready for external graph
clustering or deduplication.

## When to use enroll mode

- You have **one dataset** and want to find duplicates within it
- You're building an **entity resolution graph** where the output is
  edges (not 1:1 pairs)
- There is **no canonical reference master** — all records are peers
- Records arrive one at a time and need to be resolved against
  everything seen so far

For two-sided matching (reference master A vs query stream B), use
[live mode](live-mode.md) instead.

## Starting the server

```bash
meld enroll --config enroll_config.yaml --port 8090
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--port` | `-p` | TCP port to listen on (default: 8080) |

## Configuration

Enroll mode uses a simplified config format — single field names
instead of `field_a`/`field_b` pairs, a single `dataset:` instead of
`datasets.a`/`datasets.b`, and no `cross_map` section.

```yaml
job:
  name: entity_resolution

# Optional — pre-load a reference pool at startup. No edges are
# generated for these records. Subsequent /enroll calls score against
# this pool and add to it.
dataset:
  path: reference_entities.csv
  id_field: entity_id

embeddings:
  model: melder/arctic-embed-xs-entity-resolution
  cache_dir: cache/embeddings

blocking:
  enabled: true
  operator: and
  fields:
    - { field: country_code }

match_fields:
  - { field: legal_name, method: embedding, weight: 0.55 }
  - { field: short_name, method: fuzzy, scorer: partial_ratio, weight: 0.20 }
  - { field: country_code, method: exact, weight: 0.20 }
  - { field: lei, method: exact, weight: 0.05 }

thresholds:
  auto_match: 0.85
  review_floor: 0.60

performance:
  encoder_pool_size: 4

vector_backend: flat
top_n: 20
```

### Key differences from live mode config

| Aspect | Live mode | Enroll mode |
|--------|-----------|-------------|
| Fields | `field_a` + `field_b` | `field` |
| Datasets | `datasets.a` + `datasets.b` | `dataset` (singular, optional) |
| Embedding cache | `a_cache_dir` + `b_cache_dir` | `cache_dir` |
| Crossmap | Required (`cross_map` section) | Not used |
| Output paths | Optional (`output` section) | Optional (`output` section), same CSV/Parquet/SQLite destinations |

## Endpoints

All endpoints are under `/api/v1/`. Only enroll-specific endpoints are
mounted — the A/B, crossmap, and review endpoints from live mode are
not available.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/enroll` | Score a record against the pool, add it, return edges |
| POST | `/enroll-batch` | Score and enroll multiple records sequentially |
| POST | `/enroll/remove` | Remove a record from the pool |
| GET | `/enroll/query?id=X` | Look up a record by ID |
| GET | `/enroll/count` | Number of records in the pool |
| POST | `/exclude` | Exclude a pair (known non-match) |
| DELETE | `/exclude` | Remove an exclusion |
| GET | `/health` | Health check |
| GET | `/status` | Server status (uptime, enrollment count) |
| POST | `/admin/flush` | Trigger output build without shutting down |
| POST | `/admin/shutdown` | Graceful shutdown with final output build |

## Enrolling a record

When you enroll a record, Melder:

1. Encodes it (embedding vector)
2. Searches the pool for candidates (ANN, BM25, blocking)
3. Scores candidates on all match fields
4. Adds the record to the pool (store, vector index, BM25, blocking)
5. Returns scored edges above `review_floor`, capped at `top_n`

The record is added **after** scoring, and self-candidates are explicitly
excluded (including when replacing an existing ID).

### Request

```bash
curl -X POST http://localhost:8090/api/v1/enroll \
  -H 'Content-Type: application/json' \
  -d '{
    "record": {
      "entity_id": "ent_003",
      "legal_name": "Goldman Sachs International",
      "country_code": "GB",
      "lei": "W22LROWP2IHZNBB6K528"
    }
  }'
```

### Response

```json
{
  "id": "ent_003",
  "enrolled": true,
  "edges": [
    {
      "id": "ent_001",
      "score": 0.94,
      "field_scores": [
        { "field": "legal_name", "method": "embedding", "score": 0.97, "weight": 0.55 },
        { "field": "country_code", "method": "exact", "score": 1.0, "weight": 0.20 },
        { "field": "lei", "method": "exact", "score": 1.0, "weight": 0.05 }
      ]
    },
    {
      "id": "ent_002",
      "score": 0.67,
      "field_scores": [
        { "field": "legal_name", "method": "embedding", "score": 0.72, "weight": 0.55 },
        { "field": "country_code", "method": "exact", "score": 1.0, "weight": 0.20 },
        { "field": "lei", "method": "exact", "score": 0.0, "weight": 0.05 }
      ]
    }
  ]
}
```

The `edges` array contains all matches scoring at or above `review_floor`,
up to `top_n` results, sorted by score descending. There is no
`classification` field — the caller decides what to do with the scores.

## Batch enrollment

`POST /enroll-batch` enrolls multiple records in a single request.
Embedding fields are encoded together before changing the pool. Records are
then matched and inserted **sequentially** — record N is scored against the
pool including records 1..N-1 from the same batch, so intra-batch edges are
still discovered. Invalid records remain per-item errors; if batch encoding
fails, Melder retries encoding each valid record individually.

For individual requests, encoding also runs before acquiring the pool's
mutation lock. Concurrent clients can prepare their vectors in parallel,
but scoring and publication remain serialized: a record never searches a
partially applied enrollment.

### Request

```bash
curl -X POST http://localhost:8090/api/v1/enroll-batch \
  -H 'Content-Type: application/json' \
  -d '{
    "records": [
      { "entity_id": "ent_004", "legal_name": "Morgan Stanley & Co", "country_code": "US" },
      { "entity_id": "ent_005", "legal_name": "MS International", "country_code": "US" }
    ]
  }'
```

### Response

```json
{
  "results": [
    {
      "id": "ent_004",
      "enrolled": true,
      "edges": []
    },
    {
      "id": "ent_005",
      "enrolled": true,
      "edges": [
        { "id": "ent_004", "score": 0.78, "field_scores": [...] }
      ]
    }
  ]
}
```

## Initial dataset

The optional `dataset:` section in the config pre-loads records into the
pool at startup. **No edges are generated** for these records — they are
loaded directly into the store and indices. This is the "known entities"
reference set.

The server becomes ready only after this startup loading completes. When
embedding vectors need to be built or refreshed, encoding uses up to
`performance.encoder_pool_size` dedicated workers, separate from the
parallel tokenizer workers. This avoids nested tokenizer work blocking
on encoder sessions held by unfinished encoding jobs. No tokenizer
parallelism override is required.

Edges only come from subsequent `/enroll` or `/enroll-batch` calls. If
you need all-pairs edges within the initial dataset, use `meld run`
with the same dataset on both sides, or call `/enroll-batch` after
startup.

## No crossmap

Enroll mode has no crossmap. A record can have edges to multiple pool
members. There is no 1:1 constraint and no auto-match/review/no-match
classification. The caller is responsible for deciding which connections
are real matches — typically via graph clustering.

## Concurrency

Multiple clients can call `/enroll` simultaneously. Records are added
to the pool as they are processed, so concurrent enrollments may or may
not see each other depending on timing. This is non-deterministic by
design — the same behaviour as live mode's concurrent `add` calls.

If deterministic edge discovery is required, serialize calls or use
`/enroll-batch`.

## Output

Enroll mode has the same output pipeline as batch and live modes. The
match log written during enrollment is the canonical input for output
generation.

Each successful enrollment writes its complete returned edge list, including
an empty list, to the match log as `enroll_outcome`. These are directed
single-pool observations, **not confirmed bijective matches**. They survive
log compaction and are available to admin builds and offline export without
waiting for the asynchronous scoring-log writer. Like other WAL events,
appends are buffered, not individually fsynced; admin flush/shutdown flush
the log before building.

**Scoring log defaults to on.** With scoring logging enabled, the enrollment
outcome also persists per-field detail for output explainability. With
`scoring_log.enabled: false`, records, scores, ranks, and all returned
qualifying edges are still persisted and exported; per-field output detail
is omitted. The API response still includes per-field scores. The optional
scoring log may contain other retrieved candidates, but those do not become
enrollment relationships merely because they were scored.

### Single-pool output semantics

- `relationships` contains every retained returned edge, including rank 1,
  with `relationship_type: enroll_edge`. This means only that this query
  returned this candidate at or above `review_floor`; it does not mean
  auto-confirmation, reciprocal qualification, clustering, or a 1:1 claim.
- Existing A/B-shaped serializers use A for the candidate and B for the
  enrolling query. Both namespaces refer to the **same pool**; SQLite
  record tables contain the same pool records, including seeds and isolated
  new records. An edge never connects an ID to itself.
- `candidates` includes these same edges, including rank 1, even with scoring
  logging disabled. Rank is 1-based in returned score order per query.
  Surviving edges retain their original returned rank even if another edge
  is later invalidated or deleted. Ranks are not a global graph ordering.
- `unmatched` means **isolated in the retained evidence graph**: no outgoing
  or incoming retained edge. A seed reached by a later query is not
  unmatched. Seeds that were never scored or reached are unmatched; this
  is not proof that no qualifying neighbor exists. There is no manufactured
  best-candidate score for an unscored seed.
- Older enrollment logs that did not record returned edges cannot reconstruct those edges. Output builds replay evidence; they do not rescore the pool or discover
  seed-to-seed edges. Re-enrollment invalidates earlier evidence involving
  the updated record and replaces its outgoing results. Removing a record
  removes its incident evidence; excluding a pair removes both directed
  observations (unexcluding does not rescore or restore them). Other records are not automatically
  rescored, so the graph remains order-dependent.

**Generating outputs.** Use any of:
- `meld export --config enroll_config.yaml --out-dir export/` — reads
  the match log and produces CSVs, Parquet files, and/or the SQLite
  DB via the build pipeline.
- `POST /admin/flush` — triggers a build without shutting down.
  Returns `200 OK` after writing the files, with `enroll_edges` and `unmatched` counts.
- `POST /admin/shutdown` — build the configured outputs, then shut down.

Configure any combination of `output.csv_dir_path`,
`output.parquet_dir_path`, and `output.db_path` in the config to
control what is produced. When none is set, the match log is still
written and outputs can be generated later with `meld export`.

## Persistence

The pool persists across restarts using the same match log mechanism as
live mode. `live.db_path` is **not supported** in enroll mode and is rejected
rather than silently ignored; use `live.match_log_path` for persistence.
`output.db_path` is supported as an output artifact, not as the live store. On shutdown, the embedding index cache is saved to disk.
On restart, the match log is replayed and the cache is reloaded.

## Hooks

Hooks work in enroll mode. Since there is no crossmap or classification,
only `on_nomatch` fires — when an enrolled record has zero edges above
`review_floor`. See [Hooks](hooks.md) for configuration and examples.

## Upsert semantics

Enrolling a record with an ID that already exists replaces the existing
record, re-scores against the pool, and returns new edges.
