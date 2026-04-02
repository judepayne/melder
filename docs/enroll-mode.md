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
| Output paths | Required (`output` section) | Not used |

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

## Enrolling a record

When you enroll a record, Melder:

1. Encodes it (embedding vector)
2. Searches the pool for candidates (ANN, BM25, blocking)
3. Scores candidates on all match fields
4. Adds the record to the pool (store, vector index, BM25, blocking)
5. Returns scored edges above `review_floor`, capped at `top_n`

The record is added **after** scoring, so it never matches itself.

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
Records are processed **sequentially** — record N is scored against the
pool including records 1..N-1 from the same batch. This means
intra-batch edges are discovered.

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

## Persistence

The pool persists across restarts using the same WAL mechanism as
live mode. On shutdown, the embedding index cache is saved to disk.
On restart, the WAL is replayed and the cache is reloaded.

## Hooks

Hooks work in enroll mode. Since there is no crossmap or classification,
only `on_nomatch` fires — when an enrolled record has zero edges above
`review_floor`. See [Hooks](hooks.md) for configuration and examples.

## Upsert semantics

Enrolling a record with an ID that already exists replaces the existing
record, re-scores against the pool, and returns new edges.
