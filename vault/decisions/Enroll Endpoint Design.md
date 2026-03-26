# Enroll Endpoint Design

**Status:** Design complete, pending implementation  
**Date:** 2026-03-26

## Summary

A new `mode: enroll` for Melder's live server that provides single-pool entity resolution. Instead of matching between two distinct datasets (A vs B), records are enrolled into one growing pool and scored against everything already there. Designed for graph-based ER workflows where the output is an edge list for external clustering, not a 1:1 crossmap.

## Motivation

Melder's current live mode (`meld serve`) assumes two distinct sides — a reference pool (A) and a query stream (B). This doesn't fit workflows where:

- There is no canonical reference master — all records are peers
- The goal is to discover all pairwise connections, not assign 1:1 mappings
- The output feeds a graph clustering layer that decides entity boundaries
- Records arrive one at a time and need to be resolved against everything seen so far

The `/enroll` endpoint makes Melder a building block for graph-based entity resolution without changing the core scoring engine.

## Design Decisions

All decisions below were made in conversation on 2026-03-26.

| # | Question | Decision |
|---|----------|----------|
| 1 | Config shape | New `mode: enroll` with simplified single-side fields (`field:` not `field_a:`/`field_b:`) |
| 2 | Blocking | Same simplification — `field:` not `field_a:`/`field_b:` |
| 3 | What edges to return | Both top-N and score threshold: up to `top_n` results, but only those above `review_floor` |
| 4 | Crossmap | None. Edges are reported, no exclusive claim. Graph layer decides. |
| 5 | Concurrency | Non-deterministic is fine. No ordering guarantees between concurrent enrollments. |
| 6 | Persistence | Reuse existing storage layer (WAL for memory, SQLite for durable). No new design. |
| 7 | Response shape | No `classification` field. Just id, score, field_scores per edge. |
| 8 | Duplicate ID | Upsert semantics — replace existing record, re-score, return new edges. |
| 9 | Bulk operations | Initial dataset (config) loads pool with no edges. `enroll-batch` endpoint for scored bulk enrollment. |

## Configuration

### Enroll mode config

```yaml
mode: enroll

job:
  name: entity_resolution
  description: Single-pool entity resolution for graph clustering

dataset:                              # Optional — pre-load a reference pool
  path: reference_entities.csv
  id_field: entity_id
  format: csv

embeddings:
  model: melder/arctic-embed-xs-entity-resolution
  cache_dir: cache/embeddings         # Single cache dir (no a/b split)

blocking:
  enabled: true
  operator: and
  fields:
    - field: country_code             # Single field name, not field_a/field_b

match_fields:
  - field: legal_name                 # Single field name
    method: embedding
    weight: 0.55
  - field: short_name
    method: fuzzy
    scorer: partial_ratio
    weight: 0.20
  - field: country_code
    method: exact
    weight: 0.20
  - field: lei
    method: exact
    weight: 0.05

thresholds:
  auto_match: 0.85                    # Used only for edge scoring reference
  review_floor: 0.60                  # Minimum score threshold for returned edges

performance:
  encoder_pool_size: 4

vector_backend: usearch
top_n: 20
ann_candidates: 50
```

### Key differences from `mode: live`

| Aspect | `mode: live` | `mode: enroll` |
|--------|-------------|----------------|
| Sides | Two (A and B) | One pool |
| Field config | `field_a` + `field_b` | `field` |
| Dataset config | `datasets.a` + `datasets.b` | `dataset` (singular, optional) |
| Cache config | `a_cache_dir` + `b_cache_dir` | `cache_dir` |
| Crossmap | Required | None |
| Output config | Required (results/review/unmatched paths) | Not required |
| `cross_map` section | Required | Absent |

### Config loading

At load time, enroll-mode config is normalised into the existing internal `Config` struct:
- `field: X` expands to `field_a: X, field_b: X`
- `dataset:` maps to `datasets.a:` (B side left empty)
- `cache_dir:` maps to `a_cache_dir:` (no B cache)
- Validation skips crossmap and output path requirements

This means the scoring pipeline, blocking, vector indexing, and BM25 all work unchanged — they see a normal A-side config.

## API Endpoints

### `POST /api/v1/enroll`

Score a record against the pool, then add it to the pool.

**Request:**
```json
{
  "record": {
    "entity_id": "ent_003",
    "legal_name": "Goldman Sachs International",
    "country_code": "GB",
    "lei": "W22LROWP2IHZNBB6K528"
  }
}
```

**Response:**
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

**Semantics:**
1. Encode the record's embedding fields
2. Search the pool index (ANN + BM25 candidates, same pipeline as `b/match`)
3. Full-score candidates on all match fields
4. Filter: return up to `top_n` edges with score >= `review_floor`
5. Add the record to the pool (store + vector index + BM25 index + blocking index)
6. Return edges

Steps 1-4 run before step 5, so the record does not match against itself.

**Edge cases:**
- If `id` already exists in the pool: upsert (replace record, re-score, return new edges)
- If pool is empty: `edges` is an empty array, record is added
- If no embedding fields configured: edges scored on fuzzy/exact/BM25 only

### `POST /api/v1/enroll-batch`

Score and enroll multiple records in sequence.

**Request:**
```json
{
  "records": [
    { "entity_id": "ent_004", "legal_name": "Morgan Stanley & Co", "country_code": "US" },
    { "entity_id": "ent_005", "legal_name": "MS International", "country_code": "US" }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "ent_004",
      "enrolled": true,
      "edges": [...]
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

**Semantics:**
Records are processed sequentially within the batch. Record N is scored against the pool including records 1..N-1 from the same batch. This means intra-batch edges are discovered.

### Utility endpoints (shared with live mode)

| Endpoint | Purpose | Available in enroll mode? |
|----------|---------|--------------------------|
| `GET /api/v1/health` | Pool size, model info | Yes |
| `GET /api/v1/status` | Uptime, counters | Yes |
| `POST /api/v1/enroll/remove` | Remove a record from the pool | Yes (new) |
| `GET /api/v1/enroll/query?id=X` | Look up a record by ID | Yes (new) |
| `GET /api/v1/enroll/count` | Pool size | Yes (new) |
| All `/api/v1/a/*`, `/api/v1/b/*` | Two-sided endpoints | Not mounted |
| All `/api/v1/crossmap/*` | Crossmap ops | Not mounted |
| All `/api/v1/review/*` | Review queue | Not mounted |

## Internal Architecture

### Under the hood

Enroll mode reuses the A-side infrastructure entirely:

- **RecordStore**: A-side `MemoryStore` or `SqliteStore` holds the pool
- **VectorDB**: A-side combined index holds pool embeddings
- **BlockingIndex**: A-side blocking index
- **BM25Index**: A-side BM25 index
- **Encoder pool**: Shared, same as live mode

The scoring pipeline (`score_pool()`) runs unchanged — the enrolled record is treated as a B-side query against the A-side pool, then added to the A side. The B-side state simply doesn't exist.

### Session changes

The `Session` struct gains an `enroll()` method:

```
pub fn enroll(&self, record: Record) -> Result<EnrollResponse, SessionError>
```

Internally:
1. Extract ID from record (using `id_field` from config)
2. Call the scoring pipeline (same as `try_match` with Side::B semantics)
3. Call `upsert_record` on Side::A (adds to pool)
4. Return edges (filtered by review_floor and top_n)
5. Skip crossmap, skip review queue, skip B-side ops

### What's NOT in scope

- No graph clustering — Melder produces edges, the caller clusters
- No multi-hop transitivity — each enrollment is scored independently
- No relationship types — edges are undirected similarity scores
- No entity merging — the pool contains individual records, not merged entities

## Documentation Notes

Key points for user documentation (to be written during implementation):

1. **Initial dataset is load-only, not enrolled.** Records from the config `dataset:` are loaded into the pool at startup but do not generate edges. This is the "known entities" reference set. Edges only come from subsequent `/enroll` calls.

2. **`enroll-batch` discovers intra-batch edges.** Records within a batch are processed sequentially, so record N sees records 1..N-1. The caller gets all edges within the batch plus edges to the pre-loaded pool.

3. **No crossmap means no 1:1 constraint.** A record can have edges to multiple pool members. The caller is responsible for deciding which connections are real matches (typically via graph clustering).

4. **Concurrency is non-deterministic.** Concurrent `/enroll` calls may or may not see each other's records. If deterministic edge discovery is required, serialize calls or use `enroll-batch`.

5. **The pool persists across restarts** (same as live mode — WAL or SQLite depending on config).
