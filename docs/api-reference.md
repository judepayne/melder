← [Back to README](../README.md) | [Live Mode](live-mode.md) | [Enroll Mode](enroll-mode.md) | [CLI Reference](cli-reference.md)

# Live Mode API Reference

All endpoints are under `/api/v1/`. The server is started with
`meld serve --config config.yaml --port 8090`.

> For enroll-mode endpoints (`meld enroll`), see [Enroll Mode](enroll-mode.md).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/a/add` | Add or update an A-side record, return top matches from B |
| POST | `/b/add` | Add or update a B-side record, return top matches from A |
| POST | `/a/add-batch` | Add or update multiple A-side records in one request |
| POST | `/b/add-batch` | Add or update multiple B-side records in one request |
| POST | `/a/match` | Score an A-side record against B without storing it (read-only) |
| POST | `/b/match` | Score a B-side record against A without storing it (read-only) |
| POST | `/a/match-batch` | Score multiple A-side records against B without storing (read-only) |
| POST | `/b/match-batch` | Score multiple B-side records against A without storing (read-only) |
| POST | `/a/remove` | Remove an A-side record from all indices and break any crossmap pair |
| POST | `/b/remove` | Remove a B-side record from all indices and break any crossmap pair |
| POST | `/a/remove-batch` | Remove multiple A-side records in one request |
| POST | `/b/remove-batch` | Remove multiple B-side records in one request |
| GET | `/a/query?id=X` | Look up an A-side record and its crossmap status |
| GET | `/b/query?id=X` | Look up a B-side record and its crossmap status |
| POST | `/crossmap/confirm` | Confirm a match (add to cross-map) |
| POST | `/crossmap/break` | Break a confirmed match (remove from cross-map) |
| GET | `/crossmap/lookup?id=X&side=a` | Look up whether a record has a confirmed match |
| GET | `/crossmap/pairs` | Export all confirmed crossmap pairs (paginated) |
| GET | `/crossmap/stats` | Coverage statistics (matched/unmatched counts per side) |
| GET | `/a/unmatched` | List A-side record IDs with no crossmap pair (paginated) |
| GET | `/b/unmatched` | List B-side record IDs with no crossmap pair (paginated) |
| GET | `/review/list` | List pending review-band matches (paginated) |
| GET | `/health` | Health check |
| GET | `/status` | Detailed server status (record counts, uptime) |

> [!IMPORTANT]
> Live mode treats A and B sides identically. Adding, removing, querying,
> and matching records works the same way on both sides — same operations,
> same scoring logic, same match semantics.

## Adding a record

When you add a record to one side, the melder immediately encodes it,
searches the opposite side for matches, and returns the top results.
If the record already exists (same ID), it is updated and re-matched.

Request:

```json
POST /api/v1/a/add

{
  "record": {
    "entity_id": "ENT-001",
    "legal_name": "Acme Corporation",
    "country_code": "US"
  }
}
```

Response:

```json
{
  "id": "ENT-001",
  "status": "added",
  "matches": [
    {
      "id": "CP-042",
      "score": 0.91,
      "classification": "auto",
      "field_scores": [...]
    }
  ]
}
```

The `status` field will be `"added"` for new records or `"updated"` for
existing records that were modified.

## Removing a record

Remove a record by ID. This removes it from all indices (embedding,
blocking, unmatched set) and breaks any existing crossmap pair. The
opposite-side record in a broken pair is returned to the unmatched pool.

Request:

```json
POST /api/v1/a/remove

{
  "id": "ENT-001"
}
```

Response:

```json
{
  "status": "removed",
  "id": "ENT-001",
  "side": "a",
  "crossmap_broken": ["CP-042"]
}
```

The `crossmap_broken` array lists any opposite-side IDs whose pairing
was broken by the removal. It is omitted when empty.

## Batch operations

The batch endpoints accept multiple records in a single request. This
amortises the ONNX encoding cost across the batch — all texts are
encoded in a single model call, then scored sequentially. Maximum 1000
records per request. Empty arrays return 400.

### Add batch

Add or update multiple records at once:

```json
POST /api/v1/a/add-batch

{
  "records": [
    {"entity_id": "ENT-001", "legal_name": "Acme Corp", "country_code": "US"},
    {"entity_id": "ENT-002", "legal_name": "Globex Inc", "country_code": "GB"}
  ]
}
```

Response:

```json
{
  "results": [
    {"id": "ENT-001", "status": "added", "matches": [...], ...},
    {"id": "ENT-002", "status": "added", "matches": [...], ...}
  ]
}
```

Each entry in `results` has the same shape as a single `/add` response.
If a record fails (e.g. missing ID field), its entry has
`"status": "error"` with a message — the rest of the batch still
succeeds.

### Match batch

Score multiple records without storing them:

```json
POST /api/v1/b/match-batch

{
  "records": [
    {"counterparty_id": "CP-X", "counterparty_name": "Test Corp", "domicile": "US"},
    {"counterparty_id": "CP-Y", "counterparty_name": "Another Inc", "domicile": "GB"}
  ]
}
```

### Remove batch

Remove multiple records by ID:

```json
POST /api/v1/a/remove-batch

{
  "ids": ["ENT-001", "ENT-002", "NONEXISTENT"]
}
```

Response:

```json
{
  "results": [
    {"id": "ENT-001", "side": "a", "status": "removed"},
    {"id": "ENT-002", "side": "a", "status": "removed"},
    {"id": "NONEXISTENT", "side": "a", "status": "not_found"}
  ]
}
```

Missing IDs produce `"status": "not_found"` entries rather than failing
the request.

### Throughput

Batch endpoints are faster than sending single requests
in a loop because they amortise ONNX encoding overhead. On the 10K x
10K benchmark dataset:

| Batch size | Throughput (rec/s) | Speedup vs single |
|:----------:|-------------------:|:-----------------:|
| 1          | 221                | 0.9x              |
| 10         | 331                | 1.4x              |
| 50         | 445                | 1.8x              |
| 100        | 319                | 1.3x              |
| 500        | 325                | 1.3x              |

Batch size 50 is the sweet spot — large enough to amortise encoding,
small enough that per-batch latency stays under 200ms.

## Querying a record

Look up a record by ID to see its full contents and crossmap status
without modifying anything.

```
GET /api/v1/a/query?id=ENT-001
```

Response:

```json
{
  "id": "ENT-001",
  "side": "a",
  "record": {
    "entity_id": "ENT-001",
    "legal_name": "Acme Corporation",
    "country_code": "US"
  },
  "crossmap": {
    "status": "matched",
    "paired_id": "CP-042",
    "paired_record": {
      "counterparty_id": "CP-042",
      "counterparty_name": "ACME Corp"
    }
  }
}
```

If the record is unmatched, `crossmap.status` is `"unmatched"` and the
`paired_id` and `paired_record` fields are omitted. Returns 404 if the
record ID is not found.

## Crossmap operations

### Export pairs

List all confirmed crossmap pairs, with optional pagination:

```
GET /api/v1/crossmap/pairs?offset=0&limit=100
```

```json
{
  "total": 4523,
  "offset": 0,
  "pairs": [
    { "a_id": "ENT-001", "b_id": "CP-042" },
    { "a_id": "ENT-007", "b_id": "CP-119" }
  ]
}
```

### Coverage statistics

```
GET /api/v1/crossmap/stats
```

```json
{
  "records_a": 10000,
  "records_b": 9500,
  "crossmap_pairs": 4523,
  "matched_a": 4523,
  "matched_b": 4523,
  "unmatched_a": 5477,
  "unmatched_b": 4977,
  "coverage_a": 0.4523,
  "coverage_b": 0.4761
}
```

## Unmatched records

List record IDs that have no crossmap pair on a given side. Supports
pagination and an optional `include_records=true` parameter to return
full record data alongside each ID.

```
GET /api/v1/a/unmatched?offset=0&limit=50
GET /api/v1/b/unmatched?include_records=true&limit=10
```

Response (without `include_records`):

```json
{
  "side": "a",
  "total": 5477,
  "offset": 0,
  "records": [
    { "id": "ENT-003" },
    { "id": "ENT-009" }
  ]
}
```

Response (with `include_records=true`):

```json
{
  "side": "b",
  "total": 4977,
  "offset": 0,
  "records": [
    {
      "id": "CP-055",
      "record": {
        "counterparty_id": "CP-055",
        "counterparty_name": "Initech LLC",
        "domicile": "US"
      }
    }
  ]
}
```

## Review list

List pending review-band matches — pairs that scored between
`review_floor` and `auto_match` during upsert but were not
auto-confirmed. Resolution happens via the `/crossmap/confirm` and
`/crossmap/break` endpoints.

```
GET /api/v1/review/list?offset=0&limit=20
```

```json
{
  "total": 37,
  "offset": 0,
  "reviews": [
    {
      "id": "ENT-012",
      "side": "a",
      "candidate_id": "CP-088",
      "score": 0.74
    },
    {
      "id": "CP-201",
      "side": "b",
      "candidate_id": "ENT-055",
      "score": 0.69
    }
  ]
}
```

Reviews are sorted by score descending (highest-confidence pairs first).
Confirming or breaking a pair removes it from the review queue.
Re-upserting a record also clears its stale review entries.
