---
type: architecture
module: api
status: active
tags: [api, http, live-mode, endpoints, reference]
related_code: [src/api/handlers.rs, src/api/server.rs]
---

# API Reference

Complete HTTP API for live mode (`meld serve`). The router is built in `api/server.rs` and binds to `0.0.0.0:{port}`. All routes are prefixed `/api/v1/`.

Two modes:
- **Match mode** (default): Full A/B symmetric API
- **Enroll mode** (`meld enroll`): Single-pool enrollment API

**Middleware applied to all routes:** `CatchPananicLayer` (panic → 500 JSON), `TraceLayer` (HTTP tracing).

See [[architecture/business_logic_flow#live_mode]] for how requests flow through the system. See [[architecture/config_reference#live]] for port and WAL configuration.

---

## Record Endpoints

Symmetric — A and B sides have identical shapes. All A-side routes have a `/b/` equivalent. See [[decisions/key_decisions#Principles-Inviolable]].

### `POST /api/v1/{side}/add`

Upsert a record. Encodes embedding fields, stores the record, updates the blocking and vector indices, scores against the opposite side, claims a CrossMap pair if score ≥ `auto_match`, appends to WAL.

**Request:**
```json
{ "record": { "field_name": "value", ... } }
```
The record must contain the `id_field` configured for this side.

**Response:** `UpsertResponse` — contains `id`, `status`, and `matches` array (see Match Result Shape below).

**Side effects:** modifies state (record store, indices, CrossMap, WAL). Not idempotent — re-calling with the same ID upserts (updates) the record.

---

### `POST /api/v1/{side}/match`

Read-only similarity search. Encodes the record and scores it against the opposite side. **Does not store the record or modify any state.** Use this for duplicate detection before committing a record. See [[business/use_cases#2 Live Duplicate Detection]].

**Request:** `{ "record": { ... } }` — same shape as `/add`.

**Response:** same match result shape as `/add`, without the upsert side effects.

---

### `POST /api/v1/{side}/remove`

Remove a record from the index. Deletes from record store, blocking index, and vector index. Breaks any CrossMap pair involving this record. Appends a `remove_record` event to the WAL.

**Request:** `{ "id": "record-id" }`

**Response:** `200 OK` on success, `404` if the record was not found.

---

### `GET /api/v1/{side}/query?id=X`

Fetch a stored record's field values by ID.

**Response:** the record's fields as a JSON object. `404` if not found.

---

## Batch Record Endpoints

Same semantics as single-record endpoints; accepts and returns arrays. Max 1,000 records per request.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/{side}/add-batch` | Upsert multiple records |
| `POST` | `/api/v1/{side}/match-batch` | Read-only batch match (no state change) |
| `POST` | `/api/v1/{side}/remove-batch` | Remove multiple records by ID |

**`add-batch` / `match-batch` request:**
```json
{ "records": [ { "field": "value", ... }, ... ] }
```

**`remove-batch` request:**
```json
{ "ids": ["id-1", "id-2", ...] }
```

**`add-batch` / `match-batch` response:** array of per-record results in the same order as the input.

---

## CrossMap Endpoints

The CrossMap is the 1:1 ledger of confirmed pairs. See [[decisions/key_decisions#Principles-Inviolable]].

### `POST /api/v1/crossmap/confirm`

Manually confirm a pair without scoring. Useful for importing known matches. Appends a `crossmap_confirm` event to the WAL.

**Request:** `{ "a_id": "...", "b_id": "..." }`

**Response:** `200 OK` or `400` if either ID does not exist.

---

### `POST /api/v1/crossmap/break`

Break a confirmed pair. Both records return to the unmatched pool and may be re-matched by future upserts. Appends a `crossmap_break` event to the WAL.

**Request:** `{ "a_id": "...", "b_id": "..." }`

---

### `GET /api/v1/crossmap/lookup?id=X&side=a`

Look up the paired ID for a given record.

**Query params:** `id` (record ID), `side` (`"a"` or `"b"`).

**Response:** `{ "paired_id": "..." }` or `404` if the record has no confirmed pair.

---

### `GET /api/v1/crossmap/pairs?offset=0&limit=100`

Paginated list of all confirmed pairs.

**Query params:** `offset` (default 0), `limit` (default 100).

**Response:**
```json
{
  "pairs": [ { "a_id": "...", "b_id": "..." }, ... ],
  "total": 1234,
  "offset": 0,
  "limit": 100
}
```

---

### `GET /api/v1/crossmap/stats`

Summary statistics for the CrossMap.

**Response:**
```json
{
  "total_pairs": 1234,
  "a_matched": 1234,
  "b_matched": 1234
}
```

---

## Exclusion Endpoints

Manage known non-matching pairs. Available in both match and enroll modes.

### `POST /api/v1/exclude`

Add a pair to the exclusions set. If the pair is currently matched in the CrossMap, the match is broken first. Appends an `Exclude` event to the WAL.

**Request:** `{ "a_id": "...", "b_id": "..." }`

**Response:** `ExcludeResponse` with `excluded: true` and `match_was_broken: bool` (true if a CrossMap pair was broken).

---

### `DELETE /api/v1/exclude`

Remove a pair from the exclusions set. Appends an `Unexclude` event to the WAL.

**Request:** `{ "a_id": "...", "b_id": "..." }`

**Response:** `UnexcludeResponse` with `unexcluded: true`.

---

## Review and Unmatched Endpoints

### `GET /api/v1/review/list?offset=0&limit=100`

Paginated list of pending review-band matches (score between `review_floor` and `auto_match`). Entries are drained when either party is confirmed, broken, re-upserted, or removed.

**Response:**
```json
{
  "items": [
    { "id": "...", "side": "a", "candidate_id": "...", "score": 0.72 },
    ...
  ],
  "total": 42,
  "offset": 0,
  "limit": 100
}
```

---

### `GET /api/v1/{side}/unmatched?include_records=false&offset=0&limit=100`

Paginated list of records on the given side that are not in the CrossMap.

**Query params:** `include_records` (bool, default false — if true, full record fields are included), `offset`, `limit`.

**Response:**
```json
{
  "ids": ["id-1", "id-2", ...],
  "records": { "id-1": { "field": "value" }, ... },   // only if include_records=true
  "total": 567,
  "offset": 0,
  "limit": 100
}
```

---

## Utility Endpoints

### `GET /api/v1/health`

Liveness check. Returns `200 OK` when the server is up and ready.

### `GET /api/v1/status`

Detailed session status: record counts per side, CrossMap size, uptime, config summary.

---

## Match Result Shape

Returned by `/add`, `/match`, `/add-batch`, `/match-batch`.

```json
{
  "id": "ENT-001",
  "status": "auto",
  "matches": [
    {
      "matched_id": "CP-042",
      "score": 0.91,
      "classification": "auto",
      "from_crossmap": false,
      "field_scores": [
        {
          "field_a": "name",
          "field_b": "company_name",
          "method": "embedding",
          "score": 0.94,
          "weight": 0.6
        },
        ...
      ],
      "matched_record": { "company_name": "...", ... }   // only if top match
    }
  ]
}
```

`status` values: `"auto"` (confirmed), `"review"` (borderline), `"no_match"`. Mirrors `Classification`. `from_crossmap: true` when the result came from the CrossMap rather than live scoring (e.g., re-upsert of a record already confirmed).

---

## Error Responses

All errors return JSON: `{ "error": "message" }`.

| Status | When |
|---|---|
| `400 BAD_REQUEST` | Invalid input, upsert failure, CrossMap conflict |
| `404 NOT_FOUND` | Record not found (query, remove) |
| `500 INTERNAL_SERVER_ERROR` | Serialization errors, task join failures |

---

## Backward-Compatibility Alias

`POST /api/v1/match/b` is an alias for `POST /api/v1/b/match` (retained for older clients).

---

See also: [[architecture/scoring_algorithm]] for how match scores are computed, [[architecture/state_and_persistence]] for how upserts and CrossMap changes are persisted, [[architecture/config_reference#live]] for server configuration.
