---
type: architecture
module: general
status: active
tags: [flow, batch-mode, live-mode, pipeline]
related_code: [cli/run.rs, cli/serve.rs, batch/engine.rs, api/handlers.rs, matching/pipeline.rs]
---

# Business Logic Flow

## Batch Mode (`meld run`)

Entry point: `cli/run.rs` -> `batch/engine.rs::run_batch()`. Config is validated by `config/loader.rs` — see [[Config Reference]].

### Pipeline per B record (parallelized with Rayon):

1. **Common ID pre-match** (optional): If `common_id_field` is configured on both datasets, records with identical common IDs are auto-matched with score 1.0 before any scoring. Runs first for all records.
2. **CrossMap skip**: If the B record is already in the [[Constitution#3 CrossMap Bijection 1 1 Under One Lock|CrossMap]], skip it.
3. **Blocking filter**: If `blocking.enabled`, query the `BlockingIndex` (built from A records) with the B record to get candidate A-side IDs that share blocking key values. Supports AND/OR modes. If disabled, all A records are candidates.
4. **Candidate selection**: Search the A-side combined embedding index with the B record's combined vector. With `usearch` backend: O(log N) HNSW search. With `flat` backend: O(N) brute-force scan. Returns `top_n` nearest neighbours. If no embedding fields are configured, all blocked records pass through.
5. **Full scoring**: Each candidate is scored across all configured `match_fields` via `scoring::score_pair()` in `scoring/mod.rs`. Per-field embedding cosines are decomposed from the combined vectors (no second ONNX call). See [[Constitution#4 Combined Vector Weighted Cosine Identity]] and [[Scoring Algorithm]].
6. **Classification**: Composite score vs thresholds: `>= auto_match` -> confirmed, `>= review_floor` -> review, below -> no match.
7. **CrossMap claim**: If auto-match, atomically claim the A-B pair in the CrossMap. If the A record is already claimed, fall through to the next candidate.

### Output (batch/writer.rs):
- `results.csv`: confirmed matches
- `review.csv`: borderline pairs for human review
- `unmatched.csv`: B records with no match above review_floor

## Live Mode (`meld serve`)

Entry point: `cli/serve.rs` -> `api/server.rs` (Axum router) -> `api/handlers.rs`. Full endpoint reference: [[API Reference]].

### State: `session/mod.rs` wraps both sides as `LiveSideState` in `Arc<Session>`

### Add/Upsert flow:
1. Parse record from JSON request
2. Encode embedding fields via `EncoderPool` (or skip if text-hash unchanged -- `vectordb/texthash.rs`)
3. Store record in `DashMap`, upsert vector into combined index, update blocking index
4. Call `pipeline::score_pool()` against the opposite side -- same function as batch. See [[Constitution#2 One Scoring Pipeline]].
5. If top result >= auto_match, claim in CrossMap
6. Log to WAL (`state/upsert_log.rs`) — see [[State & Persistence#WAL]]
7. Return matches as JSON

### Try-match flow (read-only):
Same as add but does not store the record or update any indices. Encode, score, return.

### Symmetry:
A and B sides have identical `LiveSideState` structs, identical handler logic, identical API endpoints (`/a/add`, `/b/add`, etc.). See [[Constitution#1 Batch Asymmetry Live Symmetry]].

## Shared Components

| Component | Location | Used By |
|---|---|---|
| Scoring pipeline | `matching/pipeline.rs` | Batch + Live |
| Score computation | `scoring/mod.rs` | Via pipeline |
| Candidate selection | `matching/candidates.rs` | Via pipeline |
| Blocking | `matching/blocking.rs` | Batch + Live |
| CrossMap | `crossmap/mod.rs` | Batch + Live |
| Encoder pool | `encoder/mod.rs` | Batch + Live |
| Vector index trait | `vectordb/mod.rs` | Batch + Live |
| Config validation | `config/loader.rs` | All modes |

See also: [[Module Map]] for the full module directory, [[Key Decisions]] for the rationale behind major design choices, [[Performance Baselines]] for throughput numbers.
