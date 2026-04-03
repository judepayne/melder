---
type: architecture
module: general
status: active
tags: [flow, batch-mode, live-mode, pipeline]
related_code: [cli/run.rs, cli/serve.rs, batch/engine.rs, api/handlers.rs, matching/pipeline.rs]
---

# Business Logic Flow

## Batch Mode (`meld run`)

Entry point: `cli/run.rs` -> `batch/engine.rs::run_batch()`. Config is validated by `config/loader.rs` — see [[architecture/config_reference]].

### Pipeline per B record (parallelized with Rayon):

1. **Common ID pre-match** (optional): If `common_id_field` is configured on both datasets, records with identical common IDs are auto-matched with score 1.0 before any scoring. Runs first for all records.
2. **CrossMap skip**: If the B record is already in the CrossMap, skip it.
3. **Blocking filter**: If `blocking.enabled`, query the `BlockingIndex` (built from A records) with the B record to get candidate A-side IDs that share blocking key values. Supports AND/OR modes. If disabled, all A records are candidates.
4. **Candidate generation** (runs in parallel):
   - **ANN search**: Search the A-side combined embedding index with the B record's combined vector. With `usearch` backend: O(log N) HNSW search. Returns `ann_candidates` nearest neighbours.
   - **BM25 search**: Block-Max WAND via SimpleBm25, returns `bm25_candidates`.
   - **Synonym lookup**: Bidirectional index lookup for acronym matches.
5. **Union**: Deduplicate candidates by ID across all generators. Candidates found by ANY method pass through.
6. **Full scoring**: Each candidate is scored across all configured `match_fields` via `scoring::score_pair()` in `scoring/mod.rs`. Per-field embedding cosines are decomposed from the combined vectors (no second ONNX call). See [[decisions/key_decisions#Principles-Inviolable]] and [[architecture/scoring_algorithm]].
7. **Classification**: Composite score vs thresholds: `>= auto_match` -> confirmed, `>= review_floor` -> review, below -> no match.
8. **CrossMap claim**: If auto-match, atomically claim the A-B pair in the CrossMap. If the A record is already claimed, fall through to the next candidate.

### Output (batch/writer.rs):
- `results.csv`: confirmed matches
- `review.csv`: borderline pairs for human review
- `unmatched.csv`: B records with no match above review_floor

## Live Mode (`meld serve`)

Entry point: `cli/serve.rs` -> `api/server.rs` (Axum router) -> `api/handlers.rs`. Full endpoint reference: [[architecture/api_reference]].

### State: `session/mod.rs` wraps both sides as `LiveSideState` in `Arc<Session>`

### Add/Upsert flow:
1. Parse record from JSON request
2. Encode embedding fields via `EncoderPool` (or skip if text-hash unchanged -- `vectordb/texthash.rs`)
3. Store record in `DashMap`, upsert vector into combined index, update blocking index
4. Call `pipeline::score_pool()` against the opposite side -- same function as batch. See [[decisions/key_decisions#Principles-Inviolable]].
5. If top result >= auto_match, claim in CrossMap
6. Log to WAL (`state/upsert_log.rs`) — see [[architecture/state_and_persistence#WAL]]
7. Return matches as JSON

### Try-match flow (read-only):
Same as add but does not store the record or update any indices. Encode, score, return.

### Symmetry:
A and B sides have identical `LiveSideState` structs, identical handler logic, identical API endpoints (`/a/add`, `/b/add`, etc.). See [[decisions/key_decisions#Principles-Inviolable]].

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

See also: [[architecture/module_map]] for the full module directory, [[decisions/key_decisions]] for the rationale behind major design choices, [[benchmarks_and_experiments]] for throughput numbers.
