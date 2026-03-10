# Plan: Unified Embedding Index + O(log N) Pipeline Redesign

## Background and Motivation

Two problems exist in the current pipeline:

**1. The usearch HNSW index is never actually used for search.**
`VectorDB::search()` is implemented, tested, and correct, but nothing in the
production code path calls it. Every embedding comparison at runtime goes
through `VectorDB::get()` (individual point-lookup by ID) inside a brute-force
loop over all blocked candidates. The O(log N) property of HNSW is never
realised. Flat and usearch perform equivalently because both execute the same
loop.

**2. The `candidates:` config section forces users to reason about two phases
that are in conceptual conflict.**
The candidates phase and the full scoring phase both attempt to rank records,
potentially using overlapping signals (embedding similarity appears in both).
Users must configure weights in two places and understand how the two rankings
interact. This is unnecessary complexity.

---

## Design

### Core idea

All `method: embedding` match fields are combined into a single vector per
record at encoding time. A single HNSW index (one per side, one per block)
stores these combined vectors. At query time, one `search(k=top_n)` call
retrieves the top-N semantically similar candidates in O(log N). Full scoring
then runs only on those N candidates.

The user sees one scoring config with methods and weights per field. Melder
handles the rest internally.

### Mathematical basis

For embedding fields with unit-normalised per-field vectors and weights w₁, w₂:

```
combined_a = [√w₁·a₁, √w₂·a₂]
combined_b = [√w₁·b₁, √w₂·b₂]

‖combined_a‖ = √(w₁ + w₂)   (since a₁, a₂ are unit-normalised)

cos_sim(combined_a, combined_b)
  = (w₁·cos_sim(a₁,b₁) + w₂·cos_sim(a₂,b₂)) / (w₁ + w₂)
```

Therefore:

```
combined_similarity × (w₁ + w₂)
  = w₁·cos_sim(a₁,b₁) + w₂·cos_sim(a₂,b₂)
```

This **exactly** recovers the weighted sum of individual field similarities —
identical to what the current per-field full scoring computes for the embedding
component. No approximation. Generalises to any number of embedding fields.

### New pipeline (two stages)

**Stage 1 — ANN search, O(log N) with usearch**

- Encode B record's embedding fields → scale each by √weight → concatenate
  → combined query vector
- `combined_index_a.search(k=top_n, query_vec, b_record, Side::B)`
- Returns: `Vec<(a_id, combined_similarity)>`, sorted descending

**Stage 2 — Full scoring, O(top_n) ≈ constant**

For each `(a_id, combined_similarity)` from Stage 1:
- `embedding_contribution = combined_similarity × Σ(embedding_field_weights)`
- For each fuzzy field: compute `fuzzy_score × weight`
- For each exact field: compute `exact_score × weight`
- `total = embedding_contribution + Σ(other contributions)`
- Apply existing normalisation (divide by total_weight if it does not sum to 1.0)

Return sorted list. Batch takes top-1. Live returns all top_n.

### flat backend behaviour

The flat backend's `search()` is not block-aware — it searches all stored
vectors regardless of the query record's blocking key. To preserve correct
blocking semantics for flat, the Stage 1 path for flat falls back to: iterate
the blocked candidate ID list (from the BlockingIndex, as today) → `get_vec(id)`
on the combined index → manual cosine similarity of the combined vectors →
sort → truncate to top_n. This produces `combined_similarity` values and the
rest of the pipeline is identical. O(N) behaviour is preserved, unchanged from
today, but simplified to one vector lookup per candidate instead of one per
embedding field.

---

## Config changes

### Removed entirely
- `candidates:` section (all fields: `enabled`, `field_a`, `field_b`, `method`,
  `scorer`, `n`)
- `CandidatesConfig` struct
- `live.top_n` field from `LiveConfig`

### Added
- `top_n: <usize>` at the **top level** of `Config`
  - Optional, defaults to **5** if absent or zero
  - Governs: (a) how many candidates HNSW retrieves, (b) how many get
    full-scored, (c) how many results live mode returns to the caller
  - Batch mode always returns 1 result (top scorer) regardless of this value

### Unchanged
- `live.upsert_log`
- `live.crossmap_flush_secs`
- All other config sections (`blocking`, `match_fields`, `thresholds`,
  `embeddings`, `output`, `performance`, `vector_backend`, etc.)

### Before / after example

```yaml
# BEFORE
candidates:
  enabled: true
  field_a: short_name
  field_b: counterparty_name
  method: fuzzy
  scorer: wratio
  n: 10

live:
  top_n: 5
  upsert_log: bench/wal.jsonl
  crossmap_flush_secs: 10

# AFTER
top_n: 10          # optional — defaults to 5

live:
  upsert_log: bench/wal.jsonl
  crossmap_flush_secs: 10
```

---

## Vector index changes

### Before
- `FieldIndexes` holds one `Box<dyn VectorDB>` per embedding field per side,
  keyed by `"field_a/field_b"`
- Cache files: `a.legal_name__counterparty_name.index`,
  `a.short_name__counterparty_name.index` (one per field)
- Each index dimension: 384
- `search()` never called; `get()` called per candidate per field

### After
- One `Box<dyn VectorDB>` per side — the **combined embedding index**
- Cache file: `a.combined_embedding.index` (flat) or
  `a.combined_embedding.usearchdb/` (usearch)
- Index dimension: 384 × N_embedding_fields (e.g. 768 for two fields)
- `search(k=top_n)` called once per B record (usearch path)

### `FieldIndexes` role
`FieldIndexes` currently serves embedding fields only. With a single combined
index it is no longer needed for that purpose. It should be removed or
repurposed. The combined index is held directly on the engine/session state
as `combined_index_a: Box<dyn VectorDB>` and `combined_index_b: Box<dyn
VectorDB>`.

Non-embedding fields (fuzzy, exact) never touch the vector index; they work
from the record structs directly, unchanged.

---

## Per-field embedding score decomposition

Currently the output includes per-field scores, e.g. `legal_name: 0.87,
short_name: 0.71`. With a combined index only `combined_similarity` is
directly available from `search()`.

**Recommended approach — recover per-field scores at full-scoring time:**

For each of the top_n candidates, fetch the A record's combined vector via
`get(a_id)` on the combined index. Split it into per-field sub-vectors using
known field dimensions and √weight scaling. Compute individual cosine
similarities against the B record's corresponding sub-vectors (which are
already in memory from Stage 1 encoding). This adds one `get()` call per
candidate (top_n ≈ 10 calls total) — negligible cost, full interpretability
preserved.

Field offsets into the combined vector are deterministic from config order:
- Field 0: bytes 0 … 383
- Field 1: bytes 384 … 767
- etc.

Store this metadata (field order, dimensions, weights) alongside the index so
the decomposition can be performed correctly on load.

---

## Cache invalidation

Existing `.index` / `.usearchdb` cache files are **incompatible** with the new
design:
- Dimension changes (384 → 384×N)
- Structure changes (per-field → combined)

On startup, if the cache file exists but has the wrong dimension or wrong
structure, melder must detect this and force a full rebuild with a clear log
message. This makes TODO item 2 (config hash in manifest) urgent — the
combined vector structure is config-dependent, and any change to embedding
fields or weights invalidates all caches.

---

## Files to change

### `src/config/schema.rs`
- Delete `CandidatesConfig` struct and the `candidates` field on `Config`
- Delete `top_n` from `LiveConfig`
- Add `pub top_n: Option<usize>` to `Config` with `#[serde(default)]`

### `src/config/loader.rs`
- Delete all candidates validation logic (`VALID_CANDIDATE_METHODS`, field
  presence checks, method/scorer validation)
- Add `top_n` defaulting: if `None` or `Some(0)`, set `Some(5)`
- Remove candidate field references from `required_fields_*` computation
- Update any `cfg.live.top_n` references to `cfg.top_n`

### `src/vectordb/` — new function: `encode_combined_vector`
```
fn encode_combined_vector(
    record: &Record,
    embedding_fields: &[(MatchField, Side)],   // fields with method: embedding
    encoder_pool: &EncoderPool,
) -> Vec<f32>
```
For each embedding field in config order:
1. Get the field value from the record (empty string if missing)
2. Encode → unit-normalised vec (dim 384)
3. Scale by √weight: each component × √weight
4. Append to output buffer

Returns combined vector of dimension 384 × N_embedding_fields.

### `src/vectordb/` — replace `build_or_load_field_indexes`
Replace with `build_or_load_combined_index` that:
- Determines embedding fields from config
- Computes combined dimension
- Checks cache for existing combined index (validates dimension)
- If stale or absent: iterates all records, calls `encode_combined_vector`,
  upserts into a single `VectorDB` instance, saves to cache
- Returns `Box<dyn VectorDB>` (the combined index)

### `src/matching/candidates.rs`
Rewrite. New signature:
```
fn select_candidates(
    query_combined_vec: &[f32],
    top_n: usize,
    combined_index_a: &dyn VectorDB,
    blocked_ids: Option<&[String]>,   // None if blocking disabled
    pool_records: &DashMap<String, Record>,
    b_record: &Record,
    config: &Config,
) -> Vec<Candidate>
```
- **usearch path**: call `combined_index_a.search(query_combined_vec, top_n,
  b_record, Side::B)` → map results to `Candidate` structs carrying
  `combined_similarity`
- **flat path**: iterate `blocked_ids`, call `get(id)` on combined index,
  compute cosine similarity manually, sort, truncate to top_n

`Candidate` struct gains `combined_similarity: f64`; `emb_scores` HashMap is
replaced or supplemented by the decomposed per-field scores (computed
immediately from the combined vector split as described above).

### `src/matching/pipeline.rs`
- `score_pool` receives `combined_index_a` and `query_combined_vec` instead of
  `query_field_indexes` and `pool_field_indexes`
- Stage 1: call `select_candidates` (new version above)
- Stage 2: for each candidate, build `precomputed_emb_scores` from decomposed
  per-field scores; call `scoring::score_pair` unchanged
- `top_n` comes from `config.top_n`

### `src/scoring/mod.rs`
No change required. `score_pair` already accepts
`precomputed_emb_scores: Option<&HashMap<String, f64>>` and uses it as-is.
Per-field scores (decomposed from the combined vector) slot straight in.

### `src/batch/engine.rs`
- Replace `field_indexes_a` / `field_indexes_b` with
  `combined_index_a` / `combined_index_b`
- Pre-compute `query_combined_vec` for each B record before calling
  `score_pool`
- Remove all `candidates`-related setup code
- `top_n` sourced from `config.top_n` (not hardcoded 0 + `.next()`)
- Batch still takes `.into_iter().next()` on the scored list for the final
  classification decision

### `src/session/mod.rs`
- Replace `field_indexes_a` / `field_indexes_b` with combined indexes
- `top_n` sourced from `config.top_n` (not `config.live.top_n`)
- WAL replay: re-encode combined vector on record change, upsert to combined
  index

### `src/state/` (live and batch state)
- `MatchState` / `LiveSideState`: replace `FieldIndexes` embedding component
  with `combined_index: Box<dyn VectorDB>`
- Save/load: update cache paths and dimension metadata

### `testdata/configs/*.yaml`
Remove `candidates:` section from every config that has one. Add `top_n: N`
where the desired value differs from the default of 5.

Configs to update:
- `bench1kx1k.yaml`
- `bench10kx10k.yaml`
- `bench100kx100k.yaml`
- `bench_live.yaml`
- Any others with `candidates:` or `live.top_n`

### `README.md`
- Update pipeline description (remove candidates phase explanation)
- Update config reference (remove `candidates:`, document `top_n`)
- Update performance section (numbers will change — re-benchmark after
  implementation)

### `TODO.md`
- Mark item 2 (config hash in manifest) as now **urgent** — combined vector
  structure is config-dependent; cache invalidation on field/weight change is
  critical for correctness

---

## Implementation sequence

Each step is independently compilable and testable before moving to the next.

1. **Config schema and loader** — delete `CandidatesConfig`, move `top_n` to
   top level, update defaults and validation. Update all YAML configs. All
   existing tests should still pass (candidates code still compiles, just
   config struct changes).

2. **`encode_combined_vector`** — new standalone function with unit tests
   verifying the √weight scaling and concatenation. Test that
   `combined_similarity × Σ(weights)` equals the weighted sum of individual
   cosine similarities for known vectors.

3. **`build_or_load_combined_index`** — replaces field index building. Test
   cold build and warm load roundtrip. Test stale-cache detection.

4. **`candidates.rs` rewrite** — new `select_candidates` using combined index
   + `search()` for usearch, blocked-ID loop for flat. Unit test both paths.

5. **`pipeline.rs` update** — wire new `select_candidates` into `score_pool`.
   Integration test: full pipeline on small dataset, verify scores match
   current output for the same config.

6. **`batch/engine.rs` and `session/mod.rs`** — update call sites to use
   combined indexes and new `top_n`. Remove FieldIndexes embedding component.

7. **Per-field score decomposition** — implement vector split + recompute in
   the candidates → full-scoring handoff. Verify per-field scores in output
   match the current values.

8. **Remove `FieldIndexes`** — or repurpose as a thin wrapper if still useful
   for non-embedding bookkeeping. Delete dead code.

9. **Tests** — update all unit and integration tests. Add benchmarks for the
   O(log N) path to verify the improvement is real and measurable.

10. **Docs and README** — update config reference, pipeline description, and
    performance tables (re-run benchmarks).

---

## Open questions (to resolve during implementation)

1. **Cache file naming**: `combined_embedding` is proposed. Confirm this does
   not conflict with any existing cache file naming convention.

2. **Missing field values**: if a record has no value for one of the embedding
   fields, encode as empty string (consistent with current behaviour) or zero
   vector? Empty string encoding is safer — it produces a real vector, and
   similarity against another empty-string vector is ~1.0 (both "empty"), which
   is arguably correct.

3. **`FieldIndexes` fate**: fully delete or retain as a type alias / thin
   wrapper for the single combined index? Lean toward deletion to remove dead
   concepts.

4. **usearch + blocking disabled**: when `blocking.enabled: false`, usearch
   uses a single `__default__` block. `search()` then searches all records.
   This is correct behaviour — verify in tests.

5. **Re-benchmarking**: after implementation, re-run 100k benchmarks with
   usearch + `top_n: 20` to confirm O(log N) is realised and document the
   numbers in README.
