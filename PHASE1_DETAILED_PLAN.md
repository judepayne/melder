# Phase 1: BM25 Scoring + Filtering — Detailed Plan

Reference: `PHASED_PLAN_HIGH_LEVEL.md` (Phase 1), `vault/ideas/Scaling to Millions.md`

## Implementation Steps

Ordered for incremental testability. Each step produces a compilable, testable state.

---

### Step 1: Config schema changes

**Files:** `src/config/schema.rs`

**Decision: Keep `field_a` and `field_b` as `String`.** For `method: bm25`, they are empty strings (`""`). Validation (Step 2) rejects empty fields for all other methods. This avoids an invasive `Option<String>` migration across every call site.

1. Add `ann_candidates` and `bm25_candidates` to `Config`:
   ```rust
   #[serde(default)]
   pub ann_candidates: Option<usize>,
   #[serde(default)]
   pub bm25_candidates: Option<usize>,
   ```

2. Add `bm25_fields` derived field:
   ```rust
   #[serde(skip)]
   pub bm25_fields: Vec<(String, String)>,  // derived: (field_a, field_b) from fuzzy/embedding entries
   ```

No changes needed to `MatchField` struct — `field_a` and `field_b` remain `pub field_a: String` and `pub field_b: String`. For BM25 entries, serde will deserialise missing fields as empty strings (already has `#[serde(default)]`... wait, it doesn't currently have `#[serde(default)]`). So we need to add `#[serde(default)]` to `field_a` and `field_b` on `MatchField`:

```rust
#[derive(Debug, Deserialize)]
pub struct MatchField {
    #[serde(default)]
    pub field_a: String,
    #[serde(default)]
    pub field_b: String,
    pub method: String,
    #[serde(default)]
    pub scorer: Option<String>,
    pub weight: f64,
}
```

**Compile check:** `cargo build` succeeds. No existing call sites need changes — `field_a` and `field_b` are still `String`.

---

### Step 2: Config validation + defaults

**Files:** `src/config/loader.rs`

1. Add `"bm25"` to `VALID_METHODS`.

2. In `apply_defaults()`:
   - Default `ann_candidates` to 50 if not set or 0.
   - Default `bm25_candidates` to 10 if not set or 0.

3. In `validate()`:
   - For `method: bm25`: reject if `field_a` or `field_b` is set (BM25 takes no fields).
   - For all other methods: require `field_a` and `field_b` as today.
   - At most one `method: bm25` entry allowed.
   - Filter size constraints:
     - If both ANN (embedding fields exist) and BM25 enabled: `ann_candidates >= bm25_candidates >= top_n`.
     - If only ANN: `ann_candidates >= top_n`.
     - If only BM25: `bm25_candidates >= top_n`.
   - BM25 without the `bm25` feature flag: reject with clear error ("requires building with --features bm25").

4. Derive `bm25_fields`: collect `(field_a, field_b)` from `method: fuzzy` and `method: embedding` entries. Store in `cfg.bm25_fields`.

5. In `derive_required_fields()`: skip BM25 entries (no fields to add).

**Tests:** Add validation tests:
- `bm25` method accepted with no fields
- `bm25` method rejected if `field_a` or `field_b` is set
- `ann_candidates < bm25_candidates` rejected
- `bm25_candidates < top_n` rejected
- Multiple `bm25` entries rejected
- Existing configs without BM25 unchanged (regression)
- `bm25_fields` derived correctly from fuzzy/embedding entries
- Feature-flag gating: `bm25` method rejected when `bm25` feature not enabled

---

### Step 3: BM25 module — index + scorer

**Files:** New module `src/bm25/mod.rs`, `src/bm25/index.rs`, `src/bm25/scorer.rs`

All code in this module is gated behind `#[cfg(feature = "bm25")]`.

**`src/bm25/mod.rs`:**
```rust
pub mod index;
pub mod scorer;
```

**`src/bm25/index.rs`:** Wraps Tantivy.
- `BM25Index` struct holding a Tantivy `Index`, `IndexReader`, and `IndexWriter`.
- `BM25Index::build(records: &DashMap<String, Record>, fields: &[(String, String)], side: Side) -> Result<Self>`:
  - Creates an in-memory Tantivy schema with a stored `id` field and a text `content` field.
  - For each record, concatenates the text values of all `fields` (using `field_a` for Side::A, `field_b` for Side::B) into a single document.
  - Commits the index.
- `BM25Index::query(&self, text: &str, top_k: usize) -> Vec<(String, f32)>`:
  - Tokenises `text`, builds a BM25 query, returns top-K results as `(id, raw_score)` pairs.
- `BM25Index::score_pair(&self, query_text: &str, candidate_id: &str) -> f32`:
  - Returns the raw BM25 score for a specific candidate against a query.
  - Used during full scoring (Phase 3) for individual pair scoring.
- `BM25Index::upsert(&mut self, id: &str, record: &Record, fields: &[(String, String)], side: Side)`:
  - Delete old document for `id` (if exists), add new document. For live-mode real-time updates.
  - Calls `self.writer.commit()` (or batched commit — see below).
- `BM25Index::remove(&mut self, id: &str)`:
  - Delete document for `id`.

**`src/bm25/scorer.rs`:** Normalisation.
- `normalise_bm25(raw_score: f32, self_score: f32) -> f64`:
  - Returns `(raw_score / self_score).clamp(0.0, 1.0)` as f64.
  - If `self_score <= 0.0`, returns 0.0.
- `compute_self_score(index: &BM25Index, query_text: &str) -> f32`:
  - Queries the index with `query_text` and finds the score of the query document itself (or computes the theoretical max BM25 score for the query tokens).

**`src/lib.rs`:** Add `#[cfg(feature = "bm25")] pub mod bm25;`

**Cargo.toml:** Add `tantivy` as optional dependency, add `bm25` feature flag:
```toml
[features]
bm25 = ["tantivy"]

[dependencies]
tantivy = { version = "0.22", optional = true }
```

**Tests** (in `src/bm25/index.rs` and `src/bm25/scorer.rs`):
- Build index from 10 records, query returns correct top-K
- Self-score normalisation: identical query-document pair → score ~1.0
- Zero overlap → score 0.0
- Empty fields handled gracefully
- Single-token query
- Upsert updates the index correctly
- Remove deletes from the index

---

### Step 4: Wire BM25 into scoring

**Files:** `src/scoring/mod.rs`

Add `method: bm25` branch in `score_pair()`:

```rust
"bm25" => {
    // BM25 score is precomputed and passed in, similar to embedding scores.
    // Key is "bm25".
    precomputed_bm25_score.unwrap_or(0.0)
}
```

The BM25 score is precomputed by the pipeline (like embedding scores) and passed into `score_pair()`. This requires adding a new parameter:

```rust
pub fn score_pair(
    query_record: &Record,
    candidate_record: &Record,
    match_fields: &[MatchField],
    precomputed_emb_scores: Option<&std::collections::HashMap<String, f64>>,
    precomputed_bm25_score: Option<f64>,  // NEW
) -> ScoreResult {
```

For the `FieldScore` output of a BM25 entry: use `field_a: "bm25"`, `field_b: "bm25"` (or empty strings). The `method` is `"bm25"`.

**Update all call sites of `score_pair()`:**
- `src/matching/pipeline.rs` — `score_pool()`: pass BM25 score (computed in step 5)
- `src/session/mod.rs` — not directly; it calls `pipeline::score_pool()`

**Tests:** Add test for `method: bm25` scoring with precomputed value.

---

### Step 5: Wire BM25 into the candidate pipeline

**Files:** `src/matching/pipeline.rs`, `src/matching/candidates.rs`

This is the core wiring step. The pipeline changes depend on which methods are active.

**5a. Update `select_candidates()` signature** (`src/matching/candidates.rs`):

Change `top_n` parameter semantics: it now receives `ann_candidates` (not `top_n`). The caller (`score_pool`) passes the right value.

No other changes needed to `select_candidates()` itself — it already returns the top-K from ANN/flat.

**5b. Update `score_pool()`** (`src/matching/pipeline.rs`):

The function signature gains BM25 index parameters:

```rust
pub fn score_pool(
    query_id: &str,
    query_record: &Record,
    query_side: Side,
    query_combined_vec: &[f32],
    pool_records: &DashMap<String, Record>,
    pool_combined_index: Option<&dyn VectorDB>,
    blocked_ids: &[String],
    config: &Config,
    ann_candidates: usize,      // was: top_n
    bm25_index: Option<&BM25Index>,  // NEW: pool-side BM25 index
    bm25_candidates: usize,     // NEW
    top_n: usize,               // NEW: final output size
) -> Vec<MatchResult> {
```

The pipeline logic becomes:

```
1. ANN candidate selection (if embedding fields exist):
   cands = select_candidates(..., ann_candidates)
   
2. BM25 re-rank/filter (if method: bm25 in match_fields AND bm25_index is Some):
   - For each candidate in cands, compute BM25 score
   - Sort by BM25 score descending
   - Truncate to bm25_candidates
   
   OR if no ANN (no embedding fields):
   - Query bm25_index directly with bm25_candidates
   - Build candidate list from results

3. Full scoring:
   - For each candidate: compute score_pair with all methods including BM25
   - Sort by total score descending
   - Truncate to top_n
```

When neither ANN nor BM25 is active: the existing path (all blocked records scored) is preserved.

**5c. Update all callers of `score_pool()`:**

- `src/batch/engine.rs` — `run_batch()`: pass BM25 index, `ann_candidates`, `bm25_candidates`, `top_n`
- `src/session/mod.rs` — `upsert_record_inner()` and `try_match_inner()`: pass BM25 index, `ann_candidates`, `bm25_candidates`, `top_n`

**Tests:**
- Pipeline with BM25-only (no embeddings): correct number of candidates, correct scoring
- Pipeline with ANN+BM25: ANN produces `ann_candidates`, BM25 narrows to `bm25_candidates`, output is `top_n`
- Pipeline with ANN-only: unchanged behaviour
- Pipeline with neither: unchanged behaviour (blocking → score all)

---

### Step 6: Build BM25 indices at startup (batch mode)

**Files:** `src/batch/engine.rs`, `src/cli/run.rs`

In `run_batch()`, after loading A and B records:

```rust
#[cfg(feature = "bm25")]
let bm25_index_a = if has_bm25 {
    Some(BM25Index::build(&records_a_dashmap, &config.bm25_fields, Side::A)?)
} else {
    None
};

#[cfg(feature = "bm25")]
let bm25_index_b = if has_bm25 {
    Some(BM25Index::build(&b_records, &config.bm25_fields, Side::B)?)
} else {
    None
};
```

Where `has_bm25 = config.match_fields.iter().any(|mf| mf.method == "bm25")`.

Pass `bm25_index_a.as_ref()` to `score_pool()` as the pool-side BM25 index (since we're scoring B records against A, we query the A-side index).

**Note:** The batch engine currently only scores B→A. The BM25 index for B is built but not queried in batch mode. It would be used if we ever scored A→B.

**Tests:** End-to-end batch test with BM25 config (requires test data).

---

### Step 7: Integrate BM25 in live mode

**Files:** `src/session/mod.rs`, `src/state/live.rs`

**7a. Add BM25 indices to `LiveSideState`** (`src/state/live.rs`):

```rust
#[cfg(feature = "bm25")]
pub bm25_index: Option<RwLock<BM25Index>>,
```

Both A-side and B-side get their own BM25 index (symmetry principle).

**7b. Build BM25 indices during live startup** (`src/cli/serve.rs` or `src/state/live.rs`):

After loading initial A and B records, build the BM25 indices the same way as batch.

**7c. Update on upsert** (`src/session/mod.rs`):

In `upsert_record()`, after updating the blocking index and vector index:

```rust
#[cfg(feature = "bm25")]
if let Some(ref bm25_idx) = this_side.bm25_index {
    let mut idx = bm25_idx.write().unwrap_or_else(|e| e.into_inner());
    idx.upsert(&id, &record, &config.bm25_fields, side);
}
```

**7d. Remove from BM25 index** (`src/session/mod.rs`):

In `remove_record()`:

```rust
#[cfg(feature = "bm25")]
if let Some(ref bm25_idx) = this_side.bm25_index {
    let mut idx = bm25_idx.write().unwrap_or_else(|e| e.into_inner());
    idx.remove(id);
}
```

**7e. Pass BM25 index to `score_pool()`** in `upsert_record_inner()` and `try_match_inner()`:

The opposite side's BM25 index is passed as the pool-side index.

**Tests:**
- Upsert record, verify BM25 index contains it
- Remove record, verify BM25 index no longer contains it
- Match query with BM25 scoring returns expected scores
- Batch upsert updates BM25 index for all records

---

### Step 8: Feature-flag hygiene

**Files:** All files modified above

Ensure every BM25-related code path is gated behind `#[cfg(feature = "bm25")]`:
- `src/lib.rs`: `pub mod bm25` gated
- `src/config/loader.rs`: `"bm25"` in `VALID_METHODS` only when feature enabled; validation rejects `method: bm25` without feature
- `src/scoring/mod.rs`: `"bm25"` match arm gated
- `src/matching/pipeline.rs`: BM25 re-rank logic gated
- `src/batch/engine.rs`: BM25 index building gated
- `src/session/mod.rs`: BM25 index upsert/remove gated
- `src/state/live.rs`: BM25 index field gated

**Verification:**
- `cargo build` (without `bm25` feature): compiles clean
- `cargo build --features bm25`: compiles clean
- `cargo test` (without `bm25` feature): all existing tests pass
- `cargo test --all-features`: all tests pass including BM25 tests

---

### Step 9: Regression verification

**Files:** No new code — test runs only

1. Run all existing test configs without BM25 → identical output
2. Run `cargo test` → all pass
3. Run `cargo test --all-features` → all pass
4. Run `cargo fmt -- --check` → clean
5. Run `cargo clippy --all-features` → clean
6. Verify the eight pipeline modes from the pipeline table all work:
   - Blocking+ANN+BM25, Blocking+ANN, Blocking+BM25, Blocking only
   - ANN+BM25, ANN only, BM25 only, Neither

---

## Dependency Graph

```
Step 1 (schema) ──→ Step 2 (validation) ──→ Step 3 (BM25 module) ──→ Step 4 (scoring)
                                                                          ↓
                                                     Step 5 (pipeline wiring)
                                                          ↓            ↓
                                               Step 6 (batch)    Step 7 (live)
                                                          ↓            ↓
                                                     Step 8 (feature flags)
                                                          ↓
                                                     Step 9 (regression)
```

## Key Decision: `MatchField.field_a/field_b` as Empty Strings for BM25

We chose to keep `field_a` and `field_b` as `String` (not `Option<String>`) to avoid an invasive migration across every call site. For `method: bm25`, they are empty strings. Validation rejects empty fields for all other methods. The `#[serde(default)]` attribute on both fields allows BM25 YAML entries to omit them entirely. No existing code needs changes — the only new validation is ensuring BM25 entries have empty fields and non-BM25 entries have non-empty fields.

## Estimated Effort

| Step | Effort | Notes |
|---|---|---|
| 1. Config schema | Small | Add two fields to Config, add serde defaults to MatchField |
| 2. Config validation | Small | Straightforward validation rules |
| 3. BM25 module | Medium | New module, Tantivy integration, self-score normalisation |
| 4. Scoring wiring | Small | One new match arm + parameter |
| 5. Pipeline wiring | Medium | Core logic change — sequential ANN→BM25 flow |
| 6. Batch integration | Small | Build index, pass to pipeline |
| 7. Live integration | Medium | Upsert/remove hooks, RwLock, symmetry |
| 8. Feature flags | Small | Mechanical gating |
| 9. Regression | Small | Test runs only |
