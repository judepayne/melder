# melder — Build Plan

Detailed implementation plan for the melder Rust rewrite. Each task is scoped
to 1-3 hours. Tasks within a phase are sequential unless marked `[parallel]`.
Every task has a verification step — do not move on until it passes.

**Reference paths:**
- Go project: `/Users/jude/Library/CloudStorage/Dropbox/Projects/match/`
- Rust project: `/Users/jude/Library/CloudStorage/Dropbox/Projects/melder/`

**Key design doc:** `DESIGN.md` in this directory (1614 lines, 21 sections).

---

## Key Invariants (check at every phase boundary)

1. **Symmetry (live mode only):** Every feature for A exists identically for B.
   Batch mode is asymmetrical — A is the reference pool, B records are queries.
2. **Config compatibility:** Go YAML configs work unchanged (ignoring `sidecar`).
3. **API compatibility:** Same endpoints, same request/response JSON shapes.
4. **Scoring compatibility:** Same classification logic, same weight handling.
   Small float differences in embedding scores are acceptable.
5. **No `Ref` across `.await`:** DashMap guards are never held across async
   boundaries.
6. **Tests pass:** `cargo test` passes at every commit.

---

## Phase 0: Project Scaffold + Test Data

**Goal:** Cargo project compiles, test data is in place.

### 0.1 — Initialize Cargo project

- `cargo init` in the melder directory
- Set up `Cargo.toml` with initial dependencies:
  `serde`, `serde_yaml`, `serde_json`, `clap` (derive), `csv`, `anyhow`,
  `thiserror`
- Create module skeleton with empty files:
  ```
  src/lib.rs
  src/error.rs
  src/models.rs
  src/config/mod.rs
  src/config/schema.rs
  src/config/loader.rs
  ```
- `cargo build` succeeds with empty modules

**Verify:** `cargo build` and `cargo test` both pass (trivially).

### 0.2 — Copy test data and configs from Go project

- Copy from `match/testdata/` to `melder/testdata/`:
  - `dataset_a_1000.csv`, `dataset_b_1000.csv`
  - `dataset_a_10000.csv`, `dataset_b_10000.csv`
  - `dataset_a_10000.jsonl`, `dataset_b_10000.jsonl`
  - `dataset_a_10000.parquet`, `dataset_b_10000.parquet`
  - `ground_truth_crossmap.csv`
  - `generate.py`
- Copy YAML configs to `melder/testdata/configs/`:
  - `match/bench/bench_live.yaml`
  - `match/bench/bench1000x1000.yaml`
  - `match/bench/bench10000x10000.yaml`
  - `match/jobs/counterparty_recon.yaml` (has output_mapping + sidecar)
- Copy bench scripts to `melder/bench/`:
  - `match/bench/live_stress_test.py`
  - `match/bench/live_concurrent_test.py`
  - `match/bench/smoke_test.py`
  - `match/bench/crossmap_live.csv`
- Adjust any hardcoded paths in copied YAML configs (dataset paths,
  cache paths, output paths) so they are relative to `melder/`

**Verify:** All files exist at expected paths. `python3 testdata/generate.py --help` runs.

---

## Phase 1: Config + Models + Scoring

**Goal:** Config validates. Record pairs can be scored. `meld validate` works.

### 1.1 — Error types (`src/error.rs`)

- Implement `MatchrError` top-level enum with `Config`, `Data` variants
  and `#[from]` conversions (per DESIGN.md §18)
- Implement `ConfigError` enum: `MissingField`, `InvalidValue`, `WeightSum`,
  `Parse`, `Io`
- Implement `DataError` enum: `NotFound`, `MissingIdField`, `DuplicateId`,
  `Csv`, `Io`
- Stub the remaining error types (`EncoderError`, `IndexError`,
  `CrossMapError`, `SessionError`) as empty enums — they compile but have
  no variants yet

**Verify:** `cargo build` passes.

### 1.2 — Core models (`src/models.rs`)

- `pub type Record = HashMap<String, String>;`
- `Side` enum: `A`, `B` with `opposite()`, derive `Serialize`/`Deserialize`,
  `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`
- `FieldScore` struct: `field_a`, `field_b`, `method`, `score`, `weight`,
  `contribution()` method
- `MatchResult` struct: `query_id`, `matched_id`, `query_side: Side`,
  `score`, `field_scores`, `classification`, `matched_record`, `from_crossmap`
- `Classification` enum: `Auto`, `Review`, `NoMatch` with `from_score()`
  and `as_str()`

**Verify:** Unit tests:
- `Classification::from_score(0.9, 0.85, 0.60)` → `Auto`
- `Classification::from_score(0.7, 0.85, 0.60)` → `Review`
- `Classification::from_score(0.3, 0.85, 0.60)` → `NoMatch`
- `Side::A.opposite()` → `Side::B` and vice versa
- `FieldScore { score: 0.8, weight: 0.5, .. }.contribution()` → `0.4`

### 1.3 — Config schema (`src/config/schema.rs`)

- Implement all config structs per DESIGN.md §2:
  `Config`, `JobConfig`, `DatasetsConfig`, `DatasetConfig`, `CrossMapConfig`,
  `EmbeddingsConfig`, `BlockingConfig`, `BlockingFieldPair`, `MatchField`,
  `CandidatesConfig`, `FieldMapping`, `ThresholdsConfig`, `OutputConfig`,
  `LiveConfig`
- `sidecar: Option<serde_yaml::Value>` for backward compat
- `LiveConfig` includes `encoder_pool_size: Option<usize>` and
  `crossmap_flush_secs: Option<u64>`
- `#[serde(skip)]` on `required_fields_a` / `required_fields_b`
- Default functions: `default_backend() → "local"`,
  `default_operator() → "and"`

**Verify:** `serde_yaml::from_str::<Config>(yaml)` successfully deserializes
`testdata/configs/bench_live.yaml`. All fields populated, sidecar section
accepted and ignored.

### 1.4 — Config loader: parsing + defaults (`src/config/loader.rs`)

- Implement `load_config(path: &Path) -> Result<Config, ConfigError>`:
  read file, deserialize YAML
- Implement `normalise_blocking(config)`: promote legacy single
  `field_a`/`field_b` into `fields` vec, default operator to `"and"`
- Apply defaults after parse:
  - `candidates.n` → 10 if absent/zero
  - `candidates.scorer` → `"wratio"` if empty
  - `live.top_n` → 5 if absent/zero
  - `workers` → 4 if absent/zero
  - `live.encoder_pool_size` → 1 if absent/zero
  - `live.crossmap_flush_secs` → 5 if absent/zero
  - `cross_map.backend` → `"local"` if empty

**Verify:** Load `bench_live.yaml`, assert `workers == Some(4)` or default
applied. Load a config with legacy blocking syntax (create a test fixture),
verify fields vec populated and operator is `"and"`.

### 1.5 — Config validation: fields, formats, cross_map, embeddings, match_fields

Port Go `loader.go` validation rules 1-19 (reference: Go `loader_test.go`
13 test cases):

1. `job.name` required (non-empty)
2. `datasets.a.path` required
3. `datasets.a.id_field` required
4. `datasets.a.format`: infer from extension if empty
   (`.csv`/`.tsv`→csv, `.parquet`→parquet, `.jsonl`/`.ndjson`→jsonl;
   reject `.json` with helpful message, reject `.xlsx`/`.xls`; error on
   unknown)
5. Same rules for `datasets.b.*`
6. `cross_map.backend`: must be `"local"` or `"redis"`
7. `cross_map.path` required if backend is `"local"`
8. `cross_map.redis_url` required if backend is `"redis"`
9. `cross_map.a_id_field` required
10. `cross_map.b_id_field` required
11. `embeddings.model` required
12. `embeddings.a_cache` required
13. `embeddings.index_cache` required
14. At least one `match_fields` entry
15. Each: `field_a`, `field_b` non-empty
16. Each: `method` must be `"exact"`, `"fuzzy"`, or `"embedding"`
17. Each: if `method` is `"fuzzy"` and `scorer` is non-empty, scorer must
    be `"wratio"`, `"partial_ratio"`, or `"token_sort"`
18. Each: `weight > 0.0`
19. Fuzzy scorer defaults: if method is `"fuzzy"` and scorer is empty,
    default to `"wratio"`

**Verify:** Port all 13 test cases from Go's `loader_test.go`:
- Valid config loads successfully
- Missing `job.name` → error mentioning `"job.name"`
- Invalid method → error mentioning `"match_fields[0].method"`
- Invalid backend → error mentioning `"cross_map.backend"`
- Format inference from `.csv`, `.tsv`, `.parquet`, `.jsonl`, `.ndjson`
  → correct values; `.json`, `.xlsx`, `.dat` → error

### 1.6 — Config validation: weights, thresholds, blocking, output, derived fields

Continue validation rules:

20. Weights sum to 1.0 (tolerance 0.001)
21. `thresholds.auto_match` in (0.0, 1.0]
22. `thresholds.review_floor` in [0.0, 1.0)
23. `auto_match > review_floor`
24. `output.results_path` required
25. `output.review_path` required
26. `output.unmatched_path` required
27. If blocking enabled: at least one field pair, each field_a/field_b
    non-empty, operator must be `"and"` or `"or"` (case-insensitive)
28. `live.encoder_pool_size` >= 1
29. `live.crossmap_flush_secs` >= 1

Post-validation:
30. `derive_required_fields()`: collect all field names referenced in
    match_fields (field_a → A side, field_b → B side), output_mapping,
    blocking, dataset id_fields → populate `required_fields_a` /
    `required_fields_b`

**Verify:** Test cases:
- Weights summing to 0.95 → error mentioning `"match_fields"`
- `auto_match < review_floor` → error mentioning `"thresholds"`
- Blocking with invalid operator → error mentioning `"blocking.operator"`
- Derived fields for `bench_live.yaml` include `entity_id`, `legal_name`,
  `short_name`, `country_code`, `lei` on A side

### 1.7 — CSV data loader (`src/data/csv.rs`, `src/data/mod.rs`)

- Implement `load_csv(path, id_field, required_fields) -> Result<(HashMap<String, Record>, Vec<String>), DataError>`
- Returns: records map keyed by ID, plus sorted ID list
- Validations:
  - File exists (DataError::NotFound)
  - `id_field` exists in CSV headers (DataError::MissingIdField)
  - No duplicate IDs (DataError::DuplicateId)
  - All `required_fields` present in headers (log warning for missing
    optional fields, don't error)
- Handle: trim whitespace from headers, UTF-8 encoding

**Verify:** Load `testdata/dataset_a_1000.csv` with `id_field = "entity_id"`.
Assert 1000 records loaded. Assert IDs are sorted. Assert first record has
fields: `entity_id`, `legal_name`, `short_name`, `country_code`, `lei`.
Test duplicate ID detection with a synthetic CSV.

### 1.8 — Exact scorer (`src/scoring/exact.rs`, `src/scoring/mod.rs`)

- Port Go's `ExactScorer.Score()` exactly (24 lines):
  - Trim both strings
  - Both empty → 0.0
  - Case-insensitive equal → 1.0
  - Otherwise → 0.0

**Verify:** Table-driven tests:
- `("Foo", "foo")` → 1.0
- `(" Bar ", "bar")` → 1.0
- `("", "")` → 0.0
- `("a", "b")` → 0.0
- `("GB", "gb")` → 1.0
- `("  ", "  ")` → 0.0 (both empty after trim)
- `("Café", "café")` → 1.0 (Unicode case folding)

### 1.9 — Generate fuzzy golden test data

- Write `testdata/generate_fuzzy_golden.py`:
  ```python
  import json
  from rapidfuzz import fuzz
  pairs = [
      ("hello", "hello"),
      ("Hello", "hello"),
      ("", ""),
      ("abc", ""),
      # ... 50+ pairs covering:
      # identical, case diff, whitespace, partial overlap,
      # abbreviations, empty strings, Unicode, single char,
      # very long strings, token reordering, substrings
  ]
  results = []
  for a, b in pairs:
      results.append({
          "a": a, "b": b,
          "ratio": fuzz.ratio(a, b) / 100.0,
          "partial_ratio": fuzz.partial_ratio(a, b) / 100.0,
          "token_sort_ratio": fuzz.token_sort_ratio(a, b) / 100.0,
          "wratio": fuzz.WRatio(a, b) / 100.0,
      })
  with open("testdata/fuzzy_golden.json", "w") as f:
      json.dump(results, f, indent=2)
  ```
- Run it: `python3 testdata/generate_fuzzy_golden.py`

**Verify:** `fuzzy_golden.json` exists with 50+ entries, all scores in [0, 1].

### 1.10 — Fuzzy scorers (`src/fuzzy/`)

- Add `rapidfuzz` crate to `Cargo.toml`
- `src/fuzzy/mod.rs` — re-export: `ratio`, `partial_ratio`,
  `token_sort_ratio`, `wratio`
- `src/fuzzy/ratio.rs` — wrapper around `rapidfuzz::fuzz::ratio`:
  lowercase + trim inputs, call `fuzz::ratio(a.chars(), b.chars())`
- `src/fuzzy/partial_ratio.rs` — our implementation:
  Find best-aligned window of shorter string within longer string,
  compute ratio on that window. Reference: Python `rapidfuzz` source.
  ~30 lines of logic.
- `src/fuzzy/token_sort.rs` — our implementation:
  Split on whitespace, sort tokens alphabetically, rejoin with single
  space, compute ratio on the sorted strings. ~10 lines of logic.
- `src/fuzzy/wratio.rs` — our implementation:
  Try ratio, token_sort_ratio, partial_ratio; return best weighted
  result following Python WRatio semantics. ~20 lines.
- All functions: lowercase + trim inputs before processing, return 0.0-1.0

**Verify:** Load `testdata/fuzzy_golden.json`. Run each scorer on all pairs.
Assert each score within ±0.01 of expected Python value. (Tighten tolerance
to ±0.005 once implementation is stable.)

### 1.11 — CLI skeleton + `validate` command (`src/main.rs`)

- Set up `clap` derive CLI with subcommands:
  ```
  melder validate --config <path>
  melder run      --config <path> [--dry-run] [--verbose] [--limit N]
  melder serve    --config <path> [--port N] [--socket <path>]
  melder tune     --config <path> [--verbose]
  melder cache build  --config <path>
  melder cache status --config <path>
  melder cache clear  --config <path> [--index-only]
  melder review list   --config <path>
  melder review import --config <path> --file <path>
  melder crossmap stats  --config <path>
  melder crossmap export --config <path> --out <path>
  melder crossmap import --config <path> --file <path>
  ```
- All subcommands require `--config` (mandatory)
- Implement `validate`: load_config, print success message with job name
  and summary, or print error with context
- All other subcommands: print "not yet implemented" and exit

**Verify:**
- `cargo run -- validate --config testdata/configs/bench_live.yaml`
  prints success with job name
- `cargo run -- validate --config /dev/null` prints parse error
- `cargo run -- run --config testdata/configs/bench_live.yaml` prints
  "not yet implemented"
- `cargo run -- --help` shows all subcommands

---

## Phase 2: Encoder + Vector Index + Cache

**Goal:** `meld cache build` works. Embeddings generated and stored.

### 2.1 — Encoder pool (`src/encoder/pool.rs`, `src/encoder/mod.rs`)

- Add `fastembed` to `Cargo.toml`
- Populate `EncoderError` enum variants: `ModelNotFound`, `Inference`,
  `PoolExhausted`
- Implement `EncoderPool::new(model_name: &str, pool_size: usize) -> Result<Self>`:
  - Map model name string to `fastembed::EmbeddingModel` enum
    (`"all-MiniLM-L6-v2"` → `EmbeddingModel::AllMiniLML6V2`)
  - Create `pool_size` instances of `Mutex<TextEmbedding>`
  - Each instance: `TextEmbedding::try_new(InitOptions::new(model).with_show_download_progress(true))`
  - First run may download model (~90MB) — log progress
- Implement `EncoderPool::encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>`:
  - Round-robin `try_lock` across pool slots
  - If all busy, `.lock().await` on slot 0
  - Call `guard.embed(texts, None)?` (fastembed's batch API)
  - Wrap fastembed's `anyhow::Error` in `EncoderError::Inference`
- Implement `EncoderPool::encode_blocking(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>`:
  - Sync version for batch mode (uses `std::sync::Mutex` or
    `tokio::runtime::Handle::block_on`)
- Implement `EncoderPool::dim(&self) -> usize`: returns 384 for MiniLM

**Verify:** Create pool with `"all-MiniLM-L6-v2"`, pool_size=1. Encode
`["hello world", "test sentence"]`. Returns two 384-dim vectors. Dot product
of encoding "hello world" with itself ≈ 1.0 (±0.001). Two different
sentences have dot product < 1.0.

### 2.2 — Embedding scorer (`src/scoring/embedding.rs`)

- Implement `cosine_similarity(a: &[f32], b: &[f32]) -> f64`:
  dot product of two vectors, clamped to [0.0, 1.0]
  (fastembed returns L2-normalized vectors, so dot product = cosine)
- Implement `l2_normalize(v: &mut [f32])` (in-place): divide each element
  by the L2 norm. Needed for vectors loaded from cache or user-provided.
- Implement `dot_product(a: &[f32], b: &[f32]) -> f64`: sum of a[i]*b[i].
  Use `f32` arithmetic internally, return as `f64`.

**Verify:**
- `cosine_similarity([1,0], [1,0])` → 1.0
- `cosine_similarity([1,0], [0,1])` → 0.0
- `cosine_similarity([1,0], [0.707, 0.707])` ≈ 0.707
- `l2_normalize([3.0, 4.0])` → `[0.6, 0.8]`
- Encode "hello" twice with fastembed, cosine of the two → 1.0 (±0.001)

### 2.3 — Flat vector index (`src/index/flat.rs`, `src/index/mod.rs`)

- Implement `VecIndex` struct:
  ```rust
  vectors: Vec<f32>,       // N*D flat matrix, row-major
  dim: usize,
  ids: Vec<String>,        // ids[i] owns vectors[i*dim..(i+1)*dim]
  id_to_pos: HashMap<String, usize>,
  ```
- `VecIndex::new(dim: usize) -> Self` — empty index
- `VecIndex::upsert(&mut self, id: &str, vec: &[f32])`:
  - If id exists: overwrite vector in-place
  - If new: append vector + id, insert into id_to_pos
  - Panic if `vec.len() != self.dim`
- `VecIndex::remove(&mut self, id: &str) -> bool`:
  - Swap-remove: copy last row into removed row's position
  - Update `id_to_pos` for the swapped element
  - Truncate vectors and ids
  - Return true if id was found
- `VecIndex::search(&self, query: &[f32], k: usize) -> Vec<(String, f32)>`:
  - Compute dot product of query with every row
  - Partial sort (select top-K), return sorted descending
  - LLVM auto-vectorizes the dot product inner loop
- `VecIndex::search_filtered(&self, query: &[f32], k: usize, allowed: &HashSet<String>) -> Vec<(String, f32)>`:
  - Same as search but skip rows whose id is not in `allowed`
  - Complexity: O(|allowed| * D) scan if iterating allowed and looking
    up positions; or O(N) scan skipping disallowed — benchmark to decide
- `VecIndex::len()`, `VecIndex::contains(&self, id: &str) -> bool`
- `VecIndex::get(&self, id: &str) -> Option<&[f32]>` — return vector slice

**Verify:**
- Insert 100 random 384-dim unit vectors with string IDs
- Search top-5: verify ordering matches manual dot-product sort
- Upsert (replace) an existing ID: search reflects new vector
- Remove an ID: search no longer returns it, len decreased
- Swap-remove correctness: remove middle element, verify remaining
  elements all have correct id ↔ vector mapping
- search_filtered with 50 allowed IDs: only returns from that set

### 2.4 — Index cache serialization (`src/index/cache.rs`)

- Binary format:
  ```
  [4 bytes] N (u32 little-endian) — number of vectors
  [4 bytes] D (u32 little-endian) — dimension
  [N*D*4 bytes] vectors (f32 little-endian, row-major)
  [variable] N newline-separated ID strings (UTF-8)
  ```
- Implement `save_index(path: &Path, index: &VecIndex) -> Result<()>`
- Implement `load_index(path: &Path) -> Result<VecIndex>`
- Implement `save_embeddings_npy(path: &Path, index: &VecIndex) -> Result<()>`:
  write `.npy` format (numpy-compatible) for the vector matrix
- Staleness check: `is_cache_stale(path, record_count, id_hash) -> bool`
  - Compare stored (N, id_hash) header against current data
  - Return true if mismatch or file doesn't exist

**Verify:** Build VecIndex with 100 entries. Save to temp file. Load back.
Assert all vectors and IDs are identical (bitwise for f32). Test staleness:
add a record, check reports stale. Delete cache, check reports stale.

### 2.5 — State loader (`src/state/state.rs`, `src/state/mod.rs`)

- Implement `LoadOptions { load_b: bool, batch_mode: bool }`
- Implement `MatchState` struct:
  ```rust
  pub config: Config,
  pub records_a: HashMap<String, Record>,
  pub ids_a: Vec<String>,        // sorted
  pub index_a: VecIndex,
  pub records_b: Option<HashMap<String, Record>>,  // None in batch mode
  pub ids_b: Option<Vec<String>>,
  pub index_b: Option<VecIndex>,  // None in batch mode
  ```
- Implement `load_state(config, opts) -> Result<MatchState>`:
  1. Load dataset A via CSV loader → records_a + ids_a
  2. If `opts.load_b`: load dataset B → records_b + ids_b
  3. A-side cache: if `.index` exists and not stale → load_index;
     else → encode all A records (batch 256), save cache, build VecIndex
  4. B-side cache (only if `opts.load_b` and b_cache configured):
     same logic as A
  5. Progress logging: INFO every 1000 records during encoding
- Implement `primary_embedding_text(record: &Record, config: &Config) -> String`:
  - Concatenate values of all embedding-method match fields (A-side fields
    if record is A, B-side if B), space-separated
  - Fallback to first fuzzy field, then ID field

**Verify:** Load state with `bench1000x1000.yaml` config and 1K CSVs:
- Cold cache: encodes 1000 A records (~3s), saves `.index` file
- Warm cache: loads from `.index` in <100ms
- `index_a.len()` = 1000
- `records_b` is None (batch mode, load_b=false)
- Load again with `load_b=true`: both sides populated

### 2.6 — `cache` CLI commands

- Wire up `cache build --config <path>`:
  Load config. Load datasets (A always; B if b_cache configured).
  Build caches. Print summary: records encoded, cache file sizes, time.
- Wire up `cache status --config <path>`:
  Check each cache file: exists? stale? Print table with path, status,
  size, record count.
- Wire up `cache clear --config <path> [--index-only]`:
  Delete `.index` files. If not `--index-only`, also delete `.npy` files.
  Print what was deleted.

**Verify:**
- `cargo run -- cache build --config testdata/configs/bench1000x1000.yaml`
  creates cache files in the configured paths
- `cargo run -- cache status --config testdata/configs/bench1000x1000.yaml`
  shows "fresh" for all caches
- `cargo run -- cache clear --config testdata/configs/bench1000x1000.yaml`
  removes caches; subsequent status shows "missing"

---

## Phase 3: Matching Engine + Batch Mode

**Goal:** `meld run` produces output matching Go's `match run`.

**Batch semantics reminder:** Batch mode is asymmetrical. A is the reference
pool (pre-indexed). B records are the queries. For each unmatched B record,
find the best matching A record. There is no reverse pass.

### 3.1 — Composite scorer (`src/scoring/composite.rs`)

- Implement `CompositeScorer` struct holding `&Config`
- Implement `score_pair(a_record, b_record, a_vec: Option<&[f32]>, b_vec: Option<&[f32]>) -> Result<(f64, Vec<FieldScore>)>`:
  - For each match_field in config:
    - `method == "exact"`: `exact_score(a_val, b_val)`
    - `method == "fuzzy"`: dispatch to `wratio`/`partial_ratio`/`token_sort`
      based on `scorer` field (default `wratio`)
    - `method == "embedding"`: if vecs provided, `cosine_similarity(a_vec, b_vec)`;
      else skip field (exclude its weight from total)
  - Compute weighted sum + total weight
  - If `total_weight > 0 && total_weight != 1.0`: normalize
    `score = weighted_sum / total_weight`
  - Return (score, field_scores vec)
- Note: field_a values come from A record, field_b values from B record.
  Direction is NOT swapped — the config defines which field belongs to which
  side.

**Verify:** Create scorer with `bench_live.yaml` config (4 fields summing
to 1.0):
- Pair with identical country_code: exact field contributes `1.0 * 0.20`
- Pair with different LEI: exact field contributes `0.0 * 0.05`
- Mock embedding vecs with known cosine: verify contribution = `cosine * 0.55`
- Skip embedding field (no vecs): verify remaining weights re-normalize

### 3.2 — Blocking filter — linear scan for batch (`src/matching/blocking.rs`)

- Implement `apply_blocking(query_record, pool_records, blocking_config, query_side) -> Vec<String>`:
  - For each pool record, check if it passes the blocking filter
  - AND: all field pairs must match (case-insensitive, trimmed)
  - OR: any field pair match passes
  - Direction: if `query_side == B`, query uses `field_b`, pool uses `field_a`
  - Missing query value for a field: skip that constraint (all pass on
    that field)
  - Returns list of passing pool record IDs
- This is the batch-mode linear scan. Live mode uses BlockingIndex (3.3).

**Verify:** 10 synthetic A records, 3 have `country_code = "GB"`:
- B query with `domicile = "GB"`, AND mode → returns 3 IDs
- B query with `domicile = ""` (missing) → skips constraint, returns all 10
- OR mode with two fields, one matching → returns records matching either

### 3.3 — Blocking index for live mode (`src/matching/blocking.rs`)

- Implement `BlockingIndex` struct (per DESIGN.md §7):
  - `operator: BlockingOperator` (And / Or)
  - `and_index: HashMap<Vec<String>, HashSet<String>>` — composite key
  - `or_indices: Vec<HashMap<String, HashSet<String>>>` — per-field
- `BlockingIndex::new(operator, num_fields) -> Self`
- `insert(&mut self, id: &str, record: &Record, fields: &[BlockingFieldPair], side: Side)`:
  - Extract field values (lowercase, trimmed) for the appropriate side
  - AND: build composite key, insert id into that bucket
  - OR: insert id into each per-field bucket
- `remove(&mut self, id: &str, record: &Record, fields: &[BlockingFieldPair], side: Side)`:
  - Reverse of insert
- `query(&self, query_record: &Record, fields: &[BlockingFieldPair], query_side: Side) -> HashSet<String>`:
  - AND: build composite key from query, return the bucket
  - OR: union of all per-field bucket lookups
  - Missing query value: AND → return all (no filtering possible);
    OR → skip that field's bucket

**Verify:**
- Build index with 100 A records. Query with known blocking key → correct
  IDs. Test AND vs OR. Test insert/remove round-trip. Test missing query
  value semantics.

### 3.4 — CrossMap (`src/crossmap/local.rs`, `src/crossmap/mod.rs`)

- Implement `CrossMap` struct:
  ```rust
  a_to_b: HashMap<String, String>,
  b_to_a: HashMap<String, String>,
  count: usize,
  ```
- `CrossMap::new() -> Self`
- `add(&mut self, a_id: &str, b_id: &str)` — insert both directions
- `remove(&mut self, a_id: &str, b_id: &str)` — remove both directions
- `get_b(&self, a_id: &str) -> Option<&str>` — A→B lookup
- `get_a(&self, b_id: &str) -> Option<&str>` — B→A lookup
- `has_a(&self, a_id: &str) -> bool`, `has_b(&self, b_id: &str) -> bool`
- `load(path, a_field, b_field) -> Result<CrossMap>` — read CSV
- `save(&self, path, a_field, b_field) -> Result<()>` — atomic write
  (write to temp file, then `std::fs::rename`)
- `len(&self) -> usize`
- `iter(&self) -> impl Iterator<Item = (&str, &str)>`
- Populate `CrossMapError` variants in `error.rs`

**Verify:** Add 5 pairs, bidirectional lookup correct. Remove one, gone
from both sides. Save to temp, load back, identical state. Load
`bench/crossmap_live.csv` (may be empty header-only — handle that).

### 3.5 — Candidate generation (`src/matching/candidates.rs`)

- `Candidate` struct: `id: String, record: Record, embedding_score: Option<f64>`
- Implement `generate_candidates_batch(query_record, query_vec, index_a, pool_records, blocking_config, crossmap, config) -> Vec<Candidate>`:
  **Batch mode (B queries against A pool):**
  1. If CrossMap has this B record → return empty (already matched)
  2. Vector search: `index_a.search(query_vec, candidates_n)` → top-K A IDs
  3. Post-filter by blocking: keep only IDs that pass `apply_blocking`
  4. If fuzzy candidates enabled and fewer than K after filtering:
     supplement with fuzzy search on remaining A records (wratio on
     primary text field, take best remaining)
  5. Build Candidate structs with records from pool

- Implement `generate_candidates_live(query_record, query_vec, target_side_state, blocking_index, unmatched_set, config) -> Vec<Candidate>`:
  **Live mode (either direction):**
  1. Build allowed_ids = unmatched ∩ blocking_index.query()
  2. `search_filtered(query_vec, K, &allowed_ids)`
  3. Fuzzy fallback if < K results
  4. Build Candidate structs

**Verify:** With 100 indexed A records, 50 unmatched, 30 passing blocking:
- Batch mode: search returns top-K from full A index, filtering reduces
- Live mode: `search_filtered` only touches ~15 vectors (unmatched ∩ blocking)
- Fuzzy fallback triggers when vector results < K

### 3.6 — Match engine (`src/matching/engine.rs`)

- Implement `match_top_n(query_record, query_side, query_vec, state, config) -> Result<Vec<MatchResult>>`:
  1. Determine target side = `query_side.opposite()`
  2. Check CrossMap for existing mapping:
     - If found → return single result with `from_crossmap = true`,
       include matched record, score = 1.0, classification = Auto
  3. Generate candidates (batch or live variant based on state type)
  4. For each candidate: score via composite scorer
  5. Classify each: `Classification::from_score(score, auto_match, review_floor)`
  6. Sort by score descending
  7. Cap at `config.live.top_n` (live mode) or return all (batch mode)
  8. Apply output_mapping if configured (rename fields in matched_record)
  9. Populate `MatchResult` with `query_id`, `matched_id`, `query_side`

**Verify:** Load state for 1K data. Match a B record against A pool:
- Top result has a reasonable score (> 0.5 for a synthetic pair with noise)
- `query_side` = B, `query_id` = B record's ID, `matched_id` = A record's ID
- Classification is one of Auto/Review/NoMatch
- If B record is in CrossMap: returns single `from_crossmap = true` result

### 3.7 — Batch engine (`src/batch/engine.rs`)

- Implement `run_batch(config, state) -> Result<BatchResult>`:
  1. Load B dataset as flat record list (NOT into MatchState — batch mode
     does not pre-index B)
  2. Load CrossMap
  3. For each B record (optionally limited by `--limit`):
     a. Skip if already in CrossMap
     b. Compute `primary_embedding_text` for this B record
     c. Encode B record's text via encoder pool (on-the-fly, not pre-cached)
     d. Generate candidates from A pool (vector search + blocking filter)
     e. Score candidates via composite scorer
     f. Classify top result
     g. If auto_match: add to CrossMap
  4. Partition results: auto_match, review, no_match
  5. Return `BatchResult { matched, review, unmatched, stats }`
- Use Rayon `par_iter` for step 3 parallelism across B records
- Progress: INFO log every 100 B records with count/total and ETA
- Batch encoding optimisation: collect unique (A-value, B-value) text
  pairs from candidate scoring, encode in single bulk call per batch
  of B records — reuse A-side embeddings across multiple B queries

**Verify:** Run batch on 1K×1K data with `bench1000x1000.yaml`:
- Output has three categories (auto, review, no_match)
- At least some auto_matches (synthetic data has 70% matched pairs)
- At least some no_matches (10% unmatched in synthetic data)
- CrossMap updated with auto-matched pairs
- Processing time < 60s for 1K B records (with warm A-side cache)

### 3.8 — Batch output writer (`src/data/csv.rs` extension)

- `write_results_csv(path, results: &[MatchResult], config) -> Result<()>`:
  Headers: `a_id, b_id, score, classification, [field_score columns]`
- `write_review_csv(path, results: &[MatchResult], config) -> Result<()>`:
  Same format, only review-classified results
- `write_unmatched_csv(path, records: &[Record], id_field: &str) -> Result<()>`:
  All fields from unmatched B records
- Match Go output format: field order, header names, decimal precision

**Verify:** Write 10 synthetic results, read back, verify CSV is valid and
columns match expected headers.

### 3.9 — `run` CLI command

- Wire up `run --config <path> [--dry-run] [--verbose] [--limit N]`:
  1. Load and validate config
  2. Load state (batch mode: `LoadOptions { load_b: false, batch_mode: true }`)
  3. Build A-side cache if needed
  4. Initialize encoder pool (for on-the-fly B encoding)
  5. If `--dry-run`: report what would be processed, exit
  6. If `--limit N`: only process first N B records
  7. Run batch engine
  8. Write output files (results, review, unmatched)
  9. Print summary: total B records, auto-matched, review, unmatched,
     elapsed time

**Verify:** `cargo run -- run --config testdata/configs/bench1000x1000.yaml`:
- Creates three output CSV files
- Print summary shows reasonable counts (expect ~700 auto, ~100 review,
  ~200 unmatched for synthetic data — exact numbers depend on thresholds)
- Compare output record counts against Go `match run` on the same config
  and data — should be in the same ballpark (not exact due to embedding
  differences)

### 3.10 — `tune` CLI command

- Wire up `tune --config <path> [--verbose]`:
  1. Load config, load state, initialize encoder pool
  2. Run batch engine in dry-run mode (no output files, no CrossMap writes)
  3. Collect all scores
  4. Print score distribution histogram:
     Buckets: `[0.0-0.1)`, `[0.1-0.2)`, ..., `[0.9-1.0]`
     Show count and bar chart per bucket
  5. Print per-field score statistics:
     For each match_field: min, max, mean, median, std_dev
  6. Print threshold analysis at current settings:
     Count and % for auto_match, review, no_match
  7. Print suggested thresholds: find the score at the 90th percentile
     (suggest as auto_match) and 50th percentile (suggest as review_floor)

**Verify:** `cargo run -- tune --config testdata/configs/bench1000x1000.yaml`
prints histograms and statistics without panics or errors. Score distribution
should show a bimodal pattern (cluster near 0 and cluster near 0.8+).

---

## Phase 4: Live Server

**Goal:** `meld serve` passes Go project's stress tests.

**Live semantics reminder:** Live mode IS symmetrical. A-side and B-side
records can both be upserted and matched. Both sides have a VecIndex,
BlockingIndex, and unmatched set.

### 4.1 — Add async dependencies

- Add to `Cargo.toml`:
  `tokio` (features: full), `axum`, `dashmap`, `tracing`,
  `tracing-subscriber`
- Verify `main.rs` can use `#[tokio::main]` on the `serve` path
  (other commands remain sync — use `tokio::runtime::Runtime::new()`
  if needed, or conditionally start the runtime)

**Verify:** `cargo build` succeeds with all new deps. No version conflicts
(especially: `fastembed`'s pinned `ort` should not clash).

### 4.2 — WAL (`src/state/upsert_log.rs`)

- Define WAL event types:
  ```rust
  enum WalEvent {
      UpsertRecord { side: Side, record: Record },
      CrossMapConfirm { a_id: String, b_id: String },
      CrossMapBreak { a_id: String, b_id: String },
  }
  ```
- Implement `UpsertLog` struct:
  - `open(path) -> Result<Self>`: open file for append, wrap in
    `BufWriter`
  - `append(&self, event: &WalEvent) -> Result<()>`: JSON serialize +
    newline, write to buffer (no fsync). Protected by `Mutex`.
  - `flush(&self) -> Result<()>`: flush `BufWriter` to OS
  - `replay(path) -> Result<Vec<WalEvent>>`: read all lines, parse JSON.
    Tolerate truncated last line (log warning, skip it).
  - `compact(&self, a_id_field: &str, b_id_field: &str) -> Result<()>`:
    read all events, deduplicate (last-write-wins per side+id for
    UpsertRecord; keep all CrossMap events in order), rewrite atomically
- Background flush: spawn `tokio::task` that calls `flush()` every 1 second

**Verify:**
- Append 100 UpsertRecord events + 5 CrossMapConfirm events. Close.
  Replay → all 105 recovered.
- Truncate file mid-line (simulate crash). Replay → 104 recovered,
  warning logged for truncated line.
- Compact with 50 duplicate IDs: verify only latest per side+id survives,
  all CrossMap events preserved.
- Background flush: append an event, wait 1.5s, verify file size increased
  (data reached disk).

### 4.3 — Live MatchState (`src/state/state.rs` extension)

- Implement `LiveSideState` struct:
  ```rust
  pub records: DashMap<String, Record>,
  pub index: RwLock<VecIndex>,
  pub unmatched: DashSet<String>,
  pub blocking_index: RwLock<BlockingIndex>,
  ```
- Implement `LiveMatchState` struct:
  ```rust
  pub config: Config,
  pub a: LiveSideState,
  pub b: LiveSideState,
  pub crossmap: RwLock<CrossMap>,
  pub encoder_pool: EncoderPool,
  pub wal: UpsertLog,
  pub crossmap_dirty: AtomicBool,
  ```
- Implement `LiveMatchState::load(config) -> Result<Arc<LiveMatchState>>`:
  Full startup sequence per DESIGN.md §19:
  1. Init encoder pool
  2. Load A dataset → DashMap `[parallel]`
  3. Load B dataset → DashMap `[parallel]`
  4. Build/load A-side VecIndex + cache `[parallel after 2]`
  5. Build/load B-side VecIndex + cache `[parallel after 3]`
  6. Build BlockingIndex for A side
  7. Build BlockingIndex for B side
  8. Load CrossMap from CSV
  9. Build unmatched sets: iterate all A IDs, add to `a.unmatched` if
     not in CrossMap; same for B
  10. Open WAL, replay events:
      - `UpsertRecord`: insert/replace in DashMap, update indices
      - `CrossMapConfirm`: add to CrossMap
      - `CrossMapBreak`: remove from CrossMap
  11. Rebuild unmatched sets after WAL replay
  12. Log startup summary

**Verify:** Load live state with 1K data:
- `a.records.len()` = 1000, `b.records.len()` = 1000
- `a.index.read().len()` = 1000
- `a.unmatched.len()` + CrossMap.len() ≈ 1000 (some matched)
- `a.blocking_index.read().query(...)` returns expected IDs

### 4.4 — Session: upsert flow (`src/session/session.rs`)

- Implement `Session` struct:
  ```rust
  state: Arc<LiveMatchState>,
  config: Config,
  start_time: Instant,
  upsert_count: AtomicU64,
  match_count: AtomicU64,
  ```
- Implement `upsert_record(&self, side: Side, record: Record) -> Result<UpsertResponse>`:

  Per DESIGN.md §7 direct path:
  1. Extract ID from record using config's id_field. Error if missing/empty.
  2. Check if existing record:
     - If exists: check CrossMap → if matched, break the pair
       (remove from CrossMap, add both IDs back to unmatched sets,
       WAL append CrossMapBreak). Status = "updated".
     - If new: status = "added"
  3. Insert/replace record in `state.{side}.records` (DashMap)
  4. Add ID to `state.{side}.unmatched` (if not already there)
  5. Update `state.{side}.blocking_index`: remove old record if
     replacing, insert new record
  6. Compute `primary_embedding_text(record, config)`. Check if text
     changed vs old record (if replacing). If unchanged, skip encode.
  7. If text changed or new: encode via `state.encoder_pool.encode(&[text])`
  8. Write-lock `state.{side}.index`, upsert vector, drop lock
  9. WAL append `UpsertRecord { side, record }`
  10. Generate candidates from opposite side:
      - Build allowed_ids = opposite.unmatched ∩ opposite.blocking_index.query()
      - Read-lock opposite.index, search_filtered(vec, top_n, &allowed_ids)
  11. Score each candidate via composite scorer
  12. Classify top result. If score >= auto_match:
      - Write-lock CrossMap, add pair
      - Remove both IDs from their unmatched sets
      - WAL append CrossMapConfirm
      - Mark CrossMap dirty
  13. Build response: status, id, side, classification, matches list,
      old_mapping (if record was previously matched)
  14. Increment counters

  **Critical: no DashMap Ref held across any .await point.** Clone/extract
  values before async operations (encode, lock acquisition).

**Verify:**
- Upsert new A record: appears in DashMap, unmatched set, VecIndex,
  BlockingIndex. WAL has UpsertRecord event.
- Upsert same ID with changed name: status "updated", old CrossMap
  broken (if existed), new vector in index.
- Upsert B record that auto-matches: CrossMap updated, both IDs
  removed from unmatched sets, WAL has both UpsertRecord and
  CrossMapConfirm.

### 4.5 — Session: try-match (read-only)

- Implement `try_match(&self, side: Side, record: Record) -> Result<MatchResponse>`:
  1. Extract ID (for response only — do NOT insert)
  2. Check CrossMap for existing match on this ID
  3. Encode record's embedding text via pool
  4. Generate candidates from opposite side (same logic as upsert step 10)
  5. Score and classify
  6. Return matches — do NOT write to CrossMap, DashMap, WAL, or any state

**Verify:** Try-match a B record:
- Returns scored candidates
- DashMap, CrossMap, unmatched sets, WAL: all unchanged before and after
- Repeated try-match with same record: identical results

### 4.6 — Session: CrossMap management

- `confirm_match(&self, a_id: &str, b_id: &str) -> Result<ConfirmResponse>`:
  1. Validate both IDs exist in their respective DashMaps
  2. Write-lock CrossMap, add pair
  3. Remove a_id from a.unmatched, b_id from b.unmatched
  4. WAL append CrossMapConfirm
  5. Mark CrossMap dirty
  6. Return `{ status: "confirmed" }`

- `lookup_crossmap(&self, id: &str, side: Side) -> Result<LookupResponse>`:
  1. Read-lock CrossMap
  2. Look up: if side=A, get_b(id); if side=B, get_a(id)
  3. If found: fetch matched record from opposite DashMap
  4. Return `{ id, side, status: "matched"|"unmatched", paired_id, matched_record }`

- `break_crossmap(&self, a_id: &str, b_id: &str) -> Result<BreakResponse>`:
  1. Validate the pair exists in CrossMap
  2. Write-lock CrossMap, remove pair
  3. Add a_id to a.unmatched, b_id to b.unmatched
  4. WAL append CrossMapBreak
  5. Mark CrossMap dirty
  6. Return `{ status: "broken", a_id, b_id }`

**Verify:**
- Confirm: pair in CrossMap, both IDs removed from unmatched
- Lookup after confirm: returns matched status + record
- Break: pair gone, both IDs back in unmatched
- Lookup after break: returns unmatched status

### 4.7 — CrossMap background flusher

- Implement `CrossMapFlusher`:
  - Spawns a `tokio::task` that runs every `crossmap_flush_secs`
  - Checks `crossmap_dirty` AtomicBool
  - If dirty: read-lock CrossMap, call `save()`, clear dirty flag
  - On drop (shutdown): final flush

**Verify:** Confirm a match (sets dirty). Wait > flush interval. Read CSV
from disk — new pair is present. Verify multiple confirms between flushes
are all captured.

### 4.8 — HTTP handlers (`src/api/handlers.rs`)

All endpoints per DESIGN.md §4:

- `POST /api/v1/a/add`:
  Parse `{"record": {...}}`, call `session.upsert_record(Side::A, record)`,
  serialize response with `a_id`/`b_id` field mapping.
- `POST /api/v1/b/add`: same, `Side::B`
- `POST /api/v1/a/match`:
  Parse `{"record": {...}}`, call `session.try_match(Side::A, record)`
- `POST /api/v1/b/match`: same, `Side::B`
- `POST /api/v1/match/b`: backward-compat alias → `/api/v1/b/match`
- `POST /api/v1/crossmap/confirm`:
  Parse `{"a_id": "X", "b_id": "Y"}`, call `session.confirm_match()`
- `GET /api/v1/crossmap/lookup?id=X&side=a|b`:
  Parse query params, call `session.lookup_crossmap()`
- `POST /api/v1/crossmap/break`:
  Parse `{"a_id": "X", "b_id": "Y"}`, call `session.break_crossmap()`
- `GET /api/v1/health`:
  Return `{ status: "ready", model: "...", records_a: N, records_b: N, crossmap_entries: N }`
- `GET /api/v1/status`:
  Return `{ job: "...", uptime_seconds: N, upserts: N, matches: N }`

Error handling:
- Missing/malformed JSON → 400 `{"error": "..."}`
- Missing required field in record → 400
- Internal error → 500 `{"error": "internal server error"}`

Response format for match/add endpoints:
```json
{
  "status": "added|updated|already_matched|match_found|no_match",
  "id": "...",
  "side": "a|b",
  "classification": "auto|review|no_match",
  "from_crossmap": false,
  "matches": [
    { "id": "...", "score": 0.91, "classification": "auto",
      "field_scores": [...], "matched_record": {...} }
  ],
  "old_mapping": {"a_id": "...", "b_id": "..."}  // if record was re-upserted
}
```

**Verify:** Use `axum::test` (or `tower::ServiceExt`) to unit-test each
handler: correct status codes, JSON shapes match Go API exactly.

### 4.9 — HTTP server (`src/api/server.rs`)

- Implement `start_server(session, port, socket_path) -> Result<()>`:
  - Build `axum::Router` with all routes from 4.8
  - Inject `Session` via axum `State` (wrapped in `Arc`)
  - Add middleware:
    - Request logging via `tower_http::trace::TraceLayer`
    - Panic recovery via `tower_http::catch_panic::CatchPanicLayer`
  - Bind to TCP port or Unix socket
  - Graceful shutdown on SIGTERM/SIGINT via `tokio::signal`

**Verify:**
- Start server on port 8091 with 1K data
- `curl http://localhost:8091/api/v1/health` returns valid JSON
- `curl -X POST http://localhost:8091/api/v1/a/add -d '{"record":{"entity_id":"TEST-001","legal_name":"Test Corp","country_code":"GB"}}'` returns response with matches
- Send SIGINT → server shuts down cleanly

### 4.10 — `serve` CLI command + full startup/shutdown

- Wire up `serve --config <path> [--port N] [--socket <path>]`:
  - Default port: 8080
  - Full startup per DESIGN.md §19:
    1. Load and validate config
    2. `LiveMatchState::load(config)` (encoder pool, datasets, caches,
       indices, CrossMap, WAL replay)
    3. Create `Session`
    4. Start CrossMap flusher
    5. Start WAL background flusher
    6. Start HTTP server
    7. Log "ready" with listen address
  - Full shutdown per DESIGN.md §19:
    1. Stop accepting connections
    2. Drain in-flight requests (30s timeout)
    3. Flush WAL
    4. Compact WAL
    5. Final CrossMap flush
    6. Save VecIndex caches to disk
    7. Log shutdown summary

**Verify:**
- Start with `bench_live.yaml` + 1K data. Health endpoint works. Upsert
  a record. Match a record. Ctrl-C → WAL compacted, CrossMap saved.
- Restart: WAL replayed, upserted record recovered.

### 4.11 — Stress test validation

- Run `bench/smoke_test.py` against `meld serve` on port 8090:
  Fix any JSON shape mismatches, missing fields, wrong status values.
- Run `bench/live_stress_test.py --iterations 100`:
  Sequential upserts. Fix any failures. Record req/s.
- Run `bench/live_concurrent_test.py --concurrency 10 --iterations 1000`:
  Concurrent upserts. Fix any race conditions, deadlocks, or data
  corruption. Record req/s and p50/p95 latency.
- Iterate: fix failures, re-run until all three scripts pass cleanly.

**Verify:**
- All three test scripts pass with zero failures
- Sequential throughput > 200 req/s (stretch: > 400)
- Concurrent throughput > 500 req/s (stretch: > 1000)
- No panics, no deadlocks, no data corruption

---

## Phase 5: Polish + Remaining CLI

**Goal:** Feature-complete CLI, robust error handling, final benchmarks.

### 5.1 — `review` CLI commands

- `review list --config <path>`:
  Load the review CSV from `output.review_path`. Print as a formatted
  table: a_id, b_id, score, classification, field scores.
  Handle: file doesn't exist (print "no review records").

- `review import --config <path> --file <path>`:
  Read a decisions CSV with columns: `a_id`, `b_id`, `decision`
  (values: `accept` or `reject`).
  For each `accept`: add to CrossMap, save.
  For each `reject`: remove from review CSV (or mark as rejected).
  Print summary: N accepted, M rejected.

**Verify:** Run `meld run` to generate a review CSV. `review list` prints
it. Create a decisions file accepting 2 pairs. `review import` adds them
to CrossMap. `crossmap stats` confirms count increased by 2.

### 5.2 — `crossmap` CLI commands

- `crossmap stats --config <path>`:
  Load CrossMap. Print: total pairs, A-side coverage (matched / total A),
  B-side coverage (matched / total B). Requires loading datasets to get
  totals.

- `crossmap export --config <path> --out <path>`:
  Load CrossMap, write to specified CSV path. Print count.

- `crossmap import --config <path> --file <path>`:
  Read CSV with a_id_field and b_id_field columns. Add all pairs to
  CrossMap. Save. Print count imported.

**Verify:** Export after a batch run → valid CSV. Import
`testdata/ground_truth_crossmap.csv` → CrossMap populated. Stats shows
correct counts.

### 5.3 — JSONL data loader (`src/data/jsonl.rs`)

- Implement `load_jsonl(path, id_field, required_fields) -> Result<(HashMap<String, Record>, Vec<String>), DataError>`:
  - Read file line by line
  - Parse each line as `serde_json::Value`, extract to `Record`
    (flatten to string values)
  - Same validations as CSV: id_field exists, no duplicates, required
    fields present
- Wire into `src/data/mod.rs` dispatch: if format is "jsonl", use this loader

**Verify:** Load `testdata/dataset_a_10000.jsonl`. Assert 10000 records
with correct fields. Run `meld validate` with a config pointing to
JSONL files.

### 5.4 — Parquet data loader (feature-flagged) (`src/data/parquet.rs`)

- Add `parquet` (arrow-rs) as optional dep:
  `parquet = { version = "...", optional = true }`
  Feature flag: `parquet = ["dep:parquet"]`
- Implement `load_parquet(path, id_field, required_fields)`:
  - Read parquet file, iterate row groups
  - Extract string columns, build Records
  - Same validations as CSV/JSONL
- Wire into data dispatch behind `#[cfg(feature = "parquet")]`

**Verify:** `cargo test --features parquet` — load
`testdata/dataset_a_10000.parquet`, assert 10000 records. Without feature
flag: parquet paths gracefully error ("parquet support not compiled").

### 5.5 — Tracing + structured logging

- Configure `tracing-subscriber` in `main.rs`:
  - Default filter: `RUST_LOG=melder=info`
  - JSON output option for production: `--log-format json`
  - Console output with colors for development (default)
- Add `#[instrument]` and manual spans per DESIGN.md §20:
  - `melder::startup` (startup sequence)
  - `melder::request` (per HTTP request: method, path)
  - `melder::upsert` (per upsert: side, id)
  - `melder::encode` (encode latency)
  - `melder::search` (vector search latency, k, result count)
  - `melder::score` (scoring latency, candidate count)
- Add structured fields: `side`, `id`, `score`, `classification`,
  `latency_ms`, `candidates`

**Verify:** Run serve with `RUST_LOG=melder=debug`. Upsert a record.
Verify log output includes: startup span with timing, request span,
upsert span with encode/search/score sub-spans and latency.

### 5.6 — Graceful shutdown hardening

- Verify WAL is compacted on shutdown (not just flushed)
- Verify CrossMap final flush writes to disk
- Verify VecIndex caches saved on shutdown (if cache paths configured)
- Test kill-and-restart:
  1. Start serve
  2. Upsert 10 records (some auto-match)
  3. Send SIGINT
  4. Restart serve
  5. Verify: all 10 records recovered via WAL, CrossMap pairs intact,
     unmatched sets correct

**Verify:** Kill-and-restart test with zero data loss. Health endpoint
after restart shows same record counts.

### 5.7 — Error handling audit

- Search codebase for all `unwrap()` and `expect()` calls — replace
  with proper error propagation or add context strings to `expect()`
- Verify all HTTP handlers return appropriate codes:
  - 400 for: missing record envelope, missing ID field, empty ID,
    invalid side parameter, malformed JSON
  - 500 for: encoder failure, index error, IO error
- Verify startup failures produce clear messages:
  - Config file not found
  - Config validation error
  - Model download failure (network unavailable)
  - Dataset file not found
  - CSV parse error (malformed row)
  - Cache corruption (truncated file)

**Verify:** Exercise each error path manually. No panics, all produce
human-readable error messages with file/line context.

### 5.8 — Final benchmark + comparison

- Environment: same machine, same data (10K×10K), warm caches
- Start Go server: `./match serve --config bench/bench_live.yaml --port 8090`
- Start Rust server: `./melder serve --config bench/bench_live.yaml --port 8091`
- Run sequential test against both:
  `python3 bench/live_stress_test.py --iterations 1000`
  Record: req/s, p50, p95
- Run concurrent test against both:
  `python3 bench/live_concurrent_test.py --concurrency 10 --iterations 1000`
  Record: req/s, p50, p95, CPU utilization, memory
- Compare against targets:

  | Metric | Go+Python | Rust target | Actual |
  |---|---|---|---|
  | Sequential (c=1) | ~72-111 req/s | **400+** | ? |
  | Concurrent (c=10) | ~150 req/s | **1000+** | ? |
  | Encode latency | ~9-16ms | **3-5ms** | ? |
  | Machine util (c=10) | 13% | **>60%** | ? |
  | Memory (10K) | ~1GB | **<500MB** | ? |

- Document results in a BENCHMARKS.md
- If targets not met: profile with `cargo flamegraph`, identify
  bottleneck, document path to improvement

**Verify:** All benchmark numbers recorded. If below target, root cause
identified and documented.

---

## Task Summary

| Phase | Description | Tasks | Est. hours |
|---|---|---|---|
| 0 | Scaffold + test data | 2 | ~2 |
| 1 | Config + models + scoring | 11 | ~20 |
| 2 | Encoder + index + cache | 6 | ~12 |
| 3 | Matching engine + batch | 10 | ~20 |
| 4 | Live server | 11 | ~25 |
| 5 | Polish + CLI + benchmark | 8 | ~15 |
| **Total** | | **48** | **~94** |

---

## Critical Path

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                                              ↑
                                    This is the milestone:
                                    `meld serve` works and
                                    passes stress tests
```

Phase 5 tasks are mostly independent of each other and can be done in
any order. Phases 1-4 are strictly sequential — each builds on the
previous.

The single riskiest task is **2.1 (encoder pool)** — it's the first time
we integrate `fastembed` and verify ONNX model loading works on the target
platform. If this fails, we need to fall back to raw `ort`. Do this task
early and verify thoroughly before building on top of it.

The second riskiest is **4.11 (stress test validation)** — this is where
all the concurrency bugs surface. Budget extra time for debugging.
