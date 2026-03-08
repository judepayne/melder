# melder — Design

A full rewrite of the `match` Go+Python project as a single-language Rust
binary. Same functionality, same YAML config format, same HTTP API, no IPC
boundary, no GIL, true multi-threaded parallelism.

**Reference implementation:** `/Users/jude/Library/CloudStorage/Dropbox/Projects/match/`
(git repo at that root, Go module at `match/` subdirectory, Python sidecar at
`match/sidecar/pipe_main.py`). The Go+Python version is fully working and can be
run to compare behaviour and benchmark results.

---

## 1. Why Rewrite

The Go+Python architecture hits a hard performance ceiling in live mode:

| Load | Go CPU | Python CPU | Machine utilization (8 cores) |
|---|---|---|---|
| Sequential (c=1) | 9% | 25% | **4%** |
| Concurrent (c=10) | 38% | 69% | **13%** |

Three structural problems cannot be optimized away:

1. **IPC serialization.** Every sidecar call round-trips through JSON-over-pipe.
   Embedding vectors (384 floats = ~3KB JSON each) are serialized and parsed on
   both sides. This costs 1-2ms per upsert (6-12% of wall time).

2. **Single-threaded Python.** The GIL prevents parallel work within the sidecar.
   A-side and B-side encodes cannot overlap. Tokenization, numpy, FAISS calls
   are all serialized. An attempt to use 2-CPU parallel encode within Python
   hung indefinitely on the first real call.

3. **Cross-process state.** The FAISS index lives in Python; Go maintains its
   own record maps. Keeping them in sync requires careful coordination. Index
   rebuilds are O(N) because incremental updates are complex across the process
   boundary.

All three Python libraries now have production-quality Rust equivalents:

| Python | Rust equivalent | Maturity |
|---|---|---|
| `sentence-transformers` | `fastembed-rs` (ort + tokenizers) | Production (v5.12, Qdrant team) |
| `faiss` | Flat SIMD scan / HNSW | Stable |
| `rapidfuzz` | `rapidfuzz-rs` (official port, same author) | Usable (v0.5) |

HF's own production embedding server (TEI) is written in Rust with Candle.

**Current Go+Python benchmarks (M3 MacBook Air, 10K x 10K, warm cache):**
- Sequential c=1: ~72-111 req/s, p50 ~8-16ms
- Concurrent c=10: ~150 req/s, p50 ~65ms
- Concurrent c=20: ~143 req/s (saturated)

**Target Rust benchmarks:**
- Sequential c=1: 400+ req/s
- Concurrent c=10: 1000+ req/s
- Machine utilization: >60%

---

## 2. Configuration — Port in Full

The config schema is the cleanest part of the Go codebase. It is well-separated
from implementation concerns. Users should be able to point `melder` at an
existing `match` YAML config with zero changes.

### Config struct (Rust equivalent of `schema.go`)

The Go config structs are at:
`/Users/jude/Library/CloudStorage/Dropbox/Projects/match/match/internal/config/schema.go`

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub job: JobConfig,
    pub datasets: DatasetsConfig,
    pub cross_map: CrossMapConfig,
    pub embeddings: EmbeddingsConfig,
    pub blocking: BlockingConfig,
    pub match_fields: Vec<MatchField>,
    #[serde(default)]
    pub candidates: CandidatesConfig,
    #[serde(default)]
    pub output_mapping: Vec<FieldMapping>,
    pub thresholds: ThresholdsConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub live: LiveConfig,
    #[serde(default)]
    pub workers: Option<u32>,
    // `sidecar` section: accept and ignore for backward compat
    #[serde(default)]
    pub sidecar: Option<serde_yaml::Value>,

    // Derived at load time (not in YAML)
    #[serde(skip)]
    pub required_fields_a: Vec<String>,
    #[serde(skip)]
    pub required_fields_b: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct JobConfig {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct DatasetsConfig {
    pub a: DatasetConfig,
    pub b: DatasetConfig,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub path: String,
    pub id_field: String,
    #[serde(default)]
    pub format: Option<String>,   // "csv"|"parquet"|"jsonl"; inferred from ext
    #[serde(default)]
    pub encoding: Option<String>, // for csv/jsonl; defaults to utf-8
}

#[derive(Debug, Deserialize)]
pub struct CrossMapConfig {
    #[serde(default = "default_backend")]
    pub backend: String,          // "local" | "redis"
    #[serde(default)]
    pub path: Option<String>,     // for local backend
    #[serde(default)]
    pub redis_url: Option<String>,
    pub a_id_field: String,
    pub b_id_field: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsConfig {
    pub model: String,            // HF model name or local ONNX path
    pub a_index_cache: String,
    #[serde(default)]
    pub b_index_cache: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BlockingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_operator")]
    pub operator: String,         // "and" | "or"
    #[serde(default)]
    pub fields: Vec<BlockingFieldPair>,
    #[serde(default)]
    pub field_a: Option<String>,  // legacy single-field
    #[serde(default)]
    pub field_b: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BlockingFieldPair {
    pub field_a: String,
    pub field_b: String,
}

#[derive(Debug, Deserialize)]
pub struct MatchField {
    pub field_a: String,
    pub field_b: String,
    pub method: String,           // "exact" | "fuzzy" | "embedding"
    #[serde(default)]
    pub scorer: Option<String>,   // "wratio"|"partial_ratio"|"token_sort"
    pub weight: f64,
}

#[derive(Debug, Deserialize, Default)]
pub struct CandidatesConfig {
    #[serde(default)]
    pub enabled: Option<bool>,    // nil/true = enabled
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub scorer: Option<String>,
    #[serde(default)]
    pub n: Option<usize>,         // default 10
}

#[derive(Debug, Deserialize)]
pub struct FieldMapping {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Deserialize)]
pub struct ThresholdsConfig {
    pub auto_match: f64,
    pub review_floor: f64,
}

#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    pub results_path: String,
    pub review_path: String,
    pub unmatched_path: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct LiveConfig {
    #[serde(default)]
    pub top_n: Option<usize>,         // default 5
    #[serde(default)]
    pub upsert_log: Option<String>,
    #[serde(default)]
    pub encoder_pool_size: Option<usize>,  // default 1; see §7 Encoder Pool
    #[serde(default)]
    pub crossmap_flush_secs: Option<u64>,  // default 5; see §7 CrossMap Persistence
}

fn default_backend() -> String { "local".into() }
fn default_operator() -> String { "and".into() }
```

### What changed from Go

- **`sidecar` section:** Accepted and ignored (`Option<serde_yaml::Value>`)
  for backward compatibility.
- **`embeddings.model`:** Same field name, slightly different semantics. In Go
  it's a `sentence-transformers` PyTorch model name. In Rust it's an ONNX model
  (resolved by `fastembed-rs` from the same HF name like `all-MiniLM-L6-v2`,
  or a local path to a pre-exported ONNX directory).
- **Cache files:** Same path fields, different binary content. `.index` files
  will be serialized flat vectors (not FAISS). **Not compatible** with Go caches.
- **`output` section:** Could be made optional in `serve` mode (Go requires it
  even when unused — see `bench_live.yaml` line 76).
- **`cross_map.backend: "redis"`:** Kept in schema; only `local` implemented.
- **`live.encoder_pool_size`:** New field (not in Go). Controls number of
  concurrent ONNX inference sessions. Default 1. Higher values trade memory
  for throughput when concurrent upserts need parallel encoding.
- **`live.crossmap_flush_secs`:** New field (not in Go). Controls how often
  dirty CrossMap state is flushed to disk. Default 5 seconds.

### Validation rules (port from `loader.go`)

The Go validation logic is at:
`/Users/jude/Library/CloudStorage/Dropbox/Projects/match/match/internal/config/loader.go`

All of these must be ported exactly:

1. `job.name` required
2. `datasets.a.path`, `datasets.a.id_field` required; same for B
3. Format inferred from extension: `.csv`/`.tsv`->csv, `.parquet`->parquet,
   `.jsonl`/`.ndjson`->jsonl, `.json`->error, `.xlsx`->error
4. `cross_map.backend` default "local"; "local" requires `path`; "redis"
   requires `redis_url`; `a_id_field` and `b_id_field` required
5. `embeddings.model`, `a_index_cache` required
6. At least one `match_fields` entry; each needs `field_a`, `field_b`, valid
   `method`, `weight > 0`
7. **Weights must sum to 1.0** (tolerance 0.001)
8. `thresholds.auto_match` in (0, 1]; `review_floor` in [0, 1);
   `auto_match > review_floor`
9. `output` paths all required
10. Blocking: if enabled, at least one field pair; operator "and" or "or"
11. Legacy single-field blocking normalized into `fields` list
12. `candidates.n` default 10; `candidates.scorer` default "wratio"
13. `live.top_n` default 5
14. `workers` default 4
15. `live.encoder_pool_size` default 1; must be >= 1
16. `live.crossmap_flush_secs` default 5; must be >= 1
17. Derive `required_fields_a`/`required_fields_b` from match_fields +
    output_mapping + blocking + id fields

### Example config (must work identically in Go and Rust)

```yaml
job:
  name: bench_live_10000x10000
  description: Live serve stress test

datasets:
  a:
    path: testdata/dataset_a_10000.csv
    id_field: entity_id
    format: csv
  b:
    path: testdata/dataset_b_10000.csv
    id_field: counterparty_id
    format: csv

cross_map:
  backend: local
  path: bench/crossmap_live.csv
  a_id_field: entity_id
  b_id_field: counterparty_id

embeddings:
  model: all-MiniLM-L6-v2
  a_index_cache: bench/cache/a_10000.index
  b_index_cache: bench/cache/b_10000.index

blocking:
  enabled: true
  operator: and
  fields:
    - field_a: country_code
      field_b: domicile

match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.55
  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy
    scorer: partial_ratio
    weight: 0.20
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05

candidates:
  enabled: true
  method: fuzzy
  scorer: wratio
  n: 10

thresholds:
  auto_match: 0.85
  review_floor: 0.60

live:
  top_n: 5
  upsert_log: bench/live_upserts.ndjson
  # encoder_pool_size: 1      # optional, default 1
  # crossmap_flush_secs: 5    # optional, default 5

output:
  results_path:   bench/output/live_results.csv
  review_path:    bench/output/live_review.csv
  unmatched_path: bench/output/live_unmatched.csv
```

---

## 3. Core Data Models

The Go models are at:
`/Users/jude/Library/CloudStorage/Dropbox/Projects/match/match/pkg/models.go`

```rust
use std::collections::HashMap;

pub type Record = HashMap<String, String>;

/// Side represents which dataset a record belongs to.
/// Used throughout the engine to ensure symmetric handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side { A, B }

impl Side {
    pub fn opposite(&self) -> Side {
        match self { Side::A => Side::B, Side::B => Side::A }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldScore {
    pub field_a: String,
    pub field_b: String,
    pub method: String,    // "exact" | "fuzzy" | "embedding"
    pub score: f64,        // 0..1
    pub weight: f64,
}

impl FieldScore {
    pub fn contribution(&self) -> f64 { self.score * self.weight }
}

/// Internal engine result. Uses query_id/matched_id to avoid the confusing
/// a_id/b_id naming when direction reverses. Mapped to a_id/b_id at the
/// API serialization boundary based on Side.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub query_id: String,          // ID of the record being matched
    pub matched_id: String,        // ID of the candidate from the opposite pool
    pub query_side: Side,          // which side initiated the match
    pub score: f64,
    pub field_scores: Vec<FieldScore>,
    pub classification: Classification,
    pub matched_record: Option<Record>,
    pub from_crossmap: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Classification { Auto, Review, NoMatch }

impl Classification {
    pub fn as_str(&self) -> &'static str {
        match self {
            Classification::Auto => "auto",
            Classification::Review => "review",
            Classification::NoMatch => "no_match",
        }
    }

    pub fn from_score(score: f64, auto_match: f64, review_floor: f64) -> Self {
        if score >= auto_match { Classification::Auto }
        else if score >= review_floor { Classification::Review }
        else { Classification::NoMatch }
    }
}
```

### API serialization

At the HTTP boundary, `MatchResult` is serialized with `a_id`/`b_id` fields
for backward compatibility. The mapping is trivial:

```rust
// In api/handlers.rs
fn to_api_match(r: &MatchResult) -> serde_json::Value {
    let (a_id, b_id) = match r.query_side {
        Side::A => (&r.query_id, &r.matched_id),
        Side::B => (&r.matched_id, &r.query_id),
    };
    json!({ "a_id": a_id, "b_id": b_id, "score": r.score, ... })
}
```

This eliminates the error-prone field swapping in Go's `matchTopNDirected`.

---

## 4. HTTP API — Port Exactly

The Go handlers are at:
`/Users/jude/Library/CloudStorage/Dropbox/Projects/match/match/api/handlers.go`
The Go server setup is at:
`/Users/jude/Library/CloudStorage/Dropbox/Projects/match/match/api/server.go`

### Endpoints

```
POST /api/v1/a/add          upsert A record + match against B_unmatched
POST /api/v1/b/add          upsert B record + match against A_unmatched
POST /api/v1/a/match        try-match A record (read-only)
POST /api/v1/b/match        try-match B record (read-only)
POST /api/v1/crossmap/confirm
GET  /api/v1/crossmap/lookup?id=X&side=a|b
POST /api/v1/crossmap/break
GET  /api/v1/health
GET  /api/v1/status
POST /api/v1/match/b        backward-compat alias for /b/match
```

### Request format

All POST matching endpoints require `{"record": {...}}` envelope:

```json
{"record": {"entity_id": "ENT-001", "legal_name": "Barclays PLC", "country_code": "GB"}}
```

### Response format (unified for match/add)

```json
{
  "status": "added",
  "id": "ENT-001",
  "side": "a",
  "classification": "auto",
  "from_crossmap": false,
  "matches": [
    {
      "id": "CP-001",
      "score": 0.91,
      "classification": "auto",
      "field_scores": [...],
      "matched_record": {...}
    }
  ],
  "old_mapping": {"a_id": "ENT-001", "b_id": "CP-999"}
}
```

Status values: "added", "updated", "already_matched", "match_found", "no_match".

**Confirm:** `{"a_id":"X","b_id":"Y"}` -> `{"status":"confirmed"}`
**Lookup:** `?id=X&side=a` -> `{"id":"X","side":"a","status":"matched","paired_id":"Y","matched_record":{...}}`
**Break:** `{"a_id":"X","b_id":"Y"}` -> `{"status":"broken","a_id":"X","b_id":"Y"}`
**Health:** `{"status":"ready","model":"...","records_a":10000,"crossmap_entries":42}`
**Status:** `{"job":"...","uptime_seconds":3600,"matched_today":150,...}`

---

## 5. CLI Interface

```
melder
  run        --config <path>  [--dry-run] [--verbose] [--limit N]
  serve      --config <path>  [--port N]  [--socket <path>]
  validate   --config <path>
  cache
    build    --config <path>
    status   --config <path>
    clear    --config <path>  [--index-only]
  review
    list     --config <path>
    import   --config <path>  --file <path>
  crossmap
    stats    --config <path>
    export   --config <path>  --out <path>
    import   --config <path>  --file <path>
```

Use `clap` with derive macros.

---

## 6. Matching Semantics — Port Exactly

### Composite scoring

Go composite scorer: `.../match/internal/scoring/composite.go`

```
final_score = sum(field_score[i] * weight[i])
```

If weights don't sum to 1.0, normalize. (Validator requires sum = 1.0.)

### Per-field scoring

- **exact** (Go: `scoring/exact.go`, 24 lines): Case-insensitive string equality
  after trim. Both empty = 0.0. Match = 1.0. No match = 0.0.

- **fuzzy**: `rapidfuzz-rs` scorers: `wratio`, `partial_ratio`, `token_sort`.
  Returns 0.0-1.0.

- **embedding** (Go: `scoring/embedding.go`, 58 lines): Cosine similarity of
  L2-normalized vectors. `dot_product(a, b)` for unit vectors. Clamped to [0,1].

### Classification

```
score >= auto_match   -> Classification::Auto
score >= review_floor -> Classification::Review
score < review_floor  -> Classification::NoMatch
```

### Blocking

Go: `.../match/internal/matching/blocking.go`

Multi-field AND/OR pre-filter. Case-insensitive trimmed comparison. Missing
query values skip that constraint. Direction-aware (B->A or A->B).

### Candidate generation

Go: `.../match/internal/matching/candidates.go`

Two paths: (1) FuzzySearch ~9-17ms (2) Vector nearest-neighbour ~1ms.
In Rust, vector search is primary, fuzzy is fallback.

### CrossMap

Go: `.../match/internal/crossmap/local.go`

Bidirectional HashMap pair (`b_to_a`, `a_to_b`). CSV persist with atomic rename.

### Upsert-by-ID

Go: `.../match/session/session.go` (UpsertARecord line 362, UpsertBRecord line 519)

1. Extract ID. 2. If existing: replace, break CrossMap, "updated". If new:
   insert, "added". 3. WAL append. 4. Embedding change detection: skip encode
   if text unchanged. 5. Match against opposite unmatched pool. 6. Auto-confirm
   if >= auto_match.

### WAL

Go: `.../match/internal/state/upsertlog.go`

NDJSON append-only. Each line: `{"side":"a"|"b","record":{...}}`. Replay on
startup (last-write-wins). Compact on clean shutdown.

### Symmetry invariant

**Every feature for A must exist identically for B.** Hard invariant.

---

## 7. Architecture (Rust)

```
+---------------------------------------------------------------+
|  melder (single binary)                                        |
|                                                                |
|  CLI (clap)                                                    |
|    +-- BatchEngine                                             |
|    +-- LiveServer (axum)                                       |
|           |                                                    |
|       MatchState                                               |
|        +-- SideState (x2: A and B)                             |
|        |    +-- DashMap<id, Record>                             |
|        |    +-- RwLock<VecIndex>                                |
|        |    +-- DashSet<id>           (unmatched IDs)           |
|        |    +-- RwLock<BlockingIndex>                           |
|        +-- RwLock<CrossMap>                                     |
|        +-- EncoderPool               (1..N TextEmbedding)      |
|        +-- UpsertLog                 (WAL)                     |
|        +-- CacheManager                                        |
|        +-- CrossMapFlusher           (background timer)        |
|                                                                |
|  All in one process, no IPC:                                   |
|    Tokenize, Encode, Search, Fuzzy, Score, Classify            |
+---------------------------------------------------------------+
```

### Concurrency model: direct path (no coordinator)

The Go version needs a coordinator goroutine, 3-phase upsert, pool snapshots,
and BatchUpsertAndQuery — all to work around the IPC boundary.

In Rust, encoding is in-process. The primary reason for Go's coordinator
(batching IPC calls to amortize serialization) no longer applies. We use a
**direct sequential path** for each upsert:

1. Insert/update record in `DashMap` (lock-free)
2. Update `DashSet<unmatched>` and `BlockingIndex`
3. If embedding text changed: acquire encoder from pool, encode (~3-5ms), release
4. Write-lock `VecIndex`, insert/replace vector, release
5. Read-lock `VecIndex`, search candidates, release
6. Score candidates (pure computation, no locks)
7. If auto-match: write-lock `CrossMap`, add pair, update both `DashSet<unmatched>`
8. WAL append

Multiple concurrent upserts run this pipeline in parallel (each in its own
tokio task). The fine-grained locking means they only block each other at
write-lock points (steps 4, 7) for microseconds.

**Adaptive batching (deferred).** If benchmarks reveal that ONNX inference
benefits significantly from batching (larger batch = better GPU utilization),
we can add a coordinator as an optimization. This would drain a bounded channel,
group pending encodes, and dispatch a single batched inference call. The direct
path is correct first; batching is a performance optimization added if needed.

### DashMap / DashSet usage caveats

Three rules enforced throughout the codebase:

1. **Never hold a `Ref`/`RefMut` across `.await`** — this blocks the shard's
   RwLock for the entire duration the task is suspended, starving other tasks.
   Always `.clone()` or extract the value before awaiting.
2. **Never call DashMap methods from inside an iterator callback** on the same
   map — the locks are not reentrant, this will deadlock.
3. **`len()` is not atomic** across shards — acceptable for status endpoints
   but do not rely on it for correctness logic.

### Encoder pool

`N` encoder instances, each behind its own `tokio::sync::Mutex`.

**Why Mutex per instance:** `fastembed::TextEmbedding::embed()` requires
`&mut self`. A single `TextEmbedding` cannot serve concurrent encode requests.
Each pool slot is an independent instance with its own ONNX session and
tokenizer state. ONNX model weights are memory-mapped by the OS, so multiple
instances share the same physical pages — only runtime buffers (~50MB) are
duplicated per instance.

```rust
pub struct EncoderPool {
    encoders: Vec<tokio::sync::Mutex<TextEmbedding>>,
}

impl EncoderPool {
    pub fn new(model: EmbeddingModel, pool_size: usize) -> Result<Self> {
        let encoders = (0..pool_size)
            .map(|_| {
                let te = TextEmbedding::try_new(
                    InitOptions::new(model.clone())
                        .with_show_download_progress(false),
                )?;
                Ok(tokio::sync::Mutex::new(te))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { encoders })
    }

    /// Encode texts using the first available encoder in the pool.
    /// Tries each slot in round-robin order, blocks on the first available.
    pub async fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Simple approach: try_lock round-robin, fall back to awaiting slot 0
        for encoder in &self.encoders {
            if let Ok(mut guard) = encoder.try_lock() {
                return Ok(guard.embed(texts, None)?);
            }
        }
        // All busy — await the first slot
        let mut guard = self.encoders[0].lock().await;
        Ok(guard.embed(texts, None)?)
    }
}
```

**Configuration:** `live.encoder_pool_size` (default 1).

- **Pool size 1:** Safest starting point. On Apple Silicon, CPU-based ONNX
  inference with ARM NEON is fast enough for sequential upserts.
  Note: `fastembed` does not expose a CoreML execution provider for ONNX
  models — inference runs on CPU with ARM NEON vectorization. This is still
  significantly faster than Python's GIL-constrained path.
- **Pool size 2-4:** Useful when CPU cores are underutilized under high
  concurrency. Each additional instance costs ~50MB (runtime buffers; model
  weights are shared via OS memory mapping).
- **Pool size > 4:** Unlikely to help. Diminishing returns from memory
  bandwidth saturation.

### Vector index

**Decision: start with flat brute-force, graduate to `usearch` if needed.**

At our target scale (10K-100K records, 384-dimensional vectors), a flat scan
is competitive:

| Records | Flat scan (SIMD) | HNSW | Notes |
|---|---|---|---|
| 10K | ~0.3ms | ~0.1ms | Both negligible vs encode time |
| 100K | ~3ms | ~0.2ms | HNSW wins; flat still acceptable |
| 1M | ~30ms | ~0.3ms | HNSW required |

Flat scan advantages: exact results (no approximation), trivial incremental
insert/delete/update, no tuning parameters, simple serialization. This matters
because Go's FAISS IndexFlatIP also uses brute-force — the performance win
comes from eliminating IPC, not from algorithm change.

```rust
pub struct VecIndex {
    /// Dense matrix: N rows x D columns, row-major.
    vectors: Vec<f32>,
    dim: usize,
    /// Parallel array: vectors[i*dim..(i+1)*dim] belongs to ids[i].
    ids: Vec<String>,
    /// Reverse lookup for update/delete.
    id_to_pos: HashMap<String, usize>,
}

impl VecIndex {
    /// Insert or replace a vector.
    pub fn upsert(&mut self, id: &str, vec: &[f32]) { ... }

    /// Remove a vector (swap-remove for O(1)).
    pub fn remove(&mut self, id: &str) -> bool { ... }

    /// Find top-K nearest by dot product (vectors assumed L2-normalized).
    /// Scans ALL vectors. Use search_filtered for live-mode queries.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> { ... }

    /// Find top-K nearest, considering only vectors whose IDs are in `allowed`.
    /// Skips dot-product computation for IDs not in the set.
    /// Complexity: O(|allowed| * D) instead of O(N * D).
    /// This is the primary search method for live-mode upserts.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &HashSet<String>,
    ) -> Vec<(String, f32)> { ... }
}
```

**Graduation path:** If deployments exceed 100K records and flat scan becomes
a bottleneck, swap in `usearch` (v2.24, 3.9K stars, used by ClickHouse and
DuckDB). It provides: incremental insert, tombstone-based deletion,
update (remove + re-add), SIMD-accelerated distance, concurrent insert+search,
serialization to file/buffer/mmap, and built-in cosine/dot-product. The `VecIndex`
trait API stays the same — callers don't know the underlying algorithm. Trade-off:
`usearch` requires a C++ compiler at build time (single-header C++11 via `cxx`).

**Eliminated alternatives:**
- `instant-distance`: No incremental insert, no deletion, no update. Read-only
  after build. Unusable for live mode.
- `hnsw_rs`: Active maintenance, pure Rust, but no deletion and requires
  mode-toggle between insert and search (cannot do both concurrently).

### Unmatched pool tracking

**Problem (Go):** `aUnmatched()` / `bUnmatched()` do O(N) linear scans over
the full record map, checking CrossMap presence for each ID. Called on every
upsert.

**Solution:** Each `SideState` maintains a `DashSet<String>` of unmatched IDs.
Updated incrementally:

- Record inserted -> add to unmatched set
- CrossMap pair added -> remove from both sides' unmatched sets
- CrossMap pair broken -> add back to both sides' unmatched sets
- Record replaced (upsert existing) -> break old CrossMap pair first (existing
  logic), record stays in unmatched set

This turns "get unmatched pool" from O(N) scan to O(1) per-ID check, and
"iterate unmatched" from O(N) filter to O(unmatched) direct iteration.

### Blocking index

**Problem (Go):** Live mode does linear-scan blocking — O(pool_size) per query.
Batch mode uses indexed blocking but live does not.

**Solution:** Maintain a blocking index per side, updated incrementally:

```rust
/// For AND blocking (all fields must match):
/// Key = tuple of all blocking field values (lowercased, trimmed)
/// For OR blocking (any field can match):
/// One map per blocking field, merged at query time
pub struct BlockingIndex {
    operator: BlockingOperator,
    // AND: single map from composite key -> set of record IDs
    and_index: HashMap<Vec<String>, HashSet<String>>,
    // OR: one map per field, value -> set of record IDs
    or_indices: Vec<HashMap<String, HashSet<String>>>,
}

impl BlockingIndex {
    /// Insert a record's blocking values.
    pub fn insert(&mut self, id: &str, record: &Record, fields: &[BlockingFieldPair]) { ... }

    /// Remove a record from the index.
    pub fn remove(&mut self, id: &str, record: &Record, fields: &[BlockingFieldPair]) { ... }

    /// Return IDs of records that pass the blocking filter for the query.
    /// Missing query values skip that constraint (same as Go).
    pub fn query(&self, query_record: &Record, fields: &[BlockingFieldPair]) -> HashSet<String> { ... }
}
```

**Candidate generation pipeline (live mode):**

```
1. Build filter set:
   a. Start with unmatched IDs for the target side (DashSet, O(1) per check)
   b. If blocking enabled: intersect with BlockingIndex.query() result
   c. Result: allowed_ids — only these IDs are searched

2. Pre-filtered vector search:
   VecIndex.search_filtered(query_vec, K, &allowed_ids)
   Complexity: O(|allowed_ids| * D) — NOT O(N * D)
   At 10K records with 60% unmatched and 30% blocking pass rate:
     unfiltered: 10K dot products
     filtered:   ~1.8K dot products (5.5x reduction)

3. If fewer than K results from vector search:
   Supplement with fuzzy fallback — compute wratio between query and
   remaining allowed records' primary text field, take top remaining.
   This handles cases where embedding similarity misses lexically
   similar records (e.g., abbreviations, acronyms).

4. Score all candidates via composite scorer (O(K * F))
5. Sort by score descending, cap at top_n
```

If blocking is disabled, step 1b is skipped (allowed = all unmatched).
If no embedding fields exist, skip step 2 and use fuzzy search only.

**Batch mode** uses `search()` (unfiltered) since it processes all records
in bulk and blocking is applied as a post-filter.

### CrossMap persistence

**Problem (Go):** Writes entire CrossMap CSV on every auto-confirm — O(N) I/O
on the hot path.

**Solution:** Timer-based background flush with WAL-backed durability.

```
Mutation path:
  1. Update in-memory CrossMap (DashMap)
  2. Append to WAL: {"type":"crossmap_confirm","a_id":"X","b_id":"Y"}
     (reuse the existing upsert WAL file — just add a new event type)
  3. Mark CrossMap as dirty

Background flusher (every `crossmap_flush_secs`, default 5s):
  1. If dirty: write full CSV atomically (temp + rename)
  2. Clear dirty flag

Crash recovery:
  - Load CrossMap CSV (may be stale by up to flush interval)
  - Replay WAL (captures all confirms/breaks since last flush)
  - Result: identical state to pre-crash
```

The WAL already captures the causal ordering of all mutations. The periodic
CSV flush is a performance optimization (faster restart than full WAL replay)
not a correctness requirement.

### WAL write strategy

The Go version calls `fsync` after every WAL append. At 1000+ req/s this
becomes a bottleneck (SSDs sustain 10K-50K fsync/s, each ~0.1-1ms).

**Strategy: buffered write, no per-request fsync.**

```
Per upsert:
  1. Serialize WAL entry to buffer (in-memory, ~microseconds)
  2. Write to file (buffered, OS decides when to flush to disk)

Periodic (every 1s or on buffer full):
  3. Flush buffer to OS
  4. Optional: fsync (configurable, default OFF)

On shutdown:
  5. Flush + fsync + compact
```

Rationale: the WAL is a recovery optimization, not a durability guarantee.
The CrossMap CSV (flushed every 5s) is the authoritative source for confirmed
matches. The WAL captures record upserts and CrossMap mutations since the last
CSV flush. On crash, we lose at most 1 second of WAL entries — those upserts
can be re-sent by the client (upsert is idempotent by design).

If strict durability is needed (e.g., regulatory), enable `fsync` per-write
via config flag and accept the throughput ceiling (~10K req/s on NVMe).

### Fuzzy matching

**`rapidfuzz-rs` provides only `fuzz::ratio` (basic edit-distance ratio).**
The three scorers we need — `wratio`, `partial_ratio`, `token_sort_ratio` —
are NOT implemented in the Rust crate (confirmed: dormant since Dec 2023, all
fuzz scorers unchecked in the upstream porting tracker).

**We implement them ourselves** in `src/fuzzy/`. These are well-documented
composite algorithms built on top of the basic `ratio` which IS available:

```rust
/// src/fuzzy/scorer.rs

/// Split on whitespace, sort tokens, rejoin, compute ratio.
/// ~10 lines of logic on top of rapidfuzz::fuzz::ratio.
pub fn token_sort_ratio(a: &str, b: &str) -> f64 { ... }

/// Slide shorter string across longer string, find best-aligned window,
/// compute ratio on that window. ~30 lines.
pub fn partial_ratio(a: &str, b: &str) -> f64 { ... }

/// Try ratio, token_sort_ratio, partial_ratio; return best weighted result.
/// This matches Python rapidfuzz.fuzz.WRatio semantics. ~20 lines.
pub fn wratio(a: &str, b: &str) -> f64 { ... }
```

All return 0.0-1.0 (the Rust `rapidfuzz` crate already uses this range, unlike
Python's 0-100 default). Rayon for batch parallelism in `run` mode. In `serve`
mode, fuzzy scores are computed inline (microseconds per pair).

**Parity testing:** Generate golden test pairs by running Python `rapidfuzz`
on 50+ string pairs covering edge cases (empty strings, Unicode, case
differences, partial overlaps). Our implementations must match within ±0.001.

---

## 8. Project Structure

```
melder/
+-- Cargo.toml
+-- src/
|   +-- main.rs
|   +-- lib.rs               (re-exports for integration tests)
|   +-- error.rs             (error types — see §18)
|   +-- models.rs
|   +-- config/
|   |   +-- mod.rs
|   |   +-- schema.rs
|   |   +-- loader.rs
|   +-- data/
|   |   +-- mod.rs
|   |   +-- csv.rs
|   |   +-- parquet.rs       (behind feature flag)
|   |   +-- jsonl.rs
|   +-- encoder/
|   |   +-- mod.rs
|   |   +-- pool.rs
|   +-- index/
|   |   +-- mod.rs
|   |   +-- flat.rs          (brute-force SIMD — primary)
|   |   +-- cache.rs
|   +-- fuzzy/
|   |   +-- mod.rs
|   |   +-- ratio.rs         (re-export rapidfuzz::fuzz::ratio)
|   |   +-- partial_ratio.rs (our implementation)
|   |   +-- token_sort.rs    (our implementation)
|   |   +-- wratio.rs        (our implementation)
|   +-- scoring/
|   |   +-- mod.rs
|   |   +-- exact.rs
|   |   +-- fuzzy.rs
|   |   +-- embedding.rs
|   |   +-- composite.rs
|   +-- matching/
|   |   +-- mod.rs
|   |   +-- blocking.rs
|   |   +-- candidates.rs
|   |   +-- engine.rs
|   +-- crossmap/
|   |   +-- mod.rs
|   |   +-- local.rs
|   +-- state/
|   |   +-- mod.rs
|   |   +-- state.rs
|   |   +-- upsert_log.rs
|   |   +-- cache.rs
|   +-- batch/
|   |   +-- mod.rs
|   |   +-- engine.rs
|   +-- session/
|   |   +-- mod.rs
|   |   +-- session.rs
|   +-- api/
|       +-- mod.rs
|       +-- server.rs
|       +-- handlers.rs
+-- tests/
|   +-- config_test.rs       (golden file tests against Go configs)
|   +-- scoring_test.rs      (parity tests)
|   +-- fuzzy_test.rs        (golden pairs vs Python rapidfuzz output)
|   +-- integration_test.rs  (end-to-end batch run)
+-- bench/                   (reuse Go project's stress tests)
+-- testdata/
|   +-- fuzzy_golden.json    (golden string pairs + expected scores from Python)
```

---

## 9. Crate Dependencies

### Research status

All critical crates have been researched (API docs, GitHub issues, source code)
as of March 2026. Findings are embedded below. No dealbreakers found, but three
significant limitations required design adjustments (see §7).

### Decided

| Purpose | Crate | Version | Notes |
|---|---|---|---|
| Embedding inference | `fastembed` | 5.12.0 | Wraps `ort` 2.0.0-rc.11 + `tokenizers`. `EmbeddingModel::AllMiniLML6V2` is built-in (384-dim). `embed()` requires `&mut self` — one instance per pool slot, each behind `tokio::sync::Mutex`. Returns `Vec<Vec<f32>>`. Auto-downloads ONNX models to `~/.cache/`. No CoreML feature flag for ONNX models — uses CPU with ARM NEON on Apple Silicon. |
| Fuzzy matching (base) | `rapidfuzz` | 0.5.0 | Official Rust port by same author as Python rapidfuzz. **Only `fuzz::ratio` is implemented.** `partial_ratio`, `token_sort_ratio`, `wratio` are all missing (dormant since Dec 2023). Returns 0.0-1.0 (not 0-100). We use `ratio` as the primitive and implement the three missing scorers ourselves — see §7 Fuzzy Matching and §8 `fuzzy/` module. |
| HTTP server | `axum` | latest | Tokio-based, tower middleware ecosystem. |
| CLI | `clap` (derive) | latest | |
| Config | `serde` + `serde_yaml` | latest | |
| CSV | `csv` | latest | |
| JSON | `serde_json` | latest | |
| Concurrency | `dashmap` | 6.1.0 | `DashSet<K>` exists (wrapper around `DashMap<K, ()>`). Entry API available. **Caveats:** never hold `Ref` across `.await`; never call map methods inside iterator on same map; `len()` not atomic across shards. See §7 DashMap caveats. |
| Async runtime | `tokio` | latest | |
| Data parallelism | `rayon` | latest | For batch mode only. |
| Logging | `tracing` + `tracing-subscriber` | latest | Structured logging with spans. See §20. |
| Errors | `anyhow` + `thiserror` | latest | `thiserror` for library error enums, `anyhow` at application boundaries. See §18. |

### Deferred / Feature-flagged

| Purpose | Crate | Notes |
|---|---|---|
| Parquet | `parquet` (arrow-rs) | Behind `--features parquet`. Large dependency tree. |
| ANN index | `usearch` (v2.24) | Only if flat scan too slow at 100K+. Full CRUD, SIMD, concurrent insert+search, tombstone deletion. Requires C++ compiler. See §7 Vector Index. |
| Numpy compat | `ndarray-npy` | For cache import/export compatibility with Go-era `.npy` files. |

### Eliminated (with rationale)

| Crate | Why eliminated |
|---|---|
| `instant-distance` | No incremental insert, no deletion, no update. Immutable after build. Unusable for live mode. |
| `hnsw_rs` | No deletion. Requires mode-toggle between insert and search (not concurrent). Pure Rust is nice but feature gaps too large. |
| `tokenizers` (standalone) | Bundled inside `fastembed`. |
| `ort` (standalone) | Bundled inside `fastembed`. Pinned to `=2.0.0-rc.11`. Do not add a separate `ort` dependency — will cause version conflicts. |

### Version pinning note

`fastembed` pins `ort = "=2.0.0-rc.11"` (pre-release). This means:
- Do not add `ort` as a separate dependency in `Cargo.toml`
- If we ever need raw `ort` access (e.g., for custom execution providers),
  we must use the exact same version: `ort = "=2.0.0-rc.11"`
- When `ort` 2.0.0 stable ships, `fastembed` will likely update — track this

### Vector index: why flat-first

The Go version uses FAISS `IndexFlatIP` — also brute-force. The performance
gain comes from eliminating IPC, not from a better algorithm. A Rust flat scan
with SIMD on 10K x 384-dim vectors takes ~0.3ms. Adding HNSW complexity
(tuning M, ef_construction, ef_search; deletion semantics; serialization format)
is not justified until we have evidence of a bottleneck.

---

## 10. Model Provisioning

1. **`fastembed`** auto-downloads ONNX models (including `all-MiniLM-L6-v2`).
   Models are cached in `~/.cache/fastembed/` (configurable via env var).
2. **Manual:** `optimum-cli export onnx --model <name> ./onnx-model/`
   Then set `embeddings.model` to the local directory path.

---

## 11. Cache Management

Same staleness logic as Go. Cache files from Go version are **not compatible**
(different index format). `meld cache build` regenerates them.

Cache file formats:
- `.npy` — embedding vectors (N x D float32 matrix, numpy format)
- `.index` — serialized `VecIndex` (custom binary: header + flat f32 matrix +
  ID strings). Simple, fast to mmap.

Staleness detection: compare record count + hash of ID list against cached
metadata. If stale, rebuild.

---

## 12. What NOT to Port

- Python sidecar, SidecarClient, PipeClient
- Coordinator goroutine, 3-phase upsert (replaced by direct path — see §7)
- EmbeddingsA/B Go-side mirror (embeddings live in-process)
- JSON-lines IPC protocol
- `npy.go` loader (use `ndarray-npy`)
- RedisCrossMap (defer)
- `aUnmatched()` / `bUnmatched()` linear scans (replaced by DashSet — see §7)

---

## 13. Migration Phases

### Phase 1: Config + Data Loading + Scoring
Config loader/validator, CSV loader, exact/fuzzy/embedding scorers, composite.
**Deliverable:** Config validates. Record pairs can be scored.

### Phase 2: Encoder + Index
`fastembed` integration. Flat vector index. Cache management.
**Deliverable:** `meld cache build` works.

### Phase 3: Matching Engine
Blocking (with index), candidates, MatchTopN, CrossMap, batch engine.
**Deliverable:** `meld run` produces identical output to Go `match run`.

### Phase 4: Live Server
HTTP API (axum), session, upsert flow, WAL, unmatched tracking, blocking
index, CrossMap background flusher, encoder pool.
**Deliverable:** `meld serve` passes stress tests. Benchmarks vs Go.

### Phase 5: Polish
Cache CLI, review/crossmap CLI, Parquet/JSONL, graceful shutdown, docs.

---

## 14. Risks

### Confirmed (from crate research)

1. **`rapidfuzz-rs` missing scorers** — `wratio`, `partial_ratio`,
   `token_sort_ratio` do not exist. **Mitigated:** we implement them ourselves
   (~60 lines total) on top of `fuzz::ratio`. Algorithms are well-documented.
   Parity tested against Python golden outputs.

2. **`fastembed` `&mut self` on `embed()`** — cannot share a single instance
   across concurrent tasks. **Mitigated:** pool of N `Mutex<TextEmbedding>`
   instances. Each slot is independent. Model weights shared via OS mmap.

3. **`ort` pinned to pre-release** — `fastembed` pins `ort = "=2.0.0-rc.11"`.
   **Mitigated:** do not add separate `ort` dependency. Track fastembed releases
   for ort stable upgrade.

4. **No CoreML for ONNX models** — `fastembed` does not expose a CoreML
   execution provider for ONNX-based models (only for candle-based models).
   **Mitigated:** CPU inference with ARM NEON on Apple Silicon is still
   significantly faster than Python's GIL-constrained path. CoreML acceleration
   is a potential future win but not needed to hit targets.

### Remaining

5. **ONNX model compat** — validate `all-MiniLM-L6-v2` produces sensible
   embeddings early in Phase 2. Accept small float differences vs PyTorch
   if rankings are preserved.

6. **Score parity** — embedding scores will differ slightly (ONNX vs PyTorch).
   Fuzzy scores must match Python within ±0.001 (same algorithm, same math).
   Exact scores must be identical. Test with golden record pairs.

7. **Build complexity** — `ort` (via `fastembed`) needs ONNX Runtime shared
   library. On macOS this is straightforward (Homebrew or bundled). On Linux,
   may need `ORT_LIB_LOCATION` env var. Document in README.

8. **Parquet deps** — make optional feature flag to avoid bloating default build.

9. **Flat index at scale** — if deployments exceed 100K records, flat scan may
   become a bottleneck (~3ms at 100K). **Mitigated:** blocking index reduces
   effective pool size; `usearch` is a drop-in replacement behind same API.

10. **Encoder pool memory** — each additional `TextEmbedding` instance costs
    ~50MB runtime buffers (model weights shared via mmap). Default pool size 1
    is safe. Document the tradeoff.

11. **DashMap async footgun** — holding `Ref`/`RefMut` across `.await` will
    starve shard locks and effectively deadlock tokio. **Mitigated:** documented
    as hard rule in §7. Enforce via code review; consider clippy lint.

12. **`fastembed` returns `anyhow::Error`** — opaque error type, hard to match
    on programmatically. **Mitigated:** wrap in our `EncoderError::Inference`
    variant with the display string. Acceptable for our use case.

### Identified (from performance audit)

13. **VecIndex write-lock contention at scale** — every upsert takes a write
    lock on VecIndex (step 4) and every search needs a read lock (step 5).
    At high concurrency with 100K+ records, searches (~3ms under read lock)
    block inserts. **Mitigated at 10K-100K:** lock hold time is microseconds
    for insert, milliseconds for search. Acceptable. **At 1M+:** consider
    copy-on-write index or lock-free append buffer with periodic merge.

14. **WAL fsync throughput ceiling** — per-request fsync caps at ~10K-50K
    req/s on NVMe, less on SATA. **Mitigated:** default to buffered writes
    without per-request fsync. WAL is a recovery optimization; CrossMap CSV
    is the durability layer. Configurable for strict-durability deployments.

15. **Cold-start encoding at scale** — 100K records × 3ms/encode = 5 minutes.
    **Mitigated:** parallel pool encoding, batch size 256, progress logging.
    At 1M records, consider background encoding (serve immediately with
    degraded matching, encode in background, hot-swap index when ready).

---

## 15. Performance Targets

| Metric | Go+Python | Rust target |
|---|---|---|
| Sequential (c=1) | ~72-111 req/s | **400+** |
| Concurrent (c=10) | ~150 req/s | **1000+** |
| Encode latency | ~9-16ms | **3-5ms** |
| Machine util (c=10) | 13% | **>60%** |
| Startup (10K warm) | ~3s | **<1s** |
| Memory (10K) | ~1GB | **<500MB** |

### Memory budget (10K records, 384-dim embeddings)

| Component | Estimate | Notes |
|---|---|---|
| ONNX model (1 session) | ~80MB | Memory-mapped weights + runtime buffers |
| ONNX model (per extra session) | ~50MB | Weights shared via mmap, buffers are per-session |
| Records A (10K) | ~20MB | HashMap<String, HashMap<String, String>> |
| Records B (10K) | ~20MB | Same |
| VecIndex A (10K x 384) | ~15MB | 10K * 384 * 4 bytes + ID strings |
| VecIndex B (10K x 384) | ~15MB | Same |
| CrossMap | ~2MB | Two HashMaps, up to 10K entries each |
| BlockingIndex (2 sides) | ~5MB | HashMap<key, HashSet<id>> |
| Unmatched sets (2 sides) | ~2MB | DashSet<String> |
| DashMap overhead | ~10MB | Per-shard metadata |
| Tokio + axum runtime | ~10MB | |
| **Total (pool_size=1)** | **~180MB** | |
| **Total (pool_size=4)** | **~330MB** | |

Well under the 500MB target. At 100K records, multiply record/index components
by 10x → ~500MB (pool_size=1) to ~650MB (pool_size=4). Still manageable.

### Scalability ceilings

| Scale (per side) | Vector search | Memory | Startup (cold) | Bottleneck | Action needed |
|---|---|---|---|---|---|
| **10K** | 0.3ms flat | ~180MB | ~30s | None | Current design is optimal |
| **50K** | 1.5ms flat | ~350MB | ~2.5min | None | Pre-filtered search keeps effective scan small |
| **100K** | 3ms flat | ~500MB | ~5min | Flat scan if blocking selectivity is low | Evaluate `usearch`; parallel pool encoding |
| **500K** | 15ms flat | ~2.5GB | ~25min | Flat scan, memory, cold start | `usearch` required; background encoding on startup |
| **1M** | 30ms flat (unusable) | ~5GB | ~50min | Everything | `usearch` mandatory; mmap records; incremental startup |

**Key insight:** Pre-filtered search (§7) dramatically reduces effective scan
size. With blocking enabled (e.g., country code) and 60% unmatched pool, a
100K dataset may only scan ~18K vectors per query (~0.5ms) — well within budget.

**The design supports 100K per side without changes.** For 500K+, the three
changes needed are: (1) swap flat → `usearch`, (2) parallel/background startup
encoding, (3) WAL group-commit. All are designed as drop-in upgrades behind
existing APIs.

---

## 16. Reference: Key Go Source Files

Project root: `/Users/jude/Library/CloudStorage/Dropbox/Projects/match/`
Go module: `/Users/jude/Library/CloudStorage/Dropbox/Projects/match/match/`

| File | What | Lines | Phase |
|---|---|---|---|
| `internal/config/schema.go` | Config structs | 174 | 1 |
| `internal/config/loader.go` | Validate, defaults | 294 | 1 |
| `pkg/models.go` | Record, FieldScore, MatchResult | 68 | 1 |
| `internal/scoring/exact.go` | ExactScorer | 24 | 1 |
| `internal/scoring/embedding.go` | EmbeddingScorer + cosine | 58 | 2 |
| `internal/scoring/composite.go` | CompositeScorer + ScoreBatch | 349 | 1 |
| `internal/matching/blocking.go` | AND/OR blocking filter | 118 | 3 |
| `internal/matching/candidates.go` | Candidate generation | 193 | 3 |
| `internal/matching/engine.go` | MatchTopN, MatchOne | 358 | 3 |
| `internal/crossmap/local.go` | LocalCrossMap | 174 | 3 |
| `internal/state/upsertlog.go` | WAL | 226 | 4 |
| `session/session.go` | Live session + coordinator | 970 | 4 |
| `api/handlers.go` | HTTP handlers | 393 | 4 |
| `api/server.go` | HTTP server setup | 91 | 4 |
| `sidecar/pipe_main.py` | Python sidecar | ~800 | skip |

Other docs at project root:
- `DESIGN.md` — full Go+Python architecture
- `ARCHITECTURE-REVIEW.md` — bottleneck analysis, 4 rewrite options
- `OPTIMIZATIONS.md` — 8 completed optimizations, 4 to-do
- `match/CLAUDE.md` — symmetry invariant

### Running the Go version

```bash
cd /Users/jude/Library/CloudStorage/Dropbox/Projects/match/match
GONOSUMDB='*' GOFLAGS='-mod=mod' go build -o match ./cmd/match
./match validate --config bench/bench_live.yaml
./match serve --config bench/bench_live.yaml --port 8090

# Stress test (starts its own server on 8090)
cd bench && python live_stress_test.py --iterations 100
cd bench && python live_concurrent_test.py --concurrency 10 --iterations 1000

# Reset state before benchmarks
printf 'entity_id,counterparty_id\n' > bench/crossmap_live.csv
: > bench/live_upserts.ndjson
```

---

## 17. Deferred Features

- RedisCrossMap
- Absent-field handling (exclude vs score 0.0)
- Phonetic scorer
- Batch checkpointing
- TLS, Prometheus metrics
- Review workflow UI
- Excel loader
- Adaptive batching coordinator (see §7)
- HNSW vector index (see §7, §9)

---

## 18. Error Types

Use `thiserror` for typed errors within modules, `anyhow` at application
boundaries (CLI main, HTTP handlers).

```rust
/// Top-level error enum. Each module has its own error type that converts
/// into this via `From` impls.
#[derive(Debug, thiserror::Error)]
pub enum MatchrError {
    #[error("config error: {0}")]
    Config(#[from] ConfigError),

    #[error("data error: {0}")]
    Data(#[from] DataError),

    #[error("encoder error: {0}")]
    Encoder(#[from] EncoderError),

    #[error("index error: {0}")]
    Index(#[from] IndexError),

    #[error("crossmap error: {0}")]
    CrossMap(#[from] CrossMapError),

    #[error("session error: {0}")]
    Session(#[from] SessionError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing required field: {field}")]
    MissingField { field: String },
    #[error("invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },
    #[error("weights sum to {sum}, expected 1.0")]
    WeightSum { sum: f64 },
    #[error("parse error: {0}")]
    Parse(#[from] serde_yaml::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("file not found: {path}")]
    NotFound { path: String },
    #[error("missing id field '{field}' in {path}")]
    MissingIdField { field: String, path: String },
    #[error("duplicate id '{id}' in {path}")]
    DuplicateId { id: String, path: String },
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("model not found: {model}")]
    ModelNotFound { model: String },
    #[error("encoding failed: {0}")]
    Inference(String),
    #[error("pool exhausted (all {pool_size} encoders busy)")]
    PoolExhausted { pool_size: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum CrossMapError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("missing required field '{field}' in record")]
    MissingField { field: String },
    #[error("empty id in record")]
    EmptyId,
    #[error("encoder error: {0}")]
    Encoder(#[from] EncoderError),
}
```

### HTTP error responses

API handlers convert errors to JSON responses with appropriate status codes:

| Error type | HTTP status | Response body |
|---|---|---|
| `SessionError::MissingField` | 400 | `{"error":"missing required field 'entity_id' in record"}` |
| `SessionError::EmptyId` | 400 | `{"error":"empty id in record"}` |
| `EncoderError::*` | 500 | `{"error":"internal: encoding failed"}` |
| Other | 500 | `{"error":"internal server error"}` |

---

## 19. Startup and Shutdown Sequences

### Startup (`meld serve`)

```
1. Load and validate config
2. Initialize encoder pool (download model if needed — may take 30s+ first run)
3. Load dataset A → DashMap + sorted IDs    }  can run in
4. Load dataset B → DashMap + sorted IDs    }  parallel
5. Load or build A-side cache:              }  can run in
   a. If cache exists and not stale:        }  parallel
      load embeddings, build VecIndex       }  (after step 3/4)
   b. Else: batch-encode all records,       }
      save cache, build VecIndex            }
6. Load or build B-side cache (same logic)  }
7. Build BlockingIndex for both sides
8. Load CrossMap from CSV
9. Build unmatched sets (all IDs minus CrossMap entries)
10. Open WAL file
11. Replay WAL (last-write-wins for records; replay confirms/breaks for CrossMap)
12. Rebuild unmatched sets and blocking indices after WAL replay
13. Log startup summary: record counts, CrossMap entries, cache status
14. Start HTTP server
15. Log "ready" with listen address
```

Steps 3+4 can run in parallel. Steps 5+6 can run in parallel (after their
respective datasets are loaded).

**Batch encoding on cold start (steps 5b/6b):**

Cold-start encoding dominates startup time. Optimize with:

- **Batch size 256** (fastembed's default): encode 256 texts per ONNX call
  for maximum throughput. At ~3ms per text, 10K records ≈ 30s. At 100K ≈ 5min.
- **Parallel pool encoding:** If `encoder_pool_size > 1`, partition texts
  across all pool instances. With pool_size=4, 10K records ≈ 8s.
- **Progress logging:** Emit INFO log every 1000 records encoded with ETA.
- **Cache eagerly:** Write `.npy` as soon as encoding completes, before
  building VecIndex. Crash during VecIndex build can still reload embeddings.

| Records | Pool 1 | Pool 2 | Pool 4 | Notes |
|---|---|---|---|---|
| 10K | ~30s | ~16s | ~8s | Acceptable |
| 100K | ~5min | ~2.5min | ~1.3min | Progress bar essential |
| 1M | ~50min | ~25min | ~13min | Consider background encoding |

### Shutdown (SIGTERM / SIGINT)

```
1. Stop accepting new connections
2. Drain in-flight requests (timeout: 30s)
3. Flush WAL (ensure all buffered writes are on disk)
4. Compact WAL (deduplicate, keep latest per ID)
5. Flush CrossMap to CSV (final atomic write)
6. Save VecIndex to disk (if cache paths configured)
7. Log shutdown summary
```

Use `tokio::signal` for signal handling. Axum's `Server::with_graceful_shutdown`.

---

## 20. Observability

### Structured logging with `tracing`

Three log levels:

| Level | What |
|---|---|
| `INFO` | Startup/shutdown milestones, config summary, request counts |
| `DEBUG` | Per-request: endpoint, ID, score, classification, latency |
| `TRACE` | Per-field scores, candidate counts, blocking filter stats, encode timing |

### Spans

```
melder::serve                          (root span for server lifetime)
  melder::startup                      (init sequence)
    melder::load_dataset{side=a}
    melder::build_cache{side=a}
  melder::request{method=POST path=/api/v1/a/add}
    melder::upsert{side=a id=ENT-001}
      melder::encode{side=a}           (encode latency)
      melder::search{side=b k=10}      (vector search latency)
      melder::score{candidates=10}     (scoring latency)
```

### Metrics (deferred — Phase 5)

When Prometheus metrics are added, instrument:
- `melder_requests_total` (counter, by endpoint)
- `melder_request_duration_seconds` (histogram, by endpoint)
- `melder_encode_duration_seconds` (histogram)
- `melder_search_duration_seconds` (histogram)
- `melder_records_total` (gauge, by side)
- `melder_crossmap_entries` (gauge)
- `melder_encoder_pool_active` (gauge)

---

## 21. Testing Strategy

### Unit tests (per module, run with `cargo test`)

| Module | What to test | Approach |
|---|---|---|
| `config` | All 17 validation rules, defaults, legacy blocking normalization | Golden YAML files (valid + invalid). Use Go's `bench_live.yaml` as primary golden input. |
| `scoring/exact` | Case insensitive, trim, both-empty=0 | Table-driven: list of (a, b, expected_score) tuples. |
| `scoring/fuzzy` | wratio, partial_ratio, token_sort | Compare against Python `rapidfuzz` output for 20+ string pairs. |
| `scoring/composite` | Weight normalization, multi-field, embedding+fuzzy+exact mix | Mock encoder (returns fixed vectors). |
| `matching/blocking` | AND/OR, missing values skip, direction-aware | Synthetic record sets with known blocking keys. |
| `index/flat` | Insert, remove, search top-K, upsert (replace) | Random vectors, verify exact dot-product ordering. |
| `crossmap` | Add, remove, lookup, CSV round-trip, atomic write | Temp directory, verify file contents. |
| `state/upsert_log` | Append, replay, compact, crash recovery | Write entries, truncate file mid-line, verify replay handles it. |

### Integration tests (`tests/` directory)

1. **Config parity:** Load every YAML config from the Go project's `bench/`
   directory. Verify they parse and validate identically.

2. **Batch output parity:** Run `meld run` on a small dataset (100x100).
   Compare output CSV against Go `match run` output. Same records matched,
   same classifications. Scores within ±0.05 for embedding fields, exact
   match for fuzzy and exact fields.

3. **Live API parity:** Start `meld serve`, replay a sequence of 50
   upserts from the Go stress test. Compare response JSON shapes, status
   values, classification outcomes.

### Scoring parity tests

Embedding scores will differ between ONNX (Rust) and PyTorch (Python).
Acceptance criteria:
- Rank correlation (Spearman) > 0.99 for top-10 candidates
- Absolute score difference < 0.05 for any given pair
- Classification agreement > 99% (same auto/review/no_match)

### Stress tests (reuse Go project's)

- `bench/live_stress_test.py` — sequential, 100-1000 iterations
- `bench/live_concurrent_test.py` — concurrent, c=10/20, 1000 iterations

These require no modification — they test the HTTP API which is identical.
