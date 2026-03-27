---
type: overview
module: general
status: active
tags: [overview, index, onboarding]
---

# Melder — Project Overview & Index

_Single source of truth for onboarding. Read this at the start of every session.
Update it (concisely) when completing significant work._

Last updated: 2026-03-27 (Pipeline hooks + structured logging complete)

---

## What Melder Is and Why It Exists

Melder is a record matching engine: given two datasets A and B, it finds which records in B correspond to records in A. The canonical use case is a wholesale bank with a clean reference data master (A — counterparties, instruments, issuers) that receives daily vendor files (B — the same entities, but named inconsistently, abbreviated differently, or formatted to a different standard). The goal is to match each B record back to its A entity so that supplementary fields can be enriched from the master. This is called entity resolution or record linkage, and it is hard because names like "Goldman Sachs International" and "GS Intl Ltd" are the same entity to a human but very different strings to naive text comparison.

Melder's scoring pipeline is the core of the product. A match is not a binary decision — it is a composite score built from configurable per-field methods: exact equality for identifiers and codes, fuzzy string similarity for names (Levenshtein, token-sort, partial-match), BM25 for corpus-aware token scoring, and semantic embedding similarity for cases where the vocabulary gap is too large for string methods alone. Each field gets a weight; the composite score is the weighted average. Scores above `auto_match` (default 0.85) are confirmed automatically; scores in the review band are queued for a human; everything below is left unmatched. The same pipeline runs in all modes — batch and live — which means a score of 0.87 always means the same thing regardless of how it was produced.

Melder runs in two modes. **Batch mode** (`meld run`) processes a full B dataset against a full A dataset in one job, parallelised with Rayon across B records, with output to CSV. **Live mode** (`meld serve`) runs as an HTTP server: records are added to either side via API calls and matched immediately against whatever is on the opposite side, with results returned synchronously. Live mode is stateful — a WAL provides restart consistency for the in-memory backend; a SQLite backend provides full durability without a WAL.

The storage layer was a major engineering investment. At small-to-medium scale (up to a few million records), everything lives in memory (DashMap-backed MemoryStore). At extreme scale — 55M×4.5M was the design target — a columnar SQLite backend (SqliteStore) drops the memory footprint from ~100GB to ~10-12GB by streaming records from disk during scoring. Both backends implement the same `RecordStore` and `CrossMapOps` traits, so the matching pipeline is completely unaware of which is in use. This abstraction also means adding a third backend (e.g. an external database) would only require implementing the traits and wiring a new startup path.

**EXPERIMENT 12 COMPLETE — PRODUCTION CONFIGURATION FINALIZED.** The embedding fine-tuning campaign has concluded with a definitive production recommendation: **Arctic-embed-xs R22 + 50% BM25 + synonym 0.20** (name_emb=0.30, addr_emb=0.20, bm25=0.50, synonym=0.20, additive). This configuration achieves **zero overlap** (0.0003) between matched and unmatched populations, **100% combined recall** (1 missed clean + 1 missed ambiguous), and **zero false positives** in both auto-match and review. The progression from Experiment 1 to Experiment 12 reduced overlap by **560×** (0.168 → 0.0003) through systematic experimentation. See §11 for full details and [[Training Experiments Log#Experiment 12]].

Binary name: `meld`. Crate name: `melder`. Single Rust crate, no workspace.

---

## 1. Build & Test

```bash
cargo build --release --features usearch                  # standard production build
cargo build --release --features usearch,parquet-format   # all features
cargo build                                               # debug
cargo test --all-features         # REQUIRED before committing
cargo fmt -- --check && cargo clippy --all-features       # lint gate
```

Feature flags: `usearch` (HNSW ANN), `parquet-format` (Parquet I/O), `bm25` (Tantivy BM25), `simd` (SimSIMD hardware-accelerated dot product). Mode enum: `Match` (two-sided matching) or `Enroll` (single-pool entity resolution).

---

## 2. Inviolable Principles

Full text: `vault/architecture/CONSTITUTION.md`. Violating these is a bug regardless of intent.

**1. Batch Asymmetry, Live Symmetry.** Batch: B is query side, A is reference pool. Live: A and B are fully symmetric — identical struct, API, and logic.

**2. One Scoring Pipeline.** All matching — batch, live upsert, try-match — flows through `src/matching/pipeline.rs::score_pool()`. No second scoring code path exists.

**3. CrossMap Bijection (1:1 Under One Lock).** Every A maps to at most one B, enforced atomically in `src/crossmap/` via a single `RwLock`. `claim()` checks both directions before inserting. Two DashMaps were rejected (TOCTOU gap).

**4. Combined Vector = Weighted Cosine Identity.** `dot(C_a, C_b) = Σ(w_i × cos(a_i, b_i))` by scaling each L2-normalised field vector by `sqrt(w_i)` before concatenation (`src/vectordb/mod.rs`). `decompose_emb_scores()` in `pipeline.rs` reverses it — no second ONNX call ever needed.

---

## 3. Source Code Map

### Top-level files

| File | Role |
|---|---|
| `src/main.rs` | CLI entry — clap parsing + dispatch to `cli::*` only, no business logic |
| `src/lib.rs` | `pub mod` declarations only, alphabetical, no logic |
| `src/error.rs` | All error types: `MelderError`, `ConfigError`, `DataError`, `EncoderError`, `IndexError`, `CrossMapError` |
| `src/models.rs` | Core types: `pub type Record = HashMap<String, String>`, `Side`, `Classification`, `MatchResult`, `Mode` (Match/Enroll) |

### `src/config/`

| File | Key items |
|---|---|
| `schema.rs` | All config structs: `MelderConfig`, `DatasetConfig`, `EmbeddingConfig`, `MatchField`, `BlockingConfig`, `ExactPrefilterConfig`, `ThresholdConfig`, `PerformanceConfig`, `LiveConfig`, `BatchConfig`, `Mode` enum (Match/Enroll) |
| `loader.rs` | `load_config(path)` — parses YAML, validates all fields, returns `MelderConfig`; `load_enroll_config()` for enroll mode |
| `enroll_schema.rs` | `EnrollConfig` — simplified schema for single-pool mode (single `field:` instead of `field_a:`/`field_b:`, single `dataset:`, no crossmap) |

### `src/matching/`

| File | Key items |
|---|---|
| `pipeline.rs` | **`score_pool()`** — the one scoring entry point used by all modes. Also `decompose_emb_scores()` |
| `blocking.rs` | `BlockingIndex` — HashMap keyed by (field_index, value); AND/OR query modes |
| `candidates.rs` | `get_candidates()` — vector ANN search + blocked-record fallback |

### `src/scoring/`

| File | Key items |
|---|---|
| `mod.rs` | `score_pair(a, b, fields, ...)` — dispatches to per-method scorers; computes weighted composite |
| `exact.rs` | Unicode case-insensitive equality; empty → 0.0 |
| `embedding.rs` | Cosine similarity; negative cosine clamped to 0.0 |

### `src/fuzzy/`

| File | Scorer | Note |
|---|---|---|
| `wratio.rs` | `wratio` | `max(ratio, token_sort, partial_ratio)` — default for entity names |
| `ratio.rs` | `ratio` | Normalised Levenshtein |
| `partial_ratio.rs` | `partial_ratio` | Shorter slides over longer; good for substrings |
| `token_sort.rs` | `token_sort_ratio` | Sort tokens before comparing; word-order agnostic |

`partial_ratio` and `token_sort_ratio` are implemented from scratch in Melder (rapidfuzz-rs only exposes `ratio`). All use `.chars()` not bytes — correct for multi-byte UTF-8.

### `src/encoder/`

| File | Key items |
|---|---|
| `mod.rs` | `EncoderPool` — pool of ONNX sessions (`encoder_pool_size` slots, Mutex per slot). Detects local paths by heuristic (absolute, `./`, `../`, `.onnx` suffix, or resolves on disk) and loads via fastembed `UserDefinedEmbeddingModel`. Remote models use named fastembed download. Output dim auto-detected from `config.json::hidden_size`, default 384. |
| `coordinator.rs` | Optional batch coordinator — collects encode requests within `encoder_batch_wait_ms` window, dispatches as single ONNX batch. Off by default; only helps at c≥20 with large models. |

### `src/vectordb/`

| File | Key items |
|---|---|
| `mod.rs` | `VectorDB` trait: `insert`, `search`, `remove`, `contains`, `build_or_load_combined_index`. Combined vector construction with `sqrt(w)` scaling. |
| `flat.rs` | `FlatVectorDB` — O(N) brute-force cosine scan. Dev/small datasets only. |
| `usearch_backend.rs` | `UsearchVectorDB` — HNSW ANN, feature-gated. `load` vs `mmap` mode (`vector_index_mode` config). mmap = read-only, OS-managed paging, batch only. |
| `manifest.rs` | Manifest sidecar (`.manifest` file) — records model name, spec hash, blocking hash. Layer 2 of cache invalidation. |
| `texthash.rs` | `TextHashStore` — FNV-1a hash per record's embedding text. Skip ONNX re-encode if hash matches. Layer 3 of cache invalidation. 20% live throughput gain. |

Three-layer cache invalidation: (1) spec-hash in index filename (field names+weights+quantization), (2) manifest sidecar (model+blocking hash), (3) per-record text-hash diff.

### `src/crossmap/`

`CrossMapOps` trait with two implementations:
- `MemoryCrossMap` — single `RwLock<CrossMapInner>` (two plain HashMaps). `flush()` saves to CSV via stored `FlushConfig`.
- `SqliteCrossMap` — UNIQUE constraints on `a_id` and `b_id`; DELETE+INSERT for bijection; `flush()` is a no-op (write-through).

`claim()` checks both directions atomically under the write lock before inserting.

### `src/store/` (RecordStore trait)

`RecordStore` trait — 18+ methods: records (insert/get/remove/contains/len/iter), blocking index, unmatched sets, common_id index, review persistence.

| Implementation | Backing | Use case |
|---|---|---|
| `MemoryStore` | DashMap | Default — fast, all records in RAM |
| `SqliteStore` | SQLite, columnar (one column per field) | Million-scale batch or durable live mode |

SQLite connection pool: 1 writer (`Mutex<Connection>`) + N read-only (`SqliteReaderPool`, round-robin `try_lock`). Config: `sqlite_read_pool_size` (default 4), `sqlite_pool_worker_cache_mb` (default 128). Schema generated dynamically from config `required_fields`. `bulk_load()` uses single-transaction inserts with deferred index creation.

### `src/batch/`

| File | Key items |
|---|---|
| `engine.rs` | `run_batch()` — Rayon-parallelised main loop. Phases: exact prefilter → common ID → blocking → BM25 filter → ANN candidates → full scoring → classify → CrossMap claim. |
| `writer.rs` | Writes `results.csv`, `review.csv`, `unmatched.csv` |

### `src/session/` and `src/state/`

| File | Key items |
|---|---|
| `session/mod.rs` | `Session` — wraps two `LiveSideState` (match mode) or one `LiveMatchState` (enroll mode); `upsert()`, `try_match()`, `remove()`, `enroll()`, `enroll_batch()` |
| `state/live.rs` | `LiveMatchState` — `Arc<dyn RecordStore>` + `Box<dyn CrossMapOps>` + BM25 index + vector index. `load()` dispatches to `load_memory()`, `load_sqlite()`, or `load_enroll()` (single-pool, A-side only, no crossmap). Zero backend awareness at runtime — all backend ops go through trait methods. |
| `state/upsert_log.rs` | WAL — append-only log of upsert/remove events. Replayed on memory-backend startup. Skipped for SQLite (durable by construction). Compaction creates timestamped snapshots. |

### `src/api/`

| File | Key items |
|---|---|
| `server.rs` | Axum router setup, `Arc<Session>` state, graceful shutdown with `tokio::select!` on SIGTERM/Ctrl-C. Conditional router: match-mode endpoints vs enroll-mode endpoints mounted based on config mode. |
| `handlers.rs` | All HTTP handlers — add, remove, try-match, batch endpoints, crossmap, review, unmatched (match mode); enroll, enroll-batch, enroll/remove, enroll/query, enroll/count (enroll mode). Errors mapped to `StatusCode` + JSON. Full endpoint list: `vault/architecture/API Reference.md` |

### `src/cli/`

One file per subcommand: `run.rs`, `serve.rs`, `validate.rs`, `tune.rs`, `cache.rs`, `review.rs`, `crossmap.rs`, `export.rs`. Entry points are `cmd_*` functions. CLI errors use `match` + `eprintln!` + `process::exit(1)` — no `?`.

### `src/data/`

`csv.rs`, `jsonl.rs`, `parquet.rs` (parquet feature-gated). Each exposes `load_*()` (reads all into Vec) and `stream_*()` (chunked callback for SQLite batch mode).

---

## 4. Pipeline Flow

### Batch per B record (Rayon, `src/batch/engine.rs`)

1. **Exact prefilter** (`exact_prefilter` config) — O(1) hash lookup against pre-built index on A side. All configured field pairs must match exactly (AND). If all match → auto-confirm at 1.0, skip all remaining phases. Runs _before_ blocking — recovers cross-block matches (e.g. same LEI, different country code).
2. **Common ID pre-match** (`common_id_field`) — exact match on single shared ID field → auto-confirm at 1.0.
3. **CrossMap skip** — skip B records already confirmed.
4. **Blocking** (`src/matching/blocking.rs`) — `BlockingIndex` lookup; AND/OR modes. In enroll mode, `blocking_query()` takes `pool_side` parameter to exclude self-matches when query_side == pool_side.
5. **BM25 candidate filter** (optional) — Tantivy BM25 re-ranks blocked candidates; retains `bm25_candidates` top results.
6. **ANN candidate selection** (`src/matching/candidates.rs`) — searches combined embedding index for `top_n` nearest A neighbours. If no embedding fields configured, all blocked records pass through.
7. **Full scoring** (`src/scoring/mod.rs::score_pair()`) — all `match_fields` scored; embedding cosines decomposed from combined vectors.
8. **Classification** — `>= auto_match` → Auto; `>= review_floor` → Review; else NoMatch.
9. **CrossMap claim** — atomic; falls through to next candidate if A already claimed (match mode only).

### Live upsert (`src/session/mod.rs`, `src/api/handlers.rs`)

1. Parse JSON record.
2. Encode via `EncoderPool` — skip if text-hash unchanged (`src/vectordb/texthash.rs`).
3. Store in `RecordStore`, upsert combined vector index, update blocking index. BM25 index marked dirty (committed lazily before opposite-side query).
4. `score_pool()` against opposite side.
5. Claim CrossMap if auto-match.
6. WAL append (memory backend only).
7. Return JSON matches.

**Try-match** (`/a/try`, `/b/try`): same flow, read-only — no store, no WAL.

### Startup paths (`src/state/live.rs::load()`)

**Match mode:**
- `live.db_path` absent → `load_memory()`: parse CSVs, build indices, optionally replay WAL.
- `live.db_path` set, DB not found → `load_sqlite()` cold: create DB, stream CSVs into SqliteStore, build indices.
- `live.db_path` set, DB found → `load_sqlite()` warm: open existing DB (records + crossmap already durable), load reviews, skip WAL.

**Enroll mode:**
- `load_enroll()`: single-pool startup (A-side only, no B-side, no crossmap). Optional pre-load from `dataset.path`. Indices built once; records added via API.

---

## 5. Scoring Quick Reference

**Composite**: `Σ(field_score × weight) / Σ(weight)`. Weights auto-normalised — ratios matter, not absolute values. Both fields empty → 0.0. One field empty → 0.0.

**Classification thresholds**: both inclusive (≥). `auto_match` default 0.85, `review_floor` default 0.60.

| Method | File | Notes |
|---|---|---|
| `exact` | `src/scoring/exact.rs` | Unicode case-insensitive equality |
| `ratio` | `src/fuzzy/ratio.rs` | Normalised Levenshtein |
| `partial_ratio` | `src/fuzzy/partial_ratio.rs` | Best window match; substring-tolerant |
| `token_sort_ratio` | `src/fuzzy/token_sort.rs` | Sort tokens first; word-order agnostic |
| `wratio` | `src/fuzzy/wratio.rs` | max of all three above; default fuzzy scorer |
| `embedding` | `src/scoring/embedding.rs` | Cosine similarity; negative → 0.0 |
| `numeric` | `src/scoring/mod.rs` | Parse f64, equality only |
| `bm25` | `src/bm25/` (feature-gated) | Tantivy-backed, normalised by analytical self-score |

---

## 6. Key Config Fields

Full schema: `vault/architecture/Config Reference.md`. Most-used fields:

```yaml
datasets:
  a: { path, id_field, format }        # format: csv | jsonl | parquet
  b: { path, id_field, format }

embeddings:
  model: all-MiniLM-L6-v2              # HuggingFace name OR local dir path with model.onnx
  a_cache_dir / b_cache_dir
  quantized: false                     # true = 2× encoding speed, negligible quality loss

match_fields:
  - { field_a, field_b, method, weight }

blocking:
  enabled: true
  operator: and | or
  fields: [ {field_a, field_b} ]

exact_prefilter:
  enabled: true
  fields: [ {field_a, field_b} ]       # all must match (AND) → score 1.0

bm25_fields: [ {field_a, field_b} ]    # optional; derived from fuzzy/embedding fields if absent

thresholds: { auto_match: 0.85, review_floor: 0.60 }

performance:
  encoder_pool_size: 4
  vector_index_mode: load | mmap       # mmap = read-only, batch only, OS-paged
  vector_quantization: f32 | f16 | bf16
  encoder_batch_wait_ms: 0             # >0 only at c≥20 with large models

top_n: 5          # ANN candidates per B record
ann_candidates: 50
bm25_candidates: 20
min_score_gap: 0.0

live:   { db_path }   # omit = memory mode
batch:  { db_path }   # omit = memory mode; SQLite deleted after run
```

---

## 7. Performance Baselines (M3 MacBook Air, all-MiniLM-L6-v2, pool_size=4)

Full tables: `vault/architecture/Performance Baselines.md`

| Configuration | Throughput |
|---|---|
| Batch, usearch, 10k×10k (warm) | 33,738 rec/s |
| Batch, BM25-only, 10k×10k | 49,337 rec/s (fastest) |
| Batch, usearch+BM25, 10k×10k | 19,034 rec/s |
| Batch, usearch, 100k×100k (warm) | 10,539 rec/s |
| Batch, SQLite columnar, 10k×10k | 1,420 rec/s (~10-12GB RAM) |
| Live, usearch, 10k×10k warm (c=10) | 1,558 req/s, p95 25.6ms |
| Live, SQLite, 10k×10k warm (c=10) | 1,395 req/s, p95 13.6ms, 4× faster warm start |

Production: usearch backend; `quantized: true` (2× encoding, negligible quality loss); `vector_quantization: f16` (43% smaller cache). Batch endpoint sweet spot: size 50 (445 req/s, 1.8× vs single).

Accuracy (10k×10k, embeddings + exact prefilter): precision 93.3%, recall vs ceiling 92.8%, 441 FP, blocking ceiling 6,863 of 7,000.

---

## 8. Benchmarks Folder

```
benchmarks/
  data/
    generate.py              Synthetic dataset generator.
                             Key functions:
                               generate_a_with_seed(seed, n, include_addresses, out_dir)
                               generate_b_from_master(master_a_path, b_seed, n, include_addresses, out_dir)
                               generate_with_seed(seed, n, include_addresses, out_dir)  ← A+B together
                             70% matched / 20% ambiguous / 10% unmatched split.
                             B records carry hidden fields _true_a_id and _match_type for eval.

  accuracy/
    eval.py                  Standalone accuracy evaluator (uses _true_a_id / _match_type)
    10kx10k_embeddings/      Accuracy benchmark: embeddings-only config
      config.yaml, run_test.py, output/, cache/
    10kx10k_bm25/            Accuracy benchmark: BM25 config
    10kx10k_combined/        Accuracy benchmark: embeddings + BM25 combined
    training/                Fine-tuning loop (see Section 11)

  batch/
    run_all_tests.py         Runs all batch benchmark configs sequentially
    10kx10k_flat/            Batch benchmark: flat backend, 10k×10k
    10kx10k_usearch/         Batch benchmark: usearch, 10k×10k — warm/ and cold/ subdirs
    10kx10k_usearch_bm25/    Batch benchmark: usearch + BM25
    10kx10k_bm25only/        Batch benchmark: BM25-only
    10kx10k_bm25only_sqlite/ Batch benchmark: BM25-only with SQLite store
    100kx100k_usearch/       100k×100k usearch
    100kx100k_usearch_quantized/
    100kx100k_usearch_f16/
    100kx100k_usearch_mmap/  mmap vector index mode
    1Mx1M_bm25only/          1M×1M BM25-only in-memory

  live/
    run_all_tests.py         Runs all live benchmark configs
    sqlite_cache_sweep.py    Sweeps SQLite cache_mb values
    10kx10k_inject3k_flat/   Live benchmark: flat, 10k pre-loaded, 3k injected
    10kx10k_inject3k_usearch/
    10kx10k_inject3k_usearch_sqlite/
    100kx100k_inject10k_usearch/
    1Mx1M_inject10k_usearch/

  scripts/
    smoke_test.py            Quick sanity check — batch + live, small dataset
    live_stress_test.py      High-concurrency live mode stress test
    live_concurrent_test.py  Concurrent request benchmark
    live_batch_test.py       Batch-endpoint throughput (sizes 1–1000)
    cpu_monitor.py           CPU/memory monitoring during benchmarks

  experiments/
    columnar_sqlite/         Python experiment that validated columnar vs JSON-blob SQLite
                             (confirmed 2.3× speedup before implementing in Rust)
```

Each benchmark subdirectory typically contains: `config.yaml` (melder config), `run_test.py` (test runner), `output/` (gitignored results).

---

## 9. Code Style (Enforced — No Exceptions)

**Imports**: three blank-line-separated groups: (1) `std`, (2) external crates, (3) `crate::`/`super::`. Alphabetical within groups.

**Naming**: modules/files `snake_case`; structs/enums `CamelCase`; functions `snake_case`; constants `SCREAMING_SNAKE_CASE`; CLI commands `cmd_`-prefixed; test helpers `make_`-prefixed. Two-letter acronyms uppercase (`DB`), longer ones title-case (`CrossMap`).

**Errors**: `thiserror` for typed domain errors; `anyhow` for ad-hoc context at call-site boundaries. Functions return `Result<T, SpecificError>`, not `anyhow::Result`. Top-level `MelderError` has `#[from]` for all module errors. Never `unwrap()` except lock poison: `.unwrap_or_else(|e| e.into_inner())`. `expect()` only for truly impossible failures.

**Derives**: config structs `#[derive(Debug, Deserialize)]` + `#[serde(default)]`; API responses `#[derive(Debug, Serialize)]`; domain enums `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]`. Serde: `rename_all = "snake_case"`, `skip_serializing_if = "Option::is_none"`. Struct fields always `pub` — no builder pattern.

**Structure**: `mod.rs` files contain only `pub mod` and `pub use` — no logic. Business logic in named files. `lib.rs` contains only module declarations.

**Formatting**: standard rustfmt defaults (4-space indent, K&R braces, trailing commas everywhere). `// ---` banner comments to separate logical sections in long files.

**Docs**: `//!` on every file (1-3 lines); `///` on public functions (imperative voice); `//` for non-obvious logic. No doc comments on private functions or test helpers.

**Logging**: `tracing` crate, structured key-value: `info!(side = s, id = %id, "add")`. Only `info!` and `warn!`. `eprintln!()` for build-time progress, not tracing.

**Async**: `main.rs` synchronous; `tokio::runtime::Runtime::new()` created manually. CPU-bound → `tokio::task::spawn_blocking`. Axum handlers return `axum::response::Response`. State via `Arc<Session>` + `Router::with_state()`.

**Feature flags**: `#[cfg(feature = "...")]` on modules, functions, tests, and match arms. Currently: `usearch`, `parquet-format`, `bm25`, `simd`.

**Tests**: `#[cfg(test)] mod tests` at bottom of each source file. No integration test directory — everything in-crate. Table-driven for scorers `(input, expected)`. `make_` helpers. `assert!` with messages. `tempfile::tempdir()` for filesystem. `macro_rules!` for generic test suites. No external test frameworks.

---

## 10. Synthetic Fine-Tuning Loop

Entry point: `python benchmarks/accuracy/training/run.py --rounds 5` (from project root).
Full docs: `benchmarks/accuracy/training/README.md`, `vault/architecture/Training Loop.md`.

```
benchmarks/accuracy/training/
  run.py            Main orchestration — only entry point
  evaluate.py       Ground-truth classification using _true_a_id
  pairs.py          Extracts (a_text, b_text, label) training pairs
  finetune.py       sentence-transformers training (MPS/CUDA/CPU auto-detect)
  export.py         ONNX export via optimum-cli
  plot.py           Learning curve charts
  config.yaml       Base melder config template (paths substituted per round)
  requirements.txt  optimum[onnxruntime], datasets, accelerate, pyyaml, matplotlib, pandas
  master/           Fixed A-side reference master (generated once at seed=0, gitignored)
  holdout/          Fixed holdout B file (seed=9999, against master A, gitignored)
  rounds/round_N/   metrics.json + pairs.csv (committed); dataset_b.csv + output/ (gitignored)
  models/round_N/   model.onnx (gitignored, ~86MB each)
  results/          metrics.csv + learning_curve.png (committed)
```

### Loop structure (Attempt 2 — fixed A master design)

```
Setup (once):
  generate_a(seed=0)        →  master/dataset_a.csv      ← fixed for all rounds
  generate_b(seed=9999, A)  →  holdout/dataset_b.csv     ← fixed holdout

Round 0:  run base model on fresh B (seed=100) → evaluate train+holdout → extract pairs
Round N:  fine-tune(base + all pairs 0..N-1) → modelₙ → run on fresh B (seed=100+N) → evaluate
```

**Why fixed A master**: holdout B is a noisy variant of the same A entities used in training — tests noise-handling generalisation, not entity name memorisation.

### First attempt (failed)

Fully independent seeds 0–4 (train) vs 9999 (holdout). Different seeds = non-overlapping entity universes. Model memorised seeds 0–4 (training recall → 99%). Holdout collapsed to zero by round 3. Root cause: holdout design flaw.

### Second attempt (ready to run)

Already implemented: `generate_a_with_seed()` + `generate_b_from_master()` in `benchmarks/data/generate.py`; `run.py` fully rewritten; `src/encoder/mod.rs` supports local ONNX paths.

```bash
cargo build --release --features usearch
pip install -r benchmarks/accuracy/training/requirements.txt
python benchmarks/accuracy/training/run.py --rounds 5
```

Flags: `--rounds N`, `--size 10000`, `--seed-offset 100`, `--epochs 3`, `--batch-size 32`, `--resume-from N`, `--meld-binary ./target/release/meld`.

### Training pair labels

`CosineSimilarityLoss` with continuous labels: matched → 1.0, ambiguous → 0.7, FP hard negatives → 0.0.

### Timing (M3, 10k×10k)

~17s meld + ~2.1 min fine-tune per round + ~6s ONNX export. 5 rounds ≈ 25 minutes.

---

## 11. Current State

### Completed

**Experiment 8: BGE-small + LoRA + batch=128 + MNRL, 18 rounds** — Confirmed that batch size affects training signal but not capacity ceiling. Best overlap 0.070 at R12 (vs exp 2's 0.081 at R7 with batch=32). The 384-dim embedding space is the real bottleneck. Practical stopping point: R8 (overlap 0.078, recall 98.7%) for production use. Combined with BM25 at 20%, BGE-small could be viable in production with ~2× faster encoding. See [[Training Experiments Log#Experiment 8]] and [[BGE Small Training Results]].

**Experiment 9: Snowflake Arctic-embed-xs + LoRA + batch=128 + MNRL, 23 rounds** — **KEY DECISION: Arctic-embed-xs is the new recommended embedding model.** Best overlap 0.031 at R22 — best of any experiment, beating BGE-base (110M, 0.046) and BGE-small (33M, 0.070). Combined recall 99.7% from R14 onward (best of any trained model, and improved during training). Only 30 missed matches at R22 (19 clean + 11 heavy noise). Converged cleanly R17-R22 with no regression. Review FPs: 2,826 → 184 at R22 (93.5% reduction). Zero missed matches R2-R7 — briefly achieved perfect recall before overlap improvement phase. Arctic-embed-xs (22M, 6 layers) is optimal: best quality at smallest size and fastest speed. Pre-training quality (400M samples with hard negative mining) matters more than parameter count. Fewer layers = proportionally larger LoRA intervention. Arctic stretches (pushes non-matches down while keeping matches stable); BGE-small compresses. The 0.031 embedding-only overlap should drop to near-zero with BM25 (experiment 10 next). See [[Training Experiments Log#Experiment 9]] and [[Arctic Embed XS Training Results]].

**Experiment 10: Arctic-embed-xs R22 + BM25 50%** — Tested BM25 weight tuning to suppress residual false matches. BM25 at 50% eliminated overlap entirely (0.0003), achieving zero false positives in both auto-match and review. Combined recall 100% (1 missed clean + 1 missed ambiguous). This is the FINAL recommended production configuration. See [[Training Experiments Log#Experiment 10]].

**Experiment 11: Arctic-embed-xs R22 + fuzzy wratio + name:addr ratio tuning** — Tested alternative approaches to suppress residual false matches. wratio fuzzy on name (0.10) achieved no improvement (overlap 0.0011 vs exp 10's 0.0003). 75:25 name:addr ratio made things worse (overlap 0.0032, collateral damage to acronym matches). BM25 remains the superior approach. See [[Training Experiments Log#Experiment 11]].

**Experiment 12: Arctic-embed-xs R22 + weight tuning (final validation)** — Confirmed the production configuration through systematic weight tuning. Three approaches tested: (1) wratio fuzzy on name (0.10): overlap 0.0011 — no improvement; (2) 75:25 name:addr ratio: overlap 0.0032 — made things worse; (3) BM25 50%: overlap **0.0003** — eliminated overlap entirely. **FINAL PRODUCTION CONFIGURATION: Arctic-embed-xs R22 + 50% BM25 + synonym 0.20** (name_emb=0.30, addr_emb=0.20, bm25=0.50, synonym=0.20, additive). Overlap 0.0003, combined recall 100%, zero false positives. 22M params, 6 layers — fastest encoding. Progression from exp 1 to exp 12: overlap 0.168 → 0.0003 (560× improvement). See [[Training Experiments Log#Experiment 12]].

**Enroll endpoint for single-pool entity resolution** — New `mode: enroll` with 5 HTTP endpoints: `POST /api/v1/enroll`, `POST /api/v1/enroll-batch`, `POST /api/v1/enroll/remove`, `GET /api/v1/enroll/query`, `GET /api/v1/enroll/count`. New `EnrollConfig` serde schema in `src/config/enroll_schema.rs`. New `Mode` enum (Match/Enroll) in `src/config/schema.rs`. `load_enroll_config()` in `src/config/loader.rs`. `Session::enroll()` and `Session::enroll_batch()` methods. Self-match exclusion in `pipeline.rs` when query_side == pool_side. `blocking_query()` gains `pool_side` parameter for same-side blocking. `LiveMatchState::load_enroll()` for single-pool startup (A-side only, no crossmap). Conditional router: enroll endpoints only mounted in enroll mode. 6 new tests for enroll config parsing. 382 tests pass, zero clippy warnings. See [[Enroll Endpoint Design]].

**Structured logging** — Converted ~70 `eprintln!` calls in live-path files to `tracing` (info!/warn!). `--log-format json` now produces clean structured JSON for all server output including startup, encoding, WAL, shutdown. CLI-only commands (run, tune, etc.) kept as `eprintln!`.

**HuggingFace Hub model download** — Added `hf-hub` dependency. `EncoderPool::new()` auto-downloads models with `/` in the name from HuggingFace Hub (e.g. `themelder/arctic-embed-xs-entity-resolution`). Local paths detected by heuristic (absolute, `./`, `../`, `.onnx` suffix, or resolves on disk).

**Pipeline hooks** — Single long-running subprocess receiving NDJSON events on stdin. Config: `hooks: { command: "python hook.py" }` in both match and enroll mode configs. 4 event types: `on_confirm`, `on_review`, `on_nomatch`, `on_break`. `HookEvent` enum in `src/hooks/mod.rs` with custom JSON serialization. Hook writer task in `src/hooks/writer.rs` with subprocess lifecycle, exponential backoff respawn (1s-60s), disable after 5 consecutive failures. Non-blocking: scoring thread uses `try_send` on mpsc channel (~10ns), dedicated writer task handles pipe I/O. 6 injection points in `src/session/mod.rs` (2× on_confirm auto, 1× on_confirm manual, 1× on_review, 1× on_nomatch, 1× on_break). Platform dispatch: `sh -c` on Unix, `cmd /C` on Windows. Validation: empty command rejected. 392 tests pass, zero clippy warnings. Full docs page at `docs/hooks.md` with example Python script. Design spec at `HOOK_DESIGN.md`.

### In Progress

**CI/CD** — `.github/workflows/ci.yml` + `release.yml` created (macOS ARM, Linux glibc x86_64, Windows MSVC). Requires GitHub remote to activate. Homebrew/Scoop auto-update hooks not yet wired.

### Backlog (ranked)

1. Single-artifact deployment — `include_bytes!()` on ONNX weights.
2. External vector DB — Qdrant/Milvus (VectorDB trait is the interface).
3. BM25 Mutex → RwLock for BM25-heavy batch at scale.
4. Benchmark data regeneration script.

---

## 12. Key Decisions Summary

Full rationale for all decisions: `vault/decisions/Key Decisions.md`. Check before repeating a previously-rejected approach. Also check `vault/ideas/Discarded Ideas.md`.

| Decision | Choice |
|---|---|
| Combined vector index | One index per side (concat sqrt(w)×fields) — one ANN query, no quality loss |
| CrossMap locking | Single `RwLock<CrossMapInner>` — two DashMaps can't atomically check both directions |
| Text-hash skip | FNV-1a; skip ONNX if unchanged — 20% live throughput gain |
| Three-layer cache invalidation | Spec-hash filename + manifest sidecar + text-hash diff |
| RecordStore + CrossMapOps traits | Decouples pipeline from storage; MemoryStore + SqliteStore are the two impls |
| Columnar SQLite | One column per field — 2.3× faster candidate lookups vs JSON blob |
| SQLite connection pool | 1 writer + N readers (round-robin try_lock) |
| BM25 commit batching | `dirty` flag; commit only before opposite-side query — 2× live throughput |
| Exact prefilter | Pre-blocking field-pair confirmation — O(1) hash; recovers cross-block matches |
| Local ONNX encoder paths | Path heuristic → `UserDefinedEmbeddingModel` — fine-tuned models plug in directly |
| Linux CI target | glibc not musl — fastembed → openssl-sys incompatible with musl cross-compilation |
| mmap vector index | `vector_index_mode: mmap` via usearch `view()` — OS paging at 100M+ records; read-only |
| Encoding coordinator off by default | With MiniLM + pool_size≥4, parallel sessions beat batched single session |
| SIMD dot product via simsimd | Optional `simd` feature flag; single `dot_product_f32()` in `scoring::embedding` replaces 3 copies |
