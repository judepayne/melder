---
type: overview
module: general
status: active
tags: [overview, index, onboarding]
---

# Melder — Project Overview & Index

_Single source of truth for onboarding. Read this at the start of every session.
Update it (concisely) when completing significant work._

Last updated: 2026-03-30 (A1–A6 architectural refactors complete; 397 tests pass, zero clippy warnings)

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

Feature flags: `usearch` (HNSW ANN), `parquet-format` (Parquet I/O), `simd` (SimSIMD hardware-accelerated dot product). BM25 is now always compiled (no feature gate — uses lock-free `SimpleBm25`, no Tantivy dependency). Mode enum: `Match` (two-sided matching) or `Enroll` (single-pool entity resolution).

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
4. **Blocking** (`src/matching/blocking.rs`) — `BlockingIndex` lookup; AND mode only (OR removed). In enroll mode, `blocking_query()` takes `pool_side` parameter to exclude self-matches when query_side == pool_side.
5. **BM25 candidate filter** (optional) — `SimpleBm25` scores blocked candidates; retains `bm25_candidates` top results. Two query paths: exhaustive (B ≤ 5,000) or Block-Max WAND (B > 5,000) for 90-99% fewer evaluations at scale.
6. **ANN candidate selection** (`src/matching/candidates.rs`) — searches combined embedding index for `top_n` nearest A neighbours. If no embedding fields configured, all blocked records pass through.
7. **Full scoring** (`src/scoring/mod.rs::score_pair()`) — all `match_fields` scored; embedding cosines decomposed from combined vectors.
8. **Classification** — `>= auto_match` → Auto; `>= review_floor` → Review; else NoMatch.
9. **CrossMap claim** — atomic; falls through to next candidate if A already claimed (match mode only).

### Live upsert (`src/session/mod.rs`, `src/api/handlers.rs`)

1. Parse JSON record.
2. Encode via `EncoderPool` — skip if text-hash unchanged (`src/vectordb/texthash.rs`).
3. Store in `RecordStore`, upsert combined vector index, update blocking index. BM25 index updated (lock-free DashMap, instantly visible).
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
| `bm25` | `src/bm25/simple.rs` | DashMap-based lock-free scorer, normalised by analytical self-score |

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
   expansion_search: 0                  # HNSW ef parameter (usearch); 0 = usearch default
   # bm25_commit_batch_size: DEPRECATED — SimpleBm25 has instant write visibility

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
| Live, usearch+BM25 (SimpleBm25), 10k×10k (c=10) | 1,460 req/s (3.2× vs old Tantivy default) |
| Live, usearch+BM25 (WAND), 100k×100k warm (c=10) | 1,070 req/s, p50=6.7ms, p95=24.1ms, BM25 build=309ms (4.7× faster) |

Production: usearch backend; `quantized: true` (2× encoding, negligible quality loss); `vector_quantization: f16` (43% smaller cache). SimpleBm25 achieves 1,460 req/s out of the box — no tuning knobs needed (old Tantivy default was 461 req/s). Batch endpoint sweet spot: size 50 (445 req/s, 1.8× vs single).

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
    100kx100k_inject10k_usearch_bm25/  Live benchmark: 100k×100k with BM25 + embedding, arctic-embed-xs model
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

**Feature flags**: `#[cfg(feature = "...")]` on modules, functions, tests, and match arms. Currently: `usearch`, `parquet-format`, `simd`. BM25 is always compiled (no feature gate).

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

**expansion_search config knob** — Added `performance.expansion_search` (HNSW ef parameter) as a configurable field for the usearch backend. Default 0 (usearch default). Threaded through config → vectordb → usearch_backend. Documented in docs/configuration.md.

**BM25 commit batching** — Added `bm25_commit_batch_size` config flag (default: 1). Buffers N BM25 upserts before committing the Tantivy index, amortizing the expensive commit cost. At batch_size=100, live throughput improved from 461 to 1,473 req/s (3.2×) on 10k×10k with 50% embedding + 50% BM25. Documented in docs/configuration.md with trade-off analysis.

**Tracing spans** — Added lightweight info_span tracing at hot-path boundaries: upsert_record, encode_combined, onnx_encode, bm25_upsert, bm25_commit, blocking_query, score_pool, ann_candidates, bm25_candidates, full_scoring, claim_loop. Near-zero cost when not collecting.

**New benchmark** — Added benchmarks/live/10kx10k_inject50k_usearch/ with resource monitoring (CPU/GPU/memory via psutil and ioreg). Uses Arctic-embed-xs + 50% BM25 scoring config.

**Profiling findings** — CPU profiling via macOS `sample` command revealed BM25 commit_if_dirty was 40% of CPU time under load (tantivy segment finalization, FST construction, GC, lock acquisition). Encoding was 47%. Scoring pipeline was only 12%. This led directly to the batching optimization.

**SimpleBm25 — Tantivy replacement** — Replaced the Tantivy-backed BM25 index (`src/bm25/index.rs`, 1,226 lines) with a custom DashMap-based scorer (`src/bm25/simple.rs`, ~1,000 lines). Removed `tantivy = "0.22"` from Cargo.toml (~40 transitive dependencies eliminated). Key changes: (1) `LiveSideState.bm25_index` changed from `Option<RwLock<BM25Index>>` to `Option<SimpleBm25>` — no external locks needed. (2) All `RwLock` write/read lock acquisition, `commit_if_ready`, `commit_if_dirty`, and `Bm25Ctx` usage removed from `session/mod.rs`. (3) `score_pool()` signature in `pipeline.rs` now accepts pre-computed BM25 and synonym candidates — callers do candidate generation, pipeline does union + scoring. (4) `precompute_self_scores()` eliminated from batch mode (analytical self-score is O(K) at query time). (5) `bm25_commit_batch_size` config deprecated (instant write visibility, no commit concept). Results: default live throughput improved 3.2× (461→1,460 req/s) without any tuning; batch improved 6.6% (13,776→14,686 rec/s); startup 1.7s faster (no self-score pre-warming). Design docs: `CUSTOM_BM25_DESIGN.md` and `LIVE_MODE_IMPROVEMENTS.md` in project root.

**WAND BM25 implementation** — Implemented Block-Max WAND (Weak AND) scoring in `src/bm25/simple.rs` to replace the old inverted index path. Key innovations: (1) **Compact doc IDs**: Bidirectional `String <-> u32` mapping (`CompactIdMap`) reduces posting entry size from ~28 bytes to 8 bytes (u32 doc_id + u32 tf), saving ~2.2GB at 4.5M scale. (2) **BlockedPostingList**: Posting lists divided into blocks of ~128 entries with precomputed `max_tf` per block. Supports efficient block splitting on insert and block merging on remove. (3) **Block-Max WAND scorer**: Uses per-block `max_tf` to compute upper-bound BM25 scores. Skips documents whose cumulative upper bound can't beat the Kth-best score. Mathematically guaranteed to return same top-K as exhaustive scoring. (4) **Simplified API**: `score_blocked()` no longer takes `query_record` and `query_side` parameters. (5) **Two-path strategy**: Exhaustive (B ≤ 5,000) and WAND (B > 5,000). The old inverted path is deleted. Results: 100k×100k live benchmark shows 1,070 req/s (+3.4% vs exhaustive), p50=6.7ms, p95=24.1ms, BM25 build=309ms (4.7× faster). WAND's main benefit targets 4.5M scale where block sizes exceed 5,000. See [[Key Decisions#WAND BM25 Implementation]].

**OR blocking removed** — `blocking.operator` now only accepts `"and"`. `"or"` is rejected at validation with a clear error message. Removed `BlockingOperator::Or` enum, `or_indices` field, and all OR branches from `matching/blocking.rs`. Removed OR SQL path in `store/sqlite.rs`. Removed `blocking_hash_changes_on_operator` test from `vectordb/manifest.rs`. Updated `docs/configuration.md` and `docs/accuracy-and-tuning.md`. Rationale: OR blocking created overlapping blocks incompatible with per-block candidate generation (both ANN and BM25). Never used in production configs. See [[Discarded Ideas#OR blocking mode]].

**WAND BM25 implementation + OR blocking removal** — Completed WAND early-termination scoring for large BM25 blocks (B > 5,000) with compact doc IDs and block-max upper bounds. Removed OR blocking mode (incompatible with per-block candidate generation). 396 tests pass, zero clippy warnings. See [[Key Decisions#WAND BM25 Implementation]] and [[Discarded Ideas#OR blocking mode]].

**Critical bug fixes from code review** — Four critical correctness issues fixed: (1) `confirm_match` used `crossmap.add()` bypassing bijection enforcement — now breaks old pairs first then inserts (Constitution §3). (2) Batch pre-match phases (common ID + exact prefilter) used `add()` instead of `claim()` — replaced with atomic `claim()` to prevent TOCTOU races under Rayon parallelism. (3) WAL `compact()` could lose concurrent writes between flush and writer swap — now holds writer lock for entire compaction. (4) `CompactIdMap::get_or_insert` TOCTOU race could produce duplicate compact IDs under concurrent BM25 upserts — replaced with `DashMap::entry().or_insert_with()` for atomic get-or-create. 396 tests pass, zero clippy warnings.

**High-severity fixes from code review** — Seven high issues fixed: (1) `try_match` was mutating the vector index despite being documented as read-only — upsert removed. (2) `try_match` used `rayon::scope` unconditionally without the coordinator deadlock guard that `upsert_record_inner` has — added same `coordinator.is_none()` check. (3) No HTTP request body size limit — added 10MB `DefaultBodyLimit` layer to axum router. (4) `FlatVectorDB::save_index` used unsafe raw pointer cast with native endianness but load path used explicit LE — replaced with safe per-element `to_le_bytes()`. (5) `BlockingIndex::query` returned ALL records when any single blocking field was empty — now skips empty constraints and filters on available fields, matching batch-mode `passes_blocking` semantics. (6) `SessionError::MissingField` was overloaded for 5+ semantically different error types — split into `NotFound` (→404), `BatchValidation` (→422), with `status_code()` method and `run_blocking_session` handler helper. (7) `MemoryCrossMap::remove` didn't verify the pair matched before removing — now checks bijection consistency, matching `SqliteCrossMap::remove`. 397 tests pass (+1 new blocking test), zero clippy warnings.

**Low-severity fixes from code review (11 of 13 items)** — (1) wratio 0.95 early-exit removed — always runs partial_ratio now for optimal scores. (2) partial_ratio avoids redundant chars().count() on shorter string. (3) Embedding score lookup uses (&str, &str) tuple key instead of format!() allocation per candidate. (4) token_sort uses sort_unstable(). (5) Unknown fuzzy scorer and scoring method names now log warn! before fallback. (6) dot_product_f32 has debug_assert_eq on length mismatch. (7) Pipeline doc updated from 'Tantivy' to 'SimpleBm25'. (8) Dead _claimed_idx variable removed from claim loop. (9) apply_output_mapping now removes the original key (rename semantic, not copy). (10) WAL cleanup_old_files() called after compact to prevent unbounded growth. (11) Dead id_to_side field removed from UsearchVectorDB BlockState and serialization. Two items not applied: L33 (is_multiple_of) was already stable in Rust 1.83+; L39 (deny_unknown_fields) conflicts with extension fields like 'sidecar'. 397 tests pass, zero clippy warnings.

**A1 — Parameter Object for `score_pool()`** — Introduced `ScoringQuery` and `ScoringPool` structs to group logically related parameters. `ScoringQuery` holds: `id`, `record`, `side`, `combined_vec` (the record being scored). `ScoringPool` holds: `store`, `side`, `combined_index`, `blocked_ids`, `bm25_candidate_ids`, `bm25_scores_map`, `synonym_candidate_ids`, `synonym_dictionary` (the pool being scored against). New signature: `score_pool(query: &ScoringQuery, pool: &ScoringPool, config: &Config, ann_candidates: usize, top_n: usize)`. Reduced from 14 parameters to 5, making the function extensible without breaking call sites. Updated 4 production call sites (session/mod.rs ×3, batch/engine.rs ×1) and 5 test call sites. Removed `#[allow(clippy::too_many_arguments)]` — no longer needed. 397 tests pass, zero clippy warnings. See [[Key Decisions#A1 Parameter Object for score pool]].

**A3 — Load-Time Rejection of Unknown Scorer/Method Names** — Added `MatchMethod` enum (Exact, Fuzzy, Embedding, Numeric, Bm25, Synonym) and `FuzzyScorer` enum (Wratio, PartialRatio, TokenSort, Ratio) in `src/config/schema.rs`. Both use `#[derive(Deserialize)]` with `#[serde(rename_all = "snake_case")]` — invalid values rejected at YAML parse time. `FuzzyScorer::TokenSort` has `#[serde(alias = "token_sort_ratio")]` for backward compatibility (fixes the validation/dispatch discrepancy). Changed `MatchField.method` from `String` to `MatchMethod` and `MatchField.scorer` from `Option<String>` to `Option<FuzzyScorer>`. Same for `EnrollMatchField` in `enroll_schema.rs`. Removed `VALID_METHODS` and `VALID_SCORERS` constants from `loader.rs`. Removed `require_one_of` calls for method/scorer (serde does this now). Updated ~56 string comparison sites across 10+ files to use enum variants. Removed the runtime `warn!` fallback branches in `scoring/mod.rs` and `fuzzy/mod.rs` — unreachable with enum dispatch. `FieldScore.method` stays as `String` (API serialization boundary — no change to JSON output). 397 tests pass, zero clippy warnings. See [[Key Decisions#A3 Load Time Rejection of Unknown Scorer Method Names]].

**A4 — Lock Ordering Documentation for UsearchVectorDB** — Added a `## Lock ordering` section to the module-level doc comment in `src/vectordb/usearch_backend.rs`. Documents the canonical lock acquisition order: block_router → blocks → blocks[i] → record_block → text_hashes. Notes that next_key (AtomicU64) is lock-free and can be accessed at any point. Documentation-only change, no code changes. See [[Key Decisions#A4 Lock Ordering Documentation for UsearchVectorDB]].

**A5 — DashMap TOCTOU Race Fix in BM25 `decrement_stats()`** — Fixed 2 TOCTOU races in `src/bm25/simple.rs` `decrement_stats()` (lines ~384-413). Race 1: `doc_freq.get_mut()` + separate `doc_freq.remove()` → replaced with `doc_freq.remove_if_mut()` (atomic check-and-remove under single shard lock). Race 2: `postings.get_mut()` + `drop(list)` + `postings.remove()` → replaced with `postings.remove_if_mut()` (removes posting entry and drops empty list atomically). Both races could cause stale IDF statistics or lost posting entries under concurrent upsert/remove. Now impossible. Uses DashMap 6's `remove_if_mut` API. 397 tests pass, zero clippy warnings. See [[Key Decisions#A5 DashMap TOCTOU Race Fix in BM25 decrement stats]].

**A6 — Batch Error Handling Consistency** — Multiple sub-fixes: (1) `enroll_batch` (`src/session/mod.rs`): Changed from fail-fast (`?`) to per-item error collection, matching `upsert_batch`/`match_batch`. Failed items get `enrolled: false` with error message in `id` field. (2) `enroll_batch`: Added empty/MAX_BATCH_SIZE validation guards (was missing, other batch functions had them). (3) `enroll()` WAL append (`src/session/mod.rs`): Changed `let _ =` to `if let Err(e) { warn!(...) }` to match `upsert_record_inner` policy. (4) `remove_batch` (`src/session/mod.rs`): Changed `Err(_)` (all errors masked as "not_found") to distinguish `NotFound` from actual errors (store/IO errors now surfaced as `"error: ..."` in status). (5) All HTTP handlers (`src/api/handlers.rs`): Changed 7 hardcoded `StatusCode::BAD_REQUEST` to use `e.status_code()` — now BatchValidation returns 422, Encoder/Store errors return 500, etc. Affected: upsert_handler, match_handler, add_batch_handler, match_batch_handler, remove_batch_handler, enroll, enroll_batch. 397 tests pass, zero clippy warnings. See [[Key Decisions#A6 Batch Error Handling Consistency]].

**M15/A2: RecordStore trait returns Result — eliminates 35 .expect() panics in SqliteStore** — All 26 RecordStore trait methods now return `Result<T, StoreError>` instead of bare `T`. SqliteStore's 35 `.expect()` calls on SQL operations replaced with `?` propagation. MemoryStore wraps infallible returns in `Ok()`. StoreError added to error.rs with `From<rusqlite::Error>` impl and `#[from]` in both SessionError and MelderError for seamless `?` propagation through the call chain. ~135 call sites updated across 13 files: session/mod.rs (56), state/live.rs (39), batch/engine.rs (15), matching/pipeline.rs (8), matching/candidates.rs (8), state/state.rs (4), cli/ (6), api/handlers.rs (1), bm25/simple.rs (2), synonym/index.rs (5). ReviewEntry type alias added to reduce complex type. 397 tests pass, zero clippy warnings.

**Medium-severity fixes from code review (14 items)** — (1) Synonym additive bonus was diluted by weight normalization when non-synonym weights != 1.0 — separated base score and synonym bonus so normalization only applies to base. (2-3) SqliteCrossMap::add() and claim() now use rusqlite Transaction API for automatic rollback on panic, and claim() is wrapped in a proper transaction. (4) Field names validated at config load time with safe-identifier regex to prevent SQL injection via column names. (5) All RwLock .unwrap() calls in flat.rs and usearch_backend.rs replaced with .unwrap_or_else(|e| e.into_inner()) per project convention (~46 sites). (6) WAL flush() now calls sync_all() after BufWriter flush to fsync to stable storage. (7) numeric_score now uses relative tolerance (1e-9) instead of f64::EPSILON for large values. (8) BM25 field scores no longer silently lost in batch CSV — header and lookup now use 'bm25' key matching. (9) derive_required_fields now includes exact_prefilter fields. (10) EncoderPool fallback uses round-robin AtomicUsize counter instead of always blocking on slot 0. (11) EncoderCoordinator batch collection capped at 64 requests to prevent OOM. (12) WAND heap_insert replaced with BinaryHeap for O(log K) per insert instead of O(K log K). (13) Cursor-based pagination for crossmap_pairs, unmatched_records, and review_list — API breaking change: offset param replaced with cursor, response offset field replaced with next_cursor. (14) Hook events now log warn! when dropped due to full channel. M15 (SqliteStore expect → Result) deferred to separate PR. 397 tests pass, zero clippy warnings.

### In Progress

**CI/CD** — `.github/workflows/ci.yml` + `release.yml` created (macOS ARM, Linux glibc x86_64, Windows MSVC). Requires GitHub remote to activate. Homebrew/Scoop auto-update hooks not yet wired.

### Backlog (ranked)

1. Single-artifact deployment — `include_bytes!()` on ONNX weights.
2. External vector DB — Qdrant/Milvus (VectorDB trait is the interface).
3. Pipeline parallelization — rayon::scope for encode/store/BM25/synonym branches (see `LIVE_MODE_IMPROVEMENTS.md`).
4. Benchmark data regeneration script.
5. GPU/ANE encoding on Apple Silicon — CoreML EP for high-concurrency (c≥50) batched encoding.

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
| BM25: SimpleBm25 replaces Tantivy | DashMap-based lock-free scorer — no commits, no RwLock, instant write visibility. 3.2× default throughput, ~40 fewer deps |
| Exact prefilter | Pre-blocking field-pair confirmation — O(1) hash; recovers cross-block matches |
| Local ONNX encoder paths | Path heuristic → `UserDefinedEmbeddingModel` — fine-tuned models plug in directly |
| Linux CI target | glibc not musl — fastembed → openssl-sys incompatible with musl cross-compilation |
| mmap vector index | `vector_index_mode: mmap` via usearch `view()` — OS paging at 100M+ records; read-only |
| Encoding coordinator off by default | With MiniLM + pool_size≥4, parallel sessions beat batched single session |
| SIMD dot product via simsimd | Optional `simd` feature flag; single `dot_product_f32()` in `scoring::embedding` replaces 3 copies |
