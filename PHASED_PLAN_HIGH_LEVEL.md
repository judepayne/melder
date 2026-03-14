# Scaling to Millions — High-Level Phased Plan

Reference design: `vault/ideas/Scaling to Millions.md`

## Overview

Four implementation phases, ordered by value and independence. Each phase is independently shippable and testable. Phases 2 and 3 are independent of each other and can be built in either order. Phase 4 is "build when needed."

| Phase | What | Effort | Prerequisite |
|---|---|---|---|
| 1 | BM25 scoring + filtering (batch and live) | Medium | None |
| 2 | SQLite record store (live mode) | Large | None |
| 3 | Memory-mapped vector index | Small | None |
| 4 | Memory budget auto-configuration | Medium | Phases 2 + 3 |

---

## Phase 1: BM25 Scoring + Filtering

**Goal:** Add `method: bm25` as a first-class scoring method and candidate filter. Works in both batch (`meld run`) and live (`meld serve`) modes.

**What it unlocks:**
- BM25 as scoring enrichment alongside embeddings — suppresses common-token noise from untrained models
- No-embedding mode for one-off jobs — no ONNX model, no vector index, no encoding cost
- Sequential ANN → BM25 pipeline when both are enabled

### Config changes

- New `method: bm25` in `match_fields` — no `field_a`/`field_b` required
- New top-level keys: `ann_candidates` (default: 50), `bm25_candidates` (default: 10)
- Change `top_n` default from current value to 5; `top_n` becomes purely "final output size"
- Validation: `ann_candidates >= bm25_candidates >= top_n` (when both enabled)
- BM25 indexes fields from `method: fuzzy` and `method: embedding` entries only (zero-config)
- Feature flag: `bm25` in `Cargo.toml` (Tantivy is optional dependency)

### Code changes — batch mode

| Area | File(s) | Change |
|---|---|---|
| Config schema | `src/config/schema.rs` | Add `ann_candidates`, `bm25_candidates` fields to `Config`. Make `field_a`/`field_b` optional on `MatchField` (required for all methods except `bm25`). |
| Config validation | `src/config/loader.rs` | Validate filter size constraints. Validate `bm25` method has no fields. Validate `bm25` fields list derivation (from fuzzy/embedding entries). |
| BM25 module | `src/bm25/` (new) | `mod.rs` + `index.rs` + `scorer.rs`. Wraps Tantivy: build index from records + field list, query index, compute normalised score. |
| Index building | `src/batch/engine.rs` | After data load, if BM25 enabled: extract text fields, build A-side and B-side Tantivy indices. |
| Candidate selection | `src/matching/candidates.rs` | Current ANN candidate retrieval returns `ann_candidates` (not `top_n`). Add BM25 re-rank/filter step: score ANN shortlist with BM25, sort, take top `bm25_candidates`. |
| Scoring | `src/scoring/mod.rs` | Add `method: bm25` branch in `score_pair()`. Look up normalised BM25 score for the pair. |
| Pipeline | `src/matching/pipeline.rs` | Wire the sequential ANN → BM25 → score flow. Handle all four pipeline modes (ANN+BM25, ANN-only, BM25-only, neither). |
| Lib | `src/lib.rs` | Add `pub mod bm25;` (feature-gated). |

### Code changes — live mode

| Area | File(s) | Change |
|---|---|---|
| Session | `src/session/mod.rs` | Hold A-side and B-side Tantivy `Index` + `IndexWriter` alongside vector index and blocking index. |
| Upsert | `src/session/mod.rs` | On upsert: tokenise record, add document to the appropriate side's Tantivy index, commit. |
| Match | `src/session/mod.rs` | Query the opposite side's Tantivy index during candidate selection / scoring. |
| State | `src/state/live.rs` | Include BM25 index in state initialisation and shutdown. |

### Testing

- Unit tests in `src/bm25/`: index build, query, self-score normalisation, edge cases (empty fields, single token, no overlap)
- Unit tests in `src/scoring/mod.rs`: `method: bm25` scoring produces [0, 1] values
- Unit tests in `src/config/loader.rs`: validation constraints, field derivation, feature-flag gating
- Integration test in `src/matching/pipeline.rs`: BM25-only pipeline, ANN+BM25 pipeline, verify candidate counts respect `ann_candidates` / `bm25_candidates` / `top_n`
- Live-mode test: upsert records, verify BM25 index updated, match query returns BM25 scores

### Definition of done

- `cargo test --all-features` passes
- `cargo test` (without `bm25` feature) passes — no compilation errors when BM25 is disabled
- All eight pipeline modes from the pipeline table work correctly
- Existing configs without BM25 produce identical results to current behaviour (no regression)

---

## Phase 2: SQLite Record Store (Live Mode)

**Goal:** Replace in-memory `DashMap` + `HashMap` with SQLite for `meld serve`. Batch mode unchanged.

**What it unlocks:**
- Transactional consistency — single SQLite transaction per upsert instead of coordinating 6+ structures
- Fast restarts — no WAL replay, SQLite is durable
- Clean crossmap persistence — SQL constraints enforce bijection

### Code changes

| Area | File(s) | Change |
|---|---|---|
| Store trait | `src/store/` (new) | Define `RecordStore` trait: `insert`, `get`, `get_batch`, `remove`, `iter`, `len`, `blocking_lookup`. |
| Memory store | `src/store/memory.rs` | Wrap existing `DashMap` + blocking `HashMap` behind the trait. Used by `meld run`. |
| SQLite store | `src/store/sqlite.rs` | Implement trait with `rusqlite`. Tables: `records_a`, `records_b`, `blocking_a`, `blocking_b`, `crossmap`. WAL mode, explicit `cache_size`, `page_size = 8192`. |
| Session | `src/session/mod.rs` | Accept `Box<dyn RecordStore>` instead of concrete maps. |
| CLI dispatch | `src/cli/run.rs`, `src/cli/serve.rs` | `run` creates `MemoryStore`, `serve` creates `SqliteStore`. |
| State | `src/state/live.rs` | Refactor to use `RecordStore` trait. Remove in-memory crossmap/blocking structures. |
| Crossmap | `src/crossmap/mod.rs` | Implement crossmap operations via SQL (bijection enforced by unique constraints). |
| Data loading | `src/batch/engine.rs` | Load records via `RecordStore::insert` (both stores implement it). |
| Lib | `src/lib.rs` | Add `pub mod store;`. |

### Testing

- Generic test suite via `macro_rules!` (like `vectordb/tests.rs`) — runs same tests against `MemoryStore` and `SqliteStore`
- Batch regression: existing test configs produce identical output with `MemoryStore`
- Live-mode: upsert, match, crossmap operations via `SqliteStore`
- Crash recovery: kill process mid-upsert, restart, verify data integrity

### Definition of done

- All existing tests pass with `MemoryStore` (no batch regression)
- Live-mode tests pass with `SqliteStore`
- `meld serve` starts with SQLite, survives restart, state is durable
- `meld run` never touches SQLite

---

## Phase 3: Memory-Mapped Vector Index

**Goal:** Add `vector_index_mode: mmap` config option for extreme-scale batch jobs.

**What it unlocks:**
- Batch jobs at 100M+ records where vector index exceeds available RAM
- Trade latency for memory — acceptable for batch, not for live

### Code changes

| Area | File(s) | Change |
|---|---|---|
| Config | `src/config/schema.rs` | Add `vector_index_mode: load | mmap` (default: `load`). |
| Config validation | `src/config/loader.rs` | Validate value. Warn if `mmap` used with `meld serve`. |
| Usearch backend | `src/vectordb/usearch_backend.rs` | In `load()`: switch `index.load(path)` to `index.view(path)` when mode is `mmap`. |

### Testing

- Unit test: `view()` mode loads index, search returns same results as `load()` (correctness)
- Benchmark: compare search latency `load()` vs `view()` on warm/cold cache

### Definition of done

- `vector_index_mode: mmap` works for batch jobs
- Config validation warns when used with `meld serve`
- No change to default behaviour

---

## Phase 4: Memory Budget Auto-Configuration

**Goal:** Add `memory_budget: auto | <size>` config that auto-selects in-memory vs disk-backed storage.

**What it unlocks:**
- Self-tuning melder — no manual SQLite/mmap configuration
- Works on any machine size without config changes

### Prerequisites

- Phase 2 (SQLite store) — needed to switch record store to disk
- Phase 3 (mmap vector index) — needed to switch vector index to disk

### Code changes

| Area | File(s) | Change |
|---|---|---|
| Config | `src/config/schema.rs` | Add `memory_budget` field (string: "auto" or size like "24GB"). |
| Budget calculator | `src/config/loader.rs` or new `src/budget.rs` | Estimate footprint from record count + embedding dims. Compare to budget. Decide store backend + vector mode. Set SQLite `cache_size` to ~30% of budget. |
| CLI dispatch | `src/cli/run.rs`, `src/cli/serve.rs` | Use budget calculator output to select `MemoryStore` vs `SqliteStore` and `load` vs `mmap`. |

### Testing

- Unit tests: budget calculator produces correct decisions for various record counts and machine sizes
- Integration: `memory_budget: 1GB` with large dataset triggers SQLite + mmap; `memory_budget: auto` on current machine uses in-memory

### Definition of done

- `memory_budget: auto` works correctly on machines of various sizes
- `memory_budget: <size>` constrains memory usage to approximate target
- Default (no `memory_budget` key) behaves identically to current behaviour

---

## Cross-Cutting Concerns

### Feature flags

| Feature | What it gates | Always compiled? |
|---|---|---|
| `bm25` | Tantivy dependency, BM25 module, `method: bm25` config parsing | No — optional |
| `usearch` | Usearch backend (existing) | No — optional |
| `parquet-format` | Parquet data loader (existing) | No — optional |
| (none) | `rusqlite` / SQLite store | Yes — always compiled |

### Regression safety

Every phase must pass:
- `cargo test` (default features)
- `cargo test --all-features`
- `cargo fmt -- --check`
- `cargo clippy --all-features`
- Existing example configs produce identical output

### Documentation

After each phase, update:
- `vault/ideas/Scaling to Millions.md` — mark phase as complete
- `vault/architecture/Config Reference.md` — new config keys
- `vault/architecture/Business Logic Flow.md` — pipeline changes
- `vault/todo.md` — task status
