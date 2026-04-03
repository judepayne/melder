---
type: architecture
module: general
status: active
tags: [modules, structure, dependency-flow]
related_code: [lib.rs, main.rs, src/]
---

# Module Map

Melder is a single Rust crate (`melder`). Binary name is `meld`. All modules are declared in `lib.rs` (pub mod only, no logic).

## Module Directory

| Module | Files | Responsibility |
|---|---|---|
| `api/` | `server.rs`, `handlers.rs`, `mod.rs` | Axum HTTP router and request handlers for live mode. See [[architecture/api_reference]]. |
| `batch/` | `engine.rs`, `writer.rs` | Batch matching engine and CSV output writer |
| `bm25/` | `simple.rs`, `mod.rs`, `scorer.rs` | Custom DashMap-based BM25 scorer (replaced Tantivy) |
| `cli/` | `run.rs`, `serve.rs`, `validate.rs`, `tune.rs`, `cache.rs`, `review.rs`, `crossmap.rs`, `enroll.rs`, `export.rs`, `mod.rs` | One file per CLI subcommand. Entry points dispatch to business logic. |
| `config/` | `schema.rs`, `loader.rs`, `enroll_schema.rs` | YAML config structs (schema) and parse + validate (loader). See [[architecture/config_reference]]. |
| `crossmap/` | `mod.rs`, `memory.rs`, `sqlite.rs`, `traits.rs` | Confirmed match pairs with atomic 1:1 claiming. `traits.rs` defines `CrossMapOps`. `memory.rs` = RwLock-backed. `sqlite.rs` = SQLite-backed. See [[decisions/key_decisions#Principles-Inviolable]]. |
| `data/` | `csv.rs`, `jsonl.rs`, `parquet.rs` | Data loaders for each format. Parquet is feature-gated. |
| `encoder/` | `mod.rs`, `coordinator.rs` | ONNX encoder pool (mod.rs) and batch coordinator (coordinator.rs) |
| `fuzzy/` | `mod.rs`, `ratio.rs`, `partial_ratio.rs`, `token_sort.rs`, `wratio.rs` | Fuzzy string matching scorers built on rapidfuzz |
| `hooks/` | `mod.rs`, `writer.rs` | Pipeline hooks for external callbacks |
| `matching/` | `blocking.rs`, `candidates.rs`, `pipeline.rs`, `exclusions.rs`, `mod.rs` | Blocking filter, candidate selection, unified scoring pipeline, exclusions |
| `models/` | `mod.rs` (root level) | Core domain types: `Record`, `Side`, `Classification`, `MatchResult` |
| `scoring/` | `mod.rs`, `exact.rs`, `embedding.rs` | Per-field scoring dispatch and method implementations. Handles synonym scoring internally. See [[architecture/scoring_algorithm]]. |
| `session/` | `mod.rs` | Live-mode state: both sides, upsert/match logic |
| `state/` | `state.rs`, `live.rs`, `upsert_log.rs` | Shared state loading, live side state, WAL. See [[architecture/state_and_persistence]]. |
| `store/` | `mod.rs`, `memory.rs`, `sqlite.rs` | RecordStore trait implementations |
| `synonym/` | `mod.rs`, `index.rs`, `scorer.rs`, `generator.rs`, `dictionary.rs` | Synonym/acronym matching (generates acronyms, builds bidirectional index, scores) |
| `vectordb/` | `mod.rs`, `flat.rs`, `usearch_backend.rs`, `manifest.rs`, `texthash.rs`, `tests.rs` | Vector index trait, backends, cache validation |

## Top-Level Files

| File | Role |
|---|---|
| `main.rs` | CLI entry point. Clap parsing + dispatch to `cli::*` functions. No business logic. |
| `lib.rs` | Module declarations only (`pub mod`, alphabetically ordered). |
| `error.rs` | All error types: `MelderError`, `ConfigError`, `DataError`, `EncoderError`, `IndexError`, `CrossMapError` |
| `models.rs` | Core domain types (at root level for convenience) |

## Dependency Flow

```
main.rs -> cli/* -> {batch/engine, api/server, config/loader, ...}
                         |              |
                         v              v
                    matching/pipeline <-- scoring/*
                         |                   |
                         v                   v
                    candidates.rs      exact/fuzzy/embedding
                         |
                         v
                    vectordb/* (flat | usearch)
                         |
                         v
                    encoder/* (ONNX pool)
```

## Feature Flags

- `parquet-format`: enables `data/parquet.rs` and parquet dependencies
- `usearch`: enables `vectordb/usearch_backend.rs` and the usearch HNSW backend

Code is guarded with `#[cfg(feature = "...")]` on modules, functions, tests, and match arms.

See also: [[architecture/business_logic_flow]] for how these modules interact at runtime, [[decisions/key_decisions]] for the rationale behind the module structure choices.
