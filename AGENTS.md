# AGENTS.md

## Documentation & Memory Protocol

## 1. THE BOOTSTRAP RULE (Anti-Amnesia)

**CRITICAL**: At the start of every new session or major task, read the full vault before writing any code. Use @explore to read all of the following:

| Directory / File | Contents | Always read? |
|---|---|---|
| `vault/architecture/` | CONSTITUTION (inviolable principles), Module Map, Business Logic Flow, Config Reference, API Reference, Scoring Algorithm, State & Persistence, Performance Baselines | **Yes — always** |
| `vault/decisions/Key Decisions.md` | ADRs with rationale for major design choices | **Yes — always** |
| `vault/todo.md` | Current project status and backlog | **Yes — always** |
| `vault/ideas/Discarded Ideas.md` | Approaches already tried and rejected | When proposing new approaches |
| `vault/business/Use Cases.md` | Three canonical deployment patterns | When discussing use cases or new features |
| `vault/benchmarks/Benchmarks.md` | Benchmark how-to and result tables | When doing performance work |

- Do not write code until you have synchronised with the project's **Key Principles** and **Architecture**.
- If a user request contradicts a principle in `vault/architecture/CONSTITUTION.md`, say so before proceeding.

### 2. STRATEGIC DELEGATION
- You are the 'Thinker,' @doc-agent is the 'Scribe.'
- **Trigger Events**:
   - **New Principle**: If we agree on a new pattern, delegate to @doc-agent to store it in `vault/architecture/`.
   - **Key Decision**: Record the 'Why' in `vault/decisions/`.
   - **Failed Attempt**: Record in `vault/ideas/` to prevent re-trying failed logic after context compaction.
   - **Task Updates**: Tell @doc-agent to update `vault/todo.md` immediately upon progress. This is the TODO list that you are [REQUIRED] to always use for tracking todo items that we discuss. It should always be kept clean and up to date.

**Execution**: Delegate via `doc-agent: [task details]`.


## Project Overview

Melder is a high-performance record matching engine in Rust. It matches records between two datasets (A and B) using configurable pipelines of exact, fuzzy, and semantic similarity scoring. Two modes: batch (`meld run`) and live HTTP server (`meld serve`). Binary name is `meld`, crate name is `melder`.

## Build and Run

```bash
cargo build                                    # debug build
cargo build --release                          # release build
cargo build --release --features usearch       # with HNSW vector index
cargo build --release --features parquet-format # with parquet support
cargo build --release --features usearch,parquet-format  # all features
```

## Test Commands

```bash
cargo test                                     # run all tests
cargo test --all-features                      # run all tests including feature-gated
cargo test <test_name>                         # run a single test by name
cargo test --lib scoring                       # run tests in the scoring module
cargo test --lib vectordb                      # run tests in the vectordb module
```

Always run `cargo test --all-features` before committing to catch feature-gated test failures.

## Lint and Format

```bash
cargo fmt                         # format code (standard rustfmt, no custom config)
cargo fmt -- --check              # check formatting without modifying
cargo clippy --all-features       # lint with all features enabled
```

No custom `rustfmt.toml`, `.clippy.toml`, or `.editorconfig` exists -- use default rustfmt and clippy settings.

## Project Structure

```
src/
  lib.rs           # module declarations only (pub mod, alphabetically ordered)
  main.rs          # CLI entry point only (clap parsing + dispatch to cli::*)
  error.rs         # all error types (MelderError, ConfigError, DataError, etc.)
  models.rs        # core domain types (Record, Side, Classification, MatchResult)
  config/          # schema.rs (config structs) + loader.rs (parse + validate)
  api/             # server.rs (router setup) + handlers.rs (HTTP handlers)
  matching/        # blocking.rs + candidates.rs + pipeline.rs
  scoring/         # mod.rs (score_pair) + exact.rs + embedding.rs
  fuzzy/           # ratio.rs + partial_ratio.rs + token_sort.rs + wratio.rs
  encoder/         # mod.rs (EncoderPool) + coordinator.rs (batch coordinator)
  vectordb/        # trait + flat.rs + usearch_backend.rs + manifest.rs + texthash.rs
  data/            # csv.rs + jsonl.rs + parquet.rs (data loaders)
  crossmap/        # mod.rs (CrossMap: confirmed match pairs)
  session/         # mod.rs (Session: live-mode state + upsert/match logic)
  state/           # state.rs + live.rs + upsert_log.rs
  batch/           # engine.rs + writer.rs
  cli/             # one file per subcommand: run.rs, serve.rs, validate.rs, etc.
```

Key conventions:
- `mod.rs` files are minimal: only `pub mod` and `pub use` statements
- Business logic lives in named files, never in `mod.rs`
- `lib.rs` contains only module declarations, no logic

## Code Style

### Imports

Organise imports in three groups separated by blank lines:
1. `std` imports
2. External crate imports
3. Local `crate::` and `super::` imports

Each group sorted alphabetically. Use `crate::` for cross-module imports, `super::` for intra-module imports.

```rust
use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::{info, warn};

use crate::models::{Record, Side};
use crate::session::Session;
```

### Naming

- Modules and files: `snake_case` (`upsert_log.rs`, `token_sort.rs`)
- Structs/Enums: `CamelCase` (`MatchResult`, `FlatVectorDB`)
- Functions/methods: `snake_case` (`score_pool`, `encode_combined_vector`)
- Constants: `SCREAMING_SNAKE_CASE` (`VALID_METHODS`, `MAX_BATCH_SIZE`)
- CLI command functions: prefix with `cmd_` (`cmd_run`, `cmd_serve`)
- Test helpers: prefix with `make_` (`make_record`, `make_pool`)
- Two-letter acronyms uppercase (`DB`), longer ones CamelCase (`CrossMap`)

### Error Handling

Dual-layer approach: `thiserror` for typed domain errors, `anyhow` for ad-hoc context at boundaries.

- Module-level error enums with named fields (not tuple variants):
  ```rust
  #[error("invalid value for {field}: {message}")]
  InvalidValue { field: String, message: String },
  ```
- Functions return `Result<T, SpecificError>` (not `anyhow::Result`)
- The top-level `MelderError` has `#[from]` conversions for all module errors
- CLI entry points use `match` + `eprintln!` + `process::exit(1)`, not `?`
- HTTP handlers map errors to `StatusCode` + JSON response
- Never use `unwrap()` except for lock poison recovery: `.unwrap_or_else(|e| e.into_inner())`
- Use `expect()` sparingly, only for truly impossible failures

### Types and Derives

- Config structs: `#[derive(Debug, Deserialize)]` with `#[serde(default)]`
- API responses: `#[derive(Debug, Serialize)]`
- Domain enums: `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]`
- Core type alias: `pub type Record = HashMap<String, String>`
- Serde attributes: `rename_all = "snake_case"`, `skip_serializing_if = "Option::is_none"`
- Struct fields are `pub` -- no builder pattern

### Feature Flags

Two optional features: `parquet-format` and `usearch`. Guard code with `#[cfg(feature = "...")]` on modules, functions, tests, and match arms.

### Formatting

Standard rustfmt defaults (4-space indent, K&R braces). Trailing commas everywhere (structs, enums, function args, match arms). Use `// ---` banner comments to separate logical sections within long files.

### Documentation

- Module-level `//!` doc comments on every file (1-3 lines minimum)
- `///` doc comments on public functions (imperative summary, then detail)
- Inline `//` comments for non-obvious logic
- No doc comments on private functions or test helpers

### Logging

Uses `tracing` crate. Structured key-value style:
```rust
info!(side = s, id = %resp.id, matches = resp.matches.len(), "add");
warn!(side = s, error = %e, "add failed");
```
Use `eprintln!()` (not tracing) for build-time progress output. Only `info!` and `warn!` levels are used.

### Async Patterns

- `main.rs` is synchronous; async runtimes created manually via `tokio::runtime::Runtime::new()`
- CPU-bound work offloaded with `tokio::task::spawn_blocking`
- Axum handlers return `axum::response::Response` (concrete type)
- State passed as `Arc<Session>` via `Router::with_state()`
- Graceful shutdown with `tokio::select!` on Ctrl-C / SIGTERM

### Concurrency

- `DashMap` for concurrent record stores
- `RwLock` for indices (write-heavy)
- `std::sync::Mutex` for encoder pool slots
- Lock poison recovery everywhere: `.unwrap_or_else(|e| e.into_inner())`

## Test Conventions

- All tests are `#[cfg(test)] mod tests` at the bottom of each source file
- No integration test directory -- everything is in-crate
- Table-driven tests for scorers (vec of `(input, expected)` tuples)
- Test helpers at top of test module (`make_record`, `make_pool`)
- Always include assertion messages: `assert!(x, "context: {}", val)`
- Use `tempfile::tempdir()` for filesystem tests
- Feature-gated tests use `#[cfg(feature = "...")]`
- Generic test suites via `macro_rules!` (see `vectordb/tests.rs`)
- No external test frameworks -- pure `#[test]` with `assert!`/`assert_eq!`
