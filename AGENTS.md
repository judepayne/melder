# AGENTS.md

## 1. BOOTSTRAP (Anti-Amnesia)

**At the start of every session, before writing any code:**

1. Read `vault/project_overview.md` — this is the single source of truth for project state, architecture, module map, pipeline flows, benchmarks layout, current work, and backlog.
2. If a task touches an area with known historical decisions or rejected approaches, also read `vault/decisions/Key Decisions.md` and/or `vault/ideas/Discarded Ideas.md`.
3. If a user request contradicts a principle in `vault/architecture/CONSTITUTION.md` (summarised in project_overview.md §2), say so before proceeding.

Do not write code until synchronised with the current project state.

---

## 2. END-OF-SESSION UPDATE

After completing significant work, update `vault/project_overview.md` to reflect:
- New completed items (move from In Progress / Ready → Completed in §11)
- Any new backlog items
- Any new architectural facts (module changes, new config fields, new performance numbers)
- Changes to the training loop state (§10)

Keep updates concise — do not inflate the file. Delegate detailed ADRs, failed-attempt records, and deep architectural notes to the doc-agent (see §3).

---

## 3. STRATEGIC DELEGATION

You are the 'Thinker,' @doc-agent is the 'Scribe.'

**Trigger events — delegate to doc-agent:**
- **New Principle**: agree on a new invariant → `vault/architecture/`
- **Key Decision**: any non-obvious architectural choice → `vault/decisions/Key Decisions.md`
- **Failed Attempt**: record to prevent re-trying after context compaction → `vault/ideas/Discarded Ideas.md`
- **Task Updates**: progress on `vault/todo.md` — keep clean and current

**Execution**: `doc-agent: [task details]`

---

## 4. Build & Test Commands

```bash
# Builds
cargo build                                               # debug
cargo build --release                                     # release
cargo build --release --features usearch                  # standard production build
cargo build --release --features usearch,parquet-format   # all features

# Tests — always run --all-features before committing
cargo test                        # all tests
cargo test --all-features         # catches feature-gated failures
cargo test <name>                 # single test
cargo test --lib scoring          # module tests

# Lint / Format
cargo fmt                         # format (no custom rustfmt.toml)
cargo fmt -- --check
cargo clippy --all-features       # no custom .clippy.toml
```

---

## 5. Code Style

### Imports

Three groups separated by blank lines: (1) `std`, (2) external crates, (3) `crate::`/`super::`. Alphabetical within each group.

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

- Modules/files: `snake_case`
- Structs/Enums: `CamelCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- CLI command functions: `cmd_` prefix (`cmd_run`, `cmd_serve`)
- Test helpers: `make_` prefix (`make_record`, `make_pool`)
- Two-letter acronyms uppercase (`DB`); longer ones title-case (`CrossMap`)

### Error Handling

`thiserror` for typed domain errors; `anyhow` for ad-hoc context at boundaries.

- Module error enums use named fields, not tuple variants:
  ```rust
  #[error("invalid value for {field}: {message}")]
  InvalidValue { field: String, message: String },
  ```
- Functions return `Result<T, SpecificError>`, not `anyhow::Result`
- Top-level `MelderError` has `#[from]` for all module errors
- CLI entry points: `match` + `eprintln!` + `process::exit(1)` — no `?`
- HTTP handlers: map errors to `StatusCode` + JSON
- Never `unwrap()` except lock poison recovery: `.unwrap_or_else(|e| e.into_inner())`
- `expect()` only for truly impossible failures

### Types and Derives

- Config structs: `#[derive(Debug, Deserialize)]` + `#[serde(default)]`
- API responses: `#[derive(Debug, Serialize)]`
- Domain enums: `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]`
- Core type alias: `pub type Record = HashMap<String, String>`
- Serde: `rename_all = "snake_case"`, `skip_serializing_if = "Option::is_none"`
- Struct fields always `pub` — no builder pattern

### Structure

- `mod.rs` files: only `pub mod` and `pub use` — no logic
- Business logic lives in named files, never in `mod.rs`
- `lib.rs`: only module declarations

### Formatting

Standard rustfmt defaults (4-space indent, K&R braces). Trailing commas everywhere (structs, enums, function args, match arms). `// ---` banner comments to separate logical sections in long files.

### Documentation

- `//!` doc comment on every file (1-3 lines minimum)
- `///` on public functions (imperative summary, then detail)
- `//` for non-obvious logic
- No doc comments on private functions or test helpers

### Logging

`tracing` crate, structured key-value style:
```rust
info!(side = s, id = %resp.id, matches = resp.matches.len(), "add");
warn!(side = s, error = %e, "add failed");
```
Only `info!` and `warn!` levels. `eprintln!()` for build-time progress — not tracing.

### Async

- `main.rs` synchronous; async runtimes via `tokio::runtime::Runtime::new()`
- CPU-bound work: `tokio::task::spawn_blocking`
- Axum handlers return `axum::response::Response` (concrete type)
- State: `Arc<Session>` via `Router::with_state()`
- Graceful shutdown: `tokio::select!` on Ctrl-C / SIGTERM

### Concurrency

- `DashMap` for concurrent record stores
- `RwLock` for indices
- `std::sync::Mutex` for encoder pool slots
- Lock poison recovery everywhere: `.unwrap_or_else(|e| e.into_inner())`

### Feature Flags

Current features: `usearch`, `parquet-format`, `bm25`. Guard with `#[cfg(feature = "...")]` on modules, functions, tests, and match arms.

---

## 6. Test Conventions

- All tests in `#[cfg(test)] mod tests` at the bottom of each source file — no integration test directory
- Table-driven tests for scorers: `vec` of `(input, expected)` tuples
- Test helpers at top of test module (`make_record`, `make_pool`)
- Always include assertion messages: `assert!(x, "context: {}", val)`
- `tempfile::tempdir()` for filesystem tests
- Feature-gated tests: `#[cfg(feature = "...")]`
- Generic test suites via `macro_rules!` (see `vectordb/tests.rs`)
- No external test frameworks — pure `#[test]` with `assert!` / `assert_eq!`
