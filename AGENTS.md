# AGENTS.md

Purpose: compact operating instructions for LLM coding agents working in `melder`. Read this before editing code.

## 0. Bootstrap

Before code changes, read:
- `vault/project_overview.md` (source of truth: architecture, module map, current state, backlog, benchmarks, training loop)
- `vault/decisions/key_decisions.md` when touching an area with prior tradeoffs/rejections
- `vault/ideas/discarded_ideas.md` before reviving an old approach

If a requested change conflicts with the Constitution, say so before proceeding.

## 1. Core Identity

Melder is a Rust record-matching engine with:
- batch mode: `meld run` (A=reference pool, B=query side)
- live mode: `meld serve` (A/B symmetric)
- enroll mode: single-pool entity resolution

Single crate, binary `meld`, crate `melder`.

## 2. Invariants (bugs if violated)

1. Batch asymmetry, live symmetry.
2. One scoring pipeline only: all matching flows through `src/matching/pipeline.rs::score_pool()`.
3. CrossMap bijection: 1:1 A<->B enforced atomically under one lock / transaction; never bypass `claim()` semantics.
4. Combined vector identity: concatenated `sqrt(weight)`-scaled normalized field vectors must preserve weighted cosine identity.

## 3. Build / Test Matrix

Use these commands unless task scope clearly allows narrower checks:

```bash
# builds
cargo build
cargo build --release
cargo build --release --features usearch
cargo build --release --features usearch,parquet-format
cargo build --release --features usearch,gpu-encode
cargo build --release --features usearch,builtin-model

# quality gates
cargo fmt -- --check
cargo clippy --all-features
cargo test
cargo test --all-features   # required before commit
```

Feature flags currently in use:
- `usearch`: HNSW ANN backend
- `parquet-format`: Parquet IO
- `simd`: SimSIMD dot product acceleration
- `gpu-encode`: GPU ONNX inference in batch mode (CoreML macOS / CUDA Linux)
- `builtin-model`: embed ONNX weights into binary via `include_bytes!()`

Rules:
- `gpu-encode` is batch-mode acceleration; live mode should warn/ignore GPU requests rather than pretending GPU helps at batch=1.
- `vector_index_mode: mmap` is read-only; valid for batch caches, not mutable live indexing.
- Before commit, run `cargo test --all-features` unless impossible; say what blocked it.

## 4. Architecture Map

- `src/main.rs`: CLI dispatch only
- `src/lib.rs`: module declarations only
- `src/config/`: YAML schema + validation (`schema.rs`, `loader.rs`, `enroll_schema.rs`)
- `src/matching/`: pipeline, blocking, candidates, exclusions
- `src/scoring/`: exact / embedding / numeric / composite scoring
- `src/fuzzy/`: ratio, partial_ratio, token_sort, wratio
- `src/encoder/`: ONNX session pool, local/HF/builtin model loading, optional batching coordinator
- `src/vectordb/`: `VectorDB` trait; flat + usearch backends; manifest + text hash caches
- `src/store/`: `RecordStore` trait; memory + sqlite implementations
- `src/crossmap/`: `CrossMapOps`; memory + sqlite bijection implementations
- `src/bm25/`: `SimpleBm25` scorer, WAND path for large blocks
- `src/session/`: operational core for live/enroll flows
- `src/state/`: startup loading, WAL, backend construction
- `src/batch/`: batch engine + CSV writers
- `src/api/`: axum router + handlers
- `src/cli/`: one file per subcommand

## 5. Preferred Design Moves

- Extend traits rather than leaking backend-specific logic upward.
- Keep batch/live/enroll behavior aligned through shared pipeline pieces.
- Reuse existing indices / caches; avoid duplicate scoring paths and duplicate candidate generation logic.
- Prefer explicit typed config/schema enums over runtime string fallbacks.
- Prefer correctness-preserving simplifications over clever concurrency.

## 6. Code Style

- Imports in 3 groups separated by blank lines: `std`, external crates, `crate::`/`super::`; alphabetical within group.
- Names: files/modules `snake_case`; structs/enums `CamelCase`; fns `snake_case`; constants `SCREAMING_SNAKE_CASE`; CLI entrypoints `cmd_*`; test helpers `make_*`.
- `mod.rs`: only `pub mod` / `pub use`; no logic.
- `lib.rs`: declarations only.
- Every source file gets `//!`; public functions get `///`; no doc comments on private helpers.
- Use rustfmt defaults; trailing commas everywhere; `// ---` banners in long files.

## 7. Error / Logging / Concurrency Rules

- Typed domain errors with `thiserror`; boundary/context with `anyhow` only where appropriate.
- Functions should usually return `Result<T, SpecificError>`, not `anyhow::Result`.
- Top-level enums should have `#[from]` conversions for module errors.
- CLI entrypoints handle errors with `match` + `eprintln!` + `process::exit(1)`.
- HTTP handlers map errors to `StatusCode` + JSON.
- Never `unwrap()` except poison recovery: `.unwrap_or_else(|e| e.into_inner())`.
- `expect()` only for genuinely impossible states.
- Logging uses `tracing` with structured fields; only `info!` / `warn!`.
- CPU-bound async work goes through `spawn_blocking`.

## 8. Data / Matching Semantics

- `Record = HashMap<String, String>` remains the core record shape.
- Empty-vs-empty and empty-vs-nonempty scoring semantics matter; preserve existing behavior.
- Exact prefilter runs before blocking and can recover cross-block matches.
- Exclusions filter after candidate union and before scoring; if excluded pair is currently matched, break match first.
- In match mode, candidate acceptance must respect crossmap bijection.
- In enroll mode, avoid self-matches when query side equals pool side.

## 9. Storage / Persistence Rules

- Memory backend: in-RAM + WAL for live durability.
- SQLite backend: durable source of truth; no fake in-memory shadow invariants.
- Review persistence, exclusions persistence, and crossmap flushing belong in backend abstractions, not session-level backend checks.
- Do not reintroduce runtime `uses_sqlite` branching into core logic.

## 10. Encoder / GPU Notes

- Encoder supports: named fastembed models, HF Hub names, local ONNX paths, and builtin embedded model.
- Local ONNX path support depends on tokenizer/config sidecar files; preserve that contract.
- `gpu-encode` must remain batch-oriented and observable: log requested device, selected provider, and fallback/error path clearly.
- Linux GPU path means CUDA + ONNX Runtime shared-library compatibility; prefer fail-fast or explicit warning over silent CPU fallback.
- macOS GPU path uses CoreML; keep stderr suppression behavior if still needed for framework noise.

## 11. Tests

- All tests live at bottom of source files under `#[cfg(test)] mod tests`; no integration-test sprawl unless already established.
- Table-driven tests for scorers/config parsing.
- Feature-gated code needs feature-gated tests.
- Add assertion messages.
- Use `tempfile::tempdir()` for filesystem tests.
- If changing scoring, blocking, crossmap, WAL, sqlite, encoder, or feature-gated behavior, add/adjust tests in the touched module.

## 12. Docs / Session Hygiene

- After significant work, update `vault/project_overview.md` concisely: current state, completed items, new backlog, architecture changes, benchmark/perf facts, training-loop state if relevant.
- Delegate heavier documentation to doc-agent when available: decisions, failed attempts, todo cleanup, architecture notes.
- Do not bloat overview with long narratives.

## 13. Practical Guardrails

- Do not create a second scoring path.
- Do not bypass trait abstractions with backend-specific hacks unless there is no viable alternative.
- Do not silently degrade correctness for throughput.
- Do not revive OR blocking, dual-DashMap crossmap, or Tantivy BM25.
- Do not forget all-features validation when feature-gated code changed.
- When unsure, preserve invariants first, performance second.
