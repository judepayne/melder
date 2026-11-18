# AGENTS.md

Purpose: compact operating instructions for LLM coding agents working in `melder`. Read this before editing code.

## 0. Bootstrap

Before code changes, read this file completely and use the feature pointers below for targeted context. Check `.pi/todos/` before starting medium/large work so you do not duplicate active tasks.

If a requested change conflicts with the invariants in §2, say so before proceeding.

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
cargo build --release --features parquet-format
cargo build --release --features gpu-encode
cargo build --release --features builtin-model
cargo build --release --no-default-features   # pure Rust, no C++ (flat backend only)

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
- `src/data/`: CSV / JSONL / Parquet dataset loaders and streaming loaders
- `src/matching/`: shared pipeline, blocking, candidates, exclusions; verification uses that same pipeline, not a second scoring path
- `src/scoring/`: exact / embedding / numeric / synonym / composite scoring
- `src/fuzzy/`: ratio, partial_ratio, token_sort, wratio
- `src/synonym/`: acronym generation, synonym dictionary, bidirectional synonym index
- `src/encoder/`: `Encoder` trait; local ONNX pool; local/HF/builtin model loading; optional batching coordinator; subprocess remote encoder
- `src/vectordb/`: `VectorDB` trait; flat + usearch backends; manifest + text hash caches
- `src/store/`: `RecordStore` trait; memory + sqlite implementations
- `src/crossmap/`: `CrossMapOps`; memory + sqlite bijection implementations
- `src/bm25/`: `SimpleBm25` scorer, WAND path for large blocks
- `src/session/`: operational core for live/enroll flows; `assignment.rs` implements opt-in synchronous reciprocal live proposals
- `src/state/`: startup loading, WAL, backend construction
- `src/batch/`: batch engine; event-sourced match log production
- `src/output/`: build CSV / Parquet / SQLite outputs from match log + scoring log
- `src/hooks/`: long-running hook subprocess for pipeline event callbacks
- `src/api/`: axum router + handlers
- `src/cli/`: one file per subcommand (`run`, `serve`, `enroll`, `validate`, `tune`, `cache`, `crossmap`, `export`)

## 4a. Feature Pointers

When a task mentions these features, start with the listed docs/code:

- Remote/org-specific encoder execution: `docs/remote-encoder.md`, `src/encoder/subprocess/`, `tests/remote_encoder.rs`.
- Pipeline hooks: `docs/hooks.md`, `src/hooks/`.
- Output/export/scoring-log pipeline: `docs/batch-mode.md`, `docs/configuration.md#scoring_log`, `src/output/`, `src/cli/export.rs`.
- CLI utilities: `docs/cli-reference.md`, `src/cli/{validate,tune,cache,crossmap,export}.rs`.
- Threshold tuning / accuracy diagnostics: `docs/accuracy-and-tuning.md`, `src/cli/tune.rs`.
- Vector cache commands and invalidation: `docs/vector-caching.md`, `src/cli/cache.rs`, `src/vectordb/{manifest,texthash}.rs`.
- Data formats and streaming: `docs/batch-mode.md#data-formats`, `src/data/`.
- Synonym/acronym matching: `docs/scoring.md#synonym`, `src/synonym/`.
- Common ID fast path: `docs/configuration.md#datasets`, `src/config/derivation.rs`, `src/matching/pipeline.rs`.
- Live admin flush/shutdown and WAL lifecycle: `docs/live-mode.md`, `docs/api-reference.md`, `src/state/match_log.rs`.
- Live assignment: `docs/live-mode.md#live-assignment-policy`, `docs/scoring.md#live-deferred-acceptance`, `src/session/assignment.rs`, `src/config/schema.rs`. Historical bounded-repair evidence remains in `docs/local-reassignment-productionization.md` and benchmark reports; it is not the active policy.
- Batch graph assignment: `docs/configuration.md#batch`, `docs/batch-mode.md#choosing-pairs-with-the-graph-optimizer`, `docs/performance.md#batch-graph-optimizer`, `src/matching/assignment/`, `src/batch/`. Keep one promoted solver implementation; benchmark adapters must reuse it. The agreed integration contract and evidence are in `benchmarks/assignment_solver/phase-2b-config-and-docs.md`.
- Batch SQLite mode: `docs/batch-mode.md#sqlite-batch-mode-large-datasets`, `src/batch/`, `src/store/sqlite.rs`.
- Enroll-specific config/endpoints: `docs/enroll-mode.md`, `src/config/enroll_schema.rs`, `src/api/handlers.rs`.

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
- Live assignment defaults to `greedy` with no displacement work. YAML-only `live.assignment.strategy: deferred_acceptance` enables synchronous, symmetric proposals under an exclusive mutation gate; no background reconciliation. Enroll is unchanged.
- New automatic live pairs must qualify in both scoring directions, including whole-list score-gap safeguards. Displace only session-created ordinary automatic incumbents for a strictly higher current receiver-perspective score; ties retain incumbents. Manual/canonical/preloaded/restored pairs are protected.
- `max_displacements` defaults to 32 successful displacements per attempt (an initial work budget, not a proven optimum). Zero permits reciprocal free matches; at the cap, remaining shortlisted free candidates can still be tried. Existing retrieval/`top_n` bound proposals; verification of proposer/incumbent is not new proposal generation. No order-independence, accuracy or global-optimality guarantee.
- Enabled assignment uses `CrossMapOps::apply_assignment_delta()` atomically per step and version-1 `assignment_delta` WAL events. Keep legacy `CrossMapReassign` readable, but do not revive bounded one-to-two repair as active policy. Canonical metadata is B→A; audit both directional scores.
- Optional mutation-response `assignment` summaries are omitted for greedy. Candidate `matches` are not accepted links; bulk responses must reflect final CrossMap membership. Emit best-effort Break/Confirm hooks only for completed steps, with confirm source `deferred_acceptance`.
- Batch defaults to `graph` / `approximate` on both storage backends, including when `batch`, `assignment` or `strategy` is omitted. YAML-only explicit `strategy: greedy` retains local claims without an optimizer graph and rejects any explicit `mode`. Mode-only `exact` is valid with the default graph strategy. Approximate is exact through 512 distinct participating A+B endpoints per connected group, deterministic best of score-first/count-first above 512 (score-first wins ties). Explicit `mode: exact` never falls back. Reject unknown values, including `maximum_weight`; no CLI/cutoff/worker-count/tolerance knobs.
- Graph assignment uses only retained automatically eligible shared-pipeline candidates; preserve protected links, exclusions and whole-list ambiguity gates. Check eligibility before rounding assignment weights to nine decimal places; this is not a live scoring rule. Optimize total represented similarity, not pair count or identity accuracy. Fixed-graph results must not depend on edge order, worker scheduling or elapsed time.
- In enroll mode, avoid self-matches when query side equals pool side. Persist returned edges as single-pool enrollment evidence, never CrossMap claims. Flush and offline outputs include seed and isolated records. Enroll uses memory plus WAL; `output.db_path` is an export, not pool storage.

## 9. Storage / Persistence Rules

- Memory backend: in-RAM + WAL for live durability.
- SQLite backend: durable source of truth; no fake in-memory shadow invariants.
- `batch.db_path` is disposable scratch storage: never overwrite inputs or persistent outputs, even during `--dry-run`. SQLite batch does not build vector indexes; reject embedding match fields instead of scoring them as zero. Retained graph candidates/components and graph output assembly remain in memory.
- Assignment chains commit per-step prefixes, not request-wide transactions. WAL append is buffered, not per-step fsync; memory recovery uses WAL/snapshots, while SQLite retains authoritative membership without blind WAL replay. Restart reconciles metadata and protects restored links; it does not resume interrupted queues.
- Assignment append failure attempts inverse-delta rollback. Ambiguous append, rollback or postcommit metadata failure must fail assignment health and reject further optimizing mutations until recovery/restart. CrossMap remains authoritative; hooks are not an audit log.
- Older binaries cannot safely replay/export `assignment_delta` logs; disabling the strategy does not remove downgrade restrictions. Preserve legacy log readers and historical reports.
- Review persistence, exclusions persistence, and crossmap flushing belong in backend abstractions, not session-level backend checks.
- Graph batch publication is atomic for new automatic links; preserve protected pairs on rollback. Logs use `batch_assignment_started` / `batch_assignment_completed`; reject final output building/export from an incomplete graph run, never publish a partial optimum. Use `graph_assignment` as the selected-match source. Emit effective strategy/mode, group counts, largest group and exactness/certified-bound summary only after successful publication; round displayed lower bounds down.
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
- Batch graph gates: omitted batch/assignment/strategy defaults to graph/approximate on both backends, mode-only exact accepted, explicit greedy compatibility and rejection of any mode with greedy, 512/513 dispatch and effective summaries, fixed-graph determinism, independent exact oracle/objective bounds, eligibility-before-rounding, protected/conflicting pairs, atomic publication/rollback and incomplete-log export rejection. Verify CSV/Parquet/SQLite accepted pairs against final CrossMap; run default, all-features and pure-Rust checks per the build matrix.

## 12. Docs / Session Hygiene

- If a change affects user-visible behaviour, update the relevant file in `docs/` and any worked examples that would otherwise become stale.
- Keep `AGENTS.md` current when architecture, invariants, build/test expectations, or feature entry points change.
- Use `.pi/todos/` for active/backlog work tracking; close or update todos as work completes.

## 13. Practical Guardrails

- Do not create a second scoring path.
- Do not bypass trait abstractions with backend-specific hacks unless there is no viable alternative.
- Do not silently degrade correctness for throughput.
- Do not revive OR blocking, dual-DashMap crossmap, or Tantivy BM25.
- Do not forget all-features validation when feature-gated code changed.
- When unsure, preserve invariants first, performance second.
