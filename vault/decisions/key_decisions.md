---
type: decision
module: general
status: active
tags: [adr, design-decisions, rationale]
related_code: [vectordb/mod.rs, crossmap/mod.rs, vectordb/texthash.rs, vectordb/manifest.rs, encoder/coordinator.rs, encoder/subprocess/mod.rs, encoder/subprocess/slot.rs, encoder/subprocess/protocol.rs]
---

# Key Decisions

Architectural decisions that actively constrain ongoing development. Read this before changing core matching, storage, encoding, or BM25 logic.

For detailed implementation ADRs (SQLite wiring, refactoring steps, concurrency fixes), see [[decisions/key_decisions_implementation]].

---

## Principles (Inviolable)

These are non-negotiable. Violating them is a bug.

### 1. Batch Asymmetry, Live Symmetry

Batch mode: B is query side, A is reference pool. Live mode: A and B are fully symmetric. Enroll mode: single pool only.

**Why**: Enables downstream enrichment (B gets A's data). Live mode must treat both sides equally.

### 2. One Scoring Pipeline

All matching — batch, live upsert, try-match, initial match pass — flows through `pipeline::score_pool()` in `src/matching/pipeline.rs`. No second scoring code path exists.

**Why**: A score of 0.87 must mean the same thing regardless of how it was produced.

### 3. CrossMap Bijection (1:1 Under One Lock)

Every A maps to at most one B, enforced atomically via `claim()` under a single `RwLock` in `src/crossmap/`. Two DashMaps were rejected (TOCTOU gap under Rayon parallelism).

**Why**: Output model assumes 1:1 pairing. Duplicate claims produce contradictory results and silent corruption.

### 4. Combined Vector = Weighted Cosine Identity

`dot(C_a, C_b) = Σ(w_i × cos(a_i, b_i))` — scale each L2-normalised field vector by `sqrt(w_i)` before concatenation (`src/vectordb/mod.rs`). `decompose_emb_scores()` reverses it — no second ONNX call.

**Why**: ANN uses it for candidate retrieval. Breaking it silently corrupts both retrieval and per-field score reporting.

---

## Combined Vector Index (Single Index Per Side)

One combined embedding index per side. Concatenate all per-field embeddings (each scaled by sqrt(weight)) into one combined vector per record.

**Why**: One ANN query instead of N field queries. The mathematical identity (`dot(combined_A, combined_B) = sum(w_i * cos(a_i, b_i))`) means no quality loss. Per-field scores recovered cheaply via `decompose_emb_scores()` — no second ONNX call.

**Commit**: `6a2a332` (Mar 9)

---

## CrossMap: RwLock Over DashMap

Single `RwLock<CrossMapInner>` wrapping two plain `HashMap`s (a_to_b, b_to_a). `claim()` takes a write lock, checks both maps, inserts into both, releases.

**Why**: Two DashMaps cannot atomically check both maps without deadlock risk from cross-shard locking. A TOCTOU gap between checking A-to-B and B-to-A allows duplicate claims under Rayon parallelism.

**Commit**: `e440e4e` (Mar 11)

---

## Text-Hash Skip Optimization

FNV-1a hash of each record's embedding field texts. On upsert, if hash matches stored hash, skip ONNX re-encoding entirely. `TextHashStore` in `vectordb/texthash.rs`.

**Why**: Encoding is the dominant cost (3-6ms). Skipping it for unchanged text yielded 20% live throughput improvement (809→968 req/s). False-negative rate negligible.

**Commit**: `e440e4e` (Mar 11)

---

## Three-Layer Cache Invalidation

Three-layer system for cached vector indices:
1. **Spec hash in filename**: hash of embedding field names + order + weights + quantization
2. **Manifest sidecar** (`.manifest`): records model name, spec hash, blocking hash
3. **Text-hash diffing**: per-record FNV-1a hash for incremental re-encoding

**Why**: No single layer is sufficient. Together they make silent staleness impossible. Changing any config parameter forces a cache miss at the right layer.

**Files**: `vectordb/manifest.rs`, `vectordb/texthash.rs`, `vectordb/mod.rs`

---

## Encoding Coordinator (Batched ONNX Inference)

Optional batching coordinator (`encoder/coordinator.rs`) using mpsc + oneshot channels. Collects concurrent encode requests within `encoder_batch_wait_ms` then dispatches as one ONNX batch.

**Default: disabled** (`encoder_batch_wait_ms: 0`). With small models (Arctic-embed-xs) and `encoder_pool_size >= 4`, parallel independent sessions outperform batched single-session encoding. Beneficial only with larger models or very high concurrency (≥ 20 concurrent requests).

**Commit**: `e440e4e` (Mar 11)

---

## RecordStore Trait for Storage Abstraction

`RecordStore` trait with methods across 4 categories: records (`insert`, `get`, `remove`, `contains`, `len`, `iter`), blocking (`blocking_index`, `add_to_blocking_index`, `remove_from_blocking_index`), unmatched (`mark_matched`, `unmark_matched`), common ID, review persistence. Implementations: `MemoryStore` (DashMap), `SqliteStore`.

**Rule**: Extend the trait to add new storage operations. Do not reintroduce runtime `uses_sqlite` branching into session/pipeline/batch code. Do not bypass trait abstractions with backend-specific hacks.

**Commit**: `319dda8` (Mar 14)

---

## CrossMapOps Trait for CrossMap Abstraction

`CrossMapOps` trait with 13 runtime methods: add, remove, claim, break, get, has, pairs, stats, export, import, flush. Persistence methods (load/save) remain as inherent methods on `MemoryCrossMap` — backends differ fundamentally. Implementations: `MemoryCrossMap` (RwLock), `SqliteCrossMap`.

**Rule**: Same as RecordStore — no backend checks outside the trait implementations.

**Commit**: `de1ab3d` (Mar 14)

---

## Backend Abstraction Cleanup

`LiveMatchState` has zero backend awareness. `flush()` on `CrossMapOps` handles CSV vs no-op internally. Review persistence methods on `RecordStore` handle DashMap vs SQLite write-through internally. `uses_sqlite`, `sqlite_conn`, `as_any()` fields removed.

**Why**: Adding a third backend must only require implementing the traits and wiring into `load()` — not touching `LiveMatchState`, session, or pipeline.

---

## Explicit bm25_fields Config Section

Optional top-level `bm25_fields` section specifies which fields to index for BM25 independently of `match_fields`. When omitted, derived from fuzzy/embedding entries in `match_fields` (backward compatible).

**Why**: BM25-only configs no longer need ghost `match_fields` entries with `weight: 0.0`. Users control BM25 indexing explicitly.

---

## Memory-Mapped Vector Index

`performance.vector_index_mode: "load" | "mmap"` (default `"load"`). When `"mmap"`, usearch calls `index.view()` instead of `index.load()` — OS manages paging.

**Constraint**: `"mmap"` is read-only. Not suitable for `meld serve` (upserts would fail). `meld serve` warns and falls back to `"load"` if `"mmap"` is configured.

---

## Exact Prefilter: Pre-Blocking Exact Match Confirmation

New config section `exact_prefilter` with `{field_a, field_b}` pairs. Runs before blocking. If all configured field pairs match exactly (AND semantics), auto-confirms at score 1.0 immediately, recovering cross-block matches (e.g. records with matching LEI but wrong country code that would otherwise be blocked).

**Why**: O(1) hash lookup — cheaper than blocking itself. On 10k×10k, recovered 188 previously-blocked pairs (+57% of eligible). `RecordStore` gains `build_exact_index()` and `exact_lookup()` methods.

---

## Encoder Supports Local ONNX Paths

`embeddings.model` accepts: named fastembed models, HuggingFace Hub names (containing `/`), local directory paths, or `"builtin"`. Local paths detected by heuristic (absolute, `./`, `../`, `.onnx` suffix, resolves on disk). Directory must contain `model.onnx` plus four standard HuggingFace tokenizer files. Mean pooling applied; output dimension auto-detected from `config.json`'s `hidden_size`.

**Constraint**: `quantized: true` flag is ignored for local paths — point directly to the desired `.onnx` file.

---

## Remote Encoder via Subprocess (`SubprocessEncoder`)

Optional remote encoder path: `embeddings.remote_encoder_cmd` spawns a user-supplied script as a long-lived subprocess pool and delegates text→vector to it via a stdin/stdout protocol. Mutually exclusive with `embeddings.model` (validation XOR). Load-bearing for regulated/large-enterprise customers whose embedding models are walled off behind central internal services.

**Design decisions (in order of blast radius):**

1. **Dispatch: `Arc<dyn Encoder>` across every call site** — `MatchState`, `LiveMatchState`, `EncoderCoordinator`, `vectordb::*`, CLI. The trait was extracted in 3084c6e but not yet used dynamically. Rejected: an `enum EncoderImpl { Local, Remote }` for static dispatch — smaller diff but hard-codes "exactly two impls" and makes future Rust-native HTTP transport a breaking change. Mechanical type migration of ~15 files; no behaviour change on the local path.

2. **Encoder trait extension: `encode_detailed() -> Vec<EncodeResult>` as a default method.** `EncoderPool` inherits the default (no local path change). `SubprocessEncoder` overrides to surface per-record errors from the remote service (e.g. content-policy rejections) so fail-fast tracing can log each failing record individually before collapsing into `EncoderError::BatchError`. Rejected: changing `encode()`'s return type everywhere — widest blast radius for no local-path benefit.

3. **Transport: pipes only (stdin/stdout/stderr).** `sh -c` on Unix, `cmd /C` on Windows, mirroring `src/hooks/writer.rs`. Rejected: unix sockets, shared memory, in-process HTTP. The trait is transport-agnostic so a future Rust-native HTTP transport is possible without a protocol break. stderr is captured and forwarded to `tracing::debug` with a `slot=N` field.

4. **Wire format: NDJSON envelope + binary trailer.** `\n`-terminated UTF-8 JSON lines for control; a 4-byte LE `u32` length prefix followed by raw LE `f32` payload for vector responses. Rejected: pure JSON (3× larger than binary for f32 arrays), Protobuf (dependency), CBOR (dependency). The trailer is only present on successful encode responses; partial responses contain only vectors for `{"ok":true}` entries in the original order.

5. **Per-record errors vs whole-batch errors: both fail the batch (Phase 1 fail-fast).** The protocol distinguishes them (per-record: `{"error":"..."}` entries inside `results`; whole-batch: top-level `{"error":"..."}` with no `results`). Melder logs a structured tracing event for every failed record with full context (text prefix, slot, latency, model_id, command) then surfaces the failure as `EncoderError::BatchError`. A leniency mode with `encode_errors.csv` is additive and may come later if customers demonstrate real need.

6. **No tokio inside `SubprocessEncoder`.** The encoder stack is fully synchronous (`std::sync::Mutex`, OS thread in the coordinator, `std_mpsc` for results) because the `Encoder` trait is sync `&self`. Cannot use tokio internally without spinning up a private runtime or requiring callers to be on one. Rejected: `wait-timeout` crate (for whole-process waits, not per-IO). Per-call timeout uses a per-slot stdout-reader thread pushing frames onto `std::sync::mpsc`, with `encode()` doing `rx.recv_timeout(call_timeout)` — stdlib-only, no new crate.

7. **Respawn logic inline, not in a dedicated thread.** Kill-and-reap happens under the `Mutex<Slot>` guard; the next encode call on that slot runs `Slot::try_respawn` synchronously. Backoff is 1s→60s exponential (same shape as `src/hooks/writer.rs::backoff_duration`). Slot marked `Unhealthy` after 5 consecutive failures within a 60s rolling window. All slots unhealthy → `EncoderError::PoolExhausted` → current batch run aborts.

8. **Fail loudly at startup.** If any slot cannot complete its handshake within the initial respawn cycle, `SubprocessEncoder::new()` returns `EncoderError::RemoteSpawnFailed` and melder exits non-zero. Rejected: starting with a degraded pool. Rationale: "mysteriously slower than configured" is exactly the silent degradation fail-fast-loud is built to prevent. Mid-run degradation still works as expected (slots can individually become unhealthy without killing the whole run).

**Why a subprocess and not in-process HTTP?** The subprocess author owns auth, transport, retries, rate-limiting, request/response shaping, and error classification against their org's API — all the per-organisation concerns that would otherwise bloat melder with a matrix of HTTP clients. The subprocess is a clean boundary for the user to write any-language code against a stable wire protocol. See `docs/remote-encoder.md` for the full user contract and `remote_operation.md` §5 for the Phase 1 design.

**The timeout footgun**: the subprocess's own remote-service timeout must be strictly less than `performance.encoder_call_timeout_ms`. If melder's per-call timeout fires while the script's remote call is still outstanding, melder sends SIGKILL and the script cannot emit a clean error. Load-bearing for user docs — documented in `docs/remote-encoder.md` with a ★ header.

**Hardcoded lifecycle tunables (Phase 1 only; promotable to config later without protocol break)**: handshake timeout 30s; respawn backoff initial 1s, cap 60s; slot-unhealthy threshold 5 failures within 60s; shutdown grace 5s SIGTERM + 5s SIGKILL.

**Related code**: `src/encoder/subprocess/{mod.rs, slot.rs, protocol.rs}`; `src/state/state.rs::build_encoder`; `tests/fixtures/stub_encoder.py` (reference implementation with full failure-injection flags); `tests/remote_encoder.rs` (13 end-to-end integration tests); `benchmarks/batch/10kx10k_remote_encoder/cold/` (worked benchmark example).

**Commits**: `3359bd3` (trait + dispatch migration), `4bf096a` (SubprocessEncoder impl), `72d87e2` (stub + integration tests), `8ad2672` (benchmark), `6a116e9` (user docs). See also [[architecture/threading_model#Remote Encoder Threads]].

---

## Arctic-embed-xs as Recommended Embedding Model

`themelder/arctic-embed-xs-entity-resolution` (22M params, 6 layers, 384 dims) is the recommended model. Published to HuggingFace Hub; auto-downloaded on first use.

**Why over BGE-small/BGE-base**: Superior pre-training (400M samples, hard negative mining). Stretches match/non-match distributions rather than just compressing them. 2-3× faster encoding than BGE-base. Fewer layers → larger LoRA intervention per adapter.

**Experiment 9 results (23 rounds)**: overlap 0.031, combined recall 99.7%, 30 missed matches. See [[decisions/training_experiments_log#Experiment 9]].

---

## Production Configuration: Arctic-embed-xs R22 + 50% BM25

**Final configuration from Experiment 12** (confirmed production baseline):
- `name_emb: 0.30`, `addr_emb: 0.20`, `bm25: 0.50`, `synonym: 0.20` (additive, auto-normalised)
- Overlap: **0.0003**, combined recall: **100%**, zero FPs in auto-match and review
- 560× overlap improvement vs Experiment 1 baseline (0.168 → 0.0003)

**Why BM25 at 50%**: Corpus-aware token scoring complements semantic embedding similarity. At 50% weight, acts as a strong filter for residual false matches without degrading recall.

See [[decisions/training_experiments_log#Experiment 12]].

---

## Replace Tantivy BM25 Index with Custom DashMap-Based SimpleBm25

Replaced Tantivy (`src/bm25/index.rs`, 1,226 lines) with `src/bm25/simple.rs` (~1,000 lines). DashMap-based with per-doc term frequencies, global IDF stats, and block-segmented posting lists. No commit cycle — instant write visibility.

**Why**: Tantivy's commit/segment/reload cycle was fundamentally mismatched for melder's interleaved write-read access pattern. The custom implementation eliminates the architectural mismatch entirely.

**Results**: Default live throughput 3.2× (461→1,460 req/s); batch +6.6%; startup 1.7s faster; ~40 transitive crates removed.

**Commit**: Mar 28

---

## WAND BM25 Implementation

Block-Max WAND early-termination scoring in `src/bm25/simple.rs`. Two-path strategy: exhaustive (B ≤ 5,000), WAND (B > 5,000).

**Key structures**:
- `CompactIdMap`: String↔u32 bidirectional mapping. Reduces posting entry size from ~28 bytes to 8 bytes (saves ~2.2GB at 4.5M scale). Uses `DashMap::entry().or_insert_with()` for atomic ID assignment.
- `BlockedPostingList`: blocks of ~128 entries with precomputed `max_tf` per block.
- WAND scorer: per-block upper bounds skip documents whose cumulative score cannot beat the Kth-best. Uses `BinaryHeap<Reverse<OrdScore>>` for O(log K) per insert.

**Guarantee**: Returns same top-K as exhaustive scoring.

**Results**: 100k×100k live benchmark 1,070 req/s (+3.4%), BM25 build 4.7× faster.

---

## OR Blocking Removed

`blocking.operator` only accepts `"and"`. `"or"` is rejected at validation with a clear error message.

**Why**: OR blocking creates overlapping blocks incompatible with per-block candidate generation (ANN and BM25 WAND). Block-max upper bounds assume non-overlapping blocks. AND blocking is the correct model for entity resolution.

**Do not revive** — see [[ideas/discarded_ideas#OR Blocking Mode]].

---

## Exclusions as a Stateful In-Memory System

`Exclusions` struct in `src/matching/exclusions.rs`: `RwLock<HashSet<(String, String)>>`. Loaded from CSV at startup. WAL events: `Exclude`, `Unexclude`. Pipeline filter: after candidate union, before scoring — O(1) per candidate pair. If an excluded pair is currently matched, CrossMap match is broken first.

**Batch mode**: read-only from CSV. **Live mode**: mutable via API (`POST/DELETE /api/v1/exclude`), WAL-persisted, CSV-flushed on shutdown.

**Why not static-file-only**: Human review feedback requires immediate exclusion + WAL persistence + CrossMap cleanup — a static-file approach would require manual editing and server restart.

---

## Initial Match Pass at Live Startup

After datasets loaded and indices built, `Session::initial_match_pass()` scores all unmatched B records against the A pool before HTTP server starts listening. Called from `cli/serve.rs` after session creation, before `start_server()`. Suppressed via `live.skip_initial_match: true`.

**Why reuse pipeline instead of run_batch()**: Live mode already has BM25/synonym indices built. `run_batch()` would rebuild them from scratch, wasting startup time.

**Behaviour**: Skips encoding (vectors already in index); writes crossmap claims to WAL (crash-safe); adds review-band matches to review queue; fires hook events normally; skips already-matched records; no-op if either side empty.

---

## Accuracy Regression Tests (Deterministic Validation)

Two fixed-dataset accuracy tests prevent silent business logic bugs. Discovered: `score_pair` hardcoded `field_a` for candidate and `field_b` for query, working only in batch mode. In live mode with asymmetric schemas (e.g. `legal_name` vs `counterparty_name`), A→B queries silently scored zero — no error, no crash, just fewer matches. Undetected for months because the existing live CI test used `skip_initial_match: true` (only B→A direction exercised) and throughput checks ignored match quality.

**Jitter root cause (2026-04-03)**: Tests failed on CI with 8-12 pair swaps between expected and actual. Investigated and confirmed the cause was NOT ONNX non-determinism. Two different companies with identical `legal_name`, `short_name`, `country_code`, and empty LEI on the B side produced byte-identical composite scores (0.887523525953 to 12dp). With equal scores, the claim ordering depended on candidate iteration order (memory layout), which varied across runs. Fix: adding `registered_address`/`counterparty_address` as an embedding field (weight 0.20, reducing legal_name from 0.55 to 0.35) provides sufficient discriminating signal — the two companies have different addresses, breaking the score tie. After the fix, both tests produce identical results across repeated cold runs on the same machine.

**Design**: Fixed datasets with asymmetric field names (including addresses), expected outputs committed to repo, validates at crossmap/edge level.

**Live test** (`benchmarks/accuracy/live_10kx10k_inject3k/`):
- 10k A + 10k B records with asymmetric field names (legal_name/counterparty_name, country_code/domicile, lei/lei_code, registered_address/counterparty_address)
- `skip_initial_match: false` — runs initial match pass at startup (B→A direction)
- Injects 3k B records via API (triggers A→B scoring direction via crossmap claim/re-scoring)
- Validates crossmap at two checkpoints: 5,712 pairs after initial match, 6,423 after injection
- Any change to scoring logic that alters results causes CI failure

**Enroll test** (`benchmarks/accuracy/enroll_5k_inject1k/`):
- 5k single-pool dataset with registered_address, full lifecycle: enroll 1k records, validate 2,019 edges, remove 50, re-enroll 50, confirm no edges to removed records
- Tests add/score/remove/re-score paths

**CI integration**: New `accuracy` job runs both tests sequentially, fails on any regression. Runs in parallel with existing `perf` job (both depend on `test`).

**Updating expected outputs**: Run with `--update-expected` flag to regenerate baseline files after intentional scoring changes.

---

See also: [[decisions/key_decisions_implementation]] for detailed implementation ADRs (SQLite wiring, refactoring, concurrency fixes), [[ideas/discarded_ideas]] for rejected approaches.
