---
type: decision
module: general
status: active
tags: [adr, design-decisions, rationale]
related_code: [vectordb/mod.rs, crossmap/mod.rs, vectordb/texthash.rs, vectordb/manifest.rs, encoder/coordinator.rs]
---

# Key Decisions

Architectural decisions made during Melder's development, with rationale.

## Combined Vector Index (Single Index Per Side)

**Decision**: Replace per-field-per-block vector indices with a single combined embedding index per side.

**Context**: The original design stored a separate vector index per embedding field per blocking bucket. This led to multiple ANN queries per record and complex index management.

**Choice**: Concatenate all per-field embeddings (each scaled by sqrt(weight)) into one combined vector per record. Store one index per side. A single ANN query retrieves candidates ranked by the exact weighted cosine sum.

**Why**: One query instead of N. The mathematical identity (`dot(combined_A, combined_B) = sum(w_i * cos(a_i, b_i))`) means no quality loss from the consolidation. Per-field scores are recovered cheaply via `decompose_emb_scores()` -- no second ONNX call needed. See [[Constitution#4 Combined Vector Weighted Cosine Identity]] and [[Scoring Algorithm#embedding]].

**Commit**: `6a2a332` (Mar 9)

---

## CrossMap: RwLock Over DashMap

**Decision**: Use a single `RwLock<CrossMapInner>` with two plain `HashMap`s (a_to_b, b_to_a) instead of two `DashMap`s.

**Context**: The CrossMap must enforce a strict 1:1 bijection between A and B records. The `claim()` operation must atomically check both directions and insert only if neither side is already claimed.

**Choice**: Single RwLock wrapping both maps. `claim()` takes a write lock, checks both maps, inserts into both, releases.

**Why**: Two DashMaps cannot atomically check both maps without deadlock risk from cross-shard locking. A TOCTOU gap between checking A-to-B and B-to-A would allow duplicate claims under Rayon parallelism. The single lock makes the bijection invariant trivially correct. Concurrent tests prove exactly-one-winner semantics. See [[Constitution#3 CrossMap Bijection 1 1 Under One Lock]].

**Commit**: `e440e4e` (Mar 11)

---

## Text-Hash Skip Optimization

**Decision**: Compute an FNV-1a hash of each record's embedding field texts. On upsert, if the hash matches the stored hash, skip ONNX re-encoding entirely.

**Context**: In live mode, many upsert requests modify non-embedding fields (e.g., status, metadata). Re-encoding unchanged text through ONNX wastes ~3ms per request.

**Choice**: `TextHashStore` in `vectordb/texthash.rs` stores per-record hashes. On upsert, compare hash before encoding. Skip if unchanged.

**Why**: Encoding is the dominant cost in the upsert path. Skipping it for unchanged text yielded a 20% throughput improvement in live mode benchmarks (809 -> 968 req/s). The hash is cheap (FNV-1a, sub-microsecond) and the false-negative rate is negligible.

**Commit**: `e440e4e` (Mar 11)

---

## Three-Layer Cache Invalidation

**Decision**: Implement a three-layer validation system for cached vector indices: spec-hash in filename, manifest sidecar, and per-record text-hash diffing.

**Context**: Cached indices must never silently serve stale results when config or data changes. Previous iterations had bugs where changing the blocking config or model would reuse an invalid cache.

**Choice**:
1. **Spec hash in filename**: Hash of embedding field names + order + weights + quantization. Different config = different filename = old cache unreachable.
2. **Manifest sidecar** (`.manifest`): JSON file recording model name, spec hash, and blocking hash. Catches model and blocking changes that the filename hash doesn't cover.
3. **Text-hash diffing**: Per-record FNV-1a hash of source text. Detects individual record changes for incremental re-encoding.

**Why**: No single layer is sufficient. The filename hash handles weight/field changes. The manifest catches model and blocking changes. The text hash handles data changes. Together they make silent staleness impossible.

**Files**: `vectordb/manifest.rs`, `vectordb/texthash.rs`, `vectordb/mod.rs`. See [[State & Persistence#WAL]] for how this interacts with WAL replay and the `skip_deletes` / `contains()` guards.

---

## Encoding Coordinator (Batched ONNX Inference)

**Decision**: Implement an optional batching coordinator that collects concurrent encode requests within a time window and dispatches them as a single ONNX batch call.

**Context**: Under high concurrency in live mode, individual ONNX inference calls (one per request) underutilise GPU/CPU batch parallelism in the model runtime.

**Choice**: `encoder/coordinator.rs` uses mpsc + oneshot channels. Incoming encode requests are collected for up to `encoder_batch_wait_ms` milliseconds, then dispatched as one batch. Each caller receives its result via its oneshot channel.

**Why**: Amortises ONNX overhead across concurrent requests. Configurable via `performance.encoder_batch_wait_ms`. Default is 0 (disabled) because with small models (MiniLM) and `encoder_pool_size >= 4`, parallel independent sessions outperform batched single-session encoding. The coordinator shines with larger models or very high concurrency (c >= 20).

**Commit**: `e440e4e` (Mar 11)

---

See also: [[Discarded Ideas]] for the alternative approaches that were considered and rejected before each of these decisions was made.
