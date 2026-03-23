---
type: idea
module: general
status: discarded
tags: [rejected, anti-patterns, history]
related_code: [crossmap/mod.rs, vectordb/mod.rs, encoder/coordinator.rs, scoring/]
---

# Discarded Ideas

Approaches that were considered and rejected. Recorded here to prevent re-attempting them in future sessions.

## Per-Field-Per-Block Vector Indices

**What**: Store a separate vector index for each embedding field within each blocking bucket. Query each index separately and combine results.

**Why discarded**: Multiplicative complexity -- N fields x M blocks = N*M indices. Multiple ANN queries per record, complex cache management, and the separate queries could not produce the same ranking as a weighted cosine sum without a separate merge step. Replaced by the [[Key Decisions#Combined Vector Index Single Index Per Side|combined vector index]].

---

## DashMap-Based CrossMap

**What**: Use two `DashMap<String, String>` instances (a_to_b, b_to_a) for the cross-map, relying on DashMap's internal sharding for concurrency.

**Why discarded**: Cannot atomically check-and-insert across two DashMaps. DashMap shards independently, so locking a_to_b[shard_3] and b_to_a[shard_7] simultaneously risks deadlock. Even without deadlock, the TOCTOU gap between checking and inserting allows two Rayon threads to both claim the same A record for different B records. Replaced by [[Key Decisions#CrossMap RwLock Over DashMap|single RwLock]].

---

## DashMap of DashSets for CrossMap (Many-to-Many)

**What**: Early CrossMap design used `DashMap<String, DashSet<String>>` allowing one A record to map to multiple B records.

**Why discarded**: The output model (results CSV, review queue, unmatched list) assumes 1:1 pairing. Many-to-many would produce contradictory results and make the crossmap semantics ambiguous. The 1:1 bijection is the correct domain model for entity resolution: one real-world entity maps to one record on each side.

---

## Always-On Encoding Coordinator

**What**: Make the batching coordinator the default encoding path for all modes.

**Why discarded**: Benchmarking showed that with small models (MiniLM) and `encoder_pool_size >= 4`, parallel independent ONNX sessions consistently outperform batched single-session encoding. The coordinator adds latency equal to the batch window (milliseconds of waiting for more requests to accumulate). It is only beneficial with larger models or very high concurrency. Made optional via `encoder_batch_wait_ms` (default 0 = disabled).

---

## Tolerance-Based Numeric Scoring

**What**: Implement graduated numeric scoring with configurable tolerance (percentage difference, absolute range, logarithmic proximity).

**Why not yet**: The numeric scorer currently does equality-only comparison. Graduated scoring was considered but deferred because no current use case requires it. The fields where numeric comparison matters (LEI, ISIN -- identifiers that happen to be numeric) are better served by exact matching. When a genuine numeric proximity use case arises (e.g., matching financial amounts with rounding differences), the scorer can be extended. Recorded here so the design space is not re-explored from scratch.

---

## Synthetic fine-tuning with independent random seeds

**Tried:** 5-round training loop using seeds 0–4 for training datasets and seed 9999 for holdout. Full retrain from `all-MiniLM-L6-v2` each round on accumulated pairs (~9k pairs/round, labels 1.0/0.7/0.0, `CosineSimilarityLoss`).

**Outcome:** Holdout collapsed to zero matches by round 3. Training recall reached 99% (memorisation signal). The model overfit to the specific Faker-generated entity names in seeds 0–4 and lost all generalisation ability.

**Why it failed:** Different seeds produce non-overlapping entity name vocabularies. The model memorised training names rather than learning noise-invariant representations. This is a holdout design flaw, not a fundamental problem with fine-tuning.

**Do not retry** with this exact design. The fix is to share the A-side entity pool between training and holdout (same A entities, different B noise draws). See [[Training Loop]] for full results and analysis.

---

See also: [[Constitution]] for the invariants that ruled out several of these approaches, [[Key Decisions]] for the alternatives that were chosen instead.
