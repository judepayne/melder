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

## BM25 Pending Buffer on Tantivy

**What**: Buffer BM25 writes with a `dirty` flag. `upsert()` and `remove()` mark `dirty = true` without committing. A new `commit_if_dirty()` method commits + reloads + clears cache only when dirty. Session code calls `commit_if_dirty()` on the opposite-side BM25 index just before querying it.

**Why discarded**: This approach was implemented and measured a 2x throughput improvement (256 → 512 req/s on 10k×10k dataset with usearch+BM25). However, it only masked the underlying architectural problem: Tantivy's commit/segment/reload cycle is fundamentally expensive (~5-10ms per commit). The pending buffer reduced commit frequency but did not eliminate the cost. At scale, the batching window becomes a tuning knob that users must configure, and the default behavior (small batch size) remains slow. The full SimpleBm25 replacement (see [[Key Decisions#Replace Tantivy BM25 Index with Custom DashMap Based SimpleBm25]]) solves the problem at the root: no commits needed at all. SimpleBm25 achieved 3.2× improvement over the original Tantivy baseline (461 → 1,460 req/s) with zero tuning, and eliminated eventual consistency entirely.

---

## LSH-Based Blocking

**What**: Use Locality-Sensitive Hashing (MinHash or SimHash) to create a second fuzzy blocking dimension that further subdivides exact-field blocks without losing recall on hard matches (abbreviations, acronyms).

**Why discarded**: Fundamental trade-off between recall and false positive rate. To catch hard matches with Jaccard/cosine similarity < 0.10 (the cases that justify a full scoring pipeline), LSH must be tuned so loosely that 30–76% of unrelated pairs still become candidates, defeating the purpose of blocking. MinHash with R=1 requires 36–56 bands for 95–99% recall on Jaccard 0.08 pairs; SimHash with P=8 bits requires 7 hash tables for 95% recall on cosine 0.65 pairs. Both produce overlapping blocks incompatible with per-block ANN/BM25 indices. The existing architecture (exact-field blocking + ANN + BM25) is superior because it operates on full vector/token representations rather than lossy hashes. See [[LSH Blocking]] for full mathematical analysis.

---

## OR Blocking Mode

**What**: Support `blocking.operator: "or"` to allow records matching ANY blocking field (not all) to be candidates.

**Why discarded**: OR blocking creates overlapping blocks where a single record can match multiple blocking keys. This is incompatible with per-block candidate generation strategies:
- **ANN**: Candidates are selected per-block; overlapping blocks create ambiguity about which block a candidate belongs to
- **BM25 WAND**: Block-max upper bounds assume non-overlapping blocks; overlapping blocks break the guarantee

AND blocking (all fields must match) is the correct model for entity resolution. OR blocking was never used in production configs. Removed in [[Key Decisions#OR Blocking Removed]].

---

## Inverted Index Path for BM25

**What**: Maintain a separate inverted index (term → posting list) for BM25 queries, in addition to the forward index (doc → term frequencies). Use the inverted index for small blocks (exhaustive scoring) and a separate inverted path for large blocks.

**Why discarded**: The inverted index path was replaced by Block-Max WAND (see [[Key Decisions#WAND BM25 Implementation]]) which provides identical results with 90-99% fewer evaluations at scale. WAND uses the same forward index but adds per-block upper bounds to skip documents early. The inverted path is simpler but less efficient; WAND is more complex but scales better. At 4.5M scale, WAND's savings are critical.

---

See also: [[Constitution]] for the invariants that ruled out several of these approaches, [[Key Decisions]] for the alternatives that were chosen instead.
