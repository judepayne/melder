---
type: architecture
module: general
status: active
tags: [principles, constraints, correctness]
related_code: [crossmap/mod.rs, matching/pipeline.rs, vectordb/mod.rs]
---

# Constitution: Inviolable Principles

These principles are non-negotiable. Any code change that violates them is a bug, regardless of intent.

## 1. Batch Asymmetry, Live Symmetry

Batch mode is asymmetric: B is the query dataset, A is the reference pool. Each B record is matched against the A-side index. This enables enrichment of B records with A-side data downstream.

Live mode is symmetric: A and B sides have identical capabilities, identical API surfaces, and identical scoring logic. Adding a record to either side searches the opposite side for matches. All code, logic, and API structure must treat A and B equally in live mode.

## 2. One Scoring Pipeline

All matching -- batch, live upsert, and read-only try-match -- flows through `pipeline::score_pool()` in `matching/pipeline.rs`. There is no second code path that computes scores. This guarantees that a score of 0.87 means the same thing regardless of how it was produced. The moment a second scoring path exists, you get divergent match behaviour between modes that is nearly impossible to debug.

See also: [[Business Logic Flow]], [[Scoring Algorithm]], [[Module Map]]

## 3. CrossMap Bijection (1:1 Under One Lock)

Every A record maps to at most one B record, and vice versa. This is enforced atomically in `claim()` by checking both directions under a single `RwLock` before inserting (`crossmap/mod.rs`). Two `DashMap`s were rejected because cross-shard locking creates deadlock risk, and a TOCTOU gap between checking A-to-B and B-to-A would allow duplicate claims under concurrency.

This matters because the entire output model assumes 1:1 pairing. If two B records could claim the same A record, the results CSV would contain contradictory matches, the unmatched count would be wrong, and the cross-map would be silently corrupt.

See also: [[Key Decisions#CrossMap RwLock Over DashMap]]

## 4. Combined Vector = Weighted Cosine Identity

The dot product of two combined vectors exactly equals the weighted sum of per-field cosine similarities: `dot(C_a, C_b) = sum(w_i * cos(a_i, b_i))`. This is achieved by scaling each L2-normalized per-field vector by `sqrt(w_i)` before concatenation (`vectordb/mod.rs`), and relied upon in two directions:

- The ANN index uses it to retrieve the same top-N candidates that exhaustive scoring would find.
- `decompose_emb_scores()` in `matching/pipeline.rs` reverses it to recover per-field cosines without a second ONNX call.

Breaking this (forgetting the sqrt(w) scaling, or not L2-normalizing) would silently corrupt both candidate retrieval and per-field score reporting, with no error -- just wrong results. An end-to-end proof test in `vectordb/mod.rs` asserts the identity holds within tolerance.

---

See also: [[Use Cases]] for the real-world scenarios these principles protect, [[Discarded Ideas]] for approaches that were rejected in favour of these invariants.
