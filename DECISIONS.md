# Decision log

The [changelog](CHANGELOG.md) says what shipped; this log explains why. Keep IDs stable. If a choice changes, add a new entry and mark the old one **superseded by D-XXX**.

## 0.1.6 (unreleased) — accepted decisions

### D-001 — Search the whole BM25 index directly when batch blocking is off

With no blocking rule, every reference record is eligible. Search the index directly instead of rebuilding the same list of eligible IDs for every query. The results stay the same; only the redundant work goes away.

Implemented: 2026-11-15 · [Benchmark](docs/performance.md#unblocked-batch-bm25-comparison)

### D-002 — Choose batch pairs together; keep greedy as an explicit option

Use graph assignment by default so two queries competing for one reference can be considered together. Optimize total eligible score, not pair count or identity accuracy. Solve groups of up to 512 participating records exactly and larger groups approximately; callers can request exact solving at any size or avoid building the graph by choosing greedy.

Implemented: 2026-11-18 · [Batch assignment](docs/batch-mode.md#choosing-pairs-with-the-graph-optimizer)

### D-003 — Use fast CSV replay only when earlier links cannot be invalidated

Skip rescanning earlier links when the log shows that upserts preceded final outcomes, confirmed links are one-to-one, and outcomes do not conflict. Otherwise use general replay. This speeds up greedy batch output without changing recorded matches or the log format.

Implemented: 2026-11-18 · [Replay benchmark](benchmarks/assignment_solver/greedy-replay-results.md)

### D-004 — Configure live startup separately from incoming requests

Use graph assignment by default at startup, when the whole pool is available; preserve existing links and publish recoverable decisions before serving. Neither standalone batch settings nor startup settings should silently change the cost of an incoming request. Incoming requests therefore default to greedy matching.

Implemented: 2026-11-21 · [Live mode](docs/live-mode.md)

### D-005 — Offer deferred acceptance for live requests, not background repair

Let an incoming proposal displace a session-created automatic match only if it qualifies in both directions and the receiver strictly prefers it. Ties and protected links stay put. Keep this opt-in and bound the extra work per request: a stronger local pairing is possible, not guaranteed better identity accuracy.

Implemented: 2026-11-18 · [Live assignment policy](docs/live-mode.md#live-assignment-policy)

### D-006 — Prepare enrollment embeddings concurrently; apply records in order

Generate embeddings before taking the pool lock: encoding depends on the incoming text, not the current pool. Concurrent clients can encode in parallel, and bulk requests can encode many records together. Still match and apply each bulk record in input order so record 2 can match record 1.

Implemented: 2026-11-21 · [Enrollment measurements](benchmarks/enroll/optimization/results.md)

### D-007 — Separate searching a pool from assigning a match

Use one scoring pipeline for all three modes. A `SearchPool` borrows the collection and search resources; a reusable `SearchPlan` binds the fields to compare. Batch compares B fields with A fields, live can reverse the direction, and enrollment compares the same field name on two records. The plan changes, not the scorer.

For example, finding a high-scoring candidate does **not** create a link. Batch can pass the candidate to the graph solver (D-002), live can check whether the preference is reciprocal (D-005), and enrollment can return several qualifying edges. Change how similarity is calculated in the pipeline; change who gets paired in the mode’s assignment code.

Implemented: 2026-11-21 · [Shared search pipeline](src/matching/pipeline.rs)

### D-008 — Scope storage to one collection; keep cross-pool transactions outside it

`RecordCollection` provides access to one pool’s records and indexes. For batch and live, `ScopedCollection` borrows one side of the existing `RecordStore`: reading selected fields from SQLite still goes through that backend, without copying records or opening a second store. A one-to-one claim spans **both** sides, so it stays with the CrossMap transaction owner, not either collection.

Enrollment needs no such claim. `EnrollState` owns one `MemoryCollection` and a write-ahead log, and `EnrollSession` operates on them. Load a single-pool `EnrollJobConfig` rather than manufacturing a second dataset or an empty CrossMap just to reuse the scorer.

Implemented: 2026-11-21 · [Collection trait and scoped store](src/store/collection.rs)

### D-009 — Export recorded decisions, not guesses from current records

Use the live SQLite database for current links and the enrollment log for the directed edges actually returned. Export seed and isolated records even when they have no edges. CSV, Parquet and SQLite should describe the same decisions, not invent new ones by rescoring records or interpreting matching flags.

Implemented: 2026-11-18 · [Enrollment output semantics](docs/enroll-mode.md#single-pool-output-semantics)

### D-010 — Reuse eligibility for repeated blocked batch queries

Cache the backend's actual record IDs, not an inferred country-to-record mapping. For large blocks, reuse the corresponding numeric BM25 IDs from the same immutable index. Limit the cache to the batch run, 64 entries and an estimated 64 MiB: a miss still follows the normal retrieval path. This avoids live invalidation, a new index format and another config option.

Implemented: 2026-11-22 · [Performance and correctness checks](benchmarks/batch/usearch_scaling/comparison/findings.md)
