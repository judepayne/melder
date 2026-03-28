# Live Mode Performance Improvements — Detailed Design

## Overview

Two complementary changes to the live-mode upsert pipeline:

1. **Replace Tantivy with a custom DashMap-based BM25 scorer** (`SimpleBm25`) —
   eliminates the commit/segment/reload cycle that is the dominant bottleneck
   in live mode. Instant write visibility, lock-free reads, zero commit
   overhead. See `CUSTOM_BM25_DESIGN.md` (project root) for full design.

2. **Pipeline parallelization** — run the independent stages of
   `upsert_record` concurrently via `rayon::scope`, hiding BM25/synonym/store
   work under encoding latency.

Together these eliminate the two primary sources of contention (BM25 lock
convoy and sequential stage stacking) and should yield 3–5× throughput
improvement at c=10.

---

## Part 1: Replace Tantivy with SimpleBm25

### Problem

Today, every upsert takes a **write lock** on `RwLock<BM25Index>` to call
Tantivy's `delete_term` + `add_document`. Every scoring path takes a **write
lock** on the opposite side to call `commit_if_ready()` (Tantivy segment
finalization + reader reload). These write locks block all concurrent readers
and writers on that side.

The `bm25_commit_batch_size: 100` benchmark proved this is the dominant
bottleneck: throughput jumped 3.2× (461→1,473 req/s) from reducing commit
frequency alone.

The root cause is an architectural mismatch: Tantivy is designed for
batch-ingest-then-query workloads (immutable segments, compressed posting
lists, merge policies). Melder's live mode interleaves single-document writes
with immediate reads — the opposite of Tantivy's design point.

### Solution

Replace the Tantivy-backed `BM25Index` with `SimpleBm25`: a DashMap-based
scorer that stores per-document term frequencies and global IDF statistics.
Full design in `CUSTOM_BM25_DESIGN.md`.

Key properties:
- **No commit step** — upserts update DashMap entries directly, instantly
  visible to subsequent reads
- **No outer RwLock needed** — DashMap handles concurrency internally via
  shard-level locks (nanosecond hold times)
- **Exhaustive blocked-set scoring** — scores all documents in the blocked set
  (typically 200–2,000) against the query. At B=500, K=10 query terms:
  ~5,000 hash lookups ≈ 1–2µs
- **Analytical self-score** — O(K) computation from corpus stats, no sentinel
  documents or index mutation

### Changes to `LiveSideState`

```rust
// BEFORE
pub struct LiveSideState {
    pub bm25_index: Option<RwLock<BM25Index>>,  // Tantivy-backed
    // ...
}

// AFTER
pub struct LiveSideState {
    pub bm25_index: Option<SimpleBm25>,  // DashMap-based, no outer lock needed
    // ...
}
```

### Changes to Session methods

**Upsert (same-side BM25)**:
```rust
// BEFORE: write lock + Tantivy add_document
let mut idx = bm25_mtx.write().unwrap();
idx.upsert(&id, &record);

// AFTER: no lock, direct DashMap update
bm25.upsert(&id, &record);
```

**Query (opposite-side BM25)**:
```rust
// BEFORE: write lock for commit, then read lock for query
{
    let mut w = opp_bm25_mtx.write().unwrap();
    w.commit_if_ready(batch_size);
}
let guard = opp_bm25_mtx.read().unwrap();
let results = guard.query_blocked(&text, top_k, &record, side);

// AFTER: no locks at all, direct call
let results = opp_bm25.score_blocked(&query_terms, &blocked_ids, top_k);
```

### Config changes

- `bm25_commit_batch_size` becomes **obsolete** — there is no commit step.
  The field can be kept for backward compatibility but is ignored by
  `SimpleBm25`. Deprecation warning on load.

### Migration safety

- Phase 1 builds `SimpleBm25` alongside Tantivy with a parity test suite
- Phase 2 switches live mode
- Phase 3 switches batch mode
- Phase 4 removes Tantivy dependency entirely

See `CUSTOM_BM25_DESIGN.md` §Migration Plan for details.

---

## Part 2: Pipeline Parallelization

### Problem

The upsert pipeline runs 7 sequential stages on a single `spawn_blocking`
thread, even though several stages have no data dependencies between them.
The longest stage (encoding, 4–6ms) blocks all subsequent work. BM25 upsert +
query, store + blocking, and synonym work all wait in line behind encoding,
even though they only need the input record.

### Dependency Analysis

```
                     record (input)
                   /    |       \        \
                  v     v        v        v
          [Encode]  [Store+Block] [BM25]   [Synonym]
          4-6ms      0.1ms        ~2µs      0.5ms
              |        |            |          |
              +--------+------------+----------+
              |  (join: vec, blocked_ids, bm25_results, syn_candidates)
              v
          [Vec index upsert (same-side)]     0.1ms
          [ANN search (opposite-side)]       0.3ms
              |
              v
          [score_pool]                       1-2ms
          [claim + WAL]                      0.01ms
```

Key insight: **encode, store+blocking, BM25, and synonym only need the input
record**. They are fully independent. ANN search needs the encoded vector
(from encode). `score_pool` needs everything.

Note: with `SimpleBm25`, the BM25 branch cost drops from 0.5–2ms (Tantivy
commit + query) to ~2µs (DashMap upsert + blocked-set scoring). This makes
the parallelization even more effective — the BM25 branch finishes almost
instantly, and the critical path is purely the encode time.

### Design

Restructure `upsert_record` + `upsert_record_inner` to use `rayon::scope`
for the four independent branches, then run the dependent stages sequentially
after the join.

#### Branch structure

```rust
fn upsert_record(&self, side: Side, record: Record) -> Result<UpsertResponse, SessionError> {
    // ... ID extraction, existing-record handling (crossmap break, etc.) ...
    // These preamble steps remain sequential — they're fast and have
    // side effects (crossmap break, WAL append) that must happen before
    // the parallel work.

    // WAL append (sub-microsecond)
    let _ = self.state.wal.append_upsert(side, &record);

    // Store insert + blocking update (fast, needed by blocking query branch)
    store.insert(side, &id, &record);
    store.mark_unmatched(side, &id);
    store.blocking_insert(side, &id, &record);

    // --- Parallel branches ---
    let mut combined_vec: Vec<f32> = Vec::new();
    let mut blocked_ids: Vec<String> = Vec::new();
    let mut bm25_result: Option<Bm25Result> = None;
    let mut syn_candidates: Vec<String> = Vec::new();

    // Extract &self fields into locals for borrow checker
    let encoder_pool = &self.state.encoder_pool;
    let this_bm25 = self.state.side(side).bm25_index.as_ref();
    let opp_bm25 = self.state.opposite_side(side).bm25_index.as_ref();
    let opp_syn = self.state.opposite_side(side).synonym_index.as_ref();
    // ... etc

    rayon::scope(|s| {
        let combined_out = &mut combined_vec;
        let blocked_out = &mut blocked_ids;
        let bm25_out = &mut bm25_result;
        let syn_out = &mut syn_candidates;

        // Branch 1: Encode combined vector
        s.spawn(move |_| {
            *combined_out = encode_combined(encoder_pool, &record, emb_specs, is_a_side)
                .unwrap_or_default();
        });

        // Branch 2: Blocking query
        s.spawn(move |_| {
            *blocked_out = if config.blocking.enabled {
                store.blocking_query(&record, side, opp)
            } else {
                store.ids(opp)
            };
        });

        // Branch 3: BM25 upsert (same-side) + score_blocked (opposite-side)
        s.spawn(move |_| {
            if let Some(bm25) = this_bm25 {
                bm25.upsert(&id, &record);  // DashMap, no lock
            }
            if let Some(opp) = opp_bm25 {
                let query_terms = opp.tokenise_query(&record, side);
                // blocked_ids not available yet — score_blocked called after join
                // Instead, just prepare the query terms and self_score here
                let self_score = opp.analytical_self_score(&query_terms);
                *bm25_out = Some(Bm25Result {
                    query_terms,
                    self_score,
                });
            }
        });

        // Branch 4: Synonym upsert + search
        s.spawn(move |_| {
            // upsert into same-side synonym index
            // lookup on opposite-side synonym index
            *syn_out = synonym_work(...);
        });
    });

    // --- Sequential tail (after join) ---

    // BM25 candidate generation (needs blocked_ids from branch 2 + query from branch 3)
    let (bm25_candidate_ids, bm25_scores) = if let Some(ref bm25_res) = bm25_result {
        if let Some(opp) = opp_bm25 {
            opp.score_blocked(&bm25_res.query_terms, &blocked_ids, bm25_candidates_n)
        } else {
            (Vec::new(), HashMap::new())
        }
    } else {
        (Vec::new(), HashMap::new())
    };

    // Vector index upsert (same-side) — needs combined_vec from branch 1
    if !combined_vec.is_empty() {
        if let Some(ref idx) = this_side.combined_index {
            let _ = idx.upsert(&id, &combined_vec, &record, side);
        }
    }

    // ANN search (opposite-side) — needs combined_vec from branch 1
    let ann_candidates = select_candidates(&combined_vec, ...);

    // score_pool — needs everything
    let results = pipeline::score_pool(
        &id, &record, side, &combined_vec,
        store, opp, opp_combined_index,
        &blocked_ids, config,
        ann_candidates_n, top_n,
        &bm25_candidate_ids, &bm25_scores,
        &syn_candidates, syn_dict,
    );

    // Claim loop + WAL (unchanged)
    // ...
}
```

#### BM25 scoring after the join

Note a subtlety: `score_blocked` needs both `blocked_ids` (from branch 2)
and `query_terms` (from branch 3). These come from different parallel
branches, so `score_blocked` must run **after** the join. However, with
`SimpleBm25` this call takes ~1–2µs — negligible compared to encode time.

The branch 3 work is therefore limited to:
1. Same-side upsert (DashMap, ~1µs)
2. Query tokenisation + self_score computation (~5–20µs)

The actual candidate scoring happens in the sequential tail. This is the
correct design — it avoids a dependency cycle between branches 2 and 3.

#### `try_match` path

`try_match` is read-only — no BM25 or synonym upserts. Parallelization is
simpler: encode, blocking query, and BM25/synonym query preparation all run
in parallel, then scoring runs sequentially.

#### Enroll mode

`enroll()` has the same structure but scores against the same side
(pool_side = query_side). The parallel branches are identical.

---

## Part 3: score_pool Signature Refactor

### Current signature

```rust
pub fn score_pool(
    query_id, query_record, query_side, query_combined_vec,
    pool_store, pool_side, pool_combined_index,
    blocked_ids, config,
    ann_candidates, bm25_candidates, top_n,
    bm25_ctx: Option<Bm25Ctx>,          // ← carries Tantivy reference
    synonym_index, synonym_dictionary,
) -> Vec<MatchResult>
```

### New signature

```rust
pub fn score_pool(
    query_id, query_record, query_side, query_combined_vec,
    pool_store, pool_side, pool_combined_index,
    blocked_ids, config,
    ann_candidates, top_n,
    // Pre-computed candidate sources:
    bm25_candidate_ids: &[String],
    bm25_scores: &HashMap<String, f64>,
    synonym_candidate_ids: &[String],
    synonym_dictionary: Option<&SynonymDictionary>,
) -> Vec<MatchResult>
```

Changes:
- `bm25_ctx: Option<Bm25Ctx>` removed — BM25 candidate generation happens
  at the call site
- `bm25_candidates: usize` removed — caller controls candidate count
- `synonym_index` removed — synonym candidate generation happens at the call
  site
- `bm25_candidate_ids` and `bm25_scores` added — pre-computed results
- `synonym_candidate_ids` added — pre-computed results

This is a cleaner separation: candidate generation (parallelized at call site)
vs union + scoring + classification (`score_pool`).

**Batch mode impact:** Batch mode callers update to pass pre-computed
candidates. Since batch mode already computes BM25/synonym results nearby,
this is a straightforward refactor.

---

## Part 4: Expected Performance Impact

### Per-request latency

| Stage | Before | After |
|-------|--------|-------|
| Encode | 4-6ms (sequential) | 4-6ms (parallel, critical path) |
| Store + blocking | 0.1ms (sequential) | Hidden under encode |
| BM25 upsert | 0.5ms write lock (sequential) | ~1µs DashMap (parallel, hidden) |
| BM25 query | 0.5-2ms (sequential, write+read lock) | ~2µs score_blocked (after join) |
| Synonym | 0.5ms (sequential) | Hidden under encode |
| ANN search | 0.3ms (sequential) | 0.3ms (after encode) |
| score_pool | 1-2ms (sequential) | 1-2ms (after join) |
| **Total** | **~7-11ms** | **~5.5-8.5ms** |

### Throughput at c=10

- **BM25 lock contention eliminated entirely** — no RwLock, no write lock
  convoy, no commit stalls. DashMap shard locks held for nanoseconds.
- **Encode + everything else no longer sequential** — frees blocking threads
  faster, reducing queue depth.
- **Combined effect:** 3-5× throughput improvement expected (from ~1,558 to
  ~5,000-8,000 req/s).

---

## Implementation Plan

### Phase 1: Build and validate SimpleBm25

**Files:**
- New: `src/bm25/simple.rs` — `SimpleBm25` implementation
- Modified: `src/bm25/mod.rs` — add `pub mod simple`

**Testing:**
- Parity test suite: same data + queries through both Tantivy and SimpleBm25,
  assert scores match within ±5%
- Unit tests: upsert, remove, score_blocked, analytical_self_score, edge cases
- Concurrent upsert + query stress test

### Phase 2: Switch live mode to SimpleBm25

**Files:**
- `src/state/live.rs` — replace `RwLock<BM25Index>` with `SimpleBm25`
- `src/session/mod.rs` — remove write locks, commit_if_ready; use SimpleBm25
  directly

**Testing:**
- All existing live-mode tests pass
- Live benchmark at c=1 and c=10
- `cargo test --all-features`

### Phase 3: Pipeline parallelization

**Files:**
- `src/session/mod.rs` — restructure upsert/try_match/enroll with rayon::scope
- `src/matching/pipeline.rs` — new score_pool signature
- Batch mode callers — update score_pool call sites

**Testing:**
- All existing tests pass
- Live benchmark: latency and throughput at c=1, c=4, c=10
- `RAYON_NUM_THREADS=1` regression test

### Phase 4: Migrate batch mode + remove Tantivy

**Files:**
- `src/state/state.rs` — build SimpleBm25 for batch mode
- Batch scoring callers — use SimpleBm25
- Remove `src/bm25/index.rs`
- Remove `tantivy` from `Cargo.toml`

**Testing:**
- Full experiment suite for match quality parity
- Batch benchmark comparison
- `cargo build --release --features usearch` (verify clean build)

### Phase 5: Benchmarking and tuning

- Run full benchmark suite at c=1, c=4, c=10
- Compare with/without coordinator (`encoder_batch_wait_ms`)
- Profile with `perf` to identify remaining hotspots
- Update `docs/performance.md` with new numbers

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BM25 score divergence from Tantivy | Medium | Match quality regression | Parity test suite in Phase 1; ±5% tolerance for fieldnorm differences |
| Rayon scope borrow checker issues | Medium | Delays implementation | Extract &self fields into locals; use move closures |
| Tokenisation mismatch | Low | Missed candidates | Shared tokenise() function; existing sanitize_query() normalises |
| Regression in batch mode from score_pool signature change | Low | Break batch scoring | Batch test suite catches this; straightforward update |
| Thread pool exhaustion (rayon + tokio blocking) | Low | Latency spike | Rayon = num_cpus threads; tokio blocking pool is separate and elastic |
| Missing Tantivy features | Very Low | Feature gap | Melder uses: TEXT indexing, TopDocs, BooleanQuery — all covered by SimpleBm25 |

---

## Non-Goals

- **Actor model / message passing** — too invasive for the expected gain.
  SimpleBm25 eliminates the BM25 lock problem entirely without actors.
- **CrossMap optimization** — nanosecond operations, confirmed not on critical
  path.
- **WAL async batching** — sub-microsecond buffered writes, not a meaningful
  bottleneck.
- **External vector DB** — in-process HNSW is correct for single-node.
- **Synonym index pending buffer** — sub-microsecond write lock; optimise only
  if profiling shows contention after phases 1-3.
