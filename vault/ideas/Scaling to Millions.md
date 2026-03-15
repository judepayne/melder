---
type: idea
module: general
status: active
tags: [scaling, bm25, database, architecture, candidate-selection, scoring]
related_code: [src/matching/pipeline.rs, src/matching/blocking.rs, src/matching/candidates.rs, src/vectordb/mod.rs, src/scoring/mod.rs]
---

# Scaling to Millions of Records

## The Problem

The melder's current architecture assumes both datasets fit in memory and that ONNX encoding of every record is affordable upfront. This works well into the hundreds of thousands — 100k x 100k completes a warm batch run in 12 seconds with usearch. At 10M records, the memory footprint is manageable with vector quantisation (f16/bf16, already shipped): records ~5 GB, blocking indices ~1 GB, vectors ~7.5 GB with f16 — total ~13.5 GB, comfortable on a 32 GB machine. However, the real scaling bottleneck is **encoding startup time**: cold ONNX encoding of 10M records takes 30–90 minutes, which is impractical for batch jobs. The pipeline *structure* is sound — progressively narrowing candidates through cheaper phases before applying expensive scoring. What changes at scale is the implementation behind each phase.

## Summary

Three features address the main scaling constraints: **BM25 scoring and filtering** provides IDF-weighted token-overlap matching that complements embedding similarity (suppressing common tokens like "International" that untrained models over-weight), **vector quantisation** (f16/bf16, already shipped) halves the vector index memory footprint, and **database-backed storage** (SQLite) improves operational robustness for live mode. BM25 integrates into the existing scoring equation as `method: bm25` — just another weighted term alongside embedding, fuzzy, and exact. When embedding fields are present, ANN and BM25 work sequentially: ANN filters first (faster, broader semantic recall), then BM25 re-ranks and optionally narrows the shortlist (IDF-weighted token rarity). When no embedding fields are configured, BM25 serves as the sole candidate selection method — no ONNX model, no vector index, no encoding cost.

## Memory Footprint

A record is a `HashMap<String, String>`. A typical entity record has 5–10 fields averaging ~50 bytes per field (key + value) — roughly 500 bytes per record including HashMap overhead. The combined vector index stores one vector per record per embedding field; at 384 dimensions (the default model), that's 1,536 bytes per vector at f32.

**Total memory by scale (f32 vectors, single embedding field):**

| Records | Record storage | Blocking index | Vector index (f32) | Total |
|---|---|---|---|---|
| 100k | ~50 MB | ~10 MB | ~150 MB | ~210 MB |
| 1M | ~500 MB | ~100 MB | ~1.5 GB | ~2.1 GB |
| 10M | ~5 GB | ~1 GB | ~15 GB | ~21 GB |
| 50M | ~25 GB | ~5 GB | ~75 GB | ~105 GB |

At 1M records, 2 GB is nothing — any laptop handles that. At 10M, 21 GB is within reach of a 32–64 GB server. Memory only becomes a genuine constraint above 50M records.

**With vector quantisation (f16/bf16, already shipped):**

| Records | Vector index (f32) | Vector index (f16/bf16) | Total with f16 |
|---|---|---|---|
| 100k | ~150 MB | ~75 MB | ~135 MB |
| 1M | ~1.5 GB | ~750 MB | ~1.35 GB |
| 10M | ~15 GB | ~7.5 GB | ~13.5 GB |
| 50M | ~75 GB | ~37.5 GB | ~67.5 GB |

With f16 quantisation, 10M records fit comfortably on a 32 GB machine. The vector index is the dominant memory consumer at every scale — records and blocking indices are comparatively small.

## Vector Caching

The encoding startup cost — the primary scaling bottleneck — is mitigated by the vector cache, which persists encoded vectors to disk. Caching behaviour is asymmetric by design:

| | A-side | B-side |
|---|---|---|
| **Config** | `a_cache_dir` — required | `b_cache_dir` — optional |
| **Batch mode** | Encoded at startup, cached to disk. Warm start loads from cache. | Encoded every run by default. Only cached if `b_cache_dir` is configured. |
| **Live mode** | Same + saved at shutdown, `skip_deletes=true` preserves WAL-added vectors | Same |

The asymmetry is intentional. A is the reference/master dataset (stable, expensive to re-encode). B is the incoming/transient side (may change between runs — different batch files, new counterparties). So A always pays the encoding cost once and caches forever. B re-encodes each run unless the user explicitly opts into caching.

This means the encoding startup cost numbers in the Problem section are worst-case (cold start). On warm starts with a cached A-side, only new or changed records need re-encoding — the cache layer detects changes via a three-layer invalidation scheme (spec hash, manifest, text hash) and does incremental updates.

## Memory-Mapped Vector Index (usearch `view()`)

usearch exposes three loading modes for its on-disk index format:

```rust
index.load("index.usearch")  // full in-memory load — current melder behaviour
index.save("index.usearch")  // persist to disk
index.view("index.usearch")  // memory-mapped — file stays on disk, OS pages in/out
```

The current melder code (`src/vectordb/usearch_backend.rs`, `UsearchVectorDB::load()`) always uses `index.load()`, which pulls the entire HNSW graph into RAM. The `view()` call is an alternative that leaves the index on disk and lets the OS page-cache manage what stays in memory.

**What `view()` buys you at extreme scale:**

| | `load()` (current) | `view()` (mmap) |
|---|---|---|
| RAM usage | Full index in RAM | Virtual memory; OS pages in/out |
| First-query latency | Fast (warm) | Slow until OS page-cache is hot |
| ANN search throughput | Consistently fast | Unpredictable — HNSW random-access graph traversal causes frequent page faults on cold data |
| Disk requirement | Cache file only | Same cache file, accessed via mmap |

**The honest trade-off:** `view()` lets you trade RAM for disk I/O, but HNSW is one of the worst-case access patterns for mmap — graph traversal hops randomly across the file, so a cold cache means one disk read per hop. Search latency can balloon from milliseconds to seconds.

**When it becomes relevant:**

| Scale | f16 vector index | Recommendation |
|---|---|---|
| 10M records | ~7.5 GB | `load()` — fits easily on any server |
| 50M records | ~37.5 GB | `load()` — fits on a 64 GB server |
| 100M+ records | ~75 GB+ | `view()` as last resort for batch jobs tolerant of slower search; live mode needs sharding |

**Implementation note:** Switching from `load()` to `view()` in `UsearchVectorDB::load()` is a small code change — one line per block. It could be exposed as a config option (e.g. `vector_index_mode: mmap`) alongside `vector_quantisation`. For batch jobs at extreme scale where search latency is less critical than memory pressure, this is a viable escape hatch. It should not be used for live mode (`meld serve`) where latency guarantees matter.

This is a **future escape hatch for 100M+ record scale**, not a general strategy. The primary scaling levers remain quantisation (already shipped) and BM25 candidate selection (eliminates encoding startup cost).

## The Scaling Architecture

The proposal is to make the pipeline configurable across three tiers of scale, with each phase backed by an appropriate storage and retrieval strategy.

### Phase 0+1 — Common ID + Blocking (in-memory or database-backed)

Currently: in-memory `DashMap` and `HashMap`. The blocking index is a hash lookup per field.

At scale: in-memory blocking indices work fine at 10M records — memory cost is ~1 GB, which is modest compared to the vector index at 7.5–15 GB. The blocking query is a simple hash lookup per field, which is fast and scales well.

**Database-backed storage for operational robustness (live mode only).** SQLite is an embedded, in-process database — it links directly into the binary via `rusqlite`, no separate service or daemon. The melder stays a single self-contained binary. The melder automatically selects the right backend based on mode: `meld run` (batch) always uses in-memory storage for maximum speed; `meld serve` (live) always uses SQLite for operational robustness. This mixed design ensures batch jobs get the fastest possible performance while live servers benefit from durability and transactional consistency. SQLite offers operational benefits: transactional consistency (a single SQLite transaction per upsert instead of coordinating 6+ in-memory structures), fast restarts (no WAL replay — SQLite is durable), and clean crossmap persistence (SQL constraints enforce bijection instead of CSV flush). SQLite's B-tree indices handle indexed equality queries in microseconds, which is exactly the blocking workload (`WHERE country_code = ?`). Its page cache (`PRAGMA cache_size`) keeps hot index pages in process memory after the first access, so repeated blocking lookups across millions of B records are effectively in-memory speed. The database file on disk is the backing store; the working set lives in the page cache for the duration of the run. SQLite scales comfortably to 50M+ rows, handles concurrent reads via WAL mode (important for Rayon-parallelised batch scoring), and compiles everywhere without issue — unlike RocksDB which pulls in a C++ build dependency with known MSVC pain.

### Phase 2 — Sequential ANN + BM25 Candidate Selection

Currently: the melder encodes every record through the ONNX model at build time, stores the dense vectors in a combined index, and does ANN search (usearch) or brute-force scan (flat) to find `top_n` candidates.

**The new design: ANN and BM25 work sequentially, not as alternatives.**

When both embedding fields and a `method: bm25` term are configured, the pipeline runs two candidate selection stages in sequence:

```
Full block → ANN filter (top ann_candidates) → BM25 re-rank/filter (top bm25_candidates) → full scoring
```

ANN goes first because:
1. **It's faster** — O(log N) per query vs sub-millisecond for BM25, but over a much larger candidate set.
2. **It has broader recall** — semantic similarity captures synonyms, abbreviations, and translations that BM25 cannot.
3. **In production, a fine-tuned model eliminates BM25's advantage** — domain-specific training teaches the model to down-weight common tokens like "International Holdings Ltd", making BM25's IDF correction unnecessary. BM25 is most valuable with untrained, off-the-shelf models.

BM25 then operates on the ANN shortlist — a small set (typically 50–200 candidates), so the cost is trivial. It re-ranks by IDF-weighted token overlap, potentially filtering the shortlist down further. This is exactly where BM25 adds value: suppressing candidates that ANN over-ranked because common tokens inflated embedding similarity.

**The two filter sizes are user-visible controls, not implementation details:**

```yaml
ann_candidates: 200      # how many candidates ANN retrieves from the full block
bm25_candidates: 50      # how many candidates BM25 keeps after re-ranking ANN's shortlist
top_n: 20                # final results returned per record
```

The filter sizes affect scoring semantics, not just performance:

| Setting | Effect |
|---|---|
| `ann_candidates: 200, bm25_candidates: 50` | ANN casts a wide net; BM25 narrows to the best token-overlap matches |
| `ann_candidates: 50, bm25_candidates: 50` | BM25 re-ranks but doesn't filter (same size in and out) — acts purely as a scoring enrichment |
| `ann_candidates: 200, bm25_candidates: 200` | Neither method filters — both just contribute scores. Wide net, maximum recall, higher scoring cost |

**No-embedding mode (one-off jobs, no encoding cost):**

When no embedding fields are configured, ANN is skipped entirely. BM25 queries the full block directly and serves as the sole candidate selection method:

```yaml
match_fields:
  - method: bm25
    weight: 0.50
  - field_a: legal_name
    field_b: counterparty_name
    method: fuzzy
    scorer: wratio
    weight: 0.35
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.15

bm25_candidates: 100
top_n: 20
```

No ONNX model loaded, no vector index built, no encoding cost. Just tokenise, build inverted index, go. This is the fast-start path for one-off matching jobs where the user doesn't want to pay the embedding cost.

**Why BM25 works for candidate selection:**
- **Fast to build.** Tokenise and index 10M records in seconds, not the 30+ minutes that ONNX encoding would take.
- **Tiny index.** An inverted index for 10M short entity names is megabytes, not gigabytes.
- **Sub-millisecond queries.** Tantivy (Lucene for Rust) handles this natively.
- **Good enough recall for shortlisting.** BM25 won't capture "Deutsche Bank" → "German Bank" (no shared tokens), but it will capture "JPMorgan" → "JP Morgan Chase" and most real-world name variations.

**Recall limitation.** BM25 can only find candidates that share at least one token with the query. If the true match has zero token overlap — "Deutsche Bank" vs "German Bank", "IBM" vs "International Business Machines" — BM25 will miss it. When used as a secondary filter after ANN, this limitation is mitigated: ANN already found semantically similar candidates regardless of token overlap, and BM25 is only re-ranking within that set. When used as the sole filter (no-embedding mode), a generous shortlist size (e.g. `bm25_candidates: 100`) mitigates partial overlaps but cannot fix zero-overlap cases.

**How BM25 works.** BM25 scores each candidate document against a query using `Σ IDF(term) × TF_saturated(term, doc)` over shared tokens. At indexing time, every A-side record's text fields are tokenised and stored in an inverted index — a mapping from each token to the list of records containing it (posting lists). Two corpus-wide statistics are also stored: total document count (N) and average document length (avgdl). At query time, the B-side record's text is tokenised and each token's posting list is retrieved. Only records appearing in at least one posting list are scored — this is why retrieval is fast (you never scan the full 10M). IDF = `log((N - df + 0.5) / (df + 0.5) + 1)` measures term rarity: "jpmorgan" in 2 of 10M records gets IDF ~15.4; "international" in 3,400 records gets IDF ~8.0. TF_saturated applies a saturation curve and length normalisation to term frequency, but for short structured records (tf ≈ 1, dl ≈ avgdl), it collapses to roughly 1.0. In practice, **BM25 for entity matching ≈ IDF-weighted token overlap**: the rare, distinctive tokens in a name drive the score; common tokens like "Holdings" or "Inc" contribute almost nothing.

The Rust implementation would use [Tantivy](https://github.com/quickwit-oss/tantivy), which is mature, fast, and has a clean API. It supports custom tokenisers, which would be needed for entity name normalisation (strip "Ltd", "Inc", "Holdings" etc. before indexing).

### Phase 3 — Full Scoring (unified equation)

All candidates that survive the filtering phases are scored using the same weighted equation: `final_score = Σ(weight_i × score_i)`. BM25 is simply another term in this equation — `method: bm25` — alongside embedding, fuzzy, and exact.

**The `method: bm25` term requires no field specification** because it operates across all text fields in the record. The BM25 inverted index is built from all text fields at startup, and the score for a candidate pair is looked up from the index during scoring.

**BM25 score normalisation.** Raw BM25 scores are unbounded — they depend on corpus statistics (IDF values, document lengths). To fit into the [0, 1] scoring equation, BM25 scores are normalised by the query's self-score: `normalised_bm25 = raw_score / self_score`, where `self_score` is the BM25 score of the query matched against itself (the theoretical maximum). This gives a clean [0, 1] range that is comparable across queries.

**Example configurations:**

```yaml
# Production — fine-tuned model, BM25 as secondary signal
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.40
  - method: bm25
    weight: 0.20
  - field_a: legal_name
    field_b: counterparty_name
    method: fuzzy
    scorer: wratio
    weight: 0.25
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.15

ann_candidates: 100
bm25_candidates: 50
top_n: 20
```

```yaml
# One-off job — no embedding cost, BM25 as primary signal
match_fields:
  - method: bm25
    weight: 0.50
  - field_a: legal_name
    field_b: counterparty_name
    method: fuzzy
    scorer: wratio
    weight: 0.35
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.15

bm25_candidates: 100
top_n: 20
# no ann_candidates — no embedding fields, no ANN phase
```

```yaml
# Current behaviour — embedding only, no BM25 (unchanged)
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.50
  - field_a: legal_name
    field_b: counterparty_name
    method: fuzzy
    scorer: wratio
    weight: 0.30
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20

top_n: 20
# no bm25 method → no BM25 index built, no BM25 filtering
```

**All three candidate-selection layers — blocking, ANN, BM25 — are independently optional.** Each narrows the candidate set; without any of them you get brute-force O(N²). The pipeline adapts automatically based on config:

| Blocking? | ANN? | BM25? | Candidate selection pipeline |
|---|---|---|---|
| Yes | Yes | Yes | Block → ANN within block → BM25 re-rank → score |
| Yes | Yes | No | Block → ANN within block → score |
| Yes | No | Yes | Block → BM25 within block → score |
| Yes | No | No | Block → score all within block |
| No | Yes | Yes | ANN over full dataset → BM25 re-rank → score |
| No | Yes | No | ANN over full dataset → score (current no-blocking behaviour) |
| No | No | Yes | BM25 over full dataset → score |
| No | No | No | Score every A against every B — O(N²), small datasets only |

## Design Decisions

Key design decisions agreed during the scaling design process:

### BM25 field selection

BM25 automatically indexes only the fields that appear in `method: fuzzy` or `method: embedding` match field entries — the "text-like" fields the user already identified as needing soft matching. Exact-match fields like `country_code` or `currency` are not tokenised into the BM25 index. This is zero-config and avoids polluting the index with short code values.

### Candidate filter defaults and validation

**Defaults:**
- `ann_candidates`: 50
- `bm25_candidates`: 10
- `top_n`: 5

**Validation constraints (enforced at config load time):**
- If both ANN and BM25 enabled: `ann_candidates >= bm25_candidates >= top_n`
- If only ANN enabled: `ann_candidates >= top_n`
- If only BM25 enabled: `bm25_candidates >= top_n`

If the user violates these constraints, config validation rejects with a clear error message. `top_n` is always "final output size" — it never controls candidate retrieval.

### BM25 in live mode

BM25 is supported in live mode (`meld serve`) from day one. Per the Constitution's symmetry principle, BM25 indices are built for **both** A-side and B-side. Both indices are updated on every upsert via Tantivy's real-time writer API (`IndexWriter::add_document()` + batched `commit()`). The BM25 index is treated identically to the vector index and blocking index — all three are updated together on upsert:

| Index | A-side upsert | B-side upsert |
|---|---|---|
| Blocking index | Update A blocking | Update B blocking |
| Vector index | Encode + insert into A vectors | Encode + insert into B vectors |
| BM25 index | Tokenise + insert into A inverted index | Tokenise + insert into B inverted index |

### ANN and BM25 are fully optional

Both ANN (embedding) and BM25 are independently optional. The pipeline adapts based on which methods are present in `match_fields`. See the full pipeline table in Phase 3 above for all eight combinations of blocking × ANN × BM25.

---

## Memory Budget and Disk-Backed Storage

### The Competition for RAM

All three major in-memory components compete for the same RAM. Their per-record footprints at f16 quantisation (single 384-dim embedding field):

| Component | Bytes per record | 10M records | 50M records | Access pattern |
|---|---|---|---|---|
| Record store (DashMap) | ~500 B | ~5 GB | ~25 GB | Moderate — semi-sequential during candidate scoring |
| Blocking index | ~100 B | ~1 GB | ~5 GB | **Very hot** — queried for every B-side record |
| Vector index (f16) | ~750 B | ~7.5 GB | ~37.5 GB | **Random access** — HNSW graph hops are worst-case for cache misses |
| **Total** | **~1,350 B** | **~13.5 GB** | **~67.5 GB** | |

Two structural observations:

1. **The vector index is ~1.5× the size of the record store** at every scale. This ratio is constant — it depends only on embedding dimension and quantisation, not record content.
2. **The vector index benefits far more from RAM than the record store.** HNSW graph traversal is random-access — a cache miss per hop, and O(log N) hops per search. The record store is accessed more sequentially (candidates are scored one by one), so an LRU disk cache handles it well. The blocking index is the hottest structure but also the smallest.

### When Each Component Needs Disk

| Machine RAM | Safe fully in-memory up to | Record store pressure begins | Vector index pressure begins |
|---|---|---|---|
| 16 GB | ~8M records | > ~8M (record store > ~4 GB) | > ~8M (vector > ~6 GB) |
| 32 GB | ~20M records | > ~20M | > ~20M |
| 64 GB | ~40M records | > ~40M | > ~40M |

Note: **both thresholds arrive at roughly the same record count** because the 1.5× ratio is fixed. The record store and vector index move to disk together, not independently.

### SQLite as the Batch-Mode Record Store (Disk-Backed)

At 50M+ records the record store alone exceeds available RAM on any typical machine. This is the case where SQLite becomes useful **even in batch mode** — not for operational robustness (the live-mode motivation) but for **memory pressure relief**. The record store is stored on disk; a configurable in-memory page cache holds the hot subset.

SQLite provides three orthogonal memory controls:

| Pragma | What it does | Key detail |
|---|---|---|
| `PRAGMA cache_size = -N` | Page cache upper bound in RAM | Negative N = `N × 1024` bytes. **Default is `-2000` = only 2 MB** — must be overridden explicitly or performance collapses. Session-scoped; reverts on close. |
| `PRAGMA mmap_size = N` | Memory-map the DB file up to N bytes | Bypasses the page cache; OS manages hot pages via virtual memory. Useful for sequential reads; complements the page cache rather than replacing it. |
| `PRAGMA page_size = N` | B-tree page size (default 4 KB) | 8 KB or 16 KB pages reduce B-tree depth and I/Os per lookup. At ~500 bytes per record, a 4 KB page holds only ~8 records; 8 KB holds ~16. Set before any data is written. |

The cache size formula:
```sql
-- Set page cache to N megabytes:
PRAGMA cache_size = -(N * 1024);
-- Example: 4 GB cache:
PRAGMA cache_size = -4194304;
```

**The blocking index benefits automatically.** When the record store is in SQLite, the blocking index lives in a separate table with a B-tree index. Blocking queries (`WHERE country_code = ?`) are pure equality lookups — they hit the B-tree index pages, which are accessed far more frequently than record data pages and therefore stay hot in the LRU page cache naturally. No special handling is needed.

### The Memory Budget Allocation Strategy

Given a fixed RAM budget B (either detected automatically or configured), the optimal split between SQLite page cache and vector index RAM is determined by their relative access costs:

- **Blocking index pages** are kept warm by the LRU page cache automatically (most-frequently-accessed pages in SQLite). No special allocation needed.
- **Vector index (HNSW)** is the highest-priority RAM consumer. HNSW random access is catastrophically slow on cache miss — latency multiplies by O(log N × disk_latency) per search. Allocate ~70% of available budget here.
- **SQLite page cache** gets the remaining ~30%. At this allocation, the most frequently-accessed record pages (blocking index B-tree + recently scored records) remain warm.

**The ratio is ~70/30 vector/records, regardless of scale.** This follows from the 1.5× size ratio and the much higher access cost of HNSW cache misses versus B-tree record lookups.

## Implementation Phases

This doesn't need to be built all at once. A natural progression:

### 1. BM25 as scoring method and candidate filter (medium effort)

Add Tantivy as an optional dependency behind a feature flag (`bm25`). Implement `method: bm25` as a new scoring method — no field specification, operates across all text fields, normalised to [0, 1] via self-score. Build the BM25 inverted index at startup from all text fields. When `method: bm25` is present in `match_fields`, the BM25 index is built and BM25 scores are computed during Phase 3. When `bm25_candidates` is configured, BM25 also acts as a candidate filter: in sequential mode after ANN (when embedding fields exist), or as the sole filter (when no embedding fields exist). The rest of the pipeline — classification, crossmap — is unchanged. Add `ann_candidates` and `bm25_candidates` as top-level config keys alongside the existing `top_n`.

This alone unlocks two new paths: (a) BM25 as a scoring enrichment alongside embeddings, suppressing common-token noise from untrained models, and (b) the "no embedding" path for one-off jobs where the user doesn't want to pay the encoding cost.

### 2. Database-backed storage (independent, large effort)

Abstract the record store and blocking index behind a trait. Implement an on-disk backend (SQLite) for live mode (`meld serve`). Batch mode (`meld run`) always uses in-memory storage — no performance regression. The melder picks the right backend automatically based on which mode is running; no config option needed. This is an **operational robustness improvement**, not a scaling enabler. The key benefits are: transactional state consistency (a single SQLite transaction per upsert instead of coordinating 6+ in-memory structures), fast restarts (no WAL replay — SQLite is durable), and clean crossmap persistence (SQL constraints enforce bijection instead of CSV flush). It does NOT meaningfully reduce memory pressure — the vector index (the largest memory consumer) stays in memory regardless, and f16/bf16 quantisation (already shipped) halves that cost. Records and blocking indices are small relative to the vector index. `rusqlite` is always compiled in (not feature-gated) — the compile cost is negligible (~5 seconds for the SQLite C amalgamation). This is the biggest change — it touches data loading, blocking, crossmap persistence, and WAL replay. But the scoring pipeline itself is untouched. It is independent of BM25 — you can build either one without the other, in any order.

### 3. Memory-mapped vector index (future escape hatch, small effort when needed)

For 100M+ record workloads where RAM is genuinely exhausted even with f16 quantisation, switch `UsearchVectorDB::load()` from `index.load()` to `index.view()` on a per-block basis. This is a single-line change per block but trades consistent ANN search latency for lower peak RAM. Expose as a config option (`vector_index_mode: mmap`). Only meaningful for batch jobs tolerant of slower search; not suitable for live mode. Do not build this until the need is real — quantisation + BM25 covers the vast majority of realistic workloads comfortably.

## What This Doesn't Cover

- **Distributed matching.** At 100M+ records, a single machine may not have enough CPU or disk bandwidth. That requires sharded workers, a coordination layer, and a merge step — essentially a Spark-style shuffle job. Not worth designing until the need is real.
- **Incremental BM25.** The current proposal rebuilds the Tantivy index from scratch at startup. For live mode with BM25, the index would need real-time updates. Tantivy supports this, but it adds complexity.

## Key Dependencies

- [Tantivy](https://github.com/quickwit-oss/tantivy) — Rust full-text search engine (Apache 2.0 license)
- [rusqlite](https://github.com/rusqlite/rusqlite) — embedded SQLite for database-backed record storage and blocking indices

## Why Not Just Use Elasticsearch / OpenSearch?

The melder's value is in being a self-contained binary with no external dependencies. Adding an Elasticsearch requirement would fundamentally change the deployment model. The proposal here keeps everything in-process — Tantivy is an embedded library, not a service. The melder stays a single binary that you point at data and run.

---

See also: [[Constitution#2 One Scoring Pipeline]] (the scoring pipeline stays unified regardless of candidate selection strategy), [[Business Logic Flow]] (the phase structure this extends), [[Fine Tuning Embeddings]] (an alternative approach to improving recall that works at current scale), [[Discarded Ideas]] (prior scaling approaches that didn't work).
