---
type: idea
module: general
status: active
tags: [scaling, bm25, database, architecture, candidate-selection]
related_code: [src/matching/pipeline.rs, src/matching/blocking.rs, src/matching/candidates.rs, src/vectordb/mod.rs]
---

# Scaling to Millions of Records

## The Problem

The melder's current architecture assumes both datasets fit in memory and that ONNX encoding of every record is affordable upfront. This works well into the hundreds of thousands — 100k x 100k completes a warm batch run in 12 seconds with usearch. But at 4M x 10M (40 trillion possible pairs), in-memory storage becomes impractical, cold encoding takes hours, and the dense vector index grows into tens of gigabytes.

The pipeline *structure* is sound — progressively narrowing candidates through cheaper phases before applying expensive scoring. What changes at scale is the implementation behind each phase.

## The Scaling Architecture

The proposal is to make the pipeline configurable across three tiers of scale, with each phase backed by an appropriate storage and retrieval strategy.

### Phase 0+1 — Common ID + Blocking (database-backed)

Currently: in-memory `DashMap` and `HashMap`. The blocking index is a hash lookup per field.

At scale: these become database-backed indices. The blocking query becomes `SELECT id FROM a_records WHERE country_code = ?` — a simple indexed column lookup that scales to hundreds of millions of rows with sub-millisecond latency. The common ID lookup is the same pattern.

**Storage choice: SQLite.** SQLite is an embedded, in-process database — it links directly into the binary via `rusqlite`, no separate service or daemon. The melder stays a single self-contained binary. SQLite's B-tree indices handle indexed equality queries in microseconds, which is exactly the blocking workload (`WHERE country_code = ?`). Its page cache (`PRAGMA cache_size`) keeps hot index pages in process memory after the first access, so repeated blocking lookups across millions of B records are effectively in-memory speed. The database file on disk is the backing store; the working set lives in the page cache for the duration of the run. SQLite scales comfortably to 50M+ rows, handles concurrent reads via WAL mode (important for Rayon-parallelised batch scoring), and compiles everywhere without issue — unlike RocksDB which pulls in a C++ build dependency with known MSVC pain.

This would be gated behind a config option: `storage: memory` (default, current behaviour) vs `storage: disk` (database-backed). The in-memory path stays untouched for the common case.

### Phase 2 — BM25 Candidate Selection (optional, replaces embedding ANN)

Currently: the melder encodes every record through the ONNX model at build time, stores the dense vectors in a combined index, and does ANN search (usearch) or brute-force scan (flat) to find `top_n` candidates.

At scale: replace this with a BM25 inverted index for candidate selection. BM25 is a sparse bag-of-words algorithm — no neural encoding, no dense vectors, no model download. It builds an inverted index over tokenised field values and retrieves candidates by token overlap weighted by IDF (inverse document frequency — rare tokens like "Bridgewater" score high, common tokens like "International" score low).

Why BM25 works for candidate selection:
- **Fast to build.** Tokenise and index 10M records in seconds, not the 30+ minutes that ONNX encoding would take.
- **Tiny index.** An inverted index for 10M short entity names is megabytes, not gigabytes.
- **Sub-millisecond queries.** Tantivy (Lucene for Rust) handles this natively.
- **Good enough recall for shortlisting.** BM25 won't capture "Deutsche Bank" → "German Bank" (no shared tokens), but it will capture "JPMorgan" → "JP Morgan Chase" and most real-world name variations. For candidate selection — where you need the true match to be *somewhere* in the top N — this is usually sufficient.

Why BM25 is not enough for final scoring:
- No semantic understanding. Synonyms, abbreviations, translations score zero.
- TF is always 1 for structured records (each field value is short). BM25's main advantage over raw TF-IDF — the TF saturation curve — adds nothing.
- IDF is the only useful signal, and it degenerates to rarity-weighted token overlap.

**How BM25 works.** BM25 scores each candidate document against a query using `Σ IDF(term) × TF_saturated(term, doc)` over shared tokens. At indexing time, every A-side record's text fields are tokenised and stored in an inverted index — a mapping from each token to the list of records containing it (posting lists). Two corpus-wide statistics are also stored: total document count (N) and average document length (avgdl). At query time, the B-side record's text is tokenised and each token's posting list is retrieved. Only records appearing in at least one posting list are scored — this is why retrieval is fast (you never scan the full 10M). IDF = `log((N - df + 0.5) / (df + 0.5) + 1)` measures term rarity: "jpmorgan" in 2 of 10M records gets IDF ~15.4; "international" in 3,400 records gets IDF ~8.0. TF_saturated applies a saturation curve and length normalisation to term frequency, but for short structured records (tf ≈ 1, dl ≈ avgdl), it collapses to roughly 1.0. In practice, **BM25 for entity matching ≈ IDF-weighted token overlap**: the rare, distinctive tokens in a name drive the score; common tokens like "Holdings" or "Inc" contribute almost nothing.

**Recall limitation.** BM25 can only find candidates that share at least one token with the query. If the true match has zero token overlap — "Deutsche Bank" vs "German Bank", "IBM" vs "International Business Machines" — BM25 will miss it. This is the fundamental trade-off: BM25 is orders of magnitude faster than embedding-based candidate selection and scales to much larger datasets, but it sacrifices recall on semantically-similar-but-lexically-different pairs. A generous shortlist size (e.g. `bm25_candidates: 100`) mitigates this for partial overlaps, but cannot fix zero-overlap cases. For corpora where this matters, keeping embedding in the scoring equation (Option A below) provides a safety net — the BM25 shortlist just needs to contain the right candidate *somewhere* in the 100, and embedding scoring will rank it correctly.

The BM25 shortlist size would be configured independently of the existing `top_n` (which controls embedding ANN search and final result count). Because BM25 queries are so cheap — sub-millisecond even at 10M records — you can afford a generous shortlist:

```yaml
candidate_selection: bm25
bm25_candidates: 100              # how many candidates BM25 retrieves (default: 100)
top_n: 20                         # how many final results to return per record
```

A `bm25_candidates` of 100 gives the downstream scoring phase a wide enough net to find the true match even when BM25's token overlap ranking is imperfect, while still reducing a 10M candidate pool by 99.999%. The cost of scoring 100 candidates with fuzzy + exact methods is negligible (microseconds per pair). If embedding is also in the scoring equation, 100 on-the-fly ONNX calls is still only ~100-300ms per record — tractable for batch jobs.

The Rust implementation would use [Tantivy](https://github.com/quickwit-oss/tantivy), which is mature, fast, and has a clean API. It supports custom tokenisers, which would be needed for entity name normalisation (strip "Ltd", "Inc", "Holdings" etc. before indexing).

### Phase 3 — Full Scoring (with or without embedding)

Currently: all `top_n` candidates are scored across every match field. Embedding scores are recovered by slicing the combined vector — no second ONNX call.

At scale with BM25 candidate selection, two options:

**Option A — Keep embedding in the scoring equation.** Encode just the BM25 shortlist candidates on the fly during full scoring, not the entire corpus upfront. At `bm25_candidates: 100`, that's 100 ONNX inference calls per B record. At ~1-3ms per call, that's 100-300ms per record — feasible at 4M B records if parallelised across cores (Rayon). Total wall time: hours, but tractable for overnight batch jobs. The key insight: you no longer need to encode and store vectors for 10M A-side records; you only encode the shortlist.

**Option B — Drop embedding entirely.** Use only fuzzy + exact scoring. No ONNX model loaded at all. Scoring is pure CPU string operations — sub-microsecond per field pair. This is the fastest option and removes the model dependency entirely. The trade-off is lower recall on semantically similar but character-dissimilar names.

The config would make this natural:

```yaml
# Current behaviour — embedding-based candidate selection + full scoring
candidate_selection: embedding    # default
top_n: 20

# Large scale — BM25 candidates, embedding still used in scoring equation
candidate_selection: bm25
bm25_candidates: 100              # BM25 shortlist size (default: 100)
top_n: 20                         # final results per record

# Very large scale — no embedding anywhere
candidate_selection: bm25
bm25_candidates: 100
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: fuzzy
    scorer: wratio
    weight: 0.70
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.30
  # no embedding fields → no ONNX model loaded
```

## Implementation Phases

This doesn't need to be built all at once. A natural progression:

### 1. BM25 candidate selection (medium effort)

Add Tantivy as an optional dependency behind a feature flag (`bm25`). Build an inverted index at startup from the configured text fields. When `candidate_selection: bm25` is set, use it instead of the vector index for Phase 2. The rest of the pipeline — full scoring, classification, crossmap — is unchanged.

This alone unlocks the "very large scale, no embedding" path, which may be the most immediately useful for million-record batch jobs.

### 2. On-the-fly embedding for BM25 shortlists (small effort, after step 1)

When BM25 is used for candidate selection but the scoring equation includes embedding fields, encode just the shortlisted candidates during full scoring instead of pre-encoding the entire corpus. Requires a small refactor of `score_pair()` to accept a lazy encoding path.

### 3. Database-backed storage (large effort)

Abstract the record store and blocking index behind a trait. Implement an on-disk backend (SQLite or RocksDB). Gate behind `storage: disk`. This is the biggest change — it touches data loading, blocking, crossmap persistence, and WAL replay. But the scoring pipeline itself is untouched.

## What This Doesn't Cover

- **Distributed matching.** At 100M+ records, a single machine may not have enough CPU or disk bandwidth. That requires sharded workers, a coordination layer, and a merge step — essentially a Spark-style shuffle job. Not worth designing until the need is real.
- **Incremental BM25.** The current proposal rebuilds the Tantivy index from scratch at startup. For live mode with BM25, the index would need real-time updates. Tantivy supports this, but it adds complexity.
- **BM25 + ANN hybrid retrieval.** Some systems (e.g. Elasticsearch with dense vector support) combine sparse and dense retrieval. This is interesting but adds significant complexity. The simpler "BM25 or embedding, not both" approach is more tractable.

## Key Dependencies

- [Tantivy](https://github.com/quickwit-oss/tantivy) — Rust full-text search engine (Apache 2.0 license)
- [rusqlite](https://github.com/rusqlite/rusqlite) — embedded SQLite for database-backed record storage and blocking indices

## Why Not Just Use Elasticsearch / OpenSearch?

The melder's value is in being a self-contained binary with no external dependencies. Adding an Elasticsearch requirement would fundamentally change the deployment model. The proposal here keeps everything in-process — Tantivy is an embedded library, not a service. The melder stays a single binary that you point at data and run.

---

See also: [[Constitution#2 One Scoring Pipeline]] (the scoring pipeline stays unified regardless of candidate selection strategy), [[Business Logic Flow]] (the phase structure this extends), [[Fine Tuning Embeddings]] (an alternative approach to improving recall that works at current scale), [[Discarded Ideas]] (prior scaling approaches that didn't work).
