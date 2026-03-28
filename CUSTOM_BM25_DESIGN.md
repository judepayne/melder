# Custom BM25 Scorer — Design Document

## Motivation

Tantivy is a full-text search engine library designed for web-scale document
retrieval: immutable segments, compressed posting lists, skip-list
intersection, WAND early termination. Melder uses approximately 5% of this
capability — a single in-RAM index with one tokenised field, queried
immediately after writes.

The architectural mismatch is the dominant bottleneck in live mode. Tantivy's
commit/segment/reload cycle costs 0.5–5ms per commit, and every write requires
a commit before the next read can see it. The `bm25_commit_batch_size: 100`
benchmark proved this: throughput jumped 3.2× from 461 to 1,473 req/s by
reducing commit frequency alone.

Rather than working around Tantivy (pending buffers, batched commits), this
design replaces it with a purpose-built BM25 scorer that eliminates the
write-visibility problem entirely.

### Scale requirements

The initial deployment target is 2M × 2M records with concurrent live upserts.
With skewed blocking (e.g. 40% of records in one country), individual blocks
can reach 800,000 records. The design must handle this from day one — not as
a future optimisation.

---

## Design Overview

`SimpleBm25` is a single struct with two internal strategies:

- **Small blocks (B ≤ threshold):** exhaustive scoring — iterate all blocked
  docs, compute BM25 for each. O(B × K).
- **Large blocks (B > threshold):** inverted index query — look up query terms
  in posting lists partitioned by block, score only docs that share at least
  one term. O(K × log P + R × K) where R = result set size.

The threshold is configurable (default: 5,000). Both paths use the same BM25
formula, same tokenisation, same IDF statistics. The caller sees a single
`score_blocked()` interface — the strategy selection is internal.

---

## Data Structures

```rust
pub struct SimpleBm25 {
    // --- Per-document storage ---

    /// doc_id → { term → term_frequency }
    doc_terms: DashMap<String, HashMap<String, u32>>,

    /// doc_id → total token count (document length for BM25)
    doc_lengths: DashMap<String, u32>,

    // --- Global corpus statistics (for IDF) ---

    /// term → number of documents containing this term
    doc_freq: DashMap<String, usize>,

    /// Total number of indexed documents.
    total_docs: AtomicUsize,

    /// Sum of all document lengths (for avgdl computation).
    total_tokens: AtomicU64,

    // --- Inverted index (for large-block queries) ---

    /// term → sorted Vec<PostingEntry>
    /// Sorted by (block_key, doc_id) for binary-search access per block.
    postings: DashMap<String, Vec<PostingEntry>>,

    // --- Configuration ---

    /// Which record fields to concatenate and tokenise.
    fields: Vec<Bm25FieldPair>,

    /// Which side this index serves (A or B).
    side: Side,

    /// Blocking field pairs — used to extract block keys from records
    /// for posting list partitioning.
    blocking_fields: Vec<BlockingFieldPair>,

    /// Block size threshold: blocks smaller than this use exhaustive
    /// scoring; larger blocks use inverted index lookup.
    exhaustive_threshold: usize,
}

/// A single entry in a posting list.
#[derive(Clone)]
struct PostingEntry {
    /// Hash of the document's blocking key(s), for fast block filtering.
    block_key: u64,
    /// Compact document identifier (index into a side table, or the
    /// String doc_id — see §Compact Doc IDs below).
    doc_id: String,
    /// Term frequency in this document.
    tf: u32,
}
```

### Why both `doc_terms` and `postings`?

- `doc_terms` (forward index) is needed for: (a) exhaustive scoring of small
  blocks, (b) efficient upsert — removing old terms requires knowing what
  terms the document previously had.
- `postings` (inverted index) is needed for: large-block queries where
  exhaustive scoring would be too slow.

The memory overhead of maintaining both is modest: at 2M docs × 12 terms
average, `doc_terms` holds ~24M entries (~500MB with String keys), and
`postings` holds ~24M PostingEntry values (~400MB). Total ~900MB — comparable
to what Tantivy would use for a 2M-document in-RAM index.

### Block key computation

Each document's block key is computed from its blocking field values, the
same way the blocking index does it:

```rust
fn compute_block_key(record: &Record, blocking_fields: &[BlockingFieldPair], side: Side) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for bfp in blocking_fields {
        let key = match side {
            Side::A => &bfp.field_a,
            Side::B => &bfp.field_b,
        };
        if let Some(val) = record.get(key) {
            val.trim().to_lowercase().hash(&mut hasher);
        }
    }
    hasher.finish()
}
```

When blocking is disabled, all documents get the same block_key (0), and
the inverted index degenerates to a global index.

---

## Write Path: `upsert(id, record)`

```
1. Extract text:      concat configured field values → single string
2. Tokenise:          lowercase + split → Vec<String>
3. Compute TF:        count occurrences of each unique token → HashMap<String, u32>
4. Compute block_key: hash of blocking field values
5. If updating (doc_terms contains id):
   a. Load old term map from doc_terms[id]
   b. For each old term:
      - Decrement doc_freq[term] (remove entry if reaches 0)
      - Remove old PostingEntry from postings[term] for this doc_id
   c. Subtract old doc_length from total_tokens
   d. Decrement total_docs
6. Store new data:
   a. doc_terms[id] = new term→tf map
   b. doc_lengths[id] = new token count
7. Update corpus stats:
   a. For each new term: increment doc_freq[term]
   b. For each new term: insert PostingEntry into postings[term]
      (maintain sorted order by block_key)
   c. Add new token count to total_tokens
   d. Increment total_docs
```

### Cost

O(K_old + K_new) where K_old/K_new = unique tokens in old/new document.
Typically K ≈ 10–15 for entity names.

Each step touches DashMap shard locks held for nanoseconds. The most
expensive operation is maintaining sorted order in posting lists (step 7b) —
binary search for insertion point: O(log P) per term where P = posting list
length. At 2M docs with 12 terms, average posting list length is
2M × 12 / vocab_size. With a typical vocabulary of ~50k unique terms,
average P ≈ 480. So log(480) ≈ 9 comparisons per term, 9 × 12 ≈ 108
comparisons total. Sub-microsecond.

### Posting list insertion

Posting lists are sorted by `(block_key, doc_id)`. Insertion uses binary
search to find the correct position, then `Vec::insert`. At average P=480,
the `Vec::insert` shifts ~240 elements — about 2KB of memory movement.
This is fast (memcpy) but if profiling shows it as a bottleneck, the posting
list can be switched to a `BTreeMap<(u64, String), u32>` for O(log P) insert
without shifts.

### Instant write visibility

After `upsert()` returns, the document is immediately visible in both
`doc_terms` (for exhaustive scoring) and `postings` (for inverted index
queries). No commit step, no reader reload.

---

## Read Path: `score_blocked(query_text, blocked_ids, top_k)`

### Step 1: Tokenise and compute IDF

```rust
let query_terms = tokenise(query_text);  // Vec<(String, u32)> with term frequencies
let n = self.total_docs.load(Ordering::Relaxed) as f64;
let avg_dl = self.total_tokens.load(Ordering::Relaxed) as f64 / n.max(1.0);

// Pre-compute IDF for each query term
let term_idfs: Vec<(String, u32, f64)> = query_terms.iter().filter_map(|(term, tf)| {
    let df = self.doc_freq.get(term).map(|v| *v).unwrap_or(0);
    if df == 0 { return None; }  // term not in corpus — skip
    let idf = ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
    Some((term.clone(), *tf, idf))
}).collect();
```

### Step 2: Strategy selection

```rust
if blocked_ids.len() <= self.exhaustive_threshold {
    self.score_exhaustive(&term_idfs, blocked_ids, avg_dl, top_k)
} else {
    self.score_inverted(&term_idfs, blocked_ids, avg_dl, top_k)
}
```

### Exhaustive path (small blocks, B ≤ threshold)

```rust
fn score_exhaustive(&self, term_idfs, blocked_ids, avg_dl, top_k) -> Vec<(String, f64)> {
    let mut scores: Vec<(String, f64)> = blocked_ids.iter().filter_map(|doc_id| {
        let terms = self.doc_terms.get(doc_id)?;
        let dl = self.doc_lengths.get(doc_id).map(|v| *v as f64).unwrap_or(0.0);

        let mut score = 0.0;
        for (term, _qtf, idf) in term_idfs {
            let tf = terms.get(term).copied().unwrap_or(0) as f64;
            if tf > 0.0 {
                score += idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl));
            }
        }
        if score > 0.0 { Some((doc_id.clone(), score)) } else { None }
    }).collect();

    // Partial sort for top-K
    if scores.len() > top_k {
        scores.select_nth_unstable_by(top_k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores
}
```

Cost: O(B × K) where B = blocked set size, K = query terms with non-zero IDF.
At B=500, K=10: ~1–2µs.

### Inverted index path (large blocks, B > threshold)

```rust
fn score_inverted(&self, term_idfs, blocked_ids, avg_dl, top_k) -> Vec<(String, f64)> {
    // Compute the block_key for this query (all blocked_ids share the same key)
    // In practice, extract from the first blocked_id's stored block_key,
    // or compute from the query record passed alongside.
    let block_key = ...;

    // For each query term, binary-search the posting list to find entries
    // matching this block_key, accumulate scores per doc_id.
    let mut doc_scores: HashMap<String, f64> = HashMap::new();

    for (term, _qtf, idf) in term_idfs {
        if let Some(postings) = self.postings.get(term) {
            // Binary search to first entry with this block_key
            let start = postings.partition_point(|e| e.block_key < block_key);
            // Scan entries with matching block_key
            for entry in &postings[start..] {
                if entry.block_key != block_key { break; }
                let dl = self.doc_lengths.get(&entry.doc_id)
                    .map(|v| *v as f64).unwrap_or(0.0);
                let tf = entry.tf as f64;
                let term_score = idf * (tf * (K1 + 1.0))
                    / (tf + K1 * (1.0 - B_PARAM + B_PARAM * dl / avg_dl));
                *doc_scores.entry(entry.doc_id.clone()).or_insert(0.0) += term_score;
            }
        }
    }

    // Partial sort for top-K
    let mut scores: Vec<(String, f64)> = doc_scores.into_iter().collect();
    if scores.len() > top_k {
        scores.select_nth_unstable_by(top_k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores
}
```

Cost analysis for B=800,000 (large skewed block):

1. **Term lookup:** K=10 DashMap lookups for posting lists → O(K) = ~10 lookups
2. **Binary search:** for each term, find block segment in posting list.
   Average posting list P ≈ 2M × 12 / 50k_vocab ≈ 480. Binary search: O(log 480) ≈ 9.
   Total: 10 × 9 = 90 comparisons.
3. **Block segment scan:** only entries matching this block_key. If 40% of
   docs are in this block, ~40% of a posting list entry falls in this segment:
   ~192 entries per term. Scan: 10 × 192 = 1,920 entries.
4. **Score accumulation:** 1,920 HashMap inserts/updates.
5. **Partial sort:** O(R) where R = unique docs found (at most 1,920).

Total: ~2,000 operations — about 5–20µs. Compare to exhaustive scoring of
800k docs: ~8M operations, ~2–3ms. The inverted index is **100–500× faster**
for large blocks.

### Block key for inverted path

The inverted path needs the block_key to binary-search posting lists. This
is derived from the query record's blocking field values. The caller
(`Session`) already has the query record, so the block_key can be computed
and passed alongside `blocked_ids`. Alternatively, `score_blocked` can accept
an optional `block_key: Option<u64>` parameter — when present, use the
inverted path regardless of block size.

When blocking is disabled (all docs in one block), the block_key is 0 and the
inverted path scans the entire posting list — equivalent to a traditional
inverted index query.

---

## Handling Multiple Blocking Keys

A subtlety: melder supports multiple blocking field pairs (e.g. country_code
AND sector). The blocking query returns the union of matches across any
blocking pair. This means `blocked_ids` may contain documents from different
blocks.

For the inverted index path, this requires scanning multiple block segments
per posting list. The `score_blocked` method accepts a `block_keys: &[u64]`
parameter (one per blocking field pair that matched). For each query term,
binary-search to each block_key segment and scan it.

For the exhaustive path, this doesn't matter — it iterates `blocked_ids`
directly regardless of which block they came from.

---

## Self-Score (BM25 Normalisation)

Melder normalises raw BM25 scores to [0, 1] by dividing by a "self-score" —
the BM25 score a document would get if queried against itself.

The custom scorer computes self-scores analytically:

```rust
pub fn analytical_self_score(&self, query_terms: &[(String, u32)], query_len: u32) -> f64 {
    let n = self.total_docs.load(Ordering::Relaxed) as f64;
    if n == 0.0 { return 0.0; }
    let avg_dl = self.total_tokens.load(Ordering::Relaxed) as f64 / n;

    let mut score = 0.0;
    for (term, tf) in query_terms {
        let df = self.doc_freq.get(term).map(|v| *v).unwrap_or(0);
        if df == 0 { continue; }
        let idf = ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
        let tf_f = *tf as f64;
        let dl = query_len as f64;
        score += idf * (tf_f * (K1 + 1.0)) / (tf_f + K1 * (1.0 - B + B * dl / avg_dl));
    }
    score
}
```

O(K) computation, no index mutation, no sentinel documents.

---

## BM25 Formula

Standard Okapi BM25 with k1=1.2, b=0.75 (Tantivy defaults):

```
score(q, d) = Σ_{t ∈ q} IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))

where:
  IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
  N      = total_docs
  df(t)  = doc_freq[t]  (number of documents containing term t)
  tf(t,d)= term frequency of t in document d
  |d|    = doc_lengths[d]  (number of tokens in document d)
  avgdl  = total_tokens / total_docs
  k1     = 1.2
  b      = 0.75
```

Tantivy uses a fieldnorm compression step (lossy u8 mapping of document
length). The custom scorer uses exact document lengths. Scores will differ
slightly from Tantivy — this is acceptable and likely more accurate.

---

## Tokenisation

Shared between indexing and querying. Must exactly match to guarantee scoring
consistency.

```rust
fn tokenise(text: &str) -> Vec<String> {
    text.chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect()
}

fn term_frequencies(tokens: &[String]) -> HashMap<String, u32> {
    let mut freqs = HashMap::new();
    for token in tokens {
        *freqs.entry(token.clone()).or_insert(0) += 1;
    }
    freqs
}
```

This matches the behaviour of `sanitize_query()` + Tantivy's default
`SimpleTokenizer` (lowercase + split on non-alphanumeric).

---

## Thread Safety

- `DashMap` for `doc_terms`, `doc_lengths`, `doc_freq`, `postings` — shard-level
  locks, effectively lock-free for reads, nanosecond write contention
- `AtomicUsize` / `AtomicU64` for `total_docs`, `total_tokens` — lock-free
- No outer `RwLock` needed — `SimpleBm25` is safe to share via `Arc<SimpleBm25>`
  across request threads without any external synchronisation
- Concurrent upserts to the same doc_id: last writer wins (DashMap shard lock
  serialises access to the same key). Brief window where doc_terms and postings
  may be inconsistent for the same doc (old postings, new doc_terms). This is
  harmless — worst case, a single query sees slightly stale posting data for
  one document. The inconsistency resolves on the next upsert or query.

---

## Memory Usage at Scale

At 2M documents × 12 average tokens per document:

| Structure | Entries | Approx Memory |
|-----------|---------|---------------|
| `doc_terms` (DashMap<String, HashMap<String, u32>>) | 2M docs × 12 terms | ~500MB |
| `doc_lengths` (DashMap<String, u32>) | 2M entries | ~80MB |
| `doc_freq` (DashMap<String, usize>) | ~50k unique terms | ~4MB |
| `postings` (DashMap<String, Vec<PostingEntry>>) | ~50k terms × ~480 entries | ~400MB |
| `total_docs`, `total_tokens` | 2 atomics | negligible |
| **Total** | | **~1GB per side** |

At 2M × 2M, both sides together: ~2GB. This is comparable to Tantivy's
memory usage for in-RAM indices at the same scale, with the advantage of
zero commit overhead.

If memory becomes a concern, `doc_terms` can be dropped in favour of
reconstructing term maps from `postings` during upsert (slower upserts but
saves ~500MB per side). Or compact doc_ids can reduce String overhead — see
§Future Optimisations.

---

## Migration Plan

### Phase 1: Build `SimpleBm25` alongside Tantivy

- Implement `SimpleBm25` in `src/bm25/simple.rs`
- Public interface: `new`, `build` (from RecordStore), `upsert`, `remove`,
  `score_blocked`, `analytical_self_score`, `query_text_for`
- Parity test suite: feed identical data to both Tantivy and SimpleBm25,
  assert BM25 candidate rankings match (top-K sets overlap ≥ 90%) and scores
  are within ±10% tolerance (fieldnorm compression accounts for the gap)
- Unit tests for tokenisation, IDF computation, exhaustive vs inverted path,
  edge cases (empty docs, unknown terms, single-doc corpus)

### Phase 2: Switch live mode to `SimpleBm25`

- Replace `RwLock<BM25Index>` in `LiveSideState` with `Arc<SimpleBm25>`
- Update session methods: direct calls, no locks, no commit_if_ready
- Deprecate `bm25_commit_batch_size` config (log warning if set)
- Live benchmark: compare throughput at c=1 and c=10

### Phase 3: Migrate batch mode

- Build `SimpleBm25` from RecordStore during batch init
- Verify batch benchmark results and match quality parity
- Run full experiment suite

### Phase 4: Remove Tantivy dependency

- Remove `bm25/index.rs`
- Remove `tantivy` from `Cargo.toml` (~40 transitive dependencies removed)
- Clean build verification: `cargo build --release --features usearch`
- Final benchmark pass

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BM25 score divergence from Tantivy | Medium | Match quality regression | Parity test suite in Phase 1; ±10% tolerance |
| Tokenisation mismatch | Low | Missed candidates | Shared tokenise(); existing sanitize_query() normalises |
| Memory at 2M scale | Low | ~1GB/side, manageable | Comparable to Tantivy; can drop doc_terms if needed |
| Large-block inverted path correctness | Medium | Wrong candidates | Block-segmented scan is straightforward; test with synthetic skewed data |
| Posting list insertion perf (sorted Vec) | Low | Slow upserts | Switch to BTreeMap if profiling shows hotspot |
| Missing Tantivy features | Very Low | Feature gap | Melder uses TEXT + TopDocs + BooleanQuery only |

---

## Future Optimisations

### Compact doc IDs

Replace `String` doc_ids in PostingEntry with `u32` indices into a side
table. Saves ~40 bytes per entry at 2M scale (~100MB saving per side).
Requires a `DashMap<String, u32>` id→index mapping.

### SIMD scoring

The BM25 inner loop (multiply + divide per term per doc) is
SIMD-friendly. At 800k blocked docs with 10 terms, that's 8M arithmetic
operations — SIMD could reduce to ~2M vector ops. Unlikely to be needed
given the inverted index path reduces the doc count to ~2k, but available
if profiling shows scoring as a bottleneck.

### Adaptive threshold

Instead of a fixed `exhaustive_threshold`, compute it from the actual
posting list sizes for the query terms. If all query terms are rare
(short posting lists), the inverted path wins even at small block sizes.
If all terms are common (long posting lists), exhaustive may be better.
The crossover depends on selectivity.
