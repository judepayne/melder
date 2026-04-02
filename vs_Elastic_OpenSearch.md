# Melder vs Elasticsearch / OpenSearch for Entity Resolution

_Analysis of whether Elasticsearch (Elastic) or OpenSearch can replicate
Melder's entity resolution scoring pipeline, with cost, performance, and
scalability comparisons._

Last updated: 2026-04-02

---

## Contents

1. [Executive Summary](#executive-summary)
2. [Melder's Scoring Pipeline (Recap)](#melders-scoring-pipeline-recap)
3. [Elasticsearch Score Combination Methods](#elasticsearch-score-combination-methods)
4. [The Absolute Scoring Gap](#the-absolute-scoring-gap)
5. [Synonyms](#synonyms)
6. [OpenSearch Capabilities](#opensearch-capabilities)
7. [Licensing & Cost](#licensing--cost)
8. [Performance: In-Process vs Distributed](#performance-in-process-vs-distributed)
9. [Sharding and the Entity Resolution Problem](#sharding-and-the-entity-resolution-problem)
10. [Could ES/OpenSearch Be Used as a Candidate Generator?](#could-esopensearch-be-used-as-a-candidate-generator)
11. [Implications for Melder's Scalability](#implications-for-melders-scalability)
12. [Recommendation](#recommendation)

---

## Executive Summary

**Can Elasticsearch or OpenSearch replicate Melder's entity resolution
pipeline?** No, not faithfully. They can approximate candidate retrieval
(finding plausible matches), but the scoring semantics are fundamentally
different. The core issue is that ES/OpenSearch produce **relative scores**
(dependent on other documents in the result set), while Melder produces
**absolute scores** (a property of the pair alone). Entity resolution
depends on score consistency for threshold-based classification — a score
of 0.85 must always mean the same thing. This is not achievable in
ES/OpenSearch's built-in hybrid search.

If an external search system were needed, **OpenSearch is the better
choice** — it offers the same hybrid search capabilities as Elasticsearch
Platinum but is fully open source (Apache 2.0) with no licensing cost.

For Melder's current target (bank entity resolution, up to a few million
records), the in-process architecture is **5–15x faster**, produces
consistent absolute scores, costs a fraction, and is operationally trivial
(single binary, no cluster management).

---

## Melder's Scoring Pipeline (Recap)

Melder's production scoring configuration (Experiment 12):

```
Composite = Σ(field_score × weight) / Σ(weight)

Components:
  name_emb:  0.30   embedding cosine similarity (name field)
  addr_emb:  0.20   embedding cosine similarity (address field)
  bm25:      0.50   corpus-aware lexical scoring, self-score normalised to 0–1
  synonym:   0.20   additive bonus, outside normalisation denominator
```

Key properties:

1. **Per-field weighted average** — not rank fusion.
2. **Absolute 0–1 scores** — `score_pair(A, B)` returns the same value
   regardless of what other records exist in the index.
3. **Additive synonym bonus** — separated from weight normalisation.
   When synonym score is 0.0, the weight is excluded from the denominator.
4. **Exhaustive pair scoring** — every blocked candidate is scored, not
   just top-K.
5. **Per-field decomposition** — individual field scores are available
   for review UIs, debugging, and confidence assessment.
6. **CrossMap bijection** — atomic 1:1 matching (each A maps to at most
   one B), enforced under a single lock.

---

## Elasticsearch Score Combination Methods

### RRF (Reciprocal Rank Fusion)

RRF **discards actual scores** and uses only rank position:

```
RRF_score = Σ  1 / (k + rank_i)
```

Where `k` is a constant (default 60) and `rank_i` is the document's
position in each retriever's ranked list.

**Problems for entity resolution:**

- Scores are rank-derived, not calibrated. An RRF score of 0.032 has no
  inherent meaning.
- All retrievers carry **equal weight** — there is no way to say "BM25
  is 50%, embeddings are 30%."
- Adding or removing a record changes ranks and therefore changes scores
  for every other candidate.
- Requires Platinum license.

**Verdict:** Not suitable for entity resolution.

### Linear Retriever (ES 9.x, Platinum)

The `linear` retriever is a weighted combination of normalised
sub-retriever scores:

```
Score = Σ(weight_i × Normalizer(score_i))
```

Normalisers: `minmax`, `l2_norm`, or `none`.

Example:

```json
{
  "retriever": {
    "linear": {
      "retrievers": [
        {
          "retriever": { "knn": { "field": "name_vector", ... } },
          "weight": 0.30
        },
        {
          "retriever": { "knn": { "field": "addr_vector", ... } },
          "weight": 0.20
        },
        {
          "retriever": { "standard": { "query": { "match": { "name": "..." } } } },
          "weight": 0.50
        }
      ],
      "normalizer": "minmax"
    }
  }
}
```

This is the closest analog to Melder's weighted composite. It supports
per-retriever weights and multiple kNN fields.

**But the scores are relative** — see the next section.

### Legacy `boost` (Free tier)

```json
{
  "query": { "match": { "name": { "query": "...", "boost": 0.5 } } },
  "knn": { "field": "name_vector", ..., "boost": 0.3 }
}
```

Raw score sum without normalisation. Since BM25 scores (typically 5–15)
and cosine similarity (0–1) are on completely different scales, the
weights don't mean what they appear to mean.

### `script_score` (Free tier — The Escape Hatch)

You can write a custom scoring script that computes absolute scores:

```json
{
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": {
        "source": "cosineSimilarity(params.qv, 'name_vec') * 0.3 + cosineSimilarity(params.qv_addr, 'addr_vec') * 0.2",
        "params": { "qv": [...], "qv_addr": [...] }
      }
    }
  }
}
```

This would produce stable cosine scores. But:
- Runs on **every document** matching the inner query — no HNSW
  acceleration. Brute-force scan. Catastrophically slow at scale.
- You'd need to reimplement BM25 in Painless (ES's scripting language).
- No synonym bonus mechanism.
- Still no CrossMap bijection.

The one path to stable scores in ES sacrifices all the performance
benefits of having a search engine.

### Summary of Combination Methods

| Method | Weighted? | Stable Scores? | License | Melder-compatible? |
|---|---|---|---|---|
| `linear` (minmax) | Yes | No — depends on result set | Platinum | Partially, unstable scores |
| `linear` (no normaliser) | Yes | Raw scores stable but incomparable scales | Platinum | Weights meaningless |
| RRF | No — all equal | No — rank-based | Platinum | No |
| Legacy `boost` | Yes | Incomparable scales | Free | Unusable |
| `script_score` | Full control | Yes — but brute-force, no HNSW | Free | Theoretically, but too slow |

---

## The Absolute Scoring Gap

This is the fundamental incompatibility between ES/OpenSearch hybrid
search and entity resolution.

### How minmax normalisation breaks thresholds

When ES combines scores from multiple retrievers, it normalises them
onto the same scale using minmax:

```
normalised = (raw_score - min_score) / (max_score - min_score)
```

The `min_score` and `max_score` are the worst and best scores **within
the current top-K result set**.

**Example — Monday.** You query for "Goldman Sachs International". The
top-5 BM25 results are:

| Candidate | Raw BM25 | Normalised |
|---|---:|---:|
| Goldman Sachs International Ltd | 12.4 | 1.00 |
| Goldman Sachs Group Inc | 10.1 | 0.77 |
| GS International | 8.3 | 0.59 |
| Gold Standard Capital | 6.0 | 0.36 |
| Goldmark Financial | 4.2 | 0.18 |

After weighting with kNN, "GS International" gets a combined score of
**0.86**. Above your auto_match threshold of 0.85. Matched.

**Example — Tuesday.** Someone adds "Goldman Sachs Intl" to the index
(raw BM25: 11.8). Now the top-5 is:

| Candidate | Raw BM25 | Normalised |
|---|---:|---:|
| Goldman Sachs International Ltd | 12.4 | 1.00 |
| Goldman Sachs Intl | 11.8 | 0.93 |
| Goldman Sachs Group Inc | 10.1 | 0.72 |
| GS International | 8.3 | 0.50 |
| Gold Standard Capital | 6.0 | 0.27 |

The raw BM25 score for "GS International" hasn't changed (still 8.3).
But the normalised score dropped from 0.59 to 0.50 because a better
candidate appeared and shifted the min/max window. After weighting,
"GS International" now scores **0.78**. Below the threshold. Missed.

**Nothing about that pair changed.** Same query, same candidate, same
text similarity. But the score changed because the *other records in
the result set* changed.

### Why Melder doesn't have this problem

Melder's `score_pair(A, B)` computes each field score independently:

- **Embedding cosine**: `dot(vec_a, vec_b)` — a property of those two
  vectors only. Always 0–1.
- **BM25**: normalised by the analytical self-score (`score(A, A)`), not
  by other candidates. Always 0–1.
- **Fuzzy**: Levenshtein ratio of those two strings. Always 0–1.

The composite `Σ(field_score × weight) / Σ(weight)` depends only on the
pair being scored. Add a million records to the index — the score for
this specific pair is unchanged. Remove half the index — still the same.

This is what makes `auto_match: 0.85` a stable, meaningful threshold
that can be tuned once and trusted permanently. In ES/OpenSearch, you'd
need to re-tune thresholds as data evolves, and even then there's no
guarantee that a threshold working today works tomorrow.

---

## Synonyms

### Melder's approach

Synonym matching is a **separate scoring method** (`method: synonym`)
that contributes a 0–1 score independently, added to the composite with
its own weight (0.20). Crucially, it's **additive** — excluded from the
base score normalisation denominator when the synonym score is 0.0. This
prevents the synonym weight from diluting the signal for non-acronym
pairs.

### ES/OpenSearch synonyms

Synonyms are token-level expansions configured as part of an analyser:

- **Index-time**: `synonym` filter expands terms at indexing. Cannot be
  changed without reindexing.
- **Query-time**: `synonym` filter expands terms at query time. More
  flexible.
- **Synonyms API**: Dynamically updateable synonym sets via REST
  (available in free tier).

When "GS" is a synonym for "Goldman Sachs", the query expands to include
both terms. This changes the BM25 score but does not provide a separate,
additive bonus that can be weighted independently.

**Gap:** No mechanism exists in ES/OpenSearch for an additive scoring
bonus that sits outside the main score combination.

---

## OpenSearch Capabilities

### Hybrid search

OpenSearch's hybrid search uses a **search pipeline** with a
`normalization-processor`:

```json
{
  "normalization": { "technique": "min_max" },
  "combination": {
    "technique": "arithmetic_mean",
    "parameters": { "weights": [0.3, 0.5, 0.2] }
  }
}
```

This is equivalent to ES's `linear` retriever. The `arithmetic_mean`
with weights computes:

```
score = (w1 × norm_score1 + w2 × norm_score2 + ...) / (w1 + w2 + ...)
```

Same minmax normalisation, same relative scoring problem. But it's
**free** — all features are included.

### Neural search / kNN

OpenSearch supports `knn_vector` fields with HNSW (via nmslib, faiss,
or Lucene engines). Notably, OpenSearch supports **FAISS as a backend**,
which ES does not. FAISS can be more memory-efficient for large-scale
vector search.

### License

**OpenSearch is fully open source under the Apache 2.0 license.**
All features — hybrid search, neural search, kNN, ML model deployment,
normalisation processor — are free. There are no paid feature tiers.
OpenSearch became a Linux Foundation project (OpenSearch Software
Foundation) in September 2024.

---

## Licensing & Cost

### Elasticsearch

| Feature | Free/Basic | Platinum | Enterprise |
|---|---|---|---|
| Vector search, kNN | Yes | Yes | Yes |
| BM25 full-text search | Yes | Yes | Yes |
| Synonym management | Yes | Yes | Yes |
| **`linear` retriever (weighted hybrid)** | **No** | **Yes** | **Yes** |
| **RRF** | **No** | **Yes** | **Yes** |
| ML model deployment on nodes | No | Yes | Yes |

The `linear` retriever (the only weighted combination method) and RRF
are both **Platinum-only**.

### Pricing comparison (1M records, production)

| | Elasticsearch (Platinum) | OpenSearch (AWS Managed) | OpenSearch (Self-managed) | Melder |
|---|---|---|---|---|
| License | ~$131/mo starting | Included | Free (Apache 2.0) | Free (single binary) |
| Infrastructure | 3-node cluster | 3-node cluster | 3-node cluster | Single server |
| Monthly cost | $1,500–2,000 | $770–1,200 | $800–1,000 + ops | $100–200 |
| Operational overhead | Cluster management | Managed by AWS | Cluster + DevOps FTE | Zero |

### The SSPL controversy

In 2021, Elastic changed Elasticsearch from Apache 2.0 to SSPL + Elastic
License to prevent cloud providers (primarily AWS) from offering it as a
managed service. AWS responded by forking Elasticsearch 7.10.2 into
OpenSearch under Apache 2.0. In 2024, Elastic added AGPL as a third
license option. For self-hosted use, ELv2 is permissive. For offering
ES as a service to third parties, a commercial license is required.

---

## Performance: In-Process vs Distributed

### Latency comparison

| Operation | Melder (in-process) | ES/OpenSearch (distributed) |
|---|---|---|
| Vector lookup | ~10ns (memory pointer) | ~0.5–2ms (network hop + deserialise) |
| BM25 score | ~100ns (DashMap) | ~5–20ms (shard query + merge) |
| Full scoring pipeline | ~0.6ms/record (10k) | ~10–30ms/record |
| Candidate generation | ~1ms total | ~20–50ms total |

### Throughput comparison

| Configuration | Melder | ES/OpenSearch (estimated) |
|---|---|---|
| Batch, 10k×10k (warm) | 33,738 rec/s | ~2,000–5,000 rec/s |
| Batch, 100k×100k (warm) | 10,539 rec/s | ~1,000–3,000 rec/s |
| GPU encoding (1M, CoreML) | 1,828 rec/s | External inference |
| Live, 10k×10k (c=10) | 1,558 req/s, p95 25.6ms | ~200–500 req/s |

Melder is **5–15x faster** because there is zero serialisation, zero
network hops, and the HNSW index is a direct memory structure. For entity
resolution, where you score millions of pairs, this gap is enormous.

### kNN implementation comparison

| System | Implementation | Notes |
|---|---|---|
| Melder (usearch) | In-process HNSW, direct memory access | Fastest per-query |
| ES | Lucene HNSW or DiskBBQ | General-purpose, HTTP overhead |
| OpenSearch | HNSW via nmslib, faiss, or Lucene | FAISS option can be more memory-efficient |
| Standalone usearch | Same as Melder | 2–5x faster than ES at same recall |
| FAISS (IVF) | Inverted file index | GPU-acceleratable for brute-force |

---

## Sharding and the Entity Resolution Problem

### How ES/OpenSearch shard data

Documents are routed to shards via:

```
shard = hash(routing_key) % number_of_primary_shards
```

Default `routing_key` is the document `_id`. Custom routing is supported,
so `routing=country_code` naturally maps to Melder's blocking strategy.

### Problem 1: BM25 IDF per-shard

By default, ES/OpenSearch compute **IDF (Inverse Document Frequency) per
shard**, not globally. This means the same term gets different BM25 scores
depending on which shard the candidate lives on.

Example: "Goldman" appears in 100 of 200k docs on shard 1
(IDF = log(200k/100)) but 50 of 200k on shard 2 (IDF = log(200k/50)).
A match at 0.86 on one shard might score 0.83 on another shard with
different term distributions.

**Workaround:** `search_type=dfs_query_then_fetch` does a pre-pass for
global term statistics, adding a round-trip but giving correct IDF.

### Problem 2: kNN recall penalty across shards

Each shard runs HNSW independently on its subset of data. With 5 shards
and `num_candidates=100`, recall@10 typically drops 3–8% compared to
single-shard. Increasing `num_candidates` to 200 narrows the gap to
1–3% but costs latency.

### Problem 3: Cross-block matches

If you route by `country_code`, "Deutsche Bank AG" (DE) and "Deutsche
Bank London Branch" (GB) land on different shards. Melder's exact
prefilter catches these (same LEI, different country) because it runs
before blocking. In a sharded system, you'd need either:

- Route ALL queries to ALL shards (defeats the purpose of blocking)
- Maintain a secondary unrouted index for prefilter fields
- Run a separate cross-shard exact-match pass
- Accept missed cross-block matches

### The fundamental tension

Entity resolution requires comparing each record against potentially
every other record. Sharding reduces the comparison space but introduces
three risks: IDF inconsistency, kNN recall loss, and cross-block misses.
The workarounds (DFS queries, high num_candidates, secondary indices)
add latency and complexity, partially negating the scalability benefit.

---

## Could ES/OpenSearch Be Used as a Candidate Generator?

Yes. This is the architecture that makes sense if you need external
search infrastructure:

```
ES/OpenSearch (candidate retrieval)       Application Layer (scoring)
┌──────────────────────────┐             ┌──────────────────────────┐
│ For each B record:       │             │                          │
│  1. kNN (name_vec, k=50) │             │  4. Union candidates     │
│  2. kNN (addr_vec, k=50) ├────────────►│  5. score_pair() each    │
│  3. BM25 query           │             │  6. Classify (threshold) │
│                          │  candidate  │  7. CrossMap claim       │
│  Return IDs + fields     │  IDs + data │                          │
└──────────────────────────┘             └──────────────────────────┘
```

Candidate generation doesn't need stable scores — you just need a set
of plausible candidates. Top-K ranking is fine for that. The real scoring
happens outside ES with absolute scores, field decomposition, additive
synonym bonus, and bijection enforcement.

This is essentially what Melder already does internally — blocking +
BM25 filter + ANN search are all candidate generation, and `score_pair()`
is the final scoring. The difference is whether those candidate stages
run in-process or over the network.

### The performance tax

| Operation | Melder (in-process) | ES as candidate gen |
|---|---|---|
| kNN search (name) | ~0.5ms | ~5–15ms |
| kNN search (addr) | ~0.5ms | ~5–15ms |
| BM25 candidates | ~0.1ms | ~5–20ms |
| Fetch candidate records | ~10ns each | ~1–5ms |
| **Total candidate gen** | **~1ms** | **~20–50ms** |
| Full scoring (same) | ~0.1ms/candidate | ~0.1ms/candidate |

At 1M records:

- **Melder in-process**: 1M × ~1ms = ~17 minutes
- **ES candidate gen**: 1M × ~35ms = ~10 hours

The 20–50x overhead comes entirely from network serialisation, shard
coordination, and HTTP round-trips. The actual HNSW and BM25 algorithms
are comparably fast — it's the distributed architecture that costs.

---

## Implications for Melder's Scalability

### When single-machine hits its limit

| Records per side | RAM (approx) | Feasible? |
|---|---|---|
| 1M | ~10 GB | Easy |
| 5M | ~50 GB | High-end workstation |
| 10M | ~100 GB | Specialised server |
| 50M | ~500 GB | Not practical in-memory |

Beyond ~10M records per side, you need either Melder's SQLite backend
(trades RAM for disk I/O) or a distributed approach.

### Blocking IS sharding

The key insight: Melder's blocking strategy is already a natural sharding
boundary. The step from "blocking within one process" to "blocking across
processes" is smaller than it appears:

1. **Shard by blocking key.** Partition A records by blocking key
   (e.g., `country_code`). Each shard is a self-contained Melder
   instance with its own `RecordStore`, `VectorDB`, `SimpleBm25`,
   and `BlockingIndex`.

2. **Route B records to the correct shard.** Each B record's blocking
   key determines which shard to query. Since Melder uses AND blocking,
   each B record goes to exactly one shard.

3. **Cross-block recovery.** Run exact prefilter as a separate global
   pass (hash lookup against all shards). This is cheap — O(1) per
   prefilter field — and catches LEI/ISIN cross-block cases.

4. **Global CrossMap.** The bijection constraint must be global (an A
   record on shard 1 can't be matched to two different B records from
   different shards). A lightweight coordination layer or distributed
   lock would be needed. This is the hardest part.

5. **Global BM25 IDF.** Either compute global term statistics once at
   startup (Melder's `SimpleBm25` already stores them in DashMap, which
   could be replicated or shared across shards), or accept that
   per-shard IDF is close enough with reasonably uniform blocking
   distributions.

### Advantages over using ES/OpenSearch for this

- **Absolute scores preserved.** Each shard runs Melder's `score_pair()`
  with the same semantics. Global BM25 IDF can be pre-computed.
- **No kNN recall penalty.** Each shard's HNSW index covers its full
  blocking partition — no cross-shard candidate merging needed.
- **No serialisation overhead.** Within a shard, everything is
  in-process. Only the coordination layer (CrossMap, prefilter) crosses
  process boundaries.
- **Simpler operations.** Each shard is a standalone `meld` binary.
  No JVM tuning, no cluster state, no split-brain risk.

---

## Recommendation

### For Melder's current use case

Stay with the in-process architecture. It is 5–15x faster, produces
consistent absolute scores, costs a fraction of a managed search
cluster, and the GPU encoding work (pool=12, batch=256, 1,828 rec/s on
CoreML) makes the encoding bottleneck manageable even at 1M scale.

### If you need an external search system

Use OpenSearch, not Elasticsearch. OpenSearch offers the same hybrid
search capabilities as ES Platinum but is fully open source (Apache 2.0)
with no licensing cost. Use it as a candidate retrieval layer, not as the
scoring engine. Pull candidates back into Melder for scoring,
classification, and CrossMap enforcement.

### If you need horizontal scaling

Shard by blocking key with per-shard Melder instances, a global CrossMap
coordinator, and a global exact prefilter pass. This preserves absolute
scoring semantics, avoids the IDF and kNN recall problems of search
engine sharding, and keeps the per-shard performance at Melder's
in-process speed.

### Summary table

| Capability | Melder | ES (Platinum) | OpenSearch |
|---|---|---|---|
| Weighted multi-method scoring | Weighted average, absolute 0–1 | `linear` retriever, relative scores | `arithmetic_mean`, relative scores |
| Score stability | Absolute — pair-dependent only | Relative — changes with other docs | Relative |
| Synonym bonus | Additive, separate method | Token expansion only (modifies BM25) | Token expansion only |
| Exhaustive pair scoring | Yes, all blocked candidates | Top-K only | Top-K only |
| 1:1 bijection (CrossMap) | Atomic `claim()` | Not available | Not available |
| Per-field score decomposition | Yes | Not structured | Not structured |
| Batch throughput (10k×10k) | 33,738 rec/s | ~2,000–5,000 rec/s | ~2,000–5,000 rec/s |
| GPU encoding | Built-in (CoreML/CUDA) | External inference | External inference |
| License cost | Free (single binary) | ~$1,500–2,000/mo (Platinum) | Free (Apache 2.0) |
| Horizontal scaling | Not yet (single-process) | Native sharding | Native sharding |
| Operational overhead | Zero | Cluster management | Cluster management |
