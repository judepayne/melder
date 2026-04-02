# Melder — Competitive Landscape

_Where Melder sits relative to Quantexa, Elasticsearch, and OpenSearch
for entity resolution and record matching._

Last updated: 2026-04-02

---

## Contents

1. [What This Document Covers](#what-this-document-covers)
2. [The Core Problem: Why Entity Matching Is Hard to Distribute](#the-core-problem-why-entity-matching-is-hard-to-distribute)
3. [Quantexa](#quantexa)
4. [Elasticsearch and OpenSearch](#elasticsearch-and-opensearch)
5. [Cost Comparison](#cost-comparison)
6. [The In-Process Advantage](#the-in-process-advantage)
7. [Capability Matrix](#capability-matrix)
8. [Where Each Tool Wins](#where-each-tool-wins)
9. [Melder's Design Philosophy](#melders-design-philosophy)
10. [Melder's Multi-Source Capability](#melders-multi-source-capability)
11. [Recommendation](#recommendation)

---

## What This Document Covers

This is a positioning analysis for Melder against the two most likely
alternatives an enterprise evaluating entity resolution or record matching
would consider:

- **Quantexa** — the market-leading Decision Intelligence platform, built
  for large banks and government agencies, with entity resolution as a
  core capability.
- **Elasticsearch / OpenSearch** — general-purpose search engines that
  offer hybrid search (semantic similarity + text) and are sometimes
  pressed into service for matching workloads.

The analysis is written for decision-makers evaluating which tool fits
their matching use case. It is honest about where Melder is not the right
choice.

---

## The Core Problem: Why Entity Matching Is Hard to Distribute

Before comparing tools, it helps to understand why entity matching is a
fundamentally different problem from search, and why throwing more
hardware at it is not straightforward. This section matters because it
explains why a single-machine approach like Melder's is not just
cheaper and simpler — it is often more accurate as well.

### The fundamental challenge

Imagine you have two spreadsheets. Spreadsheet A is your reference data
— a million counterparty records. Spreadsheet B is a vendor file — a
million records that need to be matched against A. The task is to find,
for each row in B, the corresponding row in A (if one exists).

If both spreadsheets fit on one machine, this is straightforward. Every
record in B can be compared against every potentially matching record
in A. The scoring logic has access to the full picture.

But what if the data is too large for one machine? You need to split it
across several. And here is the core problem: **you cannot split the
data intelligently because you have not matched it yet.** The matching
is the thing you are trying to do — but you need some way of deciding
which records go to which machine *before* the matching has happened.

### How distributed systems work around this

There are two common approaches, and both involve trade-offs.

**Blocking keys.** You pick a field that you expect matching records to
share — for example, country code, or the first three letters of the
company name. Records with the same blocking key go to the same
machine. This is simple and fast, but it only works when matching
records actually share that key value. "Deutsche Bank AG" (country: DE)
and "Deutsche Bank London Branch" (country: GB) end up on different
machines and never get compared. You either miss the match or need
additional logic to recover it.

**Locality-sensitive hashing (LSH).** This is a more sophisticated
approach. Instead of using a single field, the system generates a
fingerprint of each record that is designed so that similar records are
likely to get the same fingerprint — and therefore land on the same
machine. Think of it as a fuzzy postcode: records that look alike get
sorted into the same bucket, even if they are not identical. The
trade-off is that no fingerprint is perfect. Some genuinely matching
records will get different fingerprints and end up on different machines
(missed matches), while some non-matching records will get the same
fingerprint and waste time being compared (false candidates).

Both approaches are approximations. They are educated guesses about
which records might match, made before the actual matching has
happened. Every such guess introduces the possibility of missed matches
or wasted computation.

### The consequences of splitting

Once data is distributed, several additional problems emerge:

- **Score consistency.** Some scoring methods rely on statistics about
  the full dataset — for example, how common a particular word is
  across all records. When data is split across machines, each machine
  only sees its slice. The word "Goldman" might appear in 100 records
  on one machine and 50 on another, leading to different scores for the
  same pair depending on which machine evaluates it.

- **Uniqueness constraints.** In many matching use cases, each reference
  record should match at most one incoming record. Enforcing this across
  machines requires coordination — without it, two machines might
  independently match the same reference record to different incoming
  records.

- **Iterative enrichment.** Some approaches (Quantexa's, for example)
  build entities incrementally — matching record A to B, then using the
  combined A+B entity to match C. This is inherently sequential within
  each entity cluster and cannot be easily parallelized across machines.

### The trade-off in practice

Distributed systems address these problems through various workarounds:
duplicate records across multiple machines (increasing storage and
processing costs), add re-routing logic to retry unmatched records on
other machines, run secondary global passes to catch cross-partition
misses, or simply accept a small loss in accuracy as the price of
operating at scale.

All of these add complexity, cost, and — in many cases — reduce the
quality of results compared to what you would get if the same data were
processed on a single machine with full visibility.

### The ideal: one machine, full visibility

The ideal scenario for entity matching is simple: one machine that is
large enough and fast enough to hold all the data and compare every
record against every plausible candidate. No splitting, no
fingerprinting, no cross-partition recovery. Every scoring decision has
access to the full dataset, and uniqueness constraints can be enforced
in one place.

This is not always possible. At 50 million records or more, the data
genuinely exceeds what a single machine can handle, and distribution
becomes necessary — along with all of its trade-offs. Quantexa's
Spark-based architecture exists precisely for this scale.

But for the majority of matching workloads in banking — counterparty
matching, vendor file enrichment, instrument reference data — the data
is well within the range that a single machine can handle (up to
roughly 5 million records per side). For these workloads, choosing a
distributed architecture means accepting the trade-offs of distribution
without needing the scale it provides.

This is Melder's central design bet: that for the workloads where it is
used, one machine with full visibility will produce better results,
faster and cheaper, than a cluster of machines working with partial
views of the data.

---

## Quantexa

### What it is

Quantexa is a UK-based Decision Intelligence company founded in 2016.
It is valued at $2.6 billion (March 2025, Series F), has over 900
employees, $100M+ in annual recurring revenue, and is reportedly
considering a 2026 IPO. It is a Gartner Magic Quadrant Leader for
Decision Intelligence Platforms.

Key customers include HSBC, Standard Chartered, BNY, Vodafone, Allianz,
Zurich, ING, the UK Cabinet Office, and the NHS. These are large,
regulated institutions with significant compliance requirements.

### What it does

Quantexa is a full platform, not just an entity resolution engine.
Entity resolution is one layer in a broader stack:

1. **Data Ingestion** — schema-agnostic import with AI-powered parsing,
   cleansing, and normalization. Handles structured and unstructured data.
2. **Entity Resolution** — ML-powered matching that builds unified entity
   profiles across multiple data sources.
3. **Graph Analytics** — visualizes and analyzes relationships between
   entities (ownership chains, transaction networks, fraud rings).
4. **AI and Decisioning** — scoring models, risk alerts, agentic
   copilots, and workflow automation.

### How their entity resolution works

Quantexa's approach is fundamentally different from Melder's. Rather
than scoring pairs (does record A match record B?), Quantexa builds
entity clusters iteratively:

- Start with a record. Find records that are likely the same entity.
  Merge them into a single entity profile. Use the enriched profile to
  find additional matches. Repeat.

This means a sparse record (just a name, no address) can still be
matched if an intermediate record provides the bridge. For example:
"John Citizen" at "123 Main St" matches "J. Citizen" at "123 Main St"
(address match), which in turn matches "Jonathan Citizen" at a different
address (name match with the now-enriched entity). Traditional pairwise
matching might miss the first-to-third link because neither name nor
address alone is strong enough.

Their matching is ML-based with country-specific models. They claim 99%
accuracy, 60x faster data resolution than manual processes, and 20%
record deduplication. These are marketing figures and should be validated
in a proof of concept.

### Their technical architecture

Quantexa runs on **Apache Spark** (Scala/Java) with Hadoop and
Elasticsearch as supporting infrastructure. This is a distributed big
data stack — it scales horizontally by adding nodes to a Spark cluster.
They claim to handle 60 billion+ records.

This architecture is powerful at massive scale but carries significant
infrastructure requirements. A production deployment at a Tier 1 bank
typically involves a multi-node Spark cluster (cloud or on-premises),
months of implementation, professional services engagement, and ongoing
operational support.

### Their "Dynamic Entity Resolution"

Quantexa's headline differentiator is the ability to re-resolve entities
at the time of request with different matching thresholds and data source
permissions per use case. A KYC team might resolve entities with strict
thresholds and restricted data sources, while a marketing team resolves
the same underlying data with looser thresholds and broader access.

This is genuinely useful for organizations that need entity resolution
across many departments. It avoids the problem of maintaining separate
matching instances for each team.

### Where Quantexa is the right choice

- You need to unify customer data across dozens of internal and external
  sources into a single view.
- You are solving AML, KYC, fraud detection, or sanctions screening
  where graph-based network analysis is essential.
- You have 100M+ records and need horizontal scaling across a cluster.
- You need audit trails, explainability, and regulatory compliance
  features out of the box.
- You have the budget and infrastructure team to operate a Spark-based
  platform.
- Multiple teams need different views of the same resolved entities.

### Where Quantexa is not the right choice

- Your use case is matching incoming vendor files against a reference
  master — a pairwise A-vs-B problem, not an N-way deduplication problem.
- You need real-time matching with sub-millisecond scoring latency.
- You need deterministic, stable scores where the same pair always
  produces the same number regardless of what other data exists.
  Quantexa's iterative enrichment means entity composition can affect
  downstream matching.
- You need a 1:1 matching constraint (each A maps to at most one B).
  Quantexa builds N:1 entity clusters — many records resolve to one
  entity — which is a different paradigm.
- You want something operationally simple (single binary, no cluster
  management).
- Your budget is measured in hundreds of dollars per month, not hundreds
  of thousands per year.

---

## Elasticsearch and OpenSearch

### What they are

Elasticsearch is a distributed search and analytics engine. OpenSearch
is an Apache 2.0 fork of Elasticsearch maintained by the Linux
Foundation. Both offer full-text search (BM25), semantic similarity
search, and hybrid search combining the two.

### The attraction for matching

At first glance, a search engine looks like it should solve matching:
index your reference records, then for each incoming record, run a
hybrid query combining text search and semantic similarity. The top
result is your match.

### Why it does not work for entity resolution

The fundamental problem is **score stability**. Search engines produce
relative scores — the score for a result depends on what other results
are in the set. Entity resolution requires absolute scores — the score
for a pair must be a property of that pair alone, so that a threshold
like "0.85 = auto-match" is meaningful and stable over time.

**Concrete example.** On Monday, you query for "Goldman Sachs
International". The candidate "GS International" scores 0.86 after
Elasticsearch's minmax normalization. Above your 0.85 threshold —
matched. On Tuesday, someone adds "Goldman Sachs Intl" to the index. Now
"GS International" scores 0.78 because the normalization window shifted.
Below the threshold — missed. The pair hasn't changed. The data around it
has.

Elasticsearch's `script_score` can produce stable cosine scores, but it
runs on every document (brute-force scan), sacrificing all the
performance benefits of having a search engine.

Additional gaps:
- No 1:1 matching constraint (CrossMap bijection).
- No additive synonym bonus separated from the main score.
- No per-field score decomposition for review UIs.
- BM25 scores are computed per shard, not globally (unless you use a
  slower DFS query mode).
- Semantic similarity recall drops 3-8% when data is spread across shards.

### OpenSearch vs Elasticsearch

OpenSearch offers the same hybrid search capabilities as Elasticsearch
Platinum but is fully open source (Apache 2.0). Elasticsearch's weighted
hybrid search (`linear` retriever) and RRF require a Platinum license
(~$1,500-2,000/month). OpenSearch has no paid tiers.

If an external search system is needed — for example, as a candidate
retrieval layer feeding results into a proper scoring engine — OpenSearch
is the better choice on cost alone.

### Where ES/OpenSearch could complement Melder

A search engine can serve as a **candidate generator** for an external
scoring engine. You would use semantic similarity and BM25 to retrieve
a shortlist of plausible matches, then pull those candidates back into
Melder for scoring, classification, and 1:1 enforcement. This is
essentially what Melder already does internally (blocking + BM25 +
semantic similarity search are all candidate generation), but with the
candidate stages running over the
network instead of in-process. The cost is a 20-50x latency overhead
on candidate generation.

---

## Cost Comparison

| | Quantexa | Elasticsearch (Platinum) | OpenSearch (self-managed) | Melder |
|---|---|---|---|---|
| **License** | Enterprise SaaS (undisclosed) | ~$1,500-2,000/mo | Free (Apache 2.0) | Free (single binary) |
| **Estimated annual cost** | $500K-2M+ | $18K-24K license + infra | Infrastructure only | Infrastructure only |
| **Infrastructure** | Spark cluster (3-10+ nodes) | 3-node search cluster | 3-node search cluster | Single server |
| **Monthly compute (1M records)** | $50K-200K (cloud Spark) | $1,500-3,000 | $800-1,500 | $100-200 |
| **Implementation** | Months, professional services | Weeks, self-service | Weeks, self-service | Hours, single config file |
| **Operational overhead** | Spark/Hadoop ops team | Cluster management | Cluster management | None |

Quantexa's pricing is not published and is negotiated per deal. The
estimates above are based on their ARR ($100M+) across an estimated
100-200 enterprise customers, publicly reported funding rounds, and
typical Spark infrastructure costs at Tier 1 banks. Actual costs will
vary significantly based on data volume, number of use cases, and
deployment model.

Melder's cost is essentially the cost of running one server — a modest
cloud VM for up to a few million records, or a larger instance for
10M+ scale. There is no license fee, no cluster, and no professional
services requirement.

---

## The In-Process Advantage

For datasets that fit on a single machine — and that covers most
real-world matching workloads up to roughly 5 million records per side
— there is a large and often underappreciated performance advantage to
keeping everything in one process.

### Why distribution costs so much

When a matching system is distributed across multiple machines, every
scoring operation pays a tax that has nothing to do with the actual
matching work:

- **Network round-trips.** Each candidate lookup requires sending a
  request to another machine and waiting for the response. Even on a
  fast local network, this adds 0.5-2ms per hop. In a matching pipeline
  that scores millions of pairs, those milliseconds compound quickly.
- **Serialization.** Data must be converted to bytes for transmission
  and converted back on arrival. Records, field values, similarity
  scores — all of it gets serialized and deserialized for every
  interaction between machines.
- **Coordination overhead.** Distributed systems need to agree on things
  — which machine holds which data, how to merge partial results, how
  to handle failures. This coordination has a cost in both latency and
  operational complexity.

None of this overhead exists when scoring happens inside a single
process. The similarity index, the text index, the record store, and
the scoring logic all share the same memory space. A lookup that takes
milliseconds over the network takes nanoseconds in-process.

### What this means in practice

For a workload of 1 million records matched against a reference set of
1 million:

- **Melder (single process):** roughly 17 minutes. All candidate
  retrieval and scoring happens in memory with no network hops.
- **Elasticsearch/OpenSearch (3-node cluster):** roughly 10 hours for
  the same workload. The matching logic is comparable, but every
  candidate retrieval step pays the network and serialization tax.

That is not a small difference. It is the difference between a matching
job that runs as part of a daily pipeline and one that requires
overnight batch scheduling.

At smaller scales — 100,000 records, which covers many real vendor file
matching tasks — Melder completes in seconds. A distributed system
still takes minutes, because the per-operation overhead has a floor
that does not shrink with smaller data.

### When this advantage disappears

The in-process advantage holds as long as the data fits comfortably in
the memory of a single machine. As a rough guide:

| Records per side | Memory needed | Single machine? |
|---|---|---|
| 100K | ~1 GB | Easily — any cloud VM |
| 1M | ~10 GB | Standard server |
| 5M | ~50 GB | High-memory instance |
| 10M | ~100 GB | Specialized hardware |
| 50M+ | ~500 GB+ | Needs distribution |

For the majority of matching use cases in banking — counterparty
matching, vendor file enrichment, instrument reference data — the
dataset falls well within the 1-5 million range where the in-process
approach is fastest, cheapest, and simplest.

Once you genuinely need to match 50 million records or more, a
distributed system like Quantexa's Spark-based architecture becomes
necessary. But for organizations that jump to a distributed solution
before they need one, the cost is significant: 20-50x slower candidate
retrieval, cluster infrastructure to operate, and the coordination
problems described in the earlier section on distribution.

---

## Capability Matrix

| Capability | Melder | Quantexa | ES/OpenSearch |
|---|---|---|---|
| **Pairwise A-vs-B matching** | Core use case | Possible but not primary | Not designed for this |
| **N-way entity clustering** | Via enroll mode | Core use case | Not available |
| **Score stability** | Absolute (pair-dependent only) | Entity-dependent (iterative) | Relative (result-set dependent) |
| **1:1 matching (bijection)** | Atomic CrossMap | N:1 entities (different model) | Not available |
| **Scoring methods** | Semantic similarity + BM25 + fuzzy + exact + synonym | ML models (opaque) | BM25 + semantic similarity (separate scales) |
| **Per-field score breakdown** | Yes | Claimed (explainable AI) | Not structured |
| **Graph / network analysis** | Outputs edges; graph built externally | Core capability | Not available |
| **Data ingestion / cleansing** | No (expects clean input) | Full pipeline | Not available |
| **Real-time API** | Yes (1,500+ req/s) | Dynamic ER (Spark-based) | Yes (search queries) |
| **Batch processing** | Yes (33,000+ rec/s) | Yes (Spark-based, scales out) | Not designed for batch matching |
| **Max practical scale** | ~10M per side (memory), ~50M (SQLite) | 60B+ (Spark cluster) | Billions (search index) |
| **Deployment** | Single binary | Spark cluster + supporting infra | Search cluster (3+ nodes) |
| **Regulatory compliance** | None built in | Audit trails, access controls, explainability | Basic |
| **GPU encoding** | Built-in (CoreML/CUDA) | External (Spark ML) | External inference |

---

## Where Each Tool Wins

### Melder wins when

- The problem is **matching incoming records against a reference master**
  — the classic vendor file enrichment use case.
- **Score stability matters.** A score of 0.85 must always mean the same
  thing, regardless of what other records exist. This is essential for
  setting thresholds that hold over time without re-tuning.
- **1:1 matching is required.** Each reference record maps to at most one
  incoming record, enforced atomically.
- **Low latency matters.** Real-time matching at 1,500+ requests per
  second with sub-30ms p95 latency. Melder scores in-process — no network
  hops, no serialization, no cluster coordination.
- **Operational simplicity matters.** One binary, one YAML config file,
  one server. No cluster management, no JVM tuning, no Spark expertise.
- **Budget is constrained.** $100-200/month versus $500K+/year.
- **The data fits on one machine** (up to ~10M records per side
  comfortably, higher with the SQLite backend).

### Quantexa wins when

- The problem is **building a unified entity view across many data
  sources** — dozens of internal systems plus external registries, where
  the goal is a 360-degree customer profile.
- **Graph analysis is needed.** Detecting fraud rings, tracing ownership
  chains, mapping shell company networks. (Melder does not have a graph
  layer, though its enroll mode can match across multiple datasets and
  output the edges — the "who matches whom" relationships — that a
  graph system would need. The graph itself would need to be built and
  maintained in a separate tool.)
- **Regulatory compliance is a primary concern.** AML, KYC, sanctions
  screening with audit trails, explainability, and data access controls.
- **Scale exceeds what one machine can handle.** 100M+ records requiring
  horizontal scaling.
- **Multiple teams need different matching configurations** on the same
  data (Dynamic ER).
- **Budget and infrastructure are not constraints.** You have the team
  to operate a Spark cluster and the budget for an enterprise platform.

### Elasticsearch / OpenSearch wins when

- The primary need is **search** (finding records by query), not matching
  (determining which records refer to the same entity).
- You need a **candidate retrieval layer** feeding into an external
  scoring engine.
- The matching does not require stable absolute scores or 1:1 constraints.

---

## Melder's Design Philosophy

Melder is deliberately built as a **single engine** — the core building
block from which more complex systems can be assembled. It does one
thing well: score pairs of records and determine whether they match.

This is a conscious design choice, not a limitation. Platforms like
Quantexa bundle data ingestion, entity resolution, graph analytics,
scoring models, workflow automation, and compliance tooling into a
single product. That breadth is valuable when you need all of it, but
it comes with coupling — you adopt the whole platform or none of it,
and you operate it as a unit.

Melder takes the opposite approach. It provides the matching engine and
leaves the surrounding system to the organization:

- **Data preparation** happens upstream, in whatever ETL or data
  pipeline the organization already runs.
- **Graph construction** happens downstream, using the match edges
  Melder outputs.
- **Workflow and review UIs** are built on top of Melder's API and
  match results, not locked inside a proprietary platform.
- **Compliance and audit** are handled at the system level, not
  embedded in the matching engine.

This makes Melder easy to integrate into existing infrastructure. It
fits into a data pipeline as a single step — not as a platform that
replaces the pipeline. For organizations that already have data
engineering teams, ETL tooling, and operational systems, this composable
approach avoids the cost and disruption of adopting a full platform when
all they need is better matching.

---

## Melder's Multi-Source Capability

While Melder's primary design is two-sided matching (A against B), it
also supports **single-pool entity resolution** via its enroll mode.
In enroll mode, records are added to a single pool and matched against
each other — effectively N-way deduplication within one dataset.

This means Melder can be used in a broader multi-source workflow:

1. Combine multiple source files into one dataset.
2. Run Melder in enroll mode to find duplicates and cluster related
   records.
3. Use the match results to build a unified view.

This is a simpler version of what Quantexa does with their entity
resolution layer. It lacks the graph analytics, the iterative
enrichment, the per-use-case Dynamic ER, and the regulatory compliance
features — but for organizations that need basic multi-source
deduplication without the cost and complexity of a platform like
Quantexa, it is a viable starting point.

Enroll mode supports the same scoring pipeline as two-sided matching
(semantic similarity, BM25, fuzzy, exact, synonyms) and the same real-time API,
making it straightforward to integrate into an existing data pipeline.

---

## Recommendation

**For bank reference data matching** (counterparties, instruments,
issuers against vendor files): Melder. This is the exact use case it
was built for, and it will outperform both Quantexa and Elasticsearch
on speed, cost, score stability, and operational simplicity.

**For enterprise-wide entity resolution** (single customer view across
dozens of sources, with graph analytics, AML/KYC compliance, and
multi-team access): Quantexa. This is what it was built for, and the
breadth of its platform justifies the cost at that scale.

**For search-based candidate retrieval** feeding into an external
scoring engine: OpenSearch (not Elasticsearch — same capabilities, no
license cost).

**The tools are not direct competitors.** They solve overlapping but
distinct problems at very different price points and complexity levels.
A bank could plausibly use Quantexa for enterprise-wide KYC/AML entity
resolution and Melder for daily vendor file matching in the reference
data team — the two would complement rather than conflict.
