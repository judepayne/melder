# Pipeline Enhancement: Independent Candidate Generation + Synonym Matching

## Motivation

Experiment 6 proved that BM25 composite scoring dramatically reduces overlap (0.046 → 0.002). However, 81 true matches remain stuck in the 0.44-0.60 score range — every single one is an acronym/abbreviation (e.g. "HWAG" → "Harris, Watkins and Goodwin BV"). Neither embeddings nor BM25 can bridge this gap because the relationship is structural (first letters of words), not semantic or lexical.

Additionally, the current pipeline has BM25 acting as a re-ranker of ANN candidates rather than an independent candidate generator. This means BM25 can only score candidates that ANN surfaced — it cannot find candidates that ANN missed. The same limitation would apply to any new matching method bolted onto the existing pipeline.

## Current Pipeline

```
B record
  → Blocking (O(1) HashMap)
  → ANN candidate selection (O(log N) HNSW, top ann_candidates)
  → BM25 re-rank of ANN candidates (scores ~50, keeps top bm25_candidates)
  → Full scoring (all match_fields on ~10 survivors)
  → Classification (auto/review/no-match)
```

BM25 can only see what ANN surfaced. If ANN misses a candidate (e.g. no embedding similarity on any field), BM25 never gets a chance to find it.

## Proposed Pipeline

```
B record
  → Blocking (O(1) HashMap) → blocked_ids
  → Candidate generation phase (independent, parallel):
      → Synonym lookup (O(1) HashMap) → synonym_candidates (~0-2)
      → ANN search (O(log N) HNSW)    → ann_candidates (~50)
      → BM25 search (O(log N) Tantivy) → bm25_candidates (~10-20)
  → Union all candidate sets (deduplicate by A ID)
  → Full scoring phase (all match_fields on union, ~50-70 candidates)
  → Classification
```

Each candidate generator runs independently against the blocked pool. No method filters another method's candidates. The union captures candidates found by ANY method. Full scoring is the single place where all signals combine.

### Design Principles

1. **Candidate generation is independent.** Each method surfaces its best candidates without knowledge of the others. This ensures no method's blind spot becomes the pipeline's blind spot.
2. **Scoring is unified.** All match_fields (embedding, BM25, synonym, exact, fuzzy) are evaluated on every candidate in the union. The composite score handles ranking.
3. **Pipeline is ordered by cost.** Blocking (O(1)) runs first. Candidate generators are all O(1) or O(log N). Full scoring (O(N) per candidate) runs last on the smallest set.

## Implementation Phases

### Phase 1: ANN ∪ BM25 (independent candidate generation)

Change BM25 from a re-ranker of ANN candidates to an independent candidate generator. The union of ANN and BM25 candidates flows to full scoring.

**Steps:**
1. Run existing batch benchmarks (`benchmarks/batch/`) to capture baseline performance
2. Modify `pipeline.rs` to run ANN and BM25 independently, union results
3. Re-run batch benchmarks to measure speed impact
4. If slowdown is acceptable, proceed to Phase 2

**Speed cost:** BM25 moves from scoring ~50 ANN candidates to an independent Tantivy query across the blocked pool. Tantivy is built for this (O(log N)), but the pool is larger than 50. The gain: BM25 can now surface candidates that ANN missed entirely.

### Phase 2: Synonym matching

Add synonym matching as both a candidate generator and a scoring method.

**Steps:**
1. Implement `SynonymIndex` (HashMap-based, bidirectional)
2. Implement acronym generator
3. Add `method: synonym` to the scoring dispatch in `score_pair()`
4. Add synonym candidate generation to the candidate phase (alongside ANN and BM25)
5. Add config support (`synonym_fields`, generators, optional synonym table)
6. Re-run experiment 6 with synonym matching enabled to measure impact on the 81 acronym cases

## Synonym System Design

### SynonymIndex

A bidirectional HashMap that maps normalised alternative forms to record IDs on both sides.

```
SynonymIndex {
    // "HWAG" → [("entity_123", Side::A)]
    // "Harris, Watkins and Goodwin BV" → [] (not an acronym of anything on B side)
    index: HashMap<String, Vec<(RecordId, Side)>>
}
```

**Bidirectional indexing is essential.** The abbreviation can occur on either side:
- B record "HWAG" matching A record "Harris, Watkins and Goodwin BV" (B is the acronym)
- B record "Harris, Watkins and Goodwin BV" matching A record "HWAG" (A is the acronym)

Both sides must be indexed. When building the index:
- For each A record: generate acronyms from A's name, index them pointing to A's ID with Side::A
- For each B record: generate acronyms from B's name, index them pointing to B's ID with Side::B

When querying for a B record:
- Look up B's full name in the index → finds A records that are acronyms of B's name
- Look up B's name directly → finds A records whose full name has B's name as an acronym

In practice this means: for each A record, generate acronyms and store them. For each B record, check (a) is B's name in the acronym index? and (b) generate B's acronyms and check if any A record's name matches one of them. The index supports both directions.

### Acronym Generator

Algorithmic generator that produces candidate acronyms from a full entity name.

**Input:** "Harris, Watkins and Goodwin BV"

**Steps:**
1. Tokenise the name into words
2. Classify each word: name word, connector (and, &, of, the, und, et), legal suffix (LLC, Ltd, GmbH, BV, Inc, Corp, etc.), or punctuation
3. Generate variants by combining first letters with/without each word class:
   - Names only: "HWG"
   - Names + connectors: "HWAG"
   - Names + suffixes: "HWGB"
   - Names + connectors + suffixes: "HWAGB" or "HWAGBV"
   - Hyphenated name splitting: "Gibson-Edwards" → names are ["G", "E"] or ["GE"]

**Output:** Set of candidate acronyms, all uppercased: `{"HWG", "HWAG", "HWGB", "HWAGBV"}`

**Normalisation:** All lookups are case-insensitive (uppercase both sides before lookup).

Typically 4-8 variants per record. For 10K A-side records = ~40-80K index entries. Trivial memory.

### Optional Synonym Table

A user-provided CSV or YAML file mapping known synonyms. This extends the same index with non-algorithmic equivalences.

```yaml
# synonyms.yaml
- canonical: Samuel
  alternatives: [Sam, Sammy]
- canonical: Robert
  alternatives: [Bob, Bobby, Rob]
- canonical: International Business Machines
  alternatives: [IBM]
- canonical: JPMorgan Chase
  alternatives: [JP Morgan, JPM, JPMC]
```

Or as CSV:

```csv
canonical,alternative
Samuel,Sam
Samuel,Sammy
Robert,Bob
Robert,Bobby
```

Each entry is indexed bidirectionally: "Sam" → canonical "Samuel", and the canonical "Samuel" is also stored so that a record containing "Samuel" can be matched to a query containing "Sam".

The synonym table is loaded at startup and merged into the same `SynonymIndex` used by the acronym generator. The distinction is:
- **Acronym generator:** algorithmic, derives synonyms from the record's own name field
- **Synonym table:** static, loaded from config, applies across all records

### Config

```yaml
synonym_fields:
  - field_a: legal_name
    field_b: counterparty_name
    generators:
      - type: acronym           # algorithmic acronym generation
        min_length: 3           # ignore acronyms shorter than 3 chars
    table: path/to/synonyms.csv # optional external synonym table

match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.44
  - field_a: registered_address
    field_b: counterparty_address
    method: embedding
    weight: 0.28
  - method: bm25
    weight: 0.18
  - field_a: legal_name
    field_b: counterparty_name
    method: synonym
    weight: 0.10
```

The `synonym_fields` section controls index building (which fields to generate synonyms from, which generators to use, optional table). The `method: synonym` entry in `match_fields` controls scoring (weight in composite score). These are deliberately separate — the index is built once at startup; the scoring weight is tunable independently.

### Scoring Semantics

`method: synonym` is binary:
- **1.0** if B's name (or A's name) is a known synonym/acronym of the other side's name
- **0.0** otherwise

At weight 0.10, a synonym match adds 0.10 to the composite score. An acronym pair currently scoring 0.55 would be pushed to ~0.65 (comfortably in review). With strong address match, potentially into auto-match territory.

### False Positive Considerations

Short acronyms (2 chars) could match many A records. Mitigations:
1. **Blocking** already constrains to same country code
2. **min_length config** (default 3) filters out 2-char acronyms
3. **Full scoring** handles disambiguation — a false acronym match with wrong address scores low overall
4. Synonym candidates are cheap to generate (0-2 per B record), so even a few false candidates don't affect performance
