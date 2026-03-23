---
type: idea
module: scoring
status: open
tags: [scoring, pipeline, acronyms, candidate-retrieval]
related_code: [src/matching/pipeline.rs, src/scoring/, src/matching/candidates.rs]
---

# Acronym Matching — Potential Scoring Pipeline Gap

## The Problem

Acronym/initialisation is a blind spot for all three scoring methods:

- **Embeddings**: "TRMS" has no semantic relationship to "Taylor, Reeves and Mcdaniel SRL" in vector space. No embedding model of any size will reliably bridge this — there's zero shared semantic content.
- **BM25**: no shared tokens between "TRMS" and the full name. Zero score.
- **Fuzzy (wratio)**: no character overlap beyond chance. Near-zero score.

This was identified during the synthetic fine-tuning loop (March 2026). After exhausting training improvements (loss functions, labels, hard negatives), acronym-to-full-name matching remains the one failure mode that no combination of existing scoring methods can address.

## Real-World Prevalence

Common in financial entity data:
- GSAM → Goldman Sachs Asset Management
- JPMC → JPMorgan Chase & Co
- BAML → Bank of America Merrill Lynch
- TRMS → Taylor, Reeves and Mcdaniel SRL (synthetic example)

Vendor files frequently use internal acronyms that the reference master stores as full legal names.

## Potential Solutions

### 1. Acronym Scorer (new scoring method)
A dedicated `method: acronym` scorer that:
- Extracts first letters of significant words from the A-side name
- Compares against the B-side name (and vice versa)
- Scores based on match proportion

Cheap to compute (string manipulation only), could be added to the composite alongside existing methods. Would need to handle: case insensitivity, skipping short words (the, and, of), partial acronyms (first 3 of 5 words).

### 2. Acronym Expansion in Preprocessing
Maintain a lookup table of known acronyms → expansions. Expand before encoding. Drawback: requires domain-specific maintenance.

### 3. Parallel Candidate Source
If the acronym scorer exists, it could also serve as a candidate source — bypassing ANN/BM25 entirely for acronym-pattern queries. This would require a pipeline change: candidates = union(ANN results, BM25 results, acronym matches).

## Pipeline Impact

The current pipeline filters candidates before scoring: blocking → BM25 → ANN → score. An acronym match would need to either:
- Survive the existing filters (unlikely — low embedding and BM25 scores)
- Be injected as an additional candidate source (pipeline change)

Option 2 (parallel candidate source) is the more robust approach but requires changes to `src/matching/candidates.rs` and `src/matching/pipeline.rs`.

## Status

Not yet needed — quantify the real-world prevalence of acronym-only matches in production data before investing in implementation. The synthetic generator produces acronyms at ~30% of heavy-noise records, but real vendor files may have a very different rate.
