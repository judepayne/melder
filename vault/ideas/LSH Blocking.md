---
type: idea
module: matching/blocking
status: evaluated — not viable
tags: [blocking, LSH, scaling, candidate-selection, discarded]
related_code: [src/matching/blocking.rs, src/bm25/simple.rs, src/vectordb/]
---

# LSH-Based Blocking — Analysis and Conclusion

## The Problem

Melder's blocking is currently based on exact field matches (e.g. `country_code`). This creates non-overlapping partitions — each record belongs to exactly one block. Within each block, ANN (usearch) and BM25 provide efficient candidate selection. The per-block indices are clean because blocks don't overlap.

At scale, large blocks become the bottleneck. With 1M records and 20 country codes, the average block has 50k records. ANN handles this (O(log N) search), but BM25 querying 50k posting lists per record still adds up. A second blocking dimension that further subdivides blocks without losing recall would improve throughput.

The constraint: **the second blocking key must handle fuzzy data.** Country code works because it's clean and categorical. Entity names are noisy — "Goldman Sachs International" and "GS Intl Ltd" must land in the same block, but any deterministic hash of the raw name will separate them.

Locality-Sensitive Hashing (LSH) is the standard approach for fuzzy blocking. This document evaluates whether LSH-based blocking is viable for Melder.

## How LSH Blocking Works

### MinHash LSH (for Jaccard similarity on character n-grams)

1. For each record, extract character 3-grams from the normalised name:
   - "goldman sachs" -> {"gol", "old", "ldm", "dma", "man", "an ", "n s", " sa", "sac", "ach", "chs"}

2. Compute K MinHash signatures using K independent hash functions. Each signature is the minimum hash value across all n-grams for that hash function.

3. Divide the K signatures into B bands of R rows each (K = B x R).

4. For each band, hash the R signature values together to produce a bucket ID.

5. Two records are **candidates** if they share the same bucket ID in **any** band.

### The Key Parameters

- **B** (number of bands): More bands = more chances for a pair to collide = higher recall, but also higher false positive rate and more overlapping blocks.
- **R** (rows per band): More rows per band = stricter matching per band = lower false positive rate per band, but lower recall per band.
- **K = B x R** (total hash functions): Controls overall granularity.

### The Probability Curve

For two records with Jaccard similarity s, the probability of being candidates (sharing at least one band) is:

```
P(candidate | similarity = s) = 1 - (1 - s^R)^B
```

This is an S-curve. The threshold similarity where P = 0.5 is approximately:

```
threshold ≈ (1/B)^(1/R)
```

## The Maths for Entity Resolution

Entity resolution data has a wide range of similarities between true match pairs:

| Match type | Example | Jaccard (char 3-grams) |
|---|---|---|
| Clean match | "Goldman Sachs" vs "Goldman Sachs International" | ~0.55 |
| Abbreviation | "Goldman Sachs" vs "Goldman Sachs Intl" | ~0.45 |
| Heavy abbreviation | "Goldman Sachs International" vs "GS Intl" | ~0.08 |
| Acronym | "Goldman Sachs International" vs "GSI" | ~0.03 |
| Noisy match | "Goldman Sachs" vs "Golman Sachs" (typo) | ~0.65 |

The hardest pairs (abbreviations, acronyms) have Jaccard similarity below 0.10. These are exactly the pairs that justify having a scoring pipeline in the first place — if they were easy to detect via character overlap, we wouldn't need embeddings.

### Recall vs Overlap Trade-off

For a target recall of 99% on pairs with Jaccard = 0.08 (heavy abbreviation), we need:

```
P(candidate | s = 0.08) >= 0.99
1 - (1 - 0.08^R)^B >= 0.99
(1 - 0.08^R)^B <= 0.01
```

**With R=1 (each band is a single hash — most permissive):**

```
(1 - 0.08)^B <= 0.01
0.92^B <= 0.01
B >= log(0.01) / log(0.92) = 55.2
```

Need **B = 56 bands**. Each record appears in 56 buckets. Total candidate pairs checked = 56x the work of a single non-overlapping partition.

**But R=1 also means very high false positive rate.** Two records with Jaccard = 0.01 (completely unrelated) still have:

```
P(candidate | s = 0.01) = 1 - (1 - 0.01)^56 = 1 - 0.99^56 = 0.43
```

43% of all unrelated pairs become candidates. At 1M records, that's catastrophic — essentially no blocking at all.

**With R=2:**

```
(1 - 0.08^2)^B <= 0.01
(1 - 0.0064)^B <= 0.01
0.9936^B <= 0.01
B >= log(0.01) / log(0.9936) = 718
```

Need **B = 718 bands** with R=2. Each record in 718 buckets. Completely impractical.

**With R=1 and lower recall target (95%):**

```
0.92^B <= 0.05
B >= log(0.05) / log(0.92) = 35.9 → B = 36 bands
```

36 bands, and false positive rate for unrelated pairs (s=0.01):

```
P(s=0.01) = 1 - 0.99^36 = 0.30
```

Still 30% of unrelated pairs are candidates. Not viable.

### The Fundamental Problem

The difficulty is that entity resolution requires matching pairs with very low character-level similarity (Jaccard < 0.10). To catch these pairs via LSH, you need extremely loose hashing (R=1, many bands). But loose hashing means most unrelated pairs also become candidates, defeating the purpose of blocking.

This is not a parameter tuning problem — it's a fundamental limitation of character-level LSH for this domain. The similarity measure (character n-gram Jaccard) doesn't separate true matches from non-matches at the low end. A pair like "Goldman Sachs International" vs "GS Intl" (Jaccard ~0.08) is indistinguishable from a random non-matching pair at the character level.

### What About Embedding-Based LSH (SimHash)?

Using embedding similarity instead of character similarity for LSH:

**SimHash:** Take the sign of each embedding dimension as a bit. The Hamming distance between two SimHash codes is proportional to the angular distance between the embedding vectors.

For the fine-tuned Arctic-embed-xs model, true match pairs typically have cosine similarity > 0.65 (even for acronym cases — the model was trained for this). Two records with cosine similarity c have probability of agreeing on a single random hyperplane bit:

```
P(agree on 1 bit) = 1 - arccos(c) / pi
```

For c = 0.65: P(agree) = 0.79

**SimHash blocking with P-bit prefix:**

For a P-bit prefix, the probability that two records with cosine similarity c share the same prefix:

```
P(same prefix) = (1 - arccos(c) / pi)^P
```

| Cosine sim | P=4 bits | P=6 bits | P=8 bits | P=10 bits |
|---|---|---|---|---|
| 0.90 (easy match) | 0.88 | 0.83 | 0.78 | 0.73 |
| 0.80 (typical match) | 0.75 | 0.65 | 0.56 | 0.49 |
| 0.65 (hard match) | 0.61 | 0.48 | 0.37 | 0.29 |
| 0.30 (non-match) | 0.38 | 0.24 | 0.15 | 0.09 |
| 0.10 (unrelated) | 0.30 | 0.17 | 0.09 | 0.05 |

**The same trade-off applies:** To catch hard matches (cosine 0.65) with 95% recall using multi-probe (M hash tables):

```
P(caught) = 1 - (1 - 0.61^1)^M = 1 - 0.39^M >= 0.95    (P=4 per table)
M >= log(0.05) / log(0.39) = 3.18 → M = 4
```

4 hash tables with 4-bit prefixes. Each record in 4 buckets. False positive rate for unrelated pairs (cosine 0.10):

```
P(false positive) = 1 - (1 - 0.30)^4 = 1 - 0.70^4 = 0.76
```

76% of unrelated pairs are candidates. Still not viable.

**With P=8 bits, M tables for 95% recall on cosine 0.65 pairs:**

```
1 - (1 - 0.37)^M >= 0.95
0.63^M <= 0.05
M >= log(0.05) / log(0.63) = 6.5 → M = 7
```

7 hash tables. False positive rate for unrelated (cosine 0.10):

```
P = 1 - (1 - 0.09)^7 = 1 - 0.91^7 = 0.48
```

48% false positive rate. 7x duplication. Not viable.

## Summary Table

| Method | Target recall | Bands/Tables | Duplication factor | FP rate (unrelated) | Viable? |
|---|---|---|---|---|---|
| MinHash R=1, Jaccard 0.08 | 99% | 56 | 56x | 43% | No |
| MinHash R=1, Jaccard 0.08 | 95% | 36 | 36x | 30% | No |
| SimHash P=4, cosine 0.65 | 95% | 4 | 4x | 76% | No |
| SimHash P=8, cosine 0.65 | 95% | 7 | 7x | 48% | No |
| SimHash P=8, cosine 0.80 | 95% | 3 | 3x | 23% | Marginal |

The only scenario approaching viability is SimHash with P=8 bits targeting only easy matches (cosine > 0.80) — and even then, 23% of unrelated pairs survive as candidates, and the 3x duplication adds significant overhead.

## Conclusion

**LSH-based blocking is not viable for Melder's entity resolution workload.** The fundamental problem is that true match pairs span a wide range of similarities, with the hardest cases (abbreviations, acronyms) having very low character-level or even embedding-level similarity. To catch these hard cases, LSH must be tuned so loosely that it fails to exclude unrelated pairs, producing:

1. **High duplication factor** (records appear in many buckets → overlapping blocks → incompatible with per-block ANN/BM25 indices, and the same pair gets scored multiple times)
2. **High false positive rate** (most unrelated pairs still become candidates → no meaningful candidate reduction)

The existing architecture — exact-field blocking (country_code) + ANN candidate retrieval (O(log N) per query) + BM25 candidate retrieval (inverted index lookup) — is the right design. ANN and BM25 are already doing what LSH attempts to do (find similar records without all-pairs comparison), but with much better precision because they operate on the full vector/token representation rather than a lossy hash.

Further scaling should focus on:
- Making the existing per-block ANN and BM25 indices faster (already well-optimised)
- Adding more exact blocking dimensions when the data supports it (domain-specific categorical fields)
- Distributed architecture (sharding the A-side across machines) for extreme scale

See also: [[Scaling to Millions]], [[Discarded Ideas#OR Blocking Mode]]
