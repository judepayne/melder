---
type: architecture
module: scoring
status: active
tags: [scoring, algorithm, fuzzy, embedding, composite-score]
related_code: [src/scoring/mod.rs, src/scoring/exact.rs, src/scoring/embedding.rs, src/fuzzy/, src/matching/pipeline.rs]
---

# Scoring Algorithm

How Melder computes match scores. All scoring flows through `matching/pipeline.rs::score_pool()` — there is no second code path. See [[Constitution#2 One Scoring Pipeline]].

---

## Composite Score Formula

```
composite_score = weighted_sum / total_weight

weighted_sum  = Σ (field_score_i × weight_i)
total_weight  = Σ weight_i
```

Weights are **auto-normalised**: if `|total_weight − 1.0| > 0.001`, the sum is divided by `total_weight`, producing a composite score in `[0.0, 1.0]`. If weights already sum to 1.0 (within tolerance), the raw weighted sum is used directly.

**Consequence:** weights in config are relative, not absolute. Setting all weights to 1.0, 0.5, or 100.0 produces identical scores as long as ratios are the same.

Each `FieldScore` also exposes `.contribution() = score × weight` (pre-normalisation), useful for debugging which field drove a result.

---

## Classification

Applied to the composite score after all fields are scored. See [[Config Reference#thresholds]].

```
score >= auto_match   →  Classification::Auto    ("auto")
score >= review_floor →  Classification::Review  ("review")
otherwise             →  Classification::NoMatch  ("no_match")
```

Both thresholds are **inclusive** (≥). `Classification` is serialised as `snake_case` in JSON responses.

---

## Per-Field Scoring Methods

### `exact` — `scoring/exact.rs`

Unicode case-insensitive equality.

1. Trim both values
2. Try `eq_ignore_ascii_case` (fast path for ASCII)
3. Fall back to `.to_lowercase()` comparison for non-ASCII (handles accented chars, ligatures)

| Input | Score |
|---|---|
| `"Café"` vs `"café"` | 1.0 |
| `"UK"` vs `"uk"` | 1.0 |
| `""` vs `""` (both empty after trim) | **0.0** — not evidence of a match |
| `"ABC"` vs `"DEF"` | 0.0 |

**Use for:** identifiers, codes, country fields, currencies. Anything where equality is the right comparison.

---

### `fuzzy` — `fuzzy/` module

All scorers normalise input (lowercase + trim) before scoring and return `f64` in `[0.0, 1.0]`. Both empty → 0.0. One empty → 0.0 for all scorers.

#### `ratio`

Normalised Levenshtein similarity via `rapidfuzz::fuzz::ratio`.

```
score = 2 × |common_chars| / (|a| + |b|)
```

Good baseline. Sensitive to length differences — inserting one word into a long string drops the score significantly.

---

#### `partial_ratio`

Slides the shorter string across the longer as a window; returns the maximum `ratio` across all positions.

```
"Goldman" vs "Goldman Sachs & Co."  →  high score
"Goldman" vs "Goldman"               →  1.0
```

**Use when:** the key name may be a substring of a longer legal name.

---

#### `token_sort_ratio`

Splits both strings into tokens, sorts them alphabetically, rejoins with spaces, then computes `ratio`.

```
"Morgan JP" vs "JP Morgan"  →  1.0
```

**Use when:** word order is inconsistent between datasets.

---

#### `wratio` (default)

`max(ratio, token_sort_ratio, partial_ratio)`. Takes the best of all three. Has an early-exit optimisation: if `ratio` returns 1.0, the other two are skipped.

**Use when:** you want resilience to all common name variation patterns without having to choose a specific scorer. This is the right default for most entity name fields.

**Implementation note:** `partial_ratio` and `token_sort_ratio` are implemented from scratch in Melder. Only `ratio` is pulled from the `rapidfuzz-rs` crate (which only exposes that one function at v0.5). The character-level operations use `.chars()`, not bytes — correctly handles multi-byte UTF-8.

---

### `embedding` — `scoring/embedding.rs`

Cosine similarity between pre-computed embedding vectors.

The actual cosine computation is:
1. L2-normalise both vectors in-place (no-op if zero-norm)
2. Compute dot product (f32 arithmetic, returned as f64)
3. **Clamp to `[0.0, 1.0]`** — negative cosine similarity is mapped to 0.0

**Why clamping?** Negative similarity (opposite directions in embedding space) has no meaningful interpretation for entity matching — it would perversely reduce a composite score below what a zero contribution would give. Clamping to 0.0 treats it as "no evidence of similarity", which is correct.

**Where cosines come from:** embedding scores are never recomputed from scratch during full scoring. They are recovered via `decompose_emb_scores()` in `matching/pipeline.rs` by reversing the combined vector construction — a pure dot product operation, no second ONNX call. See [[Constitution#4 Combined Vector Weighted Cosine Identity]].

---

### `numeric` — `scoring/mod.rs`

Parse-and-compare with floating-point equality.

1. Parse both values as `f64`
2. Return 1.0 if `|a − b| < f64::EPSILON`, else 0.0
3. Both empty → 0.0; either non-numeric → 0.0

**Use for:** numeric identifiers where exact equality is appropriate. Not suitable for fuzzy numeric proximity (e.g., financial amounts with rounding differences) — that case is tracked in [[Discarded Ideas#Tolerance-Based Numeric Scoring]].

---

## Candidate Selection

Before full scoring, the pipeline retrieves a shortlist of candidates from the vector index (up to `top_n`). Full scoring runs only on this shortlist — not against the entire dataset.

- **usearch backend:** O(log N) HNSW ANN search using the combined embedding vector
- **flat backend:** O(N) brute-force cosine scan
- **No embedding fields configured:** all blocked records pass through to full scoring directly

See [[Key Decisions#Combined Vector Index Single Index Per Side]] for why one combined index is sufficient.

---

## Common Gotchas

| Situation | Result | Why |
|---|---|---|
| Both fields empty | 0.0 | Empty match is not evidence of similarity |
| One field empty | 0.0 | Asymmetric data should not score as a match |
| Weights don't sum to 1.0 | OK | Auto-normalised at scoring time |
| Negative cosine similarity | 0.0 | Clamped; treated as no evidence |
| `top_n` too low | Missed matches | ANN may not retrieve the true best match; increase `top_n` if recall is poor |
| All `exact` fields, no `embedding` | No vector index built | Candidate selection is blocked-records only |

---

See also: [[Config Reference#match_fields]] for scorer configuration, [[Business Logic Flow]] for where scoring sits in the pipeline, [[Performance Baselines]] for throughput by method.
