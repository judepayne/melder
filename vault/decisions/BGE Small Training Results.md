---
type: decision
module: training
status: active
tags: [fine-tuning, embeddings, bge, training-results]
related_code: [benchmarks/accuracy/training/]
---

# BGE-small Fine-Tuning Results (March 2026)

## Best Configuration Found

**Base BGE-small (BAAI/bge-small-en-v1.5) with no fine-tuning, review_floor=0.60.**

On 1:1 synthetic holdout data (10k records, 60% matched / 10% heavy noise / 30% unmatched):

| Metric | Value |
|---|---|
| Missed (clean) | 1 |
| Missed (heavy noise) | 0 |
| Not-a-match in auto | 8 |
| Not-a-match in review | 3,004 |
| Combined recall | 99.4% |
| Precision | 88.2% |

The only problem is review queue noise: 3,004 non-matches leak into review. But from a business perspective (missed match = duplicate record in master), this is acceptable — nearly zero misses.

## Fine-Tuning Attempts (all degraded recall)

Every fine-tuning approach compressed the entire score distribution rather than stretching the gap between matches and non-matches:

| Approach | Result |
|---|---|
| CosineSimilarityLoss (labels 0.85, 0.7, 0.5) | Scores compressed downward, recall dropped 8-13pp |
| MNRL (in-batch negatives) | Preserved recall better (~85.5%) but didn't improve it |
| MNRL + hard negatives | Similar to plain MNRL |
| Neg-only (CosineSimilarityLoss on label=0 pairs) | Everything pushed down |
| Pos-only (CosineSimilarityLoss on label>0 pairs) | Everything pushed up, precision collapsed to 50% |
| MNRL + hard negatives, batch=32 | Overlap 0.081 at R7 (best), but recall degraded to 94.6% |
| MNRL + hard negatives, batch=128 | Overlap 0.070 at R12, recall degraded to 94.6% |

## Root Cause

BGE-small (33M params, 384 dims) lacks the representational capacity to independently separate match and non-match score distributions. Any weight change that affects one population affects the other in the same direction. The model cannot "stretch" — it can only "shift."

The 384-dim embedding space is the real bottleneck. Batch size affects training signal (batch=32 achieved overlap 0.081 vs batch=128's 0.070), but neither breaks through the capacity ceiling.

## Production Recommendation

**For production use with BGE-small:**
- Use base model (no fine-tuning) with review_floor=0.60 for maximum recall (99.4% combined).
- Or use fine-tuned model at R8 (batch=128, MNRL) with overlap 0.078 and recall 98.7% — practical stopping point before degradation.
- Combine with BM25 at 20% weight (see [[Key Decisions#BM25 Hybrid Scoring]]) for ~2× faster encoding with acceptable quality trade-off.

## Next Step

BGE-base (BAAI/bge-base-en-v1.5, 110M params, 768 dims) — 3x the capacity — has been tested and confirmed to stretch better than BGE-small. See [[Training Experiments Log#Experiment 5]] for results.

If larger models also hit capacity limits, the path forward is melder's multi-method scoring pipeline: combine embeddings with fuzzy (wratio) and BM25 scoring to cover each method's blind spots.
