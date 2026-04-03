---
type: guide
module: training
status: active
tags: [fine-tuning, embeddings, guide, how-to]
related_code: [benchmarks/accuracy/training/]
---

# Fine-Tuning Embedding Models

How to improve embedding quality through domain-specific fine-tuning.

## When to Consider

- General-purpose embeddings (MiniLM, BGE) underperform on your entity types
- Production config shows high overlap or poor recall
- You have labeled pairs from production (crossmap = positives, review rejections = hard negatives)

## Quick Win (Before Fine-Tuning)

Strip low-signal tokens before encoding:
```
International, Holdings, Limited, Corp, Inc, Group, plc, LLC, SA, AG
```
Five lines of preprocessing. Zero retraining.

## Fine-Tuning Steps

1. **Export training data** — from crossmap (positives) and review rejections (hard negatives)
2. **Choose base model** — Arctic-embed-xs (22M, 384d) recommended for speed; BGE-base for quality
3. **Fine-tune** — MultipleNegativesRankingLoss with hard negatives, 15-25 rounds
4. **Evaluate** — on held-out set, measure overlap and recall
5. **Export to ONNX** — via `optimum` for use with melder
6. **Deploy** — point `embeddings.model` at the ONNX file

## Production Result

| Config | Overlap | Recall |
|--------|---------|--------|
| Arctic-embed-xs R22 + 50% BM25 + synonym 0.20 | 0.0003 | 100% |

See [[decisions/training_experiments_log]] for full experiment details (1-12).
