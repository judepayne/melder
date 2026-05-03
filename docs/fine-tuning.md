← [Back to Index](./) | [Accuracy & Tuning](accuracy-and-tuning.md) | [Scoring](scoring.md)

# Fine-Tuning Embedding Models

How to improve embedding quality through domain-specific fine-tuning.

## When to Consider

- General-purpose embeddings (MiniLM, BGE) underperform on your entity types
- Production config shows high overlap or poor recall
- You have labeled pairs from production (crossmap = positives, review rejections = hard negatives)

## Quick Win (Before Fine-Tuning)

Strip low-signal tokens before encoding:

```text
International, Holdings, Limited, Corp, Inc, Group, plc, LLC, SA, AG
```

Five lines of preprocessing. Zero retraining.

## Fine-Tuning Steps

1. **Export training data** — from crossmap positives and review rejections / known non-matches as hard negatives
2. **Choose base model** — Arctic-embed-xs (22M, 384d) is recommended for speed; BGE-base offers more capacity
3. **Fine-tune** — use a ranking loss such as MultipleNegativesRankingLoss with hard negatives
4. **Evaluate** — on a held-out set, measure overlap, precision, recall, and missed matches
5. **Export to ONNX** — via `optimum` for use with Melder
6. **Deploy** — point `embeddings.model` at the exported ONNX model directory or file

## Practical Notes

- Prefer LoRA/adapters over full fine-tuning unless you have strong evidence that full fine-tuning is safe for your data.
- Keep a fixed holdout set that is never trained on.
- Watch both recall and review noise: lowering non-match scores is only useful if true matches stay above threshold.
- BM25 and synonym matching often solve problems that embeddings cannot, especially common-word false positives and acronym/full-name pairs.

## Production Result

| Config | Overlap | Recall |
|--------|---------|--------|
| Arctic-embed-xs R22 + 50% BM25 + synonym 0.20 | 0.0003 | 100% |

Historical training experiments live under `benchmarks/accuracy/science/`, especially `experiments.md`.
