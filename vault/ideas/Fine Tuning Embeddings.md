---
type: idea
module: scoring
status: active
tags: [embeddings, domain-adaptation, training, contrastive-learning]
related_code: [src/scoring/embedding.rs, src/encoder/mod.rs, src/vectordb/mod.rs]
---

# Fine-Tuning Embedding Models for Domain Accuracy

## The Problem

General-purpose sentence transformer models (e.g. `all-MiniLM-L6-v2`) are trained on web text. They do not know that words like "International", "Holdings", "Group", or "Ltd" are low-signal in financial entity names — they appear in nearly every counterparty name and are poor discriminators. BM25 handles this via corpus-specific IDF; a pre-trained embedding model cannot.

## The Fix

Fine-tune the base model on domain-specific labelled pairs using contrastive learning. The model learns to pull similar entities close together in vector space and push different entities apart, regardless of shared low-signal tokens.

## Training Data

- **Positive pairs**: two name representations of the same entity (e.g. "Goldman Sachs" / "GS Capital Partners")
- **Hard negative pairs**: names that look similar but refer to different entities (e.g. "JP Morgan International" / "Morgan Stanley International")
- **Gold source**: Melder's own crossmap is a gold source of positive pairs; review rejections are hard negatives — the training data accumulates automatically from production use

## Tooling Stack

- `sentence-transformers` (Python, Hugging Face) — main fine-tuning library
- `PyTorch` — underlying framework
- `Hugging Face Hub` — model source
- `optimum` (Hugging Face) — exports the fine-tuned model to ONNX for use with melder
- `Weights & Biases` or `MLflow` — experiment tracking
- **Hardware**: a single GPU (Google Colab T4 is sufficient for MiniLM-scale models)

## Steps

1. Export labelled pairs from melder's crossmap and review CSVs
2. Load base model via sentence-transformers
3. Fine-tune with CosineSimilarityLoss or Multiple Negatives Ranking Loss — a few epochs
4. Evaluate on a held-out set of known matches/non-matches
5. Export to ONNX via `optimum`
6. Point melder's config `embeddings.model` at the new ONNX file

## Quick Win to Try First

Strip low-signal tokens ("International", "Holdings", "Limited", "Corp", "Inc", "Group", "plc", "LLC") as a preprocessing step before encoding. Five lines of Python, zero retraining. Covers the most common case cheaply.

## The Data Flywheel

Every melder production run enriches the training dataset. Re-fine-tune quarterly; the model improves progressively on your specific entity universe.

---

See also: [[Scoring Algorithm]] for the embedding scoring method, [[Key Decisions]] for model selection rationale, [[Use Cases]] for deployment patterns where domain-specific embeddings matter most.
