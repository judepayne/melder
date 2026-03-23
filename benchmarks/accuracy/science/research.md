# Research Notes

Findings from web research on fine-tuning embedding models for entity resolution and record matching. Collected March 2026.

---

## Batch Size Matters for MNRL

MultipleNegativesRankingLoss uses in-batch items as negatives. Larger batch = more negatives per anchor = better learning signal. With batch_size=32, each anchor only sees 31 negatives — many of which are easy to distinguish.

**CachedMultipleNegativesRankingLoss** is a sentence-transformers variant that simulates very large batch sizes (e.g. 65536) without the memory cost. It caches embeddings in a first pass, then computes loss with the full virtual batch. Available natively in sentence-transformers, no custom code needed.

- Source: [Pinecone — Fine-tune Sentence Transformers with MNR Loss](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/)
- Source: [Sentence Transformers — Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)

---

## False Negatives in MNRL

MNRL assumes any two non-matching sentences in a batch are truly negative. This can fail when two genuinely similar entities appear in the same batch (e.g. two records from the same country with similar names). Our 1:1 data design mitigates this but doesn't eliminate it.

- Source: [HuggingFace — Mitigating False Negatives in MNRL](https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training)

---

## Hybrid Approach Validated

A 2025 arxiv paper on entity resolution uses exactly our strategy: embeddings for semantic blocking + fuzzy string matching for verification. Confirms the multi-method composite approach is the right architecture.

> "A scalable hybrid framework... utilizing pre-trained language models to encode structured data into semantic embedding vectors, followed by a syntactic verification stage using fuzzy string matching techniques."

- Source: [Transformer-Gather, Fuzzy-Reconsider — arxiv](https://arxiv.org/html/2509.17470v1)

---

## LinkTransformer

Open-source package treating record linkage as text retrieval (which is what melder does). Interesting for design validation — uses sentence-transformers with contrastive learning to generate record-level embeddings, matching based on embedding similarity. We've already built something more sophisticated (multi-method composite scoring, blocking, crossmap bijection).

- Source: [LinkTransformer — arxiv](https://arxiv.org/html/2309.00789v2)

---

## Fine-Tuning vs Hybrid Search

Weaviate's practical guide recommends: "Before committing to fine-tuning, explore keyword or hybrid search techniques before spending time and computing resources fine-tuning an embedding model." This validates our conclusion — combine methods rather than over-invest in fine-tuning a single model.

- Source: [Weaviate — Why, When and How to Fine-Tune](https://weaviate.io/blog/fine-tune-embedding-model)

---

## Entity Resolution in Noisy Data

Standard ER framework: blocking → block processing → entity matching → clustering. Pre-trained embeddings show strong zero-shot performance for entity resolution. Fine-tuning can reduce generalisability — models that are fine-tuned on one dataset may perform worse on others. GPT-mini retains strong generalisation while fine-tuned models lose it.

- Source: [Pre-trained Embeddings for ER — VLDB](https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf)
- Source: [Entity Resolution in Noisy Data — Towards Data Science](https://towardsdatascience.com/entity-resolution-identifying-real-world-entities-in-noisy-data-3e8c59f4f41c/)

---

## Training Tips from Sentence Transformers Docs

- Use `SentenceTransformerTrainer` (v3+) — handles device placement, logging, and evaluation automatically.
- For MNRL: larger batches are better. Use `CachedMultipleNegativesRankingLoss` if memory-limited.
- For entity matching with only positive pairs (anchor, positive): MNRL is the recommended loss function.
- Contrastive learning on diverse data demonstrates good performance in similarity learning.

- Source: [Training and Finetuning Sentence Transformers v3 — HuggingFace](https://huggingface.co/blog/train-sentence-transformers)
- Source: [Training Overview — Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html)

---

## Temperature Scaling in MNRL

MNRL has a `scale` parameter (default 20.0 in sentence-transformers) that controls the sharpness of the softmax over similarity scores. Higher temperature makes the model focus more on distinguishing hard negatives (near-miss entities that look similar); lower temperature treats all negatives more equally. This is a single-number hyperparameter that can significantly affect training behaviour. We have never tuned it.

- Source: [Sentence Transformers — MultipleNegativesRankingLoss API](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)

---

## Two-Stage Training

Train in two phases:
1. **MNRL first** — establish good ranking (matches above non-matches)
2. **Short CosineSimilarityLoss phase** — calibrate absolute scores with a very low learning rate

The intuition: MNRL gets the ordering right without constraining absolute scores. Then a brief cosine phase nudges absolute scores toward target values without the aggressive compression seen when cosine is used alone. The low learning rate in phase 2 limits the damage.

This is a common pattern in retrieval model training — "coarse then fine."

- Source: General contrastive learning literature; no single authoritative source.

---

## Matryoshka Representation Learning

Trains the model to produce useful embeddings at multiple dimensionalities simultaneously (e.g. 768, 512, 256, 128). The constraint of maintaining quality at lower dimensions acts as a regulariser and can improve full-dimension quality too. sentence-transformers supports this natively via `MatryoshkaLoss` which wraps any base loss.

Potential benefit for melder: if we eventually want to use smaller embeddings in production (faster ANN, smaller cache), a Matryoshka-trained model would allow it without retraining.

- Source: [Matryoshka Representation Learning — arxiv](https://arxiv.org/abs/2205.13147)
- Source: [Sentence Transformers — MatryoshkaLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)

---

## GISTEmbedLoss

A newer loss function in sentence-transformers that uses a "guide" model (a separate, typically larger model) to identify and filter out false negatives from in-batch samples before computing the contrastive loss. Directly addresses the false-negative problem in MNRL — where two genuinely similar entities in the same batch are incorrectly treated as negatives.

Requires a second model loaded alongside the training model, so memory cost is higher. May be worth trying if false negatives are identified as a problem in our training.

- Source: [sentence-transformers v5.3.0 release notes](https://github.com/huggingface/sentence-transformers/releases/tag/v5.3.0)
- Source: [Sentence Transformers — Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)
