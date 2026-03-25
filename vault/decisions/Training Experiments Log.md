---
type: decision
module: training
status: active
tags: [fine-tuning, embeddings, experiments, results]
related_code: [benchmarks/accuracy/training/]
---

# Training Experiments Log

Running log of embedding fine-tuning experiments. All use the 1:1 synthetic generator (60% matched / 10% heavy noise / 30% unmatched), `preserve_blocking=True`, no exact prefilter, 10k×10k datasets, full fine-tune (no LoRA), usearch backend.

Holdout results only (fixed seed 9999, same A master across all experiments).

---

## Experiment 1: BGE-small, MNRL, review_floor=0.80

**Setup:** BAAI/bge-small-en-v1.5, MultipleNegativesRankingLoss (in-batch negatives only, no hard negatives), 5 rounds, review_floor=0.80.

**Key result:** MNRL preserved recall much better than cosine similarity loss. Combined recall 85.5% vs ~82% for cosine. Zero phantom/leaked after round 1.

| | Round 0 (base) | Round 4 (best) |
|---|---|---|
| **Auto-matched** | **5,668** | **5,343** |
| Clean | 4,998 | 4,765 |
| Heavy noise | 663 | 578 |
| Not a match | 7 | 0 |
| **Review** | **774** | **544** |
| Clean | 135 | 344 |
| Heavy noise | 142 | 200 |
| Not a match | 758 | 0 |
| **Unmatched** | **3,136** | **4,113** |
| Missed (clean) | 856 | 869 |
| Missed (heavy noise) | 205 | 232 |
| Not a match | 2,247 | 3,012 |
| **Precision** | **88.2%** | **89.2%** |
| **Recall (vs ceiling)** | **83.6%** | **79.7%** |
| **Combined recall** | **85.7%** | **85.5%** |

**Observation:** Missed clean barely changed (856 → 869). MNRL stopped the compression seen with cosine loss but didn't improve recall. The model's capacity is the limit, not the training signal.

---

## Experiment 2: BGE-small, MNRL + hard negatives, review_floor=0.60

**Setup:** BAAI/bge-small-en-v1.5, MNRL with explicit hard negatives from pairs.py, 7 rounds (of 10 planned — run stopped early), review_floor lowered to 0.60.

**Key result:** Lower review_floor dramatically reduced misses in round 0 (1 missed clean!). But fine-tuning still compressed scores — missed clean climbed back to 743 by round 6.

| | Round 0 (base) | Round 6 (last) |
|---|---|---|
| **Auto-matched** | **5,668** | **4,909** |
| Clean | 4,997 | 4,406 |
| Heavy noise | 663 | 503 |
| Not a match | 8 | 0 |
| **Review** | **4,331** | **1,314** |
| Clean | 946 | 829 |
| Heavy noise | 347 | 276 |
| Not a match | 3,004 | 209 |
| **Unmatched** | **1** | **3,777** |
| Missed (clean) | 1 | 743 |
| Missed (heavy noise) | 0 | 231 |
| Not a match | 0 | 2,803 |
| **Precision** | **88.2%** | **89.8%** |
| **Recall (vs ceiling)** | **83.6%** | **73.7%** |
| **Combined recall** | **99.4%** | **87.6%** |

**Observation:** Round 0 base model + review_floor=0.60 is nearly perfect (1 missed, 99.4% combined recall). The only problem is 3,004 non-matches clogging review. Training cleaned review (3,004 → 209) but dragged matches down with it. The model compresses, not stretches.

---

## Experiment 3: BGE-small, neg-only, review_floor=0.60

**Setup:** BAAI/bge-small-en-v1.5, CosineSimilarityLoss on label=0.0 pairs only (pure "push non-matches down" signal), 2 rounds, review_floor=0.60.

**Key result:** Everything pushed down — including matches. Missed clean went from 1 → 525. Non-match in review barely moved (3,004 → 2,863).

| | Round 0 (base) | Round 1 |
|---|---|---|
| **Auto-matched** | **5,668** | **3,815** |
| Clean | 4,997 | 3,407 |
| Heavy noise | 663 | 408 |
| Not a match | 8 | 0 |
| **Review** | **4,331** | **5,317** |
| Clean | 946 | 1,995 |
| Heavy noise | 347 | 408 |
| Not a match | 3,004 | 2,863 |
| **Unmatched** | **1** | **868** |
| Missed (clean) | 1 | 525 |
| Missed (heavy noise) | 0 | 194 |
| Not a match | 0 | 149 |
| **Precision** | **88.2%** | **89.3%** |
| **Recall (vs ceiling)** | **83.6%** | **57.0%** |
| **Combined recall** | **99.4%** | **90.4%** |

**Observation:** Confirms BGE-small can't selectively move non-matches without dragging matches down. Same weights serve all pairs.

---

## Experiment 4: BGE-small, pos-only, review_floor=0.60

**Setup:** BAAI/bge-small-en-v1.5, CosineSimilarityLoss on label>0 pairs only (pure "push matches up" signal), 2 rounds, review_floor=0.60.

**Key result:** Everything pushed up — including non-matches. 2,750 non-matches auto-confirmed, precision collapsed to 50.5%.

| | Round 0 (base) | Round 1 |
|---|---|---|
| **Auto-matched** | **5,668** | **8,880** |
| Clean | 4,998 | 4,485 |
| Heavy noise | 663 | 868 |
| Not a match | 7 | 2,750 |
| **Review** | **4,331** | **425** |
| Clean | 946 | 23 |
| Heavy noise | 347 | 91 |
| Not a match | 3,005 | 0 |
| **Unmatched** | **1** | **695** |
| Missed (clean) | 1 | 382 |
| Missed (heavy noise) | 0 | 51 |
| Not a match | 0 | 262 |
| **Precision** | **88.2%** | **50.5%** |
| **Recall (vs ceiling)** | **83.6%** | **75.0%** |
| **Combined recall** | **99.4%** | **75.4%** |

**Observation:** Together with Experiment 3, proves BGE-small lacks capacity to separate populations. Neg-only pushes everything down; pos-only pushes everything up. The model can shift but cannot stretch.

---

## Experiment 5: BGE-base, MNRL + hard negatives, review_floor=0.60

**Setup:** BAAI/bge-base-en-v1.5 (110M params, 768 dims — 3x BGE-small), MNRL with hard negatives, 3 rounds, review_floor=0.60.

**Key result:** Significantly better than BGE-small. The larger model stretches rather than just compressing. Combined recall only dropped 3.3pp (99.7% → 96.4%) while review noise dropped 72% (3,005 → 834).

| | Round 0 (base) | Round 2 (last) |
|---|---|---|
| **Auto-matched** | **5,587** | **5,355** |
| Clean | 4,956 | 4,824 |
| Heavy noise | 625 | 531 |
| Not a match | 6 | 0 |
| **Review** | **4,408** | **2,172** |
| Clean | 1,005 | 938 |
| Heavy noise | 382 | 400 |
| Not a match | 3,005 | 834 |
| **Unmatched** | **5** | **2,473** |
| Missed (clean) | 1 | 216 |
| Missed (heavy noise) | 3 | 79 |
| Not a match | 1 | 2,178 |
| **Precision** | **88.7%** | **90.1%** |
| **Recall (vs ceiling)** | **82.9%** | **80.7%** |
| **Combined recall** | **99.7%** | **96.4%** |

**Observation:** BGE-base confirms the hypothesis — more model capacity enables stretching. Still converging at round 2. BGE-large (335M params, 1024 dims) is the next experiment.

---

## Experiment 6: BGE-base, MNRL + hard negatives, review_floor=0.60 (continued)

**Setup:** BAAI/bge-base-en-v1.5, MNRL with hard negatives, 5 rounds (full run), review_floor=0.60.

**Key result:** Confirmed BGE-base stretches. Combined recall 96.4% at R2 (best), degraded to 94.6% by R5. Delayed-start learning: flat R0-R1, steep R2-R3, plateau R4+.

---

## Experiment 7: BGE-small, MNRL + hard negatives, batch=32, review_floor=0.60

**Setup:** BAAI/bge-small-en-v1.5, MNRL with hard negatives, batch_size=32 (vs default 16), 7 rounds, review_floor=0.60.

**Key result:** Best overlap 0.081 at R7 (vs base 0.070). Batch size alone broke through the ceiling. But combined recall degraded continuously (98.7% → 94.6%), same compression pattern as smaller batches.

---

## Experiment 8: BGE-small, MNRL + hard negatives, batch=128, review_floor=0.60

**Setup:** BAAI/bge-small-en-v1.5, MNRL with hard negatives, batch_size=128 (4× default), 18 rounds, review_floor=0.60.

**Key result:** Best overlap 0.070 at R12 (vs exp 2's 0.081 at R7). Batch=128 broke through the old ceiling but plateaued lower. Combined recall 97.3% at best (164 missed clean), degraded to 94.6% by R17 (419 missed). Delayed-start learning confirmed: flat R0-R4, steep R5-R8, plateau R9+.

| Metric | R0 (base) | R8 (practical best) | R12 (overlap best) | R17 (final) |
|---|---|---|---|---|
| **Overlap** | 0.070 | 0.078 | 0.070 | 0.073 |
| **Combined recall** | 98.7% | 98.7% | 97.3% | 94.6% |
| **Missed (clean)** | 130 | 130 | 164 | 419 |

**Observation:** Batch=128 didn't improve the ceiling (0.070 vs 0.081 from batch=32). The 0.081 from exp 2 was a training signal problem (batch too small), not a capacity limit. But 0.070 is still above BGE-base's 0.046 — the 384-dim embedding space is the real bottleneck. Practical stopping point: R8 (overlap 0.078, recall 98.7%) for production use.

---

## Experiment 9: Snowflake Arctic-embed-xs + LoRA + batch=128 + MNRL, 23 rounds

**Setup:** Snowflake/arctic-embed-xs (22M params, 6 layers, 384 dims), MNRL with hard negatives, batch_size=128, 23 rounds, review_floor=0.60.

**Key result:** **BEST EXPERIMENT TO DATE.** Best overlap 0.031 at R22 — beats BGE-base (0.046) and BGE-small (0.070). Combined recall 99.7% from R14 onward (best of any trained model, and improved during training). Only 30 missed matches at R22 (19 clean + 11 heavy noise). Converged cleanly R17-R22 with no regression.

| Metric | R0 (base) | R8 | R14 (recall peak) | R22 (overlap best) |
|---|---|---|---|---|
| **Overlap** | 0.070 | 0.062 | 0.035 | **0.031** |
| **Combined recall** | 98.7% | 99.4% | **99.7%** | **99.7%** |
| **Missed (clean)** | 130 | 57 | 30 | 19 |
| **Missed (heavy noise)** | 0 | 0 | 0 | 11 |
| **Review FPs** | 2,826 | 1,247 | 184 | 184 |
| **Not-a-match in auto** | 131 | 0 | 0 | 0 |

**Key observations:**
1. **Pre-training quality > parameter count.** Arctic's 400M-sample pre-training with hard negative mining outweighs BGE-small's 33M params. The model stretches (separates distributions) rather than compresses.
2. **Stretching, not compression.** Arctic pushes non-matches down while keeping matches stable. BGE-small shifts everything together. This is the key difference enabling 99.7% recall.
3. **Fewer layers = larger LoRA intervention.** 6 layers vs BGE-small's 12 means each LoRA adapter has proportionally more influence, improving fine-tuning signal.
4. **Zero missed matches R2-R7.** The model briefly achieved perfect recall before the overlap improvement phase began (R8+). This is unique to Arctic.
5. **Embedding-only overlap (0.031) should drop to near-zero with BM25.** Experiment 10 will combine Arctic with BM25 to validate production viability.
6. **Smallest size, fastest speed.** 22M params vs BGE-small (33M) and BGE-base (110M). Encoding is 2–3× faster than BGE-base.

**Decision:** Arctic-embed-xs replaces BGE-small and BGE-base as the recommended embedding model for melder. See [[Key Decisions#Arctic-embed-xs as Recommended Embedding Model]].

---

## Summary of Findings

1. **Base model + low review_floor is the strongest baseline.** review_floor=0.60 with no fine-tuning achieves ~99.5% combined recall. The cost is a noisy review queue (~3,000 non-matches).

2. **Fine-tuning cleans the review queue but costs recall.** Every training approach pushes non-matches out of review. The question is how much recall it costs.

3. **BGE-small can't stretch — only shift.** 33M params / 384 dims is insufficient to independently separate match and non-match score distributions. Proven by neg-only (shifts down) and pos-only (shifts up) experiments.

4. **BGE-base can stretch.** 110M params / 768 dims has enough capacity to push non-matches down while keeping matches relatively stable. Combined recall loss of 3.3pp vs 12pp for BGE-small.

5. **MNRL > CosineSimilarityLoss.** Ranking objective preserves recall better than absolute calibration. Hard negatives provide marginal additional benefit.

6. **Batch size affects training signal, not capacity ceiling.** Batch=32 achieved overlap 0.081; batch=128 plateaued at 0.070. The 384-dim embedding space is the real bottleneck, not batch size.

7. **Acronym matching is a blind spot for all methods.** No embedding model, BM25, or fuzzy scorer can match "TRMS" to "Taylor, Reeves and Mcdaniel SRL." Documented separately in `vault/ideas/Acronym Matching.md`.
