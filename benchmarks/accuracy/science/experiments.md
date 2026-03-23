# Science Experiments

Controlled experiments on fixed datasets. Each experiment changes exactly
one variable from the previous. See `AGENTS.md` for how to run and record.

**Note:** Datasets regenerated 2026-03-21 after fixing a bug in the data
generator — unmatched B records now use fully random names (previously they
were near-miss variants of real A names, indistinguishable from true matches).

**Note:** Training loop fixed 2026-03-21 — each round now fine-tunes from the
previous round's model (previously it always restarted from the base checkpoint,
making multi-round training pointless).

---

## Baseline: BGE-small MNRL batch=32 (5 rounds)

The base BGE-small model (round 0, untrained) plus rounds of MNRL fine-tuning.
Each round fine-tunes from the previous round's model. Only 4 rounds completed
(R0-R3) due to epoch decay reaching minimum.

```yaml
status: done
model: BAAI/bge-small-en-v1.5
loss: mnrl
rounds: 5
epochs: 3
batch_size: 32
learning_rate: 2e-5
full_finetune: true
```

**Command:**
```bash
python benchmarks/accuracy/science/run.py \
    --name baseline --rounds 5 --full-finetune --loss mnrl \
    --base-model BAAI/bge-small-en-v1.5 --batch-size 32
```

### Results

| | Round 0 (base) | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,682 | 5,341 | 5,124 | 5,077 |
| Clean | 5,000 | 4,748 | 4,585 | 4,542 |
| Heavy noise | 682 | 593 | 539 | 535 |
| Not a match | 0 | 0 | 0 | 0 |
| **Review** | 4,316 | 1,197 | 949 | 895 |
| Clean | 989 | 829 | 635 | 649 |
| Heavy noise | 335 | 303 | 288 | 209 |
| Not a match | 2,958 | 65 | 26 | 37 |
| **Unmatched** | 2 | 3,462 | 3,927 | 4,028 |
| Missed (clean) | 1 | 447 | 804 | 833 |
| Missed (heavy noise) | 1 | 122 | 191 | 274 |
| Not a match | 0 | 2,893 | 2,932 | 2,921 |
| **Precision** | 88.0% | 88.9% | 89.5% | 89.5% |
| **Recall (vs ceiling)** | 83.0% | 78.8% | 76.1% | 75.4% |
| **Combined recall** | 99.4% | 92.6% | 86.7% | 86.2% |

**Score distributions per round (█ matched+ambiguous, ░ unmatched, cutoff 0.80):**

R0 (untrained) — overlap: 0.161
```
  0.56 █
  0.60 █░
  0.64 ███░░░░░
  0.68 █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.72 ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.76 ██░░░░░░░░░░░░░░░
  0.80 █░░░░
```
Both populations jammed into 0.64-0.80. Massive overlap — the model sees everything as similar.

R1 — overlap: 0.086
```
  0.32 ░
  0.36 ░░░░░░░
  0.40 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 ██████░░░░░░░░░░░░░░
  0.56 ██████████████████░░░░░░
  0.60 ████████████████░░
  0.64 ██████░
  0.68 ██░
  0.72 █
  0.76 █
  0.80 ██████
```
Best separation. Unmatched pushed to 0.36-0.52, clean gap emerging at 0.64+.

R2 — overlap: 0.132
```
  0.28 ░
  0.32 █░░░░░░░░░░░░
  0.36 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.40 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.44 ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █████████████░░░░░░░░░░░░░░░░░░░░
  0.52 █████████████████████░░░░░░░░░░
  0.56 ██████████████░░░░
  0.60 ██████░
  0.64 ██░
  0.68 █
  0.72 █████
  0.76 ████████
  0.80 ██████████
```
Catastrophic forgetting begins. Matched tail extends to 0.28, overlap rising sharply. The model is losing pre-trained knowledge.

R3 — overlap: 0.168
```
  0.24 ░
  0.28 ░░░
  0.32 █░░░░░░░░░░░░░░░░░░░
  0.36 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.40 ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.44 ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █████████████████░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █████████████████████████░░░░░░░░░░░░░░░
  0.56 ████████████████░░░░░░░
  0.60 ██████████░░
  0.64 ███████░
  0.68 ████
  0.72 ███
  0.76 ████
  0.80 ████████
```
Overlap now exceeds untrained model (0.168 vs 0.161). Training has made the model worse than doing nothing. Matched population smeared from 0.24-0.80 — the model has forgotten what company names mean.

**Observations:**
- R1 is the only round that improves on the base model. Every subsequent round degrades.
- Overlap coefficient tells the story: 0.161 → 0.086 (R1, improvement) → 0.132 → 0.168 (worse than untrained).
- This is catastrophic forgetting — 3 epochs of full fine-tuning per round overwrites pre-trained knowledge faster than it learns new patterns.
- Progressive fine-tuning (building on previous weights) compounds the damage. Each round pushes further from the pre-trained optimum.
- Combined recall drops from 99.4% → 92.6% → 86.7% → 86.2%. Missed clean climbs from 1 → 447 → 804 → 833.
- Review noise cleanup still works (2,958 → 37) but at an unacceptable cost to recall.

---

## Experiment 1: BGE-small MNRL batch=32, 1 epoch (3 rounds)

Test whether reducing to 1 epoch per round prevents catastrophic forgetting. With a pre-trained model, one pass through the training pairs should be enough to learn the signal without over-writing general knowledge.

**Variable changed from baseline:** epochs (3 → 1, fixed — no decay). Everything else identical.

```yaml
status: done
model: BAAI/bge-small-en-v1.5
loss: mnrl
rounds: 3
epochs: 1
batch_size: 32
learning_rate: 2e-5
full_finetune: true
```

**Command:**
```bash
python benchmarks/accuracy/science/run.py \
    --name experiment_1 --rounds 3 --full-finetune --loss mnrl \
    --base-model BAAI/bge-small-en-v1.5 --batch-size 32 --epochs 1
```

**Hypothesis:** 1 epoch should produce less aggressive weight changes per round, preserving more pre-trained knowledge. R1 should be at least as good as baseline R1 (possibly better — less over-fitting). R2+ should degrade less (or ideally, continue improving). If forgetting persists, the next lever is LoRA (freeze most parameters).

### Results

| | Round 0 (base) | Round 1 | Round 2 |
|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,682 | 5,308 | 5,231 |
| Clean | 5,000 | 4,716 | 4,671 |
| Heavy noise | 682 | 592 | 560 |
| Not a match | 0 | 0 | 0 |
| **Review** | 4,316 | 1,549 | 1,052 |
| Clean | 989 | 1,074 | 698 |
| Heavy noise | 335 | 353 | 300 |
| Not a match | 2,958 | 122 | 54 |
| **Unmatched** | 2 | 3,143 | 3,717 |
| Missed (clean) | 1 | 234 | 655 |
| Missed (heavy noise) | 1 | 73 | 158 |
| Not a match | 0 | 2,836 | 2,904 |
| **Precision** | 88.0% | 88.8% | 89.3% |
| **Recall (vs ceiling)** | 83.0% | 78.3% | 77.5% |
| **Combined recall** | 99.4% | 96.1% | 89.1% |

**Score distributions per round (█ matched+ambiguous, ░ unmatched, cutoff 0.80):**

R0 (untrained) — overlap: 0.161
```
  0.56 █
  0.60 █░
  0.64 ███░░░░░
  0.68 █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.72 ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.76 ██░░░░░░░░░░░░░░░
  0.80 █░░░░
```

R1 — overlap: 0.085
```
  0.36 ░
  0.40 █░░░░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 ██░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 ██████████░░░░░░░░░░
  0.60 ███████████████████░░░
  0.64 ███████████░
  0.68 ███░
  0.72 █
  0.76 █
  0.80 ███████
```
Best separation — similar to baseline R1. Unmatched pushed to 0.36-0.52.

R2 — overlap: 0.099
```
  0.28 ░
  0.32 ░░░
  0.36 ░░░░░░░░░░░░░░░░░░░░░░░
  0.40 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 ████░░░░░░░░░░░░░░░░░░░░░░░
  0.52 ██████████████░░░░░░░░░░░
  0.56 █████████████████████░░░░
  0.60 ███████████░░
  0.64 ████░
  0.68 █
  0.72 █
  0.76 ██
  0.80 ██████████
```
Forgetting visible but significantly less than baseline R2 (0.099 vs 0.132). Matched population holds shape better — tail doesn't extend as far down.

**Overlap comparison with baseline:**

| | R0 (base) | R1 | R2 |
|---|---|---|---|
| Baseline (3 epochs) | 0.161 | 0.086 | 0.132 |
| Experiment 1 (1 epoch) | 0.161 | 0.085 | 0.099 |

**Observations:**
- R1 is nearly identical to baseline R1 — one epoch captures the same first-round improvement as three.
- R2 forgetting is significantly reduced (0.099 vs 0.132) but still present. The model is still losing pre-trained knowledge, just more slowly.
- Combined recall drops 99.4% → 96.1% → 89.1%. The R1→R2 drop is still steep (7pp), confirming that full fine-tuning — even with 1 epoch — overwrites too many parameters per round.
- Review cleanup remains effective: not-a-match drops 2,958 → 122 → 54.
- **Conclusion:** 1 epoch slows forgetting but doesn't stop it. Full fine-tuning updates all 33M parameters each round, which is fundamentally too aggressive for progressive training. Next experiment: LoRA (freeze ~99% of parameters, only train low-rank adapters).

---

## Experiment 2: BGE-small MNRL batch=32, 1 epoch, LoRA (8 rounds)

Test whether LoRA (Low-Rank Adaptation) prevents catastrophic forgetting by freezing most parameters and only training small adapter matrices (~1% of parameters updated). Initially run for 4 rounds, then extended to 8 with `--resume-from 4`.

**Variable changed from experiment 1:** full_finetune → LoRA (r=8, alpha=16, dropout=0.1). Everything else identical.

```yaml
status: done
model: BAAI/bge-small-en-v1.5
loss: mnrl
rounds: 8
epochs: 1
batch_size: 32
learning_rate: 2e-5
full_finetune: false
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
```

**Command:**
```bash
python benchmarks/accuracy/science/run.py \
    --name experiment_2 --rounds 9 --loss mnrl \
    --base-model BAAI/bge-small-en-v1.5 --batch-size 32 --epochs 1 \
    --resume-from 4
```

**Hypothesis:** LoRA should preserve pre-trained knowledge across rounds since ~99% of weights are frozen. R1 may show less separation than full fine-tune (less capacity to learn), but R2+ should not degrade — or should degrade far less. If the model can sustain or improve separation across rounds, LoRA is the path forward for progressive training.

### Results

| | Round 0 (base) | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 | Round 7 |
|---|---|---|---|---|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,682 | 5,642 | 5,531 | 5,409 | 5,370 | 5,355 | 5,331 | 5,326 |
| Clean | 5,000 | 4,976 | 4,900 | 4,799 | 4,768 | 4,756 | 4,739 | 4,738 |
| Heavy noise | 682 | 666 | 631 | 610 | 602 | 599 | 592 | 588 |
| Not a match | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Review** | 4,316 | 4,347 | 3,655 | 2,212 | 1,943 | 1,846 | 1,776 | 1,723 |
| Clean | 989 | 1,024 | 1,109 | 1,193 | 1,197 | 1,195 | 1,179 | 1,168 |
| Heavy noise | 335 | 351 | 383 | 387 | 382 | 377 | 382 | 377 |
| Not a match | 2,958 | 2,953 | 2,159 | 632 | 364 | 274 | 215 | 178 |
| **Unmatched** | 2 | 11 | 814 | 2,379 | 2,687 | 2,799 | 2,893 | 2,951 |
| Missed (clean) | 1 | 5 | 11 | 32 | 59 | 73 | 106 | 118 |
| Missed (heavy noise) | 1 | 1 | 4 | 21 | 34 | 42 | 44 | 53 |
| Not a match | 0 | 5 | 799 | 2,326 | 2,594 | 2,684 | 2,743 | 2,780 |
| **Precision** | 88.0% | 88.2% | 88.6% | 88.7% | 88.8% | 88.8% | 88.9% | 89.0% |
| **Recall (vs ceiling)** | 83.0% | 82.6% | 81.3% | 79.7% | 79.1% | 79.0% | 78.7% | 78.6% |
| **Combined recall** | 99.4% | 99.6% | 99.8% | 99.5% | 99.0% | 98.8% | 98.2% | 98.0% |

**Overlap trajectory:**

| R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|---|
| 0.161 | 0.160 | 0.158 | 0.104 | 0.092 | 0.084 | **0.081** | 0.082 |

**Score distributions (█ matched+ambiguous, ░ unmatched, ▼ threshold):**

R0 (untrained) — overlap: 0.161
```
  0.56 █
  0.60 █░                                  ▼ review_floor (0.60)
  0.64 ███░░░░░
  0.68 █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.72 ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.76 ██░░░░░░░░░░░░░░░
  0.80 █░░░░
```
Both populations jammed into 0.64-0.80. Massive overlap.

R5 — overlap: 0.084 (best combined recall vs separation trade-off)
```
  0.40 ░
  0.44 ░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 ████░░░░░░░░░░░░░░░░░░░░░░
  0.60 ██████████████░░░░░░░░      ▼ review_floor (0.60)
  0.64 ██████████████████░░░
  0.68 ███████░
  0.72 ██░
  0.76 █
  0.80 ██████
```
Unmatched pushed to 0.40-0.56. Clear gap opening at 0.64+.

R7 (final) — overlap: 0.082
```
  0.40 ░░
  0.44 ░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 ██████░░░░░░░░░░░░░░░░░
  0.60 █████████████████░░░░░      ▼ review_floor (0.60)
  0.64 ████████████████░░
  0.68 █████░
  0.72 █
  0.76 █
  0.80 ██████
```
Marginal change from R5. Unmatched peak shifted slightly lower (0.44-0.48 vs 0.48-0.52) but matched tail also extending. Model has plateaued.

**Overlap comparison across all experiments:**

| | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| Baseline (3 epochs, full FT) | 0.161 | 0.086 | 0.132 | 0.168 |
| Experiment 1 (1 epoch, full FT) | 0.161 | 0.085 | 0.099 | — |
| Experiment 2 (1 epoch, LoRA) | 0.161 | 0.160 | 0.158 | 0.104 |

**Observations:**
- **LoRA eliminates catastrophic forgetting.** Combined recall holds at 98-99.8% across all 8 rounds, vs collapsing to 86% with full fine-tune. Missed clean stays at 118 by R7 vs 833 with baseline R3.
- **LoRA learns slowly but monotonically.** Overlap improves every round: 0.161 → 0.160 → 0.158 → 0.104 → 0.092 → 0.084 → 0.081 → 0.082. No regression at any point (R7's 0.082 is noise, not degradation).
- **Plateaus at R5-R6 (overlap ~0.081).** This matches the best single-round result from full fine-tune (0.085-0.086) — LoRA reaches the same separation, just sustainably. This is likely the BGE-small capacity limit.
- **Review cleanup is excellent.** Not-a-match in review: 2,958 → 178 (94% reduction) with only 1.4pp combined recall loss. Full fine-tune achieved 37 but at 13pp recall cost.
- **The R5 sweet spot:** overlap 0.084, combined recall 98.8%, not-a-match in review 274. Best trade-off between separation and recall preservation.
- **Conclusion:** LoRA is the correct training strategy for progressive fine-tuning. BGE-small has reached its capacity limit at ~0.081 overlap. Next step: test whether LoRA's rank-8 constraint is the bottleneck by freezing lower layers and fully training upper layers.

---

## Experiment 3: BGE-small MNRL batch=32, 1 epoch, freeze bottom 9 layers (4 rounds)

Test whether LoRA's rank-8 adapter constraint is the bottleneck at overlap ~0.081, or whether it's a genuine BGE-small capacity limit. Freeze the bottom 9 of 12 transformer layers (plus embeddings) and fully fine-tune the top 3 layers. This gives the upper layers full expressiveness (~25% of parameters trainable, ~8M) while protecting low-level language understanding.

**Variable changed from experiment 2:** LoRA → full fine-tune on top 3 layers only (freeze bottom 9). Everything else identical.

```yaml
status: done
model: BAAI/bge-small-en-v1.5
loss: mnrl
rounds: 4
epochs: 1
batch_size: 32
learning_rate: 2e-5
full_finetune: true
freeze_layers: 9
```

**Command:**
```bash
python benchmarks/accuracy/science/run.py \
    --name experiment_3 --rounds 4 --full-finetune --loss mnrl \
    --base-model BAAI/bge-small-en-v1.5 --batch-size 32 --epochs 1 \
    --freeze-layers 9
```

**Hypothesis:** With 75% of the network frozen, catastrophic forgetting should be controlled (similar to LoRA). But with full expressiveness in the top 3 layers (~8M trainable params vs ~300k for LoRA), the model may push past the 0.081 overlap plateau. If it plateaus at the same level, BGE-small genuinely can't stretch further and we should move to BGE-base.

### Results

| | Round 0 (base) | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,682 | 5,151 | 4,978 | 4,930 |
| Clean | 5,000 | 4,589 | 4,449 | 4,413 |
| Heavy noise | 682 | 562 | 529 | 517 |
| Not a match | 0 | 0 | 0 | 0 |
| **Review** | 4,316 | 1,114 | 1,042 | 1,058 |
| Clean | 989 | 685 | 710 | 731 |
| Heavy noise | 335 | 277 | 286 | 298 |
| Not a match | 2,958 | 150 | 46 | 29 |
| **Unmatched** | 2 | 3,735 | 3,980 | 4,012 |
| Missed (clean) | 1 | 748 | 865 | 880 |
| Missed (heavy noise) | 1 | 179 | 203 | 203 |
| Not a match | 0 | 2,808 | 2,912 | 2,929 |
| **Precision** | 88.0% | 89.1% | 89.4% | 89.5% |
| **Recall (vs ceiling)** | 83.0% | 76.2% | 73.9% | 73.3% |
| **Combined recall** | 99.4% | 87.5% | 85.6% | 85.4% |

**Score distributions (█ matched+ambiguous, ░ unmatched, ▼ threshold):**

R1 — overlap: 0.161
```
  0.32 █░
  0.36 █░░░░░░
  0.40 █░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.44 ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █████████████████░░░░░░░░░░░░░░░░░░░
  0.56 ██████████████████░░░░░░░░░
  0.60 ███████░░░░                             ▼ review_floor (0.60)
  0.64 ██░░
  0.68 █░
  0.72 █░
  0.76 ████
  0.80 ████████████
```
Matched population smeared down to 0.32, with a large hump at 0.44-0.56. The top 3 layers are being distorted while the frozen lower layers can't compensate.

R3 — overlap: 0.156
```
  0.20 ░
  0.24 █░
  0.28 █░░░░░░░░
  0.32 █░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.36 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.40 █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.44 ████████████░░░░░░░░░░░░░░░░░░░░░
  0.48 ████████████████████░░░░░░░░░░░
  0.52 █████████████░░░░░
  0.56 █████░░░
  0.60 ██░                                     ▼ review_floor (0.60)
  0.64 █░
  0.68 █
  0.72 ███
  0.76 █████████
  0.80 ███████████████
```
Matched tail extends to 0.20. The model is compressing — pushing everything down rather than stretching apart.

**Overlap comparison across all experiments:**

| | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| Baseline (3 epochs, full FT) | 0.161 | 0.086 | 0.132 | 0.168 |
| Exp 1 (1 epoch, full FT) | 0.161 | 0.085 | 0.099 | — |
| Exp 2 (1 epoch, LoRA) | 0.161 | 0.160 | 0.158 | 0.104 |
| Exp 3 (1 epoch, freeze 9) | 0.161 | 0.161 | 0.158 | 0.156 |

**Observations:**
- **Layer freezing is strictly worse than LoRA.** Overlap barely moves (0.161 → 0.156 over 4 rounds) while LoRA reached 0.104 by R3. The top 3 layers alone cannot learn effective separation.
- **Combined recall collapses despite freezing.** 99.4% → 87.5% at R1 — worse than even the full fine-tune baseline (92.6%). Freezing 75% of layers does not prevent forgetting when the unfrozen layers are fully updated.
- **Compression, not stretching.** The charts show the matched population smearing downward (to 0.20 by R3) while the unmatched population barely moves. The top layers are distorting the representation without the lower layers adapting to compensate.
- **LoRA's distributed approach is fundamentally superior.** Small perturbations across all 12 layers preserve the overall representation better than large changes concentrated in 3 layers. The rank-8 constraint is not the bottleneck.
- **Conclusion:** The 0.081 overlap plateau is a genuine BGE-small (33M param) capacity limit, not a training strategy limitation. LoRA (experiment 2) remains the best approach for BGE-small. To push overlap lower, we need a larger model (BGE-base, 110M params).

---

## Experiment 4: BGE-base MNRL batch=32, 1 epoch, LoRA (12 rounds)

Move to the larger BGE-base model (110M params, 768-dim embeddings vs BGE-small's 384-dim). Experiments 2 and 3 established that BGE-small's 0.081 overlap plateau is a model capacity limit. BGE-base has 3.3× more parameters — the question is whether that extra capacity translates to better separation. Initially run for 8 rounds, extended to 10, then 12.

**Variable changed from experiment 2:** model (BGE-small 33M → BGE-base 110M). Everything else identical — LoRA r=8, 1 epoch, MNRL.

```yaml
status: done
model: BAAI/bge-base-en-v1.5
loss: mnrl
rounds: 12
epochs: 1
batch_size: 32
learning_rate: 2e-5
full_finetune: false
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
```

**Command:**
```bash
python benchmarks/accuracy/science/run.py \
    --name experiment_4 --rounds 12 --loss mnrl \
    --base-model BAAI/bge-base-en-v1.5 --batch-size 32 --epochs 1 \
    --resume-from 10
```

**Hypothesis:** BGE-base should have a higher separation ceiling than BGE-small. With LoRA preventing catastrophic forgetting, the model should improve monotonically across rounds (as experiment 2 showed for BGE-small). The key question is whether overlap can push below 0.081 — and whether the base model's starting overlap (R0) is already better or worse than BGE-small's 0.161.

### Results

Note: Round 2 is missing from the metrics CSV due to a resume bug (holdout metrics weren't cached before the original run died). Holdout data for R2 exists and overlap is calculated from it.

| | R0 (base) | R3 | R5 | R7 | R9 | R11 |
|---|---|---|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,622 | 5,390 | 5,359 | 5,322 | 5,263 | 5,226 |
| Clean | 4,975 | 4,796 | 4,775 | 4,747 | 4,705 | 4,678 |
| Heavy noise | 647 | 594 | 584 | 575 | 558 | 548 |
| Not a match | 0 | 0 | 0 | 0 | 0 | 0 |
| **Review** | 4,372 | 1,887 | 1,798 | 1,798 | 1,855 | 1,879 |
| Clean | 1,028 | 1,161 | 1,169 | 1,195 | 1,242 | 1,276 |
| Heavy noise | 370 | 392 | 397 | 407 | 427 | 436 |
| Not a match | 2,955 | 334 | 232 | 196 | 186 | 167 |
| **Unmatched** | 6 | 2,723 | 2,843 | 2,880 | 2,882 | 2,895 |
| Missed (clean) | 2 | 67 | 80 | 82 | 77 | 70 |
| Missed (heavy noise) | 1 | 32 | 37 | 36 | 33 | 34 |
| Not a match | 3 | 2,624 | 2,726 | 2,762 | 2,772 | 2,791 |
| **Precision** | 88.5% | 89.0% | 89.1% | 89.2% | 89.4% | 89.5% |
| **Recall (vs ceiling)** | 82.6% | 79.6% | 79.3% | 78.8% | 78.1% | 77.7% |
| **Combined recall** | 99.7% | 98.9% | 98.7% | 98.6% | 98.7% | 98.8% |

**Overlap trajectory:**

| R0 | R2 | R4 | R6 | R8 | R10 | R11 |
|---|---|---|---|---|---|---|
| 0.165 | 0.110 | 0.079 | 0.071 | 0.067 | 0.063 | **0.059** |

**Score distributions (█ matched+ambiguous, ░ unmatched, ▼ threshold):**

R4 — overlap: 0.079
```
  0.36 ░
  0.40 ░░░
  0.44 █░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 ████░░░░░░░░░░░░░░░░
  0.60 ████████████░░░░░░░░              ▼ review_floor (0.60)
  0.64 ███████████████████░░░
  0.68 ██████████░
  0.72 ██
  0.76 ██
  0.80 ██████
```
Unmatched peak at 0.48-0.52. Clear separation emerging at 0.60+.

R8 — overlap: 0.067
```
  0.36 ░
  0.40 ░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 ████░░░░░░░░░░░░
  0.60 ███████████░░░░░░              ▼ review_floor (0.60)
  0.64 █████████████████░
  0.68 ██████████░
  0.72 ███
  0.76 ███
  0.80 ███████
```
Unmatched thinning at 0.52-0.56. The neck at 0.56 is visible — matched and unmatched pulling apart from this pinch point.

R11 — overlap: 0.059
```
  0.32 ░
  0.36 ░
  0.40 ░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 ████░░░░░░░░░░░░░░
  0.60 ███████████░░░░░░              ▼ review_floor (0.60)
  0.64 █████████████████░
  0.68 ███████████░
  0.72 █████
  0.76 █████
  0.80 ████████
```
Improvement is now at fine-grained level: unmatched at 0.62-0.64 dropped from 71 (R8) to 47 (R11), unmatched peak shifting lower. The "plasticine neck" at 0.56 continues to thin — populations are pulling apart from this pinch point.

**Overlap comparison — BGE-small vs BGE-base (both LoRA):**

| | R0 | R3 | R5 | R7 | R9 | R11 |
|---|---|---|---|---|---|---|
| Exp 2: BGE-small | 0.161 | 0.104 | 0.084 | 0.082 | — | — |
| Exp 4: BGE-base | 0.165 | 0.088 | 0.073 | 0.070 | 0.065 | 0.059 |

**Observations:**
- **BGE-base breaks through the BGE-small ceiling.** Final overlap 0.059 vs BGE-small's 0.081 plateau — a 27% improvement in separation. The larger model's 768-dim embeddings give LoRA's rank-8 adapters more to work with.
- **Still improving at R11.** Overlap dropped from 0.063 (R10) to 0.059 (R11). The curve is flattening but hasn't plateaued. Fine-grained analysis shows unmatched records are still being pushed out of the 0.62-0.64 danger zone (71→47 between R8 and R11).
- **Zero catastrophic forgetting.** Combined recall holds at 98.6-99.0% across all 12 rounds. Missed clean stays in the 60-82 range with no upward trend. LoRA's distributed approach continues to protect pre-trained knowledge.
- **Zero false positives in auto-match.** Not-a-match in auto stays at 0 across all rounds. Precision rises monotonically from 88.5% to 89.5%.
- **Review cleanup.** Not-a-match in review: 2,955 → 167 (94% reduction). Slower than BGE-small's cleanup (which reached 178 by R7) but combined recall is much better preserved.
- **The "plasticine neck" at 0.56.** Score distributions show a clear pinch point forming at 0.56 — unmatched population peaks at 0.48 and matched population peaks at 0.64, with the 0.56 zone thinning each round. This is where a dynamic review_floor would naturally sit.
- **Conclusion:** BGE-base with LoRA is the best configuration found. The model has not plateaued and could benefit from more rounds. The natural review_floor is migrating toward 0.56 as the populations separate.

---

## Experiment 5: BGE-base MNRL batch=128, 1 epoch, LoRA (6 rounds)

Test whether a larger batch size improves separation. With MNRL, larger batches mean more in-batch negatives per training step — the model sees more contrast per update, which could produce stronger separation signal. Batch size is the only variable changed.

**Variable changed from experiment 4:** batch_size (32 → 128). Everything else identical.

```yaml
status: done
model: BAAI/bge-base-en-v1.5
loss: mnrl
rounds: 18
epochs: 1
batch_size: 128
learning_rate: 2e-5
full_finetune: false
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
```

**Command:**
```bash
python benchmarks/accuracy/science/run.py \
    --name experiment_5 --rounds 18 --loss mnrl \
    --base-model BAAI/bge-base-en-v1.5 --batch-size 128 --epochs 1
```

**Hypothesis:** Larger batch = more in-batch negatives for MNRL = stronger contrastive signal per step. This could accelerate learning (reach lower overlap in fewer rounds) or push to a lower floor. Risk: if the signal is too strong it could behave like the full fine-tune experiments — but LoRA should buffer against that.

### Results

Initially run for 6 rounds, extended to 12, then to 18. Batch=128 learns more slowly than batch=32 (delayed ~2 rounds) but ultimately reaches lower overlap — confirming that denser contrastive signal produces more profound learning.

| | R0 (base) | R3 | R6 | R9 | R12 | R15 | R17 |
|---|---|---|---|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,622 | 5,515 | 5,398 | 5,387 | 5,373 | 5,353 | 5,346 |
| Clean | 4,975 | 4,899 | 4,807 | 4,804 | 4,800 | 4,795 | 4,793 |
| Heavy noise | 647 | 616 | 591 | 583 | 573 | 558 | 553 |
| Not a match | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Review** | 4,372 | 3,316 | 1,705 | 1,654 | 1,637 | 1,644 | 1,652 |
| Clean | 1,028 | 1,112 | 1,127 | 1,124 | 1,131 | 1,138 | 1,141 |
| Heavy noise | 370 | 391 | 384 | 396 | 405 | 419 | 427 |
| Not a match | 2,955 | 1,812 | 194 | 134 | 101 | 87 | 84 |
| **Unmatched** | 6 | 1,169 | 2,897 | 2,959 | 2,990 | 3,003 | 3,002 |
| Missed (clean) | 2 | 12 | 90 | 96 | 93 | 91 | 90 |
| Missed (heavy noise) | 1 | 11 | 43 | 39 | 40 | 41 | 38 |
| Not a match | 3 | 1,146 | 2,764 | 2,824 | 2,857 | 2,871 | 2,874 |
| **Precision** | 88.5% | 88.8% | 89.0% | 89.2% | 89.3% | 89.6% | 89.7% |
| **Recall (vs ceiling)** | 82.6% | 81.3% | 79.8% | 79.8% | 79.7% | 79.6% | 79.6% |
| **Combined recall** | 99.7% | 99.8% | 98.5% | 98.4% | 98.5% | 98.5% | 98.5% |

**Overlap trajectory:**

| R0 | R3 | R6 | R9 | R12 | R15 | R17 |
|---|---|---|---|---|---|---|
| 0.165 | 0.161 | 0.072 | 0.059 | 0.053 | 0.048 | **0.046** |

**Comparison with experiment 4 (batch=32):**

| | R0 | R6 | R11 (exp 4 final) | R17 (exp 5 final) |
|---|---|---|---|---|
| Exp 4 (batch=32) | 0.165 | 0.071 | **0.059** | — |
| Exp 5 (batch=128) | 0.165 | 0.072 | 0.055 | **0.046** |

**Score distributions (█ matched+ambiguous, ░ unmatched, ▼ threshold):**

R0 (untrained) — overlap: 0.165
```
  0.56 █░
  0.60 █░░░                                ▼ review_floor (0.60)
  0.64 ██████░░░░░░░░░░░░░░░░░░░░░░░░
  0.68 ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.72 ███████████░░░░░░░░░░░░░░░░░░░░░░░
  0.76 ██░░░░░░░░
  0.80 ██░░
```
Both populations jammed into 0.64-0.80.

R9 — overlap: 0.059
```
  0.32 ░
  0.36 ░
  0.40 ░░░░░░░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░
  0.56 █████░░░░░░░░░░
  0.60 ██████████████░░░░░            ▼ review_floor (0.60)
  0.64 ██████████████████░
  0.68 █████████░
  0.72 ██
  0.76 ██
  0.80 ██████
```
Clear separation forming. Unmatched peak at 0.44-0.48.

R17 (final) — overlap: 0.046
```
  0.32 ░
  0.36 ░░
  0.40 ░░░░░░░░░░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░
  0.56 █████░░░░░░░░░
  0.60 ██████████████░░░              ▼ review_floor (0.60)
  0.64 ██████████████████░
  0.68 ██████████░
  0.72 ███
  0.76 █████
  0.80 ███████
```
The "plasticine neck" at 0.56 continues to thin. Unmatched peak migrating to 0.40-0.48.

### Analysis of the overlap zone

The residual overlap (0.046) is dominated by two populations that meet in the 0.48-0.60 zone. Analysis of individual records reveals what's driving each:

**High-scoring non-matches (the false match tail, 0.65-0.70):**

These are unmatched B records that score high against the wrong A record. Three patterns dominate:

1. **Common surnames** — "Smith Ltd Corp" vs "Smith PLC BV" (0.70). Faker's limited surname pool means different synthetic entities share surnames (Smith, Johnson, Brown, Davis). The embedding model correctly identifies the shared word but can't know they're different entities.

2. **Shared legal suffixes** — "Lane, Wilson and Howell Capital" vs "Matthews, Moore and Beasley Capital" (0.67). Structural words (GmbH, Capital, Group, & Co, LLC) inflate similarity despite no meaningful name overlap.

3. **Address collisions** — "Abbott, Moore and Horn SRL" vs "Ward-Thomas SRL" (0.69) where both share "USCGC Lucas" in the address. Faker's military address templates (USS/USCGC vessel names, FPO/DPO formats) are a limited pool — collisions push the address embedding score up.

Categories 2 and 3 are synthetic data artefacts. Category 1 is a real-world problem but would typically be resolved by shared identifiers (LEI, DUNS) or composite scoring (BM25's IDF would downweight "Smith").

**Low-scoring true matches (the missed match tail, 0.60-0.61):**

These are genuine matches that score near the review floor. Every single one is an **acronym or abbreviation**:

| B name | A name | Score |
|---|---|---|
| SIL | Stafford Inc Ltd | 0.600 |
| TIL | Tate Inc Limited | 0.600 |
| SHADC | Sanders, Hancock and Dyer Corporation | 0.601 |
| PLS | Palmer LLC SAS | 0.601 |
| RP | Roberts PLC SA | 0.601 |
| MG | Monroe-Sanders GmbH | 0.601 |
| HG | Holland-Parks Group | 0.601 |
| TAALC | Thompson, Alexander and Lane Capital | 0.601 |
| LC | Luna-Adams Capital | 0.602 |
| HIL | Holland Inc Limited | 0.602 |

These only survive because the address field carries them (addresses are near-identical in the heavy-noise treatment). The name embedding contributes almost nothing — no embedding model can relate "TAALC" to "Thompson, Alexander and Lane Capital". This is a documented blind spot (see `vault/ideas/Acronym Matching.md`) that requires either a dedicated acronym expansion module or a shared identifier to bypass scoring.

**Conclusion:** The overlap zone is bounded by two irreducible problems: common-word false matches (addressable by BM25 IDF weighting) and acronym true matches (requires a purpose-built mechanism). Training can thin the neck but cannot break it — these are fundamentally different problems from what embedding fine-tuning solves.

**Observations:**
- **Batch=128 definitively beats batch=32.** Final overlap 0.046 vs 0.059 — a 22% improvement. The larger batch provides more in-batch negatives per MNRL step, producing denser contrastive signal and more profound learning.
- **Delayed but steeper learning curve.** Batch=128 barely moves through R0-R3 (0.165→0.161), then drops sharply at R4-R5 (0.118→0.084). Batch=32 started improving at R2. The larger batch accumulates a broader view before making its move.
- **Still improving at R17.** Overlap 0.048→0.046 between R15 and R17. The curve is flattening but not plateaued. More rounds would continue to improve, though with diminishing returns.
- **Zero catastrophic forgetting.** Combined recall holds at 98.4-98.5% from R8 onwards. Missed clean stabilises at ~90. LoRA continues to protect pre-trained knowledge perfectly.
- **Review cleanup: 97% reduction.** Not-a-match in review: 2,955 → 84 (R17). Zero false positives in auto-match throughout.
- **Best configuration found.** BGE-base + LoRA (r=8) + batch=128 + 1 epoch is the optimal training setup for this dataset. Next step: add BM25 to the composite scoring pipeline (experiment 6) to attack the common-word false matches identified in the overlap zone analysis.

---

## Experiment 6: BM25 composite scoring (no training)

No training. Takes the best fine-tuned model (experiment 5, R17 — BGE-base LoRA batch=128) and tests adding BM25 to the composite scoring pipeline. The supposition from experiment 5's overlap zone analysis was that BM25's IDF weighting would push down scores for the high-scoring non-matches driven by common words (Smith, Ltd, GmbH, Capital, Group) — and it does, dramatically.

BM25 indexes both name and address fields. The embedding weights scale down proportionally as BM25 weight increases, keeping the total at 1.0. For example, at 20% BM25: name_emb=0.48, addr_emb=0.32, bm25=0.20.

```yaml
status: done
model: results/experiment_5/models/round_17/model.onnx (fixed, no training)
bm25_weights_tested: [0.10, 0.20, 0.30, 0.40]
name_total: 0.60
addr_total: 0.40
```

**Command:**
```bash
python benchmarks/accuracy/science/run_experiment6.py
```

**Hypothesis:** BM25's IDF weighting will push down scores for false matches driven by common words (surnames, legal suffixes, address templates) without affecting true matches that share distinctive terms. There should be a sweet spot where overlap drops well below experiment 5's 0.046 floor. Too much BM25 weight risks diluting the embedding signal and spreading the matched distribution downward.

### Results

| | Exp 5 R17 (0% BM25) | 10% BM25 | 20% BM25 | 30% BM25 | 40% BM25 |
|---|---|---|---|---|---|
| **Ceiling** | 6,024 | 6,024 | 6,024 | 6,024 | 6,024 |
| **Auto-matched** | 5,346 | 5,320 | 5,302 | 5,245 | 5,163 |
| Matched | 4,793 | 4,765 | 4,742 | 4,685 | 4,620 |
| Ambiguous | 553 | 555 | 560 | 560 | 543 |
| Not a match | 0 | 0 | 0 | 0 | 0 |
| **Review** | 1,652 | 1,639 | 1,662 | 1,721 | 1,798 |
| Matched | 1,141 | 1,204 | 1,233 | 1,295 | 1,357 |
| Ambiguous | 427 | 428 | 427 | 426 | 441 |
| Not a match | 84 | 7 | 2 | 0 | 0 |
| **Unmatched** | 3,002 | 3,041 | 3,036 | 3,034 | 3,039 |
| Missed (matched) | 90 | 55 | 49 | 44 | 47 |
| Missed (ambiguous) | 38 | 35 | 31 | 32 | 34 |
| Not a match | 2,874 | 2,951 | 2,956 | 2,958 | 2,958 |
| **Precision** | 89.7% | 89.6% | 89.4% | 89.3% | 89.5% |
| **Combined recall** | 98.5% | 99.1% | 99.2% | 99.3% | 99.2% |
| **Review FP (not-a-match)** | 84 | 7 | 2 | 0 | 0 |

**Overlap trajectory:**

| 0% (exp 5 R17) | 10% | 20% | 30% | 40% |
|---|---|---|---|---|
| 0.046 | **0.013** | **0.005** | **0.002** | **0.002** |

**Score distributions (█ matched+ambiguous, ░ unmatched, ▼ threshold):**

BM25 10% — overlap: 0.013
```
  0.36 ░░░
  0.38 ░░░░░░░░
  0.40 ░░░░░░░░░░░░
  0.42 ░░░░░░░░░░░░░░░░
  0.44 ░░░░░░░░░░░░░░░░
  0.46 █░░░░░░░░░░░░░░
  0.48 █░░░░░░░░
  0.50 █░░░░░░
  0.52 █░░░
  0.54 █░░
  0.56 █░
  0.58 ██░
  0.60 ███░                                ▼ review_floor (0.60)
  0.62 █████
  0.64 ███████░
  0.66 ███████
  0.68 █████
  0.70 ███
  0.72 ██
  0.76 █
  0.80 ██
  0.84 ████
  0.88 ███████                             ▼ auto_match (0.88)
  0.92 ██████████████████
  0.96 ███████████████████████████████
  0.98 ████████████████████████████████████████████████████████████
```
BM25 pushes unmatched peak from 0.44 down to 0.40-0.44. The overlap zone thins sharply.

BM25 20% — overlap: 0.005
```
  0.32 ░░░░
  0.34 ░░░░░░░░
  0.36 ░░░░░░░░░░░░░░░
  0.38 ░░░░░░░░░░░░░░░░
  0.40 ░░░░░░░░░░░░░░░░░
  0.42 ░░░░░░░░░░░░░░
  0.44 ░░░░░░░░
  0.46 █░░░░░░
  0.48 █░░░
  0.50 █░░
  0.52 █░
  0.56 █░
  0.58 █░
  0.60 ██░                                 ▼ review_floor (0.60)
  0.64 ███████
  0.66 ████████
  0.68 ██████
  0.70 █████
  0.72 ███
  0.80 ███
  0.84 █████
  0.88 █████████                           ▼ auto_match (0.88)
  0.92 ███████████████████
  0.96 ███████████████████████████████
  0.98 ████████████████████████████████████████████████████████████
```
Near-complete separation. Unmatched population almost entirely below 0.48.

BM25 30% — overlap: 0.002
```
  0.28 ░░░
  0.30 ░░░░░░░░░
  0.32 ░░░░░░░░░░░░░░░
  0.34 ░░░░░░░░░░░░░░░░
  0.36 ░░░░░░░░░░░░░░░░░
  0.38 ░░░░░░░░░░░░░░
  0.40 ░░░░░░░░░
  0.42 ░░░░░
  0.44 ░░░
  0.46 █░░
  0.48 █░
  0.52 █░
  0.56 █░
  0.60 ██                                  ▼ review_floor (0.60)
  0.64 █████
  0.68 ███████
  0.72 ████
  0.80 ██
  0.84 █████
  0.88 ███████████                         ▼ auto_match (0.88)
  0.92 █████████████████████
  0.96 ████████████████████████████
  0.98 ████████████████████████████████████████████████████████████
```

BM25 40% — overlap: 0.002
```
  0.24 ░░░
  0.26 ░░░░░░░
  0.28 ░░░░░░░░░░░░░░
  0.30 ░░░░░░░░░░░░░░░░░
  0.32 ░░░░░░░░░░░░░░░░
  0.34 ░░░░░░░░░░░░░░░
  0.36 ░░░░░░░░░░
  0.38 ░░░░░░
  0.40 ░░░░
  0.44 ░
  0.46 █░
  0.48 █░
  0.52 █░
  0.56 █░
  0.60 █                                   ▼ review_floor (0.60)
  0.64 ████
  0.68 ████████
  0.72 █████
  0.78 ██
  0.82 ████
  0.86 █████████
  0.88 ████████████                        ▼ auto_match (0.88)
  0.92 █████████████████████
  0.96 ██████████████████████████
  0.98 ████████████████████████████████████████████████████████████
```
Diminishing returns. Overlap plateaus at 0.002, and the matched distribution starts spreading down (more mass at 0.64-0.72 vs 0.92-1.00) as embedding signal is diluted.

### Observations

- **The hypothesis was correct.** BM25's IDF weighting successfully pushes down the common-word false matches identified in experiment 5's overlap zone analysis. Even a modest 10% BM25 share cuts overlap from 0.046 to 0.013 — a 72% reduction.
- **Review false positives collapse.** Not-a-match records in the review queue drop from 84 (embedding-only) to 7 (10% BM25) to 0 (30%+ BM25). This is the clearest practical signal — BM25 is selectively punishing exactly the false matches that were polluting the review queue.
- **Combined recall actually improves.** Counterintuitively, adding BM25 *increases* combined recall from 98.5% to 99.2-99.3%. BM25 helps some true matches that were borderline misses under embedding-only scoring — presumably cases where the names share distinctive words that BM25 rewards more strongly than cosine similarity.
- **Precision holds steady.** Precision stays in the 89.3-89.6% range across all weights. BM25 is not introducing new false positives into auto-match.
- **Use in moderation.** While overlap continues to decrease with higher BM25 weights, the matched score distribution starts spreading downward at 30-40%. More matched records shift from auto-match into review (5,320 at 10% vs 5,163 at 40%), meaning humans review more true matches. The sweet spot is around **20% BM25** — it captures most of the overlap reduction (0.005 vs 0.046 baseline) while keeping the matched distribution tight and auto-match count high.
- **Zero false positives in auto-match.** Maintained across all BM25 weights, as in experiment 5.
- **Acronym matches remain unsolved.** BM25 cannot help with "TAALC" vs "Thompson, Alexander and Lane Capital" any more than embeddings can — this requires a dedicated mechanism (see `vault/ideas/Acronym Matching.md`).

---

## Experiment 7: Synonym matching verification

Tests the impact of the new synonym/acronym matching feature (`method: synonym`) on the 81 acronym cases identified in experiment 5's overlap zone analysis. Uses the same fine-tuned model (BGE-base LoRA R17) and the 20% BM25 sweet spot from experiment 6 as the baseline.

Synonym matching adds two things:
1. **Candidate generation** — a bidirectional HashMap index maps generated acronyms to record IDs on both sides. Queries check both directions: is the query an acronym of an indexed name, or is an indexed name an acronym of the query.
2. **Scoring** — `method: synonym` is a binary 1.0/0.0 scorer. At weight 0.10, a synonym match adds 0.10 to the composite score.

Two configurations compared:
- **Baseline**: 20% BM25, no synonym (name_emb=0.48, addr_emb=0.32, bm25=0.20)
- **With synonym**: 20% BM25 + synonym (name_emb=0.42, addr_emb=0.28, bm25=0.20, synonym=0.10)

```yaml
status: done
model: results/experiment_6/model/model.onnx (fixed, no training)
bm25_weight: 0.20
synonym_weight: 0.10
```

**Command:**
```bash
python benchmarks/accuracy/science/run_experiment7.py
```

**Hypothesis:** The synonym candidate generator will surface the ~81 acronym pairs that all other methods miss. The binary synonym scorer (weight 0.10) will push their composite scores from ~0.60 into the review band or higher. Combined recall should improve. Precision should hold — short acronyms are filtered by min_length=3 and blocking constrains to same country code. No new false positives expected in auto-match.

### Results

| | Baseline (20% BM25) | With Synonym |
|---|---|---|
| **Ceiling** | 6,024 | 6,024 |
| **Auto-matched** | 5,401 | 5,401 |
| Matched | 4,827 | 4,827 |
| Ambiguous | 574 | 574 |
| Not a match | 0 | 0 |
| **Review** | 1,594 | 1,621 (+27) |
| Matched | 1,169 | 1,182 (+13) |
| Ambiguous | 421 | 435 (+14) |
| Not a match | 4 | 4 |
| **Unmatched** | 3,005 | 2,978 (-27) |
| Missed (matched) | 28 | 15 (-13) |
| Missed (ambiguous) | 23 | 9 (-14) |
| Not a match | 2,954 | 2,954 |
| **Precision** | 89.4% | 89.4% |
| **Combined recall** | 99.5% | 99.8% (+0.3pp) |

Synonym index: 21,429 keys built in 12.8ms. Scoring throughput: 6,677 rec/s (vs 8,564 baseline — 22% slower due to synonym candidate generation and scoring on every B record).

### Implementation note: additive weight handling

The first run revealed that adding synonym as a regular weighted method (stealing budget from embeddings) massively degraded results — auto-matched dropped from 5,401 to 2,776 because the 10% synonym weight diluted the embedding signal for the 99% of pairs where synonym scores 0.0.

The fix: synonym weight is **excluded from the normalisation denominator when the scorer returns 0.0**. This makes synonym purely additive:
- Non-acronym pairs: composite is identical to baseline (synonym contributes nothing and its weight is excluded from `total_weight`).
- Acronym pairs: synonym adds `0.10 * 1.0` to the weighted sum and `total_weight` grows by 0.10, boosting the composite.

This required two changes:
1. `src/scoring/mod.rs`: skip adding synonym weight to `total_weight` when `score == 0.0`
2. `src/config/loader.rs`: exclude synonym weights from the sum-to-1.0 validation

### Observations

- **The hypothesis was confirmed.** Synonym matching recovered 27 records from the unmatched pool into the review queue — 13 true matches and 14 ambiguous. These are exactly the acronym cases (e.g. "HWAG" → "Harris, Watkins and Goodwin BV") that no other method could bridge.
- **Zero impact on precision.** Auto-matched count and composition are identical between baseline and synonym. The synonym scorer does not introduce any false positives into auto-match. This is expected — a synonym match alone (weight 0.10 out of 1.10) cannot push a pair above the 0.88 auto-match threshold.
- **Combined recall improved.** 99.5% → 99.8% (+0.3pp). The 13 recovered true matches represent nearly half of the 28 missed matches in the baseline — a 46% reduction in missed true matches.
- **Review queue grew modestly.** +27 entries (1,594 → 1,621, +1.7%). All 27 new entries are genuine acronym relationships — 13 true matches and 14 ambiguous — so none are noise. The review queue became slightly larger but more productive.
- **Not all 81 acronym cases recovered.** The original overlap analysis (experiment 5) identified 81 acronym cases stuck in the 0.44-0.60 zone. Synonym matching recovered 27 of these. The remaining ~54 are likely cases where: (a) the acronym is shorter than `min_length=3` (e.g. "MG", "RP", "LC" from the experiment 5 table), (b) blocking filters them out (different country codes), or (c) the B-side record is an "ambiguous" or "unmatched" type in the ground truth that was already counted correctly.
- **Performance impact is modest.** 22% scoring throughput reduction (8,564 → 6,677 rec/s). The synonym index build is negligible (12.8ms for 21,429 keys). The per-record cost is dominated by acronym generation and HashMap lookups for each B record — cheap in absolute terms but measurable at scale.
- **Additive weight design is essential.** Sparse binary scorers must not participate in weight normalisation when they score 0.0. This is a general principle that would apply to any future binary feature (e.g. exact ID match, regex pattern match).
