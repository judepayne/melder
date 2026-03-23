---
name: Training experiments current state
description: Current state of embedding fine-tuning experiments — science framework, catastrophic forgetting investigation, experiment 1 running
type: project
---

As of 2026-03-21, embedding fine-tuning experiments use a controlled "science" framework at `benchmarks/accuracy/science/`. Fixed datasets (10k×10k, 1:1 generator, seeds frozen) enable like-for-like comparison.

**Critical bugs fixed 2026-03-21:**
- Data generator: unmatched B records used near-miss names (indistinguishable from matches). Fixed to use fully random names.
- Training loop: always fine-tuned from base checkpoint, not previous round's model. Fixed to chain weights progressively.

**Current state:**
- Baseline (done): BGE-small, MNRL, batch=32, 3 epochs, full fine-tune. Shows catastrophic forgetting — overlap 0.161→0.086 (R1, only improvement)→0.132→0.168 (R3, worse than untrained).
- Experiment 1 (running on remote M1 Ultra): Same but 1 epoch/round. Testing if lighter training prevents forgetting.
- If 1 epoch insufficient, next levers: lower LR, LoRA, MNRL temperature, weight decay, layer freezing (documented in `training_ideas.md`).

**Key conclusion:** BGE-small can't stretch. Base model + review_floor=0.60 achieves ~99.5% combined recall but ~3,000 non-matches in review. Training cleans review but costs recall. The challenge is preventing catastrophic forgetting during progressive fine-tuning.

**Remote setup:** M1 Ultra Mac (64GB) via SSH, files sync via Dropbox.

**Why:** Fine-tuning embedding model before combining with melder's composite scoring (wratio, BM25).

**How to apply:** Read `benchmarks/accuracy/science/AGENTS.md` and `experiments.md` at session start. Use `output_table.py`, `score_chart.py`, `overlap.py` for analysis.
