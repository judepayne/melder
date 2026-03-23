# Science Experiments — Agent Instructions

## Purpose

This directory contains fixed datasets and a controlled training loop for comparing embedding fine-tuning approaches for melder's entity resolution pipeline. The same data is used across all experiments, enabling like-for-like comparison.

**Business context:** missed matches are far worse than review noise. A missed match means a duplicate record gets created in the reference master — expensive to detect and fix. False positives in review just cost a human 30 seconds to reject. Optimise for minimising missed matches, even at the cost of a larger review queue.

## Key Files

| File | Purpose |
|---|---|
| `experiments.md` | Numbered experiment plans, results tables, and observations |
| `research.md` | Web research findings on training approaches |
| `run.py` | Training loop (uses fixed datasets, no regeneration). `--name` is required — results go to `results/{name}/` |
| `output_table.py` | Produces holdout accuracy table from metrics.csv |
| `score_chart.py` | ASCII histogram of score distributions (matched vs unmatched). Use `--max-score 0.8` to zoom into overlap zone |
| `score_distribution.py` | Numeric score distribution table by population |
| `informal_analysis_unmatched.md` | Analysis of why unmatched records score high |
| `AGENTS.md` | This file — instructions for the agent |

Related vault docs:
- `vault/decisions/Training Experiments Log.md` — full history of pre-science experiments
- `vault/decisions/BGE Small Training Results.md` — why BGE-small can't stretch
- `vault/ideas/Acronym Matching.md` — identified blind spot for all scoring methods

## Before Running an Experiment

1. Read `experiments.md` — find the next `status: todo` experiment.
2. No cleanup needed — each experiment uses `--name` and gets its own `results/{name}/` directory.
3. Do NOT regenerate or modify anything in `master/`, `holdout/`, or `rounds/` — these are fixed.
4. The config template is at `benchmarks/accuracy/science/config.yaml` (review_floor=0.60).
5. All scripts (evaluate.py, pairs.py, finetune.py, export.py, plot.py) are local to this directory — fully self-contained.

## Running an Experiment

Use the command from the experiment's entry in `experiments.md`. All runs are from project root. Experiments run on a remote M1 Ultra Mac via SSH (Dropbox syncs files).

## After Each Experiment

1. Run `python benchmarks/accuracy/science/output_table.py --csv benchmarks/accuracy/science/results/{name}/metrics.csv` to produce the holdout table.
2. Run `python benchmarks/accuracy/science/score_chart.py --experiment {name} --round 0 --max-score 0.8` and `--round N` (last round) to show score distributions. Include R0 and final round charts in `experiments.md`.
3. Run `python benchmarks/accuracy/science/score_distribution.py --experiment {name} --round N` for detailed numeric breakdown if needed.
4. Fill in the Results table, score distribution charts, and Observations section in `experiments.md`.
5. Update `status: todo` → `status: done` in the experiment's yaml block.
6. No cleanup needed — each experiment's results live in `results/{name}/` and don't conflict.

## User's Preferred Table Format

Always present results in this format (one column per round, holdout only).

**Table width rule:** No more than 6-7 columns of numbers. For experiments with many rounds, show every other round — but always include the first (R0) and the last round. Same applies to overlap trajectory tables and markdown results tables in `experiments.md`.

```
══════════════════════════════════════════════════════════════
  [Model name] — holdout results

                                    Round 0   Round 1   ...
                                     (base)
══════════════════════════════════════════════════════════════

  Ceiling (max reachable)            X,XXX     X,XXX

  Auto-matched                       X,XXX     X,XXX
    Clean                            X,XXX     X,XXX
    Heavy noise                        XXX       XXX
    Not a match                          X         X

  Review                             X,XXX     X,XXX
    Clean                              XXX       XXX
    Heavy noise                        XXX       XXX
    Not a match                      X,XXX       XXX

  Unmatched                          X,XXX     X,XXX
    Missed (clean)                     XXX       XXX
    Missed (heavy noise)               XXX       XXX
    Not a match                      X,XXX     X,XXX

  Precision                         XX.X%     XX.X%
  Recall (vs ceiling)               XX.X%     XX.X%
  Combined recall                   XX.X%     XX.X%
══════════════════════════════════════════════════════════════
```

## Fixed Datasets

| Path | Seed | Description |
|---|---|---|
| `master/dataset_a.csv` | 0 | 10,000 A-side reference records |
| `holdout/dataset_b.csv` | 9999 | Fixed holdout B (never trained on) |
| `rounds/round_0/dataset_b.csv` | 100 | Training round 0 B |
| `rounds/round_1/dataset_b.csv` | 101 | Training round 1 B |
| ... | ... | ... |
| `rounds/round_9/dataset_b.csv` | 109 | Training round 9 B |

All B datasets use the 1:1 generator (60% matched / 10% heavy noise / 30% unmatched). Each A record maps to exactly one B record — no crossmap collisions.

## What to Compare Across Experiments

The key metrics to watch (all from holdout):

- **Missed (clean)**: should go DOWN with training. This is the primary target.
- **Missed (heavy noise)**: should go DOWN. Secondary target.
- **Not a match in review**: should go DOWN. Training's main proven benefit.
- **Not a match in auto**: should stay at 0. If it rises, precision is collapsing.
- **Combined recall**: should stay close to round 0 baseline (~99%). If it drops significantly, the model is compressing not stretching.

## Key Findings So Far

- **BGE-small (33M params) cannot stretch** — any training shifts all scores in the same direction. Proven by neg-only (shifts down) and pos-only (shifts up) experiments.
- **BGE-base (110M params) shows stretching** — combined recall drop of 3.3pp vs 12pp for small. The larger model can push non-matches down without dragging matches as far.
- **MNRL > CosineSimilarityLoss** — ranking objective preserves recall better than absolute calibration.
- **Base model + review_floor=0.60 is the strongest baseline** — ~99.5% combined recall with no training. The only problem is ~3,000 non-matches in the review queue.
- **Training's proven value is cleaning review** — pushes non-matches out of review (3,004 → 209 in best run). The cost is some recall loss.
- **Acronym matching is a blind spot** — no embedding model, BM25, or fuzzy scorer can match "TRMS" to "Taylor, Reeves and Mcdaniel SRL". Documented in `vault/ideas/Acronym Matching.md`.
- **The end-game is multi-method** — once we've optimised the embedding model, melder's composite scoring pipeline (embedding + wratio + BM25) will cover remaining gaps.

## Available Loss Functions

| Flag | Loss | Use case |
|---|---|---|
| `--loss mnrl` | MultipleNegativesRankingLoss | Default. Ranking objective with in-batch negatives + hard negatives from pairs.py |
| `--loss cached_mnrl` | CachedMultipleNegativesRankingLoss | Same as MNRL but simulates large batch sizes (e.g. 256) without memory cost |
| `--loss cosine` | CosineSimilarityLoss | Absolute calibration. Uses all pairs including label=0.0. Tends to compress scores. |
| `--loss posonly` | CosineSimilarityLoss (label>0 only) | Pure "push up" signal. Collapses precision — for diagnosis only. |
