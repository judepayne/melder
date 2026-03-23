---
name: Meld tune ground-truth enhancement
description: Planned enhancement to meld tune — use common_id_field as ground truth for two-population analysis (matched vs unmatched distributions, overlap coefficient, real precision/recall)
type: project
---

Planned enhancement to `meld tune`: use `common_id_field` as ground truth to split scores into matched vs unmatched populations. Currently tune shows one undifferentiated score histogram. With ground truth it could show:

- Two-population distribution charts (matched █ vs unmatched ░)
- Overlap coefficient (0 = perfect separation, 1 = identical)
- Real precision/recall at current thresholds
- Optimal threshold suggestions based on population separation

User workflow: provide a subset of real data where correct matches are known via a shared ID, run `meld tune`, get actionable feedback on model/weights/thresholds.

Also connects to curriculum learning idea — tune could identify easy vs hard pairs based on score distance from decision boundary.

**Why:** Current tune output is limited — percentile-based suggestions with no ground truth. The science experiment tooling (score_chart.py, overlap.py, evaluate.py) proved this analysis is essential. Building it into the Rust binary makes it accessible to all users.

**How to apply:** After experiment 4 results are analysed, explore and implement this enhancement in `src/cli/tune.rs`.
