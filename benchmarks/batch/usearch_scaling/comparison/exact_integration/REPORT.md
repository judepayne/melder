# Prepared exact keys: real batch integration

**Decision: do not promote this normalized-string prototype.** It is feasible in the shared pipeline, but these measurements show a wall-time regression, extra memory, and no credible scoring-throughput improvement. No production code/configuration was changed.

## Implementation / reproducibility

Source is a private `git archive 446d45e` at `/tmp/melder-opt.o6KzoE/integration-src`. `prepared-normalized.patch` and `reproduce.sh` recreate it. Four Rust files changed, all privately. Default `cargo build --release`, no feature overrides, target `/tmp/melder-scale.2E9P5m/target`; frozen binary `/tmp/melder-opt.o6KzoE/integration-normalized`. Baseline `/tmp/melder-opt.o6KzoE/baseline`. Binary/input hashes and environment are in each `metadata.json`.

A B→A batch SearchPlan precomputes per-record/per-binding trimmed ASCII lowercase strings once, after exact prefiltering and before scoring. Each search normalizes the query fields once; the real shared candidate loop looks up candidate keys once and routes cached equality into the existing weighted scoring/field-result implementation. Missing entries or non-ASCII values fall back to the original exact scorer. Trim uses Rust's original Unicode-aware `trim`; both empty remains zero. Unicode lowercase/context behavior is deliberately not approximated. Retrieval, union, exclusions, score accumulation, sorting and graph assignment are unchanged. The cache assumes immutable reference records (this batch fixture); it is not suitable for mutable live pools.

Preparation is inside measured process wall time and peak RSS, not the engine's scoring-only throughput interval. Pool representation is deliberately straightforward: record-ID HashMap with per-binding optional String slots (including empty slots for other methods). It adds record-ID copies, vectors and allocations. A compact ordinal/interned representation might reduce costs, but no evidence here validates such a design.

Validation: release exact scorer tests (2), shared-pipeline tests (23 before adding the new test), and the added real-loop prepared-vs-original parity test passed. The latter compares full MatchResults across empty/whitespace, ASCII, Unicode trim, accents, Turkish dotted I and Greek cases. No all-features matrix: isolated default-release benchmark prototype, not a production change/commit.

Use the [shared comparison runner](../run.py): pass `--variants exact-mixed --sizes 100000 1000000` and then `--variants exact-shortlist --sizes 100000`, separate fresh output paths, three repeats each. Seed `/tmp/melder-scale.2E9P5m/experiment`, RAYON_NUM_THREADS=20, ORT_NUM_THREADS=2, fresh COW cache per invocation, alternating binary order, three paired repeats for each requested cell. All 18 processes succeeded with exactly two warm combined-index hits. Raw full outputs remain in `/tmp/melder-opt.o6KzoE/integration-{mixed,shortlist}`; the measured rows, metadata, semantic hashes and parity summaries are archived here.

## End-to-end results

Wall/RSS columns below are separate arm medians; percentages use median of the three paired candidate/baseline ratios, not ratio of medians. RSS is macOS process peak, including preparation.

| Recipe / A size | Wall baseline → prepared (s) | Paired wall change | RSS baseline → prepared (GiB) | Paired RSS change | Paired scoring rate change |
|---|---:|---:|---:|---:|---:|
| exact-mixed / 100k | 5.26 → 5.76 | +9.5% | 2.299 → 2.314 | +0.7% | +0.7% |
| exact-mixed / 1M | 40.41 → 43.55 | +7.8% | 11.083 → 11.302 | +1.6% | −1.2% |
| exact-shortlist 200 / 100k | 3.73 → 4.11 | +7.4% | 2.163 → 2.172 | +0.6% | −2.3% |

All nine paired wall comparisons were slower (individual regressions 2.3–15.9%). At 1M, median peak RSS increased about 224 MiB (difference of arm medians). Three repeats cannot establish tight confidence bounds; scoring throughput fluctuated substantially, particularly shortlist baseline 5165–6313 records/s and candidate 5155–6170. There is no convincing speed benefit even in the exact-shortlist isolation recipe.

## Semantic evidence and limitations

* **Shortlist:** all three pairs and all six repeats had identical canonical hashes of every CSV cell, including scores/ranks and multiplicity. This is the strong end-to-end parity evidence.
* **Mixed:** none of the six pairs was fully CSV-identical. Each arm had three distinct complete semantic signatures. `parity-details.json` checks every pair of repeats, CSV decisions against WAL, and CrossMap membership against accepted relationships.
* **Mixed 1M:** all six runs had identical relationship decision sets, relationship CSV values/ranks, categories and accepted membership: 6082 matches, 2523 reviews, 1395 unmatched. Unmatched best-score WAL variation reached 0.02953 baseline/baseline versus 0.03094 across arms (also 0.03094 candidate/candidate). Exact-score identity cannot be inferred from complete mixed-output hashes.
* **Mixed 100k unexpected issue:** baseline runs and candidate repeats 1/3 had 611 matches, 276 reviews, 9113 unmatched. Candidate repeat 2 alone had 277 reviews and 9112 unmatched. CP-205102 changed from unmatched (best score 0.48580118715763093) to review against ENT-467748 (0.6283212458237539). Accepted matches remained unchanged. Baseline/baseline unmatched WAL score differences reached 0.05466; across arms/candidate repeats 0.07803. This category excursion was **not observed in the three baselines**, so mixed parity is unresolved, not passed. Shared-loop unit parity and stable shortlist parity support cached equality correctness, but do not prove this mixed excursion benign.

The code does not change candidate generation; the mixed recipe nevertheless does not provide reproducible actual candidate/score behavior across processes. No claim of identical mixed candidates or fully passed mixed parity is warranted. Investigation would require recording candidate-level retrieval/score traces or a deterministic replay fixture, not relaxing parity checks. Given the negative performance result, more variants/timings were not pursued.
