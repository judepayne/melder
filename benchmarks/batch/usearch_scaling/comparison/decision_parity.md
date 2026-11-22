# Decision parity: retained final-paired trials

Source: `/tmp/melder-opt.o6KzoE/final-paired`; 60 trials, five sizes × two recipes × three trials per binary. `decision_gate.py` calls existing `parity.load` and `parity.compare`: CSV identities, duplicate checks, CrossMap equality, full-precision decision WAL. Each row reports ranges across **3 BB, 9 BC, or 3 CC** comparisons; comparisons share trials and are not independent samples. Recipe `baseline` is distinct from binary baseline.

**Accepted edges are exactly identical in every comparison.** Review/classification parity is not universal. Rank/reason changes are zero throughout. No scientific timing claims.

## Per size and recipe

A/R/C/U = symmetric accepted-edge difference, review-edge difference, query category changes, unmatched-membership difference. UB = changed rounded CSV unmatched best scores; W = changed full-precision WAL scores (all categories); ΔW = largest unrounded WAL score delta. Counts include only common keys; membership changes are separately shown.

|Size/recipe|Arms|A/R/C/U ranges|UB range|W range|ΔW range|
|---|---|---|---|---|---|
|10000/baseline|baseline/baseline|0/0/0/0|81–94|81–94|0.0417623073–0.0778643471|
|10000/baseline|baseline/candidate|0/0/0/0|68–92|68–92|0.0476098137–0.0778643471|
|10000/baseline|candidate/candidate|0/0/0/0|66–86|66–86|0.0646751816–0.0778643471|
|10000/exact-mixed|baseline/baseline|0/0/0/0|78–92|78–92|0.042734873–0.0431420678|
|10000/exact-mixed|baseline/candidate|0/0/0/0|69–101|70–101|0.042734873–0.0779789327|
|10000/exact-mixed|candidate/candidate|0/0/0/0|85–97|85–98|0.0431420678–0.0779789327|
|100000/baseline|baseline/baseline|0/0–1/0–1/0–1|182–200|183–201|0.0668453366–0.0866956575|
|100000/baseline|baseline/candidate|0/0–1/0–1/0–1|172–197|172–198|0.0516527463–0.0866956575|
|100000/baseline|candidate/candidate|0/0–1/0–1/0–1|189–200|191–201|0.0532312579–0.0674289241|
|100000/exact-mixed|baseline/baseline|0/0/0/0|182–210|183–211|0.0535021201–0.0780260918|
|100000/exact-mixed|baseline/candidate|0/0–1/0–1/0–1|182–208|184–209|0.0537701342–0.0780260918|
|100000/exact-mixed|candidate/candidate|0/0–1/0–1/0–1|207–213|208–213|0.0620431039–0.0780260918|
|250000/baseline|baseline/baseline|0/0/0/0|179–198|179–198|0.0761438624–0.0776910926|
|250000/baseline|baseline/candidate|0/0/0/0|183–199|183–201|0.0607768686–0.0776910926|
|250000/baseline|candidate/candidate|0/0/0/0|190–205|192–205|0.0633199042–0.0761438624|
|250000/exact-mixed|baseline/baseline|0/0/0/0|186–202|186–202|0.0685294762–0.0734480986|
|250000/exact-mixed|baseline/candidate|0/0/0/0|188–211|188–211|0.0611045524–0.109067259|
|250000/exact-mixed|candidate/candidate|0/0/0/0|192–206|192–206|0.0734480986–0.109067259|
|500000/baseline|baseline/baseline|0/0/0/0|110–123|110–123|0.0566474423–0.0609022941|
|500000/baseline|baseline/candidate|0/0/0/0|104–119|104–119|0.0514053374–0.0790773381|
|500000/baseline|candidate/candidate|0/0/0/0|113–120|113–120|0.0544401787–0.0609022941|
|500000/exact-mixed|baseline/baseline|0/0/0/0|107–124|107–124|0.0558656215–0.0566474423|
|500000/exact-mixed|baseline/candidate|0/0/0/0|99–132|99–132|0.0548120647–0.0566474423|
|500000/exact-mixed|candidate/candidate|0/0/0/0|117–123|117–124|0.051740793–0.0566474423|
|1000000/baseline|baseline/baseline|0/0/0/0|8–9|8–9|0.023126363–0.0295325658|
|1000000/baseline|baseline/candidate|0/0/0/0|4–11|5–12|0.0119764446–0.0327194907|
|1000000/baseline|candidate/candidate|0/0/0/0|8–13|8–13|0.0253450699–0.0327194907|
|1000000/exact-mixed|baseline/baseline|0/0/0/0|8–12|8–12|0.0273229623–0.0309418857|
|1000000/exact-mixed|baseline/candidate|0/0/0/0|6–11|6–11|0.0115189006–0.0309418857|
|1000000/exact-mixed|candidate/candidate|0/0/0/0|10|10|0.0273229623–0.0309418857|

## Interpretation and cross-only anomalies

- 100k/baseline has review↔unmatched changes in BB, BC and CC. 100k/exact-mixed has them in BC and CC but **not BB**. Thus the candidate repeats do not justify claiming classification parity, even though accepted links never change.
- **500k/baseline cross-only score anomaly:** BC maximum unrounded delta 0.0790773381 exceeds both BB and CC maxima 0.0609022941. This cannot be described as entirely within measured same-binary variability. It does not change membership in these trials. Other ranges likewise must not be treated as equivalence bounds or proof of causation.
- 10k/exact-mixed BC and CC reach 0.0779789327 versus BB 0.0431420678; 250k/exact-mixed BC and CC reach 0.109067259 versus BB 0.0734480986. Candidate variability is not necessarily bounded by baseline variability.
- Full per-arm ranges, exact anomaly pair identities and unrounded WAL decision examples are retained in `decision-summary.json`. Unmatched CSV scores are rounded; WAL is the authoritative precision comparison. Nonzero score variability is not merely roundoff.

## Synthetic BM25 release-binary gate

`python3 benchmarks/batch/usearch_scaling/comparison/decision_gate.py --root /path/to/comparison-workspace --synthetic`

Runs actual `/tmp/melder-opt.o6KzoE/baseline` and `candidate`, serially with `RAYON_NUM_THREADS=1`, explicit graph/exact, memory and SQLite. Retained runs: `/tmp/melder-opt.o6KzoE/synthetic-decision-gate-v5`. Configs and logs remain there. Main blocks contain 31 and 5101 records (small exhaustive versus >5000 WAND dispatch); three extra records test missing first/second/both components of the composite AND key. Distinct per-record finite terms give each positive query exactly one matching document, avoiding top-k cutoff ties. Six positive queries and one no-hit query per case. This narrow deterministic gate does not stress contested graph assignment or dense WAND pruning.

All four binary comparisons have identical complete relationship/unmatched CSV rows, six accepted links and one unmatched query. All output SQLite tables/rows compare exactly, including unrounded REAL relationship scores and (SQLite backend) field scores. Memory WAL decisions compare exactly at full precision. SQLite batch emits no decision WAL; do **not** interpret its empty WAL comparison as WAL parity. Its unrounded output database is the available evidence; no hidden candidate evidence or unmatched best-score WAL is claimed. `synthetic-decision-results.json` records this availability explicitly.

Reproduce retained analysis with `python3 benchmarks/batch/usearch_scaling/comparison/decision_gate.py`. Synthetic runs refuse existing trial directories to avoid accidentally reusing CrossMaps; use a fresh root with both binaries to repeat. No production changes or commits.
