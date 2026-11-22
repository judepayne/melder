# Exact optimization: retained semantic parity investigation

## Scope and reproduction

Read-only analysis of `/tmp/melder-opt.o6KzoE/exact-e2e`; 18 retained trials,
three baseline and three candidate per size, 10,000 fixed B queries. No meld
invocations, builds, timing analysis, input writes or commits. Existing Rust edits
were inspected, not changed. Despite “exact-only optimization”, **the workload is
mixed**: embedding 0.30+0.20, BM25 0.45, synonym 0.20, exact LEI 0.05;
USearch, country blocking, ANN 50, top_n 5, BM25 default 10; review floor 0.60,
auto-match 0.88, default graph/approximate assignment.

From repository root (Python stdlib only):

```sh
python3 benchmarks/batch/usearch_scaling/comparison/parity.py \
  /tmp/melder-opt.o6KzoE/exact-e2e \
  --out benchmarks/batch/usearch_scaling/comparison/parity-results.json
```

The script compares all 15 unordered trial pairs per size: three baseline-self,
three candidate-self, nine cross-arm (including all three originally paired
trials). It preserves CSV cell strings, asserts unique relationship/query keys,
checks CSV decision keys against unrounded WAL events and accepted pairs against
CrossMap, and reports identity sets, common-row changes, ranks, reasons, numeric
score deltas, query category transitions and examples. JSON contains all 45
comparisons, not only averages. Timestamp/record-upsert events are not compared
as decisions. WAL records only final decisions, **not the complete retrieved or
scored candidate universe**. Exported ranks are therefore not hidden shortlist
ranks. Input/binary provenance remains in the retained `metadata.json` (baseline
SHA starts `c1dd9dd7`, candidate `1de3a9cd`).

## Decisions and sets

| A size | matches, every trial | reviews | unmatched |
|---:|---:|---:|---:|
| 10k | 56 | 139 | 9,805 |
| 100k | 611 | 276 or 277 | 9,113 or 9,112 |
| 1M | 6,082 | 2,523 | 1,395 |

**All accepted pair sets and CrossMaps agree exactly within each size.** All
common relationship CSV rows agree in every cell, including score, rank, reason
and source attributes. All common match/review WAL scores also agree **exactly**
(no floating-point tolerance needed); no common WAL rank/reason changes.

There is one real category/edge difference at 100k: `CP-205102` is either
unmatched at unrounded best score `0.48580118715763093`, or review of
`ENT-467748`, rank 1, score `0.6283212458237539` (`0.6283` in CSV).
The review is present only in `1-2-baseline` and `2-2-candidate`; absent in the
other four. Thus each arm independently reproduces both outcomes, with the
same 1/3 review frequency. Review-edge and unmatched-query symmetric differences
are each 0 or 1; query-category changes are 0 or 1. This is **not rounding noise**:
it crosses the review threshold. Other sizes have identical decision sets.

## Scores: variability versus arm differences

Ranges below are over all relevant pairs. “CSV” counts changed **common unmatched
rows**; the only changed cell is `best_score`. `best_a_id` is blank and unchanged,
so retained unmatched exports cannot identify which candidate supplied the best
score. WAL counts common decision entries with any changed value; all such
changes are unmatched best scores. Added/removed decisions above are separate.

| A | comparison | CSV changed | WAL changed | largest absolute WAL delta, across pairs |
|---:|---|---:|---:|---:|
| 10k | baseline/self | 88–95 | 88–95 | 0.0578097384 |
| | candidate/self | 82–95 | 83–96 | 0.0779789327 |
| | cross-arm | 79–98 | 79–98 | 0.0779789327 |
| 100k | baseline/self | 193–205 | 195–207 | 0.0780260918 |
| | candidate/self | 187–199 | 188–200 | 0.0780260918 |
| | cross-arm | 189–214 | 190–214 | 0.0780260918 |
| 1M | baseline/self | 7–10 | 7–10 | 0.0145511136 |
| | candidate/self | 5–11 | 5–11 | 0.0309418857 |
| | cross-arm | 5–12 | 6–12 | 0.0309418857 |

Unrounded deltas exceed CSV precision by orders of magnitude. E.g. 10k
`CP-348991` is `0.37190217578747264` versus `0.4498811084891157` in
baseline/candidate trial 1; baseline itself changes `CP-188360` from
`0.4133998881350768` to `0.3555901497602463` in trials 1/2. Cross-arm
variability substantially overlaps self variability; three replicates do not
establish a statistical equivalence bound or exclude an additional small
candidate effect. In particular, the 1M candidate-self envelope is wider than
baseline-self, so a baseline-only maximum is not a defensible acceptance cutoff.

## Code explanation and limits

The exact scorer change (`src/scoring/exact.rs::score`) is pointwise equivalent:
empty/nonempty was already unequal after lowercasing; both empty still returns
zero; unequal ASCII cannot become equal under ASCII lowercase after the existing
case-insensitive comparison. Non-ASCII still uses the same Unicode lowercase
fallback, including mixed ASCII/Unicode values. Nothing in this change directly
explains large variable BM25/embedding composite scores.

A concrete **pre-existing nondeterminism mechanism** is visible in
`src/bm25/simple.rs` (also present in HEAD, not just the dirty optimization):

- `partial_top_k` (~1004) partitions/truncates and sorts by score only. Equal
  scores at the cutoff have no stable record-ID tie-break.
- WAND (~690–770), `heap_insert` and `OrdScore` (~961–1001) use traversal/compact
  IDs; equal-to-threshold candidates do not replace incumbents. Compact IDs are
  assigned during index construction, not derived from stable external IDs.
- Query term frequencies and analytical self-score iterate randomized HashMaps
  (~550 and ~795), adding a smaller floating-point ordering effect.
- `pipeline::bm25_candidate_stage` retains only top-K normalized BM25 scores.
  `src/scoring/mod.rs` gives **zero BM25 contribution to candidates absent from
  that map**, even if ANN/synonym retrieval includes the candidate. Therefore a
  retrieval cutoff/tie difference can change an existing candidate's composite
  score substantially, not merely permute equal final scores. Normalized BM25
  has weight 0.45 here; the observed 0.14252 review transition is plausible under
  this mechanism. Stable final sorting cannot repair an earlier truncated set.

This explains how unchanged code can produce the observed class of differences;
self-run differences prove nondeterminism independent of the exact optimization.
It does **not prove that a particular retained query hit a BM25 cutoff tie**:
these logs lack per-source candidates, component scores and boundary ties.
USearch retrieval or other candidate-stage effects cannot be excluded from the
retained evidence. Do not attribute the 100k flip conclusively to ANN, BM25,
roundoff, or exact scoring without that trace. CSV mismatch is real; assigning
its entire cause to this optimization is unsupported.

## Recommended robust gate

1. **Separate deterministic optimization parity from stochastic end-to-end
   characterization.** Freeze/replay identical borrowed query/pool and candidate
   inputs through the one production pipeline; compare exact outputs against
   the old scorer on every observed LEI pair plus empty, whitespace, ASCII,
   Unicode/mixed-script cases. Require exact scores, decisions and ranks for
   this exact optimization, not a large numerical tolerance.
2. End-to-end: require identical accepted pairs, review edges, unmatched IDs,
   reasons and ranks under controlled retrieval; compare unrounded WAL values,
   not only four-decimal CSV hashes. Any category transition remains a failed
   strict gate, even if a baseline run also exhibits it. Current artifacts pass
   accepted-pair parity but **do not pass strict complete decision/score parity**.
3. Diagnose/fix or freeze retrieval separately: stable external-ID tie-breaking
   at every top-K boundary, deterministic construction/term accumulation, and
   capture source candidate IDs, BM25 scores and boundary ties for changed
   queries. Use the same persisted ANN index; cloned caches alone do not freeze
   rebuilt BM25 tie order. Do not introduce a second production scoring path.
4. Until deterministic replay exists, retain all self/cross-arm comparisons and
   classify every mismatch. A stochastic benchmark can show “no observed
   arm-specific regression”, not certify parity. Do not waive threshold flips,
   strip scores/ranks from hashes, or choose a global 0.08 tolerance simply
   because these trials observed it. Three repeats are inadequate for a robust
   stochastic non-inferiority bound.

Validation: analysis script ran successfully twice against all 18 retained
trials, including CSV/WAL/CrossMap assertions. No engine execution was used.
