# Batch solver library assessment

## Decision context

Jude selected maximum summed eligible edge score, with every A/B record in at
most one match and unmatched records allowed. Existing pipeline `top_n` determines
retained candidate edges. This is NOT maximum cardinality followed by score.

Two Git clones were assessed without changing Melder code/dependencies or upstream
source. Source review and executable validation were separate; initial reviewer
execution restrictions were resolved by worker validation. No implementation or
library adoption has been authorized by these findings.

## Pinned inputs

Temporary workspace: `/tmp/melder-solver-assessment.zZoPlN`.

- `https://github.com/zxqfl/flow`, commit
  `3db79ed889b4221b3ae4713fd701775890ba03a8`: Git manifest mcmf **2.0.0**, not the
  earlier inspected crates.io release 1.1.0.
- `https://github.com/spoj/florecon`, commit
  `f54389d1cb674b245f6dccbdff5fbce8a82070ea`: netsimplex **0.1.0** subcrate.

Rust/Cargo 1.94.0, Apple clang 21.0.0, arm64 macOS. MSRV and other platforms
were not tested. Full report: `/tmp/melder-solver-assessment.zZoPlN/report.md`.
Harness, lockfile, scripts, hashes and raw logs: workspace `validation/`.
These temporary artifacts must be preserved separately for exact reproduction.

## Validation

Both upstream suites passed. Each library passed 1,000 independent small sparse
oracle graphs (4,000 cold solves/library) checking capacity, bijection and summed
objective. Cases include unequal sides, unmatched vertices and random scores on
a 0.01 grid. netsimplex also passed 500 warm cost-update oracle cases.

Named diagnostic cases separately exercised ties, near ties, integer scaling,
empty graphs and a cardinality trap: one match scoring 0.9 versus two scoring
0.8 combined. Both selected the desired one-match result at ordinary scales.

Repeated canonical construction yielded identical pairs in these tests. Shuffled
insertion changed equal-optimal pairings in 4/1,000 netsimplex and 3/1,000 mcmf
cases. Sorting the shuffled inputs restored observed pair reproducibility. This
is not proof of a canonical lexicographic secondary objective.

## Findings

### mcmf: unsuitable as-is

- Builder constructs excessive source/sink supply; C++ bindings discard LEMON
  solver status. An independent C++ probe reproduced `INFEASIBLE` on the actual
  optional-matching formulation. Tiny Rust outputs can nevertheless be correct;
  do not describe all outputs as wrong.
- Costs, capacities and public totals are i32, and the native instantiation uses
  default integer arithmetic despite i64 FFI arrays. At micro-score precision,
  total-cost overflow was reproduced on 10k-per-side graphs.
- Larger scaling reproduced wrong selected paths and invalid intermediate real
  flows hidden by path decomposition. Raising precision is not a safe fix.
- Ordinary quantization can erase close score differences; that is a numerical
  policy issue distinct from overflow and discarded solver status.
- Unconditional C++ build through old gcc crate conflicts with offering this
  feature in Melder's pure-Rust configuration. A feature gate could disable it,
  but would not make the solver pure Rust.
- Wrapper MIT license does not replace bundled LEMON notices; referenced
  `lemon/LICENSE` is absent and redistribution requirements need verification.
- Pinned HEAD dates to 2018. These are clone facts, not a claim about every
  available LEMON binding or upstream solver version.

### netsimplex: closer fit, not adoption-ready

- Pure Rust, sparse arcs and explicit unmatched handling fit the required model.
- Fixed 1e-9 tolerance reproduced a warm near-tie miss: retains 0.9 after adding
  0.9000000005 alternative and reports Optimal. A fresh cold solve chooses the
  better alternative. Thus this demonstrated miss is on warm solving, not a
  demonstrated cold-batch failure.
- Source shows iteration-limit status is lost on an unchanged retry. No test
  reached the cap, so this remains source-confirmed, not runtime-reproduced.
- The general unbounded-status branch was not shown reachable for this bounded
  bipartite model; do not claim it is an established batch defect.
- Scaling requires attention (timings below). Neither status patches nor score
  quantization alone establish adequate performance at 100k.
- Pinned florecon README says archived and points to `spoj/recon`. The successor
  was not assessed. Standalone crate maturity cannot be inferred from repository
  release tags alone.

## Bounded performance measurements

Single release measurement per case, compilation excluded; not a Melder benchmark.
Sparse inputs retain five edges per A; scores in [0.85,1]. Sizes are per side.

| Size | Edges | netsimplex solve | mcmf call |
|---|---:|---:|---:|
| 1k × 1k | 5k sparse | 60.6 ms | 6.27 ms |
| 10k × 10k | 50k sparse | 6.75 s | 170 ms |
| 1k × 1k | 1,999 chain | 12.1 ms | 1.80 ms |
| 10k × 10k | 19,999 chain | 1.19 s | 25.2 ms |

mcmf timing includes internal construction/path decomposition; netsimplex timing
covers solve only. mcmf's reported totals overflow on both 10k cases, although
selected-score sums agree with netsimplex. Chain objectives have an analytic
oracle; large sparse objectives only have cross-solver agreement. 100k was not
run and memory scaling was not established. No timed run hit its 30s timeout.

## Recommendation

Do not adopt either pinned clone unchanged. mcmf needs wrapper, numeric, status,
build and licensing work. netsimplex is a plausible fork candidate but needs
numeric/status contracts and a convincing scaling evaluation, not just integration.

A focused custom sparse assignment solver is a reasonable next option if avoiding
C++ and maintaining predictable behaviour are priorities. It should implement a
standard algorithm, not a new heuristic, and reuse these independent oracle and
determinism tests. Writing our own is not automatically safer or faster.

The successor check below found no replacement optimization solver. No solver
was selected or integrated by this assessment.

## Successor repository check: spoj/recon

Cloned `https://github.com/spoj/recon` to workspace `recon/`, pinned at
`b06663d7b4f107cc961ecc9a7f33848f24347549`. HEAD records 2026-10-04. Read all
implementation and example code and inspected the complete tracked file list.

This is a successor to the financial-reconciliation project, **not a successor
Rust netsimplex solver**. Its six tracked files contain Python rule combinators,
a financial example, overrides, README, MIT license and gitignore. There is no
Cargo manifest, Rust source, network simplex or maximum-weight assignment solver
in this checkout.

`pairs()` greedily pairs equal/opposite amounts in encounter order; other helpers
sequence/filter/group rules and iterate them to a fixed point. They do not accept
weighted graph edges or maximize their sum. Groups can contain more than two
records, unlike Melder's one-to-one assignment contract.

Validation: `python3 -B recon/example.py` passed all embedded assertions. A
four-record test showed input-order sensitivity: amounts +1,+1,-1,-1 produced
(a1,b1),(a2,b2), while swapping a1/a2 in the input produced (a2,b1),(a1,b2).
Both passed its partition-validity check. This is expected for its greedy rules,
not a claimed bug in the project. Clone remained clean.

Conclusion: **not a candidate for Melder's batch optimizer**. No solver benchmark
or further scalability investigation is appropriate for this implementation.
