# Ordinary retained-graph capture

Captured both requested ordinary **warm usearch** benchmarks using Melder's
actual `score_pool()` path. No production edits, dataset edits, original config
edits, fabricated edges, or solver timings. The original persisted crossmaps are
**protected inputs**, not reset to force a more interesting workload.

## Results

| Per-side records | Protected pairs | Scored B | Candidate rows | Raw score >= .85 | Retained edges |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 5,175 | 4,825 | 96,500 | 455 | **0** |
| 100,000 | 57,960 | 42,040 | 840,800 | 7,312 | **0** |

Every threshold-Auto edge targets an already-protected A endpoint. The empty
graphs are legitimate residual graphs, **not evidence of solver scalability**.
No alternate graph was invented and no protection was removed. Neither original
config enables common-ID or exact prefilter shortcuts; canonical shortcut count
is zero. All original crossmap A/B IDs are excluded from edges (retained as
isolated declared endpoints).

Requested canonical files:

- `/tmp/melder-approx-capture/10k.graph`: `10000 10000\n`, SHA-256
  `96bcc5a4522dbfd3902d3434c756e9436153fb4eed0ca2dc2e70c08a076dcc9c`
- `/tmp/melder-approx-capture/100k.graph`: `100000 100000\n`, SHA-256
  `261369fa24cb11ef3776980c85360401b9cdc7cf51aaa7a27e7acad94d662607`

Identical small graph copies, `summary.json` (full input/cache hashes and
counts), and `provenance.json` are retained in this directory. Full raw evidence
is under `/tmp/melder-approx-capture/`: pinned binary, source snapshots,
original/isolated configs, input crossmap copies, warm cache copies and hashes,
stdout/stderr, per-scale metadata, ordinal maps, raw uncompressed scoring logs,
and independent match logs. Original datasets remain at their recorded paths.
`ordinals.json` maps all lexically sorted original record IDs to zero-based,
side-specific integer ordinals. Edge text would be sorted unique `b a weight`
with `round(score * 1_000_000_000)` using Rust f64 halfway-away semantics, **after**
raw eligibility; integer weights never substitute for threshold comparisons.

## Why this is complete

- `src/batch/engine.rs` calls `score_pool()`, then sends **all returned top_n
  candidates to scoring_log before attempting any CrossMap claims**. Neither
  retrieval nor `score_pool()` removes occupied A endpoints. Accepted-link CSVs
  and occupancy-filtered outputs are not graph sources.
- Both scales use the unchanged model `all-MiniLM-L6-v2`, country blocking,
  embedding/fuzzy/exact weights, `ann_candidates: 20`, `top_n: 20`, Auto `.85`,
  review `.60`, and no minimum gap. All copied warm cache bytes were verified
  unchanged before/after each run and against the originals. Four Rayon threads.
- Every free B has exactly one independent terminal match-log event, every
  protected B is independently accounted for, and every nonempty scored B has
  exactly one scoring entry. Here every free B has exactly 20 candidates.
  ScoringLogSender omits empty results; converter accepts such omissions **only**
  with a terminal `no_match_below` event whose `best_score` is null.
- Scoring schema 1 omits individual candidate classifications. Converter uses
  raw scores and the header thresholds, verifies the top outcome/reason, and
  reconstructs the actual pipeline gap rule: only rank 1 is demoted to Review
  when rank 2 is too close, before top_n truncation. Lower-ranked raw Autos are
  not themselves demoted by `apply_score_gap_check`, but the assignment policy rejects
  the whole downgraded list. None of these captures configures a score gap.
- Ranks are contiguous; descending-score/ascending-ID tie ordering, known IDs,
  per-query uniqueness, endpoint protection, integer range, graph uniqueness,
  and entire B coverage are checked. A candidate's later greedy acceptance or
  rejection has no role in graph eligibility.

Completeness means the **retained configured shortlist**, not all possible
cross-products, all candidate union members beyond top_n, or all true matches.
A preloaded graph is not a fresh empty-crossmap batch. These empty graphs cannot
validate nontrivial solver quality, timing, or approximation behavior.

## Reproduce and replay

From the repo root, using a **new** output directory:

```sh
cargo build --release
python3 -B benchmarks/assignment_solver/capture.py --output /tmp/melder-approx-capture-new
python3 -B -m unittest discover -s benchmarks/assignment_solver/capture -v
```

The campaign used `cargo build --release` (default features). The initially
available release binary lacked usearch: its failed attempt is preserved at
`/tmp/melder-approx-capture-no-usearch-attempt`; it did not produce graph results.
The successful run used the rebuilt/pinned binary. `provenance.json` records
binary/source hashes and dirty worktree state, not a clean reproducible-build
attestation. The capture script was subsequently hardened with additional
rank/order assertions and focused tests; the provenance capture-script hash is
that of the execution-time version. Real-artifact replay with the final script
passed for both scales with identical graph hashes and all candidate counts.

Tests include positive-half Rust-compatible quantization and neighboring floats,
raw threshold handling, rank-1 gap demotion, protection filtering, single-result
gap handling, and full real-artifact re-conversion/coverage reconciliation.
The real replay test skips explicitly if `/tmp` evidence is unavailable; it does
not pretend that the durable graph files alone establish raw-log completeness.
