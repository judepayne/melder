# Phase 2B (Phase 3): batch graph assignment — agreed user contract

Handoff for `meld run` integration. This is the agreed interface, **not a claim that the current binary supports it**. It supersedes the earlier exact-only `maximum_weight` configuration sketch. Live and enroll are unchanged.

## YAML and defaults

```yaml
batch:
  assignment:
    strategy: graph
    mode: approximate   # optional; the default only when strategy is graph
```

| YAML choice | Effective behavior |
|---|---|
| No batch assignment settings, or `strategy: greedy` without a mode | Existing greedy behavior; do not collect an optimizer graph. |
| `strategy: graph` without a mode, or with `mode: approximate` | Exact for small groups, deterministic fast matching for large groups. |
| `strategy: graph` with `mode: exact` | Exact for every group; no fallback to a heuristic. |
| `mode` supplied with omitted strategy or `strategy: greedy` | Reject with an actionable validation error: `batch.assignment.mode requires strategy: graph`. |

Use typed enums and reject unknown values. Do not expose a CLI override, cutoff, worker-count knob, or promised score-loss tolerance. Keep omission distinguishable from an explicitly supplied mode when validating.

## Suggested batch guide copy: Choosing pairs with the graph optimizer

The optimizer changes **which pairs are accepted**, not how Melder scores them. Suppose B1 scores 0.99 with A1 and 0.98 with A2; B2 scores 0.97 with A1 and 0.85 with A2. Taking the strongest pair first gives 1.84 in total. Choosing B1–A2 and B2–A1 gives 1.95. Each record still has at most one partner, and records may remain unmatched.

Melder keeps protected links fixed and considers only automatically eligible candidates on the existing shortlists. It does not widen `top_n`, reconsider excluded pairs, or turn a list demoted for ambiguity into an automatic match.

**Start with approximate.** Melder divides the candidates into independent groups: records are in the same group if you can follow a chain of candidate pairs between them. No pair crosses a group boundary. Group size counts **distinct participating records on both sides together**, not edges or records in the whole batch. Through **512 inclusive**, solve exactly. At **513 and above**, try two deterministic answers: take the highest-scoring available pairs first, and separately find a matching with the most pairs. Choose the one with the higher total score for that group; on a tie keep score-first. A million-record batch can therefore finish entirely exact.

**Choose exact if the highest possible total similarity score matters more than runtime.** The guarantee is over the eligible retained pairs, with scores rounded to nine decimal places after eligibility checks. It is not a guarantee that the shortlist contains every possible match, or that a chosen pair is the right real-world identity. Failure to finish is an error, not permission to publish a partial result as optimal.

**Repeatability:** the same Melder version, mode, retained pairs and represented weights produce the same assignments, regardless of worker scheduling or edge order. Different retrieval results can change the graph. Hardware speed or elapsed time must not decide which method runs.

## Show what actually happened

Illustrative completion summaries (numbers are examples, not new measurements):

```text
Batch assignment: graph (approximate)
Groups: 51,695 exact · 0 approximate · largest: 76 records
Result: optimal over all retained eligible pairs.
```

```text
Batch assignment: graph (approximate)
Groups: 38 exact · 2 approximate · largest: 1,420 records
Total score: at least 98.7% of the best possible over retained eligible pairs.
```

For the bound, sum exact objectives for exact groups. For each fast group use the smaller of (sum of best scores by A) and (sum of best scores by B). Divide the achieved total by the combined upper bound. Round a displayed lower bound **down**, never up. A loose bound is not measured loss or identity accuracy. If the upper bound is zero, report no positive eligible score rather than dividing by zero. Emit the completion summary only after successful publication.

## Where this goes when shipped

- `docs/configuration.md` → batch settings: YAML, defaults table, invalid mode-without-graph example; link to the batch guide.
- `docs/batch-mode.md` → “Choosing pairs with the graph optimizer”: worked example, 512/513 rule, exactness, determinism, and reading the summary. Contrast with live proposals; this is not a new CLI subcommand.
- `docs/performance.md` → batch optimizer: component size vs batch size, solver-only vs end-to-end timings, and synthetic vs captured workloads. A small absolute saving need not justify approximation.
- `AGENTS.md` → point to the shipped implementation, test gates and docs. Keep a single shared scoring pipeline and a single solver implementation.

## Resume after compaction

Phase 1 is committed at `08d0ae5`; Phase 2A is the standalone prototype under `benchmarks/assignment_solver/` on `experiment/assignment-optimization`. Proceed with Phase 2B under the active plan. The 100k connected exact cases timed out; the accepted direction is an explicit choice of guarantees, **not** a claim that exact met a 100k latency target. The prototype's earlier 256 cutoff is historical; integration and boundary tests must use **512/513**. First integration checks: default/invalid YAML, identical greedy-off behavior, exact vs approximate dispatch and summary at the boundary. Then complete graph collection, atomic publication, export and release checks in the plan. No merge or push implied.

Evidence: [approximation on captured and synthetic graphs](approx-results.md), [cutoff study](cutoff-results.md), [exact scaling](results.md).
