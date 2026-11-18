# Batch assignment solver benchmarks

The standalone benchmark adapter compiles the same solver as `meld run`, from
[`src/matching/assignment/`](../../src/matching/assignment). It accepts frozen
weighted graphs without loading records, running retrieval, or publishing pairs.
The production integration is opt-in; see the [batch guide](../../docs/batch-mode.md#choosing-pairs-with-the-graph-optimizer)
and [end-to-end CLI checks](integration-results.md). Solver-only evidence:
[baseline](results.md), [parallelism and batching](optimization-results.md),
[deterministic approximation](approx-results.md), and the [cutoff sweep](cutoff-results.md).

## Interface and guarantees

Pass `solve(n_b, n_a, &[WeightedEdge], Limits)` side-specific integer IDs in
canonical order (e.g. sorted record IDs), integer weights, and opaque payload
handles to recover the caller's metadata. The input edge slice need not be sorted.
The solver rejects duplicate pairs, invalid endpoints/weights, integer overflow,
limit or allocation errors, and detected residual invariant failures. Limits
bound input edges and active endpoints; declared isolated endpoints require no
per-record allocation. Use a process timeout in the prototype benchmark. A failed
or interrupted run publishes no successful partial matching.

Quantize **after** raw automatic-match/ambiguity/protection rules:
`round(score * 1_000_000_000)` using Rust `f64` semantics. Reject non-finite and
out-of-range scores. The objective is the exact sum of **represented** weights,
not the raw floating-point sum or the number of links. Unmatched is allowed.
The same retained integer graph yields the same canonical pairs regardless of
edge order or payload values. This does not promise a particular lexicographic
optimum, stable candidate retrieval, or identity accuracy.

## Why it is exact

Each connected component is independent. Build a unit-capacity residual network
`source -> B -> A -> sink`, with cost `-weight` on B/A edges. Zero flow is the
minimum-cost flow of cardinality zero. Initially set A potentials to minus their
largest incident weight, the sink to the minimum A potential, and the others to
zero. These potentials make all usable reduced costs nonnegative.

Dijkstra takes the cheapest residual path, breaking equal distances by canonical
node order and keeping the first predecessor. The potential update adds the
shortest distance capped at the sink distance (also for unreachable vertices).
This keeps all usable reduced costs nonnegative. The chosen path has zero new
reduced cost, so activating its reverse arcs maintains this invariant. Augmenting
produces the minimum-cost flow of the next cardinality; reverse arcs undo earlier
choices, and feasible potentials rule out improving same-cardinality cycles.
Stop if no path exists or the cheapest path's original (not reduced) cost is
nonnegative. Minimum incremental costs are nondecreasing, so a larger matching
cannot improve the objective at that point. In particular a zero-gain extension
is not selected. Check selected pair uniqueness and sum against the flow result.

Network storage is O(V+E); deterministic graph construction uses sorting and
compact indices. Per component, shortest paths take O(K (E+V) log V) time for K
augmentations, plus the final search. Allocating sparsely does not eliminate the
repeated shortest-path cost on a giant component.

## Experimental speedups

`solve_with_options` can distribute independent components over a bounded number
of workers or batch equal-gain paths within one component. The default remains
one worker, one path per search. Workers solve the same local networks and the
merge sorts by canonical `(B, A)`; scheduling and worker count never change the
selected pairs. This is separate from enabling batching, which may deliberately
select a different tied optimum but is deterministic for the same graph.

After a strictly improving shortest path, all zero-reduced-cost residual paths
have the same original cost by telescoping the potentials. Reversing their arcs
preserves feasible potentials and each additional path is still shortest. The
batch uses breadth-first levels over tight arcs and walks forward through levels
in canonical adjacency order (no recursion or cycles). Once there are no more
tight paths it returns to Dijkstra. A nonnegative original cost cannot start a
batch. This keeps the optimum and nine-decimal precision unchanged.

## Reproduce

```sh
M=benchmarks/assignment_solver/prototype/Cargo.toml
cargo test --manifest-path "$M"
cargo test --release --manifest-path "$M"
cargo clippy --manifest-path "$M" --all-targets -- -D warnings
python3 -B benchmarks/assignment_solver/run.py --timeout 25 --budget 220 --include-chain --output /tmp/b1-scale
python3 -B benchmarks/assignment_solver/run.py --matrix --workers 4 --timeout 25 --budget 480 --include-chain --output /tmp/b1-speedups
```

The benchmark output directory must be new. It holds the pinned binary, per-run
JSON, child stdout/stderr, and source hashes. Seeds are fixed and graph checksums
cover raw reconstructed scores and quantized weights. A timeout is reported as
such and makes the runner exit nonzero. These are single-process synthetic
measures, not end-to-end Melder throughput.
