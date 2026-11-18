# Live HTTP displacement workload

Python 3.9+ standard library only. Uses an EXISTING binary; never builds Rust.
From the repository root:

```sh
python3 benchmarks/assignment_live/run.py --binary ./target/release/meld --groups 100 --keep-workdir --output /tmp/assignment-live.json
python3 -m unittest discover -s benchmarks/assignment_live -p 'test_*.py' -v
python3 benchmarks/assignment_live/check_recovery.py --binary ./target/release/meld
```

The recovery check populates a memory-backed server through HTTP, flushes its
WAL, kills that test process, then verifies a quiet restart and clean shutdown
have not lost any WAL-only record, final pair, or assignment audit step.

The binary must support `live.assignment.strategy: deferred_acceptance` and
return `assignment` summaries on `/api/v1/{a,b}/add`. Both strategies run
sequentially using the exact same binary and workload, each with a fresh server,
loopback ephemeral port, and isolated temporary datasets/config/cache/crossmap/
WAL/output paths. `greedy` is the strategy-off comparison. No existing service
or user dataset is modified. A small port-selection/bind race remains; startup
errors fail the run instead of retrying requests. Temp paths are removed unless
`--keep-workdir` is passed. Use it to retain server logs on failure.

## Competition, not insertion-only traffic

Each blocking group has one opposite-side receiver and two same-side claimants:

- All three share text and a numeric `stable` value.
- Receiver and later arrival share a second numeric `competitive` value.
- Earlier incumbent has a different competitive value.

Real scoring is BM25 (0.1), numeric equality (0.6), numeric equality (0.3).
The incumbent scores at least 0.6, above the 0.55 automatic threshold; the
arrival scores at least 0.9 and is strictly better even if BM25 normalization
changes. Identical within-group text ensures lexical candidate retrieval. Groups
are isolated by blocking. Numeric values are actual equality comparisons, not
numeric-distance or mocked preference lists. There are no embeddings, common
IDs, exact prefilters, manual confirms, exclusions or score hooks.

Current live startup initializes an encoder even without embedding fields.
`unused_encoder.py` supplies ONLY its mandatory protocol handshake. It never
supplies vectors and rejects any encode request; such a call fails the run.
No model or download is needed. Matching still uses the real engine pipeline.

The ordered, unmeasured setup adds receiver then incumbent for every group,
then verifies the complete crossmap equals the incumbent pairs. Only then does
measurement start. Each measured request is a new, better claimant competing
with an automatic incumbent, NOT an update or a free receiver. The fixture is
mirrored: `--groups N` creates N A-arrival and N B-arrival groups. Each receiver
has only one slot, so one losing claimant per group is a legitimate orphan.
This is a bounded single-displacement microbenchmark, not a long-cascade,
concurrent-load, model-inference, or global optimality benchmark.

## Measurements and validity

The JSON report contains overall and per-arrival-side wall HTTP latency
p50/p95/p99 (nearest rank), sequential request throughput, elapsed wall time,
HTTP successes/errors, assignment changes, observed displacements, cap hits
(`limit_reached`), response orphan events, final unmatched totals, coverage
stats, preferred pairs and retained incumbents. HTTP timing includes JSON
serialization, connection setup, server work and response parsing. Throughput
counts completed attempts including errors; consult success/error counts.
Startup, ordered setup, audits and shutdown are excluded. Orphan events are
not deduplicated; final unmatched totals are the authoritative snapshot.

Enabled must observe `assignment.displacements > 0` on BOTH arrival sides,
finish with every preferred pair, and preserve bijection. Greedy must retain
all incumbent pairs. Both must finish with exactly N unmatched records on each
side. Missing assignment summaries, setup mismatches, request errors, unexpected
encoder calls and incorrect outcomes produce a nonzero exit and `passed: false`.
Requests are not retried (avoids hiding duplicate/update traffic).

`--cap 32` is the default displacement budget. Each gadget needs only one
successful displacement; cap hits should be zero. `--cap 0` is a useful negative
control: enabled should report zero displacements and FAIL the positive workload
contract, not silently claim to benchmark displacement. Use larger `--groups`
for steadier latency samples, and repeat runs with external order reversal if
comparing small timing differences; greedy always runs first here.
