# Live assignment: recovery and contention follow-up

Branch `experiment/assignment-optimization`, base `b9a83a4`. The follow-up is
limited to the live phase. It does not rerun the existing warm benchmarks or
start batch integration.

## Validation results

- **Memory crash/restart with a warm stale embedding.** The pre-fix release
  restored the record's updated text but scored a probe against the old vector:
  the old-only probe scored 1.0 and the new-only reciprocal proposal did not
  match. After the fix, the HTTP test passed on both flat and USearch. It verifies
  that the update WAL was flushed, the SIGKILL left the old on-disk cache bytes
  unchanged, and the first new proposal after restart uses the recovered text.
  WAL replay now checks vector hashes against final records and repairs the
  missing or stale vectors before serving, without encoding intermediate edits.
- **SQLite process-kill recovery.** The Rust test kills a child after its first
  displacement and before the next proposal, mirrored on both sides. It checks
  the audit prefix, then moves the application's audit WAL aside for the test
  and uses the real warm SQLite loader. The two committed links, unmatched
  record, protected restored partner, lookup and CLI CSV export agree. There is
  no resumed chain. This test uses a test-only pause and does not make a power-loss
  or per-request durability claim.
- **Overlapping HTTP requests.** Passed on both backends and both arrival sides.
  An encoder latch holds the first request inside the mutation gate while a
  second conflicting proposal, bulk upsert, deletion and admin flush start.
  Review ownership, orphans, the accepted pairs and the final flushed CSV agree;
  the SQLite cases also check SQL state. This is targeted overlap, not exhaustive
  scheduling coverage. Competing clients are observed in flight, not all paused
  at internal server barriers.
- **Negative acceptance.** Mirrored Rust tests on both backends reject a forward
  automatic proposal that fails its reverse threshold and prevent fallback to a
  lower automatic candidate when the top score is demoted to review by the gap.

## Multi-step live load

12 fresh-server runs on Apple M3, AC power. Each run has 100 independent,
identical-input chain fixtures (roots alternate side); each measured request
completes exactly two connected displacements and leaves one orphan. Three
repetitions for each backend/client count. All 1,200 requests complete,
with 2,400 displacements and no cap hits. Medians of per-run statistics:

| Backend | Concurrent clients | Requests/s | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Memory | 1 | 2,325 | 0.33 ms | 0.41 ms | 0.50 ms |
| Memory | 10 | 4,809 | 1.92 ms | 2.66 ms | 2.97 ms |
| SQLite | 1 | 808 | 1.11 ms | 1.43 ms | 2.43 ms |
| SQLite | 10 | 1,167 | 8.12 ms | 9.62 ms | 10.79 ms |

Short synthetic runs with two-dimensional vectors and a local test encoder;
these numbers demonstrate the chain path and the throughput/tail-latency
trade-off under this load, not production ONNX latency or degenerate worst-case
performance. They do not supersede the [warm benchmark](official-warm-report.md).

## Reproduce

From the repository root, with the release binary built from the follow-up:

```sh
python3 -B benchmarks/assignment_live/check_embedding_recovery.py --binary target/release/meld --backend flat
python3 -B benchmarks/assignment_live/check_embedding_recovery.py --binary target/release/meld --backend usearch
cargo test --lib assignment_tests
python3 -B benchmarks/assignment_live/check_concurrent_chains.py --binary target/release/meld --mode contention --backend both --output /tmp/contention.json
python3 -B benchmarks/assignment_live/check_concurrent_chains.py --binary target/release/meld --mode performance --backend both --groups 100 --repeats 3 --output /tmp/chains.json
```

The release binary used for these runs is pinned in
`/tmp/melder-live-phase1-validation/meld` (SHA-256
`25c436c31db18363f26feb1b54b83b125492efb3e7a40780f82abf2329130b9c`). That
workspace also holds full HTTP evidence and test/build logs. A new build should
report its own identity instead of claiming that binary is bitwise reproducible.

Default: 528 library + 13 integration + 1 doctest passed. All features: 533
library + 13 integration + 1 doctest passed. Strict all-target Clippy, formatting,
whitespace checks and five Python checks passed. All six specified build variants
passed (debug, release, Parquet, GPU encoder, builtin model, and no-default).
`--no-default-features`:
462 library passed and the same 34 baseline failures remained (verified by exact
failed-test names). They are not treated as passes.

The prior implementation review closure and the decision on the pre-existing
no-default failures remain separate sign-off items. This validation does not
approve batch work or a release.
