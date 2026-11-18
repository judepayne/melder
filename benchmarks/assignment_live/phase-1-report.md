# Phase 1: live assignment validation — 2026-11-18

Branch: `experiment/assignment-optimization`, based on `b9a83a4`. The new live
strategy is opt-in; batch remains greedy. No commit or merge. The full-shortlist
trial was frozen before edits in `melder-assignment-before-dexcyiz_` under the
macOS temporary directory (418 hashed files, trial patch, and baseline identity).
The trial source, rather than its branch pointer alone, is the before-state.

## Checks

| Check | Result |
| --- | --- |
| `cargo test` | 524 library, 13 integration, 1 doctest passed |
| `cargo test --all-features` | 529 library, 13 integration, 1 doctest passed |
| `cargo test --no-default-features` | 458 library passed; 34 failed, all also failing in the frozen trial (46 failures there); no new failures |
| `cargo clippy --all-features --all-targets -- -D warnings` | Passed |
| `cargo fmt -- --check`, `git diff --check` | Passed |
| `python3 -B -m unittest discover -s benchmarks/assignment_live -p 'test_*.py'` | 5 passed |

The full release feature build matrix passed before closure fixes; a fresh
default release build passed after those fixes. Post-fix all-features and
no-default suites were then compiled and executed as above. The no-default
failures are not recorded as passes or silently waived. See the full log files
under `/tmp/melder-assignment-*.log` while available.

## Live HTTP fixture

Same release binary with fresh memory-backed servers, ordered setup, and 100
measured competitive arrivals on each side. Greedy retained all 200 incumbents;
the opt-in strategy displaced all 200, with no cap hits or errors and 200
fixture-preferred final pairs. Both modes had 200 total final pairs. Every
contest has a single slot, so an orphan is expected on displacement.

| Strategy | p50 | p95 | p99 | Requests/s |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.149 ms | 0.169 ms | 3.305 ms | 4,806 |
| Deferred acceptance | 0.185 ms | 0.211 ms | 0.222 ms | 5,336 |

One sequential synthetic run per strategy, with visible noise in the greedy
latency tail. These numbers show the displacement path was exercised, not a
reliable throughput comparison or a production performance claim. Run with
`run.py --binary PATH --groups 100 --keep-workdir --output FILE`. Deterministic
fixture construction uses no random seed. Report and binary under
`/tmp/melder-assignment-live-evidence/` (`http-100-fixed.json`, `meld-fixed`).
Binary SHA-256: `cde84c2e22e3d22f29b879937e16d3eecaa49bcc7e46c3b25eb41365b03fafc2`.
Source hash manifest SHA-256: `973f799b18372cffada95df5e4977533d9dd9dd45504cb1e65cdbed060384365`.

## Review fixes and gates

- The crash/quiet-shutdown/restart regression reproduces loss of all six
  WAL-only records on the earlier implementation. The fixed release binary
  retains the six records, both final pairs, and all four audit steps. Reproduce
  with `check_recovery.py --binary PATH`. Older WAL segments are retained rather
  than deleted after compaction of only the current segment.
- Fault-injection tests verify enabled bulk removal stops immediately after a
  WAL append failure while rematching an orphan, for both sides and backends.
- WAL tests cover truncated ordinary and assignment events, including partial
  UTF-8 at EOF, and fail-closed corrupt complete frames and unknown schemas.
- Initial Partner Review required the above changes. They were implemented and
  tested. The required same-reviewer closure could not run because its saved
  session was unavailable; the initial verdict has not been superseded.
- Batch prototype has not started. The checked-in plan's go/no-go sequence
  still requires resolving review closure and disposition of the pre-existing
  no-default suite failures before advancing.
