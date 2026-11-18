# Phase 1: final review fixes and regression results

All five findings from the final review have targeted regressions. No batch work or benchmark timing claims are part of this packet.

- A delta-free segment that deletes and recreates an ID no longer loses its deletion during compaction. `check_segment_recreate.py` checks offline export before a third server start and checks the restarted server, with both sides mirrored.
- CrossMap take operations distinguish a missing pair from an SQLite error. Trigger-refused updates and removals (single and bulk, either side) preserve the record and pair and report an error.
- Changing or clearing a common ID retires its previous index entry only if this record still owns it. Canonical confirmation checks the current opposite record; duplicate IDs cannot manufacture a canonical match from stale index entries.
- Direct SQLite export reads one snapshot, derives unmatched records from the absence of a crossmap partner, and excludes resolved review owners. Tests seed stale metadata and export before any server restart.
- Deletion clears reviews using side and ID together, in the session, persisted store, and replay. A namesake on the other side keeps its unrelated reviews.

## Rerun

`cargo test`: 537 library + 13 integration + 1 doctest passed. `cargo test --all-features`: 542 + 13 + 1 passed. `cargo test --no-default-features`: 471 library passed, 34 failed; all failed names exactly matched the pre-fix log. No waiver is assumed. All six build variants passed, including no-default release. Formatting, strict all-feature/all-target Clippy, whitespace checks and the five live workload Python unit tests passed.

Release HTTP checks: `check_segment_recreate.py`, `check_recovery.py`, `check_embedding_recovery.py` (flat and USearch), and `check_concurrent_chains.py --mode contention --backend both` passed. Use `--binary /path/to/meld` on each script. The last script tested four backend/arrival-side combinations. Existing performance studies were not rerun.

Local test logs, contention output, and the pinned default release binary are in `/tmp/melder-final-fixes.ucrHQU/` (SHA-256 `99d0ff21b25154c680f3c77a30a942a70e9b19e08f4181f73a3e6921da72601d`).
