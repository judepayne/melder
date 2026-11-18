# Greedy output replay, before and after

No changes to scoring, claims, config, or log formats. The output builder checks whether upserts precede terminal outcomes, confirmed pairs are one-to-one (allowing repeat confirmations of the same pair), and no B has conflicting outcomes. In that case the repeated relationship scans cannot remove or break anything. Other logs and authoritative snapshots keep general replay. Metadata still follows event order.

## Frozen replay (isolates the change)

Apple M3 on AC, four Rayon workers, same original warm benchmark logs before and after. Two fresh exports per configuration, alternating binary order. The timer includes CLI startup, both log reads, candidate enrichment, and CSV output, but no matching. Every `relationships.csv`, `unmatched.csv`, and `candidates.csv` was byte-identical for the same source. Only CSV was timed. Source files and pinned binaries retained their hashes.

| Records per side | Baseline samples | Optimized samples | Median before → after | Speedup |
|---:|---:|---:|---:|---:|
| 10k | 1.812, 1.260 s | 1.336, 0.784 s | 1.536 → 1.060 s | 1.45× |
| 100k | 68.398, 65.770 s | 7.488, 7.549 s | 67.084 → 7.519 s | 8.92× |

## Fresh end-to-end batch smoke timings

A separate pass with the same warm USearch inputs and private copies of the caches, an empty starting CrossMap, and scoring logs enabled. One run per cell, not enough to rank small differences. All accepted output pairs agreed with their own saved CrossMap. Greedy may choose different pairs across fresh scoring runs; replaying the same recorded decisions is the equivalence check. Approximate and exact graph modes both solved every component exactly here (largest: 8/76 endpoints for 10k/100k).

| Records per side | Greedy before | Greedy after | Graph (`approximate`) | Graph (`exact`) |
|---:|---:|---:|---:|---:|
| 10k | 2.645 s | 2.064 s | 1.529 s | 1.534 s |
| 100k | 84.797 s | 24.248 s | 25.075 s | 25.386 s |

The 100k end-to-end gap seen in the [earlier integration check](integration-results.md) mostly disappears. This is not a decision to change the default or a comparison of matching quality.

## Reproduce

The baseline is `8a665f9`. Frozen logs came from the `10k-greedy-0` and `100k-greedy-0` directories of the earlier warm CLI run (`/tmp/melder-b2-final.z9pZQC/warm`). Generate equivalent fresh fixtures with the baseline binary and `run_integration.py --modes greedy --repeats 1` if those logs are unavailable. Use `compare_greedy_replay.py --before <baseline release> --after <optimized release> --capture <warm run root> --out <new dir> --repeats 2`. For full wall runs use `run_integration.py --binary <release> --capture <private warm capture> --out <new dir> --repeats 1 --modes greedy` (and `approximate exact` as desired). The local run directory `/tmp/melder-greedy-replay.hnTyOI` retains pinned binaries, logs, detailed hashes and per-run JSON in `frozen/`, `fresh-before/`, and `fresh-after/`.

Focused replay tests, 621 all-features tests, strict all-target Clippy, formatting, and independent code review passed. The fallback and same-log tests exercise duplicate confirmation metadata, review ownership, conflicting outcomes, and stateful events.
