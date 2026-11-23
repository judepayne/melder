# Batch shortlist completeness and cost

Plan: capture the ranked search result before assignment on both backends; publish a run-specific scoring log; stream all-rank edges and their field scores into configured sinks, leaving final relationship selection unchanged. Check every output against raw evidence, then compare the same job before/after with detail both on and off.

**10k A × 10k B, M3 Air, graph assignment, CSV + persistent SQLite output**. Three rotating serialized trials of each condition (24 total), default-feature release, fresh process, directories and scratch state each time. Before = `bfd16b4`. Times are whole process medians (seconds); RSS is whole-process peak, output is the sum of output directory files and persistent database, both MiB. Scoring throughput excludes output and startup. Parquet was correctness-tested, not timed.

| Storage | Logging | Wall before → after | Scoring queries/s before → after | Peak RSS MiB before → after | Output MiB before → after |
|---|---|---:|---:|---:|---:|
| Memory | Off | 10.22 → 12.20 | 58,898 → 58,710 | 592 → 632 | 13.5 → 13.5 |
| Memory | On | 11.34 → 14.17 | 48,401 → 48,928 | 804 → 645 | 134.9 → 155.1 |
| SQLite scratch | Off | 0.734 → 0.734 | 26,392 → 26,237 | 113 → 114 | 7.3 → 7.3 |
| SQLite scratch | On | 0.740 → 2.589 | 26,407 → 25,239 | 114 → 116 | 9.2 → 119.0 |

**Interpretation.** The SQLite on-path is doing substantially more work: the baseline kept about 37k selected/review field rows; the branch wrote **151,688** candidate edges plus **606,752** field rows into both files and SQL. Memory already produced the full field evidence. Its new all-rank representation costs additional writes but its on-path RSS fell. Memory startup initialized an encoder pool in each process even though this recipe doesn't score embedding fields. Subtracting the *reported* encoder initialization time per run before taking the median gives approximate remaining wall times of **0.59 → 0.50 s off** and **2.58 → 3.47 s on**. Initialization is reported to 0.1 s, so these are approximate, not isolated export timings. Don't infer a 19% default scoring regression from the unadjusted memory wall times. Do not extrapolate disk usage from record count alone: costs depend on shortlist width and field count.

**Correctness.** CLI tests compare all four edge columns and seven field columns to the raw shortlist across graph/greedy and both stores. The graph fixture selects rank two, while the losing rank-one edge and its field evidence remain available. Tests also cover DB-only, empty/isolated, reruns, and logging off. Completeness refers to the returned shortlist, not the Cartesian product or just the eligible optimizer graph.

**Reproduce** on macOS with Rust, Python 3 and PyYAML, from the candidate checkout:

```sh
export CARGO_TARGET_DIR=$(mktemp -d /tmp/meld-shortlist-target.XXXXXX)
cargo build --release --locked
export BENCH_CANDIDATE=$(mktemp /tmp/meld-shortlist-bin.XXXXXX)
cp "$CARGO_TARGET_DIR/release/meld" "$BENCH_CANDIDATE"
chmod +x "$BENCH_CANDIDATE"
python3 benchmarks/results/shortlist_benchmark.py
```

[Runner](shortlist_benchmark.py) · [per-run measurements and release/input hashes](shortlist-results.json). The runner archives the baseline commit, uses a private working directory, counts outputs after each process exits, and replaces the measurements JSON. It retains the small configs/logs privately, not the large generated databases.
