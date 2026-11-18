# Batch graph CLI integration tests

Run `cargo test --test batch_graph_cli` (also run with `--no-default-features` and `--all-features`). The memory CLI starts the existing offline Python stub encoder; there are no embedding fields or model downloads in the fixture. SQLite does not initialize an encoder.

The BM25/fuzzy fixture has a known optimum of 1.775 vs 1.325 for B1-first greedy. Greedy can also reach 1.775, so tests compare the objective it actually returned and use a single-query fixture to test omission vs explicit greedy. Distinct A and B ID fields detect swapped columns. Both backends check CrossMap membership against CSV, SQLite output, and Parquet when enabled. The fixtures also exercise protection, exclusions, whole-list ambiguity, malformed config, input-order and worker-count changes, and 512/513 active-endpoint dispatch including explicit exact mode.

The memory path re-exports a completed timestamped graph log and rejects the same log without its completion marker. The SQLite ingestion file is temporary and that path does not write a match log. See [`assignment_fault_tests.rs`](../src/batch/assignment_fault_tests.rs) for the failure matrix covering both CrossMap backends, and [the integration evidence](../benchmarks/assignment_solver/integration-results.md) for CLI checks with warm USearch caches and real model vectors.
