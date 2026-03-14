---
type: benchmark
module: general
status: active
tags: [performance, benchmarks, live-mode, batch-mode]
related_code: [bench/, testdata/configs/]
---

# Benchmarks

Copied from README.md. Last updated: 2026-03-14

---

Benchmarked on Apple Silicon, `all-MiniLM-L6-v2` model, `encoder_pool_size: 4`.

## Batch mode — flat vs usearch, top_n: 20

`flat` scans all candidates linearly; `usearch` uses an HNSW
approximate nearest-neighbour (ANN) graph for O(log N) candidate selection.
Cold builds encode from scratch and save to disk; every subsequent run
is warm. Both backends use `country_code` blocking, which limits each B
record to its country block (~5k A records out of 100k).

**Batch mode benchmarks**

10k x 10k means 10,000 records in the A set. 10,000 in B with zero initial crossmappings.
Apple Silicon M3 Macbook Air

| | flat 10k × 10k | usearch 10k × 10k | flat 100k × 100k | usearch 100k × 100k |
|---|---:|---:|---:|---:|
| Index build time (no cache) | ~16s | ~17s | ~3m | ~3m 21s |
| Index load time (pre-existing cache) | ~25ms | ~50ms | ~650ms | ~600ms |
| Scoring throughput | 4,877 rec/s | **28,937 rec/s** | 363 rec/s | **10,275 rec/s** |
| Wall time (cold) | — | — | — | 3m 34s |
| Wall time (warm) | 1.6s | 0.69s | 4m 37s | **11s** |

**Observations**

- the first build of cached indices for large use cases can be slow. vector encoding is compute intenstive. If this is a problem for you use case, set `quantized: true` in the `performance:` section of config (see example config at top). This speeds up encoding 2x.
- Thereafter, pre-built indices on disk are re-used and startup is fast.
- the 'flat' back end index store is file based and has only O(N) performance. Use only for small experiments.
- 'usearch' an in-process vector database has O(logN) performance (ANN lookup) and should be used for most real world and large use cases. Storage of vectors is slightly slower than 'flat'.

## Live mode: 3,000 add requests, 80% requiring encoding

**pre-populated caches (10k x 10k)**

Apple Silicon M3 Macbook Air
`all-MiniLM-L6-v2`, `encoder_pool_size: 4`
`c=1` means one http client submitting 3,000 requests sequentially
`c=10` means ten http clients each submitting 3,000 add requests simultaneously. 30k total.

| Metric | flat (c=1) | usearch (c=1) | flat (c=10) | usearch (c=10) |
|--------|----------:|-------------:|------------:|---------------:|
| Throughput | 349 req/s | 673 req/s | 616 req/s | 1,624 req/s |
| p50 latency | 2.2ms | 0.7ms | 13.3ms | 3.9ms |
| p95 latency | 4.4ms | 2.9ms | 39.9ms | 21.8ms |

## Live mode: 3,000 add requests, 40% requiring encoding

**pre-populated caches (10k x 10k)**

*as above*

| Metric | flat (c=1) | usearch (c=1) | flat (c=10) | usearch (c=10) |
|--------|----------:|-------------:|------------:|---------------:|
| Throughput | 366 req/s | 382 req/s | 686 req/s | 913 req/s |
| p50 latency | 2.5ms | 2.4ms | 12.6ms | 5.7ms |
| p95 latency | 3.4ms | 3.5ms | 30.8ms | 30.3ms |
| p99 latency | 5.8ms | 5.9ms | 41.2ms | 42.7ms |

Halving the encoding load has little impact; we're already encoding as fast as we can (at 40%).


## Live mode: 3,000 add requests, 80% requiring encoding

**pre-populated caches (100k x 100k)**

*as above*

| Metric | flat (c=1) | usearch (c=1) | flat (c=10) | usearch (c=10) |
|--------|----------:|-------------:|------------:|---------------:|
| Throughput | 160 req/s | 546 req/s | 251 req/s | **1,505 req/s** |
| p50 latency | 5.6ms | 0.9ms | 30.5ms | 5.4ms |
| p95 latency | 8.1ms | 3.5ms | 94.4ms | 16.6ms |

Similar to batch, at larger index sizes, usearch degrades more gracefully than flat.
The usearch c=10 result includes the text-hash skip optimisation: the 20% of
requests that only modify non-embedding fields skip ONNX encoding and complete
in ~1ms (vs ~7ms for encoding requests).

## Benchmarking

Three Python scripts in `bench/` exercise the live server. All three can
start and stop the server automatically, or connect to one you already
have running (`--no-serve`). They require only the Python standard
library -- no pip dependencies.

**`bench/smoke_test.py`** -- Quick sanity check. Starts the server,
waits for health, sends 10 upsert requests (5 A-side, 5 B-side), prints
each response with latency, and stops. Use this to verify the server
comes up cleanly and returns valid responses before running longer tests.

```bash
python bench/smoke_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml
```

**`bench/live_stress_test.py`** -- Sequential throughput and latency.
Fires N requests one at a time with a realistic operation mix: 30% new A
records, 30% new B records, 20% updates that trigger re-encoding, and
20% updates that change non-embedding fields only. Prints a summary
table with p50/p95/p99/max latency per operation type and overall
throughput.

```bash
python bench/live_stress_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml \
    --iterations 3000
```

**`bench/live_concurrent_test.py`** -- Concurrent throughput and
latency. Same operation mix as the stress test but distributes requests
across N parallel workers. Use this to measure how throughput scales
under load.

```bash
python bench/live_concurrent_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml \
    --iterations 3000 --concurrency 10
```

**`bench/live_batch_test.py`** -- Batch endpoint benchmark. Runs the
same workload through single-record endpoints and then through batch
endpoints, printing a side-by-side comparison. Also tests `match-batch`
and `remove-batch`. Use `--batch-only` to skip the single-record
baseline.

```bash
python bench/live_batch_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml \
    --records 3000 --batch-size 50
```

All four scripts accept `--no-serve` to skip starting the server, which
is useful when you want to start it yourself (e.g. with debug logging or
a different config):

```bash
# Terminal 1: start the server manually
meld serve --config testdata/configs/bench_live.yaml --port 8090

# Terminal 2: run the benchmark against it
python bench/live_concurrent_test.py --no-serve --port 8090 --iterations 3000
```

## How to Run Benchmarks

### Run all live benchmarks at once

```bash
bash bench/run_all_live.sh
```

Run from the project root. Requires `./target/release/meld` built with `--features usearch`.

Runs all 12 live scenarios: 10k + 100k datasets, flat + usearch backends, c=1 + c=10 concurrency, 80% + 40% encoding load. Takes ~15 minutes (longer on first run if caches are cold). Resets WAL and crossmap state between each scenario for clean, comparable results.

Config files used: `testdata/configs/bench_live.yaml`, `bench_live_usearch.yaml`, `bench_live_100k.yaml`, `bench_live_100k_flat.yaml`.

Results summary is printed at the end and written to `/tmp/bench_all_results.txt`.

---

### Prerequisites

```bash
# Build release binary with usearch feature
cargo build --release --features usearch

# Ensure 100k datasets exist (generate if needed)
python testdata/generate.py
```

### Batch Mode — 100k x 100k usearch

```bash
./target/release/meld run --config testdata/configs/bench100kx100k.yaml --verbose
```

Uses warm cache from `bench/cache/`. First run will build the index (~3m 24s). Subsequent runs load from cache (~700ms). Key output: scoring throughput (rec/s) and wall time.

Available batch configs:

| Config | Size | Backend |
|--------|------|---------|
| `testdata/configs/bench1kx1k.yaml` | 1k x 1k | flat |
| `testdata/configs/bench10kx10k.yaml` | 10k x 10k | usearch |
| `testdata/configs/bench100kx100k.yaml` | 100k x 100k | usearch |

### Live Mode — 100k x 100k usearch, 10k events

Two-terminal workflow:

```bash
# Terminal 1: start the server
./target/release/meld serve --config testdata/configs/bench_live_100k.yaml --port 8090

# Terminal 2: pump 10k events (10 concurrent workers)
python3 bench/live_concurrent_test.py \
    --no-serve --port 8090 \
    --iterations 10000 --concurrency 10 \
    --a-path testdata/dataset_a_100k.csv \
    --b-path testdata/dataset_b_100k.csv
```

Or as a single command (script manages the server):

```bash
python3 bench/live_concurrent_test.py \
    --config testdata/configs/bench_live_100k.yaml \
    --binary ./target/release/meld \
    --iterations 10000 --concurrency 10 \
    --a-path testdata/dataset_a_100k.csv \
    --b-path testdata/dataset_b_100k.csv
```

Available live configs:

| Config | Size | Backend |
|--------|------|---------|
| `testdata/configs/bench_live.yaml` | 10k x 10k | flat |
| `testdata/configs/bench_live_100k.yaml` | 100k x 100k | usearch |

### Live Benchmark Scripts

| Script | Purpose | Key flags |
|--------|---------|-----------|
| `bench/smoke_test.py` | Quick sanity check (10 upserts) | `--port`, `--binary` |
| `bench/live_stress_test.py` | Sequential throughput & latency | `--iterations`, `--encoding-pct` |
| `bench/live_concurrent_test.py` | Concurrent throughput | `--iterations`, `--concurrency` |
| `bench/live_batch_test.py` | Batch vs single endpoint comparison | `--port`, `--binary` |
| `bench/cpu_monitor.py` | CPU utilisation during runs | (run in parallel) |

Common flags for all scripts: `--no-serve` (use existing server), `--port`, `--binary`, `--a-path`, `--b-path`, `--seed`.

See [[Performance Baselines]] for the current baseline numbers used as the reference for regressions. See [[Key Decisions#Text-Hash Skip Optimization]] for the optimisation that accounts for the 20% encoding skip visible in the 100k results. See [[Use Cases]] for the real-world scenarios these numbers apply to.
