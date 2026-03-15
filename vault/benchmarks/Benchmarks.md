---
type: benchmark
module: general
status: active
tags: [performance, benchmarks, live-mode, batch-mode]
related_code: [benchmarks/]
---

# Benchmarks

Copied from README.md. Last updated: 2026-03-14

---

Benchmarked on Apple Silicon M3 MacBook Air, `all-MiniLM-L6-v2` model, `encoder_pool_size: 4`.

## Batch mode — flat vs usearch, top_n: 20

`flat` scans all candidates linearly; `usearch` uses an HNSW
approximate nearest-neighbour (ANN) graph for O(log N) candidate selection.
Cold builds encode from scratch and save to disk; every subsequent run
is warm. Both backends use `country_code` blocking.

10k x 10k means 10,000 records in the A set, 10,000 in B with zero initial crossmappings.

| | flat 10k × 10k | usearch 10k × 10k | flat 100k × 100k | usearch 100k × 100k |
|---|---:|---:|---:|---:|
| Index build time (no cache) | ~17s | ~17s | ~3m | ~3m 32s |
| Index load time (pre-existing cache) | ~47ms | ~78ms | ~650ms | ~640ms |
| Scoring throughput | 5,507 rec/s | **31,366 rec/s** | — | **8,735 rec/s** |
| Wall time (cold) | — | — | — | 3m 32s |
| Wall time (warm) | 2.2s | 0.7s | — | **12.8s** |

**Observations**

- The first build of cached indices for large datasets can be slow — vector encoding is compute-intensive. Set `quantized: true` in the `performance:` section to speed up encoding ~2x.
- Pre-built indices on disk are reused on subsequent runs and load in under 1 second.
- The `flat` backend has O(N) search. Use only for small experiments and development.
- The `usearch` backend has O(log N) HNSW search. Use for all real-world workloads.

## Live mode: 3,000 add requests (10k × 10k), 80% requiring encoding, c=10

Apple Silicon M3 MacBook Air. `all-MiniLM-L6-v2`, `encoder_pool_size: 4`.
`c=10` means ten HTTP clients each submitting requests simultaneously.

| Metric | flat (c=10) cold | usearch (c=10) cold | flat (c=10) warm | usearch (c=10) warm |
|--------|----------------:|--------------------:|-----------------:|--------------------:|
| Throughput | 843 req/s | 1,045 req/s | 1,113 req/s | **1,558 req/s** |
| p50 latency | 8.4ms | 5.5ms | 7.2ms | 3.5ms |
| p95 latency | 30.4ms | 29.0ms | 21.2ms | 25.6ms |

Cold = fresh index build (~18s startup). Warm = pre-built cache loaded from disk (~1.7s startup).

## Live mode: 10,000 add requests (100k × 100k usearch), 80% requiring encoding, c=10

*as above, encoder_pool_size: 6*

| Metric | cold | warm |
|--------|-----:|-----:|
| Throughput | 925 req/s | **1,325 req/s** |
| p50 latency | 8.6ms | 6.0ms |
| p95 latency | 25.0ms | 19.0ms |

Cold startup builds index from scratch (~3m 29s). Warm startup loads cache incrementally (~7s).
The 20% of requests that only modify non-embedding fields skip ONNX encoding and complete in ~1ms.

## How to Run Benchmarks

### Prerequisites

```bash
# Build release binary with usearch feature
cargo build --release --features usearch

# Ensure 100k datasets exist (generate if needed)
python3 benchmarks/data/generate.py
```

### Benchmark structure

Each benchmark is self-contained under `benchmarks/batch/` or `benchmarks/live/`:

```
benchmarks/
  data/               datasets (10k and 100k CSV + Parquet)
  scripts/            shared test scripts
  batch/
    10kx10k_flat/cold/    run_test.py + config.yaml + cache/ + output/
    10kx10k_flat/warm/
    10kx10k_usearch/cold/
    10kx10k_usearch/warm/
    100kx100k_usearch/cold/
    100kx100k_usearch/warm/
  live/
    10kx10k_flat/cold/    run_test.py + config.yaml + cache/ + output/ + wal/
    10kx10k_flat/warm/
    10kx10k_usearch/cold/
    10kx10k_usearch/warm/
    100kx100k_inject10k_usearch/cold/
    100kx100k_inject10k_usearch/warm/
```

Run any test directly:

```bash
python3 benchmarks/batch/10kx10k_usearch/cold/run_test.py
python3 benchmarks/live/10kx10k_inject3k_usearch/warm/run_test.py
```

Warm tests need to be run **twice** — the first run builds the cache, the second is the true warm measurement.

### Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `benchmarks/scripts/smoke_test.py` | Quick sanity check (10 upserts) |
| `benchmarks/scripts/live_stress_test.py` | Sequential throughput & latency |
| `benchmarks/scripts/live_concurrent_test.py` | Concurrent throughput |
| `benchmarks/scripts/live_batch_test.py` | Batch vs single endpoint comparison |
| `benchmarks/scripts/cpu_monitor.py` | CPU utilisation during runs |

See [[Performance Baselines]] for the current baseline numbers used as the reference for regressions. See [[Key Decisions#Text-Hash Skip Optimization]] for the optimisation that accounts for the 20% encoding skip visible in the 100k results. See [[Use Cases]] for the real-world scenarios these numbers apply to.
