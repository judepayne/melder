# Live Mode Performance

Benchmark results for `meld serve` live mode, measured with the stress test
scripts in `bench/`. All tests use the 10Kx10K synthetic dataset
(`testdata/configs/bench_live.yaml`).

## Setup

- **Hardware:** Apple Silicon Mac
- **Model:** all-MiniLM-L6-v2 (384-dim, ONNX via fastembed)
- **Dataset:** 10,000 A records + 10,000 B records preloaded at startup
- **Blocking:** country_code == domicile (AND mode)
- **Candidates:** top-10 by wratio fuzzy pre-filter
- **Startup time:** ~16s (encoding 20K records into VecIndex)

## Test Scripts

All scripts send a mix of operations:

| Operation | Share | Encoding? |
|-----------|-------|-----------|
| New A record (synthetic, not in dataset) | 30% | Yes |
| New B record (synthetic, not in dataset) | 30% | Yes |
| Update A — embedding field mutated | 10% | Yes |
| Update A — non-embedding field only | 10% | No |
| Update B — embedding field mutated | 10% | Yes |
| Update B — non-embedding field only | 10% | No |

**80% of requests require ONNX encoding**, 20% skip it.

### Smoke Test (`bench/smoke_test.py`)

10 upserts (5 A, 5 B). Validates basic correctness — all return 200 with
5 matches each, latencies 2-4ms.

### Sequential Stress Test (`bench/live_stress_test.py`)

3,000 sequential requests from a single client.

### Concurrent Stress Test (`bench/live_concurrent_test.py`)

3,000 requests fired from 10 parallel workers.

## Results

### Encoder Pool Size = 1

| Test | Requests | Errors | Throughput | p50 | p95 | p99 | max |
|------|----------|--------|-----------|-----|-----|-----|-----|
| Sequential | 3,000 | 0 | 374 req/s | 2.5ms | 3.4ms | 4.7ms | 16.5ms |
| Concurrent (10 workers) | 3,000 | 0 | 605 req/s | 16.4ms | 18.3ms | 21.3ms | 31.9ms |

### Encoder Pool Size = 4

| Test | Requests | Errors | Throughput | p50 | p95 | p99 | max |
|------|----------|--------|-----------|-----|-----|-----|-----|
| Sequential | 3,000 | 0 | 392 req/s | 2.5ms | 3.1ms | 3.4ms | 11.6ms |
| Concurrent (10 workers) | 3,000 | 0 | 891 req/s | 6.5ms | 29.5ms | 42.9ms | 75.3ms |

### Comparison (Concurrent, pool_size=1 vs 4)

| Metric | pool=1 | pool=4 | Change |
|--------|--------|--------|--------|
| Throughput | 605 req/s | 891 req/s | **+47%** |
| p50 | 16.4ms | 6.5ms | **-60%** |
| p95 | 18.3ms | 29.5ms | +61% |
| p99 | 21.3ms | 42.9ms | +101% |

## Observations

**Sequential performance is encoder-pool-insensitive.** With a single client
there is never contention on the encoder mutex, so pool_size=1 and pool_size=4
produce nearly identical results (~374-392 req/s, ~2.5ms p50).

**Concurrent throughput scales well with pool size.** Going from 1 to 4
encoder instances increased throughput by 47% (605 to 891 req/s) because
multiple ONNX sessions can run truly in parallel across CPU cores.

**Median latency drops dramatically, but tail latency increases.** With
pool_size=1, all concurrent requests queue fairly behind a single mutex,
producing tight latency distribution (p50=16.4ms, p99=21.3ms, only 1.3x
spread). With pool_size=4, most requests grab an encoder immediately
(p50=6.5ms), but when all 4 slots are busy simultaneously, unlucky requests
wait longer (p99=42.9ms, 6.6x spread). This is a standard concurrency
tradeoff — higher parallelism improves throughput and median latency at the
cost of wider tail variance.

**Each ONNX session uses ~50-100MB RAM.** pool_size=4 adds ~150-300MB over
pool_size=1. The default is 1; set `performance.encoder_pool_size` in the config
to increase.

**Non-embedding upserts are very fast.** The 20% of requests that only
modify non-embedding fields (lei, lei_code) skip encoding entirely and
complete in ~1-2ms regardless of pool size or concurrency.

## Configuration

Set the encoder pool size in your config YAML:

```yaml
performance:
  encoder_pool_size: 4   # default: 1

live:
  top_n: 5
```

**Guidance:** Use pool_size=1 for single-client / low-concurrency deployments.
Use pool_size=2-4 when serving multiple concurrent clients. Going beyond the
number of CPU performance cores provides diminishing returns.

## Go vs Rust Comparison

Both servers tested on the same machine, same dataset (10Kx10K), same test
scripts, same operation mix (80% encoding, 20% non-encoding). Rust uses
`encoder_pool_size: 4`.

### Sequential (c=1, 3000 requests)

| Metric | Go+Python | Rust (melder) | Speedup |
|--------|-----------|---------------|---------|
| Throughput | 57 req/s | **375 req/s** | **6.6x** |
| p50 | 16.7ms | **2.6ms** | **6.4x** |
| p95 | 21.6ms | **3.3ms** | **6.5x** |
| p99 | 26.2ms | **4.6ms** | **5.7x** |
| max | 1,754ms | **10.5ms** | **167x** |

### Concurrent (c=10, 3000 requests)

| Metric | Go+Python | Rust (melder) | Speedup |
|--------|-----------|---------------|---------|
| Throughput | 129 req/s | **914 req/s** | **7.1x** |
| p50 | 66.5ms | **6.1ms** | **10.9x** |
| p95 | 105.8ms | **29.2ms** | **3.6x** |
| p99 | 331.1ms | **41.2ms** | **8.0x** |
| max | 1,622ms | **85.1ms** | **19.1x** |

### Summary vs Targets

| Metric | Go+Python | Rust target | Rust actual | Status |
|--------|-----------|-------------|-------------|--------|
| Sequential (c=1) | 57 req/s | 400+ | **375** | Near target |
| Concurrent (c=10) | 129 req/s | 1000+ | **914** | Near target |
| Binary size | 46 MB | — | **34 MB** | 26% smaller |

### Analysis

The Rust rewrite delivers **6.5-7x throughput improvement** over the Go+Python
original. The Go server's bottleneck is the Python sidecar process used for
ONNX encoding — each encode call crosses a process boundary via gRPC, adding
~15ms latency per request. The Rust version runs ONNX inference in-process
via fastembed, eliminating this overhead entirely.

Tail latency improvement is even more dramatic — the Go server has occasional
1.5+ second spikes (likely Python GC pauses or gRPC timeouts), while the Rust
server's worst case is 85ms.

The Rust server does not quite hit the stretch targets (400 seq, 1000
concurrent) but is within 6-9%. Further optimization opportunities:
- Increase `encoder_pool_size` beyond 4 on machines with more cores
- Batch multiple concurrent encoding requests into a single ONNX call
- Use SIMD-optimized cosine similarity for the vector search
