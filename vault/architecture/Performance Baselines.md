---
type: architecture
module: general
status: active
tags: [performance, benchmarks, baselines]
related_code: [benchmarks/]
---

# Performance Baselines

Benchmarked on Apple Silicon M3 MacBook Air, `all-MiniLM-L6-v2`, `encoder_pool_size: 4`.

## Batch Mode (flat vs usearch, top_n: 20, country_code blocking)

| Metric | flat 10k x 10k | usearch 10k x 10k | flat 100k x 100k | usearch 100k x 100k |
|---|---:|---:|---:|---:|
| Index build (cold) | ~16s | ~18s | ~3m | ~3m 24s |
| Index load (warm) | ~25ms | ~50ms | ~650ms | ~700ms |
| Scoring throughput | 4,877 rec/s | 22,464 rec/s | 363 rec/s | 9,886 rec/s |
| Wall time (warm) | 1.6s | 0.85s | 4m 37s | 12s |

Key insight: usearch is 4.6x faster at 10k and 27x faster at 100k due to O(log N) vs O(N) candidate selection.

## Live Mode (3,000 adds, 80% encoding, 10k x 10k warm caches)

| Metric | flat c=1 | usearch c=1 | flat c=10 | usearch c=10 |
|---|---:|---:|---:|---:|
| Throughput | 349 req/s | 673 req/s | 616 req/s | 1,624 req/s |
| p50 latency | 2.2ms | 0.7ms | 13.3ms | 3.9ms |
| p95 latency | 4.4ms | 2.9ms | 39.9ms | 21.8ms |

## Live Mode (3,000 adds, 80% encoding, 100k x 100k warm caches)

| Metric | flat c=1 | usearch c=1 | flat c=10 | usearch c=10 |
|---|---:|---:|---:|---:|
| Throughput | 160 req/s | 546 req/s | 251 req/s | 1,448 req/s |
| p50 latency | 5.6ms | 0.9ms | 30.5ms | 5.1ms |
| p95 latency | 8.1ms | 3.5ms | 94.4ms | 19.8ms |

usearch c=10 includes text-hash skip: 20% of requests skip ONNX encoding (~1ms vs ~7ms).

## Batch Endpoint Throughput (Live Mode)

Sweet spot is batch size 50: 445 req/s (1.8x vs single requests).

## Key Performance Decisions

- usearch (HNSW) is the production backend; flat is for development only
- `encoder_pool_size: 4` is the recommended starting point
- `quantized: true` doubles encoding speed with negligible quality loss
- `vector_quantization: f16` reduces cache size 43% with no measurable quality impact
- Encoding coordinator (`encoder_batch_wait_ms`) is off by default -- only helps at c >= 20 with large models

See also: [[Business Logic Flow]], [[Key Decisions#Text-Hash Skip Optimization]], [[Benchmarks]], [[Use Cases]]
