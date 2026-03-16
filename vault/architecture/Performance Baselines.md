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

| Metric | flat 10k x 10k | usearch 10k x 10k | usearch 100k x 100k |
|---|---:|---:|---:|
| Index build (cold) | ~17s | ~17s | ~3m 32s |
| Cache load (warm) | ~47ms | ~78ms | ~640ms |
| Scoring throughput | 5,507 rec/s | 31,366 rec/s | 8,735 rec/s |
| Wall time (warm) | 2.2s | 0.7s | 12.8s |

Key insight: usearch is 5.7x faster at 10k and 12.5x faster at 100k due to O(log N) vs O(N) candidate selection.

## Live Mode (80% encoding, 10k x 10k warm caches, c=10)

| Metric | flat | usearch |
|---|---:|---:|
| Startup (cold) | ~17s | ~18s |
| Startup (warm) | ~1.6s | ~1.7s |
| Throughput (cold) | 843 req/s | 1,045 req/s |
| Throughput (warm) | 1,113 req/s | 1,558 req/s |
| p50 latency (warm) | 7.2ms | 3.5ms |
| p95 latency (warm) | 21.2ms | 25.6ms |

## Live Mode (80% encoding, 100k x 100k warm caches, c=10)

| Metric | usearch (10k events) |
|---|---:|
| Startup (cold) | ~3m 29s |
| Startup (warm) | ~7.2s |
| Throughput (cold) | 925 req/s |
| Throughput (warm) | 1,325 req/s |
| p50 latency (warm) | 6.0ms |
| p95 latency (warm) | 19.0ms |

usearch includes text-hash skip: 20% of requests skip ONNX encoding (~1ms vs ~7ms).

## Live Mode — BM25 (10k x 10k, usearch + BM25, inject 3K)

| Metric | Value |
|---|---:|
| Throughput | 615 req/s |
| p50 | ~9ms |
| p95 | ~113ms |

Config: usearch + BM25, explicit bm25_fields.

## Live Mode — SQLite vs In-Memory (10k x 10k, usearch, c=10)

| Metric | In-Memory | SQLite (warm) |
|---|---:|---:|
| Startup (warm) | ~1.9s | ~0.5s |
| Throughput (warm) | 1,616 req/s | 1,183 req/s |
| p95 latency (warm) | 20.7ms | 18.0ms |

Key findings: SQLite warm start is ~4x faster (no CSV parsing or CrossMap load). Throughput is ~25% lower due to Mutex serialization on the shared connection. However, p95 latency is actually better with SQLite — the Mutex eliminates contention spikes from concurrent DashMap access.

## Live Mode — SQLite (post-connection-pool, 10k x 10k, usearch + embeddings, inject 10K)

| Metric | In-Memory | SQLite (warm) |
|---|---:|---:|
| Throughput | 1,698 req/s | 1,395 req/s |
| p50 | 3.4ms | 6.4ms |
| p95 | 23.9ms | 13.6ms |
| Gap | — | 18% slower throughput, better tail latency |

## Batch Mode — SQLite (10k x 10k, BM25 + fuzzy + exact)

| Metric | In-Memory | SQLite (columnar) |
|---|---:|---:|
| Bulk load | N/A | 160-200K rec/s |
| Scoring | 2,289 rec/s | 1,420 rec/s |
| Gap | — | 1.6x (inherent SQLite overhead) |
| Memory | ~2GB | ~1.2GB (cache + BM25 + blocking) |

## Batch Endpoint Throughput (Live Mode)

Sweet spot is batch size 50: 445 req/s (1.8x vs single requests).

## Key Performance Decisions

- usearch (HNSW) is the production backend; flat is for development only
- `encoder_pool_size: 4` is the recommended starting point
- `quantized: true` doubles encoding speed with negligible quality loss
- `vector_quantization: f16` reduces cache size 43% with no measurable quality impact
- Encoding coordinator (`encoder_batch_wait_ms`) is off by default -- only helps at c >= 20 with large models

See also: [[Business Logic Flow]], [[Key Decisions#Text-Hash Skip Optimization]], [[Benchmarks]], [[Use Cases]]
