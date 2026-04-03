---
type: guide
module: benchmarking
status: active
tags: [benchmarks, experiments, testing, performance]
related_code: [benchmarks/]
---

# Benchmarks & Experiments

Running benchmarks and experiments to validate performance and accuracy.

---

## Folder Structure

```
benchmarks/
├── data/           # Generated test datasets (10k, 100k, 1M records)
├── batch/          # Batch mode benchmarks
├── live/           # Live mode benchmarks
├── accuracy/       # Accuracy tests
│   └── science/    # Fine-tuning experiments (rounds 1-14)
├── scripts/        # Shared scripts
└── soak/           # Long-running soak tests
```

---

## Running Benchmarks

### Batch Benchmarks

```bash
# Run all batch benchmarks
python benchmarks/batch/run_all_tests.py

# Single benchmark (warm)
python benchmarks/batch/10kx10k_usearch/warm/run_test.py
```

Key benchmarks:
- `10kx10k_usearch/warm` — 10k×10k, usearch, cached (baseline: ~31k rec/s)
- `100kx100k_usearch` — 100k×100k, usearch (~8.7k rec/s)
- `10kx10k_bm25only` — BM25 scoring only, no embeddings

### Live Benchmarks

```bash
# Run all live benchmarks
python benchmarks/live/run_all_tests.py
```

Key benchmarks:
- `10kx10k_inject3k_usearch` — 10k×10k, inject 3k events (baseline: ~1.6k req/s)
- `100kx100k_inject10k_usearch` — 100k×100k, inject 10k events

### Accuracy Tests

```bash
# Run accuracy evaluation
python benchmarks/accuracy/eval.py
```

Key tests:
- `10kx10k_combined` — Full scoring (embedding + BM25 + fuzzy + synonym)
- `10kx10k_exclusions` — Validates exclusions system
- `10kx10k_bm25only` — BM25-only scoring

---

## Experiments (Fine-Tuning)

Located in `benchmarks/accuracy/science/`:

```bash
# Run experiment
python benchmarks/accuracy/science/run_experiment12.py

# View results
cat benchmarks/accuracy/science/results/round_*.csv
```

**Experiments 1-12** explored fine-tuning (see [[decisions/training_experiments_log]]).

**Experiment 12** (R22) is production: Arctic-embed-xs + 50% BM25 + synonym 0.20.

---

## Data Generation

```bash
# Generate test data
python benchmarks/data/generate.py --count 10000 --matches 6000 --noise 1000
```

- `--exact-matches N` — first N matched B records are exact copies

---

## CI Performance Regression Tests

CI runs `perf` job on every push:
- Generates 10k data
- Runs batch (10kx10k_usearch) and live (10kx10k_inject3k_usearch)
- Fails if throughput < 70% of baseline

Baseline on GitHub ubuntu-latest:
- Batch: ~13k rec/s
- Live: ~535 req/s

---

## Performance Baselines

Benchmarked on Apple Silicon M3 MacBook Air, `all-MiniLM-L6-v2`, `encoder_pool_size: 4`.

### Batch Mode (10K × 10K, top_n: 20, country_code blocking)

| Metric | flat | usearch (warm) | BM25-only | usearch+BM25 |
|---|---:|---:|---:|---:|
| Scoring throughput | 5,507 rec/s | 33,738 rec/s | 49,337 rec/s | 19,034 rec/s |
| Wall time (warm) | 2.2s | 0.3s | 0.2s | 0.5s |
| Auto-matched | — | 5,252 | 7,824 | 5,534 |

### Batch Mode (100K × 100K, usearch, warm, top_n: 20)

| Metric | usearch |
|---|---:|
| Cache load | ~235ms (A) + ~261ms (B) |
| Scoring throughput | 10,539 rec/s |
| Wall time (warm) | 9.5s |

### Batch Mode (1M × 1M, BM25-only, in-memory)

| Metric | Value |
|---|---:|
| Data load | 1.0s (A) + 1.2s (B) |
| BM25 index build | ~1.0s |
| Scoring throughput | 1,062 rec/s |

### Live Mode (10k × 10k, warm caches, c=10)

| Metric | flat | usearch |
|---|---:|---:|
| Startup (cold) | ~17s | ~18s |
| Startup (warm) | ~1.6s | ~1.7s |
| Throughput (cold) | 843 req/s | 1,045 req/s |
| Throughput (warm) | 1,113 req/s | 1,558 req/s |
| p50 latency (warm) | 7.2ms | 3.5ms |

### Live Mode (100k × 100k, usearch, inject 10k events)

| Metric | Value |
|---|---:|
| Startup (cold) | ~3m 29s |
| Startup (warm) | ~7.2s |
| Throughput (cold) | 925 req/s |
| Throughput (warm) | 1,325 req/s |

### Live Mode — SQLite vs In-Memory (10k × 10k, usearch)

| Metric | In-Memory | SQLite (warm) |
|---|---:|---:|
| Startup (warm) | ~1.9s | ~0.5s |
| Throughput (warm) | 1,616 req/s | 1,183 req/s |

### Key Performance Settings

- usearch (HNSW) is production backend; flat is dev only
- `encoder_pool_size: 4` recommended
- `quantized: true` doubles encoding speed
- `vector_quantization: f16` reduces cache 43%

---

See also: [[architecture/Business Logic Flow]], [[decisions/Key Decisions#Text-Hash Skip Optimization]], [[business/Use Cases]]
