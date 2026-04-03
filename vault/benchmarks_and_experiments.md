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

> [!NOTE]
> Always use the `run_test.py` scripts to run benchmarks — they handle cache
> preservation, output cleanup, and timing. Don't invoke `meld run` directly
> for benchmarking. Scripts live in each benchmark's directory.

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

## CI Accuracy Regression Tests

CI runs `accuracy` job in parallel with `perf` job on every push. Two deterministic tests validate that scoring logic changes don't silently alter match results.

**Live accuracy test** (`benchmarks/accuracy/live_10kx10k_inject3k/`):
- Fixed 10k A + 10k B datasets with asymmetric field names (legal_name vs counterparty_name, country_code vs domicile, lei vs lei_code)
- Validates crossmap at two checkpoints against committed expected files:
  - After initial match pass: 5,376 pairs
  - After 3k record injection: 6,124 pairs
- Any change to scoring logic that alters results causes CI failure

**Enroll accuracy test** (`benchmarks/accuracy/enroll_5k_inject1k/`):
- Fixed 5k single-pool dataset
- Validates 2,814 edges after enrollment
- Removes 50 records, verifies they're gone
- Enrolls 50 more post-removal, validates 136 edges and confirms no edges to removed records
- Tests the full enroll lifecycle: add, score, remove, re-score

**Updating expected outputs**:
```bash
# After intentional scoring changes, regenerate baselines
python benchmarks/accuracy/live_10kx10k_inject3k/run_test.py --update-expected
python benchmarks/accuracy/enroll_5k_inject1k/run_test.py --update-expected
```

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

### Batch Mode (1M × 1M, usearch, cold, GPU encode, production config)

Mac Studio M1 Ultra, 64 GB RAM. `encoder_pool_size: 12`, `encoder_batch_size: 256`,
`encoder_device: gpu`, CoreML EP. Production scoring config (embedding + BM25 + synonym).

| Metric | Value |
|---|---:|
| GPU encoding (cold) | ~2,094s |
| BM25/synonym build | 10.4s |
| Scoring | 1,824s |
| Scoring throughput | 548 rec/s |
| Total wall time | 3,938s (~65 min) |

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

- usearch (HNSW) is the default backend; flat is dev-only or for `--no-default-features` builds
- `encoder_pool_size: 4` recommended
- `quantized: true` doubles encoding speed
- `vector_quantization: f16` reduces cache 43%

---

See also: [[architecture/Business Logic Flow]], [[decisions/Key Decisions#Text-Hash Skip Optimization]], [[business/Use Cases]]
