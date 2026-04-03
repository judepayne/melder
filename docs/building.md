← [Back to Index](./) | [Configuration](configuration.md) | [Performance](performance.md)

# Building

## Build commands

```bash
cargo build --release

# With HNSW vector index (recommended for production)
cargo build --release --features usearch

# With Parquet support
cargo build --release --features usearch,parquet-format

# With builtin embedding model (self-contained binary, no network needed at runtime)
cargo build --release --features usearch,builtin-model

# With GPU-accelerated encoding (CoreML on macOS, CUDA on Linux)
cargo build --release --features usearch,gpu-encode

# All features together
cargo build --release --features usearch,parquet-format,builtin-model,gpu-encode
```

The binary is produced at `./target/release/meld` (Windows:
`.\target\release\meld.exe`). Either add it to your PATH or invoke it
directly.

## Feature flags

| Feature | What it does |
|---------|-------------|
| `usearch` | Enables the HNSW approximate nearest-neighbour graph index for O(log N) candidate search instead of the flat backend's O(N) brute-force scan. Up to 5x faster at scale. |
| `parquet-format` | Enables reading Parquet files as input datasets. All column types (string, integer, float, boolean) are converted to strings internally. Snappy-compressed Parquet files are supported. |
| `builtin-model` | Compiles an embedding model into the binary so no network access or model download is needed at runtime. Set `model: builtin` in config to use it. See [Builtin model](#builtin-model) below. |
| `gpu-encode` | Enables GPU-accelerated ONNX encoding for batch mode. Uses CoreML on macOS and CUDA on Linux. Requires the ONNX Runtime shared library at runtime. See [GPU encoding](#gpu-encoding) below. |

> [!TIP]
> On macOS and Linux, always build with `--features usearch`. The
> usearch backend uses an HNSW graph index for O(log N) candidate
> search instead of the flat backend's O(N) brute-force scan. At 100k
> records, usearch is the difference between a 12-second warm run and a
> 4-minute one — see [Performance](performance.md) for full numbers.
>
> On Windows, `usearch` currently has a known MSVC build bug (AVX-512
> FP16 intrinsics and a missing POSIX constant) and must be omitted.
> The flat backend works correctly — just slower at scale.

## Model download

The ONNX model is downloaded automatically on first run to
`~/.cache/fastembed/` on Linux/macOS or `%LOCALAPPDATA%\fastembed\` on
Windows.

## Builtin model

Build with `--features builtin-model` to compile an embedding model
directly into the binary. The resulting binary needs no network access
and no model files on disk — ideal for air-gapped, containerised, or
edge deployments.

```bash
# Default: embeds themelder/arctic-embed-xs-entity-resolution from HuggingFace
cargo build --release --features usearch,builtin-model

# Custom model from HuggingFace
MELDER_BUILTIN_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    cargo build --release --features usearch,builtin-model

# Custom model from a local directory
MELDER_BUILTIN_MODEL=./models/my-fine-tuned-model \
    cargo build --release --features usearch,builtin-model
```

The model directory must contain the standard HuggingFace ONNX export
files: `model.onnx`, `tokenizer.json`, `config.json`,
`special_tokens_map.json`, and `tokenizer_config.json`.

To use the builtin model at runtime, set `model: builtin` in your
config:

```yaml
embeddings:
  model: builtin
```

The binary size increases by the size of the ONNX model (~90 MB for
arctic-embed-xs). The model is downloaded once at build time and cached
in the Cargo build directory — subsequent builds reuse the cached files.

## GPU encoding

Build with `--features gpu-encode` to enable GPU-accelerated ONNX
encoding for batch mode (`meld run`). This offloads the embedding
inference to the GPU, dramatically reducing encoding time for large
datasets.

```bash
cargo build --release --features usearch,gpu-encode
```

GPU encoding is **batch mode only**. It is ignored in live mode
(`meld serve`) where single-record GPU dispatch overhead exceeds
the compute savings.

### Platform setup

GPU encoding uses dynamic linking against the ONNX Runtime shared
library. The library must be present at runtime.

**macOS (Apple Silicon).** The melder auto-detects ONNX Runtime
installed via Homebrew. No manual configuration needed:

```bash
brew install onnxruntime
```

If you install ONNX Runtime to a non-standard location, set the
`ORT_DYLIB_PATH` environment variable:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
```

The CoreML execution provider is used automatically. It dispatches
compute across CPU, GPU, and the Neural Engine (ANE) based on the
model graph.

**Linux (NVIDIA GPU).** Download a CUDA-enabled ONNX Runtime build
from [Microsoft's releases](https://github.com/microsoft/onnxruntime/releases)
and set `ORT_DYLIB_PATH`:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

Requires CUDA toolkit and cuDNN to be installed. The CUDA execution
provider is used automatically.

**Windows.** GPU encoding is not currently supported on Windows. The
`gpu-encode` feature will compile but `encoder_device: gpu` in the
config will fall back to CPU with a warning.

### Configuration

Set `encoder_device: gpu` in the `performance` section of your
config file:

```yaml
performance:
  encoder_device: gpu
  encoder_pool_size: 12    # ~60% of CPU cores (see tuning guide)
  encoder_batch_size: 256  # optimal for GPU; default when device is gpu
```

See [Configuration](configuration.md#performance-field-reference) for
the full tuning guide including benchmark results.

> [!NOTE]
> If the binary was built **without** `--features gpu-encode`, setting
> `encoder_device: gpu` in the config will produce a clear error at
> startup: _"GPU encoding requires building with --features gpu-encode"_.

## Benchmark data

The repository includes a synthetic data generator and per-directory
generation scripts. Only the 10k datasets are committed (small enough
for CI); larger sizes must be regenerated after cloning.

**Prerequisites:**

```bash
pip install faker pandas pyarrow
```

### Generation scripts

Each benchmark directory has its own `generate_data.sh` that produces
exactly the datasets its benchmarks need:

| Script | Sizes | Output location |
|--------|-------|-----------------|
| `benchmarks/batch/generate_data.sh` | 10k, 100k, 1M | `benchmarks/data/` |
| `benchmarks/live/generate_data.sh` | 10k, 100k, 1M | `benchmarks/data/` |
| `benchmarks/accuracy/generate_data.sh` | 10k | `benchmarks/data/` |
| `benchmarks/accuracy/science/generate_data.sh` | 27 training rounds | `benchmarks/accuracy/science/rounds/` |

The batch and live scripts accept size arguments:

```bash
# Generate all sizes (10k + 100k + 1M)
./benchmarks/batch/generate_data.sh

# Generate specific sizes
./benchmarks/batch/generate_data.sh 10k 100k
./benchmarks/live/generate_data.sh 1M

# Accuracy benchmarks only need 10k
./benchmarks/accuracy/generate_data.sh

# Science training rounds (wraps setup_datasets.py)
./benchmarks/accuracy/science/generate_data.sh
```

All generators use seed 42 (or fixed per-round seeds for science) for
deterministic, reproducible output. The `--addresses` flag is always
passed so datasets include address fields used by some configs.

### What's committed vs generated

| Data | Status |
|------|--------|
| 10k datasets (`benchmarks/data/dataset_*_10k.*`) | Committed in git |
| 100k, 1M datasets | Gitignored — regenerate with scripts above |
| Science master + holdout (`science/master/`, `science/holdout/`) | Committed in git |
| Science round datasets (`science/rounds/`) | Gitignored — regenerate with script above |

### Special cases

- **`accuracy/10kx10k_exclusions`** generates its own data during
  Phase 0 of its `run_test.py` (uses `n_exact=1000` for exact-match
  exclusion testing). It is not covered by the accuracy generation
  script.

- **`accuracy/science/`** uses its own fixed datasets (committed
  `master/dataset_a.csv` seed 0 and `holdout/dataset_b.csv` seed 9999),
  not the shared `benchmarks/data/` pool.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG=melder=debug` | Enable debug logging |
| `RUST_LOG=melder=info` | Default log level |
| `RAYON_NUM_THREADS` | Batch scoring thread count (defaults to logical CPU count) |
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.dylib` (macOS) or `libonnxruntime.so` (Linux). Required for `gpu-encode` if ONNX Runtime is not in a standard location. On macOS, Homebrew installs are auto-detected. |
| `MELDER_BUILTIN_MODEL` | Build-time only. HuggingFace repo ID or local path for the model to embed when `--features builtin-model` is enabled. Defaults to `themelder/arctic-embed-xs-entity-resolution`. |

## Windows

The melder builds and runs on Windows. A few things to be aware of:

**Prerequisites.** Install Rust via [rustup](https://rustup.rs/) which
defaults to the MSVC toolchain on Windows. You will need the
**Visual Studio Build Tools** (or full Visual Studio) with the
"Desktop development with C++" workload installed — this provides the
MSVC compiler and linker that Rust requires.

> [!WARNING]
> The `usearch` feature currently has a known build bug on MSVC
> (AVX-512 FP16 intrinsics and a missing POSIX constant). Build without
> it on Windows — the flat vector backend works correctly, just with
> O(N) instead of O(log N) candidate search.

**Building.** The same `cargo build` commands work. The binary is
produced at `target\release\meld.exe`:

```powershell
cargo build --release --features parquet-format
.\target\release\meld.exe validate --config config.yaml
```

**Environment variables.** `RUST_LOG` and `RAYON_NUM_THREADS` work the
same way. Set them in PowerShell with `$env:RUST_LOG = "melder=debug"`
or in cmd with `set RUST_LOG=melder=debug`.

**Graceful shutdown.** On Unix, the melder listens for both Ctrl-C and
SIGTERM. On Windows, SIGTERM is not available — use Ctrl-C to trigger
a clean shutdown (in-flight requests drain, WAL is compacted, cross-map
and index caches are saved).

**Config paths.** Both forward slashes and backslashes work in YAML
config file paths (`datasets.a.path`, `output.results_path`, etc.).
