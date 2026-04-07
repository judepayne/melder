← [Back to Index](./) | [Configuration](configuration.md) | [Performance](performance.md)

# Building

## Build prerequisites

You need **Rust** (which includes Cargo) and a **C++ compiler** (for usearch, which is enabled by default).

### macOS

1. Install Xcode command-line tools (provides Clang, the C++ compiler):

```bash
xcode-select --install
```

2. Install Rust via rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

3. Verify:

```bash
rustc --version
cargo --version
cc --version
```

That's it. Homebrew is not required unless you want GPU encoding (see [GPU encoding](#macos-apple-silicon) below).

### Linux (Debian / Ubuntu)

1. Install the C++ compiler and other build tools:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev
```

2. Install Rust via rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

3. Verify:

```bash
rustc --version
cargo --version
g++ --version
```

For other distributions, install `gcc` or `clang`, `pkg-config`, and `openssl-dev` (or your distro's equivalent) before building.

### Windows

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (free). During setup, select the **"Desktop development with C++"** workload. This provides the MSVC compiler and linker.

2. Install Rust via [rustup](https://rustup.rs/). Download and run `rustup-init.exe`. It detects Visual Studio automatically.

3. Verify in a new terminal (PowerShell or cmd):

```powershell
rustc --version
cargo --version
cl
```

> [!NOTE]
> You must open a **new** terminal after installing Visual Studio Build Tools for the PATH changes to take effect.

## Build commands

```bash
# Default build (includes usearch HNSW)
cargo build --release

# Pure Rust — no C++ dependency, uses flat (brute-force) vector backend
cargo build --release --no-default-features

# Add features as needed
cargo build --release --no-default-features --features usearch
cargo build --release --no-default-features --features usearch,parquet-format
cargo build --release --no-default-features --features usearch,builtin-model
cargo build --release --no-default-features --features usearch,gpu-encode
cargo build --release --no-default-features --features usearch,parquet-format,simd,gpu-encode
```

The binary is produced at `./target/release/meld` (Windows:
`.\target\release\meld.exe`). See the [CLI Reference](cli-reference.md) for
placing it on your PATH.

## Feature flags

| Feature | Default | What it does |
|---------|---------|-------------|
| `usearch` | yes | HNSW approximate nearest-neighbour index. O(log N) candidate search instead of O(N) brute-force. Up to 5x faster at scale. Requires a C++ compiler (MSVC on Windows, GCC/Clang on Unix). |
| `simd` | no | SimSIMD hardware-accelerated dot product (NEON / SVE / AVX2 / AVX-512). Speeds up cosine similarity computation. Pure Rust — no C++ dependency. |
| `parquet-format` | no | Read Parquet files as input datasets. All column types (string, integer, float, boolean) are converted to strings internally. Snappy compression supported. |
| `builtin-model` | no | Compile an embedding model into the binary. No network access or model download needed at runtime. See [Builtin model](#builtin-model) below. |
| `gpu-encode` | no | GPU-accelerated ONNX encoding for batch mode. CoreML on macOS, CUDA on Linux. Requires the ONNX Runtime shared library at runtime. See [GPU encoding](#gpu-encoding) below. |

> [!TIP]
> `usearch` is the only default feature. To build without it (pure Rust, no
> C++ dependency), use `--no-default-features`. This is useful for quick builds
> or environments without a C++ compiler.

## System requirements

**macOS / Linux:** Xcode command-line tools (macOS) or GCC/Clang (Linux)
provide the C++ compiler needed for usearch. Rust installed via
[rustup](https://rustup.rs/).

**Windows:** Install Rust via [rustup](https://rustup.rs/) which defaults to
the MSVC toolchain. You need the **Visual Studio Build Tools** (or full
Visual Studio) with the "Desktop development with C++" workload.

**Linux GPU (CUDA):** See [GPU encoding on Linux](#linux-nvidia-gpu) below.

## Model download

The ONNX model is downloaded automatically on first run to
`~/.cache/huggingface/hub/` (Linux/macOS) or `%LOCALAPPDATA%\huggingface\hub\`
(Windows).

## Builtin model

Build with `--features builtin-model` to compile an embedding model
directly into the binary. No network access or model files on disk at
runtime — ideal for air-gapped, containerised, or edge deployments.

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

The binary size increases by ~90 MB (arctic-embed-xs). The model is
downloaded once at build time and cached in the Cargo build directory.

## GPU encoding

Build with `--features gpu-encode` to enable GPU-accelerated ONNX
encoding for batch mode (`meld run`). Offloads embedding inference to the
GPU, dramatically reducing encoding time for large datasets.

```bash
cargo build --release --features usearch,gpu-encode
```

GPU encoding is **batch mode only**. It is ignored in live mode
(`meld serve`) where single-record GPU dispatch overhead exceeds
the compute savings.

### macOS (Apple Silicon)

Auto-detects ONNX Runtime installed via Homebrew:

```bash
brew install onnxruntime
```

For non-standard locations, set `ORT_DYLIB_PATH`:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
```

The CoreML execution provider is used automatically, dispatching across
CPU, GPU, and the Neural Engine.

### Linux (NVIDIA GPU)

1. Install the CUDA toolkit (12.x recommended).
2. Download ONNX Runtime GPU (>= 1.23.x) from [Microsoft's releases](https://github.com/microsoft/onnxruntime/releases):

```bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-gpu-1.23.1.tgz
tar xzf onnxruntime-linux-x64-gpu-1.23.1.tgz
sudo cp onnxruntime-linux-x64-gpu-1.23.1/lib/libonnxruntime.so* /usr/local/lib/
sudo ln -sf /usr/local/lib/libonnxruntime.so.1.23.1 /usr/local/lib/libonnxruntime.so
sudo ldconfig
```

The library is auto-detected from standard paths (`/usr/lib/`,
`/usr/lib/x86_64-linux-gnu/`, `/usr/local/lib/`). If it's elsewhere,
set `ORT_DYLIB_PATH`:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

> [!WARNING]
> ONNX Runtime **must be >= 1.23.x**. Older versions (e.g. 1.17.x) will be
> rejected at runtime with a version mismatch error. Download the
> `linux-x64-gpu` release matching your CUDA version.

> [!NOTE]
> The usearch dependency requires GCC >= 13 to compile the bundled simsimd
> AVX-512 FP16 intrinsics on Linux. If your system has GCC 11.x (common on
> PyTorch Docker images), build without usearch:
> `cargo build --release --no-default-features --features gpu-encode`.
> This uses the flat vector backend — sufficient for GPU encoding tests and
> small-to-medium datasets.

### Windows

Follow the same steps as Linux: download a CUDA-enabled ONNX Runtime GPU build (>= 1.23.x) from [Microsoft's releases](https://github.com/microsoft/onnxruntime/releases), place `onnxruntime.dll` on your PATH or set `ORT_DYLIB_PATH`. Requires the CUDA toolkit.

### Configuration

```yaml
performance:
  encoder_device: gpu
  encoder_pool_size: 12
  encoder_batch_size: 256
```

> [!NOTE]
> If the binary was built **without** `--features gpu-encode`, setting
> `encoder_device: gpu` will produce a clear error at startup:
> _"GPU encoding requires building with --features gpu-encode"_.

## Benchmark data

The repository includes a synthetic data generator and per-directory
generation scripts. Only the 10k datasets are committed; larger sizes must
be regenerated after cloning.

**Prerequisites:**

```bash
pip install faker pandas pyarrow
```

### Generation scripts

Each benchmark directory has its own `generate_data.sh`:

| Script | Sizes | Output location |
|--------|-------|-----------------|
| `benchmarks/batch/generate_data.sh` | 10k, 100k, 1M | `benchmarks/data/` |
| `benchmarks/live/generate_data.sh` | 10k, 100k, 1M | `benchmarks/data/` |
| `benchmarks/accuracy/generate_data.sh` | 10k | `benchmarks/data/` |

```bash
# Generate all sizes (10k + 100k + 1M)
./benchmarks/batch/generate_data.sh

# Generate specific sizes
./benchmarks/batch/generate_data.sh 10k 100k
./benchmarks/live/generate_data.sh 1M
```

All generators use seed 42 for deterministic output.

### What's committed vs generated

| Data | Status |
|------|--------|
| 10k datasets | Committed in git |
| 100k, 1M datasets | Gitignored — regenerate with scripts above |

### Special cases

- **`accuracy/10kx10k_exclusions`** generates its own data during
  Phase 0 of its `run_test.py` (uses `n_exact=1000`).
- **`accuracy/science/`** is a research journal documenting fine-tuning
  experiments, not a standard benchmark suite. See
  `benchmarks/accuracy/science/experiments.md` for the full history.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG=melder=debug` | Enable debug logging |
| `RUST_LOG=melder=info` | Default log level |
| `RAYON_NUM_THREADS` | Batch scoring thread count (defaults to logical CPU count) |
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.dylib` (macOS) or `libonnxruntime.so` (Linux). Required for `gpu-encode` if ONNX Runtime is not in a standard location. On macOS, Homebrew installs are auto-detected. |
| `MELDER_BUILTIN_MODEL` | Build-time only. HuggingFace repo ID or local path for the model to embed when `--features builtin-model` is enabled. Defaults to `themelder/arctic-embed-xs-entity-resolution`. |

## Windows notes

- Same `cargo build` commands work. Binary at `target\release\meld.exe`.
- Environment variables: use `$env:RUST_LOG = "melder=debug"` (PowerShell) or
  `set RUST_LOG=melder=debug` (cmd).
- Graceful shutdown: use Ctrl-C (SIGTERM is not available on Windows).
- Config paths: both forward slashes and backslashes work in YAML paths.
- `gpu-encode` compiles but falls back to CPU with a warning.
