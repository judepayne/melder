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

# All features together
cargo build --release --features usearch,parquet-format,builtin-model
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

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG=melder=debug` | Enable debug logging |
| `RUST_LOG=melder=info` | Default log level |
| `RAYON_NUM_THREADS` | Batch scoring thread count (defaults to logical CPU count) |
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
