← [Back to README](../README.md) | [Configuration](configuration.md) | [Performance](performance.md)

# Building

## Build commands

```bash
cargo build --release

# With HNSW vector index + BM25 (recommended for production)
cargo build --release --features usearch,bm25

# With Parquet support
cargo build --release --features parquet-format

# All features together
cargo build --release --features usearch,bm25,parquet-format
```

The binary is produced at `./target/release/meld` (Windows:
`.\target\release\meld.exe`). Either add it to your PATH or invoke it
directly.

## Feature flags

| Feature | What it does |
|---------|-------------|
| `usearch` | Enables the HNSW approximate nearest-neighbour graph index for O(log N) candidate search instead of the flat backend's O(N) brute-force scan. Up to 5x faster at scale. |
| `bm25` | Adds IDF-weighted token matching via a Tantivy index. Suppresses common-token noise from untrained embedding models (e.g. "Holdings", "International"). Can be used as a scoring term alongside embedding, or as the sole candidate filter when no embedding fields are configured (fast start, no ONNX model, no vector index). |
| `parquet-format` | Enables reading Parquet files as input datasets. All column types (string, integer, float, boolean) are converted to strings internally. Snappy-compressed Parquet files are supported. |

> [!TIP]
> On macOS and Linux, always build with `--features usearch,bm25`. The
> usearch backend uses an HNSW graph index for O(log N) candidate
> search instead of the flat backend's O(N) brute-force scan. BM25
> adds IDF-weighted token matching that compensates for common-token
> noise in untrained embedding models. At 100k records, usearch is the
> difference between a 12-second warm run and a 4-minute one — see
> [Performance](performance.md) for full numbers.
>
> On Windows, `usearch` currently has a known MSVC build bug (AVX-512
> FP16 intrinsics and a missing POSIX constant) and must be omitted.
> The flat backend works correctly — just slower at scale. BM25 works
> on all platforms.

## Model download

The ONNX model is downloaded automatically on first run to
`~/.cache/fastembed/` on Linux/macOS or `%LOCALAPPDATA%\fastembed\` on
Windows.

## Environment variables

- `RUST_LOG=melder=debug` — enable debug logging
- `RUST_LOG=melder=info` — default log level
- `RAYON_NUM_THREADS` — batch scoring thread count (defaults to logical CPU count if unset)
- `--log-format json` — JSON structured log output (for production)

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
