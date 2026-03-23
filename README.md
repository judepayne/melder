
```
                                ▄▄                   
 █▄ █▄                           ██    █▄            
▄██▄██            ▄              ██    ██       ▄    
 ██ ████▄ ▄█▀█▄   ███▄███▄ ▄█▀█▄ ██ ▄████ ▄█▀█▄ ████▄
 ██ ██ ██ ██▄█▀   ██ ██ ██ ██▄█▀ ██ ██ ██ ██▄█▀ ██   
▄██▄██ ██▄▀█▄▄▄  ▄██ ██ ▀█▄▀█▄▄▄▄██▄█▀███▄▀█▄▄▄▄█▀   
```

A high-performance record matching engine written in Rust. Given two
datasets — A and B — the melder finds which records in B correspond to
records in A, using a configurable pipeline of exact, fuzzy, semantic,
BM25, and acronym similarity scoring.

This is the kind of problem that comes up in entity resolution,
reconciliation, deduplication, and data migration: you have two lists of
things that *should* refer to the same real-world entities, but the names
are spelled differently, fields are missing, and there is no shared key
to join on.

## Two modes

<ul>
<li><p><strong>Batch mode</strong> (<code>meld run</code>): Load both datasets, match every B record
against the A-side pool, and write results, review, and unmatched csvs.</p>
<blockquote><p><strong>Example use case:</strong> matching huge vendor datasets to your company's
internal reference master overnight, and extracting additional data to enrich your master with.</p></blockquote>
</li>
<li><p><strong>Live mode</strong> (<code>meld serve</code>): Start an HTTP server with both datasets
preloaded. New records can be added to either side at any time, and the melder will immediately find
and return the best matches from the opposite side. A and B sides are treated symmetrically —
both have identical capabilities.</p>
<blockquote><p><strong>Example use case:</strong> You have two master systems with independent
data setup processes, and you wish to sync them in real time.</p></blockquote>
<blockquote><p><strong>Example use case:</strong> You have a master and want to offer a fast
search facility to prevent your users setting up duplicate data.</p></blockquote>
</li>
</ul>

Both modes use the same scoring pipeline, so a match score means the
same thing regardless of how it was produced.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.85 or later
- The embedding model (~90MB) is downloaded automatically on first run

## Quick Start

```bash
# macOS / Linux — build with HNSW vector index (recommended)
cargo build --release --features usearch

# Windows — usearch has a known MSVC build bug; build without it (flat backend)
cargo build --release

# The binary is at ./target/release/meld  (Windows: .\target\release\meld.exe)

# Validate a config file
./target/release/meld validate --config config.yaml

# Run batch matching
./target/release/meld run --config config.yaml

# Start live server on port 8090
./target/release/meld serve --config config.yaml --port 8090

# Check score distribution and tune thresholds
./target/release/meld tune --config config.yaml
```

## How matching works

At its core, the melder asks a simple question for every pair of
records: *how similar are they?* The answer is a single number between
0.0 and 1.0 — the **composite score**.

You define a list of **match fields** in your config. Each entry pairs
a field from dataset A with a field from dataset B, specifies a
comparison method, and assigns a weight. The composite score is the
weighted average:

```
composite = (score_1 x weight_1) + (score_2 x weight_2) + ... + (score_n x weight_n)
```

That score is compared against two thresholds:

- **auto_match** (e.g. 0.64): at or above this, the pair is confirmed
  automatically.
- **review_floor** (e.g. 0.52): between here and auto_match, the pair
  is flagged for human review. Below this, it is discarded.

Under the hood, the melder avoids evaluating every possible pair (which
would be O(N^2)) by narrowing candidates progressively:

1. **Common ID fast-path** — exact ID matches confirmed at score 1.0
2. **Exact prefilter** — multi-field exact match before blocking
3. **Blocking** — cheap field equality eliminates impossible pairs
4. **Candidate selection** — ANN vector search, BM25, and synonym
   index each generate candidates independently; results are merged
5. **Full scoring** — candidates scored across all match fields

### Scoring methods

| Method | What it does | Best for |
|--------|-------------|----------|
| `exact` | Binary string equality | Identifiers, codes, country |
| `fuzzy` | Edit-distance similarity | Names, free text |
| `embedding` | Neural semantic similarity | Primary name field |
| `bm25` | IDF-weighted token overlap | Suppressing common-word noise |
| `synonym` | Acronym/abbreviation detection | "HSBC" vs "Hongkong and Shanghai Banking Corporation" |
| `numeric` | Numeric equality | Numeric identifiers (stub) |

See [Scoring Methods](docs/scoring.md) for detailed descriptions,
configuration examples, and trade-offs for each method.

## Documentation

| Page | Description |
|------|-------------|
| **[Scoring Methods](docs/scoring.md)** | Detailed reference for each comparison method — exact, fuzzy, embedding, BM25, synonym, numeric. Includes configuration examples and trade-offs. |
| **[Configuration](docs/configuration.md)** | Complete annotated YAML config reference. Every field, every option, every default. |
| **[Accuracy & Tuning](docs/accuracy-and-tuning.md)** | Measuring match quality, using `meld tune`, and a worked example showing the journey from 16.5% overlap to near-perfect separation. |
| **[Batch Mode](docs/batch-mode.md)** | Running `meld run` — output files, SQLite batch mode for large datasets, data formats. |
| **[Live Mode](docs/live-mode.md)** | Running `meld serve` — storage backends, persistence, WAL, crash recovery. |
| **[API Reference](docs/api-reference.md)** | HTTP API endpoints — add, remove, match, query, crossmap, review. Request/response examples. |
| **[CLI Reference](docs/cli-reference.md)** | All CLI commands — validate, run, serve, tune, cache, review, crossmap. |
| **[Performance](docs/performance.md)** | Benchmarks for batch and live mode, scaling characteristics, how to run your own benchmarks. |
| **[Vector Caching](docs/vector-caching.md)** | How embedding caches work, staleness detection, incremental encoding. |
| **[Building](docs/building.md)** | Build instructions, feature flags, Windows notes. |
| **[Batch Worked Example](examples/batch/README.md)** | Step-by-step tutorial for batch matching. |
| **[Live Worked Example](examples/live/README.md)** | Step-by-step tutorial for live mode. |
| **[Fine-Tuning Guide](vault/ideas/Fine%20Tuning%20Embeddings.md)** | How to fine-tune embedding models on your own data. |

## Project structure

```
src/
  main.rs              CLI entry point (clap)
  lib.rs               Module exports
  error.rs             Error types (Config, Data, Encoder, Index, CrossMap, Session)
  models.rs            Core types: Record, Side, MatchResult, Classification
  config/              YAML config loading and validation
  data/                Dataset loaders + streaming (csv, JSONL, Parquet)
  encoder/             ONNX encoder pool (fastembed) + batch coordinator
  vectordb/            Vector index abstraction (flat + usearch backends), combined index logic
  fuzzy/               Fuzzy string matchers (ratio, partial_ratio, token_sort, wratio)
  scoring/             Scoring dispatch (exact, fuzzy, embedding, numeric, synonym)
  bm25/                Tantivy-backed BM25 index (feature-gated)
  synonym/             Acronym generator, bidirectional index, binary scorer
  matching/            Blocking filter, candidate selection, scoring pipeline
  crossmap/            Bidirectional ID mapping (memory + SQLite backends)
  store/               Record store abstraction (MemoryStore + SqliteStore with columnar storage)
  batch/               Batch matching engine and output writers
  state/               State management (batch + live), WAL
  session/             Live session logic (upsert, match, remove, crossmap ops)
  api/                 HTTP handlers and server (axum)
  cli/                 One file per subcommand (run, serve, validate, tune, etc.)
```

## License

MIT (c) Jude Payne 2026
