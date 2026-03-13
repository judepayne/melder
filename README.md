
```
                                ▄▄                   
 █▄ █▄                           ██    █▄            
▄██▄██            ▄              ██    ██       ▄    
 ██ ████▄ ▄█▀█▄   ███▄███▄ ▄█▀█▄ ██ ▄████ ▄█▀█▄ ████▄
 ██ ██ ██ ██▄█▀   ██ ██ ██ ██▄█▀ ██ ██ ██ ██▄█▀ ██   
▄██▄██ ██▄▀█▄▄▄  ▄██ ██ ▀█▄▀█▄▄▄▄██▄█▀███▄▀█▄▄▄▄█▀   
```

A high-performance record matching engine written in Rust. Given two
datasets -- A and B -- the melder finds which records in B correspond to
records in A, using a configurable pipeline of exact, fuzzy, and
semantic similarity scoring.

This is the kind of problem that comes up in entity resolution,
reconciliation, deduplication, and data migration: you have
two lists of things that *should* refer to the same real-world entities,
but the names are spelled differently, fields are missing, and there is
no shared key to join on.

Operates in two modes:

<ul>
<li><p><strong>Batch mode</strong> (<code>meld run</code>): Load both datasets, match every B record
against the A-side pool, and write results, review, and unmatched csvs.</p>
<blockquote><p><strong>Example use case:</strong> matching huge vendor datasets to your company's
internal reference master overnight, and extracting additional data to enrich your master with.</p></blockquote>
</li>
<li><p><strong>Live mode</strong> (<code>meld serve</code>): Start an HTTP server with both datasets
preloaded. New records can be added to either side at any time, and melder will immediately find
and return the best matches from the opposite side. A and B sides are treated symmetrically —
both have identical capabilities.</p>
<blockquote><p><strong>Example use case:</strong> You have two master systems with independent
data setup processes, and you wish to sync them in real time. The machinery to create common
identifiers is outside of melder's remit, but melder strongly supports being the core of
the end-to-end processing chain.</p></blockquote>
<blockquote><p><strong>Example use case:</strong> You have a master and want to offer a fast
search facility to prevent your users setting up duplicate data.</p></blockquote>
</li>
</ul>

Both modes are optimized for high performance, supporting large dataset sizes,
fast response times, and high throughput in live mode.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Matching Algorithms](#matching-algorithms)
- [Scoring Pipeline](#scoring-pipeline)
- [Operation](#operation)
- [CLI Reference](#cli-reference)
- [Live Mode API](#live-mode-api)
- [Data Formats](#data-formats)
- [Performance](#performance)
- [How vector caching works](#how-vector-caching-works)
- [Project Structure](#project-structure)
- [Building](#building)
- [License](#license)

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.85 or later
- The embedding model (~90MB) is downloaded automatically on first run

## Quick Start

```bash
# macOS / Linux — build with HNSW vector index (strongly recommended; up to 5x faster at scale)
cargo build --release --features usearch

# Windows — usearch has a known MSVC build bug; build without it (flat backend)
cargo build --release

# The binary is at ./target/release/meld  (Windows: .\target\release\meld.exe)
# Either add it to your PATH or invoke it directly:

# Validate a config file
./target/release/meld validate --config config.yaml

# Run batch matching
./target/release/meld run --config config.yaml

# Start live server on port 8090
./target/release/meld serve --config config.yaml --port 8090

# Check score distribution and tune thresholds
./target/release/meld tune --config config.yaml
```

## Configuration

All behaviour is driven by a single YAML config file. Here is a working
example that matches a list of legal entities (A) against a list of
counterparties (B):

```yaml
# --- Job metadata (for your reference, not used by the engine) ----------
job:
  name: counterparty_recon
  description: Match entities to counterparties

# --- Datasets -----------------------------------------------------------
# Each side needs a file path and the name of the column that contains
# the unique record ID. Field names in A and B do not need to match --
# the mapping between them is defined in match_fields below.
datasets:
  a:
    path: entities.csv
    id_field: entity_id
    # common_id_field: isin    # optional -- see note below
  b:
    path: counterparties.csv
    id_field: counterparty_id
    # common_id_field: isin    # must be set on both sides or neither

# If both sides share a stable identifier (ISIN, LEI, etc.), set
# common_id_field on both datasets. Records with identical values are
# matched immediately with score 1.0 before any scoring runs.

# --- Embedding model -----------------------------------------------------
# Used by any match field with method: embedding. The model is downloaded
# automatically on first run.
#
# a_cache_dir (required): directory for the A-side combined embedding index cache.
# Created automatically on first run; loaded on subsequent runs to skip encoding.
#
# b_cache_dir (optional): directory for B-side caches. Omit to skip B-side
# caching (vectors rebuilt from scratch each run).
embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: cache
  b_cache_dir: cache              # optional — omit to skip B-side caching

# --- Vector backend -------------------------------------------------------
# Controls the embedding index used for candidate selection.
#
# flat    (default): brute-force scan — O(N) search, no dependencies.
#                    Good for development and datasets under ~10k records.
# usearch: HNSW approximate nearest-neighbour index — O(log N) search.
#           Dramatically faster at scale. Requires the usearch feature flag
#           at build time: cargo build --release --features usearch
#
# Use flat for quick tests; use usearch for any production workload.
vector_backend: usearch

# --- top_n ----------------------------------------------------------------
# How many candidates to retrieve from the embedding index before full
# scoring, and the maximum number of match results returned per record.
# A value of 20 is a good default: large enough to find the true match
# almost always, small enough that full scoring stays fast.
top_n: 20

# --- Blocking (pre-filter) -----------------------------------------------
# Before candidate selection, blocking eliminates obviously wrong candidates by
# requiring cheap field equality. Here we require the country to match --
# a record in France will never be compared against one in Japan.
# This dramatically reduces the number of pairs that need expensive
# scoring, at the cost of never finding cross-country matches.
#
# To disable blocking entirely, set enabled: false (or omit the blocking
# section). Every record will then be considered as a candidate, which
# is thorough but slower on large datasets.
blocking:
  enabled: true
  fields:
    - field_a: country_code
      field_b: domicile

# --- Match fields (full scoring) -------------------------------------------
# The match_fields list defines how each field pair is compared and how
# much it contributes to the overall score.
# Each entry pairs a field from A with a field from B, specifies how to
# compare them, and assigns a weight. The weights control how much each
# field contributes to the overall score. Weights must sum to exactly
# 1.0 -- this is validated at startup and the engine will refuse to run
# if they don't.
#
# Available methods: exact, fuzzy, embedding, numeric
# Available fuzzy scorers: ratio, partial_ratio, token_sort_ratio, wratio
match_fields:
  # Primary comparison: use the embedding model to capture semantic
  # similarity between the legal name and counterparty name. This is
  # the most powerful method -- it understands that "JP Morgan Chase"
  # and "JPMorgan" refer to the same entity even though the characters
  # are quite different. Highest weight because it is the most
  # informative signal.
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.55

  # Secondary comparison: use fuzzy character matching on the short
  # name. partial_ratio is good here because a short name like "HSBC"
  # may appear within a longer counterparty name like "HSBC Holdings".
  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy
    scorer: partial_ratio
    weight: 0.20

  # Country must match exactly. This reinforces the blocking filter --
  # even if blocking is disabled, country match still contributes to
  # the score.
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20

  # LEI (Legal Entity Identifier) is a global standard code. When both
  # sides have one it is strong evidence of a match, but many records
  # have no LEI so it gets a low weight to avoid penalising the score
  # when the field is empty.
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05

# --- Thresholds -----------------------------------------------------------
# auto_match: pairs scoring at or above this are considered confirmed
#   matches. They go straight into the results output.
# review_floor: pairs scoring between review_floor and auto_match are
#   borderline -- they are written to the review csv for a human to
#   decide. Pairs below review_floor are discarded as non-matches.
thresholds:
  auto_match: 0.85
  review_floor: 0.60

# --- Performance tuning (optional) ----------------------------------------
# All fields have sensible defaults; omit the section entirely if unsure.
performance:
  encoder_pool_size: 4            # parallel ONNX sessions (default: 1)
  quantized: true                 # INT8 quantised ONNX model (default: false)
  vector_quantization: f16        # usearch storage precision: f32, f16, bf16 (default: f32)

# --- Output paths (batch mode) -------------------------------------------
output:
  results_path: output/results.csv      # confirmed + auto matches
  review_path: output/review.csv        # borderline pairs for human review
  unmatched_path: output/unmatched.csv   # B records with no match at all
```

### Live mode config

Add a `live` section to use `meld serve`. The `--port` flag on the
command line sets the listening port (default 8080).

```yaml
live:
  upsert_log: wal.ndjson  # write-ahead log for crash recovery

top_n: 5                  # max matches returned per request (top-level field)
```

`top_n` is a top-level config field shared by both batch and live modes.

### Performance config

The `performance` section is optional. All fields have sensible defaults.

```yaml
performance:
  encoder_pool_size: 4            # parallel ONNX sessions (default: 1)
  encoder_batch_wait_ms: 0        # live mode only — batch window in ms (default: 0, disabled)
  quantized: true                 # INT8 quantised ONNX model (default: false)
  vector_quantization: f16        # usearch storage precision: f32, f16, bf16 (default: f32)
```

**`encoder_pool_size`** -- number of ONNX inference sessions to run in
parallel. Each session holds a copy of the model in memory. Higher
values increase encoding throughput at the cost of RAM. 4 is a good
starting point on machines with 4+ cores; 1 is fine for small datasets.

**`quantized`** -- load the INT8 quantised variant of the ONNX model
instead of the full FP32 model. Roughly doubles encoding speed with
negligible quality loss. Supported for `all-MiniLM-L6-v2` and
`all-MiniLM-L12-v2`; BGE models do not have quantised variants and
will error if set. Expect ~2% of borderline pairs to shift
classification bucket when switching between full-precision and
quantised on the same dataset.

**`vector_quantization`** -- controls how the usearch vector backend
stores vectors on disk and in memory. Allowed values:

| Value | Bytes per dimension | Notes |
|-------|--------------------:|-------|
| `f32` | 4 | Full precision. Default. |
| `f16` | 2 | Half precision. ~43% smaller index, negligible recall loss. |
| `bf16` | 2 | Brain float 16. Similar savings to f16, slightly different rounding. |

The primary benefit is **disk cache size**. At 100k records with 384-dim
embeddings, the usearch cache is 171 MB per side with `f32` and 98 MB
per side with `f16` -- a 43% reduction. Scoring throughput and match
quality are effectively unchanged:

| Metric | f32 | f16 |
|---|---:|---:|
| Warm scoring throughput | 9,241 rec/s | 9,346 rec/s |
| Cache size (A + B) | 346 MB | 199 MB |
| Cache load time | 612 ms | 599 ms |
| Auto-matched (of 100k) | 53,395 | 53,387 |

Only affects the `usearch` backend; the `flat` backend always stores
full f32. Changing this value invalidates existing caches (the spec
hash changes), so the first run after a change will do a cold rebuild.

> **`quantized` vs `vector_quantization`** -- these are independent
> settings that control different things.
> `quantized` controls the *ONNX encoder model* precision (FP32 vs INT8)
> -- it affects how fast text is converted into vectors.
> `vector_quantization` controls the *vector index storage* precision
> (f32/f16/bf16) -- it affects how much disk and memory the cached
> index consumes. You can use either, both, or neither.

**`encoder_batch_wait_ms`** -- live mode only. When > 0, concurrent
encoding requests are collected for up to this many milliseconds and
dispatched as a single ONNX batch call. This can improve throughput
with large models or very high concurrency (c >= 20), but adds
latency equal to the batch window. With small models (MiniLM) and
`encoder_pool_size >= 4`, leaving this at 0 (disabled) is typically
faster because parallel independent sessions outperform batched
single-session encoding. Has no effect on batch mode.

Batch scoring thread count is controlled by the `RAYON_NUM_THREADS`
environment variable (defaults to logical CPU count if unset).

## Matching Algorithms

melder supports four scoring methods, each suited to different field
types. Each field is scored independently, producing a value between
0.0 and 1.0. These per-field scores are then combined into a weighted
average to produce a single composite score for the pair.

### Exact

**Algorithm class:** Binary string equality.

**Implementation:** Custom code, no external dependencies.

Case-insensitive string comparison. Returns 1.0 on match, 0.0 otherwise.
Uses fast ASCII comparison first, falls back to full Unicode case folding
for non-ASCII input. Both-empty returns 0.0 (no evidence of a match).

**Best for:** Identifiers, codes, and categorical fields where partial
similarity is meaningless (country codes, LEIs, ISINs, currency codes).

**Trade-offs:** Zero tolerance for typos or formatting differences. Very
fast -- single string comparison with no allocation on the fast path.

### Fuzzy

**Algorithm class:** Edit distance / string similarity.

**Implementation:** Core `ratio` function from the
[rapidfuzz](https://crates.io/crates/rapidfuzz) Rust crate (Levenshtein
distance). `partial_ratio`, `token_sort_ratio`, and `wratio` are custom
implementations built on top of it -- the rapidfuzz Rust crate (v0.5)
only exposes the base ratio function.

Four scorers are available, selected via the `scorer` config field:

| Scorer | What it does | Good at |
|--------|-------------|---------|
| `ratio` | Normalized Levenshtein similarity | General string comparison |
| `partial_ratio` | Best substring alignment (sliding window) | Short names within longer strings |
| `token_sort_ratio` | Sort tokens alphabetically, then ratio | Reordered words ("Goldman Sachs" vs "Sachs Goldman") |
| `wratio` (default) | Max of ratio, partial_ratio, token_sort | Robust catch-all when you don't know the error pattern |

All scorers normalize input (lowercase, trim whitespace) before comparison
and return a score in [0.0, 1.0] where 1.0 means identical strings.

**Best for:** Names, descriptions, and free-text fields where character-level
similarity matters. Use `partial_ratio` when short names may appear within
longer legal names. Use `token_sort_ratio` when word order varies.

**Trade-offs:** `wratio` runs all three sub-scorers and takes the max, so it
costs roughly 3x a single scorer. For typical entity names (< 100 characters)
this is sub-millisecond. Pure edit distance cannot handle synonyms or
abbreviations ("Corp" vs "Corporation", "JPM" vs "J.P. Morgan") -- use
embedding scoring for that.

### Embedding

**Algorithm class:** Neural sentence embedding with cosine similarity.

**Implementation:** Encoding via the [fastembed](https://crates.io/crates/fastembed)
Rust crate, which runs ONNX Runtime inference with Sentence Transformer
models. Similarity computation and vector index are custom code.

Supported models (configured via `embeddings.model`):

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `all-MiniLM-L6-v2` | 384 | Default. Good balance of speed and quality |
| `all-MiniLM-L12-v2` | 384 | Slightly better quality, ~2x slower |
| `bge-small-en-v1.5` | 384 | BAAI family, English-optimized |
| `bge-base-en-v1.5` | 768 | Higher capacity, ~2x more memory |
| `bge-large-en-v1.5` | 1024 | Highest quality, ~4x slower than MiniLM-L6 |

How it works:

1. Each text value is fed through a transformer model that converts it
   into a dense numeric vector (e.g. 384 numbers) capturing its semantic
   meaning. Texts with similar meanings end up with similar vectors,
   regardless of exact wording.
2. Vectors are stored in an index. When a match field uses method
   `embedding`, the precomputed vectors for both records are compared
   using cosine similarity to produce the field score.
3. Cosine similarity is clamped to [0.0, 1.0]. A score of 1.0 means
   the texts have identical meaning (or are identical); a score near 0.0
   means they are unrelated.

**Best for:** The primary matching field (usually legal/company names).
Captures semantic similarity that edit distance cannot: "JP Morgan Chase"
vs "JPMorgan" scores high despite low character overlap. Handles
abbreviations, word reordering, and minor language variations.

**Trade-offs:** Requires a model download (~90MB, auto-downloaded on first
run to `~/.cache/fastembed/`). Each ONNX inference takes ~1-3ms per text.
The vector index is brute-force O(N*D), which is fine up to ~100K records
but would need an ANN index (HNSW, IVF) for millions. Each encoder pool
slot uses ~50-100MB of RAM.

### Numeric

**Algorithm class:** Numeric equality.

**Implementation:** Custom code. Currently a stub -- parses both values as
floats and returns 1.0 if equal (within f64 epsilon), 0.0 otherwise. No
range or tolerance matching.

**Best for:** Not recommended in current form. Use `exact` for numeric
identifiers. A future version may add tolerance-based comparison.

## Scoring Pipeline

To understand how matching works end-to-end, consider the config example
above where we are matching entities against counterparties.

When melder processes a B-side record -- say counterparty "JPMorgan
Chase & Co" from the US -- the pipeline works as follows:

**Step 0: Common ID fast-path.**
If `common_id_field` is configured on both datasets, melder first
checks whether the incoming record shares a common identifier (ISIN,
LEI, etc.) with any record on the opposite side. If a match is found,
the pair is immediately confirmed with score 1.0 and no further scoring
is performed. This short-circuit happens before any vector arithmetic.

**Step 1: Blocking.**
Before any scoring happens, melder eliminates records that cannot
possibly match using cheap field equality. In our config, blocking
requires `country_code == domicile`, so if the counterparty's domicile
is "US", only A-side entities with `country_code: US` are considered.
This is free to evaluate and can eliminate 95%+ of the pool in a well-
blocked job. Disable it (`enabled: false`) only if you need cross-
country matching.

**Step 2: Candidate selection via the combined embedding index.**
Among the records that passed blocking there may still be hundreds.
Rather than scoring every one of them across all match fields, melder
uses the embedding index to find the `top_n` nearest neighbours in one
fast vector lookup. This is the heart of the performance story.

Melder represents each record as a single *combined embedding vector*.
For every `method: embedding` field in your config, melder encodes
the field text into a 384-dimensional unit vector and scales it by the
square root of the field's weight. These scaled vectors are concatenated
end-to-end into one long vector per record. The mathematical consequence
is that the dot product between two combined vectors exactly equals the
weighted cosine-similarity sum that full scoring would compute for the
embedding fields alone. This means the ANN search retrieves the same
top-N candidates that an exhaustive scan would find, without scanning
everything.

With `vector_backend: usearch`, this search runs over an HNSW graph
index and takes O(log N) time regardless of pool size. With
`vector_backend: flat` it falls back to a brute-force O(N) scan.

**Step 3: Full scoring.**
Each of the `top_n` candidates is scored across all match fields:

| Field pair | Method | Score example | Weight |
|-----------|--------|---------------|--------|
| legal_name vs counterparty_name | embedding | 0.92 | 0.55 |
| short_name vs counterparty_name | fuzzy (partial_ratio) | 0.85 | 0.20 |
| country_code vs domicile | exact | 1.00 | 0.20 |
| lei vs lei_code | exact | 0.00 (one side empty) | 0.05 |

The per-field embedding scores are recovered cheaply by slicing the
combined vectors — no second model call is needed. The composite score
is the weighted average:

`(0.92 × 0.55) + (0.85 × 0.20) + (1.00 × 0.20) + (0.00 × 0.05) = 0.876`

**Step 4: Classification.**
The composite score is compared against the configured thresholds:

- **0.876 ≥ 0.85 (auto_match)?** Yes — classified as `auto`, written
  to the results file as a confirmed match.
- A score of 0.72 falls between `review_floor` (0.60) and `auto_match`
  (0.85) — classified as `review`, written to the review file for a
  human to decide.
- Below 0.60 — discarded as a non-match.

The same pipeline (`matching::pipeline::score_pool`) is used by both
batch and live modes. In batch mode, Rayon parallelises it across all B
records. In live mode, each incoming request runs it inside a
`spawn_blocking` task.

## Operation

### Batch mode

> For a hands-on walkthrough, see the
> [batch worked example](examples/batch/README.md).

Batch mode processes an entire B dataset against a pre-indexed A pool in
a single pass. Run it with:

```bash
meld run --config config.yaml
```

When the job completes, melder writes three output csvs (paths configured
in the `output` section):

| File | Contents |
|------|----------|
| `results.csv` | Confirmed matches -- pairs that scored at or above `auto_match`. Columns: `a_id`, `b_id`, `score`, `classification`, plus a per-field score column for each match field. |
| `review.csv` | Borderline pairs -- scored between `review_floor` and `auto_match`. Same columns as results. These need a human decision. |
| `unmatched.csv` | B records that had no match above `review_floor`. Contains all original B-side fields so you can inspect what failed. |

The cross-map csv is also updated: every auto-matched pair is added so
that re-running the job skips already-resolved records.

A summary is printed to stdout at the end:

```
Batch matching complete:
  Total B records: 1000
  Skipped (crossmap): 0
  Auto-matched: 712
  Review:       138
  No match:     150
  Elapsed:      8.3s
  Throughput:   120 records/sec

Output files:
  results:   output/results.csv (712 rows)
  review:    output/review.csv (138 rows)
  unmatched: output/unmatched.csv (150 rows)
```

Use `--dry-run` to validate the config and print what would be processed
without actually running. Use `--limit N` to process only the first N
B records (useful for quick sanity checks on large datasets).

### Live mode

> For a hands-on walkthrough, see the
> [live worked example](examples/live/README.md).

Live mode starts an HTTP server that holds both datasets in memory and
matches records on the fly as they arrive:

```bash
meld serve --config config.yaml --port 8090
```

**Startup.** On launch, melder loads both datasets, builds embedding
indices and blocking indices, loads the cross-map, and replays the
write-ahead log (WAL) to recover any records added since the last
restart. Progress is logged to stderr. Once ready, it prints:

```
meld serve listening on port 8090
```

**Logging.** All log output goes to stderr, so it can be redirected
independently of any stdout output. The default format is human-readable
with timestamps. Control the level with `RUST_LOG`:

```bash
# Default -- info-level messages (startup, shutdown, errors)
meld serve --config config.yaml --port 8090

# Debug -- includes per-request timing, encode/search/score spans
RUST_LOG=melder=debug meld serve --config config.yaml --port 8090

# JSON structured logs (for piping to a log aggregator)
meld serve --config config.yaml --port 8090 --log-format json

# Run in background and tail the log
meld serve --config config.yaml --port 8090 2>serve.log &
tail -f serve.log
```

**Write-ahead log.** Every record addition and cross-map change is
appended to the WAL file (configured via `live.upsert_log`, e.g.
`wal.ndjson`). This is a newline-delimited JSON file -- one event per
line. The WAL exists purely for crash recovery: if the server is killed,
the next startup replays these events to restore state. It is not
intended as an audit log. On clean shutdown the WAL is compacted
(duplicate entries collapsed) and can be inspected with:

```bash
# See recent WAL entries
tail -20 wal.ndjson

# Count events by type
jq -r .type wal.ndjson | sort | uniq -c
```

**Cross-map persistence.** Confirmed matches are held in memory and
flushed to the cross-map csv periodically (every `crossmap_flush_secs`,
default 5 seconds) and on shutdown. The cross-map file is the durable
record of which pairs have been matched.

**Shutdown.** Send Ctrl-C or SIGTERM. Melder will stop accepting new
connections, drain in-flight requests, flush and compact the WAL, save
the cross-map, and persist index caches. No data is lost.

## CLI Reference

All commands accept `--log-format json` for structured log output.
Logs go to stderr; command output goes to stdout.

### `meld validate`

Parse and validate a config file without loading data or running
anything. Catches missing fields, invalid method names, bad threshold
values, and malformed blocking rules. Use this to check a config before
committing to a long batch run.

```bash
meld validate --config config.yaml
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |

### `meld run`

Run batch matching: load both datasets, score every B record against the
A-side pool, and write three output csvs (results, review, unmatched).
The cross-map is updated with auto-matched pairs so that re-running
skips already-resolved records.

```bash
meld run --config config.yaml
meld run --config config.yaml --dry-run
meld run --config config.yaml --limit 500 --verbose
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--dry-run` | | Validate config, load data, print what would be processed, then exit. No matching or output files. |
| `--limit` | | Process only the first N B records. Useful for quick sanity checks on large datasets. |
| `--verbose` | `-v` | Print job metadata, dataset paths, and threshold values at startup. |

### `meld serve`

Start the live-mode HTTP server. Both datasets are loaded into memory,
embedding and blocking indices are built, and the write-ahead log is
replayed for crash recovery. Once ready, the server accepts requests on
the configured port. See the Live Mode API section for endpoint details.

```bash
meld serve --config config.yaml --port 8090
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--port` | `-p` | TCP port to listen on (default: 8080) |

### `meld tune`

Run the full batch pipeline without writing any output files, then print
a diagnostic report: score distribution histogram, per-field statistics
(min/max/mean/median/stddev), threshold analysis showing how the current
thresholds split your records, and suggested threshold values based on
percentiles. Use this to choose weights and thresholds before committing
to a production run.

```bash
meld tune --config config.yaml
meld tune --config config.yaml --verbose
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--verbose` | `-v` | Show current threshold values at startup. |

See [TUNE.md](TUNE.md) for a detailed guide on interpreting the tune
output, a worked example with the benchmark dataset, and the recommended
weight-tuning workflow.

### `meld cache build`

Pre-build embedding index caches for one or both sides. Encodes all
records through the ONNX model and writes the resulting vectors to disk
so that subsequent `meld run` or `meld serve` invocations start
instantly instead of re-encoding. This is especially useful when the
same dataset is matched repeatedly with different configs or thresholds.

```bash
meld cache build --config config.yaml
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |

### `meld cache status`

Show the status of each cache file: whether it exists, its size on
disk, and the number of records it contains (for index files).

```bash
meld cache status --config config.yaml
```

### `meld cache clear`

Delete stale cache files. The default behaviour is *smart*: it computes
the cache filename that the current config expects (derived from a hash
of the embedding field names, order, and weights) and deletes only files
that do **not** match — i.e. files left over from a previous config that
are now unreachable. The current valid cache is left untouched.

Use `--all` to delete everything regardless.

```bash
# Smart clear: delete stale files only (safe to run before any rebuild)
meld cache clear --config config.yaml

# Full wipe: delete all cache files including the current valid ones
meld cache clear --config config.yaml --all
```

| Flag | Description |
|------|-------------|
| `--all` | Delete all cache files, including the current valid cache. Forces a cold rebuild on the next run. |

**When to use `--all`**: after changing the embedding model, or when
you want to reclaim disk space and are happy to re-encode from scratch.

**When the smart default is enough**: after changing field weights,
adding a new match field, or renaming fields. These all change the spec
hash, so the old cache files become unreachable automatically — the
smart clear finds and removes them without touching anything current.

### `meld review list`

Print the review queue as a formatted table. The review csv is produced
by `meld run` and contains borderline pairs that scored between
`review_floor` and `auto_match`. This command reads that file and
displays it with aligned columns for easy scanning.

```bash
meld review list --config config.yaml
```

### `meld review import`

Import human decisions on review pairs. The decisions file is a csv with
columns `a_id`, `b_id`, and `decision` (either `accept` or `reject`).
Accepted pairs are added to the cross-map. Both accepted and rejected
pairs are removed from the review csv, shrinking the queue.

```bash
meld review import --config config.yaml --file decisions.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file (required) |
| `--file` | `-f` | Path to decisions csv (required) |

### `meld crossmap stats`

Show cross-map statistics: total matched pairs, and coverage as a
percentage of both A and B datasets. Loads the datasets to compute
totals.

```bash
meld crossmap stats --config config.yaml
```

### `meld crossmap export`

Export the cross-map to a csv file at a specified path. Useful for
backing up the current state or transferring matches to another system.

```bash
meld crossmap export --config config.yaml --out matches.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--out` | `-o` | Output file path (required) |

### `meld crossmap import`

Import match pairs from a csv file into the cross-map. The csv must have
columns matching the configured `a_id_field` and `b_id_field`. Pairs are
merged with any existing cross-map entries -- duplicates are ignored.

```bash
meld crossmap import --config config.yaml --file pairs.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--file` | `-f` | Input csv file path (required) |

## Live Mode API

Start the server with `meld serve --config config.yaml --port 8090`.
All endpoints are under `/api/v1/`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/a/add` | Add or update an A-side record, return top matches from B |
| POST | `/b/add` | Add or update a B-side record, return top matches from A |
| POST | `/a/add-batch` | Add or update multiple A-side records in one request |
| POST | `/b/add-batch` | Add or update multiple B-side records in one request |
| POST | `/a/match` | Score an A-side record against B without storing it (read-only) |
| POST | `/b/match` | Score a B-side record against A without storing it (read-only) |
| POST | `/a/match-batch` | Score multiple A-side records against B without storing (read-only) |
| POST | `/b/match-batch` | Score multiple B-side records against A without storing (read-only) |
| POST | `/a/remove` | Remove an A-side record from all indices and break any crossmap pair |
| POST | `/b/remove` | Remove a B-side record from all indices and break any crossmap pair |
| POST | `/a/remove-batch` | Remove multiple A-side records in one request |
| POST | `/b/remove-batch` | Remove multiple B-side records in one request |
| GET | `/a/query?id=X` | Look up an A-side record and its crossmap status |
| GET | `/b/query?id=X` | Look up a B-side record and its crossmap status |
| POST | `/crossmap/confirm` | Confirm a match (add to cross-map) |
| POST | `/crossmap/break` | Break a confirmed match (remove from cross-map) |
| GET | `/crossmap/lookup?id=X&side=a` | Look up whether a record has a confirmed match |
| GET | `/crossmap/pairs` | Export all confirmed crossmap pairs (paginated) |
| GET | `/crossmap/stats` | Coverage statistics (matched/unmatched counts per side) |
| GET | `/a/unmatched` | List A-side record IDs with no crossmap pair (paginated) |
| GET | `/b/unmatched` | List B-side record IDs with no crossmap pair (paginated) |
| GET | `/review/list` | List pending review-band matches (paginated) |
| GET | `/health` | Health check |
| GET | `/status` | Detailed server status (record counts, uptime) |

### Adding a record

When you add a record to one side, melder immediately encodes it,
searches the opposite side for matches, and returns the top results.
If the record already exists (same ID), it is updated and re-matched.

Request:

```json
POST /api/v1/a/add

{
  "record": {
    "entity_id": "ENT-001",
    "legal_name": "Acme Corporation",
    "country_code": "US"
  }
}
```

Response:

```json
{
  "id": "ENT-001",
  "status": "added",
  "matches": [
    {
      "id": "CP-042",
      "score": 0.91,
      "classification": "auto",
      "field_scores": [...]
    }
  ]
}
```

The `status` field will be `"added"` for new records or `"updated"` for
existing records that were modified.

### Removing a record

Remove a record by ID. This removes it from all indices (embedding,
blocking, unmatched set) and breaks any existing crossmap pair. The
opposite-side record in a broken pair is returned to the unmatched pool.

Request:

```json
POST /api/v1/a/remove

{
  "id": "ENT-001"
}
```

Response:

```json
{
  "status": "removed",
  "id": "ENT-001",
  "side": "a",
  "crossmap_broken": ["CP-042"]
}
```

The `crossmap_broken` array lists any opposite-side IDs whose pairing
was broken by the removal. It is omitted when empty.

### Batch operations

The batch endpoints accept multiple records in a single request. This
amortises the ONNX encoding cost across the batch -- all texts are
encoded in a single model call, then scored sequentially. Maximum 1000
records per request. Empty arrays return 400.

**Add batch** -- add or update multiple records at once:

```json
POST /api/v1/a/add-batch

{
  "records": [
    {"entity_id": "ENT-001", "legal_name": "Acme Corp", "country_code": "US"},
    {"entity_id": "ENT-002", "legal_name": "Globex Inc", "country_code": "GB"}
  ]
}
```

Response:

```json
{
  "results": [
    {"id": "ENT-001", "status": "added", "matches": [...], ...},
    {"id": "ENT-002", "status": "added", "matches": [...], ...}
  ]
}
```

Each entry in `results` has the same shape as a single `/add` response.
If a record fails (e.g. missing ID field), its entry has
`"status": "error"` with a message -- the rest of the batch still
succeeds.

**Match batch** -- score multiple records without storing them:

```json
POST /api/v1/b/match-batch

{
  "records": [
    {"counterparty_id": "CP-X", "counterparty_name": "Test Corp", "domicile": "US"},
    {"counterparty_id": "CP-Y", "counterparty_name": "Another Inc", "domicile": "GB"}
  ]
}
```

**Remove batch** -- remove multiple records by ID:

```json
POST /api/v1/a/remove-batch

{
  "ids": ["ENT-001", "ENT-002", "NONEXISTENT"]
}
```

Response:

```json
{
  "results": [
    {"id": "ENT-001", "side": "a", "status": "removed"},
    {"id": "ENT-002", "side": "a", "status": "removed"},
    {"id": "NONEXISTENT", "side": "a", "status": "not_found"}
  ]
}
```

Missing IDs produce `"status": "not_found"` entries rather than failing
the request.

**Throughput.** Batch endpoints are faster than sending single requests
in a loop because they amortise ONNX encoding overhead. On the 10K x
10K benchmark dataset:

| Batch size | Throughput (rec/s) | Speedup vs single |
|:----------:|-------------------:|:-----------------:|
| 1          | 221                | 0.9x              |
| 10         | 331                | 1.4x              |
| 50         | 445                | 1.8x              |
| 100        | 319                | 1.3x              |
| 500        | 325                | 1.3x              |

Batch size 50 is the sweet spot -- large enough to amortise encoding,
small enough that per-batch latency stays under 200ms.

### Querying a record

Look up a record by ID to see its full contents and crossmap status
without modifying anything.

```
GET /api/v1/a/query?id=ENT-001
```

Response:

```json
{
  "id": "ENT-001",
  "side": "a",
  "record": {
    "entity_id": "ENT-001",
    "legal_name": "Acme Corporation",
    "country_code": "US"
  },
  "crossmap": {
    "status": "matched",
    "paired_id": "CP-042",
    "paired_record": {
      "counterparty_id": "CP-042",
      "counterparty_name": "ACME Corp"
    }
  }
}
```

If the record is unmatched, `crossmap.status` is `"unmatched"` and the
`paired_id` and `paired_record` fields are omitted. Returns 404 if the
record ID is not found.

### Crossmap pairs

Export all confirmed crossmap pairs. Supports pagination via `offset`
and `limit` query parameters. When omitted, all pairs are returned.

```
GET /api/v1/crossmap/pairs?offset=0&limit=100
```

Response:

```json
{
  "total": 4523,
  "offset": 0,
  "pairs": [
    { "a_id": "ENT-001", "b_id": "CP-042" },
    { "a_id": "ENT-007", "b_id": "CP-119" }
  ]
}
```

`total` is the full count of crossmap pairs (regardless of pagination).
Pairs are sorted by `a_id` for deterministic ordering.

### Crossmap stats

Coverage statistics for the current crossmap. Shows matched and
unmatched counts per side, total crossmap pairs, and coverage
percentages.

```
GET /api/v1/crossmap/stats
```

Response:

```json
{
  "records_a": 10000,
  "records_b": 9500,
  "crossmap_pairs": 4523,
  "matched_a": 4523,
  "matched_b": 4523,
  "unmatched_a": 5477,
  "unmatched_b": 4977,
  "coverage_a": 0.4523,
  "coverage_b": 0.4761
}
```

### Unmatched records

List record IDs that have no crossmap pair on a given side. Supports
pagination and an optional `include_records=true` parameter to return
full record data alongside each ID.

```
GET /api/v1/a/unmatched?offset=0&limit=50
GET /api/v1/b/unmatched?include_records=true&limit=10
```

Response (without `include_records`):

```json
{
  "side": "a",
  "total": 5477,
  "offset": 0,
  "records": [
    { "id": "ENT-003" },
    { "id": "ENT-009" }
  ]
}
```

Response (with `include_records=true`):

```json
{
  "side": "b",
  "total": 4977,
  "offset": 0,
  "records": [
    {
      "id": "CP-055",
      "record": {
        "counterparty_id": "CP-055",
        "counterparty_name": "Initech LLC",
        "domicile": "US"
      }
    }
  ]
}
```

`total` is the full count of unmatched records on that side. Records
are sorted by ID for deterministic ordering.

### Review list

List pending review-band matches -- pairs that scored between
`review_floor` and `auto_match` during upsert but were not
auto-confirmed. This is a read-only endpoint; resolution happens via
the existing `/crossmap/confirm` and `/crossmap/break` endpoints.

```
GET /api/v1/review/list?offset=0&limit=20
```

Response:

```json
{
  "total": 37,
  "offset": 0,
  "reviews": [
    {
      "id": "ENT-012",
      "side": "a",
      "candidate_id": "CP-088",
      "score": 0.74
    },
    {
      "id": "CP-201",
      "side": "b",
      "candidate_id": "ENT-055",
      "score": 0.69
    }
  ]
}
```

`total` is the count of all pending reviews. Reviews are sorted by
score descending (highest-confidence pairs first). Confirming or
breaking a pair via the crossmap endpoints removes it from the review
queue. Re-upserting a record also clears its stale review entries.

### Symmetry

Live mode treats A and B sides identically. Adding, removing, querying,
and matching records works the same way on both sides. Both sides
support the same operations, the same scoring logic, and the same match
semantics.

### Persistence and restart

Live mode is designed to survive restarts. The full state —
records added via the API, confirmed crossmap pairs, and embedding
vectors — is persisted to disk and restored on the next startup.

**What is persisted:**

| Component | Mechanism | When |
|-----------|-----------|------|
| Record mutations (add, remove) | Write-ahead log (WAL) | Every API call |
| Crossmap confirmations/breaks | WAL + crossmap CSV | API call + periodic flush |
| Embedding vectors | Index cache (`.usearchdb` or `.index`) | Shutdown |
| Review queue | WAL (`ReviewMatch` events) | Every API call |

**Shutdown sequence.** When the server receives SIGTERM or Ctrl-C:

1. WAL is flushed and compacted (deduplicates per record ID, last-write-wins)
2. Crossmap CSV is flushed to disk
3. Combined embedding index caches are saved (includes all API-added vectors)

**Startup sequence.** When `meld serve` starts:

1. Dataset files (CSV/JSONL/Parquet) are loaded as the base record set
2. Embedding index caches are loaded from disk (if present and valid)
3. Blocking indices are built from the dataset records
4. Crossmap CSV is loaded
5. All WAL files are replayed in chronological order:
   - Records are inserted/removed from the in-memory store
   - Blocking indices are updated for each replayed record
   - Crossmap confirms/breaks are applied
   - Embedding vectors already in the cached index are skipped
     (no ONNX re-encoding)
6. Unmatched sets and common-ID indices are rebuilt from the final state
7. Review queue is populated from unresolved `ReviewMatch` WAL events
8. A new timestamped WAL file is opened for the current run

**What this means in practice:**

- Records added via `/a/add` or `/b/add` survive restarts. They are
  replayed from the WAL and their embedding vectors are loaded from the
  index cache — no re-encoding required.
- Confirmed crossmap pairs survive via both the crossmap CSV and WAL
  replay (belt and suspenders).
- Blocking works correctly for WAL-replayed records. A new record added
  after restart will find WAL-replayed records on the opposite side as
  match candidates.
- The review queue is rebuilt from WAL events, minus any pairs that were
  subsequently confirmed or broken.
- The base dataset files are never modified. The WAL captures the delta.

**WAL files.** Each server run creates a new timestamped WAL file (e.g.
`wal_20260312T143207Z.ndjson`). On startup, all WAL files matching the
configured base path are discovered and replayed in lexicographic
(chronological) order. Each run's WAL is compacted at shutdown to
deduplicate per-record events. Old WAL files accumulate across runs;
delete them manually if disk space is a concern (only the most recent
compacted file is needed for full recovery).

## Data Formats

melder reads csv (default), JSONL/NDJSON, and Parquet files. The format is
inferred from the file extension or set explicitly in the config:

```yaml
datasets:
  a:
    path: data.jsonl
    format: jsonl   # csv, jsonl, ndjson, parquet
```

Parquet support requires a feature flag at build time:

```bash
cargo build --release --features parquet-format
```

All column types (string, integer, float, boolean) are converted to strings
to match the csv/JSONL behaviour. Snappy-compressed parquet files are
supported.

## Performance

Benchmarked on Apple Silicon, `all-MiniLM-L6-v2` model, `encoder_pool_size: 4`.

### Batch mode — flat vs usearch, top_n: 20

`flat` scans all candidates linearly; `usearch` uses an HNSW
approximate nearest-neighbour (ANN) graph for O(log N) candidate selection.
Cold builds encode from scratch and save to disk; every subsequent run
is warm. Both backends use `country_code` blocking, which limits each B
record to its country block (~5k A records out of 100k).

**Batch mode benchmarks**

10k x 10k means 10,000 records in the A set. 10,000 in B with zero initial crossmappings.
Apple Silicon M3 Macbook Air

| | flat 10k × 10k | usearch 10k × 10k | flat 100k × 100k | usearch 100k × 100k |
|---|---:|---:|---:|---:|
| Index build time (no cache) | ~16s | ~18s | ~3m | ~3m 24s |
| Index load time (pre-existing cache) | ~25ms | ~50ms | ~650ms | ~700ms |
| Scoring throughput | 4,877 rec/s | **22,464 rec/s** | 363 rec/s | **9,886 rec/s** |
| Wall time (cold) | — | — | — | 3m 36s |
| Wall time (warm) | 1.6s | 0.85s | 4m 37s | **12s** |

**Observations**

- the first build of cached indices for large use cases can be slow. vector encoding is compute intenstive. If this is a problem for you use case, set `quantized: true` in the `performance:` section of config (see example config at top). This speeds up encoding 2x.
- Thereafter, pre-built indices on disk are re-used and startup is fast.
- the 'flat' back end index store is file based and has only O(N) performance. Use only for small experiments.
- 'usearch' an in-process vector database has O(logN) performance (ANN lookup) and should be used for most real world and large use cases. Storage of vectors is slightly slower than 'flat'.

### Live mode: 3,000 add requests, 80% requiring encoding

**pre-populated caches (10k x 10k)**

Apple Silicon M3 Macbook Air
`all-MiniLM-L6-v2`, `encoder_pool_size: 4`
`c=1` means one http client submitting 3,000 requests sequentially
`c=10` means ten http clients each submitting 3,000 add requests simultaneously. 30k total.

| Metric | flat (c=1) | usearch (c=1) | flat (c=10) | usearch (c=10) |
|--------|----------:|-------------:|------------:|---------------:|
| Throughput | 349 req/s | 673 req/s | 616 req/s | **1,624 req/s** |
| p50 latency | 2.2ms | 0.7ms | 13.3ms | 3.9ms |
| p95 latency | 4.4ms | 2.9ms | 39.9ms | 21.8ms |
| p99 latency | 9.7ms | 3.3ms | 64.4ms | 35.4ms |

### Live mode: 3,000 add requests, 40% requiring encoding

**pre-populated caches (10k x 10k)**

*as above*

| Metric | flat (c=1) | usearch (c=1) | flat (c=10) | usearch (c=10) |
|--------|----------:|-------------:|------------:|---------------:|
| Throughput | 407 req/s | 928 req/s | 890 req/s | **2,474 req/s** |
| p50 latency | 1.9ms | 0.6ms | 10.1ms | 2.4ms |
| p95 latency | 4.4ms | 3.0ms | 21.0ms | 11.1ms |
| p99 latency | 8.0ms | 3.5ms | 36.5ms | 24.3ms |

Halving the encoding load gives a meaningful speedup on usearch: the
text-hash skip optimisation means non-encoding requests complete in
under 1ms, pulling the average down significantly.


### Live mode: 3,000 add requests, 80% requiring encoding

**pre-populated caches (100k x 100k)**

*as above*

| Metric | flat (c=1) | usearch (c=1) | flat (c=10) | usearch (c=10) |
|--------|----------:|-------------:|------------:|---------------:|
| Throughput | 160 req/s | 546 req/s | 251 req/s | **1,448 req/s** |
| p50 latency | 5.6ms | 0.9ms | 30.5ms | 5.1ms |
| p95 latency | 8.1ms | 3.5ms | 94.4ms | 19.8ms |
| p99 latency | 11.3ms | 3.9ms | 199.7ms | 31.2ms |

At 100k records, usearch is 3.4x faster than flat at c=1 and 5.8x
faster at c=10. The usearch c=10 result includes text-hash skip: the
20% of requests that only modify non-embedding fields skip ONNX
encoding and complete in ~1ms (vs ~7ms for encoding requests).

### Benchmarking

Three Python scripts in `bench/` exercise the live server. All three can
start and stop the server automatically, or connect to one you already
have running (`--no-serve`). They require only the Python standard
library -- no pip dependencies.

**`bench/smoke_test.py`** -- Quick sanity check. Starts the server,
waits for health, sends 10 upsert requests (5 A-side, 5 B-side), prints
each response with latency, and stops. Use this to verify the server
comes up cleanly and returns valid responses before running longer tests.

```bash
python bench/smoke_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml
```

**`bench/live_stress_test.py`** -- Sequential throughput and latency.
Fires N requests one at a time with a realistic operation mix: 30% new A
records, 30% new B records, 20% updates that trigger re-encoding, and
20% updates that change non-embedding fields only. Prints a summary
table with p50/p95/p99/max latency per operation type and overall
throughput.

```bash
python bench/live_stress_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml \
    --iterations 3000
```

**`bench/live_concurrent_test.py`** -- Concurrent throughput and
latency. Same operation mix as the stress test but distributes requests
across N parallel workers. Use this to measure how throughput scales
under load.

```bash
python bench/live_concurrent_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml \
    --iterations 3000 --concurrency 10
```

**`bench/live_batch_test.py`** -- Batch endpoint benchmark. Runs the
same workload through single-record endpoints and then through batch
endpoints, printing a side-by-side comparison. Also tests `match-batch`
and `remove-batch`. Use `--batch-only` to skip the single-record
baseline.

```bash
python bench/live_batch_test.py --binary ./target/release/meld \
    --config testdata/configs/bench_live.yaml \
    --records 3000 --batch-size 50
```

All four scripts accept `--no-serve` to skip starting the server, which
is useful when you want to start it yourself (e.g. with debug logging or
a different config):

```bash
# Terminal 1: start the server manually
meld serve --config testdata/configs/bench_live.yaml --port 8090

# Terminal 2: run the benchmark against it
python bench/live_concurrent_test.py --no-serve --port 8090 --iterations 3000
```

See [LIVE_PERFORMANCE.md](LIVE_PERFORMANCE.md) for full benchmark details
including encoder pool size comparisons and Go baseline comparison.

## How vector caching works

Melder uses a sentence-transformer model (default: `all-MiniLM-L6-v2`) to
convert each record's text fields into dense numeric vectors — fingerprints
that capture meaning rather than characters. Two records about the same
entity produce vectors that point in nearly the same direction, even if the
wording differs completely. This is how `method: embedding` scoring works.

Encoding is expensive: running 10,000 records through the ONNX model takes
around 8 seconds. To avoid repeating this work, melder caches the encoded
vectors to disk after the first run.

### The combined embedding index

Rather than storing a separate vector index for every embedding field,
melder builds a single *combined index* per side. For each record, the
vectors for all embedding fields are scaled by the square root of their
weights and concatenated into one long vector. This combined vector has a
useful property: searching for the nearest combined vectors is exactly
equivalent to finding the records with the highest weighted cosine
similarity across all embedding fields — the same ranking that full scoring
would produce. A single ANN search therefore retrieves the right candidates
without needing multiple per-field lookups.

The cache filename encodes a hash of the field names, their order, and their
weights. Changing any of these produces a different filename, so the old
cache is ignored and a fresh one is built automatically. `meld cache clear`
(without `--all`) uses this same hash to identify and delete only the
now-unreachable stale files.

### Config options

```yaml
embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: cache                 # required
  b_cache_dir: cache                 # optional — omit to skip B-side caching
```

`a_cache_dir` is required — the A-side combined index is always cached to
disk. On first run the directory is created and populated; on subsequent
runs the index loads in milliseconds (~170ms for 100k records).

`b_cache_dir` is optional. When set, the B-side combined index is also
saved to disk. When omitted, B vectors are re-encoded from scratch on
every run.

### Batch mode lifecycle

```
First run:
  A index: check manifest → missing → cold build → save index + manifest + texthash
  B index: same

Subsequent runs (same data, same config):
  A index: manifest fresh → load index + texthash → diff: 0 changed → return (ms)
  B index: same

Subsequent runs (some records changed):
  A index: manifest fresh → load → diff: N records changed → re-encode N → save
  B index: same

Config changed (model / spec / blocking):
  A index: manifest mismatch → log reason → cold build
  B index: same

Scoring (per B record):
  1. Look up B record's combined vector in the B index
  2. Search the A combined index for the top_n nearest neighbours (O(log N)
     with usearch, O(N) with flat)
  3. Score each candidate across all match fields
  4. Classify and output
```

Setting `b_cache_dir` is especially valuable when tuning thresholds or
weights — encoding is done once and the score distribution can be explored
cheaply on subsequent runs.

### Live mode lifecycle

```
Startup:
  A index: load from a_cache_dir (or encode + build if stale)
  B index: load from b_cache_dir (or encode; only saved if configured)
  Both indices live in memory for fast concurrent access.

During operation (upsert / try-match):
  Encode the record's embedding fields into a combined vector.
  Upsert the combined vector into the in-memory index.
  Search the opposite side's index for top_n nearest neighbours.

Shutdown:
  Save combined indices to their cache directories.
```

### Staleness and invalidation

Cache validation is multi-layered and runs automatically on every startup.
No manual intervention is needed for any of the scenarios below.

**Layer 1 — Config hash (manifest check).** A `.manifest` sidecar is
stored alongside each cache file. It records a hash of the embedding
field spec (field names, order, weights), the blocking configuration
(enabled/operator/fields), and the model name. On load, these hashes are
compared against the current config. Any mismatch triggers an immediate
cold rebuild with a clear log message explaining what changed:

```
Warning: A combined index cache invalidated (blocking config changed), rebuilding from scratch.
Warning: B combined index cache invalidated (embedding model changed), rebuilding from scratch.
```

This catches config changes that previously caused silent wrong-result
bugs — for example, adding a new blocking field used to silently reuse an
index that was built without that field.

**Layer 2 — Text-hash deduplication (incremental encoding).** After the
manifest check passes, the engine computes a FNV-1a hash of each record's
source text and compares it against the stored hashes in a `.texthash`
sidecar. Records whose text hasn't changed since the last run are skipped:
their cached vectors are reused as-is. Only records whose text actually
changed (or that are new) are re-encoded through the ONNX model.

This means:
- Recurring batch jobs where most records are stable only re-encode the
  changed minority — avoiding the full encoding cost on every run.
- Datasets that change content but not size (delete one record, insert
  another with the same count) are detected correctly: the changed
  records' hashes differ, so their vectors are updated automatically.

If more than 90% of records change in a single run, a full cold rebuild
is triggered instead (more efficient batching outweighs the incremental
overhead at that point).

**Spec hash → cache filename.** If you change a field's weight, rename a
field, or add/remove an embedding field, the spec hash embedded in the
cache filename changes and the old cache file becomes unreachable. The
engine builds a fresh index automatically; `meld cache clear` (smart mode)
finds and removes the now-unreachable old file along with its sidecars.

**Cache files produced per index:**

| File | Contents |
|------|----------|
| `*.index` | Flat backend: combined vectors (binary). Usearch: key mapping manifest only. |
| `*.usearchdb/` | Usearch backend: HNSW graph files, one per block. |
| `*.index.manifest` | Config hashes, model name, record count, build timestamp. |
| `*.index.texthash` | Per-record FNV-1a hashes of source text. |

`meld cache status` now prints the model, spec hash, blocking hash,
record count, and build timestamp from each manifest:

```
  A cache          bench/cache (1 index files, 52.3 MB)
    model=all-MiniLM-L6-v2 spec=a3f7c2b1 blocking=deadbeef records=100000 built=2026-03-10T14:22:05Z
```

`meld cache clear` and `meld cache clear --all` both delete the sidecars
alongside the index files they belong to.

## Project Structure

```
src/
  main.rs              CLI entry point (clap)
  lib.rs               Module exports
  error.rs             Error types (Config, Data, Encoder, Index, CrossMap, Session)
  models.rs            Core types: Record, Side, MatchResult, Classification
  config/              YAML config loading and validation
  data/                Dataset loaders (csv, JSONL, Parquet)
  encoder/             ONNX encoder pool (fastembed)
  vectordb/            Vector index abstraction (flat + usearch backends), combined index logic
  fuzzy/               Fuzzy string matchers (ratio, partial_ratio, token_sort, wratio)
  scoring/             Scoring dispatch (exact, fuzzy, embedding, numeric)
  matching/            Blocking filter, candidate selection, scoring pipeline
  crossmap/            Bidirectional ID mapping with csv persistence
  batch/               Batch matching engine and output writers
  state/               State management (batch + live), WAL
  session/             Live session logic (add, match, crossmap ops)
  api/                 HTTP handlers and server (axum)
```

## Building

```bash
cargo build --release

# With HNSW approximate nearest-neighbour index (recommended for production)
cargo build --release --features usearch

# With Parquet support
cargo build --release --features parquet-format

# Both features together
cargo build --release --features usearch,parquet-format
```

The ONNX model is downloaded automatically on first run to
`~/.cache/fastembed/` on Linux/macOS or `%LOCALAPPDATA%\fastembed\` on
Windows.

### Environment

- `RUST_LOG=melder=debug` -- enable debug logging
- `RUST_LOG=melder=info` -- default log level
- `--log-format json` -- JSON structured log output (for production)

### Windows

Melder builds and runs on Windows. A few things to be aware of:

**Prerequisites.** Install Rust via [rustup](https://rustup.rs/) which
defaults to the MSVC toolchain on Windows. You will need the
**Visual Studio Build Tools** (or full Visual Studio) with the
"Desktop development with C++" workload installed -- this provides the
MSVC compiler and linker that Rust requires. If you enable the `usearch`
feature, you will also need **CMake** on your PATH (the usearch crate
compiles C++ via a build script).

**Building.** The same `cargo build` commands work. The binary is
produced at `target\release\meld.exe`:

```powershell
cargo build --release --features usearch,parquet-format
.\target\release\meld.exe validate --config config.yaml
```

**Environment variables.** `RUST_LOG` and `RAYON_NUM_THREADS` work the
same way. Set them in PowerShell with `$env:RUST_LOG = "melder=debug"`
or in cmd with `set RUST_LOG=melder=debug`.

**Graceful shutdown.** On Unix, melder listens for both Ctrl-C and
SIGTERM. On Windows, SIGTERM is not available -- use Ctrl-C to trigger
a clean shutdown (in-flight requests drain, WAL is compacted, cross-map
and index caches are saved).

**Unix sockets.** The `--socket` CLI option for `meld serve` is not
available on Windows. Use `--port` instead (the default).

**Config paths.** Both forward slashes and backslashes work in YAML
config file paths (`datasets.a.path`, `output.results_path`, etc.).

## License

MIT (c) Jude Payne 2026
