
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
preloaded. New records can be added to either side at any time, and the melder will immediately find
and return the best matches from the opposite side. A and B sides are treated symmetrically —
both have identical capabilities.</p>
<blockquote><p><strong>Example use case:</strong> You have two master systems with independent
data setup processes, and you wish to sync them in real time. The machinery to create common
identifiers is outside of the melder's remit, but the melder strongly supports being the core of
the end-to-end processing chain.</p></blockquote>
<blockquote><p><strong>Example use case:</strong> You have a master and want to offer a fast
search facility to prevent your users setting up duplicate data.</p></blockquote>
</li>
</ul>

Both modes use the same scoring pipeline, so a match score means the
same thing regardless of how it was produced.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [How Matching Works](#how-matching-works)
  - [The scoring equation](#the-scoring-equation)
  - [What happens under the hood](#what-happens-under-the-hood)
  - [Scoring methods in detail](#scoring-methods-in-detail)
- [Configuration](#configuration)
  - [Live mode config](#live-mode-config)
  - [Performance config](#performance-config)
- [In Operation](#in-operation)
  - [Batch mode](#batch-mode)
  - [Live mode](#live-mode)
  - [Live mode API](#live-mode-api)
- [CLI Reference](#cli-reference)
- [Data Formats](#data-formats)
- [Building](#building)
- [Performance](#performance)
  - [Benchmarking](#benchmarking)
- [How Vector Caching Works](#how-vector-caching-works)
- [Project Structure](#project-structure)
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

## How Matching Works

### The scoring equation

At its core, the melder asks a simple question for every pair of
records: *how similar are they?* The answer is a single number between
0.0 and 1.0 — the **composite score**.

You control how that score is calculated by defining a list of
**match fields** in your config. Each entry pairs a field from dataset A
with a field from dataset B, specifies a comparison method, and assigns
a weight. The composite score is the weighted average of all the
per-field scores:

```
composite = (score_1 × weight_1) + (score_2 × weight_2) + ... + (score_n × weight_n)
```

Here is a concrete example. Suppose you are matching legal entities (A)
against counterparties (B) and you define these match fields:

```yaml
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding          # semantic similarity — understands meaning
    weight: 0.55

  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy              # character-level similarity
    scorer: partial_ratio
    weight: 0.20

  - field_a: country_code
    field_b: domicile
    method: exact              # binary equality
    weight: 0.20

  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05
```

This is your scoring equation. It says: the name comparison matters
most (55%), character-level matching on the short name adds a safety
net (20%), country must agree (20%), and a matching LEI code is a
small bonus (5%). Weights must sum to 1.0.

For a pair where the names are semantically similar (0.92), the short
name partially matches (0.85), the countries agree (1.00), but neither
side has an LEI (0.00), the composite score is:

```
(0.92 × 0.55) + (0.85 × 0.20) + (1.00 × 0.20) + (0.00 × 0.05) = 0.876
```

That single number is then compared against two thresholds you set:

- **auto_match** (e.g. 0.85): at or above this, the pair is confirmed
  automatically.
- **review_floor** (e.g. 0.60): between here and auto_match, the pair
  is flagged for human review. Below this, it is discarded.

This scoring equation is the only thing you need to understand to use
the melder. It is deliberately simple — a weighted average that you can
read, reason about, and fine-tune by adjusting weights and thresholds.

### What happens under the hood

The scoring equation is easy to understand, but evaluating it naively
against every possible pair would be far too slow. If you have 100,000
records on each side, that is 10 billion pair comparisons. The melder
solves this by breaking the work into phases, each one narrowing the
field before the next begins.

**Phase 0 — Common ID fast-path.** If both datasets share a stable
identifier (ISIN, LEI, etc.) and you configure `common_id_field`, the
melder checks for exact ID matches first. Any pair that shares a common
ID is confirmed immediately with score 1.0 — no scoring needed. This
short-circuit runs before anything else.

**Phase 1 — Blocking.** Before any scoring happens, the melder
eliminates records that cannot possibly match using cheap field equality.
For example, if you configure blocking on country code, a record from
the US will never be compared against one from Japan. You can block on
multiple fields at once, combined with AND (all must match) or OR (any
one is enough). This is a simple equality check per field, and it can
eliminate 95%+ of the candidate pool. It costs almost nothing and saves
enormous work downstream.

**Phase 2 — Candidate selection.** Among the records that survived
blocking, there may still be hundreds or thousands. Rather than scoring
every one of them across all match fields, the melder uses an embedding
index to find the `top_n` nearest neighbours in a single fast vector
lookup.

This works because the melder takes the `embedding` terms from your
scoring equation and uses them to build a vector index of all records.
Each record's embedding fields are encoded by a neural language model
into dense numeric vectors — fingerprints that capture meaning rather
than characters. These vectors are combined into a single index per
side, weighted according to your config. A nearest-neighbour search over
this index retrieves the records most likely to score highest under your
full equation, without evaluating every pair.

With the `usearch` backend (HNSW graph index), this search runs in
O(log N) time regardless of pool size. The `flat` backend falls back to
a brute-force O(N) scan.

> [!NOTE]
> If your config has no `embedding` fields, no vector index is built.
> All records that pass blocking go straight to full scoring.

**Phase 3 — Full scoring and classification.** Each of the `top_n`
candidates is now scored across *all* match fields — embedding, fuzzy,
exact, and numeric. The per-field scores are combined into the composite
score using your weights. The composite is compared against your
thresholds and classified as auto-match, review, or no-match.

One subtlety: the per-field embedding scores computed during full
scoring are recovered directly from the combined vectors by slicing them
apart — there is no second pass through the neural model. This is
possible because of a mathematical property of the combined index
construction (the dot product of two combined vectors exactly equals the
weighted sum of per-field cosine similarities).

> [!TIP]
> Use `meld tune` to see the score distribution across your dataset
> before committing to production thresholds. It runs the full pipeline
> without writing output files and prints a histogram with suggested
> threshold values. See [TUNE.md](TUNE.md) for a detailed walkthrough.

### Scoring methods in detail

The melder supports four comparison methods. Each one takes two field
values and returns a score between 0.0 and 1.0.

#### Exact

Binary string equality. Case-insensitive (ASCII fast path, full Unicode
fallback). Returns 1.0 on match, 0.0 otherwise. Both fields empty
returns 0.0 — absence of data is not evidence of a match.

**Best for:** identifiers, codes, and categorical fields where partial
similarity is meaningless — country codes, LEIs, ISINs, currency codes.

**Trade-off:** zero tolerance for typos or formatting differences. Very
fast — single string comparison with no allocation.

#### Fuzzy

Edit-distance-based string similarity, built on normalised Levenshtein
distance. Four scorers are available, selected via the `scorer` config
field:

| Scorer | What it does | Good at |
|--------|-------------|---------|
| `ratio` | Normalised Levenshtein similarity | General string comparison |
| `partial_ratio` | Best substring alignment (sliding window) | Short names within longer strings |
| `token_sort_ratio` | Sort tokens alphabetically, then ratio | Reordered words ("Goldman Sachs" vs "Sachs Goldman") |
| `wratio` (default) | Max of ratio, partial_ratio, token_sort | Robust catch-all when you don't know the error pattern |

All scorers normalise input (lowercase, trim whitespace) before
comparison. A score of 1.0 means identical strings after normalisation.

**Best for:** names, descriptions, and free-text fields where
character-level similarity matters. Use `partial_ratio` when short names
may appear within longer legal names. Use `token_sort_ratio` when word
order varies.

**Trade-off:** `wratio` runs all three sub-scorers and takes the max, so
it costs roughly 3x a single scorer — but still sub-millisecond for
typical entity names. Pure edit distance cannot handle synonyms or
abbreviations ("Corp" vs "Corporation", "JPM" vs "J.P. Morgan") — use
embedding for that.

#### Embedding

Neural sentence embedding with cosine similarity. Each text value is fed
through a transformer model (default: `all-MiniLM-L6-v2`) that converts
it into a dense numeric vector capturing its semantic meaning. Two texts
about the same entity produce vectors pointing in nearly the same
direction, even if the wording differs completely.

Supported models (configured via `embeddings.model`):

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `all-MiniLM-L6-v2` | 384 | Default. Good balance of speed and quality |
| `all-MiniLM-L12-v2` | 384 | Slightly better quality, ~2x slower |
| `bge-small-en-v1.5` | 384 | BAAI family, English-optimised |
| `bge-base-en-v1.5` | 768 | Higher capacity, ~2x more memory |
| `bge-large-en-v1.5` | 1024 | Highest quality, ~4x slower than MiniLM-L6 |

**Best for:** the primary matching field (usually the entity or company
name). Captures semantic similarity that edit distance cannot: "JP Morgan
Chase" vs "JPMorgan" scores high despite low character overlap. Handles
abbreviations, word reordering, and minor language variations.

**Trade-off:** requires a model download (~90MB, auto-downloaded on first
run). Each ONNX inference takes ~1–3ms per text. Each encoder pool slot
uses ~50–100MB of RAM.

> [!NOTE]
> For domain-specific use cases (e.g. counterparty reconciliation),
> general-purpose models can be fine-tuned on your own matched pairs to
> improve accuracy. The melder's own crossmap output is the training data
> source. See [Fine Tuning Embeddings](vault/ideas/Fine%20Tuning%20Embeddings.md) for a guide.

#### Numeric

Numeric equality. Parses both values as floating-point numbers and
returns 1.0 if equal (within machine epsilon), 0.0 otherwise. No range
or tolerance matching — this is currently a stub. Use `exact` for
numeric identifiers. A future version may add tolerance-based comparison.

## Configuration

All behaviour is driven by a single YAML config file. The annotated
example below shows every field with its purpose and default.
Optional fields are marked `optional (default: ...)`. Fields with no
such annotation are required.

```yaml
# =============================================================================
# MELDER — COMPLETE CONFIG REFERENCE
# Every field is shown. Optional fields are marked with their default.
# Fields with no annotation are required.
# =============================================================================

# --- Job metadata ------------------------------------------------------------
# For your reference. Not used by the engine.
job:
  name: counterparty_recon
  description: Match entities to counterparties  # optional (default: "")

# --- Datasets ----------------------------------------------------------------
# Field names in A and B do not need to match — the mapping is in match_fields.
datasets:
  a:
    path: data/entities.csv         # path to CSV, JSONL, or Parquet
    id_field: entity_id             # column whose value is the unique record key
    common_id_field: lei            # optional — stable shared identifier (LEI, ISIN, etc.).
                                    #   Records where both sides share the same value are
                                    #   matched immediately with score 1.0, before any scoring.
                                    #   Must be set on both sides or neither.
    format: csv                     # optional — "csv", "jsonl", or "parquet".
                                    #   Inferred from file extension if omitted.
    encoding: utf-8                 # optional — character encoding for csv/jsonl. Default: utf-8.
  b:
    path: data/counterparties.csv
    id_field: counterparty_id
    common_id_field: lei            # optional — must mirror datasets.a.common_id_field
    format: csv                     # optional
    encoding: utf-8                 # optional

# --- Cross-map ---------------------------------------------------------------
# Persistent record of confirmed A↔B pairs. In batch mode this is a CSV file.
# In live mode with live.db_path it is stored in SQLite instead.
cross_map:
  backend: local                    # optional (default: "local") — the only supported backend
  path: crossmap.csv                # optional (default: "crossmap.csv")
  a_id_field: entity_id             # A-side ID column written into the cross-map output
  b_id_field: counterparty_id       # B-side ID column written into the cross-map output

# --- Embedding model ---------------------------------------------------------
# Used by any match field with method: embedding. Model weights (~90 MB) are
# downloaded automatically from HuggingFace on first run.
embeddings:
  model: all-MiniLM-L6-v2          # HuggingFace model name or local ONNX path.
                                    #   Supported models and their dimensions:
                                    #     all-MiniLM-L6-v2   — 384-dim, fast, good default
                                    #     all-MiniLM-L12-v2  — 384-dim, slightly better, ~2x slower
                                    #     bge-small-en-v1.5  — 384-dim, English-optimised
                                    #     bge-base-en-v1.5   — 768-dim, higher capacity
                                    #     bge-large-en-v1.5  — 1024-dim, highest quality, ~4x slower
  a_cache_dir: cache/a             # directory for the A-side combined embedding index.
                                    #   Created automatically on first run; loaded on subsequent
                                    #   runs to skip re-encoding.
  b_cache_dir: cache/b             # optional — same for B-side. Omit to skip B-side caching
                                    #   (B vectors are rebuilt from scratch on every run).

# --- Vector backend ----------------------------------------------------------
# Controls the embedding index used for candidate selection (Phase 2).
#
#   flat    (default) — brute-force O(N) scan. No extra build dependency.
#                       Good for development and datasets under ~10k records.
#   usearch — HNSW approximate nearest-neighbour graph. O(log N) search.
#             Up to 5× faster at scale. Requires:
#             cargo build --release --features usearch
vector_backend: usearch             # "flat" | "usearch" (default: "flat")

# --- Candidate selection sizes -----------------------------------------------
# Controls the progressive narrowing of candidates before full scoring.
# Required relationship: ann_candidates >= bm25_candidates >= top_n.
top_n: 20                           # Maximum candidates passed to full scoring and maximum
                                    #   results returned per record. Default: 5.
ann_candidates: 200                 # Candidates the ANN (embedding) index retrieves per query.
                                    #   Default: 50. Only used when embedding fields are present.
                                    #   Larger values improve recall at the cost of scoring time.
bm25_candidates: 50                 # Candidates BM25 keeps after re-ranking the ANN shortlist
                                    #   (when both are configured), or retrieves directly from
                                    #   the block (when BM25 is the only filter).
                                    #   Default: 10. Only used when method: bm25 is in match_fields.
                                    #   Requires: cargo build --release --features bm25

# --- Blocking (pre-filter) ---------------------------------------------------
# Before candidate selection, blocking eliminates impossible candidates by
# requiring cheap field equality. A record from France will never be compared
# against one from Japan when blocking on country. Typically eliminates 95%+
# of pairs at almost no cost.
#
# Multiple field pairs can be combined with AND (all must match) or OR (any
# field match is enough). Omit this section or set enabled: false to disable —
# every record then becomes a candidate (thorough but slow on large datasets).
blocking:
  enabled: true                     # optional (default: false)
  operator: "and"                   # "and" (default) — all fields must match (intersection)
                                    # "or"            — any one field matching is enough (union)
  fields:
    - field_a: country_code
      field_b: domicile
    # - field_a: currency           # add more field pairs as needed
    #   field_b: ccy

# --- Match fields (the scoring equation) ------------------------------------
# Defines how similarity is measured. Each entry pairs a field from A with a
# field from B, names a comparison method, and assigns a weight.
# Weights must sum to exactly 1.0 — validated at startup.
#
# Methods:
#   exact     — binary string equality (case-insensitive). Returns 1.0 or 0.0.
#               Best for: identifiers, codes, categoricals (country, currency, ISIN).
#
#   fuzzy     — edit-distance similarity. Select a scorer:
#                 wratio (default)   — max of ratio, partial_ratio, token_sort; robust catch-all
#                 partial_ratio      — best substring alignment; good for short names in long strings
#                 token_sort_ratio   — sort tokens then compare; handles word-order variations
#                 ratio              — normalised Levenshtein; general string comparison
#               Best for: names, free text. Cannot handle synonyms — use embedding for that.
#
#   embedding — neural semantic similarity via the configured model and cosine distance.
#               Understands synonyms, abbreviations, and translations. Builds a vector
#               index at startup; requires a_cache_dir to be set in embeddings.
#               Best for: the primary entity name field.
#
#   numeric   — numeric equality (parses both values as float). Returns 1.0 or 0.0.
#               Use exact for numeric identifiers; this is a stub for now.
#
#   bm25      — IDF-weighted token overlap across all fuzzy/embedding text fields.
#               Do NOT specify field_a or field_b — it operates automatically.
#               Suppresses common-token noise from untrained models (e.g. "Holdings",
#               "International"). Use as a scoring term alongside embedding, or as
#               the sole candidate filter when no embedding fields are configured
#               (fast start, no ONNX model, no vector index).
#               Requires: cargo build --release --features bm25
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding               # semantic similarity — understands meaning
    weight: 0.55

  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy
    scorer: partial_ratio           # wratio | partial_ratio | token_sort_ratio | ratio
    weight: 0.20                    # scorer defaults to wratio if omitted

  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20

  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05

  # BM25 example (uncomment to enable; re-balance weights to sum to 1.0):
  # - method: bm25
  #   weight: 0.10                  # no field_a / field_b — operates across all text fields

# --- Output mapping (optional) -----------------------------------------------
# Copy fields from A-side records into the results CSV under a new column name.
# Useful for enriching output without adding fields to match_fields.
output_mapping:
  - from: sector                    # field name in the A-side record
    to: ref_sector                  # column name in the results/review CSV

# --- Thresholds --------------------------------------------------------------
# auto_match:   pairs scoring >= this are confirmed automatically → results CSV.
# review_floor: pairs scoring between here and auto_match → review CSV for human decision.
#               Pairs scoring below review_floor are discarded.
# Constraint: 0 < review_floor < auto_match <= 1.0
thresholds:
  auto_match: 0.85
  review_floor: 0.60

# --- Output paths (batch mode) -----------------------------------------------
# Paths for the three output CSVs written by meld run.
output:
  results_path: output/results.csv      # confirmed matches (score >= auto_match)
  review_path: output/review.csv        # borderline pairs for human review
  unmatched_path: output/unmatched.csv  # B records with no match above review_floor

# --- Live mode (meld serve) --------------------------------------------------
# Ignored by meld run. Omit this section for pure batch usage.
live:
  upsert_log: wal.ndjson            # optional — path for the write-ahead log. Every record add/remove
                                    #   and crossmap change is appended here for crash recovery.
                                    #   Compacted on clean shutdown.
  crossmap_flush_secs: 5            # optional (default: 5) — how often to flush the in-memory
                                    #   crossmap to disk (seconds). Ignored when using
                                    #   live.db_path (SQLite writes through immediately).
  db_path: data/live.db             # optional — path to a SQLite database for durable live-mode
                                    #   storage. When set:
                                    #   • Cold start (no DB file): datasets loaded from CSV,
                                    #     records inserted into SQLite, crossmap stored in SQLite.
                                    #   • Warm start (DB exists): opens existing DB directly —
                                    #     no WAL replay needed; restarts are instant.
                                    #   Default: not set (in-memory MemoryStore + MemoryCrossMap).
  sqlite_cache_mb: 64               # optional (default: 64) — SQLite page cache size in MB.
                                    #   Controls PRAGMA cache_size. Only relevant when db_path is set.

# --- Performance tuning ------------------------------------------------------
# All fields are optional with sensible defaults. Omit the section if unsure.
performance:
  encoder_pool_size: 4              # Number of concurrent ONNX inference sessions.
                                    #   Each session holds a model copy in RAM.
                                    #   Default: 1. Good starting point: match your core count.
  quantized: true                   # Load the INT8 quantised ONNX model variant.
                                    #   ~2× faster encoding, negligible quality loss.
                                    #   Default: false.
                                    #   Supported: all-MiniLM-L6-v2, all-MiniLM-L12-v2.
                                    #   BGE models have no quantised variant — will error if set.
  encoder_batch_wait_ms: 0          # Live mode only. Collects concurrent encoding requests into
                                    #   a single ONNX batch call for up to this many milliseconds.
                                    #   Default: 0 (disabled — each request encodes independently).
                                    #   Raise to 1–10 only for very high concurrency (c >= 20)
                                    #   with large models; adds latency equal to the window.
                                    #   Has no effect on batch mode.
  vector_quantization: f16          # Storage precision for vectors in the usearch index.
                                    #   f32 (default) — full 32-bit precision, 4 bytes/dimension.
                                    #   f16           — half precision, 2 bytes/dim. ~43% smaller
                                    #                   index and RAM footprint, negligible recall loss.
                                    #   bf16          — brain float 16, same size as f16, slightly
                                    #                   different rounding behaviour.
                                    #   No effect with the flat backend.
                                    #   Changing this invalidates the cache (forces cold rebuild).
  vector_index_mode: load           # How to load the usearch HNSW index from its cache file.
                                    #   "load" (default) — read the full index into RAM. Consistent
                                    #   search latency. Safe for meld run and meld serve.
                                    #   "mmap" — memory-map the file; OS pages in/out on demand.
                                    #   Lower peak RAM at extreme scale (100M+ records) but
                                    #   unpredictable cold-cache latency. READ-ONLY: do not use
                                    #   with meld serve (live upserts write to the index).
                                    #   No effect with the flat backend.
```

### Field reference

**`encoder_pool_size`** — number of ONNX inference sessions to run in
parallel. Each session holds a copy of the model in memory (~50–100 MB
per slot). Higher values increase encoding throughput at the cost of RAM.
4 is a good starting point on machines with 4+ cores; 1 is fine for
small datasets.

**`quantized`** — load the INT8 quantised variant of the ONNX model
instead of the full FP32 model. Roughly doubles encoding speed with
negligible quality loss. Supported for `all-MiniLM-L6-v2` and
`all-MiniLM-L12-v2`; BGE models do not have quantised variants and
will error if set. Expect ~2% of borderline pairs to shift
classification bucket when switching between full-precision and
quantised on the same dataset.

**`vector_quantization`** — controls how the usearch vector backend
stores vectors on disk and in memory. Allowed values:

| Value | Bytes per dimension | Notes |
|-------|--------------------:|-------|
| `f32` | 4 | Full precision. Default. |
| `f16` | 2 | Half precision. ~43% smaller index, negligible recall loss. |
| `bf16` | 2 | Brain float 16. Similar savings to f16, slightly different rounding. |

The primary benefit is **disk cache size**. At 100k records with 384-dim
embeddings, the usearch cache is 171 MB per side with `f32` and 98 MB
per side with `f16` — a 43% reduction. Scoring throughput and match
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

> [!NOTE]
> **`quantized` vs `vector_quantization`** — these are independent
> settings that control different things.
> `quantized` controls the *ONNX encoder model* precision (FP32 vs INT8)
> — it affects how fast text is converted into vectors.
> `vector_quantization` controls the *vector index storage* precision
> (f32/f16/bf16) — it affects how much disk and memory the cached
> index consumes. You can use either, both, or neither.

**`encoder_batch_wait_ms`** — live mode only. When > 0, concurrent
encoding requests are collected for up to this many milliseconds and
dispatched as a single ONNX batch call. This can improve throughput
with large models or very high concurrency (c >= 20), but adds
latency equal to the batch window. With small models (MiniLM) and
`encoder_pool_size >= 4`, leaving this at 0 (disabled) is typically
faster because parallel independent sessions outperform batched
single-session encoding. Has no effect on batch mode.

**`vector_index_mode`** — how the usearch HNSW index is loaded from
its on-disk cache. `"load"` (default) pulls the full graph into RAM,
giving consistent search latency. `"mmap"` memory-maps the file and
lets the OS page-cache decide what stays in RAM — useful at extreme
scale (100M+ records) where the index does not fit in memory, but HNSW
random-access traversal causes frequent page faults on a cold cache, so
latency is unpredictable. Not suitable for `meld serve` because upserts
write to the index.

Batch scoring thread count is controlled by the `RAYON_NUM_THREADS`
environment variable (defaults to logical CPU count if unset).

## In Operation

### Batch mode

> For a hands-on walkthrough, see the
> [batch worked example](examples/batch/README.md).

Batch mode processes an entire B dataset against a pre-indexed A pool in
a single pass. Run it with:

```bash
meld run --config config.yaml
```

When the job completes, the melder writes three output csvs (paths
configured in the `output` section):

| File | Contents |
|------|----------|
| `results.csv` | Confirmed matches — pairs that scored at or above `auto_match`. Columns: `a_id`, `b_id`, `score`, `classification`, plus a per-field score column for each match field. |
| `review.csv` | Borderline pairs — scored between `review_floor` and `auto_match`. Same columns as results. These need a human decision. |
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
  Index build:  14.2s
  Scoring time: 8.3s
  Total elapsed:22.5s
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

Live mode starts an HTTP server that matches records on the fly as they
arrive. It supports two storage backends:

- **In-memory** (default) — records and crossmap held in RAM, persisted
  via a write-ahead log (WAL) and crossmap CSV.
- **SQLite** (set `live.db_path`) — records, crossmap, and review queue
  stored in a SQLite database. Durable by default, instant warm restarts.

SQLite throughput is ~10-15% lower than in-memory at the same scale
(~1,300 vs ~1,530 req/s at 10k, c=10) due to Mutex serialization on the
shared connection. Tail latencies (p95, p99) are actually better with
SQLite. The `live.sqlite_cache_mb` setting controls the SQLite page cache
(default 64 MB) but has negligible impact on throughput — the bottleneck
is lock contention, not I/O.

```bash
meld serve --config config.yaml --port 8090
```

**Startup.** What happens at launch depends on the storage backend:

*In-memory (no `db_path`):* Both datasets are loaded from CSV, embedding
and blocking indices are built, the crossmap CSV is loaded, and the WAL
is replayed to recover any records added since the last restart.

*SQLite — cold start (DB file does not exist):* Datasets are loaded from
CSV and inserted into a new SQLite database. If a crossmap CSV exists,
its pairs are imported into SQLite as a one-time migration. Embedding
and blocking indices are built as normal.

*SQLite — warm start (DB file exists):* The database is opened directly.
No CSV loading, no WAL replay — the DB is the source of truth. Only
embedding index caches are loaded from disk. This is fast.

Once ready, the server prints:

```
meld serve listening on port 8090
```

**Logging.** All log output goes to stderr, so it can be redirected
independently of any stdout output. Control the level with `RUST_LOG`:

```bash
# Default — info-level messages (startup, shutdown, errors)
meld serve --config config.yaml --port 8090

# Debug — includes per-request timing, encode/search/score spans
RUST_LOG=melder=debug meld serve --config config.yaml --port 8090

# JSON structured logs (for piping to a log aggregator)
meld serve --config config.yaml --port 8090 --log-format json

# Run in background and tail the log
meld serve --config config.yaml --port 8090 2>serve.log &
tail -f serve.log
```

**Write-ahead log.** Every record addition and cross-map change is
appended to the WAL file (configured via `live.upsert_log`, e.g.
`wal.ndjson`). This is a newline-delimited JSON file — one event per
line. In in-memory mode, the WAL is essential for crash recovery: if the
server is killed, the next startup replays these events to restore
state. In SQLite mode, the WAL is still written as a redundant safety
net but is not needed for recovery. On clean shutdown the WAL is
compacted (duplicate entries collapsed) and can be inspected with:

```bash
# See recent WAL entries
tail -20 wal.ndjson

# Count events by type
jq -r .type wal.ndjson | sort | uniq -c
```

**Cross-map persistence.** How confirmed matches are persisted depends
on the storage backend:

- *In-memory:* The crossmap is held in RAM and flushed to the crossmap
  CSV periodically (every `crossmap_flush_secs`, default 5 seconds) and
  on shutdown. The CSV is the durable record of which pairs have been
  matched.
- *SQLite:* Every confirm/break is written to the database immediately.
  The crossmap CSV is never updated. Use the `/crossmap/pairs` API
  endpoint or query the `crossmap` table in the SQLite DB directly to
  export pairs.

**Shutdown.** Send Ctrl-C or SIGTERM. The melder will stop accepting new
connections, drain in-flight requests, flush and compact the WAL, save
the cross-map (in-memory mode) or no-op (SQLite mode), and persist
index caches. No data is lost.

#### Persistence and restart

Live mode is designed to survive restarts. The full state — records
added via the API, confirmed crossmap pairs, and embedding vectors — is
persisted to disk and restored on the next startup.

##### In-memory mode (default)

**What is persisted:**

| Component | Mechanism | When |
|-----------|-----------|------|
| Record mutations (add, remove) | Write-ahead log (WAL) | Every API call |
| Crossmap confirmations/breaks | WAL + crossmap CSV | API call + periodic flush |
| Embedding vectors | Index cache (`.usearchdb` or `.index`) | Shutdown |
| Review queue | WAL (`ReviewMatch` events) | Every API call |

**Shutdown sequence:**

1. WAL is flushed and compacted (deduplicates per record ID, last-write-wins)
2. Crossmap CSV is flushed to disk
3. Combined embedding index caches are saved (includes all API-added vectors)

**Startup sequence:**

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
(chronological) order. Each run's WAL is compacted at shutdown. Old WAL
files accumulate across runs; delete them manually if disk space is a
concern (only the most recent compacted file is needed for full
recovery).

##### SQLite mode (`live.db_path` set)

**What is persisted:**

| Component | Mechanism | When |
|-----------|-----------|------|
| Records | SQLite `records` table | Immediately on every add/remove |
| Crossmap pairs | SQLite `crossmap` table | Immediately on every confirm/break |
| Review queue | SQLite `reviews` table | Immediately on every review-band match |
| Embedding vectors | Index cache (`.usearchdb` or `.index`) | Shutdown |
| WAL | Same as in-memory mode | Every API call (redundant safety net) |

**Shutdown sequence:**

1. WAL is flushed and compacted
2. Combined embedding index caches are saved
3. (No crossmap CSV flush — SQLite is already durable)

**Startup sequence (warm — DB exists):**

1. SQLite database is opened directly — records, crossmap, and reviews
   are already there
2. Embedding index caches are loaded from disk
3. Blocking indices are rebuilt from the SQLite records
4. A new WAL file is opened

No CSV loading. No WAL replay. Restarts are fast.

**Startup sequence (cold — no DB file):**

1. A new SQLite database is created
2. Dataset files (CSV/JSONL/Parquet) are loaded and inserted into SQLite
3. If a crossmap CSV exists at `cross_map.path`, its pairs are imported
   into SQLite (one-time migration — the CSV is not updated afterwards)
4. Embedding indices are built or loaded from cache
5. Blocking indices are built
6. A new WAL file is opened

**Migrating from in-memory to SQLite:** Add `live.db_path` to your
config and restart. The first startup is a cold start — datasets are
loaded from CSV, the crossmap CSV is imported into the database, and the
WAL is written as a redundant log. From the second startup onwards, the
database is the sole source of truth and restarts are instant. The
crossmap CSV is never written to again.

### Live mode API

All endpoints are under `/api/v1/`. The server is started with
`meld serve --config config.yaml --port 8090`.

#### Endpoints

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

> [!IMPORTANT]
> Live mode treats A and B sides identically. Adding, removing, querying,
> and matching records works the same way on both sides — same operations,
> same scoring logic, same match semantics.

#### Adding a record

When you add a record to one side, the melder immediately encodes it,
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

#### Removing a record

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

#### Batch operations

The batch endpoints accept multiple records in a single request. This
amortises the ONNX encoding cost across the batch — all texts are
encoded in a single model call, then scored sequentially. Maximum 1000
records per request. Empty arrays return 400.

**Add batch** — add or update multiple records at once:

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
`"status": "error"` with a message — the rest of the batch still
succeeds.

**Match batch** — score multiple records without storing them:

```json
POST /api/v1/b/match-batch

{
  "records": [
    {"counterparty_id": "CP-X", "counterparty_name": "Test Corp", "domicile": "US"},
    {"counterparty_id": "CP-Y", "counterparty_name": "Another Inc", "domicile": "GB"}
  ]
}
```

**Remove batch** — remove multiple records by ID:

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

Batch size 50 is the sweet spot — large enough to amortise encoding,
small enough that per-batch latency stays under 200ms.

#### Querying a record

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

#### Crossmap operations

**Export pairs** — list all confirmed crossmap pairs, with optional
pagination:

```
GET /api/v1/crossmap/pairs?offset=0&limit=100
```

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

**Coverage statistics:**

```
GET /api/v1/crossmap/stats
```

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

#### Unmatched records

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

#### Review list

List pending review-band matches — pairs that scored between
`review_floor` and `auto_match` during upsert but were not
auto-confirmed. Resolution happens via the `/crossmap/confirm` and
`/crossmap/break` endpoints.

```
GET /api/v1/review/list?offset=0&limit=20
```

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

Reviews are sorted by score descending (highest-confidence pairs first).
Confirming or breaking a pair removes it from the review queue.
Re-upserting a record also clears its stale review entries.

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
the configured port. See [Live mode API](#live-mode-api) for endpoint
details.

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
merged with any existing cross-map entries — duplicates are ignored.

```bash
meld crossmap import --config config.yaml --file pairs.csv
```

| Flag | Short | Description |
|------|-------|-------------|
| `--file` | `-f` | Input csv file path (required) |

## Data Formats

The melder reads CSV (default), JSONL/NDJSON, and Parquet files. The
format is inferred from the file extension or can be set explicitly in
the config:

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

All column types (string, integer, float, boolean) are converted to
strings internally to provide a uniform interface across formats.
Snappy-compressed Parquet files are supported.

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

> [!TIP]
> On macOS and Linux, always build with `--features usearch`. The
> usearch backend uses an HNSW graph index for O(log N) candidate
> search instead of the flat backend's O(N) brute-force scan. At
> 100k records this is the difference between a 12-second warm run and
> a 4-minute one — see [Performance](#performance) for full numbers.
>
> On Windows, `usearch` currently has a known MSVC build bug (AVX-512
> FP16 intrinsics and a missing POSIX constant) and must be omitted.
> The flat backend works correctly — just slower at scale.

The ONNX model is downloaded automatically on first run to
`~/.cache/fastembed/` on Linux/macOS or `%LOCALAPPDATA%\fastembed\` on
Windows.

### Environment

- `RUST_LOG=melder=debug` — enable debug logging
- `RUST_LOG=melder=info` — default log level
- `--log-format json` — JSON structured log output (for production)

### Windows

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

## Performance

Benchmarked on Apple M3 MacBook Air, `all-MiniLM-L6-v2` model,
`encoder_pool_size: 4`.

### Batch mode

`flat` scans all candidates linearly; `usearch` uses an HNSW
approximate nearest-neighbour (ANN) graph for O(log N) candidate
selection. Cold builds encode vectors from scratch and save them to
disk; every subsequent run is warm. Both backends use `country_code`
blocking.

10k x 10k means 10,000 records on each side with zero initial
crossmappings.

| | flat 10k x 10k | usearch 10k x 10k | flat 100k x 100k | usearch 100k x 100k |
|---|---:|---:|---:|---:|
| Index build time (no cache) | ~17s | ~17s | ~3m | ~3m 32s |
| Index load time (cached) | ~47ms | ~78ms | ~650ms | ~640ms |
| Scoring throughput | 5,507 rec/s | **31,366 rec/s** | — | **8,735 rec/s** |
| Wall time (cold) | — | — | — | 3m 32s |
| Wall time (warm) | 2.2s | 0.7s | — | **12.8s** |

> [!TIP]
> The first build of cached indices for large datasets can be slow —
> vector encoding is compute-intensive. If this is a problem, set
> `quantized: true` in the `performance` section to roughly double
> encoding speed. Thereafter, pre-built indices on disk are reused
> and startup is fast.

- The `flat` backend is file-based with O(N) search performance. Use
  only for small experiments and development.
- The `usearch` backend is an in-process HNSW vector database with
  O(log N) search. Use for any real-world workload.

### Live mode

Pre-populated caches (10k x 10k). `c=1` means one HTTP client
submitting 3,000 requests sequentially. `c=10` means ten concurrent
clients each submitting 3,000 requests (30k total). 80% of requests
require ONNX encoding; 20% modify only non-embedding fields and skip
the model entirely.

| Metric | flat (c=10) cold | usearch (c=10) cold | flat (c=10) warm | usearch (c=10) warm |
|--------|----------------:|--------------------:|-----------------:|--------------------:|
| Throughput | 843 req/s | 1,045 req/s | 1,113 req/s | **1,558 req/s** |
| p50 latency | 8.4ms | 5.5ms | 7.2ms | 3.5ms |
| p95 latency | 30.4ms | 29.0ms | 21.2ms | 25.6ms |

Cold = fresh index build on startup. Warm = pre-built cache loaded from disk (~1.7s startup vs ~18s cold).

At 100k x 100k (80% encoding, c=10, 10k events), usearch reaches **1,325 req/s** warm
with p50 latency of 6.0ms and p95 of 19.0ms.

When fewer requests require encoding (40% instead of 80%), throughput
improves further: the text-hash skip optimisation means non-encoding
requests complete in under 1ms.

| Metric (40% encoding) | flat (c=10) | usearch (c=10) |
|--------|------------:|---------------:|
| Throughput | 890 req/s | **2,474 req/s** |
| p50 latency | 10.1ms | 2.4ms |
| p95 latency | 21.0ms | 11.1ms |

### Benchmarking

Each benchmark is a self-contained directory with its own `config.yaml` and
`run_test.py`. All scripts require only the Python standard library — no pip
dependencies.

#### Running individual tests

```bash
# Single batch test — run from the project root
python3 benchmarks/batch/10kx10k_usearch/cold/run_test.py
python3 benchmarks/batch/10kx10k_usearch/warm/run_test.py

# Single live test
python3 benchmarks/live/10kx10k_inject3k_usearch/cold/run_test.py
python3 benchmarks/live/100kx100k_inject10k_usearch/warm/run_test.py
```

Cold tests wipe their cache and rebuild from scratch. Warm tests preserve the
cache — run them twice if the cache is empty: the first run builds it, the
second is the true warm measurement.

#### Running the full suite

> [!WARNING]
> A full suite run takes a long time. The 100k cold tests alone encode
> 200,000 records through the ONNX model (~3.5 minutes each). Expect
> **45–60 minutes** for all batch tests and **60–90 minutes** for all live
> tests on Apple Silicon. Budget accordingly.

```bash
# All batch benchmarks (cold then warm for each size/backend)
python3 benchmarks/batch/run_all_tests.py

# All live benchmarks (cold then warm for each size/backend)
python3 benchmarks/live/run_all_tests.py
```

Both scripts stream each test's output to the terminal as it runs, then print
a summary table at the end. Because cold tests build the embedding cache, the
immediately following warm test needs only one pass — the cache is already hot.

#### Helper scripts

Four scripts in `benchmarks/scripts/` exercise the live server directly and
can start/stop it automatically or connect to one you already have running
(`--no-serve`):

**`benchmarks/scripts/smoke_test.py`** — Quick sanity check. Starts the server,
sends 10 upsert requests, prints each response with latency, and stops.
Use this to verify the server comes up cleanly before running longer tests.

```bash
python3 benchmarks/scripts/smoke_test.py --binary ./target/release/meld \
    --config benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml
```

**`benchmarks/scripts/live_stress_test.py`** — Sequential throughput and latency.
Fires N requests one at a time with a realistic operation mix (30% new
A, 30% new B, 20% embedding updates, 20% non-embedding updates). Prints
p50/p95/p99/max latency per operation type and overall throughput.

```bash
python3 benchmarks/scripts/live_stress_test.py --binary ./target/release/meld \
    --config benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml \
    --iterations 3000
```

**`benchmarks/scripts/live_concurrent_test.py`** — Concurrent throughput. Same
operation mix but distributed across N parallel workers. Use this to
measure how throughput scales under load.

```bash
python3 benchmarks/scripts/live_concurrent_test.py --binary ./target/release/meld \
    --config benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml \
    --iterations 3000 --concurrency 10
```

**`benchmarks/scripts/live_batch_test.py`** — Batch endpoint benchmark. Runs the
same workload through single-record and batch endpoints, printing a
side-by-side comparison. Use `--batch-only` to skip the single-record
baseline.

```bash
python3 benchmarks/scripts/live_batch_test.py --binary ./target/release/meld \
    --config benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml \
    --records 3000 --batch-size 50
```

All four scripts accept `--no-serve` to skip starting the server:

```bash
# Terminal 1: start the server manually
meld serve --config benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml --port 8090

# Terminal 2: run the benchmark against it
python3 benchmarks/scripts/live_concurrent_test.py --no-serve --port 8090 --iterations 3000
```

## How Vector Caching Works

The melder uses a sentence-transformer model (default:
`all-MiniLM-L6-v2`) to convert each record's text fields into dense
numeric vectors — fingerprints that capture meaning rather than
characters. Two records about the same entity produce vectors that point
in nearly the same direction, even if the wording differs completely.
This is how `method: embedding` scoring works.

Encoding is expensive: running 10,000 records through the ONNX model
takes around 8 seconds. To avoid repeating this work, the melder caches
the encoded vectors to disk after the first run.

### The combined embedding index

Rather than storing a separate vector index for every embedding field,
the melder builds a single *combined index* per side. For each record,
the vectors for all embedding fields are scaled by the square root of
their weights and concatenated into one long vector. This combined
vector has a useful property: searching for the nearest combined vectors
is exactly equivalent to finding the records with the highest weighted
cosine similarity across all embedding fields — the same ranking that
full scoring would produce. A single nearest-neighbour search therefore
retrieves the right candidates without needing multiple per-field
lookups.

The cache filename encodes a hash of the field names, their order, and
their weights. Changing any of these produces a different filename, so
the old cache is ignored and a fresh one is built automatically.
`meld cache clear` (without `--all`) uses this same hash to identify
and delete only the now-unreachable stale files.

### Config options

```yaml
embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: cache                 # required
  b_cache_dir: cache                 # optional — omit to skip B-side caching
```

`a_cache_dir` is required — the A-side combined index is always cached
to disk. On first run the directory is created and populated; on
subsequent runs the index loads in milliseconds (~170ms for 100k
records).

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
weights — encoding is done once and the score distribution can be
explored cheaply on subsequent runs.

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

Cache validation is multi-layered and runs automatically on every
startup. No manual intervention is needed.

**Layer 1 — Config hash (manifest check).** A `.manifest` sidecar is
stored alongside each cache file. It records a hash of the embedding
field spec (field names, order, weights), the blocking configuration,
and the model name. On load, these hashes are compared against the
current config. Any mismatch triggers an immediate cold rebuild with a
clear log message explaining what changed:

```
Warning: A combined index cache invalidated (blocking config changed), rebuilding from scratch.
Warning: B combined index cache invalidated (embedding model changed), rebuilding from scratch.
```

**Layer 2 — Text-hash deduplication (incremental encoding).** After the
manifest check passes, the engine computes a FNV-1a hash of each
record's source text and compares it against the stored hashes in a
`.texthash` sidecar. Records whose text has not changed are skipped —
their cached vectors are reused. Only records whose text actually
changed (or that are new) are re-encoded through the ONNX model.

This means recurring batch jobs where most records are stable only
re-encode the changed minority. If more than 90% of records change in a
single run, a full cold rebuild is triggered instead (more efficient
batching outweighs the incremental overhead at that point).

**Spec hash in the filename.** If you change a field's weight, rename a
field, or add/remove an embedding field, the spec hash embedded in the
cache filename changes and the old cache file becomes unreachable. The
engine builds a fresh index automatically; `meld cache clear` (smart
mode) finds and removes the now-unreachable old file along with its
sidecars.

**Cache files produced per index:**

| File | Contents |
|------|----------|
| `*.index` | Flat backend: combined vectors (binary). Usearch: key mapping manifest only. |
| `*.usearchdb/` | Usearch backend: HNSW graph files, one per block. |
| `*.index.manifest` | Config hashes, model name, record count, build timestamp. |
| `*.index.texthash` | Per-record FNV-1a hashes of source text. |

`meld cache status` prints the model, spec hash, blocking hash,
record count, and build timestamp from each manifest:

```
  A cache          benchmarks/batch/100kx100k_usearch/warm/cache (1 index files, 52.3 MB)
    model=all-MiniLM-L6-v2 spec=a3f7c2b1 blocking=deadbeef records=100000 built=2026-03-10T14:22:05Z
```

`meld cache clear` and `meld cache clear --all` both delete the
sidecars alongside the index files they belong to.

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

## License

MIT (c) Jude Payne 2026
