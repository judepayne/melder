← [Back to Index](./) | [Scoring Methods](scoring.md) | [CLI Reference](cli-reference.md)

# Configuration Reference

All behaviour is driven by a single YAML config file. The annotated example below shows every field with its purpose and default. Optional fields are marked `optional (default: ...)`. Fields with no such annotation are required.

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

# --- Exclusions --------------------------------------------------------------
# Known non-matching pairs to exclude from scoring. Pairs can be loaded from
# CSV at startup and added/removed at runtime via the API. If an excluded pair
# is currently matched, the match is broken automatically. The CSV file is
# updated on shutdown with any runtime changes.
exclusions:
  path: exclusions.csv              # optional — omit section or path to disable
  a_id_field: entity_id             # column name for A-side IDs
  b_id_field: counterparty_id       # column name for B-side IDs

# --- Embedding model ---------------------------------------------------------
# Used by any match field with method: embedding. Model weights (~90 MB) are
# downloaded automatically from HuggingFace on first run.
embeddings:
  model: all-MiniLM-L6-v2          # Model name, local path, or "builtin".
                                    #   Resolution order:
                                    #     "builtin"          — use the model compiled into the
                                    #                          binary (requires --features builtin-model)
                                    #     Local path          — ./models/my-model or /abs/path/model.onnx
                                    #     user/repo           — downloaded from HuggingFace Hub
                                    #     Named model         — fastembed built-in:
                                    #       all-MiniLM-L6-v2   — 384-dim, fast, good default
                                    #       all-MiniLM-L12-v2  — 384-dim, slightly better, ~2x slower
                                    #       bge-small-en-v1.5  — 384-dim, English-optimised
                                    #       bge-base-en-v1.5   — 768-dim, higher capacity
                                    #       bge-large-en-v1.5  — 1024-dim, highest quality, ~4x slower
  a_cache_dir: cache/a             # directory for the A-side combined embedding index.
                                    #   Created automatically on first run; loaded on subsequent
                                    #   runs to skip re-encoding.
  b_cache_dir: cache/b             # optional — same for B-side. Omit to skip B-side caching
                                    #   (B vectors are rebuilt from scratch on every run).

# --- Vector backend ----------------------------------------------------------
# Controls the embedding index used for candidate selection (Phase 2).
#
#   flat    — brute-force O(N) scan. Pure Rust, no C++ dependency.
#             Good for development and datasets under ~10k records.
#   usearch (default) — HNSW approximate nearest-neighbour graph. O(log N)
#             search. Up to 5× faster at scale. Included by default; opt out
#             with --no-default-features if you need a pure-Rust build.
vector_backend: usearch             # "flat" | "usearch" (default: "usearch")

# --- BM25 fields (optional) ---------------------------------------------------
# Which text fields to index for BM25 scoring. Preferred: define fields inline
# on the method: bm25 entry in match_fields (see below). Alternatively, set
# them here as a top-level section. Cannot use both (error if both present).
#
# When neither inline nor top-level fields are set, they are derived
# automatically from fuzzy/embedding match_fields entries.
#
# bm25_fields:
#   - field_a: legal_name
#     field_b: counterparty_name

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
bm25_commit_batch_size: 1           # optional (default: 1) — how many BM25 upserts to buffer
                                    #   before committing the Tantivy index. Each commit is
                                    #   expensive (~2-5ms): segment finalization, FST construction,
                                    #   garbage collection. Batching amortizes this.
                                    #   1 = commit every upsert (maximum accuracy).
                                    #   50-200 = recommended for high-throughput live workloads.
                                    #   See "BM25 commit batching" in docs/performance.md.

# --- Exact prefilter (pre-blocking exact match) ------------------------------
# Confirms pairs where ALL configured field pairs match exactly before
# blocking or scoring runs. AND semantics: every pair must match (non-empty)
# for an immediate auto-confirm at score 1.0.
#
# Runs before blocking — recovers pairs that blocking would miss due to
# mismatched blocking keys (e.g. wrong country code but matching LEI).
# Extremely fast: O(1) hash lookup per B record (MemoryStore) or indexed
# SQL query (SqliteStore).
#
# Use for globally unique identifiers only (LEI, ISIN, national ID etc.).
# A match on "Limited" or a country code is not a unique identifier.
# Omit this section or set enabled: false to skip.
exact_prefilter:
  enabled: true
  fields:
    - field_a: lei                    # field name in dataset A
      field_b: lei_code               # corresponding field name in dataset B
    # - field_a: isin                 # add more pairs as needed — ALL must match
    #   field_b: isin_code

# --- Blocking (pre-filter) ---------------------------------------------------
# Before candidate selection, blocking eliminates impossible candidates by
# requiring cheap field equality. A record from France will never be compared
# against one from Japan when blocking on country. Typically eliminates 95%+
# of pairs at almost no cost.
#
# Multiple field pairs are combined with AND (all must match).
# Omit this section or set enabled: false to disable — every record then
# becomes a candidate (thorough but slow on large datasets).
blocking:
  enabled: true                     # optional (default: false)
  fields:
    - field_a: country_code
      field_b: domicile
    # - field_a: currency           # add more field pairs as needed
    #   field_b: ccy

# --- Match fields (the scoring equation) ------------------------------------
# Defines how similarity is measured. Each entry pairs a field from A with a
# field from B, names a comparison method, and assigns a weight.
# Weights must sum to exactly 1.0 — validated at startup. Exception: synonym
# weights are additive and excluded from the sum-to-1.0 check.
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
#   bm25      — IDF-weighted token overlap across indexed text fields.
#               Specify which fields to index via the inline `fields` key (preferred)
#               or the top-level `bm25_fields` section (legacy). When neither is set,
#               fields are derived from fuzzy/embedding entries automatically.
#               Suppresses common-token noise from untrained models (e.g. "Holdings",
#               "International"). Use as a scoring term alongside embedding, or as
#               the sole candidate filter when no embedding fields are configured
#               (fast start, no ONNX model, no vector index).
#
#   synonym   — acronym/abbreviation matching. Binary 1.0/0.0. Builds a bidirectional
#               index of generated acronyms at startup. Weight is ADDITIVE — not
#               included in the sum-to-1.0 check. See docs/scoring.md for details.
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
  #   weight: 0.10
  #   fields:                        # which text fields BM25 indexes
  #     - field_a: legal_name
  #       field_b: counterparty_name
  #     - field_a: short_name
  #       field_b: counterparty_name

  # Synonym example (uncomment to enable; weight is additive, not counted in 1.0):
  # - field_a: legal_name
  #   field_b: counterparty_name
  #   method: synonym
  #   weight: 0.20                  # flat +0.20 bonus when acronym match found

# --- Synonym dictionary (optional) -------------------------------------------
# Path to a CSV file of user-provided synonym equivalences. Each row lists
# 2+ terms that should be treated as synonyms. Supplements the auto-generated
# acronym index — useful for equivalences that cannot be derived algorithmically.
# See docs/scoring.md#synonym-dictionary for format details.
#
# synonym_dictionary:
#   path: data/synonyms.csv

# --- Output mapping (optional) -----------------------------------------------
# Copy fields from A-side records into the results CSV under a new column name.
# Useful for enriching output without adding fields to match_fields.
output_mapping:
  - from: sector                    # field name in the A-side record
    to: ref_sector                  # column name in the results/review CSV

# --- Thresholds --------------------------------------------------------------
# auto_match:    pairs scoring >= this are confirmed automatically → results CSV.
# review_floor:  pairs scoring between here and auto_match → review CSV for human decision.
#                Pairs scoring below review_floor are discarded.
# min_score_gap: (optional) minimum margin between rank-1 and rank-2 candidate
#                required to auto-confirm. When set, a top candidate that clears
#                auto_match but whose lead over rank-2 is less than min_score_gap
#                is downgraded to review instead of auto-confirmed. Single-
#                candidate results are never downgraded. Omit to disable (default).
# Constraints: 0 < review_floor < auto_match <= 1.0
#              0.0 <= min_score_gap < 1.0  (if set)
thresholds:
  auto_match: 0.85
  review_floor: 0.60
  # min_score_gap: 0.10   # uncomment to enable confidence-margin check

# --- Output paths (batch mode) -----------------------------------------------
# Paths for the three output CSVs written by meld run.
output:
  results_path: output/results.csv      # confirmed matches (score >= auto_match)
  review_path: output/review.csv        # borderline pairs for human review
  unmatched_path: output/unmatched.csv  # B records with no match above review_floor

# --- Batch mode SQLite (meld run) ---------------------------------------------
# When batch.db_path is set, meld run stores records in SQLite instead of
# in-memory DashMap. Use this for datasets that exceed available RAM.
# The database is created fresh each run and deleted on completion.
# Omit this section (or leave db_path unset) for the default in-memory path.
batch:
  db_path: batch.db                 # optional — triggers SQLite batch mode.
                                    #   The file is created fresh and deleted after the run.
                                    #   Default: not set (in-memory storage).
  sqlite_read_pool_size: 8          # optional — read connection pool size.
                                    #   Default: num_cpus (matches Rayon parallelism).
  sqlite_pool_worker_cache_mb: 128  # optional — page cache per read connection in MB.
                                    #   Total read cache = pool_size × this value.
                                    #   Default: 128.
  sqlite_cache_mb: 64               # optional — write connection page cache in MB.
                                    #   Default: 64.

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
  sqlite_cache_mb: 64               # optional (default: 64) — SQLite page cache for the write
                                    #   connection in MB. Only relevant when db_path is set.
  sqlite_read_pool_size: 4          # optional (default: 4) — number of read-only SQLite connections.
                                    #   Concurrent reads (get, blocking_query, etc.) are served from
                                    #   this pool. Higher values reduce lock contention under load.
  sqlite_pool_worker_cache_mb: 128  # optional (default: 128) — page cache per read connection in MB.
                                    #   Total read cache memory = pool_size × this value.
  skip_initial_match: false         # optional (default: false) — when false, all unmatched B records
                                    #   are scored against A at startup before the API begins listening.
                                    #   Set to true to skip this pass and start accepting requests
                                    #   immediately. Useful for large datasets where you want the API
                                    #   up fast and will trigger matching selectively via the API.

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
  expansion_search: 0               # optional (default: 0) — HNSW search beam width (ef parameter)
                                    #   for the usearch index. Controls how many graph nodes are
                                    #   explored when searching for the top-k nearest neighbours.
                                    #   Higher values improve recall at the cost of slower search.
                                    #   0 = usearch default. Typical range: 16-256.
                                    #   Must be >= ann_candidates. No effect with the flat backend.
  encoder_device: gpu               # optional (default: not set = CPU) — ONNX encoding device.
                                    #   "cpu"  — use CPU for encoding (default behaviour).
                                    #   "gpu"  — use CoreML (macOS) or CUDA (Linux).
                                    #   Requires building with --features gpu-encode.
                                    #   Batch mode only — ignored in live mode with a warning.
  encoder_batch_size: 256           # optional — number of texts per ONNX inference call.
                                    #   Default: 64 (CPU), 256 (GPU).
                                    #   GPU benefits from larger batches to amortise kernel launch
                                    #   overhead. Values above 256 risk GPU memory pressure at
                                    #   higher pool sizes.
                                    #   Batch mode only.
```

### Performance field reference

**`encoder_pool_size`** — number of ONNX inference sessions to run in
parallel. Each session holds a copy of the model in memory (~50–100 MB
per slot). Higher values increase encoding throughput at the cost of RAM.
4 is a good starting point on machines with 4+ cores; 1 is fine for
small datasets.

> [!TIP]
> **Tuning `encoder_pool_size` on multi-core machines.** Each ONNX
> session internally spawns its own thread pool for intra-op parallelism.
> By default ONNX Runtime sets the number of intra-op threads to the
> total number of physical cores. With `encoder_pool_size: 4` on a
> 20-core machine, that means 4 sessions × 20 threads = 80 logical
> threads competing for 20 cores — significant contention.
>
> For best throughput, aim for `pool_size × intra_op_threads ≈ physical
> cores`. You can control the per-session thread count with the
> **`ORT_NUM_THREADS`** environment variable:
>
> ```bash
> # 10 sessions × 2 threads = 20 threads on a 20-core machine
> ORT_NUM_THREADS=2 meld serve --config config.yaml --port 8080
> ```
>
> As a rule of thumb: raise `encoder_pool_size` to match your HTTP
> concurrency (or close to it), and lower `ORT_NUM_THREADS` so the
> product stays near your physical core count. This eliminates both
> mutex contention (requests waiting for a free encoder slot) and CPU
> contention (too many intra-op threads fighting for cores).

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

**`expansion_search`** — HNSW search beam width (`ef` parameter) for
the usearch backend. Controls how many graph nodes are explored when
searching for the top-k nearest neighbours. Higher values improve recall
(fewer missed true matches) at the cost of slower search. The default
(`0`) delegates to usearch's built-in default. Typical values range from
16 to 256 and must be >= `ann_candidates`. At small dataset scales
(10k-50k) the HNSW graph is compact enough that the default works well.
At 1M+ records, tuning this upward can meaningfully improve recall. No
effect with the `flat` backend. Changing this value does **not**
invalidate caches — it only affects search-time behaviour.

**`encoder_device`** — ONNX encoding device for batch mode. When set
to `"gpu"`, embedding inference is offloaded to CoreML (macOS) or
CUDA (Linux). Requires building with `--features gpu-encode`. See
[Building: GPU encoding](building.md#gpu-encoding) for platform setup.

Ignored in live mode (`meld serve`) — GPU encoding does not improve
single-record latency because the kernel launch overhead exceeds the
compute savings at batch size 1. If set in a live-mode config, the melder
prints a warning and falls back to CPU.

> [!TIP]
> **Tuning GPU encoding for maximum throughput.** GPU encoding
> benefits from multiple concurrent ONNX sessions to keep the GPU fed
> while CPU cores handle tokenisation. The optimal settings depend on
> your hardware:
>
> - **`encoder_pool_size`**: set to ~60% of your CPU core count. Each
>   session needs CPU time for tokenisation and data marshalling; too
>   many sessions starve the CPU, too few leave the GPU idle.
>
> - **`encoder_batch_size`**: 256 is optimal for most configurations.
>   Larger batches (512) cause GPU memory pressure when combined with
>   many concurrent sessions. Smaller batches (64) underutilise the GPU.
>
> - Avoid a `pool_size x batch_size` product above ~3,000. Beyond that,
>   concurrent GPU memory usage degrades throughput sharply.
>
> Benchmarked on M1 Ultra (20 cores, 64 GPU cores, 64 GB) with
> all-MiniLM-L6-v2 on a 1M x 1M dataset:
>
> | pool | batch | rec/s | vs CPU baseline |
> |-----:|------:|------:|:---:|
> | 12 | 256 | **1,828** | **8.7x** |
> | 8 | 256 | 1,677 | 8.0x |
> | 16 | 128 | 1,718 | 8.2x |
> | 12 | 512 | 765 | 3.6x (memory pressure) |
> | 16 | 512 | 473 | 2.3x (memory pressure) |
>
> When `encoder_device` is `"gpu"` and `encoder_pool_size` is not set,
> the melder defaults to ~60% of available CPU cores and logs the chosen
> value.

**`encoder_batch_size`** — number of texts sent per ONNX inference
call during batch encoding. Default: 64 for CPU (tuned for Apple
Silicon cache locality), 256 for GPU (amortises kernel launch overhead).
Batch mode only.

Batch scoring thread count is controlled by the `RAYON_NUM_THREADS`
environment variable (defaults to logical CPU count if unset).

### BM25 commit batching

**`bm25_commit_batch_size`** — controls how many BM25 upserts are
buffered before the Tantivy index is committed. This is a top-level
config field (not under `performance`).

Each Tantivy commit is expensive: it finalizes index segments, builds
FST term dictionaries, serializes postings, and garbage-collects old
segment files. Under high concurrency this commit takes a write lock
that serializes all workers, creating a major throughput bottleneck.

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `1` (default) | Commit after every upsert. Every BM25 query sees the very latest records. | Maximum accuracy. Low-throughput or latency-insensitive workloads. |
| `50`-`200` | Commit after N upserts. Newly inserted records may not be visible to BM25 queries until the batch completes. | High-throughput live workloads where the embedding index provides a parallel candidate path. |

**Accuracy trade-off**: when `bm25_commit_batch_size > 1`, a record
inserted between commits is invisible to BM25 queries but still
immediately visible to the embedding (ANN) index. In a typical 50/50
embedding + BM25 config, the embedding path finds the candidate
independently, so the combined score drops by at most the BM25 weight
(e.g. from 0.92 to ~0.46) only for the brief window before the next
commit. In practice this rarely affects match outcomes because:

1. The embedding index has no commit delay — candidates are found
   immediately via ANN search.
2. The BM25 window is short (100 upserts at 1,400 req/s = ~70ms).
3. The `try_match` endpoint (query-only, no upsert) always forces a
   full commit before querying, so explicit match requests see the
   latest data regardless of batch size.

**Benchmark**: on a 10k x 10k dataset with 50% embedding + 50% BM25
scoring, `bm25_commit_batch_size: 100` improved live-mode throughput
from 461 req/s to 1,473 req/s (3.2x) and reduced p50 latency from
11.5ms to 5.1ms.

```yaml
# Recommended for production live workloads with BM25 scoring:
bm25_commit_batch_size: 100
```

For detailed descriptions of each scoring method including worked examples, see [Scoring Methods](scoring.md).
