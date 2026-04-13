← [Back to Index](./) | [Scoring Methods](scoring.md) | [CLI Reference](cli-reference.md)

# Configuration Reference

All behaviour is driven by a single YAML config file. This page describes every config section. Optional fields are noted with their defaults. Fields without a default annotation are required.

For detailed descriptions of each scoring method including worked examples, see [Scoring Methods](scoring.md).

---

## `job`

Metadata for your reference. Not used by the engine.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | no | `""` | A label for this config. Appears in output filenames and logs. |
| `description` | no | `""` | Free-text description of what this config does. |

```yaml
job:
  name: counterparty_recon
  description: Match entities to counterparties
```

---

## `datasets`

Paths to your input files. Field names in A and B do not need to match — the mapping is in `match_fields`.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `path` | **yes** | — | Path to CSV, JSONL, or Parquet file. |
| `id_field` | **yes** | — | Column whose value is the unique record key. |
| `common_id_field` | no | `None` | Shared identifier (LEI, ISIN, etc.). Records sharing the same value are matched immediately at score 1.0 before any scoring. Must be set on both sides or neither. |
| `format` | no | inferred | `"csv"`, `"jsonl"`, or `"parquet"`. Inferred from file extension if omitted. |
| `encoding` | no | `"utf-8"` | Character encoding for CSV/JSONL files. |

```yaml
datasets:
  a:
    path: data/entities.csv
    id_field: entity_id
    common_id_field: lei        # optional — must match on both sides or neither
    format: csv                 # optional — inferred from extension
  b:
    path: data/counterparties.csv
    id_field: counterparty_id
    format: csv
```

---

## `cross_map`

Persistent record of confirmed A↔B pairs. In batch mode this is a CSV file. In live mode with `live.db_path` it is stored in SQLite.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `backend` | no | `"local"` | Storage backend. Only `"local"` is supported. |
| `path` | no | `"crossmap.csv"` | Path to the cross-map CSV file. |
| `a_id_field` | **yes** | — | A-side ID column written into the cross-map output. |
| `b_id_field` | **yes** | — | B-side ID column written into the cross-map output. |

```yaml
cross_map:
  backend: local
  path: crossmap.csv
  a_id_field: entity_id
  b_id_field: counterparty_id
```

---

## `exclusions`

Known non-matching pairs to exclude from scoring. Pairs can be loaded from CSV at startup and added/removed at runtime via the API. If an excluded pair is currently matched, the match is broken automatically.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `path` | **yes** | — | Path to CSV file with columns matching `a_id_field` and `b_id_field`. Omit this section entirely to disable. |
| `a_id_field` | **yes** | — | Column name for A-side IDs in the exclusions CSV. |
| `b_id_field` | **yes** | — | Column name for B-side IDs in the exclusions CSV. |

```yaml
exclusions:
  path: exclusions.csv
  a_id_field: entity_id
  b_id_field: counterparty_id
```

---

## `embeddings`

Used by any match field with `method: embedding`. Model weights are downloaded automatically on first run.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | **yes** | — | Model identifier. Resolution order: `"builtin"` → local path → HuggingFace repo (`user/repo`) → named fastembed model (`all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5`, `bge-large-en-v1.5`). |
| `a_cache_dir` | **yes** | — | Directory for the A-side combined embedding index cache. Created automatically. |
| `b_cache_dir` | no | `None` | Same for B-side. Omit to skip B-side caching (vectors rebuilt every run). |

**Model resolution order:**

1. `"builtin"` — uses the model compiled into the binary (requires `--features builtin-model`)
2. Local path — `./models/my-model` or `/abs/path/model.onnx`
3. HuggingFace Hub — any `user/repo` name with a `/`
4. Named fastembed model — `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, etc.

```yaml
embeddings:
  model: themelder/arctic-embed-xs-entity-resolution
  a_cache_dir: cache/a
  b_cache_dir: cache/b
```

**Remote encoder alternative.** If your organisation requires embeddings
to run behind a central internal service, set
`embeddings.remote_encoder_cmd` instead of `embeddings.model` (exactly
one must be set — setting both is a validation error). Melder will
spawn your user-supplied script as a subprocess pool and talk to it
over a stdin/stdout protocol. See [Remote Encoder](remote-encoder.md)
for the full contract.

```yaml
embeddings:
  remote_encoder_cmd: "python scripts/my_encoder.py --env prod"
  a_cache_dir: cache/a
performance:
  encoder_pool_size: 8          # required with remote_encoder_cmd
  encoder_call_timeout_ms: 60000 # optional
```

---

## `vector_backend`

Controls the embedding index used for candidate selection.

| Value | Description |
|-------|-------------|
| `"usearch"` (default) | HNSW approximate nearest-neighbour graph. O(log N) search. Requires a C++ compiler. |
| `"flat"` | Brute-force O(N) scan. Pure Rust, no C++ dependency. Good for datasets under ~10k records. |

```yaml
vector_backend: usearch
```

---

## `top_n`, `ann_candidates`, `bm25_candidates`

Controls the progressive narrowing of candidates before full scoring.

**Required relationship:** `ann_candidates >= bm25_candidates >= top_n`.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `top_n` | no | `5` | Maximum candidates passed to full scoring and maximum results returned per record. |
| `ann_candidates` | no | `50` | Candidates the ANN (embedding) index retrieves per query. Larger values improve recall at the cost of scoring time. |
| `bm25_candidates` | no | `10` | Candidates BM25 keeps after re-ranking. Only used when `method: bm25` is in match_fields. |

```yaml
top_n: 20
ann_candidates: 200
bm25_candidates: 50
```

---

## `bm25_fields`

Which text fields to index for BM25 scoring. Preferred: define fields inline on the `method: bm25` entry in `match_fields`. Set them here as a top-level section only if you have no BM25 match field entries. Cannot use both.

When neither inline nor top-level fields are set, they are derived automatically from fuzzy/embedding `match_fields` entries.

```yaml
bm25_fields:
  - field_a: legal_name
    field_b: counterparty_name
```

---

## `exact_prefilter`

Pre-blocking exact match phase. If ALL configured field pairs match exactly (AND semantics), the pair is auto-confirmed at score 1.0 immediately — before blocking or scoring runs. Recovers pairs that blocking would miss due to mismatched blocking keys (e.g. wrong country code but matching LEI).

Use for globally unique identifiers only (LEI, ISIN, national ID). Omit this section to skip.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `enabled` | no | `false` | Set to `true` to enable. |
| `fields` | yes | — | List of field pairs that must all match exactly. |

```yaml
exact_prefilter:
  enabled: true
  fields:
    - field_a: lei
      field_b: lei_code
```

---

## `blocking`

Before candidate selection, blocking eliminates impossible candidates by requiring cheap field equality. Typically eliminates 95%+ of pairs at almost no cost.

Multiple field pairs are combined with AND (all must match). Omit this section or set `enabled: false` to disable — every record then becomes a candidate.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `enabled` | no | `false` | Set to `true` to enable blocking. |
| `fields` | yes | — | List of field pairs that must all match for two records to be candidates. |
| `operator` | no | `"and"` | Only `"and"` is supported. |

```yaml
blocking:
  enabled: true
  fields:
    - field_a: country_code
      field_b: domicile
```

---

## `match_fields`

The scoring equation. Each entry pairs a field from A with a field from B, names a comparison method, and assigns a weight. Weights must sum to exactly 1.0. Exception: synonym weights are additive and excluded from the sum-to-1.0 check.

### Methods

| Method | Description | Best for |
|--------|-------------|----------|
| `exact` | Binary string equality (case-insensitive). Returns 1.0 or 0.0. | Identifiers, codes, categoricals (country, currency, ISIN). |
| `fuzzy` | Edit-distance similarity. Select a scorer: `wratio` (default), `partial_ratio`, `token_sort_ratio`, `ratio`. | Names, free text. |
| `embedding` | Neural semantic similarity via cosine distance. Understands synonyms, abbreviations, translations. | The primary entity name field. |
| `numeric` | Numeric equality (parses both as float). Returns 1.0 or 0.0. | Numeric identifiers. |
| `bm25` | IDF-weighted token overlap. Specify fields via the inline `fields` key or `bm25_fields`. | Corpus-aware scoring alongside embedding. |
| `synonym` | Acronym/abbreviation matching. Binary 1.0/0.0. Weight is additive (not counted in 1.0). | Matching abbreviations like "HSBC" ↔ "Hongkong and Shanghai Banking Corporation". |

### Per-field options

| Field | Required | Description |
|-------|----------|-------------|
| `field_a` | **yes** | Field name in dataset A. |
| `field_b` | **yes** | Field name in dataset B. |
| `method` | **yes** | Scoring method (see table above). |
| `weight` | **yes** | Weight in the composite score. Must sum to 1.0 (excluding synonym). |
| `scorer` | no | Fuzzy scorer: `wratio` (default), `partial_ratio`, `token_sort_ratio`, `ratio`. Only for `method: fuzzy`. |
| `fields` | no | BM25 field pairs to index. Only for `method: bm25`. |

```yaml
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.55
  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy
    scorer: partial_ratio
    weight: 0.20
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05
  # Synonym (additive weight, not counted in 1.0):
  - field_a: legal_name
    field_b: counterparty_name
    method: synonym
    weight: 0.20
```

---

## `synonym_dictionary`

Optional CSV file of user-provided synonym equivalences. Each row lists 2+ terms treated as synonyms. Supplements the auto-generated acronym index.

| Field | Required | Description |
|-------|----------|-------------|
| `path` | **yes** | Path to CSV file. See [Scoring Methods](scoring.md) for format details. |

```yaml
synonym_dictionary:
  path: data/synonyms.csv
```

---

## `output_mapping`

Copy fields from A-side records into the output under a new column name. Useful for enriching output without adding fields to `match_fields`.

| Field | Required | Description |
|-------|----------|-------------|
| `from` | **yes** | Field name in the A-side record. |
| `to` | **yes** | Column name in the output CSV. |

```yaml
output_mapping:
  - from: sector
    to: ref_sector
  - from: country_code
    to: domicile
```

---

## `thresholds`

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `auto_match` | **yes** | — | Pairs scoring >= this value are confirmed automatically. |
| `review_floor` | no | `0.0` | Pairs scoring between this and `auto_match` go to review. Below this, pairs are discarded. |
| `min_score_gap` | no | disabled | Minimum margin between rank-1 and rank-2 candidate for auto-confirm. Single-candidate results are never downgraded. |

**Constraints:** `0 < review_floor < auto_match <= 1.0`

```yaml
thresholds:
  auto_match: 0.85
  review_floor: 0.60
```

---

## `output`

Controls where results are written. At least one of `csv_dir_path`, `parquet_dir_path`, or `db_path` must be set. Multiple can be set simultaneously.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `csv_dir_path` | no | `None` | Directory for `relationships.csv`, `unmatched.csv`, and `candidates.csv`. Created if it does not exist. |
| `parquet_dir_path` | no | `None` | Directory for `relationships.parquet`, `unmatched.parquet`, and `candidates.parquet`. Requires `--features parquet-format`. Created if it does not exist. |
| `db_path` | no | `None` | Path for the output SQLite database with tables and analytical views. |
| `cleanup_match_log` | no | `false` | Batch mode only. When true, deletes the match log after a successful build. |

```yaml
output:
  csv_dir_path: output/
  parquet_dir_path: output/parquet/
  db_path: output/results.db
```

---

## `scoring_log`

Per-field explainability data. When enabled, records every scored query's full top_n candidate set with per-field breakdowns. Produces `candidates.csv` (and `candidates.parquet` when Parquet output is configured) and populates the `field_scores` table in the output SQLite database. Also enables the `near_misses`, `runner_ups`, and `relationship_detail` views in the SQLite output.

Works in all three modes: `meld run` (batch), `meld serve` (live), and `meld enroll`. In enroll mode it is enabled by default, since enrollment has no other persistent relationship output; disable explicitly with `scoring_log.enabled: false` if not needed.

> [!NOTE]
> The SQLite batch path (`batch.db_path`) does not produce scoring log output.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `enabled` | no | `false` | Enable scoring log output. |
| `compression` | no | `"zstd"` | `"zstd"` (recommended, ~5-10x compression) or `"none"`. |
| `rotation_size_mb` | no | not set | Reserved for future use. Not currently implemented. |

```yaml
scoring_log:
  enabled: true
  compression: zstd
```

### What the scoring log produces

| Output | Description |
|--------|-------------|
| `{job_name}.scoring_log.ndjson.zst` | Raw scoring log file (one line per scored record with full candidate set) |
| `candidates.csv` | Rank-2+ candidates that did not make the final match (written by `build_outputs`) |
| `field_scores` DB table | Per-field score breakdowns for all candidates (SQLite output only) |

### Explainability views (SQLite output)

These views require the scoring log to be populated:

- **`near_misses`** — best unmatched candidates below review floor
- **`runner_ups`** — candidates at ranks 2+
- **`relationship_detail`** — joins relationships with per-field scores

---

## `batch`

Settings for `meld run`. Controls in-memory vs SQLite storage for large datasets.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `db_path` | no | `None` | When set, stores records in SQLite instead of RAM. File is created fresh and deleted after the run. |
| `sqlite_read_pool_size` | no | num_cpus | Read connection pool size. |
| `sqlite_pool_worker_cache_mb` | no | `128` | Page cache per read connection in MB. |
| `sqlite_cache_mb` | no | `64` | Write connection page cache in MB. |

```yaml
batch:
  db_path: batch.db
```

---

## `live`

Settings for `meld serve`. Ignored by `meld run`.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `match_log_path` | no | `"wal.ndjson"` | Path for the match log (WAL). Every event is appended here; compacted on clean shutdown. |
| `crossmap_flush_secs` | no | `5` | How often to flush the in-memory crossmap to disk. Ignored with `db_path`. |
| `db_path` | no | `None` | SQLite database for durable storage. Cold start loads from CSV; warm start opens directly (instant restarts). |
| `sqlite_cache_mb` | no | `64` | SQLite page cache for the write connection in MB. |
| `sqlite_read_pool_size` | no | `4` | Number of read-only SQLite connections. |
| `sqlite_pool_worker_cache_mb` | no | `128` | Page cache per read connection in MB. |
| `skip_initial_match` | no | `false` | When true, skips the startup scoring pass and begins accepting requests immediately. |

```yaml
live:
  db_path: data/live.db
  match_log_path: wal.ndjson
```

---

## Memory management

Melder’s memory use usually comes from four places:

1. **record storage** — input records, review queue, crossmap, blocking data
2. **vector indices** — the combined embedding indices for A and B
3. **encoder sessions** — one copy of the ONNX model per `encoder_pool_size` slot
4. **BM25 and other runtime structures** — token statistics, candidate buffers, etc.

No single setting caps total process memory. Different settings control different parts of the footprint.

### Main memory levers

| Lever | Affects | Best for | What it does | Typical tradeoff |
|---|---|---|---|---|
| `batch.db_path` | record storage | batch | Stores records in SQLite instead of RAM | Lower memory, slower than pure in-memory |
| `live.db_path` | record storage | live / enroll | Stores records, crossmap, and reviews in SQLite | Lower memory, modest throughput cost |
| `batch.sqlite_cache_mb`, `batch.sqlite_read_pool_size`, `batch.sqlite_pool_worker_cache_mb` | SQLite page cache | batch | Controls SQLite cache budget in batch mode | Lower cache = lower RAM, more disk I/O |
| `live.sqlite_cache_mb`, `live.sqlite_read_pool_size`, `live.sqlite_pool_worker_cache_mb` | SQLite page cache | live / enroll | Controls SQLite cache budget in live mode | Lower cache = lower RAM, more disk I/O |
| `performance.vector_quantization: f16` or `bf16` | vector indices | all usearch modes | Stores vectors at lower precision | Much smaller indices, usually negligible quality loss |
| `performance.vector_index_mode: mmap` | vector indices | warm batch runs only | Memory-maps usearch index instead of copying it into heap | Lower memory, less predictable latency |
| `performance.encoder_pool_size` | encoder model memory | all embedding modes | Controls number of parallel ONNX sessions | Lower value saves RAM, reduces encode throughput |
| `performance.quantized: true` | encoder model memory | local ONNX embedding runs | Uses INT8 encoder model instead of full-precision model | Lower memory, often faster, model-dependent |

### Which knobs to use in which mode

**Batch mode**
- If records are main problem, use `batch.db_path`.
- If vectors are main problem, use `vector_quantization: f16` first.
- For very large **warm** usearch runs, try `vector_index_mode: mmap`.
- Lower `encoder_pool_size` if model copies dominate RAM.
- Note: current SQLite batch path is for SQLite-backed record storage; embedding-heavy runs still need their own vector-memory tuning.

**Live mode (`meld serve`)**
- Use `live.db_path` if you want records offloaded from RAM.
- Use `vector_quantization: f16` to shrink vector indices.
- Keep `vector_index_mode: load`. `mmap` is read-only and not suitable for live upserts.
- Reduce `encoder_pool_size` if you need to trade throughput for lower memory.

**Enroll mode**
- Same guidance as live mode: mutable index, so use `load`, not `mmap`.
- `vector_quantization` and lower `encoder_pool_size` are the main vector/encoder levers.

### What these settings usually save

These are rough rules of thumb, not guarantees:

- **SQLite record storage (`batch.db_path`, `live.db_path`)**  
  Usually biggest win when records dominate memory. Can turn “does not fit in RAM” jobs into runnable ones. Cost is extra I/O and lower throughput.

- **`vector_quantization: f16` / `bf16`**  
  Usually cuts vector-index size by roughly **40–50%** with little or no noticeable quality loss. Best first lever for embedding-heavy jobs.

- **`vector_index_mode: mmap`**  
  Often saves close to **one full in-memory copy of the usearch index** on warm batch runs, because the graph is traversed from a memory-mapped file instead of copied into heap memory.  
  However, this is **not a hard RAM cap**: the OS may still keep much of the mapped file resident if the workload touches it heavily. Expect lower private heap usage, but RSS depends on access pattern.

- **Lower `encoder_pool_size`**  
  Saves roughly one model copy per slot removed. Good when startup or encode memory is too high. Cost is lower encoding throughput.

- **`performance.quantized: true`**  
  Can reduce encoder memory and usually improve encoding speed, but depends on the model path you are using.

### Approximate SQLite cache budget

If you use SQLite storage, the page-cache budget is roughly:

- **batch:**  
  `batch.sqlite_cache_mb + batch.sqlite_read_pool_size × batch.sqlite_pool_worker_cache_mb`
- **live / enroll:**  
  `live.sqlite_cache_mb + live.sqlite_read_pool_size × live.sqlite_pool_worker_cache_mb`

This covers SQLite cache only. It does **not** include vector indices, encoder models, BM25, or other heap allocations.

### Recommended order of attack

If memory is too high:

1. Move records to SQLite (`batch.db_path` or `live.db_path`) if record storage is large.
2. Set `vector_quantization: f16` if embeddings are enabled.
3. Reduce `encoder_pool_size` if encoder slots are large.
4. For large warm **batch** usearch runs, experiment with `vector_index_mode: mmap`.

Best settings depend heavily on dataset size, number of embedding fields, model choice, and workload shape. Start with these levers, then benchmark on your own data.

---

## `performance`

All fields are optional with sensible defaults. Omit the section if unsure.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `encoder_pool_size` | no | `1` | Number of concurrent ONNX inference sessions. Good starting point: match your core count. |
| `quantized` | no | `false` | Load INT8 quantised ONNX model. ~2x faster encoding, negligible quality loss. Not all models support this. |
| `encoder_batch_wait_ms` | no | `0` | Live mode only. Collects concurrent requests into a batch for up to N ms. Default 0 (disabled). |
| `vector_quantization` | no | `"f32"` | Storage precision for vectors: `"f32"`, `"f16"`, or `"bf16"`. Changing invalidates cache. |
| `vector_index_mode` | no | `"load"` | `"load"` reads index into RAM; `"mmap"` memory-maps. Do not use `"mmap"` with `meld serve`. |
| `expansion_search` | no | `0` | HNSW search beam width. Must be >= `ann_candidates`. Changing does not invalidate cache. |
| `encoder_device` | no | `"cpu"` | `"cpu"` or `"gpu"`. GPU requires `--features gpu-encode`. Batch mode only. |
| `encoder_batch_size` | no | `64` (CPU), `256` (GPU) | Texts per ONNX inference call. Batch mode only. |

```yaml
performance:
  encoder_pool_size: 4
  encoder_device: gpu
  encoder_batch_size: 256
```

### Performance field reference

**`encoder_pool_size`** — number of ONNX inference sessions to run in parallel. Each session holds a copy of the model in memory (~50–100 MB per slot). Higher values increase encoding throughput at the cost of RAM. 4 is a good starting point on machines with 4+ cores; 1 is fine for small datasets.

> [!TIP]
> **Tuning `encoder_pool_size` on multi-core machines.** Each ONNX session internally spawns its own thread pool for intra-op parallelism. By default ONNX Runtime sets the number of intra-op threads to the total number of physical cores. With `encoder_pool_size: 4` on a 20-core machine, that means 4 sessions × 20 threads = 80 logical threads competing for 20 cores — significant contention.
>
> For best throughput, aim for `pool_size × intra_op_threads ≈ physical cores`. You can control the per-session thread count with the **`ORT_NUM_THREADS`** environment variable:
>
> ```bash
> # 10 sessions × 2 threads = 20 threads on a 20-core machine
> ORT_NUM_THREADS=2 meld serve --config config.yaml --port 8080
> ```
>
> As a rule of thumb: raise `encoder_pool_size` to match your HTTP concurrency (or close to it), and lower `ORT_NUM_THREADS` so the product stays near your physical core count. This eliminates both mutex contention (requests waiting for a free encoder slot) and CPU contention (too many intra-op threads fighting for cores).

**`quantized`** — load the INT8 quantised variant of the ONNX model instead of the full FP32 model. Roughly doubles encoding speed with negligible quality loss. Supported for `all-MiniLM-L6-v2` and `all-MiniLM-L12-v2`; BGE models do not have quantised variants and will error if set.

**`vector_quantization`** — controls how the usearch vector backend stores vectors on disk and in memory.

| Value | Bytes per dimension | Notes |
|-------|--------------------:|-------|
| `f32` | 4 | Full precision. Default. |
| `f16` | 2 | Half precision. ~43% smaller index, negligible recall loss. |
| `bf16` | 2 | Brain float 16. Similar savings to f16, slightly different rounding. |

The primary benefit is **disk cache size**. At 100k records with 384-dim embeddings, the usearch cache is 171 MB per side with `f32` and 98 MB per side with `f16`. Changing this value invalidates existing caches.

> [!NOTE]
> **`quantized` vs `vector_quantization`** — these are independent settings. `quantized` controls the ONNX encoder model precision (FP32 vs INT8). `vector_quantization` controls the vector index storage precision (f32/f16/bf16). You can use either, both, or neither.

**`vector_index_mode`** — how the usearch HNSW index is loaded from its on-disk cache. `"load"` (default) pulls the full graph into RAM. `"mmap"` memory-maps the file — useful at extreme scale where the index does not fit in memory, but latency is unpredictable. Not suitable for `meld serve` because upserts write to the index.

**`expansion_search`** — HNSW search beam width (`ef` parameter). Controls how many graph nodes are explored when searching for nearest neighbours. Higher values improve recall at the cost of speed. Must be >= `ann_candidates`. No effect with the `flat` backend. Does **not** invalidate caches.

**`encoder_device`** — ONNX encoding device for batch mode. When set to `"gpu"`, embedding inference is offloaded to CoreML (macOS) or CUDA (Linux). Requires `--features gpu-encode`. See [Building: GPU encoding](building.md#gpu-encoding) for platform setup. Ignored in live mode.

> [!TIP]
> **Tuning GPU encoding for maximum throughput.** GPU encoding benefits from multiple concurrent ONNX sessions to keep the GPU fed while CPU cores handle tokenisation. Set `encoder_pool_size` to ~60% of your CPU core count. Use `encoder_batch_size: 256`. Avoid a `pool_size × batch_size` product above ~3,000 — concurrent GPU memory usage degrades throughput beyond that.

**`encoder_batch_size`** — texts per ONNX inference call. Default: 64 for CPU, 256 for GPU. Batch mode only.

---

## Example config

This is a working config taken from `benchmarks/batch/10kx10k_usearch/cold/config.yaml`. It matches 10k entity records against 10k counterparty records using a combination of embedding, fuzzy, and exact scoring.

```yaml
job:
  name: batch_10kx10k_usearch_cold
  description: Batch benchmark — 10k × 10k, usearch backend, cold run

datasets:
  a:
    path: benchmarks/data/dataset_a_10k.csv
    id_field: entity_id
    format: csv
  b:
    path: benchmarks/data/dataset_b_10k.csv
    id_field: counterparty_id
    format: csv

cross_map:
  backend: local
  path: benchmarks/batch/10kx10k_usearch/cold/crossmap.csv
  a_id_field: entity_id
  b_id_field: counterparty_id

embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: benchmarks/batch/10kx10k_usearch/cold/cache
  b_cache_dir: benchmarks/batch/10kx10k_usearch/cold/cache

blocking:
  enabled: true
  operator: and
  fields:
    - field_a: country_code
      field_b: domicile

match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.55
  - field_a: short_name
    field_b: counterparty_name
    method: fuzzy
    scorer: partial_ratio
    weight: 0.20
  - field_a: country_code
    field_b: domicile
    method: exact
    weight: 0.20
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05

thresholds:
  auto_match: 0.85
  review_floor: 0.60

output:
  csv_dir_path: benchmarks/batch/10kx10k_usearch/cold/output

performance:
  encoder_pool_size: 4
  vector_index_mode: "load"

vector_backend: usearch
top_n: 20
ann_candidates: 20
```

### What's happening in this config

**Datasets** — Two CSV files with asymmetric column names. A-side has `entity_id`, `legal_name`, `short_name`, `country_code`, `lei`. B-side has `counterparty_id`, `counterparty_name`, `domicile`, `lei_code`. The `match_fields` section bridges these differences.

**Blocking** — Records are only compared when their country matches (`country_code` ↔ `domicile`). This eliminates ~95% of candidate pairs before any scoring runs.

**Scoring** — Four weighted fields summing to 1.0:
- **Embedding (0.55)** — the primary signal. Legal names are converted to vectors and compared by cosine similarity. This catches semantic matches like "Apple Inc" ↔ "Apple Incorporated".
- **Fuzzy (0.20)** — partial string matching of short names against counterparty names. Catches abbreviations and truncations.
- **Exact country (0.20)** — binary equality, already guaranteed by blocking but scored for completeness in the composite.
- **Exact LEI (0.05)** — binary equality on the Legal Entity Identifier. High-signal when present.

**Thresholds** — Pairs scoring 0.85+ are auto-matched. Pairs between 0.60 and 0.85 go to the review queue. Below 0.60 is discarded.

**Performance** — Four parallel ONNX sessions for encoding. The usearch HNSW index retrieves 20 ANN candidates per query, of which up to 20 are fully scored.

**Cross-map** — Persisted to a CSV file. On re-runs, already-matched pairs are skipped.
