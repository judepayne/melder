---
type: architecture
module: config
status: active
tags: [config, schema, reference, yaml]
related_code: [src/config/schema.rs, src/config/loader.rs]
---

# Config Reference

Complete YAML config schema. All fields are validated at startup by `config/loader.rs`. See [[Use Cases]] for worked examples per deployment pattern.

## Top-Level Structure

```yaml
job:              # JobConfig            — required
datasets:         # DatasetsConfig       — required
cross_map:        # CrossMapConfig       — required
embeddings:       # EmbeddingsConfig     — required
blocking:         # BlockingConfig       — default: disabled
match_fields:     # Vec<MatchField>      — required
output_mapping:   # Vec<FieldMapping>    — default: []
thresholds:       # ThresholdsConfig     — required
output:           # OutputConfig         — batch mode only
live:             # LiveConfig           — live mode only
performance:      # PerformanceConfig    — default: all None/false
vector_backend:   # String              — default: "flat"
top_n:            # usize               — default: 5
```

---

## `job`

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | String | required | Job identifier |
| `description` | String | `""` | Human description |

---

## `datasets`

```yaml
datasets:
  a:
    path: "data/master.csv"
    id_field: "entity_id"
    common_id_field: "lei"   # optional; must be set on both sides if set on either
    format: "csv"            # optional; inferred from extension if absent
  b:
    path: "data/vendor.csv"
    id_field: "vendor_id"
    common_id_field: "lei"
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `path` | String | required | File path |
| `id_field` | String | required | Column used as the record's unique ID |
| `common_id_field` | Option\<String\> | None | Shared key for pre-match (e.g. LEI, ISIN, CUSIP). Records with matching values are auto-confirmed at score 1.0 before any scoring runs. |
| `format` | Option\<String\> | inferred | `"csv"`, `"parquet"` (feature-gated), or `"jsonl"` |
| `encoding` | Option\<String\> | UTF-8 | File encoding |

`common_id_field` must be set on **both** sides if set on either side. It is the fastest path to high match rates when a partial shared key exists.

---

## `cross_map`

```yaml
cross_map:
  backend: "local"
  path: "output/crossmap.csv"
  a_id_field: "master_id"
  b_id_field: "vendor_id"
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `backend` | String | `"local"` | Only supported value |
| `path` | Option\<String\> | `"crossmap.csv"` | Output file path |
| `a_id_field` | String | required | Column header for A IDs in the CSV |
| `b_id_field` | String | required | Column header for B IDs in the CSV |

See [[State & Persistence#CrossMap Persistence]] for flush mechanics and crash safety.

---

## `embeddings`

```yaml
embeddings:
  model: "all-MiniLM-L6-v2"
  a_cache_dir: "cache/a"
  b_cache_dir: "cache/b"   # optional
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | String | required | HuggingFace model name; downloaded and cached at first run |
| `a_cache_dir` | String | required | Directory for A-side combined index cache |
| `b_cache_dir` | Option\<String\> | None | B-side cache dir; omit to skip B-side vector caching |

Recommended model: `all-MiniLM-L6-v2` (384 dimensions, fast, strong quality for entity matching). Cache files are named `combined_{side}_{spec_hash}.idx`. Changing field names, order, weights, or `performance.vector_quantization` produces a new spec hash and forces a cold rebuild. See [[Key Decisions#Three-Layer Cache Invalidation]].

---

## `blocking`

```yaml
blocking:
  enabled: true
  operator: "and"     # "and" | "or"
  fields:
    - field_a: "country"
      field_b: "country_code"
    - field_a: "currency"
      field_b: "ccy"
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `false` | Must be `true` to activate blocking |
| `operator` | String | `"and"` | `"and"`: must match all pairs. `"or"`: any pair match is sufficient. |
| `fields` | Vec\<BlockingFieldPair\> | `[]` | List of `{field_a, field_b}` pairs |
| `field_a` / `field_b` | Option\<String\> | None | Legacy single-field syntax; promoted to `fields` at load time |

Blocking is critical for performance at scale. With country blocking on a 100k × 100k dataset, each query searches ~5k candidates instead of 100k. See [[Performance Baselines]]. Use `operator: "or"` when records may reach each other via multiple field routes.

---

## `match_fields`

```yaml
match_fields:
  - field_a: "name"
    field_b: "company_name"
    method: "embedding"
    weight: 0.6
  - field_a: "country"
    field_b: "domicile"
    method: "exact"
    weight: 0.2
  - field_a: "short_name"
    field_b: "alias"
    method: "fuzzy"
    scorer: "wratio"
    weight: 0.2
```

| Field | Type | Notes |
|---|---|---|
| `field_a` | String | Field name on the A-side record |
| `field_b` | String | Field name on the B-side record |
| `method` | String | `"exact"`, `"fuzzy"`, `"embedding"`, or `"numeric"` |
| `scorer` | Option\<String\> | Fuzzy only: `"wratio"` (default), `"partial_ratio"`, `"token_sort_ratio"`, `"ratio"` |
| `weight` | f64 | Relative weight; weights are auto-normalised and need not sum to 1.0 |

Fields with `method: "embedding"` are concatenated into the combined vector (each scaled by `sqrt(weight)`). See [[Scoring Algorithm]] for the full algorithm and [[Constitution#4 Combined Vector Weighted Cosine Identity]] for the math.

### Method selection guide

| Method | Best for |
|---|---|
| `"exact"` | Identifiers, codes, standardised fields (country codes, currencies, ISINs) |
| `"fuzzy"` (wratio) | Short free-text names where word order may vary ("JP Morgan" vs "JPMorgan") |
| `"fuzzy"` (partial_ratio) | Short string contained within a longer one ("Goldman" in "Goldman Sachs & Co.") |
| `"fuzzy"` (token_sort_ratio) | Multi-word names where word order is inconsistent |
| `"embedding"` | Long free text, descriptions, semantic similarity across paraphrase |
| `"numeric"` | Numeric identifiers where equality is the right comparison |

---

## `thresholds`

```yaml
thresholds:
  auto_match: 0.85
  review_floor: 0.60
```

| Field | Notes |
|---|---|
| `auto_match` | Score >= this → `Classification::Auto` (confirmed, written to results.csv / CrossMap) |
| `review_floor` | Score >= this but < auto_match → `Classification::Review` (borderline, written to review.csv / review queue) |

Both thresholds are **inclusive** (≥). Below `review_floor` → `Classification::NoMatch`. Use `meld tune` to inspect the score distribution before setting production thresholds. See [[Scoring Algorithm#Classification]].

---

## `output` (batch mode only)

```yaml
output:
  results_path: "output/results.csv"
  review_path: "output/review.csv"
  unmatched_path: "output/unmatched.csv"
```

---

## `output_mapping`

```yaml
output_mapping:
  - from: "sector"
    to: "ref_sector"
  - from: "lei"
    to: "ref_lei"
```

Enriches batch output by copying A-side fields (with optional rename) into the results CSV alongside the matched B record. Used to stamp A-side reference data attributes onto matched B records for downstream enrichment. Fields not present on a matched A record are omitted from that row.

---

## `live`

```yaml
live:
  upsert_log: "state/events.wal"
  crossmap_flush_secs: 5
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `upsert_log` | Option\<String\> | `"bench/upsert.wal"` | WAL base path. Each server run creates a new timestamped file: `events_20260311T184207Z.wal`. |
| `crossmap_flush_secs` | Option\<u64\> | `5` | How often the CrossMap is flushed to disk when dirty |

See [[State & Persistence#WAL]] for how the WAL file naming, replay, and compaction work.

---

## `performance`

```yaml
performance:
  encoder_pool_size: 4
  quantized: false
  encoder_batch_wait_ms: 0
  vector_quantization: "f32"
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `encoder_pool_size` | Option\<usize\> | `1` | Concurrent ONNX encoder sessions. **Recommended: 4.** Diminishing returns above 8. |
| `quantized` | bool | `false` | INT8-quantized ONNX model — ~2× faster encoding, negligible quality loss. Recommended for datasets > 50k records. |
| `encoder_batch_wait_ms` | Option\<u64\> | `0` | Coordinator batch window (ms). `0` = disabled. Only beneficial at concurrency ≥ 20 with large models. See [[Key Decisions#Encoding Coordinator Batched ONNX Inference]]. |
| `vector_quantization` | Option\<String\> | `"f32"` | Cache precision: `"f32"` (default), `"f16"` (43% smaller, no measurable quality loss), `"bf16"`. usearch backend only. |

See [[Performance Baselines]] for the throughput impact of each setting.

---

## `vector_backend`

```yaml
vector_backend: "usearch"   # "flat" (default) | "usearch"
```

| Value | Algorithm | When to use |
|---|---|---|
| `"flat"` | O(N) brute-force scan | Development and small datasets (< 10k records) |
| `"usearch"` | O(log N) HNSW ANN | Production — any dataset > 10k records |

`usearch` requires `cargo build --features usearch`. See [[Performance Baselines]] for throughput comparison. See [[Key Decisions#Combined Vector Index Single Index Per Side]] for why one index per side is sufficient.

---

## `top_n`

```yaml
top_n: 20
```

Controls both the ANN search width (number of candidates retrieved from the vector index) and the maximum number of matches returned per API response. Default is 5 in most configs; 20 is typical for batch mode where higher recall matters. Higher values improve recall at the cost of more full-scoring work per record.

---

See also: [[Business Logic Flow]] for how config drives the pipeline at runtime, [[State & Persistence]] for persistence-related config (`live`, `cross_map`, `embeddings` cache dirs).
