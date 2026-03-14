//! Config schema — Rust equivalent of Go `schema.go`.
//!
//! Users should be able to point `meld` at an existing `match` YAML config
//! with zero changes.

use serde::Deserialize;

/// Top-level configuration parsed from YAML.
#[derive(Debug, Deserialize)]
pub struct Config {
    pub job: JobConfig,
    pub datasets: DatasetsConfig,
    pub cross_map: CrossMapConfig,
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub blocking: BlockingConfig,
    pub match_fields: Vec<MatchField>,
    #[serde(default)]
    pub output_mapping: Vec<FieldMapping>,
    pub thresholds: ThresholdsConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub live: LiveConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    /// Vector storage backend: "flat" (brute-force, default) or "usearch"
    /// (per-block HNSW, requires building with `--features usearch`).
    #[serde(default = "default_vector_backend")]
    pub vector_backend: String,
    /// Maximum final results to return per record. Defaults to 5.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// How many candidates ANN retrieves from the full block. Defaults to 50.
    /// Must be >= bm25_candidates >= top_n when both ANN and BM25 are enabled.
    #[serde(default)]
    pub ann_candidates: Option<usize>,
    /// How many candidates BM25 keeps after re-ranking ANN's shortlist (when
    /// both are enabled), or retrieves directly from the full block (when BM25
    /// is the sole filter). Defaults to 10.
    #[serde(default)]
    pub bm25_candidates: Option<usize>,
    /// Optional memory budget for auto-configuring record store and vector index
    /// backends.
    ///
    /// - `"auto"`: detects available RAM at startup and uses 80% as the budget.
    /// - `"24GB"`, `"512MB"`, etc.: explicit size strings (TB, GB, MB, KB, B).
    ///
    /// When the estimated record or vector index footprint exceeds the budget
    /// thresholds (30% for records, 70% for vectors), melder automatically
    /// selects SQLite for the record store and/or mmap for the vector index.
    ///
    /// Default: `None` (no budget limit — fully in-memory, current behaviour).
    #[serde(default)]
    pub memory_budget: Option<String>,
    // Derived at load time (not in YAML). Populated by `compute_required_fields`.
    #[serde(skip)]
    pub required_fields_a: Vec<String>,
    #[serde(skip)]
    pub required_fields_b: Vec<String>,
    /// Derived at load time: (field_a, field_b) pairs from fuzzy/embedding
    /// match fields, used to determine which fields BM25 indexes.
    #[serde(skip)]
    pub bm25_fields: Vec<(String, String)>,
}

#[derive(Debug, Deserialize)]
pub struct JobConfig {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct DatasetsConfig {
    pub a: DatasetConfig,
    pub b: DatasetConfig,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub path: String,
    pub id_field: String,
    /// Optional shared identifier field (e.g. LEI). If set on one side,
    /// it must be set on both. Records sharing a common ID are
    /// auto-matched before any scoring takes place.
    #[serde(default)]
    pub common_id_field: Option<String>,
    /// "csv" | "parquet" | "jsonl"; inferred from extension if absent.
    #[serde(default)]
    pub format: Option<String>,
    /// For csv/jsonl; defaults to utf-8.
    #[serde(default)]
    pub encoding: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CrossMapConfig {
    /// "local" (only supported backend). Defaults to "local".
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Path for local backend.
    #[serde(default)]
    pub path: Option<String>,
    pub a_id_field: String,
    pub b_id_field: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsConfig {
    /// HuggingFace model name or local ONNX path.
    pub model: String,
    /// Directory for A-side combined embedding index cache. Created
    /// automatically on first run; loaded on subsequent runs to skip encoding.
    pub a_cache_dir: String,
    /// Directory for B-side combined embedding index cache. Optional — omit
    /// to skip B-side caching (vectors rebuilt from scratch each run).
    #[serde(default)]
    pub b_cache_dir: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct BlockingConfig {
    #[serde(default)]
    pub enabled: bool,
    /// "and" | "or". Defaults to "and".
    #[serde(default = "default_operator")]
    pub operator: String,
    #[serde(default)]
    pub fields: Vec<BlockingFieldPair>,
    /// Legacy single-field syntax — promoted to `fields` vec at load time.
    #[serde(default)]
    pub field_a: Option<String>,
    #[serde(default)]
    pub field_b: Option<String>,
}

#[derive(Debug, Deserialize, serde::Serialize, Clone)]
pub struct BlockingFieldPair {
    pub field_a: String,
    pub field_b: String,
}

#[derive(Debug, Deserialize)]
pub struct MatchField {
    /// Empty for `method: bm25` (operates across all text fields).
    #[serde(default)]
    pub field_a: String,
    /// Empty for `method: bm25` (operates across all text fields).
    #[serde(default)]
    pub field_b: String,
    /// "exact" | "fuzzy" | "embedding" | "numeric" | "bm25"
    pub method: String,
    /// For fuzzy: "wratio" | "partial_ratio" | "token_sort_ratio" | "ratio"
    #[serde(default)]
    pub scorer: Option<String>,
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct FieldMapping {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Deserialize)]
pub struct ThresholdsConfig {
    pub auto_match: f64,
    pub review_floor: f64,
}

#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    pub results_path: String,
    pub review_path: String,
    pub unmatched_path: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct LiveConfig {
    #[serde(default)]
    pub upsert_log: Option<String>,
    /// How often dirty CrossMap state is flushed to disk (seconds). Default 5.
    #[serde(default)]
    pub crossmap_flush_secs: Option<u64>,
    /// Path to the SQLite database file for durable live-mode storage.
    ///
    /// When set, live mode uses `SqliteStore` + `SqliteCrossMap` instead of
    /// in-memory `MemoryStore` + `MemoryCrossMap`. The database is created
    /// on first run (cold start) and reused on subsequent runs (warm start).
    ///
    /// Default: `None` (use in-memory storage, current behavior).
    #[serde(default)]
    pub db_path: Option<String>,
}

/// Performance tuning — applies to both batch and live modes.
#[derive(Debug, Deserialize, Default)]
pub struct PerformanceConfig {
    /// Number of concurrent ONNX inference sessions. Default 1.
    #[serde(default)]
    pub encoder_pool_size: Option<usize>,
    /// Use the INT8-quantised ONNX model variant. ~2x faster encoding,
    /// negligible quality loss. Default false.
    #[serde(default)]
    pub quantized: bool,
    /// How long (ms) the encoding coordinator waits to collect concurrent
    /// requests before dispatching them as a single ONNX batch call.
    ///
    /// - `None` or `0`: coordinator disabled, each request encodes
    ///   independently (current behaviour, zero overhead).
    /// - `1–10`: recommended for concurrent workloads (c >= 4). Requests
    ///   arriving within this window are batched, amortising ONNX overhead.
    ///
    /// Only affects live mode (`meld serve`). Batch mode always encodes
    /// in large batches regardless of this setting.
    #[serde(default)]
    pub encoder_batch_wait_ms: Option<u64>,
    /// Scalar quantization for vectors stored in the usearch HNSW index.
    ///
    /// - `"f32"` (default): full 32-bit precision, 1,536 bytes per 384-dim vector.
    /// - `"f16"`: half precision, 768 bytes per vector. Negligible recall loss,
    ///   ~50% smaller indices, faster search due to reduced memory bandwidth.
    /// - `"bf16"`: brain float 16, same size as f16. Wider exponent range,
    ///   slightly less mantissa precision. Also negligible recall loss.
    ///
    /// Has no effect when `vector_backend` is `flat`.
    /// Changing this value invalidates the usearch index cache (forces rebuild).
    #[serde(default)]
    pub vector_quantization: Option<String>,
    /// How the usearch HNSW index is loaded from the cache file.
    ///
    /// - `"load"` (default): full in-memory load. Consistent search latency,
    ///   higher peak RAM. Safe for both `meld run` and `meld serve`.
    /// - `"mmap"`: memory-mapped — OS pages in/out on demand. Lower peak RAM
    ///   but unpredictable cold-cache latency. **Read-only: do not use with
    ///   `meld serve`** (upserts write to the index and will fail).
    ///
    /// Has no effect when `vector_backend` is `flat`.
    #[serde(default)]
    pub vector_index_mode: Option<String>,
}

fn default_backend() -> String {
    "local".into()
}

fn default_vector_backend() -> String {
    "flat".into()
}

fn default_operator() -> String {
    "and".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_bench_live_yaml() {
        let yaml = std::fs::read_to_string("benchmarks/live/10kx10k_usearch/warm/config.yaml")
            .expect("failed to read bench_live.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize bench_live.yaml");

        assert_eq!(config.job.name, "live_10kx10k_usearch_warm");
        assert_eq!(config.datasets.a.path, "benchmarks/data/dataset_a_10k.csv");
        assert_eq!(config.datasets.b.id_field, "counterparty_id");
        assert_eq!(config.cross_map.backend, "local");
        assert_eq!(config.embeddings.model, "all-MiniLM-L6-v2");
        assert!(config.blocking.enabled);
        assert_eq!(config.match_fields.len(), 4);
        assert_eq!(config.match_fields[0].method, "embedding");
        assert!((config.match_fields[0].weight - 0.55).abs() < f64::EPSILON);
        assert!((config.thresholds.auto_match - 0.85).abs() < f64::EPSILON);
        assert!((config.thresholds.review_floor - 0.60).abs() < f64::EPSILON);
        assert_eq!(config.top_n, Some(5));
        assert_eq!(config.performance.encoder_pool_size, Some(4));
    }

    #[test]
    fn deserialize_counterparty_recon_with_sidecar() {
        // sidecar section in YAML is silently ignored by serde (no deny_unknown_fields)
        let yaml = std::fs::read_to_string("tests/fixtures/counterparty_recon.yaml")
            .expect("failed to read counterparty_recon.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize counterparty_recon.yaml");

        assert_eq!(config.job.name, "counterparty_recon");
        assert_eq!(config.output_mapping.len(), 4);
        assert_eq!(config.output_mapping[0].from, "sector");
        assert_eq!(config.output_mapping[0].to, "ref_sector");
        assert_eq!(config.performance.encoder_pool_size, Some(2));
    }

    #[test]
    fn deserialize_bench1kx1k() {
        let yaml = std::fs::read_to_string("benchmarks/batch/10kx10k_flat/cold/config.yaml")
            .expect("failed to read bench1kx1k.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize bench1kx1k.yaml");

        assert_eq!(config.datasets.a.path, "benchmarks/data/dataset_a_10k.csv");
        // top_n: 20 is set explicitly in bench1kx1k.yaml
        assert_eq!(config.top_n, Some(20));
    }
}
