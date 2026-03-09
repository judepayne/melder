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
    pub candidates: CandidatesConfig,
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
    /// Deprecated: use `performance.workers` instead. Kept for backward compat.
    #[serde(default)]
    pub workers: Option<u32>,

    // Derived at load time (not in YAML). Populated by `compute_required_fields`.
    #[serde(skip)]
    pub required_fields_a: Vec<String>,
    #[serde(skip)]
    pub required_fields_b: Vec<String>,
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
    /// Directory for A-side per-field vector index caches. Created
    /// automatically on first run; loaded on subsequent runs to skip encoding.
    pub a_cache_dir: String,
    /// Directory for B-side per-field vector index caches. Optional — omit
    /// to skip B-side caching (vectors rebuilt from scratch each run).
    #[serde(default)]
    pub b_cache_dir: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
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
    pub field_a: String,
    pub field_b: String,
    /// "exact" | "fuzzy" | "embedding" | "numeric"
    pub method: String,
    /// For fuzzy: "wratio" | "partial_ratio" | "token_sort_ratio" | "ratio"
    #[serde(default)]
    pub scorer: Option<String>,
    pub weight: f64,
}

#[derive(Debug, Deserialize, Default)]
pub struct CandidatesConfig {
    /// nil/true = enabled.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// A-side field to score for candidate selection.
    #[serde(default)]
    pub field_a: Option<String>,
    /// B-side field to score for candidate selection.
    #[serde(default)]
    pub field_b: Option<String>,
    /// Scoring method: "fuzzy", "embedding", or "exact".
    #[serde(default)]
    pub method: Option<String>,
    /// Fuzzy scorer (only used when method is "fuzzy"). Default "wratio".
    #[serde(default)]
    pub scorer: Option<String>,
    /// Number of top candidates to pass to full scoring. Default 10.
    #[serde(default)]
    pub n: Option<usize>,
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
    /// Default 5.
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub upsert_log: Option<String>,
    /// Number of concurrent ONNX inference sessions. Default 1.
    #[serde(default)]
    pub encoder_pool_size: Option<usize>,
    /// How often dirty CrossMap state is flushed to disk (seconds). Default 5.
    #[serde(default)]
    pub crossmap_flush_secs: Option<u64>,
}

/// Performance tuning — applies to both batch and live modes.
#[derive(Debug, Deserialize, Default)]
pub struct PerformanceConfig {
    /// Number of concurrent ONNX inference sessions. Default 1.
    #[serde(default)]
    pub encoder_pool_size: Option<usize>,
    /// Rayon parallel scoring threads. Default 4.
    #[serde(default)]
    pub workers: Option<u32>,
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
        let yaml = std::fs::read_to_string("testdata/configs/bench_live.yaml")
            .expect("failed to read bench_live.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize bench_live.yaml");

        assert_eq!(config.job.name, "bench_live_10000x10000");
        assert_eq!(config.datasets.a.path, "testdata/dataset_a_10000.csv");
        assert_eq!(config.datasets.b.id_field, "counterparty_id");
        assert_eq!(config.cross_map.backend, "local");
        assert_eq!(config.embeddings.model, "all-MiniLM-L6-v2");
        assert!(config.blocking.enabled);
        assert_eq!(config.match_fields.len(), 4);
        assert_eq!(config.match_fields[0].method, "embedding");
        assert!((config.match_fields[0].weight - 0.55).abs() < f64::EPSILON);
        assert!((config.thresholds.auto_match - 0.85).abs() < f64::EPSILON);
        assert!((config.thresholds.review_floor - 0.60).abs() < f64::EPSILON);
        assert_eq!(config.live.top_n, Some(5));
        assert_eq!(config.performance.encoder_pool_size, Some(4));
        assert_eq!(config.performance.workers, Some(4));
    }

    #[test]
    fn deserialize_counterparty_recon_with_sidecar() {
        // sidecar section in YAML is silently ignored by serde (no deny_unknown_fields)
        let yaml = std::fs::read_to_string("testdata/configs/counterparty_recon.yaml")
            .expect("failed to read counterparty_recon.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize counterparty_recon.yaml");

        assert_eq!(config.job.name, "counterparty_recon");
        assert_eq!(config.output_mapping.len(), 4);
        assert_eq!(config.output_mapping[0].from, "sector");
        assert_eq!(config.output_mapping[0].to, "ref_sector");
        assert_eq!(config.performance.workers, Some(4));
        assert_eq!(config.performance.encoder_pool_size, Some(2));
    }

    #[test]
    fn deserialize_bench1000x1000() {
        let yaml = std::fs::read_to_string("testdata/configs/bench1000x1000.yaml")
            .expect("failed to read bench1000x1000.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize bench1000x1000.yaml");

        assert_eq!(config.datasets.a.path, "testdata/dataset_a_1000.csv");
        assert_eq!(config.candidates.scorer, Some("wratio".into()));
        assert_eq!(config.candidates.n, Some(10));
    }
}
