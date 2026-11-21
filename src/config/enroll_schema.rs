//! Enroll-mode YAML schema — single-pool entity resolution.
//!
//! Native jobs use a single pool without paired matching state.

use serde::Deserialize;

use super::schema::{
    DatasetConfig, ExclusionsConfig, HooksConfig, OutputConfig, PerformanceConfig,
    ScoringLogConfig, SynonymDictionaryConfig,
};

// ---------------------------------------------------------------------------
// Enroll-specific field types (single `field` instead of `field_a`/`field_b`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollMatchField {
    /// Name used to look up the value in any record in this pool.
    /// Empty for `method: bm25` when using inline `fields`.
    #[serde(default)]
    pub field: String,
    /// Scoring method. Invalid values are rejected at YAML parse time.
    pub method: super::schema::MatchMethod,
    /// Fuzzy scorer variant. Only meaningful when `method == Fuzzy`.
    #[serde(default)]
    pub scorer: Option<super::schema::FuzzyScorer>,
    pub weight: f64,
    /// For `method: bm25` only — which fields the BM25 index covers.
    #[serde(default)]
    pub fields: Option<Vec<EnrollBm25Field>>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EnrollBm25Field {
    pub field: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EnrollBlockingField {
    pub field: String,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct EnrollBlockingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "super::schema::default_operator")]
    pub operator: String,
    #[serde(default)]
    pub fields: Vec<EnrollBlockingField>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct EnrollExactPrefilterConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub fields: Vec<EnrollBlockingField>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollEmbeddingsConfig {
    /// HuggingFace model name or local ONNX path. Mutually exclusive
    /// with `remote_encoder_cmd`.
    #[serde(default)]
    pub model: String,
    /// Shell command to spawn a user-supplied encoder subprocess.
    /// Mutually exclusive with `model`. See `docs/remote-encoder.md`.
    #[serde(default)]
    pub remote_encoder_cmd: Option<String>,
    /// Single cache directory for the pool's embedding index.
    pub cache_dir: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollDatasetConfig {
    pub path: String,
    pub id_field: String,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub encoding: Option<String>,
    #[serde(default)]
    pub common_id_field: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollSynonymFieldConfig {
    pub field: String,
    #[serde(default = "super::schema::default_synonym_generators")]
    pub generators: Vec<super::schema::SynonymGenerator>,
}

// ---------------------------------------------------------------------------
// Top-level enroll config
// ---------------------------------------------------------------------------

/// YAML schema for `meld enroll`. Clean single-pool configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollJobConfig {
    pub job: super::schema::JobConfig,

    /// Optional initial dataset to pre-load into the pool (no edges generated).
    #[serde(default = "default_dataset", deserialize_with = "deserialize_dataset")]
    pub dataset: DatasetConfig,

    pub embeddings: EnrollEmbeddingsConfig,

    #[serde(default)]
    pub blocking: EnrollBlockingConfig,

    #[serde(default)]
    pub exact_prefilter: EnrollExactPrefilterConfig,

    pub match_fields: Vec<EnrollMatchField>,

    pub thresholds: super::schema::ThresholdsConfig,

    #[serde(default)]
    pub live: EnrollRuntimeConfig,

    #[serde(default)]
    pub performance: PerformanceConfig,

    #[serde(default)]
    pub hooks: HooksConfig,

    #[serde(default = "super::schema::default_vector_backend")]
    pub vector_backend: String,

    #[serde(default)]
    pub top_n: Option<usize>,

    #[serde(default)]
    pub ann_candidates: Option<usize>,

    #[serde(default)]
    pub bm25_candidates: Option<usize>,

    #[serde(default)]
    pub bm25_fields: Vec<EnrollBm25Field>,

    #[serde(default)]
    pub synonym_fields: Vec<EnrollSynonymFieldConfig>,

    #[serde(default)]
    pub synonym_dictionary: Option<SynonymDictionaryConfig>,

    /// Known non-matching pairs to exclude from scoring.
    #[serde(default)]
    pub exclusions: ExclusionsConfig,

    #[serde(default)]
    pub output: OutputConfig,

    #[serde(default)]
    pub scoring_log: ScoringLogConfig,

    /// Derived record fields, never accepted from YAML.
    #[serde(skip)]
    pub required_fields: Vec<String>,
}

fn deserialize_dataset<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<DatasetConfig, D::Error> {
    let seed = EnrollDatasetConfig::deserialize(deserializer)?;
    Ok(DatasetConfig {
        path: seed.path,
        id_field: seed.id_field,
        format: seed.format,
        encoding: seed.encoding,
        common_id_field: seed.common_id_field,
    })
}

fn default_dataset() -> DatasetConfig {
    DatasetConfig {
        path: String::new(),
        id_field: "id".into(),
        format: None,
        encoding: None,
        common_id_field: None,
    }
}

/// Persistence settings for the single-pool runtime.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollRuntimeConfig {
    /// Base path for enrollment WAL and snapshots.
    #[serde(default)]
    pub match_log_path: Option<String>,
}
