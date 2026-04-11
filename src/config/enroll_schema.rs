//! Enroll-mode YAML schema — single-pool entity resolution.
//!
//! Deserialized from YAML when `meld enroll` is invoked, then normalised
//! into the engine-facing `Config` via `into_config()`.

use serde::Deserialize;

use crate::error::ConfigError;

use super::schema::{
    BatchConfig, Bm25FieldPair, Config, CrossMapConfig, DatasetConfig, DatasetsConfig,
    EmbeddingsConfig, ExclusionsConfig, HooksConfig, LiveConfig, Mode, OutputConfig,
    PerformanceConfig, ScoringLogConfig, SynonymDictionaryConfig,
};

// ---------------------------------------------------------------------------
// Enroll-specific field types (single `field` instead of `field_a`/`field_b`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct EnrollMatchField {
    /// Single field name — used for both sides internally.
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
pub struct EnrollBm25Field {
    pub field: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EnrollBlockingField {
    pub field: String,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct EnrollBlockingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "super::schema::default_operator")]
    pub operator: String,
    #[serde(default)]
    pub fields: Vec<EnrollBlockingField>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct EnrollExactPrefilterConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub fields: Vec<EnrollBlockingField>,
}

#[derive(Debug, Deserialize)]
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
pub struct EnrollConfig {
    pub job: super::schema::JobConfig,

    /// Optional initial dataset to pre-load into the pool (no edges generated).
    #[serde(default)]
    pub dataset: Option<EnrollDatasetConfig>,

    pub embeddings: EnrollEmbeddingsConfig,

    #[serde(default)]
    pub blocking: EnrollBlockingConfig,

    #[serde(default)]
    pub exact_prefilter: EnrollExactPrefilterConfig,

    pub match_fields: Vec<EnrollMatchField>,

    pub thresholds: super::schema::ThresholdsConfig,

    #[serde(default)]
    pub live: LiveConfig,

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
    pub bm25_commit_batch_size: Option<usize>,

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
}

// ---------------------------------------------------------------------------
// Conversion to engine Config
// ---------------------------------------------------------------------------

impl EnrollConfig {
    /// Convert enroll-mode YAML into the engine-facing `Config`.
    ///
    /// Expands single `field` names to `field_a = field_b`, maps
    /// `dataset` to `datasets.a`, and fills B-side/crossmap/output
    /// with inert defaults.
    pub fn into_config(self) -> Result<Config, ConfigError> {
        // Dataset: map to A-side, B-side is a dummy copy
        let a_dataset = match self.dataset {
            Some(d) => DatasetConfig {
                path: d.path,
                id_field: d.id_field.clone(),
                common_id_field: d.common_id_field,
                format: d.format,
                encoding: d.encoding,
            },
            None => DatasetConfig {
                path: String::new(),
                id_field: String::new(),
                common_id_field: None,
                format: None,
                encoding: None,
            },
        };

        // B-side mirrors A-side field names (enroll mode: field_a == field_b)
        let b_dataset = DatasetConfig {
            path: String::new(),
            id_field: a_dataset.id_field.clone(),
            common_id_field: None,
            format: None,
            encoding: None,
        };

        // Expand match fields: field -> field_a = field_b
        let match_fields: Vec<super::schema::MatchField> = self
            .match_fields
            .into_iter()
            .map(|ef| {
                let bm25_fields = ef.fields.map(|fs| {
                    fs.into_iter()
                        .map(|f| Bm25FieldPair {
                            field_a: f.field.clone(),
                            field_b: f.field,
                        })
                        .collect()
                });
                super::schema::MatchField {
                    field_a: ef.field.clone(),
                    field_b: ef.field,
                    method: ef.method,
                    scorer: ef.scorer,
                    weight: ef.weight,
                    fields: bm25_fields,
                }
            })
            .collect();

        // Expand blocking fields
        let blocking = super::schema::BlockingConfig {
            enabled: self.blocking.enabled,
            operator: self.blocking.operator,
            fields: self
                .blocking
                .fields
                .into_iter()
                .map(|f| super::schema::BlockingFieldPair {
                    field_a: f.field.clone(),
                    field_b: f.field,
                })
                .collect(),
            field_a: None,
            field_b: None,
        };

        // Expand exact prefilter fields
        let exact_prefilter = super::schema::ExactPrefilterConfig {
            enabled: self.exact_prefilter.enabled,
            fields: self
                .exact_prefilter
                .fields
                .into_iter()
                .map(|f| super::schema::BlockingFieldPair {
                    field_a: f.field.clone(),
                    field_b: f.field,
                })
                .collect(),
        };

        // Expand BM25 fields
        let bm25_fields: Vec<Bm25FieldPair> = self
            .bm25_fields
            .into_iter()
            .map(|f| Bm25FieldPair {
                field_a: f.field.clone(),
                field_b: f.field,
            })
            .collect();

        // Expand synonym fields
        let synonym_fields: Vec<super::schema::SynonymFieldConfig> = self
            .synonym_fields
            .into_iter()
            .map(|sf| super::schema::SynonymFieldConfig {
                field_a: sf.field.clone(),
                field_b: sf.field,
                generators: sf.generators,
            })
            .collect();

        // Embeddings: cache_dir maps to a_cache_dir
        let embeddings = EmbeddingsConfig {
            model: self.embeddings.model,
            remote_encoder_cmd: self.embeddings.remote_encoder_cmd,
            a_cache_dir: self.embeddings.cache_dir,
            b_cache_dir: None,
        };

        // Dummy crossmap (not used in enroll mode)
        let cross_map = CrossMapConfig {
            backend: "local".into(),
            path: None,
            a_id_field: a_dataset.id_field.clone(),
            b_id_field: a_dataset.id_field.clone(),
        };

        Ok(Config {
            mode: Mode::Enroll,
            job: self.job,
            datasets: DatasetsConfig {
                a: a_dataset,
                b: b_dataset,
            },
            cross_map,
            exclusions: self.exclusions,
            embeddings,
            blocking,
            exact_prefilter,
            match_fields,
            output_mapping: Vec::new(),
            thresholds: self.thresholds,
            output: self.output,
            scoring_log: self.scoring_log,
            batch: BatchConfig::default(),
            live: self.live,
            performance: self.performance,
            hooks: self.hooks,
            vector_backend: self.vector_backend,
            top_n: self.top_n,
            ann_candidates: self.ann_candidates,
            bm25_candidates: self.bm25_candidates,
            bm25_commit_batch_size: self.bm25_commit_batch_size,
            required_fields_a: Vec::new(),
            required_fields_b: Vec::new(),
            bm25_fields,
            synonym_fields,
            synonym_dictionary: self.synonym_dictionary,
        })
    }
}
