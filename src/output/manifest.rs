//! Output manifest and build report types.

use crate::config::{Config, Mode};

/// Snapshot of config needed to build output schema.
pub struct OutputManifest {
    pub mode: Mode,
    pub job_name: String,
    pub a_id_field: String,
    pub b_id_field: String,
    pub a_fields: Vec<String>,
    pub b_fields: Vec<String>,
    pub auto_match: f64,
    pub review_floor: f64,
    pub min_score_gap: Option<f64>,
    pub top_n: usize,
    pub model: String,
}

impl OutputManifest {
    pub fn from_config(config: &Config) -> Self {
        Self {
            mode: config.mode,
            job_name: config.job.name.clone(),
            a_id_field: config.datasets.a.id_field.clone(),
            b_id_field: config.datasets.b.id_field.clone(),
            a_fields: config.required_fields_a.clone(),
            b_fields: config.required_fields_b.clone(),
            auto_match: config.thresholds.auto_match,
            review_floor: config.thresholds.review_floor,
            min_score_gap: config.thresholds.min_score_gap,
            top_n: config.top_n.unwrap_or(5),
            model: config.embeddings.model.clone(),
        }
    }
}

/// Report from a build_outputs call.
pub struct BuildReport {
    pub a_record_count: usize,
    pub b_record_count: usize,
    pub match_count: usize,
    pub review_count: usize,
    pub no_match_count: usize,
    pub broken_count: usize,
    pub warnings: Vec<String>,
    pub elapsed_secs: f64,
}
