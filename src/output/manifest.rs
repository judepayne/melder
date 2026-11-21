//! Output manifest and build report types.

use crate::config::{Config, EnrollJobConfig, Mode};

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
    /// Adapt the single pool to the historical A/B output and log schema.
    pub fn from_enroll_job(config: &EnrollJobConfig) -> Self {
        Self {
            mode: Mode::Enroll,
            job_name: config.job.name.clone(),
            a_id_field: config.dataset.id_field.clone(),
            b_id_field: config.dataset.id_field.clone(),
            a_fields: config.required_fields.clone(),
            b_fields: config.required_fields.clone(),
            auto_match: config.thresholds.auto_match,
            review_floor: config.thresholds.review_floor,
            min_score_gap: config.thresholds.min_score_gap,
            top_n: config.top_n.unwrap_or(5),
            model: config.embeddings.model.clone(),
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self {
            mode: Mode::Match,
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

/// Locate a native enrollment scoring log using its unchanged on-disk namespace.
pub(crate) fn enroll_scoring_log_path(config: &EnrollJobConfig) -> Option<std::path::PathBuf> {
    if !config.scoring_log.enabled {
        return None;
    }
    let dir = config.output.csv_dir_path.as_deref().unwrap_or(".");
    let suffix = if config.scoring_log.compression != "none" {
        "ndjson.zst"
    } else {
        "ndjson"
    };
    let path = std::path::Path::new(dir).join(format!("{}.scoring_log.{suffix}", config.job.name));
    path.exists().then_some(path)
}

/// Report from a build_outputs call.
pub struct BuildReport {
    pub a_record_count: usize,
    pub b_record_count: usize,
    pub match_count: usize,
    pub enroll_edge_count: usize,
    pub review_count: usize,
    pub no_match_count: usize,
    pub broken_count: usize,
    pub warnings: Vec<String>,
    pub elapsed_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_enroll_manifest_preserves_common_id_columns() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.yaml");
        for field in ["label", "common"] {
            std::fs::write(
                &path,
                format!(
                    "job: {{name: compat}}\ndataset: {{path: seed.csv, id_field: id, common_id_field: common}}\nembeddings: {{model: m, cache_dir: cache}}\nvector_backend: flat\nmatch_fields: [{{field: {field}, method: exact, weight: 1}}]\nthresholds: {{auto_match: 0.9, review_floor: 0.5}}\n"
                ),
            )?;
            let config = crate::config::load_enroll_job_config(&path)?;
            let native = OutputManifest::from_enroll_job(&config);
            assert_eq!(native.a_fields, config.required_fields, "seed columns");
            assert_eq!(native.b_fields, native.a_fields, "symmetric pool columns");
            assert!(
                native.b_fields.iter().any(|f| f == "common"),
                "common ID is available at both endpoints with match field {field}"
            );
        }
        Ok(())
    }
}
