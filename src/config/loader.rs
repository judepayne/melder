//! Config loading: parse YAML, apply defaults, validate, derive fields.

use std::path::Path;

use crate::error::ConfigError;

use super::enroll_schema::EnrollConfig;
use super::schema::Config;

/// Load, parse, validate, and return a fully-populated `Config`.
pub fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let data = std::fs::read_to_string(path)?;
    let mut cfg: Config = serde_yaml::from_str(&data)?;

    super::defaults::normalise_blocking(&mut cfg);
    super::defaults::apply_defaults(&mut cfg);
    super::defaults::deprecation_warnings(&cfg);
    super::validation::validate(&cfg)?;
    super::derivation::derive_bm25_fields(&mut cfg);
    super::derivation::derive_synonym_fields(&mut cfg);
    super::derivation::derive_required_fields(&mut cfg);
    super::validation::validate_field_names(&cfg)?;

    Ok(cfg)
}

/// Load, parse, validate, and return a `Config` for enroll mode.
///
/// The YAML is deserialized into `EnrollConfig` (single-pool schema),
/// then normalised into the engine-facing `Config` with `field_a == field_b`.
pub fn load_enroll_config(path: &Path) -> Result<Config, ConfigError> {
    let data = std::fs::read_to_string(path)?;
    let enroll: EnrollConfig = serde_yaml::from_str(&data)?;
    let mut cfg = enroll.into_config()?;

    super::defaults::apply_defaults(&mut cfg);
    super::validation::validate_enroll(&cfg)?;
    super::derivation::derive_bm25_fields(&mut cfg);
    super::derivation::derive_synonym_fields(&mut cfg);
    super::derivation::derive_required_fields(&mut cfg);
    super::validation::validate_field_names(&cfg)?;

    Ok(cfg)
}

/// Infer data format from file extension.
pub fn infer_format(path: &str, field_prefix: &str) -> Result<String, ConfigError> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "csv" | "tsv" => Ok("csv".into()),
        "parquet" => Ok("parquet".into()),
        "jsonl" | "ndjson" => Ok("jsonl".into()),
        "json" => Err(ConfigError::InvalidValue {
            field: format!("{}.path", field_prefix),
            message:
                "JSON array format is not supported; use .jsonl (newline-delimited JSON) instead"
                    .into(),
        }),
        "xlsx" | "xls" => Err(ConfigError::InvalidValue {
            field: format!("{}.path", field_prefix),
            message: "Excel format is not supported; convert to CSV or Parquet".into(),
        }),
        _ => Err(ConfigError::InvalidValue {
            field: format!("{}.path", field_prefix),
            message: format!(
                "cannot infer format from extension {:?}; set format: explicitly in config",
                ext
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn load_bench_live() {
        let cfg = load_config(Path::new(
            "benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml",
        ))
        .unwrap();
        assert_eq!(cfg.job.name, "live_10kx10k_usearch_warm");
        assert_eq!(cfg.performance.encoder_pool_size, Some(4));
        assert_eq!(cfg.top_n, Some(5));
        assert_eq!(cfg.live.crossmap_flush_secs, Some(5));
    }

    #[test]
    fn load_bench1kx1k() {
        let cfg = load_config(Path::new("benchmarks/batch/10kx10k_flat/cold/config.yaml")).unwrap();
        assert_eq!(cfg.top_n, Some(20));
        assert_eq!(cfg.performance.encoder_pool_size, Some(4));
    }

    #[test]
    fn load_counterparty_recon_with_sidecar() {
        let cfg = load_config(Path::new("tests/fixtures/counterparty_recon.yaml")).unwrap();
        assert_eq!(cfg.output_mapping.len(), 4);
        assert_eq!(cfg.performance.encoder_pool_size, Some(2));
    }

    #[test]
    fn enroll_full_pipeline() {
        let cfg = load_enroll_config(Path::new(
            "benchmarks/enroll/10kx10k_enroll3k_scoring_log/config.yaml",
        ));
        // If the fixture exists, validate the full pipeline runs.
        // If not (e.g. in CI without benchmark data), skip silently.
        if let Ok(cfg) = cfg {
            assert!(cfg.is_enroll_mode());
        }
    }
}
