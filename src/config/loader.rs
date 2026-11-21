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
    super::validation::validate_scratch(&cfg, Some(path))?;
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
    fn load_batch_assignment_modes() {
        use crate::config::schema::BatchAssignmentMode;

        let base = r#"
vector_backend: flat
job: { name: test }
datasets:
  a: { path: a.csv, id_field: id }
  b: { path: b.csv, id_field: id }
cross_map: { backend: local, path: cm.csv, a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: name, field_b: name, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: output }
"#;
        let dir = tempfile::tempdir().expect("temporary config directory");
        let path = dir.path().join("config.yaml");
        for (suffix, mode) in [
            ("", Some(BatchAssignmentMode::Approximate)),
            ("batch: {}", Some(BatchAssignmentMode::Approximate)),
            (
                "batch: { assignment: {} }",
                Some(BatchAssignmentMode::Approximate),
            ),
            (
                "batch: { assignment: { mode: approximate } }",
                Some(BatchAssignmentMode::Approximate),
            ),
            (
                "batch: { assignment: { mode: exact } }",
                Some(BatchAssignmentMode::Exact),
            ),
            ("batch: { assignment: { strategy: greedy } }", None),
            (
                "batch: { assignment: { strategy: graph } }",
                Some(BatchAssignmentMode::Approximate),
            ),
            (
                "batch: { assignment: { strategy: graph, mode: approximate } }",
                Some(BatchAssignmentMode::Approximate),
            ),
            (
                "batch: { assignment: { strategy: graph, mode: exact } }",
                Some(BatchAssignmentMode::Exact),
            ),
        ] {
            std::fs::write(&path, format!("{base}\n{suffix}")).expect("write config");
            let cfg = load_config(&path).expect("load valid batch assignment");
            assert_eq!(cfg.batch.assignment.graph_mode(), mode, "{suffix}");
            assert_eq!(
                cfg.live.assignment,
                Default::default(),
                "batch assignment must not change live defaults"
            );
        }
        for assignment in [
            "{ strategy: greedy, mode: approximate }",
            "{ strategy: greedy, mode: exact }",
            "{ strategy: graph, mode: invalid }",
            "{ strategy: maximum_weight }",
        ] {
            std::fs::write(
                &path,
                format!("{base}\nbatch: {{ assignment: {assignment} }}"),
            )
            .expect("write invalid config");
            assert!(load_config(&path).is_err(), "reject {assignment}");
        }
    }

    #[test]
    fn load_live_startup_assignment_modes() {
        use crate::config::schema::{BatchAssignmentMode, LiveAssignmentStrategy};

        let base = std::fs::read_to_string("benchmarks/batch/10kx10k_flat/cold/config.yaml")
            .expect("read baseline config");
        let dir = tempfile::tempdir().expect("temporary config directory");
        let path = dir.path().join("config.yaml");
        // Force unrelated policies away from startup's graph/approximate defaults.
        let base = format!(
            "{base}\nbatch: {{ assignment: {{ strategy: greedy }} }}\nlive:\n  assignment: {{ strategy: deferred_acceptance }}\n"
        );
        for (startup, expected) in [
            ("", Some(BatchAssignmentMode::Approximate)),
            ("  startup: {}", Some(BatchAssignmentMode::Approximate)),
            (
                "  startup: { assignment: { mode: exact } }",
                Some(BatchAssignmentMode::Exact),
            ),
            ("  startup: { assignment: { strategy: greedy } }", None),
        ] {
            std::fs::write(&path, format!("{base}{startup}\n")).expect("write startup config");
            let cfg = load_config(&path).expect("load valid startup config");
            assert_eq!(
                cfg.live.startup.assignment.graph_mode(),
                expected,
                "{startup}"
            );
            assert_eq!(
                cfg.batch.assignment.graph_mode(),
                None,
                "batch stays greedy"
            );
            assert_eq!(
                cfg.live.assignment.strategy,
                LiveAssignmentStrategy::DeferredAcceptance,
                "incoming requests keep their independent policy"
            );
        }
        for assignment in [
            "{ strategy: greedy, mode: approximate }",
            "{ strategy: greedy, mode: exact }",
        ] {
            std::fs::write(
                &path,
                format!("{base}  startup: {{ assignment: {assignment} }}\n"),
            )
            .expect("write invalid startup config");
            let error = load_config(&path).expect_err("greedy startup must reject mode");
            assert!(
                matches!(error, ConfigError::InvalidValue { ref field, .. } if field == "live.startup.assignment.mode"),
                "startup error must identify its own setting: {error}"
            );
        }
    }

    #[test]
    fn load_bench_live() {
        let cfg = load_config(Path::new(
            "benchmarks/live/10kx10k_inject3k_flat/warm/config.yaml",
        ))
        .unwrap();
        assert_eq!(cfg.job.name, "live_10kx10k_flat_warm");
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

    #[cfg(feature = "usearch")]
    #[test]
    fn enroll_full_pipeline() {
        let cfg = load_enroll_config(Path::new(
            "benchmarks/enroll/10kx10k_enroll3k_scoring_log/config.yaml",
        ))
        .expect("load the checked-in USearch enroll configuration");
        assert!(cfg.is_enroll_mode());
    }
}
