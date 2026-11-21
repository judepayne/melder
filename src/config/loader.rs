//! Config loading: parse YAML, apply defaults, validate, derive fields.

use std::path::Path;

use crate::error::ConfigError;

use super::enroll_schema::EnrollJobConfig;
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

/// Load and validate native single-pool enroll YAML without paired configuration.
pub fn load_enroll_job_config(
    path: &Path,
) -> Result<super::enroll_schema::EnrollJobConfig, ConfigError> {
    let data = std::fs::read_to_string(path)?;
    let mut cfg: EnrollJobConfig = serde_yaml::from_str(&data)?;
    super::enroll_native::apply_defaults(&mut cfg);
    super::enroll_validation::validate_enroll(&cfg)?;
    super::enroll_native::derive_bm25_fields(&mut cfg);
    super::enroll_native::derive_synonym_fields(&mut cfg);
    super::enroll_native::derive_required_fields(&mut cfg);
    super::enroll_validation::validate_field_names(&cfg)?;
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
    fn native_enroll_defaults_derivation_and_validation() -> Result<(), Box<dyn std::error::Error>>
    {
        let base = r#"
vector_backend: flat
job: { name: native }
embeddings: { model: m, cache_dir: cache }
match_fields:
  - { field: name, method: fuzzy, weight: 0.7 }
  - { method: bm25, weight: 0.3 }
  - { field: alias, method: synonym, weight: 0.4 }
thresholds: { auto_match: 0.9, review_floor: 0.5 }
"#;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("enroll.yaml");
        let cases = [
            base.to_string(),
            format!("{base}\ndataset: {{ path: '', id_field: key, common_id_field: '' }}"),
            format!(
                "{base}\nblocking: {{ enabled: false, fields: [] }}\nexact_prefilter: {{ enabled: false, fields: [{{ field: 'ignored field' }}] }}"
            ),
            format!("{base}\nsynonym_fields: [{{ field: explicit_alias }}]"),
            format!("{base}\nlive: {{ match_log_path: enroll-log }}"),
            base.replace("field: alias", "field: name"),
            base.replace(
                "method: bm25, weight: 0.3",
                "method: bm25, weight: 0.3, fields: []",
            ),
            base.replace(
                "method: bm25, weight: 0.3",
                "method: bm25, weight: 0.3, fields: [{ field: text }]",
            ),
            format!("{base}\nbm25_fields: [{{ field: text }}]"),
            format!(
                "{base}\ndataset: {{ path: seed.csv, id_field: key, common_id_field: common }}\nblocking: {{ enabled: true, fields: [{{ field: region }}] }}\nexact_prefilter: {{ enabled: true, fields: [{{ field: code }}] }}"
            ),
            format!("{base}\ntop_n: 0\nann_candidates: 0\nbm25_candidates: 0"),
        ];
        for yaml in cases {
            std::fs::write(&path, yaml)?;
            let native = load_enroll_job_config(&path)?;
            assert_eq!(native.job.name, "native", "job name is retained");
            assert_eq!(native.match_fields.len(), 3, "all scoring fields retained");
            assert_eq!(
                native.embeddings.cache_dir, "cache",
                "single cache retained"
            );
            assert!(
                native.required_fields.contains(&"name".to_string()),
                "scoring fields required"
            );
        }
        std::fs::write(&path, base)?;
        let native = load_enroll_job_config(&path)?;
        assert_eq!(native.dataset.id_field, "id", "empty pool gets default ID");
        assert!(native.dataset.path.is_empty(), "seed dataset is optional");
        assert_eq!(native.top_n, Some(5), "native top_n default");
        assert_eq!(native.ann_candidates, Some(50), "native ANN default");
        assert_eq!(native.bm25_candidates, Some(10), "native BM25 default");
        assert_eq!(
            native.performance.encoder_pool_size,
            Some(1),
            "local encoder default"
        );
        assert!(
            native.live.match_log_path.is_none(),
            "persistence is opt-in"
        );
        assert_eq!(
            native.match_fields[0].scorer,
            Some(super::super::schema::FuzzyScorer::Wratio),
            "default fuzzy scorer"
        );
        assert_eq!(
            native
                .bm25_fields
                .iter()
                .map(|f| f.field.as_str())
                .collect::<Vec<_>>(),
            vec!["name"],
            "BM25 derives text fields"
        );
        assert_eq!(
            native
                .synonym_fields
                .iter()
                .map(|f| f.field.as_str())
                .collect::<Vec<_>>(),
            vec!["alias"],
            "synonym fields derive from scoring fields"
        );
        for field in ["id", "name", "alias"] {
            assert!(
                native.required_fields.iter().any(|f| f == field),
                "required pool field: {field}"
            );
        }
        for suffix in [
            "live: { db_path: pool.db }",
            "live: { assignment: { strategy: deferred_acceptance } }",
            "blocking: { enabled: true, operator: or, fields: [{ field: name }] }",
            "exact_prefilter: { enabled: true }",
            "performance: { encoder_batch_size: 0 }",
            "performance: { vector_index_mode: invalid }",
            "performance: { vector_quantization: invalid }",
            "performance: { encoder_call_timeout_ms: 0 }",
            "performance: { encoder_device: invalid }",
            "thresholds: { auto_match: 0.5, review_floor: 0.6 }",
            "thresholds: { auto_match: 0.9, review_floor: 0.5, min_score_gap: 1.0 }",
            "vector_backend: invalid",
            "job: { name: ' ' }",
            "synonym_dictionary: { path: ' ' }",
            "blocking: { enabled: true }",
            "live: { startup: { assignment: { strategy: greedy, mode: exact } } }",
            "hooks: { command: ' ' }",
            "bm25_candidates: 1",
            "dataset: { path: data.json, id_field: id }",
        ] {
            let mut yaml: serde_yaml::Value = serde_yaml::from_str(base)?;
            let overrides: serde_yaml::Mapping = serde_yaml::from_str(suffix)?;
            yaml.as_mapping_mut()
                .expect("base is a mapping")
                .extend(overrides);
            std::fs::write(&path, serde_yaml::to_string(&yaml)?)?;
            assert!(
                load_enroll_job_config(&path).is_err(),
                "reject invalid native setting: {suffix}"
            );
        }
        for yaml in [
            base.replace("weight: 0.7", "weight: 0.6"),
            base.replace("field: name", "field: 'bad field'"),
            base.replace(
                "method: bm25, weight: 0.3",
                "method: bm25, weight: 0.3, fields: [{ field: text }]",
            ) + "\nbm25_fields: [{ field: other }]",
            base.replace("model: m", "remote_encoder_cmd: encoder"),
            base.replace(
                "method: bm25, weight: 0.3",
                "field: name, method: bm25, weight: 0.3",
            ),
        ] {
            std::fs::write(&path, yaml)?;
            assert!(
                load_enroll_job_config(&path).is_err(),
                "native rejects invalid config"
            );
        }
        Ok(())
    }

    #[test]
    fn enroll_rejects_paired_only_yaml() -> Result<(), Box<dyn std::error::Error>> {
        let base = "job: { name: native }\nvector_backend: flat\nembeddings: { model: m, cache_dir: cache }\nmatch_fields: [{ field: name, method: exact, weight: 1.0 }]\nthresholds: { auto_match: 0.9, review_floor: 0.5 }\n";
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("enroll.yaml");
        for extra in [
            "datasets: {}",
            "bm25_commit_batch_size: 100",
            "required_fields: [ignored]",
            "live: { skip_initial_match: true }",
            "live: { crossmap_flush_secs: 5 }",
            "live: { startup: {} }",
            "live: { assignment: { strategy: greedy } }",
            "dataset: { path: seed.csv }",
            "dataset: { id_field: id }",
            "dataset: { path: seed.csv, id_field: id, extra: ignored }",
            "cross_map: {}",
            "batch: {}",
            "output_mapping: []",
            "blocking: { fields: [{ field_a: name, field_b: name }] }",
            "exact_prefilter: { fields: [{ field_a: name, field_b: name }] }",
            "bm25_fields: [{ field_a: name, field_b: name }]",
            "synonym_fields: [{ field_a: name, field_b: name }]",
        ] {
            std::fs::write(&path, format!("{base}{extra}\n"))?;
            assert!(
                load_enroll_job_config(&path).is_err(),
                "reject paired setting: {extra}"
            );
        }
        for (from, to) in [
            ("field: name", "field_a: name, field_b: name"),
            ("cache_dir: cache", "a_cache_dir: a, b_cache_dir: b"),
        ] {
            std::fs::write(&path, base.replace(from, to))?;
            assert!(
                load_enroll_job_config(&path).is_err(),
                "reject paired fields: {to}"
            );
        }
        Ok(())
    }

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
        let cfg = load_enroll_job_config(Path::new(
            "benchmarks/enroll/10kx10k_enroll3k_scoring_log/config.yaml",
        ))
        .expect("load the checked-in USearch enroll configuration");
        assert!(!cfg.dataset.id_field.is_empty(), "native pool ID field");
    }
}
