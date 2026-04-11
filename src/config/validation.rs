//! Config validation: shared, match-mode, and enroll-mode validators.

use crate::error::ConfigError;

use super::schema::{Config, DatasetConfig, MatchMethod};

/// Valid cross-map backends.
const VALID_BACKENDS: &[&str] = &["local"];

/// Valid vector storage backends.
const VALID_VECTOR_BACKENDS: &[&str] = &["flat", "usearch"];

/// Valid vector quantization formats.
const VALID_VECTOR_QUANTIZATIONS: &[&str] = &["f32", "f16", "bf16"];

/// Valid vector index modes.
const VALID_VECTOR_INDEX_MODES: &[&str] = &["load", "mmap"];

/// Valid encoder devices.
const VALID_ENCODER_DEVICES: &[&str] = &["cpu", "gpu"];

/// Tolerance for field-weight sum validation (allows floating-point
/// accumulation error, e.g. 10 fields × 0.1 = 0.9999…8).
const WEIGHT_SUM_EPSILON: f64 = 0.01;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Validate a match-mode config (batch or live).
pub(crate) fn validate(cfg: &Config) -> Result<(), ConfigError> {
    // Match-mode specific checks
    validate_dataset(&cfg.datasets.a, "datasets.a")?;
    validate_dataset(&cfg.datasets.b, "datasets.b")?;

    // cross_map
    require_one_of(&cfg.cross_map.backend, VALID_BACKENDS, "cross_map.backend")?;
    if cfg.cross_map.backend == "local" && cfg.cross_map.path.as_deref().unwrap_or("").is_empty() {
        return Err(ConfigError::MissingField {
            field: "cross_map.path".into(),
        });
    }
    require_non_empty(&cfg.cross_map.a_id_field, "cross_map.a_id_field")?;
    require_non_empty(&cfg.cross_map.b_id_field, "cross_map.b_id_field")?;

    // common_id_field symmetry
    let a_common = cfg.datasets.a.common_id_field.as_deref().unwrap_or("");
    let b_common = cfg.datasets.b.common_id_field.as_deref().unwrap_or("");
    if !a_common.is_empty() && b_common.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "datasets.b.common_id_field".into(),
            message: "must be set when datasets.a.common_id_field is set".into(),
        });
    }
    if !b_common.is_empty() && a_common.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "datasets.a.common_id_field".into(),
            message: "must be set when datasets.b.common_id_field is set".into(),
        });
    }

    // output: at least one destination
    if cfg.output.csv_dir_path.is_none()
        && cfg.output.parquet_dir_path.is_none()
        && cfg.output.db_path.is_none()
    {
        return Err(ConfigError::InvalidValue {
            field: "output".into(),
            message: "at least one of output.csv_dir_path, output.parquet_dir_path, or output.db_path must be set".into(),
        });
    }

    // live flush interval
    if let Some(secs) = cfg.live.crossmap_flush_secs
        && secs < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "live.crossmap_flush_secs".into(),
            message: "must be >= 1".into(),
        });
    }

    // Blocking: validate both field_a and field_b
    validate_blocking(cfg, false)?;

    // Exact prefilter: validate both field_a and field_b
    validate_exact_prefilter(cfg, false)?;

    // vector_quantization / vector_index_mode flat-backend notes
    if let Some(ref vq) = cfg.performance.vector_quantization {
        require_one_of(
            vq,
            VALID_VECTOR_QUANTIZATIONS,
            "performance.vector_quantization",
        )?;
        if cfg.vector_backend == "flat" && vq != "f32" {
            eprintln!(
                "Note: vector_quantization {:?} has no effect with flat backend",
                vq
            );
        }
    }
    if let Some(ref vim) = cfg.performance.vector_index_mode {
        require_one_of(
            vim,
            VALID_VECTOR_INDEX_MODES,
            "performance.vector_index_mode",
        )?;
        if cfg.vector_backend == "flat" {
            eprintln!(
                "Note: vector_index_mode {:?} has no effect with flat backend",
                vim
            );
        }
    }

    // Shared checks
    validate_common(cfg)?;

    Ok(())
}

/// Validate an enroll-mode config.
pub(crate) fn validate_enroll(cfg: &Config) -> Result<(), ConfigError> {
    // Enroll: dataset A is optional (empty pool is valid)
    if !cfg.datasets.a.path.is_empty() {
        validate_dataset(&cfg.datasets.a, "dataset")?;
    }

    // Blocking: enroll only checks field_a (field_a == field_b after into_config)
    validate_blocking(cfg, true)?;

    // Exact prefilter: enroll only checks field_a
    validate_exact_prefilter(cfg, true)?;

    // vector_quantization / vector_index_mode (no flat-backend notes for enroll)
    if let Some(ref vq) = cfg.performance.vector_quantization {
        require_one_of(
            vq,
            VALID_VECTOR_QUANTIZATIONS,
            "performance.vector_quantization",
        )?;
    }
    if let Some(ref vim) = cfg.performance.vector_index_mode {
        require_one_of(
            vim,
            VALID_VECTOR_INDEX_MODES,
            "performance.vector_index_mode",
        )?;
    }

    // Shared checks
    validate_common(cfg)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared validation (called by both match and enroll)
// ---------------------------------------------------------------------------

fn validate_common(cfg: &Config) -> Result<(), ConfigError> {
    // job.name
    require_non_empty(&cfg.job.name, "job.name")?;

    // exclusions
    if let Some(ref p) = cfg.exclusions.path
        && !p.is_empty()
    {
        require_non_empty(&cfg.exclusions.a_id_field, "exclusions.a_id_field")?;
        require_non_empty(&cfg.exclusions.b_id_field, "exclusions.b_id_field")?;
    }

    // embeddings: exactly one of `model` / `remote_encoder_cmd`
    validate_embeddings_source(cfg)?;
    require_non_empty(&cfg.embeddings.a_cache_dir, "embeddings.a_cache_dir")?;

    // at least one match field
    if cfg.match_fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: "at least one match field required".into(),
        });
    }

    // validate each match field + accumulate weight sum
    validate_match_fields(cfg)?;

    // thresholds
    validate_thresholds(cfg)?;

    // performance
    validate_performance(cfg)?;

    // vector_backend
    require_one_of(&cfg.vector_backend, VALID_VECTOR_BACKENDS, "vector_backend")?;
    if cfg.vector_backend == "usearch" && !cfg!(feature = "usearch") {
        return Err(ConfigError::InvalidValue {
            field: "vector_backend".into(),
            message: "usearch backend requires building with --features usearch".into(),
        });
    }

    // synonym_dictionary
    if let Some(ref sd) = cfg.synonym_dictionary {
        require_non_empty(&sd.path, "synonym_dictionary.path")?;
    }

    // hooks
    validate_hooks(cfg)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Component validators
// ---------------------------------------------------------------------------

fn validate_match_fields(cfg: &Config) -> Result<(), ConfigError> {
    let mut weight_sum = 0.0_f64;
    let mut bm25_count = 0usize;

    for (i, mf) in cfg.match_fields.iter().enumerate() {
        let prefix = format!("match_fields[{}]", i);

        if mf.method == MatchMethod::Bm25 {
            if !mf.field_a.is_empty() || !mf.field_b.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: prefix.to_string(),
                    message:
                        "bm25 method operates across all text fields; field_a and field_b must be omitted"
                            .into(),
                });
            }
            bm25_count += 1;
        } else {
            require_non_empty(&mf.field_a, &format!("{}.field_a", prefix))?;
            // field_b may equal field_a in enroll mode; just check it's non-empty
            require_non_empty(&mf.field_b, &format!("{}.field_b", prefix))?;
        }

        if mf.weight < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: format!("{}.weight", prefix),
                message: "must be >= 0".into(),
            });
        }
        if mf.method != MatchMethod::Synonym {
            weight_sum += mf.weight;
        }
    }

    // At most one BM25 entry
    if bm25_count > 1 {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: format!("at most one bm25 entry allowed, found {}", bm25_count),
        });
    }

    // Conflicting BM25 field definitions
    let has_inline_bm25_fields = cfg
        .match_fields
        .iter()
        .any(|mf| mf.method == MatchMethod::Bm25 && mf.fields.is_some());
    if has_inline_bm25_fields && !cfg.bm25_fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "bm25_fields".into(),
            message: "BM25 fields defined both inline on the match_fields entry and as \
                      a top-level bm25_fields section — use one or the other"
                .into(),
        });
    }

    // Validate inline BM25 fields
    if let Some(bm25_mf) = cfg
        .match_fields
        .iter()
        .find(|mf| mf.method == MatchMethod::Bm25)
        && let Some(ref fields) = bm25_mf.fields
    {
        for (i, pair) in fields.iter().enumerate() {
            let prefix = format!("match_fields[bm25].fields[{}]", i);
            require_non_empty(&pair.field_a, &format!("{}.field_a", prefix))?;
            require_non_empty(&pair.field_b, &format!("{}.field_b", prefix))?;
        }
    }

    // Validate top-level bm25_fields
    for (i, pair) in cfg.bm25_fields.iter().enumerate() {
        let prefix = format!("bm25_fields[{}]", i);
        require_non_empty(&pair.field_a, &format!("{}.field_a", prefix))?;
        require_non_empty(&pair.field_b, &format!("{}.field_b", prefix))?;
    }

    // Candidate count constraints
    let has_bm25 = bm25_count > 0;
    let has_embedding = cfg
        .match_fields
        .iter()
        .any(|mf| mf.method == MatchMethod::Embedding);
    let top_n = cfg.top_n.unwrap_or(5);
    let ann_candidates = cfg.ann_candidates.unwrap_or(50);
    let bm25_candidates = cfg.bm25_candidates.unwrap_or(10);

    if has_embedding && ann_candidates < top_n {
        return Err(ConfigError::InvalidValue {
            field: "ann_candidates".into(),
            message: format!(
                "must be >= top_n ({}) when ANN is enabled, got {}",
                top_n, ann_candidates
            ),
        });
    }
    if has_bm25 && bm25_candidates < top_n {
        return Err(ConfigError::InvalidValue {
            field: "bm25_candidates".into(),
            message: format!(
                "must be >= top_n ({}) when BM25 is enabled, got {}",
                top_n, bm25_candidates
            ),
        });
    }

    // Weights sum to 1.0
    if (weight_sum - 1.0).abs() > WEIGHT_SUM_EPSILON {
        return Err(ConfigError::WeightSum { sum: weight_sum });
    }

    Ok(())
}

fn validate_thresholds(cfg: &Config) -> Result<(), ConfigError> {
    if cfg.thresholds.auto_match <= 0.0 || cfg.thresholds.auto_match > 1.0 {
        return Err(ConfigError::InvalidValue {
            field: "thresholds.auto_match".into(),
            message: "must be in range (0, 1]".into(),
        });
    }
    if cfg.thresholds.review_floor < 0.0 || cfg.thresholds.review_floor >= 1.0 {
        return Err(ConfigError::InvalidValue {
            field: "thresholds.review_floor".into(),
            message: "must be in range [0, 1)".into(),
        });
    }
    if cfg.thresholds.auto_match <= cfg.thresholds.review_floor {
        return Err(ConfigError::InvalidValue {
            field: "thresholds".into(),
            message: format!(
                "auto_match ({:.2}) must be greater than review_floor ({:.2})",
                cfg.thresholds.auto_match, cfg.thresholds.review_floor
            ),
        });
    }
    if let Some(gap) = cfg.thresholds.min_score_gap
        && !(0.0..1.0).contains(&gap)
    {
        return Err(ConfigError::InvalidValue {
            field: "thresholds.min_score_gap".into(),
            message: "must be in range [0.0, 1.0)".into(),
        });
    }
    Ok(())
}

fn validate_performance(cfg: &Config) -> Result<(), ConfigError> {
    let remote_set = cfg
        .embeddings
        .remote_encoder_cmd
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    if let Some(pool) = cfg.performance.encoder_pool_size
        && pool < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "performance.encoder_pool_size".into(),
            message: "must be >= 1".into(),
        });
    }
    // When remote_encoder_cmd is set, pool size must be explicit. No default.
    // See docs/remote-encoder.md — startup cost scales with pool size, so
    // we require the operator to think about it deliberately.
    if remote_set && cfg.performance.encoder_pool_size.is_none() {
        return Err(ConfigError::InvalidValue {
            field: "performance.encoder_pool_size".into(),
            message: "required when embeddings.remote_encoder_cmd is set (no default). \
                      See docs/remote-encoder.md"
                .into(),
        });
    }
    if let Some(ref dev) = cfg.performance.encoder_device {
        if remote_set {
            // Silently ignored (documented). Info log for operator visibility.
            tracing::info!(
                "performance.encoder_device is ignored when embeddings.remote_encoder_cmd is set"
            );
        } else {
            require_one_of(dev, VALID_ENCODER_DEVICES, "performance.encoder_device")?;
            if dev == "gpu" && !cfg!(feature = "gpu-encode") {
                return Err(ConfigError::InvalidValue {
                    field: "performance.encoder_device".into(),
                    message: "GPU encoding requires building with --features gpu-encode".into(),
                });
            }
        }
    }
    if let Some(bs) = cfg.performance.encoder_batch_size
        && bs < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "performance.encoder_batch_size".into(),
            message: "must be >= 1".into(),
        });
    }
    if let Some(ms) = cfg.performance.encoder_call_timeout_ms
        && ms < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "performance.encoder_call_timeout_ms".into(),
            message: "must be >= 1".into(),
        });
    }
    Ok(())
}

/// Validate that exactly one of `embeddings.model` / `embeddings.remote_encoder_cmd`
/// is set. Empty-string `model` is treated as unset to avoid requiring
/// `Option<String>` migration across every existing test fixture.
fn validate_embeddings_source(cfg: &Config) -> Result<(), ConfigError> {
    let has_model = !cfg.embeddings.model.trim().is_empty();
    let has_remote = cfg
        .embeddings
        .remote_encoder_cmd
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    match (has_model, has_remote) {
        (false, false) => Err(ConfigError::MissingField {
            field: "embeddings.model or embeddings.remote_encoder_cmd".into(),
        }),
        (true, true) => Err(ConfigError::InvalidValue {
            field: "embeddings".into(),
            message: "set exactly one of `model` or `remote_encoder_cmd` — not both".into(),
        }),
        _ => Ok(()),
    }
}

/// Validate blocking config. When `enroll_mode` is true, only checks field_a
/// (enroll has field_a == field_b).
fn validate_blocking(cfg: &Config, enroll_mode: bool) -> Result<(), ConfigError> {
    if !cfg.blocking.enabled {
        return Ok(());
    }
    if cfg.blocking.fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "blocking".into(),
            message: "at least one field pair required when enabled".into(),
        });
    }
    for (i, fp) in cfg.blocking.fields.iter().enumerate() {
        let prefix = format!("blocking.fields[{}]", i);
        require_non_empty(&fp.field_a, &format!("{}.field_a", prefix))?;
        if !enroll_mode {
            require_non_empty(&fp.field_b, &format!("{}.field_b", prefix))?;
        }
    }
    let op = cfg.blocking.operator.to_lowercase();
    if op != "and" {
        return Err(ConfigError::InvalidValue {
            field: "blocking.operator".into(),
            message: format!(
                "must be \"and\" (OR blocking is no longer supported), got {:?}",
                cfg.blocking.operator
            ),
        });
    }
    Ok(())
}

/// Validate exact_prefilter config. When `enroll_mode` is true, only checks field_a.
fn validate_exact_prefilter(cfg: &Config, enroll_mode: bool) -> Result<(), ConfigError> {
    if !cfg.exact_prefilter.enabled {
        return Ok(());
    }
    if cfg.exact_prefilter.fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "exact_prefilter".into(),
            message: "at least one field pair required when enabled".into(),
        });
    }
    for (i, fp) in cfg.exact_prefilter.fields.iter().enumerate() {
        let prefix = format!("exact_prefilter.fields[{}]", i);
        require_non_empty(&fp.field_a, &format!("{}.field_a", prefix))?;
        if !enroll_mode {
            require_non_empty(&fp.field_b, &format!("{}.field_b", prefix))?;
        }
    }
    Ok(())
}

fn validate_hooks(cfg: &Config) -> Result<(), ConfigError> {
    if let Some(ref cmd) = cfg.hooks.command
        && cmd.trim().is_empty()
    {
        return Err(ConfigError::InvalidValue {
            field: "hooks.command".into(),
            message: "must be non-empty if specified".into(),
        });
    }
    Ok(())
}

pub(crate) fn validate_dataset(d: &DatasetConfig, prefix: &str) -> Result<(), ConfigError> {
    require_non_empty(&d.path, &format!("{}.path", prefix))?;
    require_non_empty(&d.id_field, &format!("{}.id_field", prefix))?;

    match d.format.as_deref() {
        Some(fmt) if !fmt.is_empty() => {
            require_one_of(
                fmt,
                &["csv", "parquet", "jsonl"],
                &format!("{}.format", prefix),
            )?;
        }
        _ => {
            super::loader::infer_format(&d.path, prefix)?;
        }
    }

    Ok(())
}

/// Validate that all field names used as SQL column names are safe identifiers.
pub(crate) fn validate_field_names(cfg: &Config) -> Result<(), ConfigError> {
    fn is_safe_identifier(s: &str) -> bool {
        !s.is_empty()
            && s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    }

    for name in cfg.required_fields_a.iter().chain(&cfg.required_fields_b) {
        if !is_safe_identifier(name) {
            return Err(ConfigError::InvalidValue {
                field: "field name".into(),
                message: format!(
                    "{:?} is not a safe identifier (must match [a-zA-Z_][a-zA-Z0-9_.-]*)",
                    name,
                ),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_non_empty(value: &str, field: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        return Err(ConfigError::MissingField {
            field: field.into(),
        });
    }
    Ok(())
}

fn require_one_of(value: &str, valid: &[&str], field: &str) -> Result<(), ConfigError> {
    if !valid.contains(&value) {
        return Err(ConfigError::InvalidValue {
            field: field.into(),
            message: format!(
                "unsupported value {:?}; must be one of: {}",
                value,
                valid.join(", ")
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::defaults::{apply_defaults, normalise_blocking};
    use crate::config::loader::infer_format;

    /// Helper: minimal valid YAML for tests that add custom match fields.
    fn base_yaml_with_match_fields(match_fields_yaml: &str) -> String {
        format!(
            r#"
job:
  name: test
datasets:
  a: {{ path: "a.csv", id_field: id }}
  b: {{ path: "b.csv", id_field: id }}
cross_map: {{ backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }}
embeddings: {{ model: m, a_cache_dir: i }}
match_fields:
{}
thresholds: {{ auto_match: 0.85, review_floor: 0.6 }}
output: {{ csv_dir_path: /tmp/test/output }}
"#,
            match_fields_yaml
        )
    }

    fn parse_and_validate(yaml: &str) -> Result<(), ConfigError> {
        let mut cfg: Config = serde_yaml::from_str(yaml).map_err(ConfigError::Parse)?;
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg)
    }

    fn parse_and_validate_enroll(yaml: &str) -> Result<(), ConfigError> {
        let enroll: crate::config::enroll_schema::EnrollConfig =
            serde_yaml::from_str(yaml).map_err(ConfigError::Parse)?;
        let mut cfg = enroll.into_config()?;
        apply_defaults(&mut cfg);
        validate_enroll(&cfg)
    }

    // --- Format inference ---

    #[test]
    fn format_inference() {
        assert_eq!(infer_format("data.csv", "test").unwrap(), "csv");
        assert_eq!(infer_format("data.tsv", "test").unwrap(), "csv");
        assert_eq!(infer_format("data.parquet", "test").unwrap(), "parquet");
        assert_eq!(infer_format("data.jsonl", "test").unwrap(), "jsonl");
        assert_eq!(infer_format("data.ndjson", "test").unwrap(), "jsonl");
        assert!(infer_format("data.json", "test").is_err());
        assert!(infer_format("data.xlsx", "test").is_err());
        assert!(infer_format("data.xls", "test").is_err());
        assert!(infer_format("data.dat", "test").is_err());
    }

    // --- Match-mode validation ---

    #[test]
    fn validation_missing_job_name() {
        let yaml = r#"
job:
  name: ""
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(err.to_string().contains("job.name"), "got: {}", err);
    }

    #[test]
    fn validation_invalid_method() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: magic, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let result: Result<Config, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "invalid method should fail at parse time");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("magic") || err.contains("unknown variant"),
            "error should mention the invalid value, got: {}",
            err
        );
    }

    #[test]
    fn validation_invalid_backend() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: s3, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("cross_map.backend"),
            "got: {}",
            err
        );
    }

    #[test]
    fn validation_weight_sum() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 0.5 }
  - { field_a: g, field_b: g, method: fuzzy, weight: 0.45 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("weights sum"), "got: {}", err);
    }

    #[test]
    fn validation_auto_match_lt_review_floor() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.50, review_floor: 0.80 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("thresholds"), "got: {}", err);
    }

    #[test]
    fn validation_blocking_invalid_operator() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
blocking:
  enabled: true
  operator: xor
  fields:
    - { field_a: c, field_b: c }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("blocking.operator"),
            "got: {}",
            err
        );
    }

    #[test]
    fn validation_invalid_vector_backend() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
vector_backend: milvus
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("vector_backend"), "got: {}", err);
    }

    #[test]
    fn validation_vector_backend_defaults_to_usearch() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.vector_backend, "usearch");
    }

    #[test]
    fn validation_vector_backend_usearch_without_feature() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
vector_backend: usearch
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        if cfg!(feature = "usearch") {
            validate(&cfg).unwrap();
        } else {
            let err = validate(&cfg).unwrap_err();
            assert!(
                err.to_string().contains("--features usearch"),
                "got: {}",
                err
            );
        }
    }

    #[test]
    fn vector_quantization_f16_accepted() {
        parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
performance:
  vector_quantization: f16
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap();
    }

    #[test]
    fn vector_quantization_bf16_accepted() {
        parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
performance:
  vector_quantization: bf16
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap();
    }

    #[test]
    fn vector_quantization_invalid_rejected() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
performance:
  vector_quantization: i8
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("vector_quantization"),
            "got: {}",
            err
        );
    }

    #[test]
    fn vector_quantization_none_defaults_to_f32_behavior() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.performance.vector_quantization, None);
    }

    #[test]
    fn vector_index_mode_none_is_valid() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.performance.vector_index_mode, None);
    }

    #[test]
    fn vector_index_mode_load_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  vector_index_mode: load\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn vector_index_mode_mmap_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  vector_index_mode: mmap\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn vector_index_mode_invalid_rejected() {
        let yaml = format!(
            "{}\nperformance:\n  vector_index_mode: load_from_disk\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(
            err.to_string().contains("vector_index_mode"),
            "got: {}",
            err
        );
    }

    // --- BM25 validation ---

    #[test]
    fn bm25_accepted_without_fields() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { method: bm25, weight: 0.2 }",
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn bm25_rejected_with_field_a() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { field_a: x, method: bm25, weight: 0.2 }",
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(err.to_string().contains("match_fields[1]"), "got: {}", err);
    }

    #[test]
    fn bm25_rejected_with_field_b() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { field_b: x, method: bm25, weight: 0.2 }",
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(err.to_string().contains("match_fields[1]"), "got: {}", err);
    }

    #[test]
    fn bm25_multiple_rejected() {
        let yaml = base_yaml_with_match_fields(
            "  - { method: bm25, weight: 0.5 }\n  - { method: bm25, weight: 0.5 }",
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(err.to_string().contains("at most one bm25"), "got: {}", err);
    }

    #[test]
    fn bm25_candidates_lt_top_n_rejected() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
top_n: 20
bm25_candidates: 5
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 0.8 }
  - { method: bm25, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(err.to_string().contains("bm25_candidates"), "got: {}", err);
    }

    #[test]
    fn bm25_always_accepted() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { method: bm25, weight: 0.2 }",
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn bm25_inline_fields_parsed() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 0.8 }
  - method: bm25
    weight: 0.2
    fields:
      - { field_a: name_a, field_b: name_b }
      - { field_a: desc_a, field_b: desc_b }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        let bm25_mf = cfg
            .match_fields
            .iter()
            .find(|mf| mf.method == MatchMethod::Bm25)
            .unwrap();
        let fields = bm25_mf.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].field_a, "name_a");
    }

    #[test]
    fn bm25_inline_and_toplevel_errors() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
bm25_fields:
  - { field_a: top_a, field_b: top_b }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 0.8 }
  - method: bm25
    weight: 0.2
    fields:
      - { field_a: inline_a, field_b: inline_b }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(err.to_string().contains("bm25_fields"), "got: {}", err);
    }

    #[test]
    fn bm25_toplevel_still_works() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
bm25_fields:
  - { field_a: name_a, field_b: name_b }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 0.8 }
  - { method: bm25, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        parse_and_validate(yaml).unwrap();
    }

    // --- min_score_gap ---

    #[test]
    fn min_score_gap_accepted() {
        parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6, min_score_gap: 0.10 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap();
    }

    #[test]
    fn min_score_gap_zero_accepted() {
        parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6, min_score_gap: 0.0 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap();
    }

    #[test]
    fn min_score_gap_negative_rejected() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6, min_score_gap: -0.05 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("min_score_gap"), "got: {}", err);
    }

    #[test]
    fn min_score_gap_one_rejected() {
        let err = parse_and_validate(
            r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6, min_score_gap: 1.0 }
output: { csv_dir_path: /tmp/test/output }
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("min_score_gap"), "got: {}", err);
    }

    #[test]
    fn min_score_gap_absent_is_none() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.thresholds.min_score_gap, None);
    }

    // --- Hooks ---

    #[test]
    fn hooks_empty_command_rejected() {
        let yaml = format!(
            "{}\nhooks:\n  command: \"\"",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(err.to_string().contains("hooks.command"), "got: {}", err);
    }

    #[test]
    fn hooks_valid_command_accepted() {
        let yaml = format!(
            "{}\nhooks:\n  command: \"python hook.py\"",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn hooks_absent_defaults_to_none() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert!(cfg.hooks.command.is_none());
    }

    // --- Remote encoder (XOR with model) ---

    #[test]
    fn remote_encoder_xor_neither_set_rejected() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(
            err.to_string()
                .contains("embeddings.model or embeddings.remote_encoder_cmd"),
            "got: {err}"
        );
    }

    #[test]
    fn remote_encoder_xor_both_set_rejected() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings:
  model: m
  remote_encoder_cmd: "python enc.py"
  a_cache_dir: i
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
performance:
  encoder_pool_size: 4
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(err.to_string().contains("exactly one of"), "got: {err}");
    }

    #[test]
    fn remote_encoder_requires_pool_size() {
        // remote_encoder_cmd set but no encoder_pool_size → reject.
        // Defaults do NOT fill in a pool size when remote_encoder_cmd is
        // set, so the normal loader path surfaces the error.
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings:
  remote_encoder_cmd: "python enc.py"
  a_cache_dir: i
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(err.to_string().contains("encoder_pool_size"), "got: {err}");
    }

    #[test]
    fn remote_encoder_valid_config_accepted() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings:
  remote_encoder_cmd: "python enc.py"
  a_cache_dir: i
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
performance:
  encoder_pool_size: 4
  encoder_call_timeout_ms: 30000
"#;
        parse_and_validate(yaml).unwrap();
    }

    #[test]
    fn remote_encoder_zero_timeout_rejected() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings:
  remote_encoder_cmd: "python enc.py"
  a_cache_dir: i
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
performance:
  encoder_pool_size: 4
  encoder_call_timeout_ms: 0
"#;
        let err = parse_and_validate(yaml).unwrap_err();
        assert!(
            err.to_string().contains("encoder_call_timeout_ms"),
            "got: {err}"
        );
    }

    // --- Encoder ---

    #[test]
    fn encoder_device_cpu_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_device: cpu\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn encoder_device_invalid_rejected() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_device: tpu\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(err.to_string().contains("encoder_device"), "got: {}", err);
    }

    #[test]
    fn encoder_device_gpu_without_feature() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_device: gpu\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        if cfg!(feature = "gpu-encode") {
            parse_and_validate(&yaml).unwrap();
        } else {
            let err = parse_and_validate(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("--features gpu-encode"),
                "got: {}",
                err
            );
        }
    }

    #[test]
    fn encoder_batch_size_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_batch_size: 128\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        parse_and_validate(&yaml).unwrap();
    }

    #[test]
    fn encoder_batch_size_zero_rejected() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_batch_size: 0\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let err = parse_and_validate(&yaml).unwrap_err();
        assert!(
            err.to_string().contains("encoder_batch_size"),
            "got: {}",
            err
        );
    }

    #[test]
    fn encoder_device_gpu_accepted_with_feature() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_device: gpu\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        if cfg!(feature = "gpu-encode") {
            parse_and_validate(&yaml).unwrap();
        }
    }

    #[test]
    fn encoder_device_cpu_with_batch_size_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  encoder_device: cpu\n  encoder_batch_size: 32\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        parse_and_validate(&yaml).unwrap();
    }

    // --- Enroll-mode validation ---

    #[test]
    fn enroll_config_basic_parse() {
        parse_and_validate_enroll(
            r#"
job:
  name: enroll_test
embeddings:
  model: all-MiniLM-L6-v2
  cache_dir: cache/emb
match_fields:
  - { field: legal_name, method: embedding, weight: 0.55 }
  - { field: country_code, method: exact, weight: 0.25 }
  - { field: short_name, method: fuzzy, weight: 0.20 }
thresholds:
  auto_match: 0.85
  review_floor: 0.60
"#,
        )
        .unwrap();
    }

    #[test]
    fn enroll_config_with_dataset() {
        parse_and_validate_enroll(
            r#"
job:
  name: enroll_with_data
dataset:
  path: reference.csv
  id_field: entity_id
embeddings:
  model: all-MiniLM-L6-v2
  cache_dir: cache/emb
match_fields:
  - { field: name, method: exact, weight: 1.0 }
thresholds:
  auto_match: 0.85
  review_floor: 0.60
"#,
        )
        .unwrap();
    }

    #[test]
    fn enroll_config_with_blocking() {
        parse_and_validate_enroll(
            r#"
job:
  name: enroll_blocking
embeddings:
  model: m
  cache_dir: c
blocking:
  enabled: true
  operator: and
  fields:
    - { field: country }
match_fields:
  - { field: name, method: exact, weight: 1.0 }
thresholds:
  auto_match: 0.85
  review_floor: 0.60
"#,
        )
        .unwrap();
    }

    #[test]
    fn enroll_config_weight_sum_rejected() {
        let err = parse_and_validate_enroll(
            r#"
job:
  name: bad_weights
embeddings:
  model: m
  cache_dir: c
match_fields:
  - { field: f, method: exact, weight: 0.5 }
thresholds:
  auto_match: 0.85
  review_floor: 0.60
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("weights sum"), "got: {}", err);
    }

    #[test]
    fn enroll_config_no_dataset_allowed() {
        parse_and_validate_enroll(
            r#"
job:
  name: empty_pool
embeddings:
  model: m
  cache_dir: c
match_fields:
  - { field: f, method: exact, weight: 1.0 }
thresholds:
  auto_match: 0.85
  review_floor: 0.60
"#,
        )
        .unwrap();
    }

    #[test]
    fn enroll_config_bm25_with_inline_fields() {
        parse_and_validate_enroll(
            r#"
job:
  name: enroll_bm25
embeddings:
  model: m
  cache_dir: c
match_fields:
  - { field: name, method: embedding, weight: 0.8 }
  - method: bm25
    weight: 0.2
    fields:
      - { field: name }
      - { field: address }
thresholds:
  auto_match: 0.85
  review_floor: 0.60
"#,
        )
        .unwrap();
    }
}
