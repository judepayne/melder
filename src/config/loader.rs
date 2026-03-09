//! Config loading: parse YAML, apply defaults, validate, derive fields.

use std::collections::HashSet;
use std::path::Path;

use crate::error::ConfigError;

use super::schema::{BlockingFieldPair, Config, DatasetConfig};

/// Valid match methods.
const VALID_METHODS: &[&str] = &["exact", "fuzzy", "embedding", "numeric"];

/// Valid fuzzy scorers.
const VALID_SCORERS: &[&str] = &["wratio", "partial_ratio", "token_sort", "ratio"];

/// Valid candidate selection methods.
const VALID_CANDIDATE_METHODS: &[&str] = &["exact", "fuzzy", "embedding"];

/// Valid cross-map backends.
const VALID_BACKENDS: &[&str] = &["local", "redis"];

/// Valid data formats.
const VALID_FORMATS: &[&str] = &["csv", "parquet", "jsonl"];

/// Load, parse, validate, and return a fully-populated `Config`.
pub fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let data = std::fs::read_to_string(path)?;
    let mut cfg: Config = serde_yaml::from_str(&data)?;

    normalise_blocking(&mut cfg);
    apply_defaults(&mut cfg);
    validate(&cfg)?;
    derive_required_fields(&mut cfg);

    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

fn apply_defaults(cfg: &mut Config) {
    // candidates — infer enabled from presence of config fields
    let candidates_has_config = cfg.candidates.field_a.is_some()
        || cfg.candidates.field_b.is_some()
        || cfg.candidates.method.is_some();
    let candidates_enabled = cfg.candidates.enabled.unwrap_or(candidates_has_config);
    if candidates_enabled {
        if cfg.candidates.n.is_none() || cfg.candidates.n == Some(0) {
            cfg.candidates.n = Some(10);
        }
        if cfg.candidates.method.as_deref() == Some("fuzzy")
            && cfg.candidates.scorer.as_deref().unwrap_or("").is_empty()
        {
            cfg.candidates.scorer = Some("wratio".into());
        }
    }

    // live
    if cfg.live.top_n.is_none() || cfg.live.top_n == Some(0) {
        cfg.live.top_n = Some(5);
    }
    if cfg.live.crossmap_flush_secs.is_none() || cfg.live.crossmap_flush_secs == Some(0) {
        cfg.live.crossmap_flush_secs = Some(5);
    }

    // performance — merge from old locations for backward compat, then default.
    // Priority: performance.encoder_pool_size > live.encoder_pool_size > default(1)
    if cfg.performance.encoder_pool_size.is_none() || cfg.performance.encoder_pool_size == Some(0) {
        cfg.performance.encoder_pool_size = match cfg.live.encoder_pool_size {
            Some(v) if v > 0 => Some(v),
            _ => Some(1),
        };
    }
    // Priority: performance.workers > top-level workers > default(4)
    if cfg.performance.workers.is_none() || cfg.performance.workers == Some(0) {
        cfg.performance.workers = match cfg.workers {
            Some(v) if v > 0 => Some(v),
            _ => Some(4),
        };
    }
    // Sync old fields so any code still reading them gets consistent values
    cfg.live.encoder_pool_size = cfg.performance.encoder_pool_size;
    cfg.workers = cfg.performance.workers;

    // cross_map backend already has serde default, but belt-and-suspenders
    if cfg.cross_map.backend.is_empty() {
        cfg.cross_map.backend = "local".into();
    }

    // fuzzy scorer defaults
    for mf in &mut cfg.match_fields {
        if mf.method == "fuzzy" && mf.scorer.as_deref().unwrap_or("").is_empty() {
            mf.scorer = Some("wratio".into());
        }
    }
}

// ---------------------------------------------------------------------------
// Blocking normalisation
// ---------------------------------------------------------------------------

fn normalise_blocking(cfg: &mut Config) {
    if cfg.blocking.fields.is_empty() {
        if let Some(ref fa) = cfg.blocking.field_a {
            let fb = cfg.blocking.field_b.clone().unwrap_or_default();
            cfg.blocking.fields.push(BlockingFieldPair {
                field_a: fa.clone(),
                field_b: fb,
            });
        }
    }
    if cfg.blocking.operator.is_empty() {
        cfg.blocking.operator = "and".into();
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate(cfg: &Config) -> Result<(), ConfigError> {
    // 1. job.name
    require_non_empty(&cfg.job.name, "job.name")?;

    // 2-5. datasets
    validate_dataset(&cfg.datasets.a, "datasets.a")?;
    validate_dataset(&cfg.datasets.b, "datasets.b")?;

    // 6. cross_map.backend
    require_one_of(&cfg.cross_map.backend, VALID_BACKENDS, "cross_map.backend")?;

    // 7. cross_map.path required if local
    if cfg.cross_map.backend == "local" {
        if cfg.cross_map.path.as_deref().unwrap_or("").is_empty() {
            return Err(ConfigError::MissingField {
                field: "cross_map.path".into(),
            });
        }
    }

    // 8. cross_map.redis_url required if redis
    if cfg.cross_map.backend == "redis" {
        if cfg.cross_map.redis_url.as_deref().unwrap_or("").is_empty() {
            return Err(ConfigError::MissingField {
                field: "cross_map.redis_url".into(),
            });
        }
    }

    // 9-10. cross_map id fields
    require_non_empty(&cfg.cross_map.a_id_field, "cross_map.a_id_field")?;
    require_non_empty(&cfg.cross_map.b_id_field, "cross_map.b_id_field")?;

    // 11-12. embeddings
    require_non_empty(&cfg.embeddings.model, "embeddings.model")?;
    require_non_empty(&cfg.embeddings.a_index_cache, "embeddings.a_index_cache")?;

    // 14a. common_id_field: if set on one side, must be set on both
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

    // 14. at least one match_fields
    if cfg.match_fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: "at least one match field required".into(),
        });
    }

    // 15-19. validate each match field + accumulate weight sum
    let mut weight_sum = 0.0_f64;
    for (i, mf) in cfg.match_fields.iter().enumerate() {
        let prefix = format!("match_fields[{}]", i);
        require_non_empty(&mf.field_a, &format!("{}.field_a", prefix))?;
        require_non_empty(&mf.field_b, &format!("{}.field_b", prefix))?;
        require_one_of(&mf.method, VALID_METHODS, &format!("{}.method", prefix))?;

        if mf.method == "fuzzy" {
            if let Some(ref scorer) = mf.scorer {
                if !scorer.is_empty() {
                    require_one_of(scorer, VALID_SCORERS, &format!("{}.scorer", prefix))?;
                }
            }
        }

        if mf.weight <= 0.0 {
            return Err(ConfigError::InvalidValue {
                field: format!("{}.weight", prefix),
                message: "must be > 0".into(),
            });
        }
        weight_sum += mf.weight;
    }

    // 20. weights sum to 1.0
    if (weight_sum - 1.0).abs() > 0.001 {
        return Err(ConfigError::WeightSum { sum: weight_sum });
    }

    // 21. thresholds.auto_match in (0, 1]
    if cfg.thresholds.auto_match <= 0.0 || cfg.thresholds.auto_match > 1.0 {
        return Err(ConfigError::InvalidValue {
            field: "thresholds.auto_match".into(),
            message: "must be in range (0, 1]".into(),
        });
    }

    // 22. thresholds.review_floor in [0, 1)
    if cfg.thresholds.review_floor < 0.0 || cfg.thresholds.review_floor >= 1.0 {
        return Err(ConfigError::InvalidValue {
            field: "thresholds.review_floor".into(),
            message: "must be in range [0, 1)".into(),
        });
    }

    // 23. auto_match > review_floor
    if cfg.thresholds.auto_match <= cfg.thresholds.review_floor {
        return Err(ConfigError::InvalidValue {
            field: "thresholds".into(),
            message: format!(
                "auto_match ({:.2}) must be greater than review_floor ({:.2})",
                cfg.thresholds.auto_match, cfg.thresholds.review_floor
            ),
        });
    }

    // 24-26. output paths
    require_non_empty(&cfg.output.results_path, "output.results_path")?;
    require_non_empty(&cfg.output.review_path, "output.review_path")?;
    require_non_empty(&cfg.output.unmatched_path, "output.unmatched_path")?;

    // 27. blocking
    if cfg.blocking.enabled {
        if cfg.blocking.fields.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "blocking".into(),
                message: "at least one field pair required when enabled".into(),
            });
        }
        for (i, fp) in cfg.blocking.fields.iter().enumerate() {
            let prefix = format!("blocking.fields[{}]", i);
            require_non_empty(&fp.field_a, &format!("{}.field_a", prefix))?;
            require_non_empty(&fp.field_b, &format!("{}.field_b", prefix))?;
        }
        let op = cfg.blocking.operator.to_lowercase();
        if op != "and" && op != "or" {
            return Err(ConfigError::InvalidValue {
                field: "blocking.operator".into(),
                message: format!("must be \"and\" or \"or\", got {:?}", cfg.blocking.operator),
            });
        }
    }

    // 28. candidates
    // If the candidates section is entirely omitted (enabled=None, no fields/method),
    // treat as disabled. If the section is present with fields, treat as enabled.
    let candidates_has_config = cfg.candidates.field_a.is_some()
        || cfg.candidates.field_b.is_some()
        || cfg.candidates.method.is_some();
    let candidates_enabled = cfg.candidates.enabled.unwrap_or(candidates_has_config);
    if candidates_enabled {
        let fa = cfg.candidates.field_a.as_deref().unwrap_or("");
        let fb = cfg.candidates.field_b.as_deref().unwrap_or("");
        if fa.is_empty() {
            return Err(ConfigError::MissingField {
                field: "candidates.field_a".into(),
            });
        }
        if fb.is_empty() {
            return Err(ConfigError::MissingField {
                field: "candidates.field_b".into(),
            });
        }
        let method = cfg.candidates.method.as_deref().unwrap_or("");
        if method.is_empty() {
            return Err(ConfigError::MissingField {
                field: "candidates.method".into(),
            });
        }
        require_one_of(method, VALID_CANDIDATE_METHODS, "candidates.method")?;
        if method == "fuzzy" {
            if let Some(ref scorer) = cfg.candidates.scorer {
                if !scorer.is_empty() {
                    require_one_of(scorer, VALID_SCORERS, "candidates.scorer")?;
                }
            }
        }
    }

    // 29-31. performance + live constraints
    if let Some(pool) = cfg.performance.encoder_pool_size {
        if pool < 1 {
            return Err(ConfigError::InvalidValue {
                field: "performance.encoder_pool_size".into(),
                message: "must be >= 1".into(),
            });
        }
    }
    if let Some(w) = cfg.performance.workers {
        if w < 1 {
            return Err(ConfigError::InvalidValue {
                field: "performance.workers".into(),
                message: "must be >= 1".into(),
            });
        }
    }
    if let Some(secs) = cfg.live.crossmap_flush_secs {
        if secs < 1 {
            return Err(ConfigError::InvalidValue {
                field: "live.crossmap_flush_secs".into(),
                message: "must be >= 1".into(),
            });
        }
    }

    Ok(())
}

fn validate_dataset(d: &DatasetConfig, prefix: &str) -> Result<(), ConfigError> {
    require_non_empty(&d.path, &format!("{}.path", prefix))?;
    require_non_empty(&d.id_field, &format!("{}.id_field", prefix))?;

    // Format: infer from extension if not explicit
    match d.format.as_deref() {
        Some(fmt) if !fmt.is_empty() => {
            require_one_of(fmt, VALID_FORMATS, &format!("{}.format", prefix))?;
        }
        _ => {
            // Infer — just validate that the extension is recognisable
            infer_format(&d.path, prefix)?;
        }
    }

    Ok(())
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

// ---------------------------------------------------------------------------
// Derived fields
// ---------------------------------------------------------------------------

fn derive_required_fields(cfg: &mut Config) {
    let mut seen_a = HashSet::new();
    let mut seen_b = HashSet::new();

    // id fields
    seen_a.insert(cfg.datasets.a.id_field.clone());
    seen_b.insert(cfg.datasets.b.id_field.clone());

    // common_id fields
    if let Some(ref cid) = cfg.datasets.a.common_id_field {
        if !cid.is_empty() {
            seen_a.insert(cid.clone());
        }
    }
    if let Some(ref cid) = cfg.datasets.b.common_id_field {
        if !cid.is_empty() {
            seen_b.insert(cid.clone());
        }
    }

    // match fields
    for mf in &cfg.match_fields {
        seen_a.insert(mf.field_a.clone());
        seen_b.insert(mf.field_b.clone());
    }

    // output mapping (from side A)
    for om in &cfg.output_mapping {
        seen_a.insert(om.from.clone());
    }

    // blocking fields
    if cfg.blocking.enabled {
        for fp in &cfg.blocking.fields {
            seen_a.insert(fp.field_a.clone());
            seen_b.insert(fp.field_b.clone());
        }
    }

    let mut a: Vec<String> = seen_a.into_iter().collect();
    let mut b: Vec<String> = seen_b.into_iter().collect();
    a.sort();
    b.sort();
    cfg.required_fields_a = a;
    cfg.required_fields_b = b;
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

    #[test]
    fn load_bench_live() {
        let cfg = load_config(Path::new("testdata/configs/bench_live.yaml")).unwrap();
        assert_eq!(cfg.job.name, "bench_live_10000x10000");
        assert_eq!(cfg.performance.encoder_pool_size, Some(4));
        assert_eq!(cfg.performance.workers, Some(4));
        assert_eq!(cfg.live.top_n, Some(5));
        assert_eq!(cfg.live.crossmap_flush_secs, Some(5)); // default applied
    }

    #[test]
    fn load_bench1000x1000() {
        let cfg = load_config(Path::new("testdata/configs/bench1000x1000.yaml")).unwrap();
        assert_eq!(cfg.candidates.n, Some(10));
        assert_eq!(cfg.candidates.scorer, Some("wratio".into()));
        assert_eq!(cfg.performance.encoder_pool_size, Some(4));
        assert_eq!(cfg.performance.workers, Some(8));
    }

    #[test]
    fn load_counterparty_recon_with_sidecar() {
        let cfg = load_config(Path::new("testdata/configs/counterparty_recon.yaml")).unwrap();
        assert!(cfg.sidecar.is_some());
        assert_eq!(cfg.output_mapping.len(), 4);
        assert_eq!(cfg.performance.workers, Some(4));
        assert_eq!(cfg.performance.encoder_pool_size, Some(2));
    }

    #[test]
    fn derived_fields_bench_live() {
        let cfg = load_config(Path::new("testdata/configs/bench_live.yaml")).unwrap();
        // A side should include: entity_id, legal_name, short_name, country_code, lei
        assert!(cfg.required_fields_a.contains(&"entity_id".to_string()));
        assert!(cfg.required_fields_a.contains(&"legal_name".to_string()));
        assert!(cfg.required_fields_a.contains(&"short_name".to_string()));
        assert!(cfg.required_fields_a.contains(&"country_code".to_string()));
        assert!(cfg.required_fields_a.contains(&"lei".to_string()));
        // B side should include: counterparty_id, counterparty_name, domicile, lei_code
        assert!(cfg
            .required_fields_b
            .contains(&"counterparty_id".to_string()));
        assert!(cfg
            .required_fields_b
            .contains(&"counterparty_name".to_string()));
        assert!(cfg.required_fields_b.contains(&"domicile".to_string()));
        assert!(cfg.required_fields_b.contains(&"lei_code".to_string()));
    }

    #[test]
    fn legacy_blocking_normalised() {
        // counterparty_recon.yaml uses legacy single field_a/field_b blocking
        let cfg = load_config(Path::new("testdata/configs/counterparty_recon.yaml")).unwrap();
        assert!(cfg.blocking.enabled);
        assert_eq!(cfg.blocking.fields.len(), 1);
        assert_eq!(cfg.blocking.fields[0].field_a, "country_code");
        assert_eq!(cfg.blocking.fields[0].field_b, "domicile");
        assert_eq!(cfg.blocking.operator, "and");
    }

    #[test]
    fn format_inference() {
        assert_eq!(infer_format("data.csv", "test").unwrap(), "csv");
        assert_eq!(infer_format("data.tsv", "test").unwrap(), "csv");
        assert_eq!(infer_format("data.parquet", "test").unwrap(), "parquet");
        assert_eq!(infer_format("data.jsonl", "test").unwrap(), "jsonl");
        assert_eq!(infer_format("data.ndjson", "test").unwrap(), "jsonl");

        // These should error
        assert!(infer_format("data.json", "test").is_err());
        assert!(infer_format("data.xlsx", "test").is_err());
        assert!(infer_format("data.xls", "test").is_err());
        assert!(infer_format("data.dat", "test").is_err());
    }

    #[test]
    fn validation_missing_job_name() {
        let yaml = r#"
job:
  name: ""
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_index_cache: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
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
embeddings: { model: m, a_index_cache: i }
match_fields:
  - { field_a: f, field_b: f, method: magic, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("match_fields[0].method"),
            "got: {}",
            err
        );
    }

    #[test]
    fn validation_invalid_backend() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: s3, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_index_cache: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("cross_map.backend"),
            "got: {}",
            err
        );
    }

    #[test]
    fn validation_weight_sum() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_index_cache: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 0.5 }
  - { field_a: g, field_b: g, method: fuzzy, weight: 0.45 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("weights sum"), "got: {}", err);
    }

    #[test]
    fn validation_auto_match_lt_review_floor() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_index_cache: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.50, review_floor: 0.80 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("thresholds"), "got: {}", err);
    }

    #[test]
    fn validation_blocking_invalid_operator() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_index_cache: i }
blocking:
  enabled: true
  operator: xor
  fields:
    - { field_a: c, field_b: c }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("blocking.operator"),
            "got: {}",
            err
        );
    }

    #[test]
    fn fuzzy_scorer_defaults_to_wratio() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_index_cache: i }
match_fields:
  - { field_a: f, field_b: f, method: fuzzy, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.match_fields[0].scorer, Some("wratio".into()));
    }
}
