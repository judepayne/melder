//! Config loading: parse YAML, apply defaults, validate, derive fields.

use std::collections::HashSet;
use std::path::Path;

use crate::error::ConfigError;

use super::enroll_schema::EnrollConfig;
use super::schema::{BlockingFieldPair, Config, DatasetConfig};

/// Valid match methods.
const VALID_METHODS: &[&str] = &["exact", "fuzzy", "embedding", "numeric", "bm25", "synonym"];

/// Valid fuzzy scorers.
const VALID_SCORERS: &[&str] = &["wratio", "partial_ratio", "token_sort", "ratio"];

/// Valid cross-map backends.
const VALID_BACKENDS: &[&str] = &["local"];

/// Valid vector storage backends.
const VALID_VECTOR_BACKENDS: &[&str] = &["flat", "usearch"];

/// Valid vector quantization types (for usearch index storage).
const VALID_VECTOR_QUANTIZATIONS: &[&str] = &["f32", "f16", "bf16"];

/// Valid vector index load modes.
const VALID_VECTOR_INDEX_MODES: &[&str] = &["load", "mmap"];

/// Valid data formats.
const VALID_FORMATS: &[&str] = &["csv", "parquet", "jsonl"];

/// Load, parse, validate, and return a fully-populated `Config`.
pub fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let data = std::fs::read_to_string(path)?;
    let mut cfg: Config = serde_yaml::from_str(&data)?;

    normalise_blocking(&mut cfg);
    apply_defaults(&mut cfg);
    deprecation_warnings(&cfg);
    validate(&cfg)?;
    derive_bm25_fields(&mut cfg);
    derive_synonym_fields(&mut cfg);
    derive_required_fields(&mut cfg);

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

    apply_defaults(&mut cfg);
    validate_enroll(&cfg)?;
    derive_bm25_fields(&mut cfg);
    derive_synonym_fields(&mut cfg);
    derive_required_fields(&mut cfg);

    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

fn apply_defaults(cfg: &mut Config) {
    // top_n: default 5
    if cfg.top_n.is_none() || cfg.top_n == Some(0) {
        cfg.top_n = Some(5);
    }

    // live
    if cfg.live.crossmap_flush_secs.is_none() || cfg.live.crossmap_flush_secs == Some(0) {
        cfg.live.crossmap_flush_secs = Some(5);
    }

    // Default encoder_pool_size to 1 if not set.
    if cfg.performance.encoder_pool_size.is_none() || cfg.performance.encoder_pool_size == Some(0) {
        cfg.performance.encoder_pool_size = Some(1);
    }

    // cross_map backend already has serde default, but belt-and-suspenders
    if cfg.cross_map.backend.is_empty() {
        cfg.cross_map.backend = "local".into();
    }

    // ann_candidates: default 50
    if cfg.ann_candidates.is_none() || cfg.ann_candidates == Some(0) {
        cfg.ann_candidates = Some(50);
    }

    // bm25_candidates: default 10
    if cfg.bm25_candidates.is_none() || cfg.bm25_candidates == Some(0) {
        cfg.bm25_candidates = Some(10);
    }

    // fuzzy scorer defaults
    for mf in &mut cfg.match_fields {
        if mf.method == "fuzzy" && mf.scorer.as_deref().unwrap_or("").is_empty() {
            mf.scorer = Some("wratio".into());
        }
    }
}

// ---------------------------------------------------------------------------
// Deprecation warnings
// ---------------------------------------------------------------------------

fn deprecation_warnings(cfg: &Config) {
    if cfg.bm25_commit_batch_size.is_some() {
        eprintln!(
            "WARNING: bm25_commit_batch_size is deprecated and ignored. \
             SimpleBm25 has instant write visibility — no commit batching needed."
        );
    }
}

// ---------------------------------------------------------------------------
// Blocking normalisation
// ---------------------------------------------------------------------------

fn normalise_blocking(cfg: &mut Config) {
    if cfg.blocking.fields.is_empty()
        && let Some(ref fa) = cfg.blocking.field_a
    {
        let fb = cfg.blocking.field_b.clone().unwrap_or_default();
        cfg.blocking.fields.push(BlockingFieldPair {
            field_a: fa.clone(),
            field_b: fb,
        });
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
    if cfg.cross_map.backend == "local" && cfg.cross_map.path.as_deref().unwrap_or("").is_empty() {
        return Err(ConfigError::MissingField {
            field: "cross_map.path".into(),
        });
    }

    // 9-10. cross_map id fields
    require_non_empty(&cfg.cross_map.a_id_field, "cross_map.a_id_field")?;
    require_non_empty(&cfg.cross_map.b_id_field, "cross_map.b_id_field")?;

    // 11-12. embeddings
    require_non_empty(&cfg.embeddings.model, "embeddings.model")?;
    require_non_empty(&cfg.embeddings.a_cache_dir, "embeddings.a_cache_dir")?;

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
    let mut bm25_count = 0usize;
    for (i, mf) in cfg.match_fields.iter().enumerate() {
        let prefix = format!("match_fields[{}]", i);
        require_one_of(&mf.method, VALID_METHODS, &format!("{}.method", prefix))?;

        if mf.method == "bm25" {
            // BM25 takes no field_a/field_b — reject if set.
            if !mf.field_a.is_empty() || !mf.field_b.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: prefix.to_string(),
                    message: "bm25 method operates across all text fields; field_a and field_b must be omitted".into(),
                });
            }
            bm25_count += 1;
        } else {
            // Non-BM25 methods require field_a and field_b.
            require_non_empty(&mf.field_a, &format!("{}.field_a", prefix))?;
            require_non_empty(&mf.field_b, &format!("{}.field_b", prefix))?;
        }

        if mf.method == "fuzzy"
            && let Some(ref scorer) = mf.scorer
            && !scorer.is_empty()
        {
            require_one_of(scorer, VALID_SCORERS, &format!("{}.scorer", prefix))?;
        }

        if mf.weight < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: format!("{}.weight", prefix),
                message: "must be >= 0".into(),
            });
        }
        // Synonym weights are excluded from the sum-to-1.0 check because
        // the scorer excludes synonym from the normalisation denominator
        // when it scores 0.0 (which is ~99% of pairs). This makes synonym
        // purely additive — it only affects the composite when it fires.
        if mf.method != "synonym" {
            weight_sum += mf.weight;
        }
    }

    // At most one BM25 entry allowed.
    if bm25_count > 1 {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: format!("at most one bm25 entry allowed, found {}", bm25_count),
        });
    }

    // Check for conflicting BM25 field definitions (inline + top-level).
    let has_inline_bm25_fields = cfg
        .match_fields
        .iter()
        .any(|mf| mf.method == "bm25" && mf.fields.is_some());
    if has_inline_bm25_fields && !cfg.bm25_fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "bm25_fields".into(),
            message: "BM25 fields defined both inline on the match_fields entry and as \
                      a top-level bm25_fields section — use one or the other"
                .into(),
        });
    }

    // Validate inline BM25 fields (if provided on the match_fields entry).
    if let Some(bm25_mf) = cfg.match_fields.iter().find(|mf| mf.method == "bm25")
        && let Some(ref fields) = bm25_mf.fields
    {
        for (i, pair) in fields.iter().enumerate() {
            let prefix = format!("match_fields[bm25].fields[{}]", i);
            require_non_empty(&pair.field_a, &format!("{}.field_a", prefix))?;
            require_non_empty(&pair.field_b, &format!("{}.field_b", prefix))?;
        }
    }

    // Validate explicit top-level bm25_fields entries (if provided).
    for (i, pair) in cfg.bm25_fields.iter().enumerate() {
        let prefix = format!("bm25_fields[{}]", i);
        require_non_empty(&pair.field_a, &format!("{}.field_a", prefix))?;
        require_non_empty(&pair.field_b, &format!("{}.field_b", prefix))?;
    }

    // Candidate count constraints — each generator is independent.
    let has_bm25 = bm25_count > 0;
    let has_embedding = cfg.match_fields.iter().any(|mf| mf.method == "embedding");
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

    // 23b. min_score_gap in [0.0, 1.0) if set
    if let Some(gap) = cfg.thresholds.min_score_gap
        && !(0.0..1.0).contains(&gap)
    {
        return Err(ConfigError::InvalidValue {
            field: "thresholds.min_score_gap".into(),
            message: "must be in range [0.0, 1.0)".into(),
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

    // 28. exact_prefilter
    if cfg.exact_prefilter.enabled {
        if cfg.exact_prefilter.fields.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "exact_prefilter".into(),
                message: "at least one field pair required when enabled".into(),
            });
        }
        for (i, fp) in cfg.exact_prefilter.fields.iter().enumerate() {
            let prefix = format!("exact_prefilter.fields[{}]", i);
            require_non_empty(&fp.field_a, &format!("{}.field_a", prefix))?;
            require_non_empty(&fp.field_b, &format!("{}.field_b", prefix))?;
        }
    }

    // 29. performance + live constraints
    if let Some(pool) = cfg.performance.encoder_pool_size
        && pool < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "performance.encoder_pool_size".into(),
            message: "must be >= 1".into(),
        });
    }
    if let Some(secs) = cfg.live.crossmap_flush_secs
        && secs < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "live.crossmap_flush_secs".into(),
            message: "must be >= 1".into(),
        });
    }

    // 29. vector_backend
    require_one_of(&cfg.vector_backend, VALID_VECTOR_BACKENDS, "vector_backend")?;
    if cfg.vector_backend == "usearch" && !cfg!(feature = "usearch") {
        return Err(ConfigError::InvalidValue {
            field: "vector_backend".into(),
            message: "usearch backend requires building with --features usearch".into(),
        });
    }

    // 30. vector_quantization
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

    // 30b. synonym_dictionary
    if let Some(ref sd) = cfg.synonym_dictionary {
        require_non_empty(&sd.path, "synonym_dictionary.path")?;
    }

    // 31. vector_index_mode
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

    // 32. hooks
    validate_hooks(cfg)?;

    Ok(())
}

/// Validation for enroll mode — skips B-side datasets, crossmap, and output paths.
fn validate_enroll(cfg: &Config) -> Result<(), ConfigError> {
    // 1. job.name
    require_non_empty(&cfg.job.name, "job.name")?;

    // 2. dataset (A side) — only required if a path is set (dataset is optional)
    if !cfg.datasets.a.path.is_empty() {
        validate_dataset(&cfg.datasets.a, "dataset")?;
    } else if cfg.datasets.a.id_field.is_empty() {
        // No dataset provided — id_field still needed for enroll requests.
        // The user must set id_field somewhere. For now, allow empty (set via dataset).
    }

    // 3. embeddings
    require_non_empty(&cfg.embeddings.model, "embeddings.model")?;
    require_non_empty(&cfg.embeddings.a_cache_dir, "embeddings.cache_dir")?;

    // 4. at least one match_fields
    if cfg.match_fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: "at least one match field required".into(),
        });
    }

    // 5. validate each match field + accumulate weight sum
    let mut weight_sum = 0.0_f64;
    let mut bm25_count = 0usize;
    for (i, mf) in cfg.match_fields.iter().enumerate() {
        let prefix = format!("match_fields[{}]", i);
        require_one_of(&mf.method, VALID_METHODS, &format!("{}.method", prefix))?;

        if mf.method == "bm25" {
            if !mf.field_a.is_empty() || !mf.field_b.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: prefix.to_string(),
                    message: "bm25 method operates across all text fields; field must be omitted"
                        .into(),
                });
            }
            bm25_count += 1;
        } else {
            // In enroll mode, field_a == field_b (both set by into_config)
            require_non_empty(&mf.field_a, &format!("{}.field", prefix))?;
        }

        if mf.method == "fuzzy"
            && let Some(ref scorer) = mf.scorer
            && !scorer.is_empty()
        {
            require_one_of(scorer, VALID_SCORERS, &format!("{}.scorer", prefix))?;
        }

        if mf.weight < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: format!("{}.weight", prefix),
                message: "must be >= 0".into(),
            });
        }
        if mf.method != "synonym" {
            weight_sum += mf.weight;
        }
    }

    if bm25_count > 1 {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: format!("at most one bm25 entry allowed, found {}", bm25_count),
        });
    }

    // Check for conflicting BM25 field definitions
    let has_inline_bm25_fields = cfg
        .match_fields
        .iter()
        .any(|mf| mf.method == "bm25" && mf.fields.is_some());
    if has_inline_bm25_fields && !cfg.bm25_fields.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "bm25_fields".into(),
            message: "BM25 fields defined both inline and as a top-level section".into(),
        });
    }

    // Candidate count constraints
    let has_bm25 = bm25_count > 0;
    let has_embedding = cfg.match_fields.iter().any(|mf| mf.method == "embedding");
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

    // 6. weights sum to 1.0
    if (weight_sum - 1.0).abs() > 0.001 {
        return Err(ConfigError::WeightSum { sum: weight_sum });
    }

    // 7. thresholds
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

    // 8. blocking
    if cfg.blocking.enabled {
        if cfg.blocking.fields.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "blocking".into(),
                message: "at least one field required when enabled".into(),
            });
        }
        for (i, fp) in cfg.blocking.fields.iter().enumerate() {
            let prefix = format!("blocking.fields[{}]", i);
            require_non_empty(&fp.field_a, &format!("{}.field", prefix))?;
        }
        let op = cfg.blocking.operator.to_lowercase();
        if op != "and" && op != "or" {
            return Err(ConfigError::InvalidValue {
                field: "blocking.operator".into(),
                message: format!("must be \"and\" or \"or\", got {:?}", cfg.blocking.operator),
            });
        }
    }

    // 9. exact_prefilter
    if cfg.exact_prefilter.enabled {
        if cfg.exact_prefilter.fields.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "exact_prefilter".into(),
                message: "at least one field required when enabled".into(),
            });
        }
        for (i, fp) in cfg.exact_prefilter.fields.iter().enumerate() {
            let prefix = format!("exact_prefilter.fields[{}]", i);
            require_non_empty(&fp.field_a, &format!("{}.field", prefix))?;
        }
    }

    // 10. performance
    if let Some(pool) = cfg.performance.encoder_pool_size
        && pool < 1
    {
        return Err(ConfigError::InvalidValue {
            field: "performance.encoder_pool_size".into(),
            message: "must be >= 1".into(),
        });
    }

    // 11. vector_backend
    require_one_of(&cfg.vector_backend, VALID_VECTOR_BACKENDS, "vector_backend")?;
    if cfg.vector_backend == "usearch" && !cfg!(feature = "usearch") {
        return Err(ConfigError::InvalidValue {
            field: "vector_backend".into(),
            message: "usearch backend requires building with --features usearch".into(),
        });
    }

    // 12. vector_quantization
    if let Some(ref vq) = cfg.performance.vector_quantization {
        require_one_of(
            vq,
            VALID_VECTOR_QUANTIZATIONS,
            "performance.vector_quantization",
        )?;
    }

    // 13. synonym_dictionary
    if let Some(ref sd) = cfg.synonym_dictionary {
        require_non_empty(&sd.path, "synonym_dictionary.path")?;
    }

    // 14. vector_index_mode
    if let Some(ref vim) = cfg.performance.vector_index_mode {
        require_one_of(
            vim,
            VALID_VECTOR_INDEX_MODES,
            "performance.vector_index_mode",
        )?;
    }

    // hooks
    validate_hooks(cfg)?;

    Ok(())
}

/// Validate hooks configuration (shared by match and enroll validators).
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

/// Populate `bm25_fields` from the best available source.
///
/// Priority:
/// 1. Inline `fields` on the BM25 match_fields entry (preferred)
/// 2. Top-level `bm25_fields` section (backward compatible)
/// 3. Auto-derive from fuzzy/embedding match field entries
///
/// If both (1) and (2) are set, `validate()` will have already rejected
/// the config. This function only resolves the winning source.
fn derive_bm25_fields(cfg: &mut Config) {
    // Check for inline fields on the BM25 match_fields entry.
    let inline_fields: Option<Vec<super::schema::Bm25FieldPair>> = cfg
        .match_fields
        .iter()
        .find(|mf| mf.method == "bm25")
        .and_then(|mf| mf.fields.clone());

    if let Some(fields) = inline_fields {
        cfg.bm25_fields = fields;
        return;
    }

    if !cfg.bm25_fields.is_empty() {
        // User provided top-level bm25_fields — keep them.
        return;
    }

    // Fallback: derive from fuzzy/embedding match fields.
    cfg.bm25_fields = cfg
        .match_fields
        .iter()
        .filter(|mf| mf.method == "fuzzy" || mf.method == "embedding")
        .map(|mf| super::schema::Bm25FieldPair {
            field_a: mf.field_a.clone(),
            field_b: mf.field_b.clone(),
        })
        .collect();
}

/// Populate `synonym_fields` if not explicitly set in config.
///
/// When the user omits `synonym_fields`, derive them from `method: synonym`
/// entries in `match_fields` with default generators (acronym, min_length=3).
/// When set explicitly, the user controls generator options.
fn derive_synonym_fields(cfg: &mut Config) {
    if !cfg.synonym_fields.is_empty() {
        // User provided explicit synonym_fields — keep them.
        return;
    }
    cfg.synonym_fields = cfg
        .match_fields
        .iter()
        .filter(|mf| mf.method == "synonym")
        .map(|mf| super::schema::SynonymFieldConfig {
            field_a: mf.field_a.clone(),
            field_b: mf.field_b.clone(),
            generators: super::schema::default_synonym_generators(),
        })
        .collect();
}

fn derive_required_fields(cfg: &mut Config) {
    let mut seen_a = HashSet::new();
    let mut seen_b = HashSet::new();

    // id fields
    seen_a.insert(cfg.datasets.a.id_field.clone());
    seen_b.insert(cfg.datasets.b.id_field.clone());

    // common_id fields
    if let Some(ref cid) = cfg.datasets.a.common_id_field
        && !cid.is_empty()
    {
        seen_a.insert(cid.clone());
    }
    if let Some(ref cid) = cfg.datasets.b.common_id_field
        && !cid.is_empty()
    {
        seen_b.insert(cid.clone());
    }

    // match fields (skip BM25 — no field_a/field_b)
    for mf in &cfg.match_fields {
        if mf.method == "bm25" {
            continue;
        }
        seen_a.insert(mf.field_a.clone());
        seen_b.insert(mf.field_b.clone());
    }

    // bm25_fields (explicit or derived — fields must be loadable from datasets)
    for pair in &cfg.bm25_fields {
        seen_a.insert(pair.field_a.clone());
        seen_b.insert(pair.field_b.clone());
    }

    // synonym_fields (explicit or derived)
    for sf in &cfg.synonym_fields {
        seen_a.insert(sf.field_a.clone());
        seen_b.insert(sf.field_b.clone());
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
        let cfg = load_config(Path::new(
            "benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml",
        ))
        .unwrap();
        assert_eq!(cfg.job.name, "live_10kx10k_usearch_warm");
        assert_eq!(cfg.performance.encoder_pool_size, Some(4));
        assert_eq!(cfg.top_n, Some(5));
        assert_eq!(cfg.live.crossmap_flush_secs, Some(5)); // default applied
    }

    #[test]
    fn load_bench1kx1k() {
        let cfg = load_config(Path::new("benchmarks/batch/10kx10k_flat/cold/config.yaml")).unwrap();
        // top_n: 20 is set explicitly in bench1kx1k.yaml
        assert_eq!(cfg.top_n, Some(20));
        assert_eq!(cfg.performance.encoder_pool_size, Some(4));
    }

    #[test]
    fn load_counterparty_recon_with_sidecar() {
        // sidecar section in YAML is silently ignored
        let cfg = load_config(Path::new("tests/fixtures/counterparty_recon.yaml")).unwrap();
        assert_eq!(cfg.output_mapping.len(), 4);
        assert_eq!(cfg.performance.encoder_pool_size, Some(2));
    }

    #[test]
    fn derived_fields_bench_live() {
        let cfg = load_config(Path::new(
            "benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml",
        ))
        .unwrap();
        // A side should include: entity_id, legal_name, short_name, country_code, lei
        assert!(cfg.required_fields_a.contains(&"entity_id".to_string()));
        assert!(cfg.required_fields_a.contains(&"legal_name".to_string()));
        assert!(cfg.required_fields_a.contains(&"short_name".to_string()));
        assert!(cfg.required_fields_a.contains(&"country_code".to_string()));
        assert!(cfg.required_fields_a.contains(&"lei".to_string()));
        // B side should include: counterparty_id, counterparty_name, domicile, lei_code
        assert!(
            cfg.required_fields_b
                .contains(&"counterparty_id".to_string())
        );
        assert!(
            cfg.required_fields_b
                .contains(&"counterparty_name".to_string())
        );
        assert!(cfg.required_fields_b.contains(&"domicile".to_string()));
        assert!(cfg.required_fields_b.contains(&"lei_code".to_string()));
    }

    #[test]
    fn legacy_blocking_normalised() {
        // counterparty_recon.yaml uses legacy single field_a/field_b blocking
        let cfg = load_config(Path::new("tests/fixtures/counterparty_recon.yaml")).unwrap();
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
embeddings: { model: m, a_cache_dir: i }
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
embeddings: { model: m, a_cache_dir: i }
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
embeddings: { model: m, a_cache_dir: i }
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
embeddings: { model: m, a_cache_dir: i }
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
embeddings: { model: m, a_cache_dir: i }
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
embeddings: { model: m, a_cache_dir: i }
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
    fn validation_invalid_vector_backend() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("vector_backend"), "got: {}", err);
    }

    #[test]
    fn validation_vector_backend_defaults_to_flat() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.vector_backend, "flat");
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        // Without the usearch feature compiled in, this should fail.
        // With the usearch feature, this should pass.
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
    fn fuzzy_scorer_defaults_to_wratio() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
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

    #[test]
    fn vector_quantization_f16_accepted() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.performance.vector_quantization, Some("f16".to_string()));
    }

    #[test]
    fn vector_quantization_bf16_accepted() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(
            cfg.performance.vector_quantization,
            Some("bf16".to_string())
        );
    }

    #[test]
    fn vector_quantization_invalid_rejected() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("vector_quantization"),
            "got: {}",
            err
        );
    }

    // --- BM25 validation tests ---

    /// Helper: minimal valid YAML for tests that add BM25 match fields.
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
output: {{ results_path: r, review_path: rv, unmatched_path: u }}
"#,
            match_fields_yaml
        )
    }

    #[test]
    fn bm25_accepted_without_fields() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
    }

    #[test]
    fn bm25_rejected_with_field_a() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { field_a: x, method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("must be omitted"), "got: {}", err);
    }

    #[test]
    fn bm25_rejected_with_field_b() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { field_b: x, method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("must be omitted"), "got: {}", err);
    }

    #[test]
    fn bm25_multiple_rejected() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.6 }\n  - { method: bm25, weight: 0.2 }\n  - { method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("at most one bm25"), "got: {}", err);
    }

    #[test]
    fn ann_and_bm25_candidates_independent() {
        // ann_candidates < bm25_candidates is valid — generators are independent.
        let yaml = format!(
            r#"
job:
  name: test
datasets:
  a: {{ path: "a.csv", id_field: id }}
  b: {{ path: "b.csv", id_field: id }}
cross_map: {{ backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }}
embeddings: {{ model: m, a_cache_dir: i }}
ann_candidates: 5
bm25_candidates: 10
match_fields:
  - {{ field_a: f, field_b: f, method: embedding, weight: 0.8 }}
  - {{ method: bm25, weight: 0.2 }}
thresholds: {{ auto_match: 0.85, review_floor: 0.6 }}
output: {{ results_path: r, review_path: rv, unmatched_path: u }}
"#
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap(); // should pass — independent constraints
    }

    #[test]
    fn bm25_candidates_lt_top_n_rejected() {
        let yaml = format!(
            r#"
job:
  name: test
datasets:
  a: {{ path: "a.csv", id_field: id }}
  b: {{ path: "b.csv", id_field: id }}
cross_map: {{ backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }}
embeddings: {{ model: m, a_cache_dir: i }}
bm25_candidates: 2
top_n: 5
match_fields:
  - {{ field_a: f, field_b: f, method: exact, weight: 0.8 }}
  - {{ method: bm25, weight: 0.2 }}
thresholds: {{ auto_match: 0.85, review_floor: 0.6 }}
output: {{ results_path: r, review_path: rv, unmatched_path: u }}
"#
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("bm25_candidates"), "got: {}", err);
    }

    #[test]
    fn bm25_fields_derived_from_fuzzy_and_embedding() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: name_a, field_b: name_b, method: fuzzy, weight: 0.3 }\n  - { field_a: desc_a, field_b: desc_b, method: embedding, weight: 0.3 }\n  - { field_a: code_a, field_b: code_b, method: exact, weight: 0.2 }\n  - { method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);
        // Should contain fuzzy and embedding fields, NOT exact
        assert_eq!(cfg.bm25_fields.len(), 2);
        assert!(
            cfg.bm25_fields
                .iter()
                .any(|p| p.field_a == "name_a" && p.field_b == "name_b"),
            "should contain name_a/name_b"
        );
        assert!(
            cfg.bm25_fields
                .iter()
                .any(|p| p.field_a == "desc_a" && p.field_b == "desc_b"),
            "should contain desc_a/desc_b"
        );
    }

    #[test]
    fn bm25_fields_explicit_overrides_derivation() {
        let yaml = format!(
            r#"
job: {{ name: test }}
datasets:
  a: {{ path: "a.csv", id_field: id }}
  b: {{ path: "b.csv", id_field: id }}
cross_map: {{ backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }}
embeddings: {{ model: m, a_cache_dir: i }}
bm25_fields:
  - {{ field_a: custom_a, field_b: custom_b }}
match_fields:
  - {{ field_a: name_a, field_b: name_b, method: fuzzy, weight: 0.8 }}
  - {{ method: bm25, weight: 0.2 }}
thresholds: {{ auto_match: 0.85, review_floor: 0.6 }}
output: {{ results_path: r, review_path: rv, unmatched_path: u }}
"#
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);
        // Explicit bm25_fields should NOT be overridden by derivation
        assert_eq!(cfg.bm25_fields.len(), 1);
        assert_eq!(cfg.bm25_fields[0].field_a, "custom_a");
        assert_eq!(cfg.bm25_fields[0].field_b, "custom_b");

        // And those fields should appear in required_fields
        derive_required_fields(&mut cfg);
        assert!(
            cfg.required_fields_a.contains(&"custom_a".to_string()),
            "explicit bm25 field_a should be in required_fields_a"
        );
        assert!(
            cfg.required_fields_b.contains(&"custom_b".to_string()),
            "explicit bm25 field_b should be in required_fields_b"
        );
    }

    #[test]
    fn bm25_always_accepted() {
        // BM25 is always compiled in — method: bm25 should always be accepted.
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
    }

    #[test]
    fn bm25_not_in_required_fields() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 0.8 }\n  - { method: bm25, weight: 0.2 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);
        derive_required_fields(&mut cfg);
        // BM25 has empty field_a/field_b — they should NOT appear in required fields
        assert!(
            !cfg.required_fields_a.contains(&String::new()),
            "empty string should not be in required_fields_a"
        );
        assert!(
            !cfg.required_fields_b.contains(&String::new()),
            "empty string should not be in required_fields_b"
        );
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
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 0.8 }
  - method: bm25
    weight: 0.2
    fields:
      - { field_a: name_a, field_b: name_b }
      - { field_a: addr_a, field_b: addr_b }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);

        assert_eq!(cfg.bm25_fields.len(), 2, "inline fields should be used");
        assert_eq!(cfg.bm25_fields[0].field_a, "name_a");
        assert_eq!(cfg.bm25_fields[1].field_a, "addr_a");
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
  - { field_a: old_a, field_b: old_b }
match_fields:
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 0.8 }
  - method: bm25
    weight: 0.2
    fields:
      - { field_a: new_a, field_b: new_b }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("both inline"),
            "should reject both inline and top-level: {}",
            err
        );
    }

    #[test]
    fn bm25_toplevel_still_works() {
        // Backward compatibility: top-level bm25_fields without inline fields
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
bm25_fields:
  - { field_a: custom_a, field_b: custom_b }
match_fields:
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 0.8 }
  - { method: bm25, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);

        assert_eq!(cfg.bm25_fields.len(), 1, "top-level should still work");
        assert_eq!(cfg.bm25_fields[0].field_a, "custom_a");
    }

    #[test]
    fn bm25_inline_takes_priority_over_derivation() {
        // Inline fields should be used even when fuzzy/embedding fields exist
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 0.5 }
  - { field_a: desc_a, field_b: desc_b, method: fuzzy, weight: 0.3 }
  - method: bm25
    weight: 0.2
    fields:
      - { field_a: only_this_a, field_b: only_this_b }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);

        assert_eq!(
            cfg.bm25_fields.len(),
            1,
            "inline should override auto-derivation"
        );
        assert_eq!(cfg.bm25_fields[0].field_a, "only_this_a");
    }

    #[test]
    fn ann_candidates_defaults() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        assert_eq!(cfg.ann_candidates, Some(50));
        assert_eq!(cfg.bm25_candidates, Some(10));
    }

    #[test]
    fn vector_quantization_none_defaults_to_f32_behavior() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        // When not set, vector_quantization is None (callers default to "f32")
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
        assert_eq!(
            cfg.performance.vector_index_mode, None,
            "unset vector_index_mode should remain None"
        );
    }

    #[test]
    fn vector_index_mode_load_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  vector_index_mode: load\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(
            cfg.performance.vector_index_mode.as_deref(),
            Some("load"),
            "vector_index_mode: load should be accepted"
        );
    }

    #[test]
    fn vector_index_mode_mmap_accepted() {
        let yaml = format!(
            "{}\nperformance:\n  vector_index_mode: mmap\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(
            cfg.performance.vector_index_mode.as_deref(),
            Some("mmap"),
            "vector_index_mode: mmap should be accepted"
        );
    }

    #[test]
    fn min_score_gap_accepted() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.thresholds.min_score_gap, Some(0.10));
    }

    #[test]
    fn min_score_gap_zero_accepted() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.thresholds.min_score_gap, Some(0.0));
    }

    #[test]
    fn min_score_gap_negative_rejected() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("min_score_gap"), "got: {}", err);
    }

    #[test]
    fn min_score_gap_one_rejected() {
        let yaml = r#"
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
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("min_score_gap"), "got: {}", err);
    }

    #[test]
    fn min_score_gap_absent_is_none() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.thresholds.min_score_gap, None);
    }

    #[test]
    fn vector_index_mode_invalid_rejected() {
        let yaml = format!(
            "{}\nperformance:\n  vector_index_mode: load_from_disk\n",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("vector_index_mode"),
            "error should mention vector_index_mode, got: {}",
            err
        );
    }

    // --- Enroll config tests ---

    #[test]
    fn enroll_config_basic_parse() {
        let yaml = r#"
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
"#;
        let enroll: super::super::enroll_schema::EnrollConfig = serde_yaml::from_str(yaml).unwrap();
        let mut cfg = enroll.into_config().unwrap();
        apply_defaults(&mut cfg);
        validate_enroll(&cfg).unwrap();

        assert!(cfg.is_enroll_mode(), "should be enroll mode");
        assert_eq!(cfg.job.name, "enroll_test");
        assert_eq!(cfg.embeddings.model, "all-MiniLM-L6-v2");
        assert_eq!(cfg.embeddings.a_cache_dir, "cache/emb");

        // field -> field_a == field_b
        assert_eq!(cfg.match_fields[0].field_a, "legal_name");
        assert_eq!(cfg.match_fields[0].field_b, "legal_name");
        assert_eq!(cfg.match_fields[1].field_a, "country_code");
        assert_eq!(cfg.match_fields[1].field_b, "country_code");
    }

    #[test]
    fn enroll_config_with_dataset() {
        let yaml = r#"
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
"#;
        let enroll: super::super::enroll_schema::EnrollConfig = serde_yaml::from_str(yaml).unwrap();
        let mut cfg = enroll.into_config().unwrap();
        apply_defaults(&mut cfg);
        validate_enroll(&cfg).unwrap();

        assert_eq!(cfg.datasets.a.path, "reference.csv");
        assert_eq!(cfg.datasets.a.id_field, "entity_id");
        // B-side mirrors A-side id_field
        assert_eq!(cfg.datasets.b.id_field, "entity_id");
        assert!(
            cfg.datasets.b.path.is_empty(),
            "B-side path should be empty"
        );
    }

    #[test]
    fn enroll_config_with_blocking() {
        let yaml = r#"
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
"#;
        let enroll: super::super::enroll_schema::EnrollConfig = serde_yaml::from_str(yaml).unwrap();
        let mut cfg = enroll.into_config().unwrap();
        apply_defaults(&mut cfg);
        validate_enroll(&cfg).unwrap();

        assert!(cfg.blocking.enabled);
        assert_eq!(cfg.blocking.fields.len(), 1);
        assert_eq!(cfg.blocking.fields[0].field_a, "country");
        assert_eq!(cfg.blocking.fields[0].field_b, "country");
    }

    #[test]
    fn enroll_config_weight_sum_rejected() {
        let yaml = r#"
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
"#;
        let enroll: super::super::enroll_schema::EnrollConfig = serde_yaml::from_str(yaml).unwrap();
        let mut cfg = enroll.into_config().unwrap();
        apply_defaults(&mut cfg);
        let err = validate_enroll(&cfg).unwrap_err();
        assert!(err.to_string().contains("weights sum"), "got: {}", err);
    }

    #[test]
    fn enroll_config_no_dataset_allowed() {
        let yaml = r#"
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
"#;
        let enroll: super::super::enroll_schema::EnrollConfig = serde_yaml::from_str(yaml).unwrap();
        let mut cfg = enroll.into_config().unwrap();
        apply_defaults(&mut cfg);
        validate_enroll(&cfg).unwrap(); // should pass — empty pool is valid
        assert!(cfg.datasets.a.path.is_empty());
    }

    #[test]
    fn enroll_config_bm25_with_inline_fields() {
        let yaml = r#"
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
"#;
        let enroll: super::super::enroll_schema::EnrollConfig = serde_yaml::from_str(yaml).unwrap();
        let mut cfg = enroll.into_config().unwrap();
        apply_defaults(&mut cfg);
        validate_enroll(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);

        assert_eq!(cfg.bm25_fields.len(), 2);
        assert_eq!(cfg.bm25_fields[0].field_a, "name");
        assert_eq!(cfg.bm25_fields[0].field_b, "name");
        assert_eq!(cfg.bm25_fields[1].field_a, "address");
        assert_eq!(cfg.bm25_fields[1].field_b, "address");
    }

    // --- Hooks config tests ---

    #[test]
    fn hooks_empty_command_rejected() {
        let yaml = format!(
            "{}\nhooks:\n  command: \"\"",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        let err = validate(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("hooks.command"),
            "expected hooks.command error, got: {}",
            err
        );
    }

    #[test]
    fn hooks_valid_command_accepted() {
        let yaml = format!(
            "{}\nhooks:\n  command: \"python hook.py\"",
            base_yaml_with_match_fields(
                "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }"
            )
        );
        let mut cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        assert_eq!(cfg.hooks.command.as_deref(), Some("python hook.py"),);
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

    #[test]
    fn expansion_search_defaults_to_none() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            cfg.performance.expansion_search, None,
            "expansion_search should default to None when absent"
        );
    }

    #[test]
    fn expansion_search_explicit_value() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
performance:
  expansion_search: 32
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            cfg.performance.expansion_search,
            Some(32),
            "expansion_search should deserialize explicit value"
        );
    }

    #[test]
    fn bm25_commit_batch_size_defaults_to_none() {
        let yaml = base_yaml_with_match_fields(
            "  - { field_a: f, field_b: f, method: exact, weight: 1.0 }",
        );
        let cfg: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            cfg.bm25_commit_batch_size, None,
            "bm25_commit_batch_size should default to None when absent"
        );
    }

    #[test]
    fn bm25_commit_batch_size_explicit_value() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: f, field_b: f, method: exact, weight: 1.0 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { results_path: r, review_path: rv, unmatched_path: u }
bm25_commit_batch_size: 100
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            cfg.bm25_commit_batch_size,
            Some(100),
            "bm25_commit_batch_size should deserialize explicit value"
        );
    }
}
