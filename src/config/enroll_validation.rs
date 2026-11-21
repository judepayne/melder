//! Native single-pool validation using shared component constraints.

use crate::error::ConfigError;

use super::enroll_schema::EnrollJobConfig;
use super::schema::MatchMethod;
use super::validation::{
    require_non_empty, require_one_of, validate_dataset, validate_embeddings_source,
    validate_hooks, validate_performance, validate_thresholds,
};

/// Valid vector quantization formats.
const VALID_VECTOR_QUANTIZATIONS: &[&str] = &["f32", "f16", "bf16"];

/// Valid vector index modes.
const VALID_VECTOR_INDEX_MODES: &[&str] = &["load", "mmap"];

pub(super) fn validate_enroll(cfg: &EnrollJobConfig) -> Result<(), ConfigError> {
    // An empty seed pool is valid.
    if !cfg.dataset.path.is_empty() {
        validate_dataset(&cfg.dataset, "dataset")?;
    }

    // Validate single-pool retrieval fields.
    validate_blocking(cfg)?;

    // Exact prefilter
    validate_exact_prefilter(cfg)?;

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
// Single-pool orchestration of shared component validation
// ---------------------------------------------------------------------------

fn validate_common(cfg: &EnrollJobConfig) -> Result<(), ConfigError> {
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
    validate_embeddings_source(
        &cfg.embeddings.model,
        cfg.embeddings.remote_encoder_cmd.as_deref(),
    )?;
    require_non_empty(&cfg.embeddings.cache_dir, "embeddings.a_cache_dir")?;

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
    validate_thresholds(&cfg.thresholds)?;

    // performance
    validate_performance(
        &cfg.performance,
        cfg.embeddings.remote_encoder_cmd.as_deref(),
    )?;

    super::validation::validate_vector_backend(&cfg.vector_backend)?;

    // synonym_dictionary
    if let Some(ref sd) = cfg.synonym_dictionary {
        require_non_empty(&sd.path, "synonym_dictionary.path")?;
    }

    // hooks
    validate_hooks(&cfg.hooks)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Component validators
// ---------------------------------------------------------------------------

fn validate_match_fields(cfg: &EnrollJobConfig) -> Result<(), ConfigError> {
    let mut weight_sum = 0.0_f64;
    let mut bm25_count = 0usize;

    for (i, mf) in cfg.match_fields.iter().enumerate() {
        let prefix = format!("match_fields[{}]", i);

        if mf.method == MatchMethod::Bm25 {
            if !mf.field.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: prefix.to_string(),
                    message:
                        "bm25 method operates across all text fields; field_a and field_b must be omitted"
                            .into(),
                });
            }
            bm25_count += 1;
        } else {
            require_non_empty(&mf.field, &format!("{}.field", prefix))?;
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
            require_non_empty(&pair.field, &format!("{}.field", prefix))?;
        }
    }

    // Validate top-level bm25_fields
    for (i, pair) in cfg.bm25_fields.iter().enumerate() {
        let prefix = format!("bm25_fields[{}]", i);
        require_non_empty(&pair.field, &format!("{}.field", prefix))?;
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

    super::validation::validate_candidates_and_weight(
        has_embedding,
        has_bm25,
        top_n,
        ann_candidates,
        bm25_candidates,
        weight_sum,
    )?;

    Ok(())
}

// Validate single-pool blocking.
fn validate_blocking(cfg: &EnrollJobConfig) -> Result<(), ConfigError> {
    if !super::validation::validate_retrieval_enabled(
        cfg.blocking.enabled,
        cfg.blocking.fields.len(),
        "blocking",
    )? {
        return Ok(());
    }
    for (i, fp) in cfg.blocking.fields.iter().enumerate() {
        let prefix = format!("blocking.fields[{}]", i);
        require_non_empty(&fp.field, &format!("{}.field", prefix))?;
    }
    super::validation::validate_blocking_operator(&cfg.blocking.operator)?;

    Ok(())
}

// Validate single-pool exact prefilter.
fn validate_exact_prefilter(cfg: &EnrollJobConfig) -> Result<(), ConfigError> {
    if !super::validation::validate_retrieval_enabled(
        cfg.exact_prefilter.enabled,
        cfg.exact_prefilter.fields.len(),
        "exact_prefilter",
    )? {
        return Ok(());
    }
    for (i, fp) in cfg.exact_prefilter.fields.iter().enumerate() {
        let prefix = format!("exact_prefilter.fields[{}]", i);
        require_non_empty(&fp.field, &format!("{}.field", prefix))?;
    }
    Ok(())
}

/// Validate that all field names used as SQL column names are safe identifiers.
pub(crate) fn validate_field_names(cfg: &EnrollJobConfig) -> Result<(), ConfigError> {
    super::validation::validate_identifiers(&cfg.required_fields)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
