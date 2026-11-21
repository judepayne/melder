//! Defaults and derivation for native single-pool jobs.
use super::enroll_schema::EnrollJobConfig;
use super::schema::{FuzzyScorer, MatchMethod};
pub(crate) fn apply_defaults(cfg: &mut EnrollJobConfig) {
    super::defaults::apply_component_defaults(
        &mut cfg.top_n,
        &mut cfg.performance,
        cfg.embeddings.remote_encoder_cmd.as_deref(),
        &mut cfg.ann_candidates,
        &mut cfg.bm25_candidates,
    );
    // fuzzy scorer defaults
    for mf in &mut cfg.match_fields {
        if mf.method == MatchMethod::Fuzzy && mf.scorer.is_none() {
            mf.scorer = Some(FuzzyScorer::Wratio);
        }
    }
}

pub(crate) fn derive_bm25_fields(cfg: &mut EnrollJobConfig) {
    // Check for inline fields on the BM25 match_fields entry.
    let inline_fields: Option<Vec<super::enroll_schema::EnrollBm25Field>> = cfg
        .match_fields
        .iter()
        .find(|mf| mf.method == MatchMethod::Bm25)
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
        .filter(|mf| mf.method == MatchMethod::Fuzzy || mf.method == MatchMethod::Embedding)
        .map(|mf| super::enroll_schema::EnrollBm25Field {
            field: mf.field.clone(),
        })
        .collect();
}

/// Populate `synonym_fields` if not explicitly set in config.
///
/// When the user omits `synonym_fields`, derive them from `method: synonym`
/// entries in `match_fields` with default generators (acronym, min_length=3).
/// When set explicitly, the user controls generator options.
///
/// Warns when an explicit `synonym_fields` section is redundant — i.e. it
/// matches exactly what auto-derivation would produce. This nudges users
/// toward the simpler config form.
pub(crate) fn derive_synonym_fields(cfg: &mut EnrollJobConfig) {
    if !cfg.synonym_fields.is_empty() {
        // User provided explicit synonym_fields — check if redundant.
        warn_if_synonym_fields_redundant(cfg);
        return;
    }
    cfg.synonym_fields = cfg
        .match_fields
        .iter()
        .filter(|mf| mf.method == MatchMethod::Synonym)
        .map(|mf| super::enroll_schema::EnrollSynonymFieldConfig {
            field: mf.field.clone(),
            generators: super::schema::default_synonym_generators(),
        })
        .collect();
}

// Warn when explicit synonym fields duplicate the derived defaults.
fn warn_if_synonym_fields_redundant(cfg: &EnrollJobConfig) {
    let derived: Vec<&str> = cfg
        .match_fields
        .iter()
        .filter(|mf| mf.method == MatchMethod::Synonym)
        .map(|mf| mf.field.as_str())
        .collect();

    if cfg.synonym_fields.len() != derived.len() {
        return; // Different count — not redundant.
    }

    let all_default_generators = cfg.synonym_fields.iter().all(|sf| {
        sf.generators.len() == 1
            && sf.generators[0].gen_type == "acronym"
            && sf.generators[0].min_length == 3
    });

    if !all_default_generators {
        return; // Custom generators — user needs the explicit section.
    }

    let explicit: Vec<&str> = cfg
        .synonym_fields
        .iter()
        .map(|sf| sf.field.as_str())
        .collect();

    if explicit == derived {
        eprintln!(
            "NOTE: synonym_fields section is redundant — it duplicates what would be \
             auto-derived from the method: synonym entry in match_fields. \
             You can safely remove the synonym_fields section."
        );
    }
}

pub(crate) fn derive_required_fields(cfg: &mut EnrollJobConfig) {
    cfg.required_fields = super::derivation::required_fields(
        &cfg.dataset.id_field,
        cfg.dataset.common_id_field.as_deref(),
        cfg.match_fields
            .iter()
            .filter(|f| f.method != MatchMethod::Bm25)
            .map(|f| f.field.as_str())
            .chain(cfg.bm25_fields.iter().map(|f| f.field.as_str()))
            .chain(cfg.synonym_fields.iter().map(|f| f.field.as_str()))
            .chain(
                cfg.blocking
                    .fields
                    .iter()
                    .filter(|_| cfg.blocking.enabled)
                    .map(|f| f.field.as_str()),
            )
            .chain(
                cfg.exact_prefilter
                    .fields
                    .iter()
                    .filter(|_| cfg.exact_prefilter.enabled)
                    .map(|f| f.field.as_str()),
            ),
    );
}
