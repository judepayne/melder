//! Derived config fields: BM25 fields, synonym fields, required fields.

use std::collections::HashSet;

use super::schema::{Config, MatchMethod};

/// Populate `bm25_fields` from the best available source.
///
/// Priority:
/// 1. Inline `fields` on the BM25 match_fields entry (preferred)
/// 2. Top-level `bm25_fields` section (backward compatible)
/// 3. Auto-derive from fuzzy/embedding match field entries
///
/// If both (1) and (2) are set, `validate()` will have already rejected
/// the config. This function only resolves the winning source.
pub(crate) fn derive_bm25_fields(cfg: &mut Config) {
    // Check for inline fields on the BM25 match_fields entry.
    let inline_fields: Option<Vec<super::schema::Bm25FieldPair>> = cfg
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
///
/// Warns when an explicit `synonym_fields` section is redundant — i.e. it
/// matches exactly what auto-derivation would produce. This nudges users
/// toward the simpler config form.
pub(crate) fn derive_synonym_fields(cfg: &mut Config) {
    if !cfg.synonym_fields.is_empty() {
        // User provided explicit synonym_fields — check if redundant.
        warn_if_synonym_fields_redundant(cfg);
        return;
    }
    cfg.synonym_fields = cfg
        .match_fields
        .iter()
        .filter(|mf| mf.method == MatchMethod::Synonym)
        .map(|mf| super::schema::SynonymFieldConfig {
            field_a: mf.field_a.clone(),
            field_b: mf.field_b.clone(),
            generators: super::schema::default_synonym_generators(),
        })
        .collect();
}

/// Warn when an explicit `synonym_fields` section duplicates what would be
/// auto-derived from `method: synonym` entries in `match_fields`.
fn warn_if_synonym_fields_redundant(cfg: &Config) {
    let derived: Vec<(&str, &str)> = cfg
        .match_fields
        .iter()
        .filter(|mf| mf.method == MatchMethod::Synonym)
        .map(|mf| (mf.field_a.as_str(), mf.field_b.as_str()))
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

    let explicit: Vec<(&str, &str)> = cfg
        .synonym_fields
        .iter()
        .map(|sf| (sf.field_a.as_str(), sf.field_b.as_str()))
        .collect();

    if explicit == derived {
        eprintln!(
            "NOTE: synonym_fields section is redundant — it duplicates what would be \
             auto-derived from the method: synonym entry in match_fields. \
             You can safely remove the synonym_fields section."
        );
    }
}

pub(crate) fn derive_required_fields(cfg: &mut Config) {
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
        if mf.method == MatchMethod::Bm25 {
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

    // exact_prefilter fields
    if cfg.exact_prefilter.enabled {
        for fp in &cfg.exact_prefilter.fields {
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::defaults::{apply_defaults, normalise_blocking};
    use crate::config::loader::load_config;
    use crate::config::validation::validate;

    #[test]
    fn derived_fields_bench_live() {
        let cfg = load_config(Path::new(
            "benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml",
        ))
        .unwrap();
        assert!(cfg.required_fields_a.contains(&"entity_id".to_string()));
        assert!(cfg.required_fields_a.contains(&"legal_name".to_string()));
        assert!(cfg.required_fields_a.contains(&"short_name".to_string()));
        assert!(cfg.required_fields_a.contains(&"country_code".to_string()));
        assert!(cfg.required_fields_a.contains(&"lei".to_string()));
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
    fn bm25_fields_derived_from_fuzzy_and_embedding() {
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
  - { method: bm25, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);

        assert_eq!(cfg.bm25_fields.len(), 2);
        assert_eq!(cfg.bm25_fields[0].field_a, "name_a");
        assert_eq!(cfg.bm25_fields[0].field_b, "name_b");
        assert_eq!(cfg.bm25_fields[1].field_a, "desc_a");
        assert_eq!(cfg.bm25_fields[1].field_b, "desc_b");
    }

    #[test]
    fn bm25_fields_explicit_overrides_derivation() {
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
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 0.5 }
  - { field_a: desc_a, field_b: desc_b, method: fuzzy, weight: 0.3 }
  - { method: bm25, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);

        assert_eq!(cfg.bm25_fields.len(), 1, "explicit should override");
        assert_eq!(cfg.bm25_fields[0].field_a, "custom_a");
    }

    #[test]
    fn bm25_not_in_required_fields() {
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
  - { method: bm25, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);
        derive_synonym_fields(&mut cfg);
        derive_required_fields(&mut cfg);

        assert!(
            !cfg.required_fields_a.contains(&"bm25".to_string()),
            "BM25 pseudo-field should not be in required_fields"
        );
    }

    #[test]
    fn bm25_inline_takes_priority_over_derivation() {
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
output: { csv_dir_path: /tmp/test/output }
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
    fn synonym_fields_auto_derived_from_match_fields() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
match_fields:
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 1.0 }
  - { field_a: name_a, field_b: name_b, method: synonym, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);
        derive_synonym_fields(&mut cfg);

        assert_eq!(cfg.synonym_fields.len(), 1);
        assert_eq!(cfg.synonym_fields[0].field_a, "name_a");
        assert_eq!(cfg.synonym_fields[0].field_b, "name_b");
        assert_eq!(cfg.synonym_fields[0].generators.len(), 1);
        assert_eq!(cfg.synonym_fields[0].generators[0].gen_type, "acronym");
        assert_eq!(cfg.synonym_fields[0].generators[0].min_length, 3);
    }

    #[test]
    fn synonym_fields_explicit_with_custom_generators_kept() {
        let yaml = r#"
job:
  name: test
datasets:
  a: { path: "a.csv", id_field: id }
  b: { path: "b.csv", id_field: id }
cross_map: { backend: local, path: "cm.csv", a_id_field: a, b_id_field: b }
embeddings: { model: m, a_cache_dir: i }
synonym_fields:
  - field_a: name_a
    field_b: name_b
    generators:
      - { type: acronym, min_length: 4 }
match_fields:
  - { field_a: name_a, field_b: name_b, method: embedding, weight: 1.0 }
  - { field_a: name_a, field_b: name_b, method: synonym, weight: 0.2 }
thresholds: { auto_match: 0.85, review_floor: 0.6 }
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        validate(&cfg).unwrap();
        derive_bm25_fields(&mut cfg);
        derive_synonym_fields(&mut cfg);

        assert_eq!(cfg.synonym_fields.len(), 1);
        assert_eq!(cfg.synonym_fields[0].generators[0].min_length, 4);
    }
}
