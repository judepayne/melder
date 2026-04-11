//! Config defaults, deprecation warnings, and blocking normalisation.

use super::schema::{BlockingFieldPair, Config, FuzzyScorer, MatchMethod};

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

pub(crate) fn apply_defaults(cfg: &mut Config) {
    // top_n: default 5
    if cfg.top_n.is_none() || cfg.top_n == Some(0) {
        cfg.top_n = Some(5);
    }

    // live
    if cfg.live.crossmap_flush_secs.is_none() || cfg.live.crossmap_flush_secs == Some(0) {
        cfg.live.crossmap_flush_secs = Some(5);
    }

    // Default encoder_pool_size to 1 if not set — but only for local
    // encoders. When `embeddings.remote_encoder_cmd` is set the operator
    // MUST specify pool size explicitly; validation will reject a missing
    // value with a clear error. See docs/remote-encoder.md for why this
    // is not silently defaulted.
    let remote_set = cfg
        .embeddings
        .remote_encoder_cmd
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if !remote_set
        && (cfg.performance.encoder_pool_size.is_none()
            || cfg.performance.encoder_pool_size == Some(0))
    {
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
        if mf.method == MatchMethod::Fuzzy && mf.scorer.is_none() {
            mf.scorer = Some(FuzzyScorer::Wratio);
        }
    }
}

// ---------------------------------------------------------------------------
// Deprecation warnings
// ---------------------------------------------------------------------------

pub(crate) fn deprecation_warnings(cfg: &Config) {
    if cfg.bm25_commit_batch_size.is_some() {
        tracing::warn!(
            "bm25_commit_batch_size is deprecated and ignored. \
             SimpleBm25 has instant write visibility — no commit batching needed."
        );
    }
}

// ---------------------------------------------------------------------------
// Blocking normalisation
// ---------------------------------------------------------------------------

pub(crate) fn normalise_blocking(cfg: &mut Config) {
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::loader::load_config;

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
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        assert_eq!(cfg.match_fields[0].scorer, Some(FuzzyScorer::Wratio));
    }

    #[test]
    fn ann_candidates_defaults() {
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
output: { csv_dir_path: /tmp/test/output }
"#;
        let mut cfg: Config = serde_yaml::from_str(yaml).unwrap();
        normalise_blocking(&mut cfg);
        apply_defaults(&mut cfg);
        assert_eq!(cfg.ann_candidates, Some(50));
        assert_eq!(cfg.bm25_candidates, Some(10));
    }

    #[test]
    fn legacy_blocking_normalised() {
        let cfg = load_config(Path::new("tests/fixtures/counterparty_recon.yaml")).unwrap();
        assert!(cfg.blocking.enabled);
        assert_eq!(cfg.blocking.fields.len(), 1);
        assert_eq!(cfg.blocking.fields[0].field_a, "country_code");
        assert_eq!(cfg.blocking.fields[0].field_b, "domicile");
        assert_eq!(cfg.blocking.operator, "and");
    }

    #[test]
    fn expansion_search_defaults_to_none() {
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
output: { csv_dir_path: /tmp/test/output }
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.performance.expansion_search, None);
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
output: { csv_dir_path: /tmp/test/output }
performance:
  expansion_search: 32
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.performance.expansion_search, Some(32));
    }

    #[test]
    fn bm25_commit_batch_size_defaults_to_none() {
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
output: { csv_dir_path: /tmp/test/output }
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.bm25_commit_batch_size, None);
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
output: { csv_dir_path: /tmp/test/output }
bm25_commit_batch_size: 100
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.bm25_commit_batch_size, Some(100));
    }

    #[test]
    fn encoder_device_defaults_to_none() {
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
output: { csv_dir_path: /tmp/test/output }
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.performance.encoder_device, None);
        assert_eq!(cfg.performance.encoder_batch_size, None);
    }
}
