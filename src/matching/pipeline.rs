//! Unified scoring pipeline shared by batch and live modes.
//!
//! ## Pipeline stages
//!
//!   1. Blocking filter — done by the *caller* before invoking score_pool;
//!      `blocked_ids` is passed in as a pre-queried slice.
//!   2. Independent candidate generation — each method runs in parallel:
//!      - ANN search (O(log N) HNSW, `ann_candidates`)
//!      - BM25 search (Block-Max WAND via SimpleBm25, `bm25_candidates`)
//!      - Synonym lookup (inverted index)
//!   3. Union — deduplicate candidates by ID across all generators.
//!   4. Full scoring — score candidates on all match_fields, classify, sort,
//!      truncate to `top_n`.
//!
//! Each candidate generator runs independently against the blocked pool. No
//! method filters another method's candidates — the union captures candidates
//! found by ANY method. Full scoring is the single place where all signals
//! combine.
//!
//! Both batch and live call `score_pool()` with the same arguments; the only
//! difference is where the data comes from (MatchState vs LiveSideState).
//!
//! ## Why blocked_ids is passed in rather than BlockingIndex
//!
//! In live mode the blocking index is a `RwLock<BlockingIndex>`. Previously
//! `score_pool` held the read lock for its entire duration (~5-15ms at 100k
//! scale), starving opposite-side writes. The caller now takes the lock only
//! for the `.query()` call (~1µs) and passes the resulting owned `Vec<String>`
//! here. The pipeline is therefore lock-free for its entire execution.

use std::collections::HashMap;

use tracing::info_span;

use crate::config::{Config, MatchMethod};
use crate::error::StoreError;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;
use crate::scoring::embedding::dot_product_f32;
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// The record being scored against the pool.
pub struct ScoringQuery<'a> {
    pub id: &'a str,
    pub record: &'a Record,
    pub side: Side,
    pub combined_vec: &'a [f32],
}

/// The pool (opposite side) and pre-computed candidate sets to score against.
pub struct ScoringPool<'a> {
    pub store: &'a dyn RecordStore,
    pub side: Side,
    pub combined_index: Option<&'a dyn VectorDB>,
    pub blocked_ids: &'a [String],
    /// Pre-computed BM25 candidate IDs (from caller).
    pub bm25_candidate_ids: &'a [String],
    /// Pre-computed BM25 scores keyed by pool record ID.
    pub bm25_scores_map: &'a HashMap<String, f64>,
    /// Pre-computed synonym candidate IDs (from caller).
    pub synonym_candidate_ids: &'a [String],
    /// Optional synonym dictionary for per-field synonym scoring.
    pub synonym_dictionary: Option<&'a crate::synonym::dictionary::SynonymDictionary>,
    /// Known non-matching pairs to skip during scoring.
    pub exclusions: &'a crate::matching::exclusions::Exclusions,
}

/// Score a single query record against the opposite-side pool.
///
/// Runs independent candidate generators (ANN, BM25, synonym), unions their
/// results, then scores all candidates on every configured match_field.
/// Returns a sorted `Vec<MatchResult>` (descending by score, then ascending
/// by matched record ID for equal scores), truncated to `top_n` entries.
///
/// BM25 and synonym candidate generation is handled by the *caller* — this
/// function receives pre-computed candidate IDs and BM25 scores via
/// `ScoringPool`. ANN candidate generation still runs internally.
pub fn score_pool(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
) -> Vec<MatchResult> {
    score_pool_internal(query, pool, config, ann_candidates, top_n, &[], false)
        .expect("lenient scoring does not propagate store errors")
}

/// Score the ordinary candidate union plus caller-validated verification IDs.
///
/// The caller supplies at most two verification IDs, validates current vectors,
/// and provides their normalized BM25 scores in `pool.bm25_scores_map` (absent
/// hits mean zero). Verification IDs outside `blocked_ids` are ignored. All
/// candidates share exclusion, scoring, ranking, gap and truncation handling;
/// use `top_n = 0` to retain the entire union for verification.
///
/// Candidate record fetch errors are propagated instead of suppressed.
pub fn score_pool_verified(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
    verification_ids: &[String],
) -> Result<Vec<MatchResult>, StoreError> {
    score_pool_internal(
        query,
        pool,
        config,
        ann_candidates,
        top_n,
        verification_ids,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn score_pool_internal(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
    verification_ids: &[String],
    strict: bool,
) -> Result<Vec<MatchResult>, StoreError> {
    if pool.blocked_ids.is_empty() {
        return Ok(Vec::new());
    }

    let has_embeddings = !query.combined_vec.is_empty();

    // Pool-side fields needed for scoring (used by get_many_fields to
    // avoid full JSON deserialization in SQLite).
    let scoring_fields: Vec<String> = config
        .match_fields
        .iter()
        .filter(|mf| mf.method != MatchMethod::Bm25 && mf.method != MatchMethod::Embedding)
        .map(|mf| match pool.side {
            Side::A => mf.field_a.clone(),
            Side::B => mf.field_b.clone(),
        })
        .collect();

    // --- Stage 2: Independent candidate generation ---
    //
    // Each generator runs independently against the blocked pool. The union
    // of all candidate sets flows to full scoring.

    // ANN candidates (empty if no embedding fields configured).
    let ann_cands = {
        let _span = info_span!("ann_candidates").entered();
        candidates::select_candidates_checked(
            query.combined_vec,
            ann_candidates,
            pool.combined_index,
            pool.blocked_ids,
            pool.store,
            pool.side,
            query.record,
            query.side,
            &config.vector_backend,
            strict,
        )?
    };

    // --- Stage 3: Union candidates ---
    //
    // Merge ANN, BM25, and synonym candidate sets, deduplicating by ID.
    // Candidates not in the ANN set need their records fetched from the store.
    let ann_id_set: std::collections::HashSet<&str> =
        ann_cands.iter().map(|c| c.id.as_str()).collect();

    // Collect non-ANN candidate IDs (BM25 + synonym + verification), deduplicated.
    // Only caller-validated verification IDs within the blocking pool may be added.
    let verified_ids = verification_ids
        .iter()
        .filter(|id| pool.blocked_ids.contains(id));
    let mut extra_ids: Vec<String> = Vec::new();
    let mut extra_id_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for id in pool
        .bm25_candidate_ids
        .iter()
        .chain(pool.synonym_candidate_ids.iter())
        .chain(verified_ids)
    {
        if !ann_id_set.contains(id.as_str()) && extra_id_set.insert(id.clone()) {
            extra_ids.push(id.clone());
        }
    }

    // Fetch records for non-ANN candidates.
    let extra_cands: Vec<candidates::Candidate> = if extra_ids.is_empty() {
        Vec::new()
    } else {
        let fetched = if scoring_fields.is_empty() {
            pool.store.get_many(pool.side, &extra_ids)
        } else {
            pool.store
                .get_many_fields(pool.side, &extra_ids, &scoring_fields)
        };
        let records = match fetched {
            Ok(records) => records,
            Err(e) if strict => return Err(e),
            Err(e) => {
                tracing::warn!(error = %e, "failed to fetch extra candidate records");
                Vec::new()
            }
        };
        records
            .into_iter()
            .map(|(id, record)| candidates::Candidate {
                id,
                record,
                combined_dot: 0.0,
            })
            .collect()
    };

    // Build the full union: ANN candidates + extra candidates (BM25/synonym).
    if ann_cands.is_empty() && extra_cands.is_empty() {
        return Ok(Vec::new());
    }

    // Self-match exclusion: when query_side == pool_side (enroll mode),
    // the query record may appear in the pool index. Filter it out before
    // scoring so a record never matches itself.
    let filter_self = query.side == pool.side;

    // --- Stage 4: Full scoring with per-field decomposition ---
    let _span = info_span!("full_scoring").entered();
    let emb_specs = crate::vectordb::embedding_field_specs(config);
    let has_emb_specs = has_embeddings && !emb_specs.is_empty();

    let mut results: Vec<MatchResult> = Vec::with_capacity(ann_cands.len() + extra_cands.len());

    // Score all candidates through the same path.
    for cand in ann_cands.iter().chain(extra_cands.iter()) {
        // Skip self-match in enroll mode
        if filter_self && cand.id == query.id {
            continue;
        }
        // Skip known non-matching pairs
        if !pool.exclusions.is_empty() && pool.exclusions.contains(query.id, &cand.id) {
            continue;
        }
        results.push(score_candidate(
            cand,
            query.id,
            query.record,
            query.side,
            query.combined_vec,
            pool.combined_index,
            config,
            pool.bm25_scores_map,
            &emb_specs,
            has_emb_specs,
            pool.synonym_dictionary,
        ));
    }

    // Sort by score descending, breaking ties by record ID ascending.
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.matched_id.cmp(&b.matched_id))
    });

    // Assign rank (1-based, capped at u8::MAX) after sorting.
    if results.len() > 254 {
        tracing::warn!(
            candidates = results.len(),
            "more than 254 candidates — ranks above 255 will be capped"
        );
    }
    for (i, r) in results.iter_mut().enumerate() {
        r.rank = Some((i + 1).min(255) as u8);
    }

    // Apply score gap check before truncation so rank-2 is still present.
    if let Some(min_gap) = config.thresholds.min_score_gap {
        apply_score_gap_check(&mut results, min_gap);
    }

    // Truncate to top_n if specified.
    if top_n > 0 {
        results.truncate(top_n);
    }

    // Attach matched record (with output mapping) to the top result only.
    if let Some(top) = results.first_mut() {
        let matched = ann_cands
            .iter()
            .chain(extra_cands.iter())
            .find(|c| c.id == top.matched_id);
        if let Some(cand) = matched {
            top.matched_record = Some(apply_output_mapping(&cand.record, config));
        }
    }

    Ok(results)
}

/// Score a single candidate through full per-field scoring.
///
/// Decomposes combined embedding vectors into per-field cosines, combines
/// with precomputed BM25 scores (if any), and runs all non-embedding,
/// non-BM25 scoring methods (exact, fuzzy, numeric).
#[allow(clippy::too_many_arguments)]
fn score_candidate(
    cand: &candidates::Candidate,
    query_id: &str,
    query_record: &Record,
    query_side: Side,
    query_combined_vec: &[f32],
    pool_combined_index: Option<&dyn VectorDB>,
    config: &Config,
    bm25_scores: &HashMap<String, f64>,
    emb_specs: &[(String, String, f64)],
    has_emb_specs: bool,
    synonym_dictionary: Option<&crate::synonym::dictionary::SynonymDictionary>,
) -> MatchResult {
    // Decompose combined vecs → per-field cosine similarities.
    let emb_scores: Option<HashMap<(&str, &str), f64>> = if has_emb_specs {
        pool_combined_index
            .and_then(|idx| idx.get(&cand.id).ok().flatten())
            .map(|pool_vec| decompose_emb_scores(query_combined_vec, &pool_vec, emb_specs))
    } else {
        None
    };

    let precomputed_bm25 = bm25_scores.get(&cand.id).copied();

    let score_result = scoring::score_pair(
        query_record,
        &cand.record,
        &config.match_fields,
        emb_scores.as_ref(),
        precomputed_bm25,
        synonym_dictionary,
        query_side.opposite(),
    );

    scoring::build_match_result(
        query_id,
        &cand.id,
        query_side,
        score_result,
        config.thresholds.auto_match,
        config.thresholds.review_floor,
        None,
        false,
    )
}

/// Independent synonym candidate generation stage.
///
/// Looks up each synonym match field value in the synonym index and returns
/// the union of all matching record IDs that are not already in the ANN or
/// BM25 candidate sets.
pub fn synonym_candidate_stage(
    index: &crate::synonym::index::SynonymIndex,
    query_record: &Record,
    query_side: Side,
    synonym_fields: &[crate::config::SynonymFieldConfig],
) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for sf in synonym_fields {
        let query_field = match query_side {
            Side::A => &sf.field_a,
            Side::B => &sf.field_b,
        };
        let query_value = query_record
            .get(query_field)
            .map(|s| s.as_str())
            .unwrap_or("");
        if query_value.trim().is_empty() {
            continue;
        }

        let min_len = sf.generators.first().map(|g| g.min_length).unwrap_or(3);

        let candidates = index.lookup(query_value, &sf.field_a, &sf.field_b, min_len);
        for id in candidates {
            if seen.insert(id.clone()) {
                result.push(id);
            }
        }
    }

    result
}

/// Decompose two combined embedding vectors into per-field cosine similarities.
///
/// Given combined vectors built as `[√w₁·a₁, √w₂·a₂, …]` (dim = 384×N):
///   - Sub-slice `i` is `√wᵢ·aᵢ` (query) and `√wᵢ·bᵢ` (pool)
///   - `dot(√wᵢ·aᵢ, √wᵢ·bᵢ) = wᵢ · cosᵢ`
///   - Therefore `cosᵢ = dot(sub_query_i, sub_pool_i) / wᵢ`
///
/// Returns a `HashMap<"field_a/field_b", cosᵢ>` compatible with
/// `scoring::score_pair`'s `precomputed_emb_scores` argument.
fn decompose_emb_scores<'a>(
    query_combined: &[f32],
    pool_combined: &[f32],
    emb_specs: &'a [(String, String, f64)],
) -> HashMap<(&'a str, &'a str), f64> {
    if emb_specs.is_empty() || query_combined.is_empty() || pool_combined.is_empty() {
        return HashMap::new();
    }
    let field_dim = query_combined.len() / emb_specs.len();
    if field_dim == 0 {
        return HashMap::new();
    }
    debug_assert_eq!(
        query_combined.len() % emb_specs.len(),
        0,
        "combined vector length {} not evenly divisible by {} fields",
        query_combined.len(),
        emb_specs.len()
    );
    if !query_combined.len().is_multiple_of(emb_specs.len()) {
        tracing::warn!(
            vec_len = query_combined.len(),
            fields = emb_specs.len(),
            "combined vector not evenly divisible by field count — last field will be truncated"
        );
    }

    let mut scores = HashMap::with_capacity(emb_specs.len());

    for (i, (field_a, field_b, weight)) in emb_specs.iter().enumerate() {
        let start = i * field_dim;
        let end = start + field_dim;

        if end > query_combined.len() || end > pool_combined.len() {
            tracing::warn!(
                field_a,
                field_b,
                end,
                query_len = query_combined.len(),
                pool_len = pool_combined.len(),
                "embedding decomposition truncated — field scored as 0.0"
            );
            break;
        }

        let q_sub = &query_combined[start..end];
        let p_sub = &pool_combined[start..end];

        let dot: f32 = dot_product_f32(q_sub, p_sub);

        let cos = if *weight > 0.0 {
            (dot as f64 / weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        scores.insert((field_a.as_str(), field_b.as_str()), cos);
    }

    scores
}

/// Apply output_mapping to a matched record: rename fields per config.
/// Rename fields in a matched record per output_mapping config.
///
/// Each mapping renames `from` → `to`: the original key is removed
/// and the new key is inserted with the same value.
fn apply_output_mapping(record: &Record, config: &Config) -> Record {
    if config.output_mapping.is_empty() {
        return record.clone();
    }

    let mut result = record.clone();
    for mapping in &config.output_mapping {
        if let Some(val) = result.remove(&mapping.from) {
            result.insert(mapping.to.clone(), val);
        }
    }
    result
}

/// Downgrade the top result from Auto to Review when the score gap to rank-2
/// is below `min_gap`.
///
/// A high composite score with a close second candidate indicates model
/// uncertainty — the top match is more likely to be a false positive than
/// when the winning margin is large. Downgrading to Review surfaces these
/// borderline cases for human inspection rather than auto-confirming them.
///
/// Only applies when there are ≥ 2 results. Single-candidate results are
/// left unchanged: no rank-2 exists, so there is no ambiguity signal.
fn apply_score_gap_check(results: &mut [MatchResult], min_gap: f64) {
    if results.len() < 2 {
        return;
    }
    if results[0].classification != Classification::Auto {
        return;
    }
    let gap = results[0].score - results[1].score;
    if gap < min_gap {
        results[0].classification = Classification::Review;
        results[0].reason = Some("downgraded".into());
    }
}

/// Classify the top match result.
pub fn top_classification(results: &[MatchResult]) -> Classification {
    results
        .first()
        .map(|r| r.classification)
        .unwrap_or(Classification::NoMatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingConfig;
    use crate::store::memory::MemoryStore;
    use crate::vectordb::flat::FlatVectorDB;

    fn make_match_result(score: f64, classification: Classification) -> MatchResult {
        MatchResult {
            query_id: "q".to_string(),
            matched_id: "m".to_string(),
            query_side: Side::B,
            score,
            field_scores: vec![],
            classification,
            matched_record: None,
            from_crossmap: false,
            rank: None,
            reason: None,
        }
    }

    // --- apply_score_gap_check ---

    #[test]
    fn score_gap_check_downgrades_auto_when_gap_too_small() {
        let mut results = vec![
            make_match_result(0.90, Classification::Auto),
            make_match_result(0.87, Classification::Auto),
        ];
        apply_score_gap_check(&mut results, 0.10);
        // gap = 0.03 < 0.10 → top result downgraded
        assert_eq!(
            results[0].classification,
            Classification::Review,
            "expected Review when gap < min_gap"
        );
        assert_eq!(
            results[1].classification,
            Classification::Auto,
            "rank-2 must be untouched"
        );
    }

    #[test]
    fn score_gap_check_preserves_auto_when_gap_sufficient() {
        let mut results = vec![
            make_match_result(0.95, Classification::Auto),
            make_match_result(0.80, Classification::Auto),
        ];
        apply_score_gap_check(&mut results, 0.10);
        // gap = 0.15 >= 0.10 → no change
        assert_eq!(
            results[0].classification,
            Classification::Auto,
            "expected Auto when gap >= min_gap"
        );
    }

    #[test]
    fn score_gap_check_noop_with_single_result() {
        let mut results = vec![make_match_result(0.90, Classification::Auto)];
        apply_score_gap_check(&mut results, 0.10);
        // only one result — no rank-2 to compare against
        assert_eq!(results[0].classification, Classification::Auto);
    }

    #[test]
    fn score_gap_check_noop_when_top_is_not_auto() {
        let mut results = vec![
            make_match_result(0.75, Classification::Review),
            make_match_result(0.74, Classification::Review),
        ];
        apply_score_gap_check(&mut results, 0.10);
        // top is Review, not Auto — nothing to downgrade
        assert_eq!(results[0].classification, Classification::Review);
    }

    #[test]
    fn score_gap_check_empty_results_is_noop() {
        let mut results: Vec<MatchResult> = vec![];
        apply_score_gap_check(&mut results, 0.10); // must not panic
    }

    #[test]
    fn decompose_emb_scores_single_field() {
        let w = 1.0_f64;
        let dim = 4usize;

        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];

        let q = a.clone();
        let p = b.clone();

        let specs = vec![("f_a".to_string(), "f_b".to_string(), w)];
        let scores = decompose_emb_scores(&q, &p, &specs);

        let cos = *scores.get(&("f_a", "f_b")).unwrap();
        assert!(cos.abs() < 0.001, "expected ~0.0, got {}", cos);
        let _ = dim;
    }

    #[test]
    fn decompose_emb_scores_self_match() {
        let w = 0.55_f64;
        let sqrt_w = w.sqrt() as f32;

        let scaled = vec![sqrt_w, 0.0, 0.0, 0.0];
        let specs = vec![("f_a".to_string(), "f_b".to_string(), w)];
        let scores = decompose_emb_scores(&scaled, &scaled, &specs);

        let cos = *scores.get(&("f_a", "f_b")).unwrap();
        assert!((cos - 1.0).abs() < 0.001, "expected ~1.0, got {}", cos);
    }

    #[test]
    fn decompose_emb_scores_two_fields_recovers_both() {
        let w1 = 0.55_f64;
        let w2 = 0.20_f64;
        let sqrt_w1 = w1.sqrt() as f32;
        let sqrt_w2 = w2.sqrt() as f32;

        let f1_a = vec![sqrt_w1, 0.0];
        let f1_b = vec![sqrt_w1, 0.0];

        let f2_a = vec![0.0, sqrt_w2];
        let f2_b = vec![sqrt_w2, 0.0];

        let mut q = f1_a.clone();
        q.extend_from_slice(&f2_a);
        let mut p = f1_b.clone();
        p.extend_from_slice(&f2_b);

        let specs = vec![
            ("f1_a".to_string(), "f1_b".to_string(), w1),
            ("f2_a".to_string(), "f2_b".to_string(), w2),
        ];
        let scores = decompose_emb_scores(&q, &p, &specs);

        let cos1 = *scores.get(&("f1_a", "f1_b")).unwrap();
        let cos2 = *scores.get(&("f2_a", "f2_b")).unwrap();

        assert!(
            (cos1 - 1.0).abs() < 0.001,
            "cos1 expected ~1.0, got {}",
            cos1
        );
        assert!(cos2.abs() < 0.001, "cos2 expected ~0.0, got {}", cos2);
    }

    // --- score_pool union tests ---

    fn make_record(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_store() -> MemoryStore {
        MemoryStore::new(&BlockingConfig::default())
    }

    fn make_config_exact(field_a: &str, field_b: &str) -> Config {
        let yaml = format!(
            r#"
job:
  name: test
datasets:
  a: {{ path: x.csv, id_field: id, format: csv }}
  b: {{ path: x.csv, id_field: id, format: csv }}
cross_map:
  path: /tmp/test/crossmap.csv
  a_id_field: id
  b_id_field: id
embeddings:
  model: test
  a_cache_dir: /tmp/test
match_fields:
  - field_a: {field_a}
    field_b: {field_b}
    method: exact
    weight: 1.0
thresholds:
  auto_match: 0.85
  review_floor: 0.60
output:
  csv_dir_path: /tmp/test/output
vector_backend: flat
"#
        );
        // Use load pipeline: parse → defaults → validate → derive.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        crate::config::load_config(tmp.path()).unwrap()
    }

    fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as i32 as f32) / (i32::MAX as f32)
            })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn score_pool_empty_blocked_ids_returns_empty() {
        let store = make_store();
        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Foo")]);
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &[],
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 5);
        assert!(results.is_empty(), "empty blocked_ids → no results");
    }

    #[test]
    fn score_pool_no_embeddings_no_bm25_returns_empty() {
        let store = make_store();
        store
            .insert(
                Side::A,
                "a1",
                &make_record(&[("id", "a1"), ("name", "Foo Corp")]),
            )
            .unwrap();

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Foo Corp")]);
        let blocked = vec!["a1".to_string()];
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        // No embeddings (empty vec), no BM25 candidates → both generators
        // return empty → union is empty → no results.
        let results = score_pool(&q, &p, &config, 50, 5);
        assert!(
            results.is_empty(),
            "no candidate generators configured → no results"
        );
    }

    #[test]
    fn score_pool_ann_only_scores_candidates() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Insert two A records with identical names to the query.
        let a1 = make_record(&[("id", "a1"), ("name", "Foo Corp")]);
        let a2 = make_record(&[("id", "a2"), ("name", "Bar Inc")]);
        store.insert(Side::A, "a1", &a1).unwrap();
        store.insert(Side::A, "a2", &a2).unwrap();

        let v1 = unit_vec(dim, 1);
        let v2 = unit_vec(dim, 2);
        idx.upsert("a1", &v1, &a1, Side::A).unwrap();
        idx.upsert("a2", &v2, &a2, Side::A).unwrap();

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Foo Corp")]);
        let blocked = vec!["a1".to_string(), "a2".to_string()];
        let query_vec = unit_vec(dim, 1); // identical to a1
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &query_vec,
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx as &dyn VectorDB),
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 5);
        assert!(
            !results.is_empty(),
            "ANN should surface candidates for scoring"
        );

        // a1 has exact name match → score 1.0. a2 doesn't match → score 0.0.
        let top = &results[0];
        assert_eq!(top.matched_id, "a1", "exact name match should rank first");
        assert!(
            (top.score - 1.0).abs() < 0.01,
            "exact match score should be ~1.0, got {}",
            top.score
        );
    }

    #[test]
    fn score_pool_results_sorted_descending() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Three A records: one exact match, one partial, one no match.
        let a1 = make_record(&[("id", "a1"), ("name", "Acme Corp")]);
        let a2 = make_record(&[("id", "a2"), ("name", "Other Ltd")]);
        let a3 = make_record(&[("id", "a3"), ("name", "Acme Corp")]);
        store.insert(Side::A, "a1", &a1).unwrap();
        store.insert(Side::A, "a2", &a2).unwrap();
        store.insert(Side::A, "a3", &a3).unwrap();

        for (id, seed, rec) in [("a1", 1u64, &a1), ("a2", 2, &a2), ("a3", 3, &a3)] {
            idx.upsert(id, &unit_vec(dim, seed), rec, Side::A).unwrap();
        }

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Acme Corp")]);
        let blocked = vec!["a1".to_string(), "a2".to_string(), "a3".to_string()];
        let query_vec = unit_vec(dim, 99);
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &query_vec,
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx as &dyn VectorDB),
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 5);

        // Results must be sorted descending by score.
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results not sorted: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn score_pool_equal_scores_use_id_order_before_truncation() {
        let store = make_store();
        for (id, name) in [
            ("ENT-000025", "Gibson Group"),
            ("ENR-0000094", "Gibson Group"),
            ("a-no-match", "Other Company"),
        ] {
            store
                .insert(Side::A, id, &make_record(&[("name", name)]))
                .expect("test records should insert");
        }
        let config = make_config_exact("name", "name");
        let record = make_record(&[("name", "Gibson Group")]);
        let bm25_scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let expected = ["ENR-0000094", "ENT-000025", "a-no-match"];

        // Cover both enroll (same side) and batch/live (opposite side),
        // with every input permutation and cutoffs inside/outside the tie.
        for side in [Side::A, Side::B] {
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let ids: Vec<String> = order.iter().map(|&i| expected[i].to_string()).collect();
                let query = ScoringQuery {
                    id: "query",
                    record: &record,
                    side,
                    combined_vec: &[],
                };
                let pool = ScoringPool {
                    store: &store,
                    side: Side::A,
                    combined_index: None,
                    blocked_ids: &ids,
                    bm25_candidate_ids: &ids,
                    bm25_scores_map: &bm25_scores,
                    synonym_candidate_ids: &[],
                    synonym_dictionary: None,
                    exclusions: &exclusions,
                };
                for top_n in [0, 1, 2, 3] {
                    let results = score_pool(&query, &pool, &config, 50, top_n);
                    let count = if top_n == 0 { 3 } else { top_n };
                    let actual: Vec<&str> = results
                        .iter()
                        .map(|result| result.matched_id.as_str())
                        .collect();
                    assert_eq!(
                        actual,
                        expected[..count],
                        "score then ID ordering must ignore input order: side={side:?}, order={order:?}, top_n={top_n}",
                    );
                    for (i, result) in results.iter().enumerate() {
                        assert_eq!(result.rank, Some((i + 1) as u8), "rank follows tie order");
                        assert_eq!(
                            result.score,
                            if i < 2 { 1.0 } else { 0.0 },
                            "ID tie-breaking must not change scores or outrank higher scores",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn verified_union_is_deduplicated_before_gap_and_truncation() {
        let store = make_store();
        let idx = FlatVectorDB::new(2);
        let record = make_record(&[("name", "Acme")]);
        for id in ["a1", "a2"] {
            store
                .insert(Side::A, id, &record)
                .expect("test record should insert");
        }
        idx.upsert("a1", &[1.0, 0.0], &record, Side::A)
            .expect("test vector should insert");
        idx.upsert("a2", &[0.0, 1.0], &record, Side::A)
            .expect("test vector should insert");
        let mut config = make_config_exact("name", "name");
        config.vector_backend = "flat".into();
        config.thresholds.min_score_gap = Some(0.1);
        let ids = vec!["a1".into(), "a2".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::B,
            combined_vec: &[1.0, 0.0],
        };
        let pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx),
            blocked_ids: &ids,
            bm25_candidate_ids: &ids[..1],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &ids[..1],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        let ordinary = score_pool(&query, &pool, &config, 1, 0);
        assert_eq!(ordinary.len(), 1, "retrieval limit must stay unchanged");
        assert_eq!(ordinary[0].classification, Classification::Auto);
        let empty_verified = score_pool_verified(&query, &pool, &config, 1, 0, &[])
            .expect("empty verification should score");
        assert_eq!(empty_verified[0].score, ordinary[0].score);
        assert_eq!(empty_verified[0].classification, ordinary[0].classification);

        for verification in [ids.clone(), vec!["a2".into(), "a2".into()]] {
            for top_n in [0, 1] {
                let results = score_pool_verified(&query, &pool, &config, 1, top_n, &verification)
                    .expect("verification should score");
                assert_eq!(results.len(), if top_n == 0 { 2 } else { 1 });
                assert_eq!(results[0].matched_id, "a1", "ties use ID order");
                assert_eq!(results[0].rank, Some(1));
                assert_eq!(
                    results[0].classification,
                    Classification::Review,
                    "verification runner-up must affect gap even with top_n=1",
                );
                assert_eq!(results[0].reason.as_deref(), Some("downgraded"));
                if top_n == 0 {
                    assert_eq!(results[1].matched_id, "a2");
                    assert_eq!(results[1].rank, Some(2));
                }
            }
        }
    }

    #[test]
    fn verified_ids_respect_blocking_exclusions_and_self_filter() {
        let store = make_store();
        let record = make_record(&[("name", "Acme")]);
        for id in ["query", "excluded", "outside", "allowed"] {
            store
                .insert(Side::A, id, &record)
                .expect("test record should insert");
        }
        let config = make_config_exact("name", "name");
        let ids = vec!["query".into(), "excluded".into(), "allowed".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        exclusions.add("query", "excluded");
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::A,
            combined_vec: &[],
        };
        let mut pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &ids,
            bm25_candidate_ids: &[],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        for verification in [
            vec!["query".into(), "excluded".into()],
            vec!["outside".into(), "outside".into()],
        ] {
            let results = score_pool_verified(&query, &pool, &config, 1, 0, &verification)
                .expect("filtered verification should succeed");
            assert!(results.is_empty(), "ineligible IDs must never be scored");
        }
        let verification = vec!["allowed".into()];
        assert_eq!(
            score_pool_verified(&query, &pool, &config, 1, 0, &verification)
                .expect("eligible verification should score")
                .len(),
            1,
            "verification can add an otherwise unretrieved candidate",
        );
        pool.blocked_ids = &[];
        assert!(
            score_pool_verified(&query, &pool, &config, 1, 0, &verification)
                .expect("empty block should succeed")
                .is_empty(),
            "verification cannot bypass an empty block",
        );
    }

    #[test]
    fn verified_bm25_uses_supplied_scores_and_missing_hits_are_zero() {
        let store = make_store();
        let record = Record::new();
        let ids = vec!["scored".to_owned(), "zero".to_owned()];
        for id in &ids {
            store
                .insert(Side::A, id, &record)
                .expect("test record should insert");
        }
        let mut config = make_config_exact("name", "name");
        config.match_fields[0].method = MatchMethod::Bm25;
        let scores = HashMap::from([("scored".into(), 0.93)]);
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        let pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &ids,
            bm25_candidate_ids: &[],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        let results = score_pool_verified(&query, &pool, &config, 1, 0, &ids)
            .expect("BM25 verification should score");
        assert_eq!(results.len(), 2, "zero BM25 hit must not omit the pair");
        assert_eq!(results[0].matched_id, "scored");
        assert!(
            (results[0].score - 0.93).abs() < 1e-9,
            "use normalized BM25"
        );
        assert_eq!(results[1].score, 0.0, "absent hit means zero");
    }

    #[test]
    fn verified_propagates_extra_fetch_errors_but_greedy_stays_lenient() {
        let dir = tempfile::tempdir().expect("test directory should create");
        let columns = vec!["name".into()];
        let (store, _, writer) = crate::store::sqlite::open_sqlite(
            &dir.path().join("records.db"),
            &BlockingConfig::default(),
            None,
            &columns,
            &columns,
        )
        .expect("test store should open");
        writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .execute_batch("DROP TABLE a_records")
            .expect("test table should drop to inject read failure");
        let record = Record::new();
        let ids = vec!["unreadable".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        // Cover both full-record and projected-field extra fetches, and
        // failures from every extra-candidate source, even with no verification.
        for method in [MatchMethod::Exact, MatchMethod::Bm25] {
            let mut config = make_config_exact("name", "name");
            config.match_fields[0].method = method;
            for source in 0..3 {
                let pool = ScoringPool {
                    store: &store,
                    side: Side::A,
                    combined_index: None,
                    blocked_ids: &ids,
                    bm25_candidate_ids: if source == 0 { &ids } else { &[] },
                    bm25_scores_map: &scores,
                    synonym_candidate_ids: if source == 1 { &ids } else { &[] },
                    synonym_dictionary: None,
                    exclusions: &exclusions,
                };
                let verification = if source == 2 { ids.as_slice() } else { &[] };
                assert!(
                    score_pool_verified(&query, &pool, &config, 1, 0, verification).is_err(),
                    "strict extra fetch failure must propagate: source={source}",
                );
                assert!(
                    score_pool(&query, &pool, &config, 1, 0).is_empty(),
                    "greedy fetch failure must retain its lenient behavior",
                );
            }
        }
    }

    #[test]
    fn score_pool_top_n_truncates() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Insert 10 A records.
        let mut blocked = Vec::new();
        for i in 0..10 {
            let id = format!("a{}", i);
            let rec = make_record(&[("id", &id), ("name", &format!("Corp {}", i))]);
            store.insert(Side::A, &id, &rec).unwrap();
            idx.upsert(&id, &unit_vec(dim, i as u64), &rec, Side::A)
                .unwrap();
            blocked.push(id);
        }

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Corp 0")]);
        let query_vec = unit_vec(dim, 99);
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &query_vec,
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx as &dyn VectorDB),
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 3);
        assert!(
            results.len() <= 3,
            "top_n=3 should truncate, got {}",
            results.len()
        );
    }
}
