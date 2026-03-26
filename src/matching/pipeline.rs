//! Unified scoring pipeline shared by batch and live modes.
//!
//! ## Pipeline stages
//!
//!   1. Blocking filter — done by the *caller* before invoking score_pool;
//!      `blocked_ids` is passed in as a pre-queried slice.
//!   2. Independent candidate generation — each method runs in parallel:
//!      - ANN search (O(log N) HNSW, `ann_candidates`)
//!      - BM25 search (O(log N) Tantivy, `bm25_candidates`)
//!      - (Future: synonym lookup, etc.)
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

use crate::bm25::scorer::normalise_bm25;
use crate::config::Config;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;
use crate::scoring::embedding::dot_product_f32;
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// BM25 context passed into the pipeline.
///
/// Carries an immutable reference to the pool-side BM25 index, the
/// concatenated query text, and a pre-computed self-score for normalisation.
pub struct Bm25Ctx<'a> {
    pub index: &'a crate::bm25::index::BM25Index,
    pub query_text: String,
    /// Pre-computed self-score for BM25 normalisation. Populated from cache
    /// (batch pre-compute or prior event) or analytically on cache miss.
    pub self_score: f32,
}

impl<'a> Bm25Ctx<'a> {
    /// Create a new BM25 context.
    ///
    /// The self-score is resolved eagerly: cache hit → O(1), cache miss →
    /// analytical computation → O(unique tokens), ~5–20µs.
    pub fn new(index: &'a crate::bm25::index::BM25Index, query_text: String) -> Self {
        let self_score = index
            .cached_self_score(&query_text)
            .unwrap_or_else(|| index.analytical_self_score(&query_text));
        Self {
            index,
            query_text,
            self_score,
        }
    }
}

/// Score a single query record against the opposite-side pool.
///
/// Runs independent candidate generators (ANN, BM25), unions their results,
/// then scores all candidates on every configured match_field. Returns a
/// sorted `Vec<MatchResult>` (descending by score), truncated to `top_n`
/// entries. The top result carries an attached `matched_record` with output
/// mapping applied.
#[allow(clippy::too_many_arguments)]
pub fn score_pool(
    query_id: &str,
    query_record: &Record,
    query_side: Side,
    query_combined_vec: &[f32],
    pool_store: &dyn RecordStore,
    pool_side: Side,
    pool_combined_index: Option<&dyn VectorDB>,
    blocked_ids: &[String],
    config: &Config,
    ann_candidates: usize,
    bm25_candidates: usize,
    top_n: usize,
    bm25_ctx: Option<Bm25Ctx>,
    synonym_index: Option<&crate::synonym::index::SynonymIndex>,
    synonym_dictionary: Option<&crate::synonym::dictionary::SynonymDictionary>,
) -> Vec<MatchResult> {
    if blocked_ids.is_empty() {
        return Vec::new();
    }

    let has_embeddings = !query_combined_vec.is_empty();

    // Pool-side fields needed for scoring (used by get_many_fields to
    // avoid full JSON deserialization in SQLite).
    let scoring_fields: Vec<String> = config
        .match_fields
        .iter()
        .filter(|mf| mf.method != "bm25" && mf.method != "embedding")
        .map(|mf| match pool_side {
            Side::A => mf.field_a.clone(),
            Side::B => mf.field_b.clone(),
        })
        .collect();

    // --- Stage 2: Independent candidate generation ---
    //
    // Each generator runs independently against the blocked pool. The union
    // of all candidate sets flows to full scoring.

    // ANN candidates (empty if no embedding fields configured).
    let ann_cands = candidates::select_candidates(
        query_combined_vec,
        ann_candidates,
        pool_combined_index,
        blocked_ids,
        pool_store,
        pool_side,
        query_record,
        query_side,
        &config.vector_backend,
    );

    // BM25 candidates (empty if no BM25 configured).
    let (bm25_cand_ids, bm25_scores_map) = {
        let has_bm25 = config.match_fields.iter().any(|mf| mf.method == "bm25");
        if has_bm25 {
            if let Some(ctx) = bm25_ctx {
                bm25_candidate_stage(&ctx, bm25_candidates, query_record, query_side)
            } else {
                (Vec::new(), HashMap::new())
            }
        } else {
            (Vec::new(), HashMap::new())
        }
    };

    // Synonym candidates (empty if no synonym_fields configured).
    let synonym_cand_ids = if !config.synonym_fields.is_empty() {
        if let Some(idx) = synonym_index {
            synonym_candidate_stage(idx, query_record, query_side, &config.synonym_fields)
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // --- Stage 3: Union candidates ---
    //
    // Merge ANN, BM25, and synonym candidate sets, deduplicating by ID.
    // Candidates not in the ANN set need their records fetched from the store.
    let ann_id_set: std::collections::HashSet<&str> =
        ann_cands.iter().map(|c| c.id.as_str()).collect();

    // Collect non-ANN candidate IDs (BM25 + synonym), deduplicated.
    let mut extra_ids: Vec<String> = Vec::new();
    let mut extra_id_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for id in bm25_cand_ids.iter().chain(synonym_cand_ids.iter()) {
        if !ann_id_set.contains(id.as_str()) && extra_id_set.insert(id.clone()) {
            extra_ids.push(id.clone());
        }
    }

    // Fetch records for non-ANN candidates.
    let extra_cands: Vec<candidates::Candidate> = if extra_ids.is_empty() {
        Vec::new()
    } else {
        let records = if scoring_fields.is_empty() {
            pool_store.get_many(pool_side, &extra_ids)
        } else {
            pool_store.get_many_fields(pool_side, &extra_ids, &scoring_fields)
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
        return Vec::new();
    }

    // Self-match exclusion: when query_side == pool_side (enroll mode),
    // the query record may appear in the pool index. Filter it out before
    // scoring so a record never matches itself.
    let filter_self = query_side == pool_side;

    // --- Stage 4: Full scoring with per-field decomposition ---
    let emb_specs = crate::vectordb::embedding_field_specs(config);
    let has_emb_specs = has_embeddings && !emb_specs.is_empty();

    let mut results: Vec<MatchResult> = Vec::with_capacity(ann_cands.len() + extra_cands.len());

    // Score all candidates through the same path.
    for cand in ann_cands.iter().chain(extra_cands.iter()) {
        // Skip self-match in enroll mode
        if filter_self && cand.id == query_id {
            continue;
        }
        results.push(score_candidate(
            cand,
            query_id,
            query_record,
            query_side,
            query_combined_vec,
            pool_combined_index,
            config,
            &bm25_scores_map,
            &emb_specs,
            has_emb_specs,
            synonym_dictionary,
        ));
    }

    // Sort by score descending.
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

    results
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
    let emb_scores: Option<HashMap<String, f64>> = if has_emb_specs {
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

/// Independent BM25 candidate generation stage.
///
/// Queries the Tantivy index directly with blocking filters to surface the
/// top `bm25_candidates` records. Returns candidate IDs and their normalised
/// BM25 scores.
///
/// This runs independently of ANN — it can find candidates that ANN missed
/// entirely (e.g. records with shared distinctive tokens but low embedding
/// similarity).
fn bm25_candidate_stage(
    ctx: &Bm25Ctx,
    bm25_candidates: usize,
    query_record: &Record,
    query_side: Side,
) -> (Vec<String>, HashMap<String, f64>) {
    if ctx.query_text.is_empty() {
        return (Vec::new(), HashMap::new());
    }

    // Query BM25 index with blocking filter — Tantivy skips documents
    // from the wrong block entirely (no wasted scoring).
    let raw_results =
        ctx.index
            .query_blocked(&ctx.query_text, bm25_candidates, query_record, query_side);

    let norm_ceiling = ctx.self_score;

    let scored: Vec<(String, f64)> = raw_results
        .into_iter()
        .map(|(id, raw)| {
            let norm = normalise_bm25(raw, norm_ceiling);
            (id, norm)
        })
        .collect();

    let ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
    let map: HashMap<String, f64> = scored.into_iter().collect();
    (ids, map)
}

/// Independent synonym candidate generation stage.
///
/// For each configured synonym field, looks up the query record's field value
/// in the opposite-side synonym index. Returns deduplicated candidate IDs.
///
/// Typically produces 0-2 candidates per query — synonym matches are rare
/// but critical for acronym recovery.
fn synonym_candidate_stage(
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
fn decompose_emb_scores(
    query_combined: &[f32],
    pool_combined: &[f32],
    emb_specs: &[(String, String, f64)],
) -> HashMap<String, f64> {
    if emb_specs.is_empty() || query_combined.is_empty() || pool_combined.is_empty() {
        return HashMap::new();
    }
    let field_dim = query_combined.len() / emb_specs.len();
    if field_dim == 0 {
        return HashMap::new();
    }

    let mut scores = HashMap::with_capacity(emb_specs.len());

    for (i, (field_a, field_b, weight)) in emb_specs.iter().enumerate() {
        let start = i * field_dim;
        let end = start + field_dim;

        if end > query_combined.len() || end > pool_combined.len() {
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

        let key = format!("{}/{}", field_a, field_b);
        scores.insert(key, cos);
    }

    scores
}

/// Apply output_mapping to a matched record: rename fields per config.
fn apply_output_mapping(record: &Record, config: &Config) -> Record {
    if config.output_mapping.is_empty() {
        return record.clone();
    }

    let mut result = record.clone();
    for mapping in &config.output_mapping {
        if let Some(val) = record.get(&mapping.from) {
            result.insert(mapping.to.clone(), val.clone());
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

        let cos = *scores.get("f_a/f_b").unwrap();
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

        let cos = *scores.get("f_a/f_b").unwrap();
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

        let cos1 = *scores.get("f1_a/f1_b").unwrap();
        let cos2 = *scores.get("f2_a/f2_b").unwrap();

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
  results_path: /tmp/test/results.csv
  review_path: /tmp/test/review.csv
  unmatched_path: /tmp/test/unmatched.csv
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
        let query = make_record(&[("id", "q1"), ("name", "Foo")]);

        let results = score_pool(
            "q1",
            &query,
            Side::B,
            &[],
            &store,
            Side::A,
            None,
            &[],
            &config,
            50,
            10,
            5,
            None,
            None,
            None,
        );

        assert!(results.is_empty(), "empty blocked_ids → no results");
    }

    #[test]
    fn score_pool_no_embeddings_no_bm25_returns_empty() {
        let store = make_store();
        store.insert(
            Side::A,
            "a1",
            &make_record(&[("id", "a1"), ("name", "Foo Corp")]),
        );

        let config = make_config_exact("name", "name");
        let query = make_record(&[("id", "q1"), ("name", "Foo Corp")]);
        let blocked = vec!["a1".to_string()];

        // No embeddings (empty vec), no BM25 context → both generators
        // return empty → union is empty → no results.
        let results = score_pool(
            "q1",
            &query,
            Side::B,
            &[],
            &store,
            Side::A,
            None,
            &blocked,
            &config,
            50,
            10,
            5,
            None,
            None,
            None,
        );

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
        store.insert(Side::A, "a1", &a1);
        store.insert(Side::A, "a2", &a2);

        let v1 = unit_vec(dim, 1);
        let v2 = unit_vec(dim, 2);
        idx.upsert("a1", &v1, &a1, Side::A).unwrap();
        idx.upsert("a2", &v2, &a2, Side::A).unwrap();

        let config = make_config_exact("name", "name");
        let query = make_record(&[("id", "q1"), ("name", "Foo Corp")]);
        let blocked = vec!["a1".to_string(), "a2".to_string()];
        let query_vec = unit_vec(dim, 1); // identical to a1

        let results = score_pool(
            "q1",
            &query,
            Side::B,
            &query_vec,
            &store,
            Side::A,
            Some(&idx as &dyn VectorDB),
            &blocked,
            &config,
            50,
            10,
            5,
            None,
            None,
            None,
        );

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
        store.insert(Side::A, "a1", &a1);
        store.insert(Side::A, "a2", &a2);
        store.insert(Side::A, "a3", &a3);

        for (id, seed, rec) in [("a1", 1u64, &a1), ("a2", 2, &a2), ("a3", 3, &a3)] {
            idx.upsert(id, &unit_vec(dim, seed), rec, Side::A).unwrap();
        }

        let config = make_config_exact("name", "name");
        let query = make_record(&[("id", "q1"), ("name", "Acme Corp")]);
        let blocked = vec!["a1".to_string(), "a2".to_string(), "a3".to_string()];
        let query_vec = unit_vec(dim, 99);

        let results = score_pool(
            "q1",
            &query,
            Side::B,
            &query_vec,
            &store,
            Side::A,
            Some(&idx as &dyn VectorDB),
            &blocked,
            &config,
            50,
            10,
            5,
            None,
            None,
            None,
        );

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
    fn score_pool_top_n_truncates() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Insert 10 A records.
        let mut blocked = Vec::new();
        for i in 0..10 {
            let id = format!("a{}", i);
            let rec = make_record(&[("id", &id), ("name", &format!("Corp {}", i))]);
            store.insert(Side::A, &id, &rec);
            idx.upsert(&id, &unit_vec(dim, i as u64), &rec, Side::A)
                .unwrap();
            blocked.push(id);
        }

        let config = make_config_exact("name", "name");
        let query = make_record(&[("id", "q1"), ("name", "Corp 0")]);
        let query_vec = unit_vec(dim, 99);

        let results = score_pool(
            "q1",
            &query,
            Side::B,
            &query_vec,
            &store,
            Side::A,
            Some(&idx as &dyn VectorDB),
            &blocked,
            &config,
            50, // ann_candidates
            10, // bm25_candidates
            3,  // top_n = 3
            None,
            None,
            None,
        );

        assert!(
            results.len() <= 3,
            "top_n=3 should truncate, got {}",
            results.len()
        );
    }
}
