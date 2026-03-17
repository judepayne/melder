//! Unified scoring pipeline shared by batch and live modes.
//!
//! Four stages:
//!   1. Blocking filter — done by the *caller* before invoking score_pool;
//!      `blocked_ids` is passed in as a pre-queried slice.
//!   2. ANN candidate selection — ANN search to get `ann_candidates`
//!   3. BM25 re-rank/filter — narrow to `bm25_candidates` (optional)
//!   4. Full scoring — score candidates on all match_fields, classify, sort,
//!      truncate to `top_n`
//!
//! Both batch and live call `score_pool()` with the same arguments; the only
//! difference is where the data comes from (MatchState vs LiveSideState).
//!
//! ## Sequential pipeline modes
//!
//! Depending on which methods are configured, the pipeline degrades gracefully:
//! - ANN+BM25: block → ANN(ann_candidates) → BM25(bm25_candidates) → score(top_n)
//! - ANN only: block → ANN(ann_candidates) → score(top_n)
//! - BM25 only: block → BM25(bm25_candidates) → score(top_n)
//! - Neither:   block → score all → truncate(top_n)
//!
//! ## Why blocked_ids is passed in rather than BlockingIndex
//!
//! In live mode the blocking index is a `RwLock<BlockingIndex>`. Previously
//! `score_pool` held the read lock for its entire duration (~5-15ms at 100k
//! scale), starving opposite-side writes. The caller now takes the lock only
//! for the `.query()` call (~1µs) and passes the resulting owned `Vec<String>`
//! here. The pipeline is therefore lock-free for its entire execution.

use std::collections::HashMap;

use crate::config::Config;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// Optional BM25 context passed into the pipeline.
///
/// When the `bm25` feature is enabled, this carries an immutable reference to
/// the pool-side BM25 index, the concatenated query text, and a pre-computed
/// self-score for normalisation. When disabled, this is a zero-size unit struct.
#[cfg(feature = "bm25")]
pub struct Bm25Ctx<'a> {
    pub index: &'a crate::bm25::index::BM25Index,
    pub query_text: String,
    /// Pre-computed self-score for BM25 normalisation. Populated from cache
    /// (batch pre-compute or prior event) or analytically on cache miss.
    pub self_score: f32,
}

#[cfg(not(feature = "bm25"))]
pub struct Bm25Ctx;

#[cfg(feature = "bm25")]
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

#[cfg(not(feature = "bm25"))]
impl Bm25Ctx {
    /// No-op constructor for when BM25 is not compiled in.
    pub fn none() -> Self {
        Self
    }
}

/// Score a single query record against the opposite-side pool.
///
/// Returns a sorted `Vec<MatchResult>` (descending by score), truncated to
/// `top_n` entries. The top result carries an attached `matched_record` with
/// output mapping applied.
///
/// Parameters:
/// - `query_id`:             ID of the query record
/// - `query_record`:         the query record itself
/// - `query_side`:           which side the query belongs to (A or B)
/// - `query_combined_vec`: pre-encoded combined embedding vector for the query
///   (empty slice if no embedding fields configured)
/// - `pool_store`: record store for looking up pool-side records
/// - `pool_side`: which side of the store the pool records are on
/// - `pool_combined_index`: combined embedding index for the pool side
///   (`None` if no embedding fields configured)
/// - `blocked_ids`: IDs pre-selected by the caller's blocking query;
///   pass all pool IDs if blocking is disabled
/// - `config`: job config
/// - `ann_candidates`: how many candidates ANN retrieves from the full block
/// - `bm25_candidates`: how many candidates BM25 keeps after re-ranking
/// - `top_n`: max results to return (0 = no limit)
/// - `pool_bm25_index`: pool-side BM25 index for candidate filtering and
///   scoring (`None` if BM25 not configured or feature not enabled)
/// - `query_bm25_text`: concatenated query text for BM25 queries (empty if
///   no BM25)
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
) -> Vec<MatchResult> {
    if blocked_ids.is_empty() {
        return Vec::new();
    }

    let has_embeddings = !query_combined_vec.is_empty();
    let has_bm25 = config.match_fields.iter().any(|mf| mf.method == "bm25");

    // Compute the pool-side fields needed for scoring (used by get_many_fields
    // to avoid full JSON deserialization in SQLite).
    let scoring_fields: Vec<String> = config
        .match_fields
        .iter()
        .filter(|mf| mf.method != "bm25" && mf.method != "embedding")
        .map(|mf| match pool_side {
            Side::A => mf.field_a.clone(),
            Side::B => mf.field_b.clone(),
        })
        .collect();

    // --- Fast path: BM25-only (no embeddings) ---
    // When BM25 is the sole candidate filter, query the BM25 index directly
    // with blocking filters and fetch only the survivors. This avoids
    // loading all blocked records into memory (e.g. 50K records per country
    // bucket at 1M scale) when only ~20 will be scored.
    #[cfg(feature = "bm25")]
    if has_bm25
        && !has_embeddings
        && let Some(ctx) = bm25_ctx
    {
        use crate::bm25::scorer::normalise_bm25;

        // Query BM25 index with blocking filter — Tantivy skips
        // documents from the wrong block entirely (no wasted scoring).
        // Returns exactly the top bm25_candidates from the matching block.
        let raw_results =
            ctx.index
                .query_blocked(&ctx.query_text, bm25_candidates, query_record, query_side);

        // Use pre-computed self-score for normalisation (computed eagerly
        // at Bm25Ctx construction — cache hit or analytical fallback).
        let norm_ceiling = ctx.self_score;

        let scored: Vec<(String, f64)> = raw_results
            .into_iter()
            .map(|(id, raw)| {
                let norm = normalise_bm25(raw, norm_ceiling);
                (id, norm)
            })
            .collect();

        if scored.is_empty() {
            return Vec::new();
        }

        // Fetch only the BM25 survivors (typically ~20 records)
        let survivor_ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
        let bm25_scores_map: HashMap<String, f64> = scored.into_iter().collect();
        let survivor_records = if scoring_fields.is_empty() {
            pool_store.get_many(pool_side, &survivor_ids)
        } else {
            pool_store.get_many_fields(pool_side, &survivor_ids, &scoring_fields)
        };

        let survivor_cands: Vec<candidates::Candidate> = survivor_records
            .into_iter()
            .map(|(id, record)| candidates::Candidate {
                id,
                record,
                combined_dot: 0.0,
            })
            .collect();

        // Score the survivors using the same path as the standard pipeline
        let mut results: Vec<MatchResult> = survivor_cands
            .iter()
            .map(|cand| {
                let bm25_score = bm25_scores_map.get(&cand.id).copied();
                let score_result = scoring::score_pair(
                    query_record,
                    &cand.record,
                    &config.match_fields,
                    None, // no embedding scores in BM25-only path
                    bm25_score,
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
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if top_n > 0 {
            results.truncate(top_n);
        }

        return results;
    }

    // --- Standard path: ANN candidate selection → BM25 re-rank → full scoring ---

    // --- Stage 2: ANN candidate selection ---
    let cands = candidates::select_candidates(
        query_combined_vec,
        ann_candidates,
        pool_combined_index,
        blocked_ids,
        pool_store,
        pool_side,
        query_record,
        query_side,
        &config.vector_backend,
        &scoring_fields,
    );

    if cands.is_empty() {
        return Vec::new();
    }

    // --- Stage 3: BM25 re-rank / filter ---
    // Collect candidate IDs after ANN; BM25 may narrow this list.
    #[cfg(feature = "bm25")]
    let (cand_ids_after_bm25, bm25_scores_map) = if has_bm25 {
        if let Some(ctx) = bm25_ctx {
            bm25_filter_stage(
                &cands,
                Some(ctx.index),
                &ctx.query_text,
                ctx.self_score,
                has_embeddings,
                bm25_candidates,
                blocked_ids,
            )
        } else {
            (
                cands.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
                HashMap::new(),
            )
        }
    } else {
        (
            cands.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            HashMap::new(),
        )
    };

    #[cfg(not(feature = "bm25"))]
    let (cand_ids_after_bm25, bm25_scores_map): (Vec<String>, HashMap<String, f64>) = {
        let _ = has_bm25;
        let _ = bm25_candidates;
        let _ = bm25_ctx;
        (cands.iter().map(|c| c.id.clone()).collect(), HashMap::new())
    };

    // Filter candidates to those that survived BM25 (or all if no BM25)
    let final_cands: Vec<&candidates::Candidate> = cands
        .iter()
        .filter(|c| cand_ids_after_bm25.contains(&c.id))
        .collect();

    if final_cands.is_empty() {
        return Vec::new();
    }

    // --- Stage 4: Full scoring with per-field decomposition ---
    let emb_specs = crate::vectordb::embedding_field_specs(config);
    let has_emb_specs = has_embeddings && !emb_specs.is_empty();

    let mut results: Vec<MatchResult> = final_cands
        .iter()
        .map(|cand| {
            // Decompose combined vecs → per-field cosine similarities.
            let emb_scores: Option<HashMap<String, f64>> = if has_emb_specs {
                pool_combined_index
                    .and_then(|idx| idx.get(&cand.id).ok().flatten())
                    .map(|pool_vec| decompose_emb_scores(query_combined_vec, &pool_vec, &emb_specs))
            } else {
                None
            };

            let precomputed_bm25 = bm25_scores_map.get(&cand.id).copied();

            let score_result = scoring::score_pair(
                query_record,
                &cand.record,
                &config.match_fields,
                emb_scores.as_ref(),
                precomputed_bm25,
            );

            scoring::build_match_result(
                query_id,
                &cand.id,
                query_side,
                score_result,
                config.thresholds.auto_match,
                config.thresholds.review_floor,
                None, // matched_record attached lazily below
                false,
            )
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to top_n if specified
    if top_n > 0 {
        results.truncate(top_n);
    }

    // Attach matched record (with output mapping) to the top result only.
    if let Some(top) = results.first_mut()
        && let Some(cand) = cands.iter().find(|c| c.id == top.matched_id)
    {
        top.matched_record = Some(apply_output_mapping(&cand.record, config));
    }

    results
}

/// BM25 filtering stage. When ANN produced candidates, re-rank them by BM25
/// and keep the top `bm25_candidates`. When no ANN (no embeddings), query
/// the BM25 index directly.
///
/// `self_score` is the pre-computed self-score from the `Bm25Ctx`.
///
/// Returns `(surviving_ids, id→normalised_bm25_score)`.
#[cfg(feature = "bm25")]
fn bm25_filter_stage(
    ann_cands: &[candidates::Candidate],
    pool_bm25_index: Option<&crate::bm25::index::BM25Index>,
    query_bm25_text: &str,
    self_score: f32,
    has_embeddings: bool,
    bm25_candidates: usize,
    blocked_ids: &[String],
) -> (Vec<String>, HashMap<String, f64>) {
    use crate::bm25::scorer::normalise_bm25;

    let bm25_idx = match pool_bm25_index {
        Some(idx) => idx,
        None => {
            return (
                ann_cands.iter().map(|c| c.id.clone()).collect(),
                HashMap::new(),
            );
        }
    };

    if query_bm25_text.is_empty() {
        return (
            ann_cands.iter().map(|c| c.id.clone()).collect(),
            HashMap::new(),
        );
    }

    if has_embeddings {
        // ANN+BM25 mode: re-rank ANN candidates by BM25.
        let candidate_ids: Vec<String> = ann_cands.iter().map(|c| c.id.clone()).collect();
        let (raw_scores, _index_norm) =
            bm25_idx.score_candidates(query_bm25_text, &candidate_ids, bm25_candidates * 3);

        // Use the pre-computed self-score for normalisation.
        let norm_ceiling = self_score;

        let mut scored: Vec<(String, f64)> = candidate_ids
            .into_iter()
            .map(|id| {
                let raw = raw_scores.get(&id).copied().unwrap_or(0.0);
                let norm = normalise_bm25(raw, norm_ceiling);
                (id, norm)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(bm25_candidates);

        let ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
        let map: HashMap<String, f64> = scored.into_iter().collect();
        (ids, map)
    } else {
        // BM25-only mode: query the index directly
        let raw_results = bm25_idx.query(query_bm25_text, bm25_candidates * 3);
        let norm_ceiling = self_score;
        let blocked_set: std::collections::HashSet<&str> =
            blocked_ids.iter().map(|s| s.as_str()).collect();

        let mut scored: Vec<(String, f64)> = raw_results
            .into_iter()
            .filter(|(id, _)| blocked_set.contains(id.as_str()))
            .map(|(id, raw)| {
                let norm = normalise_bm25(raw, norm_ceiling);
                (id, norm)
            })
            .take(bm25_candidates)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
        let map: HashMap<String, f64> = scored.into_iter().collect();
        (ids, map)
    }
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

        let dot: f32 = q_sub.iter().zip(p_sub.iter()).map(|(a, b)| a * b).sum();

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
}
