//! Unified scoring pipeline shared by batch and live modes.
//!
//! Three stages:
//!   1. Blocking filter  — now done by the *caller* before invoking score_pool;
//!                         `blocked_ids` is passed in as a pre-queried slice.
//!   2. Candidate selection — ANN search (usearch) or flat scan to get top_n
//!   3. Full scoring — score candidates on all match_fields, classify, sort
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

use dashmap::DashMap;

use crate::config::Config;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;
use crate::vectordb::VectorDB;

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
/// - `query_combined_vec`:   pre-encoded combined embedding vector for the query
///                           (empty slice if no embedding fields configured)
/// - `pool_records`:         opposite-side records (DashMap)
/// - `pool_combined_index`:  combined embedding index for the pool side
///                           (`None` if no embedding fields configured)
/// - `blocked_ids`:          IDs pre-selected by the caller's blocking query;
///                           pass all pool IDs if blocking is disabled
/// - `config`:               job config
/// - `top_n`:                max candidates from ANN search and max results to
///                           return (0 = no limit, flat path only)
pub fn score_pool(
    query_id: &str,
    query_record: &Record,
    query_side: Side,
    query_combined_vec: &[f32],
    pool_records: &DashMap<String, Record>,
    pool_combined_index: Option<&dyn VectorDB>,
    blocked_ids: &[String],
    config: &Config,
    top_n: usize,
) -> Vec<MatchResult> {
    if blocked_ids.is_empty() {
        return Vec::new();
    }

    // --- Stage 2: Candidate selection (ANN or flat scan) ---
    let cands = candidates::select_candidates(
        query_combined_vec,
        top_n,
        pool_combined_index,
        blocked_ids,
        pool_records,
        query_record,
        query_side,
        &config.vector_backend,
    );

    if cands.is_empty() {
        return Vec::new();
    }

    // --- Stage 3: Full scoring with per-field decomposition ---
    let emb_specs = crate::vectordb::embedding_field_specs(config);
    let has_embeddings = !query_combined_vec.is_empty() && !emb_specs.is_empty();

    let mut results: Vec<MatchResult> = cands
        .iter()
        .map(|cand| {
            // Decompose combined vecs → per-field cosine similarities.
            let emb_scores: Option<HashMap<String, f64>> = if has_embeddings {
                pool_combined_index
                    .and_then(|idx| idx.get(&cand.id).ok().flatten())
                    .map(|pool_vec| decompose_emb_scores(query_combined_vec, &pool_vec, &emb_specs))
            } else {
                None
            };

            let score_result = scoring::score_pair(
                query_record,
                &cand.record,
                &config.match_fields,
                emb_scores.as_ref(),
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
    if let Some(top) = results.first_mut() {
        if let Some(cand) = cands.iter().find(|c| c.id == top.matched_id) {
            top.matched_record = Some(apply_output_mapping(&cand.record, config));
        }
    }

    results
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
