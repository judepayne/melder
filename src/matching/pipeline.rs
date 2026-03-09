//! Unified scoring pipeline shared by batch and live modes.
//!
//! Three stages:
//!   1. Blocking filter  — narrow the pool to records sharing a blocking key
//!   2. Candidate selection — score blocked records on one field pair, keep top N
//!   3. Full scoring — score candidates on all match_fields, classify, sort
//!
//! Both batch and live call `score_pool()` with the same arguments; the only
//! difference is where the data comes from (MatchState vs LiveSideState).

use std::collections::HashMap;

use dashmap::DashMap;

use crate::config::Config;
use crate::index::VecIndex;
use crate::matching::blocking::BlockingIndex;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;

/// Score a single query record against the opposite-side pool.
///
/// Returns a sorted `Vec<MatchResult>` (descending by score), truncated to
/// `top_n` entries. The top result carries an attached `matched_record` with
/// output mapping applied.
///
/// Parameters:
/// - `query_id`:     ID of the query record
/// - `query_record`: the query record itself
/// - `query_vec`:    the query record's embedding vector
/// - `query_side`:   which side the query belongs to (A or B)
/// - `pool_records`: opposite-side records (DashMap — works for both batch and live)
/// - `pool_index`:   opposite-side VecIndex
/// - `blocking_index`: optional BlockingIndex for the opposite side
/// - `config`:       job config
/// - `top_n`:        max results to return (0 = unlimited)
pub fn score_pool(
    query_id: &str,
    query_record: &Record,
    query_vec: &[f32],
    query_side: Side,
    pool_records: &DashMap<String, Record>,
    pool_index: &VecIndex,
    blocking_index: Option<&BlockingIndex>,
    config: &Config,
    top_n: usize,
) -> Vec<MatchResult> {
    // --- Stage 1: Blocking filter ---
    let candidate_ids: Vec<String> = if config.blocking.enabled {
        if let Some(bi) = blocking_index {
            bi.query(query_record, query_side).into_iter().collect()
        } else {
            // No blocking index available — fall through to all pool records
            pool_records.iter().map(|e| e.key().clone()).collect()
        }
    } else {
        pool_records.iter().map(|e| e.key().clone()).collect()
    };

    if candidate_ids.is_empty() {
        return Vec::new();
    }

    // --- Stage 2: Candidate selection ---
    let cands = candidates::select_candidates(
        query_record,
        query_vec,
        &candidate_ids,
        pool_records,
        pool_index,
        query_side,
        config,
    );

    if cands.is_empty() {
        return Vec::new();
    }

    // --- Stage 3: Full scoring ---
    // Pre-compute embedding field keys for score_pair precomputed_emb_scores.
    let emb_keys: Vec<String> = config
        .match_fields
        .iter()
        .filter(|mf| mf.method == "embedding")
        .map(|mf| format!("{}/{}", mf.field_a, mf.field_b))
        .collect();

    let mut results: Vec<MatchResult> = cands
        .iter()
        .map(|cand| {
            // Build precomputed embedding scores
            let emb_scores = if !emb_keys.is_empty() {
                if let Some(sim) = cand.embedding_score {
                    let mut map = HashMap::with_capacity(emb_keys.len());
                    for key in &emb_keys {
                        map.insert(key.clone(), sim);
                    }
                    Some(map)
                } else {
                    None
                }
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
