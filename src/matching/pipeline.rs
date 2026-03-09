//! Unified scoring pipeline shared by batch and live modes.
//!
//! Three stages:
//!   1. Blocking filter  — narrow the pool to records sharing a blocking key
//!   2. Candidate selection — score blocked records on one field pair, keep top N
//!   3. Full scoring — score candidates on all match_fields, classify, sort
//!
//! Both batch and live call `score_pool()` with the same arguments; the only
//! difference is where the data comes from (MatchState vs LiveSideState).

use dashmap::DashMap;

use crate::config::Config;
use crate::matching::blocking::BlockingIndex;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;
use crate::vectordb::field_vectors::FieldVectors;

/// Score a single query record against the opposite-side pool.
///
/// Returns a sorted `Vec<MatchResult>` (descending by score), truncated to
/// `top_n` entries. The top result carries an attached `matched_record` with
/// output mapping applied.
///
/// Parameters:
/// - `query_id`:          ID of the query record
/// - `query_record`:      the query record itself
/// - `query_side`:        which side the query belongs to (A or B)
/// - `pool_records`:      opposite-side records (DashMap)
/// - `query_field_vecs`:  per-field embedding vectors for the query side
/// - `pool_field_vecs`:   per-field embedding vectors for the pool (opposite) side
/// - `blocking_index`:    optional BlockingIndex for the opposite side
/// - `config`:            job config
/// - `top_n`:             max results to return (0 = unlimited)
pub fn score_pool(
    query_id: &str,
    query_record: &Record,
    query_side: Side,
    pool_records: &DashMap<String, Record>,
    query_field_vecs: &FieldVectors,
    pool_field_vecs: &FieldVectors,
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
        query_id,
        query_record,
        &candidate_ids,
        pool_records,
        query_field_vecs,
        pool_field_vecs,
        query_side,
        config,
    );

    if cands.is_empty() {
        return Vec::new();
    }

    // --- Stage 3: Full scoring ---
    let mut results: Vec<MatchResult> = cands
        .iter()
        .map(|cand| {
            // Use per-field embedding scores from the candidate.
            let emb_scores = if !cand.emb_scores.is_empty() {
                Some(&cand.emb_scores)
            } else {
                None
            };

            let score_result =
                scoring::score_pair(query_record, &cand.record, &config.match_fields, emb_scores);

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
