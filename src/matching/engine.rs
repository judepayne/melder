//! Match engine: score candidates and produce MatchResults.

use std::collections::HashMap;

use crate::config::Config;
use crate::models::{MatchResult, Record, Side};
use crate::scoring;

use super::candidates::Candidate;

/// Score a list of candidates against a query record, returning sorted MatchResults.
///
/// For each candidate:
/// 1. Compute precomputed embedding scores (if vectors available)
/// 2. Score via composite scorer
/// 3. Classify
/// 4. Sort by score descending
/// 5. Apply output_mapping only to the top result (lazy cloning)
pub fn score_candidates(
    query_id: &str,
    query_record: &Record,
    _query_vec: Option<&[f32]>,
    query_side: Side,
    candidates: &[Candidate<'_>],
    config: &Config,
) -> Vec<MatchResult> {
    // Pre-compute embedding score key/value once (same for all embedding fields)
    let emb_keys: Vec<String> = config
        .match_fields
        .iter()
        .filter(|mf| mf.method == "embedding")
        .map(|mf| format!("{}/{}", mf.field_a, mf.field_b))
        .collect();

    let mut results: Vec<MatchResult> = candidates
        .iter()
        .map(|candidate| {
            // Build precomputed embedding scores
            let emb_scores = if let Some(score) = candidate.embedding_score {
                let mut map = HashMap::with_capacity(emb_keys.len());
                for key in &emb_keys {
                    map.insert(key.clone(), score);
                }
                map
            } else {
                HashMap::new()
            };

            let emb_ref = if emb_scores.is_empty() {
                None
            } else {
                Some(&emb_scores)
            };

            // Score the pair (borrows candidate.record, no clone)
            let score_result = scoring::score_pair(
                query_record,
                candidate.record,
                &config.match_fields,
                emb_ref,
            );

            // Defer output mapping — we'll only apply it to the top result
            scoring::build_match_result(
                query_id,
                &candidate.id,
                query_side,
                score_result,
                config.thresholds.auto_match,
                config.thresholds.review_floor,
                None, // no matched_record yet — applied lazily below
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

    // Apply output_mapping only to the top result
    if let Some(top) = results.first_mut() {
        // Find the candidate record for the top result
        if let Some(cand) = candidates.iter().find(|c| c.id == top.matched_id) {
            top.matched_record = Some(apply_output_mapping(cand.record, config));
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
