//! Candidate generation: narrowing blocked records to the top-N before full scoring.
//!
//! The pipeline is: blocking filter → candidate selection → full scoring.
//! This module implements the candidate selection stage.

use std::collections::HashMap;

use dashmap::DashMap;

use crate::config::Config;
use crate::models::{Record, Side};
use crate::vectordb::field_indexes::FieldIndexes;

/// A candidate record from the opposite pool, with per-field embedding scores.
///
/// Owns a clone of the pool record. Since we only keep N candidates
/// (default 10), the cloning cost is negligible.
#[derive(Debug)]
pub struct Candidate {
    pub id: String,
    pub record: Record,
    /// Per-field embedding cosine similarities, keyed by "field_a/field_b".
    /// Populated for all embedding match fields where both query and candidate
    /// vectors are available.
    pub emb_scores: HashMap<String, f64>,
}

/// Select top-N candidates from blocked records using the configured
/// candidate scoring method.
///
/// If candidates is disabled (`enabled: false`), returns all blocked records
/// with per-field embedding scores computed from the field vectors.
///
/// If candidates is enabled, scores each blocked record on the configured
/// `field_a`/`field_b` using the configured `method` (fuzzy, embedding, exact),
/// sorts descending, and returns the top `n`.
///
/// Each returned candidate carries precomputed per-field embedding scores
/// for use in the full scoring stage.
pub fn select_candidates(
    query_id: &str,
    query_record: &Record,
    candidate_ids: &[String],
    pool_records: &DashMap<String, Record>,
    query_field_indexes: &FieldIndexes,
    pool_field_indexes: &FieldIndexes,
    query_side: Side,
    config: &Config,
) -> Vec<Candidate> {
    // Collect embedding field keys for per-field scoring.
    let emb_keys: Vec<(String, String, String)> = crate::vectordb::embedding_field_keys(config);

    let has_config = config.candidates.field_a.is_some()
        || config.candidates.field_b.is_some()
        || config.candidates.method.is_some();
    let candidates_enabled = config.candidates.enabled.unwrap_or(has_config);

    if !candidates_enabled {
        // No candidate filtering — pass all blocked records through.
        // Compute per-field embedding scores for each.
        return candidate_ids
            .iter()
            .filter_map(|cid| {
                pool_records.get(cid).map(|entry| {
                    let emb_scores = compute_emb_scores(
                        query_id,
                        cid,
                        &emb_keys,
                        query_field_indexes,
                        pool_field_indexes,
                    );
                    Candidate {
                        id: cid.clone(),
                        record: entry.value().clone(),
                        emb_scores,
                    }
                })
            })
            .collect();
    }

    // Candidates enabled — score on the configured field pair.
    let n = config.candidates.n.unwrap_or(10);
    let method = config.candidates.method.as_deref().unwrap_or("fuzzy");
    let scorer = config.candidates.scorer.as_deref().unwrap_or("wratio");

    // Determine which field names to use for candidate scoring.
    // field_a is on the A-side record, field_b is on the B-side record.
    let field_a = config.candidates.field_a.as_deref().unwrap_or("");
    let field_b = config.candidates.field_b.as_deref().unwrap_or("");

    // Get the query-side field value.
    let query_field = match query_side {
        Side::A => field_a,
        Side::B => field_b,
    };
    let query_val = query_record
        .get(query_field)
        .map(|s| s.as_str())
        .unwrap_or("");

    // Opposite-side field name.
    let pool_field = match query_side {
        Side::A => field_b,
        Side::B => field_a,
    };

    // Build the field_key for embedding candidate scoring if applicable.
    let cand_emb_field_key = format!("{}/{}", field_a, field_b);

    // Score each candidate on the configured field pair.
    let mut scored: Vec<(String, Record, f64)> = candidate_ids
        .iter()
        .filter_map(|cid| {
            pool_records.get(cid).map(|entry| {
                let rec = entry.value();
                let score = match method {
                    "embedding" => {
                        // Use per-field indexes for embedding candidate scoring.
                        let q_vec = query_field_indexes.get_vec(query_id, &cand_emb_field_key);
                        let c_vec = pool_field_indexes.get_vec(cid, &cand_emb_field_key);
                        match (q_vec, c_vec) {
                            (Some(qv), Some(cv)) => {
                                crate::scoring::embedding::cosine_similarity(&qv, &cv)
                            }
                            _ => 0.0,
                        }
                    }
                    "fuzzy" => {
                        let pool_val = rec.get(pool_field).map(|s| s.as_str()).unwrap_or("");
                        crate::fuzzy::score(scorer, query_val, pool_val)
                    }
                    "exact" => {
                        let pool_val = rec.get(pool_field).map(|s| s.as_str()).unwrap_or("");
                        crate::scoring::exact::score(query_val, pool_val)
                    }
                    _ => 0.0,
                };
                (cid.clone(), rec.clone(), score)
            })
        })
        .collect();

    // Sort by score descending, take top N.
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);

    // Build Candidate structs with per-field embedding scores for full scoring.
    scored
        .into_iter()
        .map(|(cid, rec, _)| {
            let emb_scores = compute_emb_scores(
                query_id,
                &cid,
                &emb_keys,
                query_field_indexes,
                pool_field_indexes,
            );
            Candidate {
                id: cid,
                record: rec,
                emb_scores,
            }
        })
        .collect()
}

/// Compute per-field embedding cosine similarities between query and candidate.
fn compute_emb_scores(
    query_id: &str,
    cand_id: &str,
    emb_keys: &[(String, String, String)],
    query_fi: &FieldIndexes,
    pool_fi: &FieldIndexes,
) -> HashMap<String, f64> {
    let mut scores = HashMap::with_capacity(emb_keys.len());
    for (field_key, _field_a, _field_b) in emb_keys {
        let q_vec = query_fi.get_vec(query_id, field_key);
        let c_vec = pool_fi.get_vec(cand_id, field_key);
        if let (Some(qv), Some(cv)) = (q_vec, c_vec) {
            let sim = crate::scoring::embedding::cosine_similarity(&qv, &cv);
            scores.insert(field_key.clone(), sim);
        }
    }
    scores
}
