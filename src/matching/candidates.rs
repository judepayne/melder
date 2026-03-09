//! Candidate generation: narrowing blocked records to the top-N before full scoring.
//!
//! The pipeline is: blocking filter → candidate selection → full scoring.
//! This module implements the candidate selection stage.

use dashmap::DashMap;

use crate::config::Config;
use crate::index::VecIndex;
use crate::models::{Record, Side};

/// A candidate record from the opposite pool, with optional embedding score.
///
/// Owns a clone of the pool record. Since we only keep N candidates
/// (default 10), the cloning cost is negligible.
#[derive(Debug)]
pub struct Candidate {
    pub id: String,
    pub record: Record,
    pub embedding_score: Option<f64>,
}

/// Select top-N candidates from blocked records using the configured
/// candidate scoring method.
///
/// If candidates is disabled (`enabled: false`), returns all blocked records
/// with embedding scores looked up from the indices.
///
/// If candidates is enabled, scores each blocked record on the configured
/// `field_a`/`field_b` using the configured `method` (fuzzy, embedding, exact),
/// sorts descending, and returns the top `n`.
///
/// Each returned candidate carries a precomputed embedding score (if both
/// vectors are available in the indices) for use in the full scoring stage.
pub fn select_candidates(
    query_record: &Record,
    query_vec: &[f32],
    candidate_ids: &[String],
    pool_records: &DashMap<String, Record>,
    pool_index: &VecIndex,
    query_side: Side,
    config: &Config,
) -> Vec<Candidate> {
    let has_config = config.candidates.field_a.is_some()
        || config.candidates.field_b.is_some()
        || config.candidates.method.is_some();
    let candidates_enabled = config.candidates.enabled.unwrap_or(has_config);

    if !candidates_enabled {
        // No candidate filtering — pass all blocked records through.
        // Look up embedding scores for each if vectors are available.
        return candidate_ids
            .iter()
            .filter_map(|cid| {
                pool_records.get(cid).map(|entry| {
                    let emb_score = pool_index.get(cid).map(|a_vec| {
                        crate::scoring::embedding::cosine_similarity(query_vec, a_vec)
                    });
                    Candidate {
                        id: cid.clone(),
                        record: entry.value().clone(),
                        embedding_score: emb_score,
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

    // Score each candidate on the configured field pair.
    let mut scored: Vec<(String, Record, f64)> = candidate_ids
        .iter()
        .filter_map(|cid| {
            pool_records.get(cid).map(|entry| {
                let rec = entry.value();
                let score = match method {
                    "embedding" => {
                        // Use precomputed vectors from the index.
                        pool_index
                            .get(cid)
                            .map(|a_vec| {
                                crate::scoring::embedding::cosine_similarity(query_vec, a_vec)
                            })
                            .unwrap_or(0.0)
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

    // Build Candidate structs with embedding scores for full scoring.
    scored
        .into_iter()
        .map(|(cid, rec, _)| {
            let emb_score = pool_index
                .get(&cid)
                .map(|a_vec| crate::scoring::embedding::cosine_similarity(query_vec, a_vec));
            Candidate {
                id: cid,
                record: rec,
                embedding_score: emb_score,
            }
        })
        .collect()
}
