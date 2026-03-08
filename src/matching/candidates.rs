//! Candidate generation: finding potential matches from the opposite pool.

use std::collections::{HashMap, HashSet};

use crate::config::Config;
use crate::crossmap::CrossMap;
use crate::index::VecIndex;
use crate::matching::blocking;
use crate::models::{Record, Side};

/// A candidate record from the opposite pool, with optional embedding score.
///
/// Borrows the record from the pool to avoid cloning during candidate generation.
/// Records are only cloned when needed (e.g., for output mapping on the top result).
#[derive(Debug)]
pub struct Candidate<'a> {
    pub id: String,
    pub record: &'a Record,
    pub embedding_score: Option<f64>,
}

/// Generate candidates for batch mode: B queries against A pool.
///
/// 1. If CrossMap has this B record → return empty (already matched)
/// 2. Vector search top-K from A index
/// 3. Post-filter by blocking
/// 4. Build Candidate structs
pub fn generate_candidates_batch<'a>(
    query_id: &str,
    query_record: &Record,
    query_vec: &[f32],
    index_a: &VecIndex,
    pool_records: &'a HashMap<String, Record>,
    config: &Config,
    crossmap: &CrossMap,
) -> Vec<Candidate<'a>> {
    // Skip if already matched
    if crossmap.has_b(query_id) {
        return vec![];
    }

    let k = config.candidates.n.unwrap_or(10);

    // Vector search: top-K from A index
    let search_results = index_a.search(query_vec, k * 3); // oversample to compensate for blocking filter

    // Get blocking-passing IDs
    let blocking_ids: HashSet<String> = if config.blocking.enabled {
        blocking::apply_blocking(query_record, pool_records, &config.blocking, Side::B)
            .into_iter()
            .collect()
    } else {
        // No blocking — all pass
        pool_records.keys().cloned().collect()
    };

    // Filter search results by blocking
    let mut candidates: Vec<Candidate<'a>> = search_results
        .into_iter()
        .filter(|(id, _)| blocking_ids.contains(id))
        .filter_map(|(id, score)| {
            pool_records.get(&id).map(|rec| Candidate {
                id,
                record: rec,
                embedding_score: Some(score as f64),
            })
        })
        .take(k)
        .collect();

    // If we got fewer than K candidates, supplement with fuzzy fallback
    // (records that pass blocking but weren't in vector top-K)
    if candidates.len() < k {
        let existing_ids: HashSet<&str> = candidates.iter().map(|c| c.id.as_str()).collect();

        // Score remaining by fuzzy on primary text field
        let need = k - candidates.len();
        let query_text = get_primary_text(query_record, config, Side::B);
        let mut scored: Vec<(&str, &'a Record, f64)> = blocking_ids
            .iter()
            .filter(|id| !existing_ids.contains(id.as_str()))
            .filter_map(|id| pool_records.get(id).map(|rec| (id.as_str(), rec)))
            .map(|(id, rec)| {
                let pool_text = get_primary_text(rec, config, Side::A);
                let score = crate::fuzzy::wratio(&query_text, &pool_text);
                (id, rec, score)
            })
            .collect();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(need);

        for (id, rec, _) in scored {
            candidates.push(Candidate {
                id: id.to_string(),
                record: rec,
                embedding_score: None,
            });
        }
    }

    candidates
}

/// Generate candidates for batch mode using a pre-built BlockingIndex for O(1) lookups.
///
/// Same logic as `generate_candidates_batch` but uses the indexed blocking
/// instead of linear-scanning all A records.
pub fn generate_candidates_batch_indexed<'a>(
    query_id: &str,
    query_record: &Record,
    query_vec: &[f32],
    index_a: &VecIndex,
    pool_records: &'a HashMap<String, Record>,
    config: &Config,
    crossmap: &CrossMap,
    blocking_index: Option<&blocking::BlockingIndex>,
) -> Vec<Candidate<'a>> {
    // Skip if already matched
    if crossmap.has_b(query_id) {
        return vec![];
    }

    let k = config.candidates.n.unwrap_or(10);

    // Vector search: top-K from A index
    let search_results = index_a.search(query_vec, k * 3);

    // Get blocking-passing IDs via index (O(1)) or no-filter
    let blocking_ids: HashSet<String> = if config.blocking.enabled {
        if let Some(bi) = blocking_index {
            bi.query(query_record, Side::B)
        } else {
            blocking::apply_blocking(query_record, pool_records, &config.blocking, Side::B)
                .into_iter()
                .collect()
        }
    } else {
        pool_records.keys().cloned().collect()
    };

    // Filter search results by blocking
    let mut candidates: Vec<Candidate<'a>> = search_results
        .into_iter()
        .filter(|(id, _)| blocking_ids.contains(id))
        .filter_map(|(id, score)| {
            pool_records.get(&id).map(|rec| Candidate {
                id,
                record: rec,
                embedding_score: Some(score as f64),
            })
        })
        .take(k)
        .collect();

    // If we got fewer than K candidates, supplement with fuzzy fallback
    if candidates.len() < k {
        let existing_ids: HashSet<&str> = candidates.iter().map(|c| c.id.as_str()).collect();
        let need = k - candidates.len();
        let query_text = get_primary_text(query_record, config, Side::B);
        let mut scored: Vec<(&str, &'a Record, f64)> = blocking_ids
            .iter()
            .filter(|id| !existing_ids.contains(id.as_str()))
            .filter_map(|id| pool_records.get(id).map(|rec| (id.as_str(), rec)))
            .map(|(id, rec)| {
                let pool_text = get_primary_text(rec, config, Side::A);
                let score = crate::fuzzy::wratio(&query_text, &pool_text);
                (id, rec, score)
            })
            .collect();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(need);

        for (id, rec, _) in scored {
            candidates.push(Candidate {
                id: id.to_string(),
                record: rec,
                embedding_score: None,
            });
        }
    }

    candidates
}

/// Generate candidates for live mode: either side queries the opposite pool.
///
/// Uses search_filtered with the intersection of unmatched and blocking sets.
pub fn generate_candidates_live<'a>(
    query_record: &Record,
    query_vec: &[f32],
    target_index: &VecIndex,
    target_records: &'a HashMap<String, Record>,
    blocking_index: Option<&crate::matching::blocking::BlockingIndex>,
    unmatched: &HashSet<String>,
    query_side: Side,
    config: &Config,
) -> Vec<Candidate<'a>> {
    let k = config.live.top_n.unwrap_or(5);

    // Build allowed_ids = unmatched ∩ blocking_index.query()
    let allowed: HashSet<String> = if let Some(bi) = blocking_index {
        let blocking_ids = bi.query(query_record, query_side);
        unmatched.intersection(&blocking_ids).cloned().collect()
    } else {
        unmatched.clone()
    };

    if allowed.is_empty() {
        return vec![];
    }

    // Filtered vector search
    let search_results = target_index.search_filtered(query_vec, k, &allowed);

    let mut candidates: Vec<Candidate<'a>> = search_results
        .into_iter()
        .filter_map(|(id, score)| {
            target_records.get(&id).map(|rec| Candidate {
                id,
                record: rec,
                embedding_score: Some(score as f64),
            })
        })
        .collect();

    // Fuzzy fallback if fewer than K results
    if candidates.len() < k {
        let existing_ids: HashSet<&str> = candidates.iter().map(|c| c.id.as_str()).collect();
        let need = k - candidates.len();
        let query_text = get_primary_text(query_record, config, query_side);

        let mut extras: Vec<(&str, &'a Record, f64)> = allowed
            .iter()
            .filter(|id| !existing_ids.contains(id.as_str()))
            .filter_map(|id| {
                target_records.get(id).map(|rec| {
                    let opposite_side = query_side.opposite();
                    let pool_text = get_primary_text(rec, config, opposite_side);
                    let score = crate::fuzzy::wratio(&query_text, &pool_text);
                    (id.as_str(), rec, score)
                })
            })
            .collect();

        extras.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        extras.truncate(need);

        for (id, rec, _) in extras {
            candidates.push(Candidate {
                id: id.to_string(),
                record: rec,
                embedding_score: None,
            });
        }
    }

    candidates
}

/// Get the primary text for a record (for fuzzy fallback scoring).
///
/// Uses the first fuzzy or embedding field value for the given side.
fn get_primary_text(record: &Record, config: &Config, side: Side) -> String {
    for mf in &config.match_fields {
        let field = match side {
            Side::A => &mf.field_a,
            Side::B => &mf.field_b,
        };
        if mf.method == "embedding" || mf.method == "fuzzy" {
            if let Some(val) = record.get(field) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    String::new()
}
