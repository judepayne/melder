//! Shared search utilities for vector backends.
//!
//! Extracts common ranking logic used by both flat and usearch backends:
//! NaN filtering, top-K partial sort, and descending score sort.

use std::cmp::Ordering;

use super::SearchResult;

/// Compare two f32 scores in descending order, treating NaN as less than
/// any real value (sorts NaN to the end).
#[inline]
pub fn cmp_score_desc(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

/// Sort `SearchResult`s by score descending and truncate to `k`.
pub fn sort_and_truncate(results: &mut Vec<SearchResult>, k: usize) {
    results.sort_by(|a, b| cmp_score_desc(a.score, b.score));
    results.truncate(k);
}

/// Select the top-K from indexed scores: filter NaN, partial-sort, final sort.
///
/// Used by the flat backend where scores are `(index, dot_product)` pairs.
/// Returns `(index, score)` pairs sorted descending by score, truncated to `k`.
pub fn top_k_by_score(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    // Filter out NaN scores (corrupted vectors) before ranking.
    scores.retain(|(_, s)| !s.is_nan());

    if scores.is_empty() || k == 0 {
        return vec![];
    }

    let k = k.min(scores.len());
    scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| cmp_score_desc(a.1, b.1));
    scores.truncate(k);
    scores.sort_by(|a, b| cmp_score_desc(a.1, b.1));
    scores
}
