//! BM25 score normalisation.
//!
//! Raw BM25 scores are unbounded. We normalise by dividing by the self-score
//! (query matched against itself = theoretical max), giving a [0, 1] range.

/// Normalise a raw BM25 score to [0.0, 1.0].
///
/// `raw_score` is the BM25 score of a candidate against the query.
/// `self_score` is the BM25 score of the query against itself (theoretical max).
///
/// Returns `(raw_score / self_score).clamp(0.0, 1.0)`. If `self_score` is zero
/// or negative (empty query, no indexed tokens), returns 0.0.
pub fn normalise_bm25(raw_score: f32, self_score: f32) -> f64 {
    if self_score <= 0.0 {
        return 0.0;
    }
    ((raw_score as f64) / (self_score as f64)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_match() {
        // Same score as self → 1.0
        let score = normalise_bm25(5.0, 5.0);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn partial_match() {
        let score = normalise_bm25(2.5, 5.0);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_raw() {
        let score = normalise_bm25(0.0, 5.0);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_self_score() {
        let score = normalise_bm25(3.0, 0.0);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn negative_self_score() {
        let score = normalise_bm25(3.0, -1.0);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn clamped_above_one() {
        // Raw score exceeding self-score (rare but possible with IDF variance)
        let score = normalise_bm25(6.0, 5.0);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn negative_raw_clamped() {
        let score = normalise_bm25(-1.0, 5.0);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }
}
