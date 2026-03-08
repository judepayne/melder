//! WRatio scorer.
//!
//! Tries `ratio`, `token_sort_ratio`, and `partial_ratio` and returns the
//! best result. This approximates Python `rapidfuzz.fuzz.WRatio` semantics.

use super::partial_ratio::partial_ratio_normalized;
use super::ratio::ratio_normalized;
use super::token_sort::token_sort_ratio_normalized;

/// Compute WRatio similarity (0.0..=1.0).
///
/// Takes the maximum of `ratio`, `token_sort_ratio`, and `partial_ratio`.
/// This matches the intent of Python's `fuzz.WRatio` which tries multiple
/// strategies and returns the best score.
///
/// Normalizes (lowercase + trim) once and passes to all sub-scorers.
pub fn wratio(a: &str, b: &str) -> f64 {
    let a_norm = a.trim().to_lowercase();
    let b_norm = b.trim().to_lowercase();

    if a_norm.is_empty() && b_norm.is_empty() {
        return 0.0;
    }

    let r = ratio_normalized(&a_norm, &b_norm);
    if (r - 1.0).abs() < f64::EPSILON {
        return 1.0;
    }

    let ts = token_sort_ratio_normalized(&a_norm, &b_norm);
    let pr = partial_ratio_normalized(&a_norm, &b_norm);

    r.max(ts).max(pr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical() {
        assert!((wratio("hello", "hello") - 1.0).abs() < 0.001);
    }

    #[test]
    fn reordered() {
        // Token reordering should be caught by token_sort_ratio
        let r = wratio("John Smith", "Smith John");
        assert!((r - 1.0).abs() < 0.001, "wratio = {}", r);
    }

    #[test]
    fn substring() {
        // partial_ratio should catch substring matches
        let r = wratio("bar", "foobar");
        assert!(r > 0.8, "wratio = {}", r);
    }

    #[test]
    fn both_empty() {
        assert!((wratio("", "")).abs() < 0.001);
    }

    #[test]
    fn completely_different() {
        let r = wratio("abc", "xyz");
        assert!(r < 0.5, "wratio = {}", r);
    }
}
