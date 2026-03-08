//! Ratio scorer — wrapper around `rapidfuzz::fuzz::ratio`.

use rapidfuzz::fuzz;

/// Compute normalized Levenshtein similarity (0.0..=1.0).
///
/// Both inputs are lowercased and trimmed before comparison.
pub fn ratio(a: &str, b: &str) -> f64 {
    let a = a.trim().to_lowercase();
    let b = b.trim().to_lowercase();
    ratio_normalized(&a, &b)
}

/// Compute ratio on already-normalized (lowercased, trimmed) inputs.
pub(crate) fn ratio_normalized(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    fuzz::ratio(a.chars(), b.chars())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical() {
        assert!((ratio("hello", "hello") - 1.0).abs() < 0.001);
    }

    #[test]
    fn case_insensitive() {
        assert!((ratio("Hello", "hello") - 1.0).abs() < 0.001);
    }

    #[test]
    fn both_empty() {
        assert!((ratio("", "")).abs() < 0.001);
    }

    #[test]
    fn one_empty() {
        assert!((ratio("abc", "")).abs() < 0.001);
    }

    #[test]
    fn similar() {
        let r = ratio("kitten", "sitting");
        assert!(r > 0.4 && r < 0.8, "ratio = {}", r);
    }
}
