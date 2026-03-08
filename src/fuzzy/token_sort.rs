//! Token sort ratio scorer.
//!
//! Splits on whitespace, sorts tokens alphabetically, rejoins with a single
//! space, then computes `ratio` on the sorted strings.

use super::ratio::ratio_normalized;

/// Compute token-sort ratio similarity (0.0..=1.0).
///
/// Both inputs are lowercased and trimmed, then split into tokens which are
/// sorted alphabetically and rejoined. The `ratio` of the sorted strings
/// is returned.
pub fn token_sort_ratio(a: &str, b: &str) -> f64 {
    let a = a.trim().to_lowercase();
    let b = b.trim().to_lowercase();
    token_sort_ratio_normalized(&a, &b)
}

/// Compute token-sort ratio on already-normalized (lowercased, trimmed) inputs.
pub(crate) fn token_sort_ratio_normalized(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let sorted_a = sort_tokens(a);
    let sorted_b = sort_tokens(b);

    ratio_normalized(&sorted_a, &sorted_b)
}

fn sort_tokens(s: &str) -> String {
    let mut tokens: Vec<&str> = s.split_whitespace().collect();
    tokens.sort();
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reordered_tokens() {
        // "John Smith" vs "Smith John" — same tokens, different order
        let r = token_sort_ratio("John Smith", "Smith John");
        assert!((r - 1.0).abs() < 0.001, "token_sort_ratio = {}", r);
    }

    #[test]
    fn identical() {
        assert!((token_sort_ratio("hello world", "hello world") - 1.0).abs() < 0.001);
    }

    #[test]
    fn both_empty() {
        assert!((token_sort_ratio("", "")).abs() < 0.001);
    }

    #[test]
    fn extra_whitespace() {
        let r = token_sort_ratio("  John   Smith  ", "Smith John");
        assert!((r - 1.0).abs() < 0.001, "token_sort_ratio = {}", r);
    }
}
