//! Fuzzy string matching scorers.
//!
//! Provides `ratio`, `partial_ratio`, `token_sort_ratio`, and `wratio`.
//! All functions: lowercase + trim inputs, return 0.0..=1.0.
//!
//! `ratio` wraps `rapidfuzz::fuzz::ratio`. The remaining three scorers are
//! implemented from scratch since `rapidfuzz-rs` 0.5 only implements `ratio`.

mod partial_ratio;
mod ratio;
mod token_sort;
mod wratio;

pub use self::partial_ratio::partial_ratio;
pub use self::ratio::ratio;
pub use self::token_sort::token_sort_ratio;
pub use self::wratio::wratio;

use crate::config::FuzzyScorer;

/// Dispatch to the configured fuzzy scorer.
pub fn score(scorer: FuzzyScorer, a: &str, b: &str) -> f64 {
    match scorer {
        FuzzyScorer::Ratio => ratio(a, b),
        FuzzyScorer::PartialRatio => partial_ratio(a, b),
        FuzzyScorer::TokenSort => token_sort_ratio(a, b),
        FuzzyScorer::Wratio => wratio(a, b),
    }
}
