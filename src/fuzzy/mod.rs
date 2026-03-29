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

/// Dispatch to a named scorer.
pub fn score(scorer: &str, a: &str, b: &str) -> f64 {
    match scorer {
        "ratio" => ratio(a, b),
        "partial_ratio" => partial_ratio(a, b),
        "token_sort" | "token_sort_ratio" => token_sort_ratio(a, b),
        "wratio" => wratio(a, b),
        unknown => {
            tracing::warn!(
                scorer = unknown,
                "unknown fuzzy scorer, falling back to wratio"
            );
            wratio(a, b)
        }
    }
}
