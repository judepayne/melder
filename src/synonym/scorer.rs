//! Binary synonym scorer.
//!
//! Returns 1.0 if the two values have a synonym relationship (acronym match
//! or dictionary equivalence), 0.0 otherwise.

use super::dictionary::SynonymDictionary;
use super::generator::generate_acronyms;

/// Default minimum acronym length used by the scorer.
const DEFAULT_MIN_LENGTH: usize = 3;

/// Score a synonym relationship between two field values.
///
/// Checks three things:
/// 1. Is `a_val` an acronym of `b_val`?
/// 2. Is `b_val` an acronym of `a_val`?
/// 3. Are `a_val` and `b_val` equivalent in the dictionary?
///
/// Returns 1.0 if any check matches, 0.0 otherwise.
pub fn score(
    a_val: &str,
    b_val: &str,
    min_length: usize,
    dictionary: Option<&SynonymDictionary>,
) -> f64 {
    let a_trimmed = a_val.trim();
    let b_trimmed = b_val.trim();

    if a_trimmed.is_empty() || b_trimmed.is_empty() {
        return 0.0;
    }

    let a_upper = a_trimmed.to_uppercase();
    let b_upper = b_trimmed.to_uppercase();

    // Check: is a_val an acronym of b_val?
    let b_acronyms = generate_acronyms(b_trimmed, min_length);
    if b_acronyms.contains(&a_upper) {
        return 1.0;
    }

    // Check: is b_val an acronym of a_val?
    let a_acronyms = generate_acronyms(a_trimmed, min_length);
    if a_acronyms.contains(&b_upper) {
        return 1.0;
    }

    // Check: dictionary equivalence.
    if let Some(dict) = dictionary {
        if dict.is_equivalent(a_trimmed, b_trimmed) {
            return 1.0;
        }
    }

    0.0
}

/// Score with default min_length and no dictionary.
pub fn score_default(a_val: &str, b_val: &str) -> f64 {
    score(a_val, b_val, DEFAULT_MIN_LENGTH, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acronym_match_forward() {
        assert_eq!(
            score("HWAG", "Harris, Watkins and Goodwin BV", 3, None),
            1.0,
            "HWAG is an acronym of the full name"
        );
    }

    #[test]
    fn acronym_match_reverse() {
        assert_eq!(
            score("Harris, Watkins and Goodwin BV", "HWAG", 3, None),
            1.0,
            "reverse direction should also match"
        );
    }

    #[test]
    fn no_match() {
        assert_eq!(
            score("Completely Different Corp", "HWAG", 3, None),
            0.0,
            "unrelated strings → 0.0"
        );
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(
            score("hwag", "Harris, Watkins and Goodwin BV", 3, None),
            1.0,
            "case should not matter"
        );
    }

    #[test]
    fn empty_values() {
        assert_eq!(score("", "something", 3, None), 0.0);
        assert_eq!(score("something", "", 3, None), 0.0);
        assert_eq!(score("", "", 3, None), 0.0);
    }

    #[test]
    fn trms_example() {
        assert_eq!(
            score("TRMS", "Taylor, Reeves and Mcdaniel SRL", 3, None),
            1.0,
            "names + suffixes variant"
        );
    }

    #[test]
    fn single_word_no_match() {
        assert_eq!(
            score("KPMG", "KPMG", 3, None),
            0.0,
            "identical single words are not an acronym relationship"
        );
    }

    #[test]
    fn gsam() {
        assert_eq!(score("GSAM", "Goldman Sachs Asset Management", 3, None), 1.0,);
    }
}
