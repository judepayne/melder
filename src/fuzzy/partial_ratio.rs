//! Partial ratio scorer.
//!
//! Finds the best-aligned window of the shorter string within the longer
//! string and computes `ratio` on that window. This matches the Python
//! `rapidfuzz.fuzz.partial_ratio` semantics.

use super::ratio::ratio_normalized;

/// Compute partial ratio similarity (0.0..=1.0).
///
/// Slides the shorter (lowercased, trimmed) string across the longer one,
/// computing `ratio` at each window position, and returns the maximum.
pub fn partial_ratio(a: &str, b: &str) -> f64 {
    let a = a.trim().to_lowercase();
    let b = b.trim().to_lowercase();
    partial_ratio_normalized(&a, &b)
}

/// Compute partial ratio on already-normalized (lowercased, trimmed) inputs.
pub(crate) fn partial_ratio_normalized(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    let (shorter, longer) = if a.chars().count() <= b.chars().count() {
        (a, b)
    } else {
        (b, a)
    };

    let short_len = shorter.chars().count();
    if short_len == 0 {
        return 0.0;
    }

    // Check if shorter is an exact substring of longer
    if longer.contains(shorter) {
        return 1.0;
    }

    // Slide a window of `short_len` chars across `longer`.
    // Pre-compute char→byte offset map to avoid per-window String allocation.
    let char_offsets: Vec<usize> = longer.char_indices().map(|(i, _)| i).collect();
    let long_len = char_offsets.len();
    let mut best = 0.0_f64;

    let windows = long_len - short_len + 1;
    for i in 0..windows {
        let byte_start = char_offsets[i];
        let byte_end = if i + short_len < long_len {
            char_offsets[i + short_len]
        } else {
            longer.len()
        };
        let window = &longer[byte_start..byte_end];
        let r = ratio_normalized(shorter, window);
        if r > best {
            best = r;
        }
        if (best - 1.0).abs() < f64::EPSILON {
            break;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substring_match() {
        // "bar" is a substring of "foobar" → should score very high
        let r = partial_ratio("bar", "foobar");
        assert!(r > 0.95, "partial_ratio = {}", r);
    }

    #[test]
    fn identical() {
        assert!((partial_ratio("hello", "hello") - 1.0).abs() < 0.001);
    }

    #[test]
    fn both_empty() {
        assert!((partial_ratio("", "")).abs() < 0.001);
    }

    #[test]
    fn one_empty() {
        assert!((partial_ratio("abc", "")).abs() < 0.001);
    }

    #[test]
    fn no_overlap() {
        let r = partial_ratio("abc", "xyz");
        assert!(r < 0.5, "partial_ratio = {}", r);
    }
}
