//! Exact scorer: case-insensitive string equality after trim.

/// Score two strings for exact equality (case-insensitive, trimmed).
///
/// - Both empty after trim → 0.0 (no evidence of a match)
/// - Case-insensitive equal → 1.0
/// - Otherwise → 0.0
pub fn score(a: &str, b: &str) -> f64 {
    let a = a.trim();
    let b = b.trim();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a.eq_ignore_ascii_case(b) {
        1.0
    } else if a.is_ascii() && b.is_ascii() {
        // The ASCII comparison is conclusive; do not allocate lowercase copies.
        0.0
    } else if a.to_lowercase() == b.to_lowercase() {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_driven() {
        let cases = vec![
            ("Foo", "foo", 1.0),
            (" Bar ", "bar", 1.0),
            ("", "", 0.0),
            ("a", "b", 0.0),
            ("GB", "gb", 1.0),
            ("  ", "  ", 0.0), // both empty after trim
            ("Café", "café", 1.0),
            ("CAFÉ", "café", 1.0),
            ("K", "k", 1.0),  // Unicode lowercase can equal an ASCII value.
            ("ß", "ss", 0.0), // Lowercasing is not full Unicode case folding.
            ("\u{2003}", "nonempty", 0.0),
            ("hello", "hello", 1.0),
            ("ABC", "abc", 1.0),
            ("abc", "def", 0.0),
            ("  hello  ", "hello", 1.0),
        ];

        for (a, b, expected) in cases {
            let result = score(a, b);
            assert!(
                (result - expected).abs() < f64::EPSILON,
                "score({:?}, {:?}) = {}, expected {}",
                a,
                b,
                result,
                expected
            );
        }
    }
}
