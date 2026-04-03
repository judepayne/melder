pub mod embedding;
pub mod exact;

use crate::config::{FuzzyScorer, MatchField, MatchMethod};
use crate::fuzzy;
use crate::models::{Classification, FieldScore, MatchResult, Record, Side};

/// Result of scoring a single record pair across all match fields.
#[derive(Debug, Clone)]
pub struct ScoreResult {
    pub field_scores: Vec<FieldScore>,
    pub total: f64,
}

/// Score a query record against a candidate record across all configured
/// match fields.
///
/// - For `exact` fields: case-insensitive string equality
/// - For `fuzzy` fields: dispatches to the configured scorer
/// - For `embedding` fields: uses pre-computed cosine similarity if provided,
///   otherwise scores 0.0 (embedding scorer injected at a higher level)
/// - For `numeric` fields: 1.0 if both values parse as equal f64, else 0.0
/// - For `bm25` fields: uses pre-computed normalised BM25 score if provided
///
/// `precomputed_emb_scores`: optional per-field embedding cosine similarities
/// keyed by "field_a/field_b". When present, used instead of computing.
///
/// `precomputed_bm25_score`: optional normalised BM25 score [0.0, 1.0],
/// computed by the pipeline before full scoring.
pub fn score_pair(
    query_record: &Record,
    candidate_record: &Record,
    match_fields: &[MatchField],
    precomputed_emb_scores: Option<&std::collections::HashMap<(&str, &str), f64>>,
    precomputed_bm25_score: Option<f64>,
    synonym_dictionary: Option<&crate::synonym::dictionary::SynonymDictionary>,
    pool_side: Side,
) -> ScoreResult {
    let mut field_scores = Vec::with_capacity(match_fields.len());
    let mut base_weighted_sum = 0.0_f64; // non-synonym methods
    let mut synonym_bonus = 0.0_f64; // additive synonym contribution
    let mut total_weight = 0.0_f64; // denominator (excludes synonym)

    for mf in match_fields {
        let score = if mf.method == MatchMethod::Bm25 {
            // BM25 score is precomputed by the pipeline, like embedding scores.
            // No per-field record values — BM25 operates across all text fields.
            precomputed_bm25_score.unwrap_or(0.0)
        } else {
            // field_a always refers to the A-side column, field_b to the B-side.
            // Pick the correct field based on which side the candidate (pool) is on.
            let (cand_field, query_field) = match pool_side {
                Side::A => (&mf.field_a, &mf.field_b),
                Side::B => (&mf.field_b, &mf.field_a),
            };
            let a_val = candidate_record
                .get(cand_field)
                .map(|s| s.as_str())
                .unwrap_or("");
            let b_val = query_record
                .get(query_field)
                .map(|s| s.as_str())
                .unwrap_or("");

            match mf.method {
                MatchMethod::Exact => exact::score(a_val, b_val),
                MatchMethod::Fuzzy => {
                    let scorer = mf.scorer.unwrap_or(FuzzyScorer::Wratio);
                    fuzzy::score(scorer, a_val, b_val)
                }
                MatchMethod::Embedding => {
                    // Tuple key avoids a format!() allocation per candidate.
                    precomputed_emb_scores
                        .and_then(|m| m.get(&(mf.field_a.as_str(), mf.field_b.as_str())))
                        .copied()
                        .unwrap_or(0.0)
                }
                MatchMethod::Numeric => numeric_score(a_val, b_val),
                MatchMethod::Synonym => {
                    crate::synonym::scorer::score(a_val, b_val, 3, synonym_dictionary)
                }
                MatchMethod::Bm25 => unreachable!("handled above"),
            }
        };

        let score = if score.is_nan() { 0.0 } else { score };

        let fs = FieldScore {
            field_a: if mf.method == MatchMethod::Bm25 {
                "bm25".to_string()
            } else {
                mf.field_a.clone()
            },
            field_b: if mf.method == MatchMethod::Bm25 {
                "bm25".to_string()
            } else {
                mf.field_b.clone()
            },
            method: mf.method.as_str().to_string(),
            score,
            weight: mf.weight,
        };

        // Synonym is a flat additive bonus: +weight when it fires (score=1.0),
        // +0.0 when it doesn't. Its contribution is never diluted by weight
        // normalization — only the base (non-synonym) methods normalize among
        // themselves.
        if mf.method == MatchMethod::Synonym {
            synonym_bonus += score * mf.weight;
        } else {
            base_weighted_sum += score * mf.weight;
            total_weight += mf.weight;
        }
        field_scores.push(fs);
    }

    // Normalize the base score if non-synonym weights don't sum to 1.0,
    // then add the synonym bonus on top, and clamp to [0, 1].
    let base = if total_weight > 0.0 && (total_weight - 1.0).abs() > 0.001 {
        base_weighted_sum / total_weight
    } else {
        base_weighted_sum
    };
    let total = if (base + synonym_bonus).is_nan() {
        0.0
    } else {
        base + synonym_bonus
    }
    .clamp(0.0, 1.0);

    ScoreResult {
        field_scores,
        total,
    }
}

/// Build a full `MatchResult` from a score result.
#[allow(clippy::too_many_arguments)]
pub fn build_match_result(
    query_id: &str,
    matched_id: &str,
    query_side: Side,
    score_result: ScoreResult,
    auto_match: f64,
    review_floor: f64,
    matched_record: Option<Record>,
    from_crossmap: bool,
) -> MatchResult {
    let classification = Classification::from_score(score_result.total, auto_match, review_floor);
    MatchResult {
        query_id: query_id.to_string(),
        matched_id: matched_id.to_string(),
        query_side,
        score: score_result.total,
        field_scores: score_result.field_scores,
        classification,
        matched_record,
        from_crossmap,
    }
}

/// Numeric scorer: returns 0.0 or 1.0 based on numeric equality.
///
/// Tries to parse both values as f64. If both parse and are equal
/// (within relative tolerance of 1e-9), returns 1.0. Otherwise returns 0.0.
fn numeric_score(a: &str, b: &str) -> f64 {
    let a = a.trim();
    let b = b.trim();
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(va), Ok(vb)) => {
            // Use relative tolerance for large values, absolute for small.
            // f64::EPSILON (~2.2e-16) is effectively bit-equality and far
            // too tight for values like 1_000_000.0.
            let max_abs = va.abs().max(vb.abs());
            let tol = if max_abs > 1.0 { max_abs * 1e-9 } else { 1e-9 };
            if (va - vb).abs() < tol { 1.0 } else { 0.0 }
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_fields() -> Vec<MatchField> {
        vec![
            MatchField {
                field_a: "country_code".into(),
                field_b: "domicile".into(),
                method: MatchMethod::Exact,
                scorer: None,
                weight: 0.20,
                fields: None,
            },
            MatchField {
                field_a: "short_name".into(),
                field_b: "counterparty_name".into(),
                method: MatchMethod::Fuzzy,
                scorer: Some(FuzzyScorer::PartialRatio),
                weight: 0.20,
                fields: None,
            },
            MatchField {
                field_a: "lei".into(),
                field_b: "lei_code".into(),
                method: MatchMethod::Exact,
                scorer: None,
                weight: 0.05,
                fields: None,
            },
        ]
    }

    #[test]
    fn score_exact_match() {
        let a = make_record(&[
            ("country_code", "GB"),
            ("short_name", "Acme Corp"),
            ("lei", "ABC123"),
        ]);
        let b = make_record(&[
            ("domicile", "gb"),
            ("counterparty_name", "Acme Corp"),
            ("lei_code", "abc123"),
        ]);
        let fields = make_fields();
        let result = score_pair(&b, &a, &fields, None, None, None, Side::A);

        // country_code: exact match → 1.0 * 0.20 = 0.20
        // short_name/counterparty_name: fuzzy partial_ratio identical → ~1.0 * 0.20 = 0.20
        // lei: exact match → 1.0 * 0.05 = 0.05
        // Total: 0.45 (but weights only sum to 0.45, so normalized: 1.0)
        assert!(
            result.total > 0.95,
            "expected near 1.0, got {}",
            result.total
        );
        assert_eq!(result.field_scores.len(), 3);
    }

    #[test]
    fn score_exact_match_pool_side_b() {
        // Same data as score_exact_match, but now A is the query and B is the pool.
        // field_a columns are on the query (A), field_b columns are on the candidate (B).
        let a = make_record(&[
            ("country_code", "GB"),
            ("short_name", "Acme Corp"),
            ("lei", "ABC123"),
        ]);
        let b = make_record(&[
            ("domicile", "gb"),
            ("counterparty_name", "Acme Corp"),
            ("lei_code", "abc123"),
        ]);
        let fields = make_fields();
        // query=A, candidate=B, pool_side=Side::B
        let result = score_pair(&a, &b, &fields, None, None, None, Side::B);

        assert!(
            result.total > 0.95,
            "expected near 1.0 with Side::B pool, got {}",
            result.total
        );
        assert_eq!(result.field_scores.len(), 3);
    }

    #[test]
    fn score_no_match() {
        let a = make_record(&[
            ("country_code", "US"),
            ("short_name", "Alpha Corp"),
            ("lei", "XYZ999"),
        ]);
        let b = make_record(&[
            ("domicile", "GB"),
            ("counterparty_name", "Beta Inc"),
            ("lei_code", "ABC123"),
        ]);
        let fields = make_fields();
        let result = score_pair(&b, &a, &fields, None, None, None, Side::A);

        // country_code: "US" vs "GB" → 0.0
        // short_name: "Alpha Corp" vs "Beta Inc" → low
        // lei: "XYZ999" vs "ABC123" → 0.0
        assert!(
            result.total < 0.3,
            "expected low score, got {}",
            result.total
        );
    }

    #[test]
    fn build_match_result_auto() {
        let sr = ScoreResult {
            field_scores: vec![],
            total: 0.9,
        };
        let mr = build_match_result("q1", "m1", Side::B, sr, 0.85, 0.60, None, false);
        assert_eq!(mr.classification, Classification::Auto);
        assert_eq!(mr.query_id, "q1");
        assert_eq!(mr.matched_id, "m1");
    }

    #[test]
    fn build_match_result_review() {
        let sr = ScoreResult {
            field_scores: vec![],
            total: 0.7,
        };
        let mr = build_match_result("q1", "m1", Side::B, sr, 0.85, 0.60, None, false);
        assert_eq!(mr.classification, Classification::Review);
    }

    #[test]
    fn build_match_result_no_match() {
        let sr = ScoreResult {
            field_scores: vec![],
            total: 0.3,
        };
        let mr = build_match_result("q1", "m1", Side::B, sr, 0.85, 0.60, None, false);
        assert_eq!(mr.classification, Classification::NoMatch);
    }

    #[test]
    fn numeric_score_equal() {
        assert!((numeric_score("42", "42") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn numeric_score_not_equal() {
        assert!((numeric_score("42", "43")).abs() < f64::EPSILON);
    }

    #[test]
    fn numeric_score_non_numeric() {
        assert!((numeric_score("foo", "bar")).abs() < f64::EPSILON);
    }

    #[test]
    fn numeric_score_both_empty() {
        assert!((numeric_score("", "")).abs() < f64::EPSILON);
    }

    #[test]
    fn precomputed_embedding_score() {
        let a = make_record(&[("legal_name", "Acme Corp")]);
        let b = make_record(&[("counterparty_name", "Acme Corporation")]);
        let fields = vec![MatchField {
            field_a: "legal_name".into(),
            field_b: "counterparty_name".into(),
            method: MatchMethod::Embedding,
            scorer: None,
            weight: 1.0,
            fields: None,
        }];
        let mut emb = HashMap::new();
        emb.insert(("legal_name", "counterparty_name"), 0.92);

        let result = score_pair(&b, &a, &fields, Some(&emb), None, None, Side::A);
        assert!((result.total - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn precomputed_bm25_score() {
        let a = make_record(&[("country_code", "GB")]);
        let b = make_record(&[("domicile", "GB")]);
        let fields = vec![
            MatchField {
                field_a: "country_code".into(),
                field_b: "domicile".into(),
                method: MatchMethod::Exact,
                scorer: None,
                weight: 0.7,
                fields: None,
            },
            MatchField {
                field_a: String::new(),
                field_b: String::new(),
                method: MatchMethod::Bm25,
                scorer: None,
                weight: 0.3,
                fields: None,
            },
        ];
        let result = score_pair(&b, &a, &fields, None, Some(0.8), None, Side::A);
        // exact: 1.0 * 0.7 = 0.7
        // bm25: 0.8 * 0.3 = 0.24
        // total = 0.94
        assert!(
            (result.total - 0.94).abs() < 0.001,
            "expected 0.94, got {}",
            result.total
        );
        assert_eq!(result.field_scores.len(), 2);

        let bm25_fs = &result.field_scores[1];
        assert_eq!(bm25_fs.method, "bm25");
        assert_eq!(bm25_fs.field_a, "bm25");
        assert_eq!(bm25_fs.field_b, "bm25");
        assert!((bm25_fs.score - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn bm25_score_defaults_to_zero_when_none() {
        let a = make_record(&[]);
        let b = make_record(&[]);
        let fields = vec![MatchField {
            field_a: String::new(),
            field_b: String::new(),
            method: MatchMethod::Bm25,
            scorer: None,
            weight: 1.0,
            fields: None,
        }];
        let result = score_pair(&b, &a, &fields, None, None, None, Side::A);
        assert!(
            result.total.abs() < f64::EPSILON,
            "expected 0.0, got {}",
            result.total
        );
    }
}
