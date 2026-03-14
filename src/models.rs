//! Core domain models shared across modules.

use std::collections::HashMap;

/// A record is a flat string→string map. Field names are lowercase.
pub type Record = HashMap<String, String>;

/// Side represents which dataset a record belongs to.
/// Used throughout the engine to ensure symmetric handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    A,
    B,
}

impl Side {
    pub fn opposite(&self) -> Side {
        match self {
            Side::A => Side::B,
            Side::B => Side::A,
        }
    }
}

/// Score contribution from a single match field.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldScore {
    pub field_a: String,
    pub field_b: String,
    pub method: String, // "exact" | "fuzzy" | "embedding" | "numeric" | "bm25"
    pub score: f64,     // 0.0..=1.0
    pub weight: f64,
}

impl FieldScore {
    pub fn contribution(&self) -> f64 {
        self.score * self.weight
    }
}

/// Internal engine result. Uses query_id/matched_id to avoid the confusing
/// a_id/b_id naming when direction reverses. Mapped to a_id/b_id at the
/// API serialization boundary based on `query_side`.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub query_id: String,
    pub matched_id: String,
    pub query_side: Side,
    pub score: f64,
    pub field_scores: Vec<FieldScore>,
    pub classification: Classification,
    pub matched_record: Option<Record>,
    pub from_crossmap: bool,
}

/// Match classification based on thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    Auto,
    Review,
    NoMatch,
}

impl Classification {
    pub fn as_str(&self) -> &'static str {
        match self {
            Classification::Auto => "auto",
            Classification::Review => "review",
            Classification::NoMatch => "no_match",
        }
    }

    pub fn from_score(score: f64, auto_match: f64, review_floor: f64) -> Self {
        if score >= auto_match {
            Classification::Auto
        } else if score >= review_floor {
            Classification::Review
        } else {
            Classification::NoMatch
        }
    }
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_auto() {
        assert_eq!(
            Classification::from_score(0.9, 0.85, 0.60),
            Classification::Auto
        );
    }

    #[test]
    fn classification_review() {
        assert_eq!(
            Classification::from_score(0.7, 0.85, 0.60),
            Classification::Review
        );
    }

    #[test]
    fn classification_no_match() {
        assert_eq!(
            Classification::from_score(0.3, 0.85, 0.60),
            Classification::NoMatch
        );
    }

    #[test]
    fn classification_boundary_auto() {
        // Exactly at auto_match threshold → Auto
        assert_eq!(
            Classification::from_score(0.85, 0.85, 0.60),
            Classification::Auto
        );
    }

    #[test]
    fn classification_boundary_review() {
        // Exactly at review_floor threshold → Review
        assert_eq!(
            Classification::from_score(0.60, 0.85, 0.60),
            Classification::Review
        );
    }

    #[test]
    fn side_opposite() {
        assert_eq!(Side::A.opposite(), Side::B);
        assert_eq!(Side::B.opposite(), Side::A);
    }

    #[test]
    fn field_score_contribution() {
        let fs = FieldScore {
            field_a: "a".into(),
            field_b: "b".into(),
            method: "exact".into(),
            score: 0.8,
            weight: 0.5,
        };
        assert!((fs.contribution() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn classification_as_str() {
        assert_eq!(Classification::Auto.as_str(), "auto");
        assert_eq!(Classification::Review.as_str(), "review");
        assert_eq!(Classification::NoMatch.as_str(), "no_match");
    }
}
