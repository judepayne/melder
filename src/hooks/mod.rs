//! Pipeline hooks — event notifications via a long-running subprocess.
//!
//! Events (match confirmed, review queued, no match, match broken) are
//! serialized as newline-delimited JSON and written to the subprocess stdin.

pub mod writer;

use serde::Serialize;

use crate::models::{FieldScore, Side};

// ---------------------------------------------------------------------------
// Hook event types
// ---------------------------------------------------------------------------

/// A pipeline event dispatched to the hook subprocess.
#[derive(Debug, Clone)]
pub enum HookEvent {
    /// Match confirmed — auto (threshold) or manual (API).
    Confirm {
        a_id: String,
        b_id: String,
        score: f64,
        source: String,
        field_scores: Vec<FieldScore>,
    },
    /// Pair entered the review queue.
    Review {
        a_id: String,
        b_id: String,
        score: f64,
        field_scores: Vec<FieldScore>,
    },
    /// No match found for a record.
    NoMatch {
        side: Side,
        id: String,
        best_score: Option<f64>,
        best_candidate_id: Option<String>,
    },
    /// A previously confirmed match was broken.
    Break { a_id: String, b_id: String },
    /// A pair was excluded (known non-match).
    Exclude {
        a_id: String,
        b_id: String,
        match_was_broken: bool,
    },
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

/// Wire format for hook events. Uses a `type` discriminator field rather
/// than serde's default enum tagging — cleaner for external consumers.
#[derive(Serialize)]
struct HookEventJson<'a> {
    #[serde(rename = "type")]
    event_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    a_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    side: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_candidate_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_scores: Option<&'a [FieldScore]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    match_was_broken: Option<bool>,
}

impl HookEvent {
    /// Serialize to a JSON line (with trailing newline).
    pub fn to_json_line(&self) -> String {
        let wire = match self {
            HookEvent::Confirm {
                a_id,
                b_id,
                score,
                source,
                field_scores,
            } => HookEventJson {
                event_type: "on_confirm",
                a_id: Some(a_id),
                b_id: Some(b_id),
                side: None,
                id: None,
                score: Some(*score),
                source: Some(source),
                best_score: None,
                best_candidate_id: None,
                field_scores: if field_scores.is_empty() {
                    None
                } else {
                    Some(field_scores)
                },
                match_was_broken: None,
            },
            HookEvent::Review {
                a_id,
                b_id,
                score,
                field_scores,
            } => HookEventJson {
                event_type: "on_review",
                a_id: Some(a_id),
                b_id: Some(b_id),
                side: None,
                id: None,
                score: Some(*score),
                source: None,
                best_score: None,
                best_candidate_id: None,
                field_scores: if field_scores.is_empty() {
                    None
                } else {
                    Some(field_scores)
                },
                match_was_broken: None,
            },
            HookEvent::NoMatch {
                side,
                id,
                best_score,
                best_candidate_id,
            } => HookEventJson {
                event_type: "on_nomatch",
                a_id: None,
                b_id: None,
                side: Some(match side {
                    Side::A => "a",
                    Side::B => "b",
                }),
                id: Some(id),
                score: None,
                source: None,
                best_score: *best_score,
                best_candidate_id: best_candidate_id.as_deref(),
                field_scores: None,
                match_was_broken: None,
            },
            HookEvent::Break { a_id, b_id } => HookEventJson {
                event_type: "on_break",
                a_id: Some(a_id),
                b_id: Some(b_id),
                side: None,
                id: None,
                score: None,
                source: None,
                best_score: None,
                best_candidate_id: None,
                field_scores: None,
                match_was_broken: None,
            },
            HookEvent::Exclude {
                a_id,
                b_id,
                match_was_broken,
            } => HookEventJson {
                event_type: "on_exclude",
                a_id: Some(a_id),
                b_id: Some(b_id),
                side: None,
                id: None,
                score: None,
                source: None,
                best_score: None,
                best_candidate_id: None,
                field_scores: None,
                match_was_broken: Some(*match_was_broken),
            },
        };
        let mut json = serde_json::to_string(&wire).unwrap_or_default();
        json.push('\n');
        json
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirm_event_json() {
        let event = HookEvent::Confirm {
            a_id: "A1".into(),
            b_id: "B1".into(),
            score: 0.95,
            source: "auto".into(),
            field_scores: vec![FieldScore {
                field_a: "name".into(),
                field_b: "name".into(),
                method: "exact".into(),
                score: 1.0,
                weight: 1.0,
            }],
        };
        let json = event.to_json_line();
        assert!(json.ends_with('\n'), "should end with newline");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "on_confirm");
        assert_eq!(parsed["a_id"], "A1");
        assert_eq!(parsed["b_id"], "B1");
        assert_eq!(parsed["score"], 0.95);
        assert_eq!(parsed["source"], "auto");
        assert!(parsed["field_scores"].is_array());
    }

    #[test]
    fn review_event_json() {
        let event = HookEvent::Review {
            a_id: "A1".into(),
            b_id: "B1".into(),
            score: 0.73,
            field_scores: vec![],
        };
        let json = event.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "on_review");
        assert_eq!(parsed["score"], 0.73);
        // Empty field_scores omitted
        assert!(parsed.get("field_scores").is_none());
    }

    #[test]
    fn nomatch_event_json_with_candidate() {
        let event = HookEvent::NoMatch {
            side: Side::B,
            id: "B99".into(),
            best_score: Some(0.42),
            best_candidate_id: Some("A55".into()),
        };
        let json = event.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "on_nomatch");
        assert_eq!(parsed["side"], "b");
        assert_eq!(parsed["id"], "B99");
        assert_eq!(parsed["best_score"], 0.42);
        assert_eq!(parsed["best_candidate_id"], "A55");
    }

    #[test]
    fn nomatch_event_json_no_candidate() {
        let event = HookEvent::NoMatch {
            side: Side::A,
            id: "A1".into(),
            best_score: None,
            best_candidate_id: None,
        };
        let json = event.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "on_nomatch");
        assert!(parsed.get("best_score").is_none());
        assert!(parsed.get("best_candidate_id").is_none());
    }

    #[test]
    fn break_event_json() {
        let event = HookEvent::Break {
            a_id: "A1".into(),
            b_id: "B1".into(),
        };
        let json = event.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "on_break");
        assert_eq!(parsed["a_id"], "A1");
        assert_eq!(parsed["b_id"], "B1");
        // Minimal — no score, no field_scores
        assert!(parsed.get("score").is_none());
    }
}
