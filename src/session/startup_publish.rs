//! Publish graph startup decisions into the live session before accepting requests.

use crate::batch::RecordOutcome;
use crate::error::SessionError;
use crate::hooks::HookEvent;
use crate::models::Side;
use crate::state::live::{ReviewEntry, review_queue_key};
use crate::state::match_log::{MatchLog, MatchLogEvent};

use super::Session;

impl Session {
    pub(super) fn publish_startup_graph(
        &self,
        outcomes: Vec<RecordOutcome>,
    ) -> Result<(), SessionError> {
        let _guard = self.assignment_guard()?;
        let mut events = Vec::with_capacity(outcomes.len());
        let mut pairs = Vec::new();
        for outcome in &outcomes {
            events.push(match outcome {
                RecordOutcome::Auto(result) => {
                    if self
                        .state
                        .exclusions
                        .contains(&result.matched_id, &result.query_id)
                    {
                        return Err(SessionError::Assignment("excluded startup pair".into()));
                    }
                    pairs.push((result.matched_id.as_str(), result.query_id.as_str()));
                    MatchLogEvent::CrossMapConfirm {
                        a_id: result.matched_id.clone(),
                        b_id: result.query_id.clone(),
                        score: Some(result.score),
                        rank: result.rank,
                        reason: result.reason.clone(),
                    }
                }
                RecordOutcome::Review(result) => MatchLogEvent::ReviewMatch {
                    id: result.query_id.clone(),
                    side: Side::B,
                    candidate_id: result.matched_id.clone(),
                    score: result.score,
                    rank: result.rank,
                    reason: result.reason.clone(),
                },
                RecordOutcome::NoMatch(id, _, score) => MatchLogEvent::NoMatchBelow {
                    query_id: id.clone(),
                    query_side: Side::B,
                    best_candidate_id: None,
                    best_score: *score,
                },
                RecordOutcome::Scored(_) => {
                    return Err(SessionError::Assignment(
                        "unresolved startup scoring outcome".into(),
                    ));
                }
            });
        }
        let reviews: Vec<_> = outcomes
            .iter()
            .filter_map(|o| match o {
                RecordOutcome::Review(r) => {
                    Some((review_queue_key(Side::B, &r.query_id, &r.matched_id), r))
                }
                _ => None,
            })
            .collect();
        let review_rows: Vec<_> = reviews
            .iter()
            .map(|(key, r)| {
                (
                    key.as_str(),
                    r.query_id.as_str(),
                    r.matched_id.as_str(),
                    r.score,
                )
            })
            .collect();
        for &(a, b) in &pairs {
            if self.state.crossmap.has_a(a) || self.state.crossmap.has_b(b) {
                return Err(SessionError::Assignment(
                    "startup publication conflicts with existing pairs".into(),
                ));
            }
        }
        // Serialize before modifying authoritative membership. One frame prevents
        // memory recovery or offline export from observing a prefix of the graph.
        let frame = MatchLog::serialize_startup_assignment(events)?;
        if let Err(error) = self
            .append_assignment(&frame)
            .and_then(|()| self.state.wal.flush())
        {
            // No live or durable membership has changed yet. A complete frame
            // may be replayed after an ambiguous flush, but never a partial graph.
            let error =
                SessionError::Assignment(format!("startup WAL publication failed: {error}"));
            self.assignment_failed(&error);
            return Err(error);
        }
        // SQLite commits membership, review removals/inserts and matched flags
        // together. It never replays memory WAL events into authoritative state.
        if !self.state.crossmap.publish_startup(&pairs, &review_rows)? {
            return Err(SessionError::Assignment(
                "startup publication conflicts with existing pairs".into(),
            ));
        }
        for outcome in &outcomes {
            match outcome {
                RecordOutcome::Auto(result) => {
                    self.state.review_queue.retain(|_, row| {
                        !((row.side == Side::A && row.id == result.matched_id)
                            || (row.side == Side::B && row.id == result.query_id))
                    });
                    self.state.store.mark_matched(Side::A, &result.matched_id)?;
                    self.state.store.mark_matched(Side::B, &result.query_id)?;
                    // Startup pairs are not request-created automatic incumbents:
                    // never enroll them in deferred acceptance's displacement set.
                    self.protect_assignment(&result.matched_id, &result.query_id);
                }
                RecordOutcome::Review(result) => {
                    self.state.review_queue.insert(
                        review_queue_key(Side::B, &result.query_id, &result.matched_id),
                        ReviewEntry {
                            id: result.query_id.clone(),
                            side: Side::B,
                            candidate_id: result.matched_id.clone(),
                            score: result.score,
                        },
                    );
                }
                _ => {}
            }
        }
        self.state.mark_crossmap_dirty();
        // Emit best-effort hooks only after the full publication is complete.
        for outcome in outcomes {
            if let RecordOutcome::Auto(result) = outcome {
                self.send_hook(HookEvent::Confirm {
                    a_id: result.matched_id,
                    b_id: result.query_id,
                    score: result.score,
                    source: "startup_graph_assignment".into(),
                    field_scores: result.field_scores,
                });
            }
        }
        Ok(())
    }
}
