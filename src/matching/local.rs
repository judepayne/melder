//! Bounded augmentation of an unmatched query by relocating one automatic match.
//!
//! This is not global assignment: candidates are limited to the full scored
//! shortlists supplied by callers from `score_pool()`, bounded by `top_n`.
//! Only one incumbent is relocated; callers decide eligible pair provenance.

use crate::crossmap::CrossMapOps;
use crate::error::CrossMapError;
use crate::models::{Classification, MatchResult, Side};
use crate::state::match_log::{MatchLog, MatchLogEvent, ReassignmentMatch};

/// One existing pair replaced by two qualifying pairs.
#[derive(Debug, Clone)]
pub struct LocalPlan {
    pub old_a_id: String,
    pub old_b_id: String,
    pub incoming: MatchResult,
    pub incumbent: MatchResult,
}

/// Return the canonical A/B IDs for a directional scoring result.
pub fn pair_ids(result: &MatchResult) -> (String, String) {
    match result.query_side {
        Side::A => (result.query_id.clone(), result.matched_id.clone()),
        Side::B => (result.matched_id.clone(), result.query_id.clone()),
    }
}

/// Whether a result is an ordinary, finite, automatic scored candidate.
pub fn eligible(result: &MatchResult, auto_match: f64) -> bool {
    result.classification == Classification::Auto
        && matches!(result.reason.as_deref(), None | Some("top_scoring"))
        && result.score.is_finite()
        && result.score >= auto_match
}

/// Keep all automatically acceptable shortlisted hints, without records.
pub fn hints(results: &[MatchResult], auto_match: f64) -> Vec<MatchResult> {
    if gap_demoted(results) {
        return Vec::new();
    }
    results
        .iter()
        .filter(|r| eligible(r, auto_match))
        .map(|r| {
            let mut hint = r.clone();
            hint.matched_record = None;
            hint
        })
        .collect()
}

/// Find the highest combined-score augmentation of a currently unmatched query.
///
/// `incumbent_candidates` must return an empty list for protected or unknown
/// pairs. Batch can use immutable cached scores; live callers must re-score
/// against a mutation-stable pool. There is no recursive displacement.
pub fn find_plan<E>(
    crossmap: &dyn CrossMapOps,
    side: Side,
    query_id: &str,
    results: &[MatchResult],
    auto_match: f64,
    mut incumbent_candidates: impl FnMut(&str, &str, &str) -> Result<Vec<MatchResult>, E>,
) -> Result<Option<LocalPlan>, E> {
    if gap_demoted(results) || partner(crossmap, side, query_id).is_some() {
        return Ok(None);
    }
    let mut best: Option<LocalPlan> = None;
    for incoming in results {
        if !eligible(incoming, auto_match)
            || incoming.query_id != query_id
            || incoming.query_side != side
        {
            continue;
        }
        let Some(incumbent_id) = partner(crossmap, side.opposite(), &incoming.matched_id) else {
            continue;
        };
        if incumbent_id == query_id {
            continue;
        }
        let (old_a, old_b) = match side {
            Side::A => (incumbent_id.as_str(), incoming.matched_id.as_str()),
            Side::B => (incoming.matched_id.as_str(), incumbent_id.as_str()),
        };
        let alternatives = incumbent_candidates(&incumbent_id, old_a, old_b)?;
        if gap_demoted(&alternatives) {
            continue;
        }
        for alternative in &alternatives {
            if !eligible(alternative, auto_match)
                || alternative.query_id != incumbent_id
                || alternative.query_side != side
                || alternative.matched_id == incoming.matched_id
                || partner(crossmap, side.opposite(), &alternative.matched_id).is_some()
            {
                continue;
            }
            let plan = LocalPlan {
                old_a_id: old_a.to_string(),
                old_b_id: old_b.to_string(),
                incoming: incoming.clone(),
                incumbent: alternative.clone(),
            };
            let better = best.as_ref().is_none_or(|current| {
                let score = plan.incoming.score + plan.incumbent.score;
                let current_score = current.incoming.score + current.incumbent.score;
                score > current_score
                    || (score == current_score
                        && (pair_ids(&plan.incoming), pair_ids(&plan.incumbent))
                            < (pair_ids(&current.incoming), pair_ids(&current.incumbent)))
            });
            if better {
                best = Some(plan);
            }
        }
    }
    Ok(best)
}

fn gap_demoted(results: &[MatchResult]) -> bool {
    results
        .first()
        .is_some_and(|r| r.reason.as_deref() == Some("downgraded"))
}

fn partner(crossmap: &dyn CrossMapOps, side: Side, id: &str) -> Option<String> {
    match side {
        Side::A => crossmap.get_b(id),
        Side::B => crossmap.get_a(id),
    }
}

impl LocalPlan {
    /// Construct one replayable event containing both replacement scores.
    pub fn event(&self) -> MatchLogEvent {
        let replacement = |result: &MatchResult| {
            let (a_id, b_id) = pair_ids(result);
            ReassignmentMatch {
                a_id,
                b_id,
                score: result.score,
                rank: result.rank,
                reason: result.reason.clone(),
            }
        };
        MatchLogEvent::CrossMapReassign {
            old_a_id: self.old_a_id.clone(),
            old_b_id: self.old_b_id.clone(),
            replacements: [replacement(&self.incoming), replacement(&self.incumbent)],
        }
    }

    /// Atomically replace the pair, and append one WAL frame if configured.
    ///
    /// Callers serialize overlapping mutation and log emission. If appending
    /// fails, compare-and-replace restores the old pair before returning error.
    /// As with ordinary confirms, an acknowledged append is buffered until flush.
    pub fn commit(
        &self,
        crossmap: &dyn CrossMapOps,
        log: Option<&MatchLog>,
    ) -> Result<bool, CrossMapError> {
        let bytes = log
            .map(|_| MatchLog::serialize_event(&self.event()))
            .transpose()?;
        self.commit_logged(crossmap, || {
            if let (Some(log), Some(bytes)) = (log, bytes) {
                log.append_bytes(&bytes)?;
            }
            Ok(())
        })
    }

    fn commit_logged(
        &self,
        crossmap: &dyn CrossMapOps,
        append: impl FnOnce() -> std::io::Result<()>,
    ) -> Result<bool, CrossMapError> {
        let (a1, b1) = pair_ids(&self.incoming);
        let (a2, b2) = pair_ids(&self.incumbent);
        let old = [(self.old_a_id.as_str(), self.old_b_id.as_str())];
        let new = [(a1.as_str(), b1.as_str()), (a2.as_str(), b2.as_str())];
        if !crossmap.replace_pairs(&old, &new)? {
            return Ok(false);
        }
        if let Err(error) = append() {
            if !crossmap.replace_pairs(&new, &old)? {
                return Err(CrossMapError::Io(std::io::Error::other(format!(
                    "reassignment log failed ({error}); rollback preconditions changed"
                ))));
            }
            return Err(CrossMapError::Io(error));
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::crossmap::MemoryCrossMap;

    use super::*;

    fn make_result(side: Side, query: &str, target: &str, score: f64) -> MatchResult {
        MatchResult {
            query_id: query.into(),
            matched_id: target.into(),
            query_side: side,
            score,
            field_scores: Vec::new(),
            classification: Classification::from_score(score, 0.8, 0.6),
            matched_record: None,
            from_crossmap: false,
            rank: Some(1),
            reason: Some("top_scoring".into()),
        }
    }

    fn make_plan() -> LocalPlan {
        LocalPlan {
            old_a_id: "a1".into(),
            old_b_id: "b1".into(),
            incoming: make_result(Side::B, "b2", "a1", 0.97),
            incumbent: make_result(Side::B, "b1", "a2", 0.98),
        }
    }

    #[test]
    fn one_incumbent_augmentation_is_symmetric() -> Result<(), Box<dyn Error>> {
        for side in [Side::A, Side::B] {
            let map = MemoryCrossMap::new();
            assert!(map.claim("a1", "b1"), "install incumbent");
            let (query, incumbent, target, alternative) = match side {
                Side::A => ("a2", "a1", "b1", "b2"),
                Side::B => ("b2", "b1", "a1", "a2"),
            };
            let results = [make_result(side, query, target, 0.97)];
            let plan = find_plan::<()>(&map, side, query, &results, 0.8, |id, a, b| {
                assert_eq!((id, a, b), (incumbent, "a1", "b1"));
                Ok(vec![
                    make_result(side, incumbent, target, 0.99),
                    make_result(side, incumbent, alternative, 0.98),
                ])
            })
            .map_err(|_| "unexpected scoring error")?
            .ok_or("missing augmentation")?;
            assert!(plan.commit(&map, None)?, "commit augmentation");
            assert_eq!(map.len(), 2, "strictly more pairs");
            assert_eq!(map.get_b("a1").as_deref(), Some("b2"), "new partner");
            assert_eq!(map.get_b("a2").as_deref(), Some("b1"), "relocated partner");
        }
        Ok(())
    }

    #[test]
    fn protected_demoted_and_below_threshold_alternatives_are_not_moved() {
        let map = MemoryCrossMap::new();
        assert!(map.claim("a1", "b1"), "install incumbent");
        let incoming = [make_result(Side::B, "b2", "a1", 0.97)];
        for mode in ["protected", "threshold", "gap", "canonical"] {
            let mut alternative = make_result(Side::B, "b1", "a2", 0.98);
            match mode {
                "threshold" => alternative.score = 0.79,
                "gap" => alternative.classification = Classification::Review,
                "canonical" => alternative.reason = Some("canonical".into()),
                _ => {}
            }
            let result = find_plan::<()>(&map, Side::B, "b2", &incoming, 0.8, |_, _, _| {
                Ok(if mode == "protected" {
                    vec![]
                } else {
                    vec![alternative.clone()]
                })
            });
            assert!(matches!(result, Ok(None)), "must not move: {mode}");
            assert_eq!(map.len(), 1, "no mutation during planning");
        }
    }

    #[test]
    fn score_gap_demotion_cannot_be_bypassed_via_the_second_candidate() {
        let map = MemoryCrossMap::new();
        assert!(map.claim("a1", "b1"), "install incumbent");
        let incoming = [make_result(Side::B, "b2", "a1", 0.97)];
        let mut demoted = make_result(Side::B, "b1", "a1", 0.99);
        demoted.classification = Classification::Review;
        demoted.reason = Some("downgraded".into());
        let alternatives = vec![demoted, make_result(Side::B, "b1", "a2", 0.98)];
        assert!(
            hints(&alternatives, 0.8).is_empty(),
            "do not cache auto alternatives for a demoted query"
        );
        let plan = find_plan::<()>(&map, Side::B, "b2", &incoming, 0.8, |_, _, _| {
            Ok(alternatives.clone())
        });
        assert!(
            matches!(plan, Ok(None)),
            "second candidate must not bypass incumbent score-gap demotion"
        );
    }

    #[test]
    fn full_shortlists_include_late_conflicts_and_incumbent_alternatives() {
        for rank in [4, 5] {
            let map = MemoryCrossMap::new();
            assert!(map.claim("a1", "b1"), "install movable incumbent");
            let mut incoming = Vec::new();
            let mut alternatives = vec![make_result(Side::B, "b1", "a1", 0.99)];
            for i in 0..rank - 1 {
                let target = format!("protected_a{i}");
                assert!(
                    map.claim(&target, &format!("protected_b{i}")),
                    "install protected blocker {i}"
                );
                incoming.push(make_result(Side::B, "b2", &target, 0.99 - i as f64 * 0.01));
                if i < rank - 2 {
                    alternatives.push(make_result(Side::B, "b1", &target, 0.98 - i as f64 * 0.01));
                }
            }
            incoming.push(make_result(Side::B, "b2", "a1", 0.94));
            alternatives.push(make_result(Side::B, "b1", "a2", 0.94));
            for top_n in 1..=rank {
                let incoming_hints = hints(&incoming[..top_n], 0.8);
                let alternative_hints = hints(&alternatives[..top_n], 0.8);
                assert_eq!(incoming_hints.len(), top_n, "retain every qualifying hint");
                assert_eq!(alternative_hints.len(), top_n, "retain every alternative");
                let plan =
                    find_plan::<()>(&map, Side::B, "b2", &incoming_hints, 0.8, |id, _, _| {
                        Ok(if id == "b1" {
                            alternative_hints.clone()
                        } else {
                            Vec::new()
                        })
                    });
                assert!(
                    matches!(plan, Ok(Some(_))) == (top_n == rank),
                    "repair requires rank {rank} in both supplied shortlists: top_n={top_n}"
                );
                assert_eq!(map.len(), rank, "planning leaves all pairs unchanged");
            }
        }
    }

    #[test]
    fn full_shortlist_search_does_not_follow_recursive_chains() {
        let map = MemoryCrossMap::new();
        assert!(map.claim("a1", "b1"), "install incumbent");
        assert!(map.claim("a2", "b3"), "occupy next alternative");
        let incoming = [make_result(Side::B, "b2", "a1", 0.97)];
        let plan = find_plan::<()>(&map, Side::B, "b2", &incoming, 0.8, |id, _, _| {
            assert_eq!(id, "b1", "only score the directly displaced incumbent");
            Ok(vec![make_result(Side::B, "b1", "a2", 0.98)])
        });
        assert!(matches!(plan, Ok(None)), "no recursive displacement of b3");
    }

    #[test]
    fn highest_combined_score_wins_across_the_full_shortlist() -> Result<(), Box<dyn Error>> {
        let map = MemoryCrossMap::new();
        assert!(map.claim("a1", "b1"), "first incumbent");
        assert!(map.claim("a3", "b3"), "second incumbent");
        assert!(map.claim("a5", "b5"), "third incumbent");
        assert!(map.claim("a7", "b7"), "fourth incumbent");
        let incoming = [
            make_result(Side::B, "b2", "a1", 0.98),
            make_result(Side::B, "b2", "a3", 0.97),
            make_result(Side::B, "b2", "a5", 0.96),
            make_result(Side::B, "b2", "a7", 0.95),
        ];
        let plan = find_plan::<()>(&map, Side::B, "b2", &incoming, 0.8, |id, _, _| {
            Ok(vec![match id {
                "b7" => make_result(Side::B, id, "a8", 0.96),
                _ => make_result(Side::B, id, "a2", 0.81),
            }])
        })
        .map_err(|_| "unexpected scoring error")?
        .ok_or("missing plan")?;
        assert_eq!(
            plan.old_a_id, "a7",
            "choose best combined score, not first conflict"
        );
        Ok(())
    }

    #[test]
    fn log_failure_rolls_back_both_replacements() {
        let map = MemoryCrossMap::new();
        assert!(map.claim("a1", "b1"), "install incumbent");
        let result =
            make_plan().commit_logged(&map, || Err(std::io::Error::other("injected log error")));
        assert!(result.is_err(), "log error must not acknowledge success");
        assert_eq!(map.len(), 1, "restore original cardinality");
        assert_eq!(
            map.get_b("a1").as_deref(),
            Some("b1"),
            "restore original pair"
        );
        assert!(!map.has_a("a2") && !map.has_b("b2"), "no half-replacements");
    }

    #[test]
    fn stale_plan_does_not_append_a_log_event() {
        let map = MemoryCrossMap::new();
        assert!(map.claim("a1", "someone_else"), "incumbent changed");
        let mut appended = false;
        let result = make_plan().commit_logged(&map, || {
            appended = true;
            Ok(())
        });
        assert!(matches!(result, Ok(false)), "reject stale plan");
        assert!(!appended, "no event for rejected plan");
    }
}
