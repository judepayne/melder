//! Synchronous live proposals and atomic, audited assignment steps.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::bm25::scorer::normalise_bm25;
use crate::error::SessionError;
use crate::matching::{local, pipeline};
use crate::models::{MatchResult, Side};
use crate::state::live::{ReviewEntry, review_queue_key};
use crate::state::match_log::{
    AssignmentDelta, AssignmentEndpoint, AssignmentMatch, AssignmentPair, MatchLog, MatchLogEvent,
};
use crate::vectordb::texthash::compute_text_hash;

use super::{Session, response::AssignmentSummary};

pub(super) type Entity = (Side, String);

#[derive(Default)]
pub(super) struct LiveAssignmentState {
    gate: Mutex<()>,
    automatic: Mutex<HashSet<(String, String)>>,
    unhealthy: AtomicBool,
    next_attempt: AtomicU64,
    #[cfg(test)]
    fail_next_append: AtomicBool,
    #[cfg(test)]
    pause_after_displacement: Mutex<Option<std::path::PathBuf>>,
}

impl Session {
    /// Flush durable membership without snapshotting a proposal before its WAL event.
    pub(crate) fn flush_assignment_crossmap(&self) -> Result<(), crate::error::MelderError> {
        let gate = self.assignment_guard().map_err(anyhow::Error::new)?;
        if gate.is_some() {
            self.flush_assignment_log().map_err(anyhow::Error::new)?;
        }
        self.state.flush_crossmap()
    }

    pub(crate) fn flush_assignment_log(&self) -> Result<(), SessionError> {
        self.state
            .wal
            .flush()
            .map_err(SessionError::from)
            .inspect_err(|e| self.assignment_failed(e))
    }

    /// Serialize optimizing live mutations or single-pool enrollment and snapshots.
    /// Call only at operation boundaries; helpers under this guard must not relock.
    pub(crate) fn assignment_guard(&self) -> Result<Option<MutexGuard<'_, ()>>, SessionError> {
        if let Some(gate) = &self.enroll_gate {
            return gate
                .lock()
                .map(Some)
                .map_err(|_| SessionError::AssignmentUnhealthy);
        }
        let Some(state) = &self.assignment else {
            return Ok(None);
        };
        let guard = state
            .gate
            .lock()
            .map_err(|_| SessionError::AssignmentUnhealthy)?;
        if state.unhealthy.load(Ordering::Relaxed) {
            return Err(SessionError::AssignmentUnhealthy);
        }
        Ok(Some(guard))
    }

    #[cfg(test)]
    pub(super) fn fail_next_assignment_append(&self) {
        if let Some(state) = &self.assignment {
            state.fail_next_append.store(true, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    pub(super) fn pause_after_assignment_displacement(&self, ready: std::path::PathBuf) {
        if let Some(state) = &self.assignment {
            *state
                .pause_after_displacement
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(ready);
        }
    }

    pub(super) fn append_assignment(&self, bytes: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if self
            .assignment
            .as_ref()
            .is_some_and(|state| state.fail_next_append.swap(false, Ordering::Relaxed))
        {
            return Err(std::io::Error::other("injected assignment append failure"));
        }
        self.state.wal.append_bytes(bytes)
    }

    pub(super) fn assignment_failed(&self, error: &SessionError) {
        if let Some(state) = &self.assignment
            && matches!(
                error,
                SessionError::Wal(_)
                    | SessionError::Store(_)
                    | SessionError::CrossMap(_)
                    | SessionError::Assignment(_)
            )
        {
            state.unhealthy.store(true, Ordering::Relaxed);
        }
    }

    pub(super) fn forget_assignment(&self, a: &str, b: &str) {
        if let Some(state) = &self.assignment {
            state
                .automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&(a.into(), b.into()));
        }
    }

    pub(super) fn can_confirm_canonical(&self, side: Side, id: &str, target: &str) -> bool {
        let Some(state) = &self.assignment else {
            return true;
        };
        if self.state.exclusions.contains(id, target) {
            return false;
        }
        let Some(incumbent) = self.assigned_partner(side.opposite(), target) else {
            return true;
        };
        if incumbent == id {
            return true;
        }
        let pair = if side == Side::A {
            (incumbent, target.to_owned())
        } else {
            (target.to_owned(), incumbent)
        };
        state
            .automatic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&pair)
    }

    pub(super) fn protect_assignment(&self, a: &str, b: &str) {
        if let Some(state) = &self.assignment {
            state
                .automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(old_a, old_b)| old_a != a && old_b != b);
        }
    }

    pub(super) fn assigned_partner(&self, side: Side, id: &str) -> Option<String> {
        match side {
            Side::A => self.state.crossmap.get_b(id),
            Side::B => self.state.crossmap.get_a(id),
        }
    }

    pub(super) fn clear_assignment_review(&self, side: Side, id: &str) -> Result<(), SessionError> {
        self.state.store.remove_query_reviews(side, id)?;
        self.state
            .review_queue
            .retain(|_, entry| entry.side != side || entry.id != id);
        Ok(())
    }

    pub(super) fn assignment_classification(
        &self,
        side: Side,
        id: &str,
        partner: Option<&str>,
    ) -> &'static str {
        if partner.is_some() {
            "auto"
        } else if self
            .state
            .review_queue
            .iter()
            .any(|entry| entry.side == side && entry.id == id)
        {
            "review"
        } else {
            "no_match"
        }
    }

    pub(super) fn refresh_assignment_summary(
        &self,
        summary: &mut AssignmentSummary,
        side: Side,
        id: &str,
    ) {
        summary.partner = self.assigned_partner(side, id);
        summary
            .unmatched
            .retain(|e| self.assigned_partner(e.side, &e.id).is_none());
        for (side, id) in
            std::iter::once((side, id)).chain(summary.changes.iter().flat_map(|delta| {
                delta
                    .replacements
                    .iter()
                    .flat_map(|p| [(Side::A, p.a_id.as_str()), (Side::B, p.b_id.as_str())])
            }))
        {
            if self.assigned_partner(side, id).is_none()
                && !summary
                    .unmatched
                    .iter()
                    .any(|e| e.side == side && e.id == id)
            {
                summary.unmatched.push(AssignmentEndpoint {
                    side,
                    id: id.into(),
                });
            }
        }
    }

    fn current_vector(&self, side: Side, id: &str, record: &crate::models::Record) -> bool {
        self.emb_specs.is_empty()
            || self
                .state
                .side(side)
                .combined_index
                .as_ref()
                .is_some_and(|index| {
                    index.text_hash_for(id)
                        == Some(compute_text_hash(record, &self.emb_specs, side))
                })
    }

    fn assignment_scores(
        &self,
        side: Side,
        id: &str,
        verify: &[String],
    ) -> Result<Vec<MatchResult>, SessionError> {
        let config = &self.state.config;
        let store = &self.state.store;
        let Some(record) = store.get(side, id)? else {
            return Ok(vec![]);
        };
        if !self.current_vector(side, id, &record) {
            tracing::info!(?side, id, "assignment skipped stale query vector");
            return Ok(vec![]);
        }
        let vector = if self.emb_specs.is_empty() {
            vec![]
        } else {
            self.state
                .side(side)
                .combined_index
                .as_ref()
                .and_then(|idx| idx.get(id).ok().flatten())
                .unwrap_or_default()
        };
        if !self.emb_specs.is_empty() && vector.is_empty() {
            return Ok(vec![]);
        }
        let other = side.opposite();
        let pool = self.state.side(other);
        let blocked = if config.blocking.enabled {
            store.blocking_query(&record, side, other)?
        } else {
            store.ids(other)?
        };
        let mut keep = vec![];
        for target in verify {
            if blocked.contains(target)
                && !self.state.exclusions.contains(id, target)
                && let Some(candidate) = store.get(other, target)?
                && self.current_vector(other, target, &candidate)
            {
                keep.push(target.clone());
            }
        }
        if keep.len() != verify.len() {
            return Ok(vec![]);
        }
        let (bm25_ids, bm25_scores) = if let Some(bm25) = &pool.bm25_index {
            let query = bm25.query_text_for(&record, side);
            let self_score = bm25.analytical_self_score(&query);
            let hits = bm25.score_blocked(&query, &blocked, config.bm25_candidates.unwrap_or(10));
            let ids: Vec<_> = hits.iter().map(|(id, _)| id.clone()).collect();
            let mut scores: HashMap<_, _> = hits
                .into_iter()
                .map(|(id, raw)| (id, normalise_bm25(raw, self_score)))
                .collect();
            for (id, raw) in bm25.score_blocked(&query, &keep, keep.len()) {
                scores.insert(id, normalise_bm25(raw, self_score));
            }
            (ids, scores)
        } else {
            (vec![], HashMap::new())
        };
        let index = pool
            .synonym_index
            .as_ref()
            .map(|i| i.read().unwrap_or_else(|e| e.into_inner()));
        let synonyms = index.as_ref().map_or_else(Vec::new, |i| {
            pipeline::synonym_candidate_stage(i, &record, side, &config.synonym_fields)
        });
        let pool = pipeline::ScoringPool {
            store: store.as_ref(),
            side: other,
            combined_index: pool.combined_index.as_deref(),
            blocked_ids: &blocked,
            bm25_candidate_ids: &bm25_ids,
            bm25_scores_map: &bm25_scores,
            synonym_candidate_ids: &synonyms,
            synonym_dictionary: self.state.synonym_dictionary.as_deref(),
            exclusions: &self.state.exclusions,
        };
        let top_n = if verify.is_empty() {
            config.top_n.unwrap_or(5)
        } else {
            0
        };
        let results = pipeline::score_pool_verified(
            &pipeline::ScoringQuery {
                id,
                record: &record,
                side,
                combined_vec: &vector,
            },
            &pool,
            config,
            config.ann_candidates.unwrap_or(50),
            top_n,
            &keep,
        )?;
        // Do not remove a stale candidate and thereby manufacture a clear score gap.
        for candidate in &results {
            let Some(current) = store.get(other, &candidate.matched_id)? else {
                return Ok(vec![]);
            };
            if !self.current_vector(other, &candidate.matched_id, &current) {
                tracing::info!(?side, id, candidate = %candidate.matched_id, "assignment skipped stale candidate vector");
                return Ok(vec![]);
            }
        }
        if verify.is_empty()
            && let Some(log) = &self.scoring_log
        {
            log.send(id, side, &results);
        }
        Ok(results)
    }

    fn commit_assignment(
        &self,
        delta: &AssignmentDelta,
        primary: &MatchResult,
    ) -> Result<(), SessionError> {
        let map = &self.state.crossmap;
        let old: Vec<_> = delta
            .expected
            .iter()
            .map(|p| (p.a_id.as_str(), p.b_id.as_str()))
            .collect();
        let new: Vec<_> = delta
            .replacements
            .iter()
            .map(|p| (p.a_id.as_str(), p.b_id.as_str()))
            .collect();
        delta.validate()?;
        let bytes = MatchLog::serialize_event(&MatchLogEvent::AssignmentDelta(delta.clone()))?;
        if !map.apply_assignment_delta(&old, &new)? {
            return Err(SessionError::Assignment(
                "pair ownership changed while applying proposal".into(),
            ));
        }
        if let Err(error) = self.append_assignment(&bytes) {
            if let Some(state) = &self.assignment {
                state.unhealthy.store(true, Ordering::Relaxed);
            }
            match map.apply_assignment_delta(&new, &old) {
                Ok(true) => return Err(error.into()),
                result => {
                    return Err(SessionError::Assignment(format!(
                        "WAL append failed ({error}); rollback did not complete: {result:?}"
                    )));
                }
            }
        }
        self.state.mark_crossmap_dirty();
        for endpoint in &delta.newly_unmatched {
            self.state
                .store
                .mark_unmatched(endpoint.side, &endpoint.id)?;
            self.clear_assignment_review(endpoint.side, &endpoint.id)?;
        }
        for (a, b) in &new {
            for (side, id) in [(Side::A, *a), (Side::B, *b)] {
                self.state.store.mark_matched(side, id)?;
                self.clear_assignment_review(side, id)?;
            }
        }
        if let Some(state) = &self.assignment {
            let mut automatic = state.automatic.lock().unwrap_or_else(|e| e.into_inner());
            for (a, b) in &old {
                automatic.remove(&(a.to_string(), b.to_string()));
            }
            automatic.extend(new.iter().map(|(a, b)| (a.to_string(), b.to_string())));
        }
        for p in &delta.expected {
            self.send_hook(crate::hooks::HookEvent::Break {
                a_id: p.a_id.clone(),
                b_id: p.b_id.clone(),
            });
        }
        for p in &delta.replacements {
            self.send_hook(crate::hooks::HookEvent::Confirm {
                a_id: p.a_id.clone(),
                b_id: p.b_id.clone(),
                score: primary.score,
                source: "deferred_acceptance".into(),
                field_scores: primary.field_scores.clone(),
            });
        }
        #[cfg(test)]
        if !delta.expected.is_empty()
            && let Some(state) = &self.assignment
            && let Some(ready) = state
                .pause_after_displacement
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
        {
            self.state.wal.flush()?;
            std::fs::write(ready, b"committed")?;
            loop {
                std::thread::park();
            }
        }
        Ok(())
    }

    fn finish_unmatched(
        &self,
        side: Side,
        id: &str,
        results: &[MatchResult],
    ) -> Result<(), SessionError> {
        self.clear_assignment_review(side, id)?;
        let best = results.first();
        if let Some(r) = best
            .filter(|r| r.score.is_finite() && r.score >= self.state.config.thresholds.review_floor)
        {
            let reason = if local::eligible(r, self.state.config.thresholds.auto_match) {
                Some("assignment_conflict".into())
            } else {
                r.reason.clone()
            };
            let key = review_queue_key(side, id, &r.matched_id);
            self.state.wal.append(&MatchLogEvent::ReviewMatch {
                id: id.into(),
                side,
                candidate_id: r.matched_id.clone(),
                score: r.score,
                rank: r.rank,
                reason,
            })?;
            self.state
                .store
                .persist_review(&key, id, side, &r.matched_id, r.score)?;
            self.state.review_queue.insert(
                key,
                ReviewEntry {
                    id: id.into(),
                    side,
                    candidate_id: r.matched_id.clone(),
                    score: r.score,
                },
            );
            let (a_id, b_id) = local::pair_ids(r);
            self.send_hook(crate::hooks::HookEvent::Review {
                a_id,
                b_id,
                score: r.score,
                field_scores: r.field_scores.clone(),
            });
        } else {
            self.state.wal.append(&MatchLogEvent::NoMatchBelow {
                query_id: id.into(),
                query_side: side,
                best_candidate_id: best.map(|r| r.matched_id.clone()),
                best_score: best.map(|r| r.score),
            })?;
            self.send_hook(crate::hooks::HookEvent::NoMatch {
                side,
                id: id.into(),
                best_score: best.map(|r| r.score),
                best_candidate_id: best.map(|r| r.matched_id.clone()),
            });
        }
        Ok(())
    }

    pub(super) fn reconcile_assignment(
        &self,
        root: Entity,
        seeds: Vec<Entity>,
    ) -> Result<(AssignmentSummary, Vec<MatchResult>), SessionError> {
        let result = self.reconcile_assignment_inner(root, seeds);
        if let Err(error) = &result {
            self.assignment_failed(error);
        }
        result
    }

    fn reconcile_assignment_inner(
        &self,
        root: Entity,
        seeds: Vec<Entity>,
    ) -> Result<(AssignmentSummary, Vec<MatchResult>), SessionError> {
        let state = self
            .assignment
            .as_ref()
            .ok_or_else(|| SessionError::Assignment("strategy not enabled".into()))?;
        let attempt_id = state.next_attempt.fetch_add(1, Ordering::Relaxed);
        let threshold = self.state.config.thresholds.auto_match;
        let cap = self.state.config.live.assignment.max_displacements;
        let mut queue = VecDeque::from(seeds);
        let mut attempted = HashSet::new();
        let mut affected = vec![];
        let mut listed = HashSet::new();
        let mut lists: HashMap<Entity, Vec<MatchResult>> = HashMap::new();
        let mut summary = AssignmentSummary {
            strategy: "deferred_acceptance",
            displacements: 0,
            limit_reached: false,
            changes: vec![],
            unmatched: vec![],
            partner: None,
        };
        while let Some((side, id)) = queue.pop_front() {
            if self.assigned_partner(side, &id).is_some()
                || !self.state.store.contains(side, &id)?
            {
                continue;
            }
            if listed.insert((side, id.clone())) {
                affected.push((side, id.clone()));
            }
            let key = (side, id.clone());
            let forward: &[MatchResult] = match lists.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(self.assignment_scores(side, &id, &[])?)
                }
            };
            if ambiguous(forward) {
                continue;
            }
            let receiver_side = side.opposite();
            for p in forward {
                if !local::eligible(p, threshold)
                    || !attempted.insert((side, id.clone(), p.matched_id.clone()))
                {
                    continue;
                }
                let receiver = &p.matched_id;
                let incumbent = self.assigned_partner(receiver_side, receiver);
                let mut verify = vec![id.clone()];
                if let Some(old) = &incumbent {
                    let (a, b) = if side == Side::A {
                        (old, receiver)
                    } else {
                        (receiver, old)
                    };
                    if !state
                        .automatic
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .contains(&(a.clone(), b.clone()))
                    {
                        continue;
                    }
                    verify.push(old.clone());
                }
                let reverse = self.assignment_scores(receiver_side, receiver, &verify)?;
                if ambiguous(&reverse) {
                    continue;
                }
                let Some(new) = reverse
                    .iter()
                    .find(|r| r.matched_id == id && local::eligible(r, threshold))
                else {
                    continue;
                };
                let mut previous_receiver_score = None;
                if let Some(old) = &incumbent {
                    let Some(prior) = reverse
                        .iter()
                        .find(|r| &r.matched_id == old && r.score.is_finite())
                    else {
                        continue;
                    };
                    if new.score <= prior.score {
                        continue;
                    }
                    if summary.displacements == cap {
                        summary.limit_reached = true;
                        continue;
                    }
                    previous_receiver_score = Some(prior.score);
                }
                let (primary, other) = if side == Side::B { (p, new) } else { (new, p) };
                let (a_id, b_id) = local::pair_ids(p);
                let mut expected = vec![];
                let mut newly_unmatched = vec![];
                if let Some(old) = &incumbent {
                    expected.push(if side == Side::A {
                        AssignmentPair {
                            a_id: old.clone(),
                            b_id: b_id.clone(),
                        }
                    } else {
                        AssignmentPair {
                            a_id: a_id.clone(),
                            b_id: old.clone(),
                        }
                    });
                    newly_unmatched.push(AssignmentEndpoint {
                        side,
                        id: old.clone(),
                    });
                }
                let delta = AssignmentDelta {
                    version: 1,
                    attempt_id,
                    step_index: summary.changes.len(),
                    proposer: AssignmentEndpoint {
                        side,
                        id: id.clone(),
                    },
                    expected,
                    replacements: vec![AssignmentMatch {
                        a_id,
                        b_id,
                        score_b_to_a: primary.score,
                        score_a_to_b: other.score,
                        rank: if side == Side::B { p.rank } else { None },
                        reason: primary.reason.clone(),
                    }],
                    newly_unmatched,
                    previous_receiver_score,
                };
                self.commit_assignment(&delta, primary)?;
                if listed.insert((receiver_side, receiver.clone())) {
                    affected.push((receiver_side, receiver.clone()));
                }
                summary.changes.push(delta);
                if let Some(old) = incumbent {
                    summary.displacements += 1;
                    queue.push_front((side, old));
                }
                break;
            }
        }
        for (side, id) in affected {
            if self.assigned_partner(side, &id).is_none() && self.state.store.contains(side, &id)? {
                self.finish_unmatched(
                    side,
                    &id,
                    lists.get(&(side, id.clone())).map_or(&[], Vec::as_slice),
                )?;
                summary.unmatched.push(AssignmentEndpoint { side, id });
            }
        }
        summary.partner = self.assigned_partner(root.0, &root.1);
        Ok((summary, lists.remove(&root).unwrap_or_default()))
    }
}

fn ambiguous(results: &[MatchResult]) -> bool {
    results
        .first()
        .is_some_and(|r| r.reason.as_deref() == Some("downgraded"))
}
