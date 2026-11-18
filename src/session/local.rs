//! Opportunistic one-incumbent augmentation for live sessions.
//!
//! Ordinary mutations share a read gate and keep their existing concurrency.
//! Repair never waits for those operations: it needs an exclusive, quiescent
//! window to re-score current records and commit one atomic pair replacement.

use std::sync::TryLockError;
use std::sync::atomic::Ordering;

use crate::error::SessionError;
use crate::matching::local;
use crate::models::{MatchResult, Record, Side};
use crate::vectordb::texthash::compute_text_hash;

use super::Session;

impl Session {
    /// Remember a newly established automatic pair during an exclusive repair.
    pub(super) fn remember_automatic(&self, result: &MatchResult) {
        self.remember_automatic_at(result, self.local_protection_epoch.load(Ordering::Acquire));
    }

    /// Register only if no operator protection intervened after the claim began.
    pub(super) fn remember_automatic_at(&self, result: &MatchResult, provenance: u64) {
        if local::eligible(result, self.state.config.thresholds.auto_match) {
            let mut known = self
                .local_automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (a, b) = local::pair_ids(result);
            if self.local_protection_epoch.load(Ordering::Acquire) == provenance
                && self.state.crossmap.get_b(&a).as_deref() == Some(b.as_str())
            {
                known.insert((a, b));
            }
        }
    }

    /// Discard an optimization hint when its exact pair is broken or replaced.
    pub(super) fn forget_local_pair(&self, a_id: &str, b_id: &str) {
        self.local_automatic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(a_id.to_string(), b_id.to_string()));
    }

    /// Protect an explicit confirmation and invalidate any delayed registrations.
    pub(super) fn forget_local_endpoints(&self, a_id: &str, b_id: &str) {
        let mut known = self
            .local_automatic
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.local_protection_epoch.fetch_add(1, Ordering::AcqRel);
        known.retain(|(a, b)| a != a_id && b != b_id);
    }

    fn vector_is_current(&self, side: Side, id: &str, record: &Record) -> bool {
        self.emb_specs.is_empty()
            || self
                .state
                .side(side)
                .combined_index
                .as_ref()
                .is_some_and(|idx| {
                    idx.text_hash_for(id) == Some(compute_text_hash(record, &self.emb_specs, side))
                })
    }

    /// Re-score a stored query, declining inconsistent cached vector snapshots.
    pub(super) fn score_stored(
        &self,
        side: Side,
        id: &str,
    ) -> Result<Vec<MatchResult>, SessionError> {
        let config = &self.state.config;
        let store = &self.state.store;
        let Some(record) = store.get(side, id)? else {
            return Ok(Vec::new());
        };
        if !self.vector_is_current(side, id, &record) {
            return Ok(Vec::new());
        }
        let combined = match self.state.side(side).combined_index.as_ref() {
            Some(idx) if !self.emb_specs.is_empty() => match idx.get(id) {
                Ok(Some(vector)) => vector,
                _ => return Ok(Vec::new()),
            },
            _ => Vec::new(),
        };
        let pool_side = side.opposite();
        let blocked = if config.blocking.enabled {
            store.blocking_query(&record, side, pool_side)?
        } else {
            store.ids(pool_side)?
        };
        let results = self.score_against_pool(
            id,
            &record,
            side,
            &combined,
            &blocked,
            self.state.side(pool_side),
            pool_side,
            config,
            config.ann_candidates.unwrap_or(50),
            config.bm25_candidates.unwrap_or(10),
            config.top_n.unwrap_or(5),
            &config.synonym_fields,
        );
        if results
            .first()
            .is_some_and(|r| r.reason.as_deref() == Some("downgraded"))
        {
            return Ok(Vec::new());
        }
        // Read-only query APIs can cache a vector without replacing its record.
        // Never relocate against such a vector, or against a removed target.
        let mut current = Vec::with_capacity(results.len());
        for result in results {
            if let Some(target) = store.get(pool_side, &result.matched_id)?
                && self.vector_is_current(pool_side, &result.matched_id, &target)
            {
                current.push(result);
            }
        }
        Ok(current)
    }

    fn clear_resolved_query_reviews(&self, side: Side, id: &str) -> Result<(), SessionError> {
        self.state.store.remove_query_reviews(side, id)?;
        self.state
            .review_queue
            .retain(|_, entry| entry.side != side || entry.id != id);
        Ok(())
    }

    /// Try one strictly cardinality-increasing move without waiting for mutations.
    pub(super) fn try_local_augmentation(
        &self,
        side: Side,
        id: &str,
        record: &Record,
        previous: &[MatchResult],
    ) -> Result<Option<Vec<MatchResult>>, SessionError> {
        let crossmap = self.state.crossmap.as_ref();
        let threshold = self.state.config.thresholds.auto_match;
        let eligible_conflict = previous.iter().any(|result| {
            if !local::eligible(result, threshold) {
                return false;
            }
            let incumbent = match side {
                Side::A => crossmap.get_a(&result.matched_id),
                Side::B => crossmap.get_b(&result.matched_id),
            };
            incumbent.is_some_and(|other| {
                let pair = match side {
                    Side::A => (other, result.matched_id.clone()),
                    Side::B => (result.matched_id.clone(), other),
                };
                self.local_automatic
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .contains(&pair)
            })
        });
        if !eligible_conflict {
            return Ok(None);
        }
        let _exclusive = match self.local_mutations.try_write() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
        };
        let Some(current) = self.state.store.get(side, id)? else {
            return Ok(None);
        };
        let fields = match side {
            Side::A => &self.state.config.required_fields_a,
            Side::B => &self.state.config.required_fields_b,
        };
        // Stores may materialize absent columns as empty strings or omit
        // unconfigured payload fields. Compare the matching-relevant snapshot.
        if !fields.iter().all(|field| {
            current.get(field).map(String::as_str).unwrap_or("")
                == record.get(field).map(String::as_str).unwrap_or("")
        }) {
            return Ok(None);
        }
        let fresh = self.score_stored(side, id)?;
        let plan = local::find_plan(crossmap, side, id, &fresh, threshold, |other, a, b| {
            let known = self
                .local_automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(a.to_string(), b.to_string()));
            if known {
                self.score_stored(side, other)
            } else {
                Ok(Vec::new())
            }
        })?;
        let Some(plan) = plan else {
            return Ok(None);
        };
        if !plan.commit(crossmap, Some(&self.state.wal))? {
            return Ok(None);
        }
        self.state.mark_crossmap_dirty();
        self.forget_local_pair(&plan.old_a_id, &plan.old_b_id);
        self.send_hook(crate::hooks::HookEvent::Break {
            a_id: plan.old_a_id.clone(),
            b_id: plan.old_b_id.clone(),
        });
        for result in [&plan.incoming, &plan.incumbent] {
            let (a_id, b_id) = local::pair_ids(result);
            self.state.store.mark_matched(Side::A, &a_id)?;
            self.state.store.mark_matched(Side::B, &b_id)?;
            self.clear_resolved_query_reviews(Side::A, &a_id)?;
            self.clear_resolved_query_reviews(Side::B, &b_id)?;
            self.remember_automatic(result);
            self.send_hook(crate::hooks::HookEvent::Confirm {
                a_id,
                b_id,
                score: result.score,
                source: "local".to_string(),
                field_scores: result.field_scores.clone(),
            });
        }
        tracing::info!(old_a_id = %plan.old_a_id, old_b_id = %plan.old_b_id, "local match augmentation");
        Ok(Some(fresh))
    }
}
