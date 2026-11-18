//! `CrossMapOps` trait: runtime interface for bidirectional record-pair mapping.
//!
//! Persistence details differ between backends: `MemoryCrossMap` flushes to
//! CSV via `flush()`, while `SqliteCrossMap` is write-through (flush is a
//! no-op). The trait's `flush()` method abstracts this.

use std::collections::HashSet;

use crate::error::CrossMapError;

pub(super) fn validate_assignment_delta<'a>(
    expected: &[(&'a str, &'a str)],
    replacements: &[(&str, &str)],
) -> Option<(HashSet<&'a str>, HashSet<&'a str>)> {
    if expected.is_empty() && replacements.is_empty() {
        return None;
    }
    fn endpoints<'a>(pairs: &[(&'a str, &'a str)]) -> Option<(HashSet<&'a str>, HashSet<&'a str>)> {
        let mut a_ids = HashSet::with_capacity(pairs.len());
        let mut b_ids = HashSet::with_capacity(pairs.len());
        for &(a, b) in pairs {
            if !a_ids.insert(a) || !b_ids.insert(b) {
                return None;
            }
        }
        Some((a_ids, b_ids))
    }
    let expected_endpoints = endpoints(expected)?;
    endpoints(replacements)?;
    Some(expected_endpoints)
}

/// Result of an atomic `confirm()` call: any record IDs that were displaced
/// from a prior pairing and now require store-side `mark_unmatched`.
///
/// `displaced_b` is the B-side that `a_id` used to be paired with.
/// `displaced_a` is the A-side that `b_id` used to be paired with.
/// Both are `None` if neither side had a prior pairing, or if the requested
/// pair was already installed (idempotent confirm).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ConfirmOutcome {
    pub displaced_b: Option<String>,
    pub displaced_a: Option<String>,
}

/// Runtime operations on a bidirectional A↔B record-pair mapping.
///
/// Thread-safe: all methods take `&self`. Implementations must be `Send + Sync`.
pub trait CrossMapOps: Send + Sync {
    /// Flush pending state to durable storage.
    ///
    /// For `MemoryCrossMap` this saves to CSV. For `SqliteCrossMap` this
    /// is a no-op (mutations are immediately durable).
    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Insert a pair unconditionally in both directions.
    fn add(&self, a_id: &str, b_id: &str) -> Result<(), CrossMapError>;

    /// Remove a pair unconditionally in both directions. No-op if absent.
    fn remove(&self, a_id: &str, b_id: &str);

    /// Remove the pair only if `a_id` is currently mapped to exactly `b_id`.
    ///
    /// Returns `true` and removes both directions if the pair matched.
    /// Returns `false` (no-op) if the pair was absent or mapped differently.
    fn remove_if_exact(&self, a_id: &str, b_id: &str) -> bool;

    /// Atomically remove the pair keyed by `a_id` and return the paired B-id.
    /// Returns `Ok(None)` only when absent; storage failures are errors.
    fn take_a(&self, a_id: &str) -> Result<Option<String>, CrossMapError>;

    /// Atomically remove the pair keyed by `b_id` and return the paired A-id.
    /// Returns `Ok(None)` only when absent; storage failures are errors.
    fn take_b(&self, b_id: &str) -> Result<Option<String>, CrossMapError>;

    /// Atomically claim the `(a_id, b_id)` pair if neither side is taken.
    ///
    /// Returns `true` on success; `false` if either side was already mapped.
    fn claim(&self, a_id: &str, b_id: &str) -> bool;

    /// Atomically install `(a_id, b_id)`, displacing any prior partners on
    /// either side. Bijection is preserved under a single lock / transaction —
    /// concurrent `claim()`/`confirm()` callers cannot race in and leave
    /// orphaned half-pairs (Constitution §3).
    ///
    /// Returns the displaced ids so the caller can update store-side
    /// metadata (e.g. `mark_unmatched`). If `a_id` is already paired with
    /// `b_id`, this is an idempotent no-op and returns `Default`.
    fn confirm(&self, a_id: &str, b_id: &str) -> Result<ConfirmOutcome, CrossMapError>;

    /// Atomically replace exactly the expected pairs with the replacements.
    ///
    /// Each slice must contain one or two pairs, with unique A and B endpoints
    /// within each slice. Every expected pair must still exist, and replacement
    /// endpoints must be free or owned by an expected pair being removed.
    /// Returns `Ok(false)` without changes for invalid input, stale expectations,
    /// or unrelated owners. Storage failures return an error without changes.
    fn replace_pairs(
        &self,
        expected: &[(&str, &str)],
        replacements: &[(&str, &str)],
    ) -> Result<bool, CrossMapError>;

    /// Atomically apply an arbitrary-size delta of canonical A/B pairs.
    ///
    /// Each slice must be separately bijective. Expected pairs must exist in
    /// both directions; replacement endpoints must be free or owned by expected
    /// pairs. Empty expected claims free endpoints; empty replacements removes
    /// pairs. Both empty, duplicates, stale expectations, or unrelated owners
    /// return `Ok(false)` without changes. Storage failures roll back and return
    /// a typed error. Validation and mutation share one lock or transaction.
    fn apply_assignment_delta(
        &self,
        expected: &[(&str, &str)],
        replacements: &[(&str, &str)],
    ) -> Result<bool, CrossMapError>;

    /// A→B lookup.
    fn get_b(&self, a_id: &str) -> Option<String>;

    /// B→A lookup.
    fn get_a(&self, b_id: &str) -> Option<String>;

    /// Whether `a_id` is currently mapped.
    fn has_a(&self, a_id: &str) -> bool;

    /// Whether `b_id` is currently mapped.
    fn has_b(&self, b_id: &str) -> bool;

    /// Number of pairs.
    fn len(&self) -> usize;

    /// Whether the map is empty.
    fn is_empty(&self) -> bool;

    /// Collect all pairs as owned `(a_id, b_id)` tuples.
    fn pairs(&self) -> Vec<(String, String)>;
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    type PairDelta<'a> = (&'a [(&'a str, &'a str)], &'a [(&'a str, &'a str)]);

    use super::CrossMapOps;

    pub(crate) fn check_assignment_delta(cm: &dyn CrossMapOps) {
        assert_eq!(
            cm.apply_assignment_delta(&[], &[]).ok(),
            Some(false),
            "empty delta is invalid"
        );
        let original = [("a1", "b1"), ("a2", "b2"), ("a3", "b3")];
        assert_eq!(
            cm.apply_assignment_delta(&[], &original).ok(),
            Some(true),
            "free group claim"
        );
        let cyclic = [("a1", "b2"), ("a2", "b3"), ("a3", "b1")];
        assert_eq!(
            cm.apply_assignment_delta(&original, &cyclic).ok(),
            Some(true),
            "cyclic replacement with overlapping endpoints"
        );
        assert_eq!(
            cm.apply_assignment_delta(&cyclic, &original).ok(),
            Some(true),
            "inverse restores cycle"
        );
        let invalid: Vec<PairDelta<'_>> = vec![
            (&[], &[("a1", "b1")]),
            (&[], &[("a1", "free-b")]),
            (&[], &[("free-a", "b1")]),
            (&[("a1", "stale")], &[]),
            (&[("stale", "b1")], &[]),
            (&[("a1", "b1"), ("missing", "missing")], &[]),
            (&[("a1", "b1"), ("a1", "b2")], &[]),
            (&[("a1", "b1"), ("a2", "b1")], &[]),
            (&[("a1", "b1")], &[("a1", "b2")]),
            (&[("a1", "b1")], &[("a2", "b1")]),
            (&[], &[("free-a", "x"), ("free-a", "y")]),
            (&[], &[("x", "free-b"), ("y", "free-b")]),
            (&original, &[("a1", "b1"), ("a1", "b1")]),
        ];
        for (expected, replacements) in invalid {
            let mut before = cm.pairs();
            before.sort();
            assert_eq!(
                cm.apply_assignment_delta(expected, replacements).ok(),
                Some(false),
                "reject {expected:?} -> {replacements:?}"
            );
            let mut after = cm.pairs();
            after.sort();
            assert_eq!(after, before, "rejection must not mutate");
            for (a, b) in &before {
                assert_eq!(
                    cm.get_a(b).as_ref(),
                    Some(a),
                    "reverse state must survive rejection"
                );
            }
        }
        assert_eq!(
            cm.apply_assignment_delta(&[("a1", "b1")], &[("a1", "new-b")])
                .ok(),
            Some(true),
            "one-for-one displacement"
        );
        assert_eq!(cm.get_a("b1"), None, "old partner is orphaned");
        assert_eq!(
            cm.apply_assignment_delta(&[("a1", "new-b")], &[("a1", "b1")])
                .ok(),
            Some(true),
            "inverse restores displaced pair"
        );
        assert_eq!(
            cm.apply_assignment_delta(&original, &[]).ok(),
            Some(true),
            "remove all"
        );
        assert!(cm.is_empty(), "all endpoints must be removed");
        for (a, b) in original {
            assert_eq!(cm.get_b(a), None, "forward endpoint freed");
            assert_eq!(cm.get_a(b), None, "reverse endpoint freed");
        }

        let owned: Vec<_> = (0..1024)
            .map(|i| (format!("a{i}"), format!("b{i}")))
            .collect();
        let expected: Vec<_> = owned
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let rotated: Vec<_> = (0..owned.len())
            .map(|i| (owned[i].0.as_str(), owned[(i + 1) % owned.len()].1.as_str()))
            .collect();
        assert_eq!(
            cm.apply_assignment_delta(&[], &expected).ok(),
            Some(true),
            "large free claim"
        );
        assert_eq!(
            cm.apply_assignment_delta(&expected, &rotated).ok(),
            Some(true),
            "large cycle has no bounded shape limit"
        );
        for (a, b) in &rotated {
            assert_eq!(cm.get_b(a).as_deref(), Some(*b), "large forward mapping");
            assert_eq!(cm.get_a(b).as_deref(), Some(*a), "large reverse mapping");
        }
        assert_eq!(
            cm.apply_assignment_delta(&rotated, &[]).ok(),
            Some(true),
            "large removal"
        );
        assert!(cm.is_empty(), "large inverse leaves no pairs");
    }

    pub(crate) fn check_assignment_delta_races(make: impl Fn() -> Arc<dyn CrossMapOps>) {
        for free_claim in [false, true] {
            for _ in 0..20 {
                let cm = make();
                let expected = if free_claim {
                    vec![]
                } else {
                    vec![("a1", "b1")]
                };
                if !free_claim {
                    assert!(cm.claim("a1", "b1"), "seed pair");
                }
                let barrier = Arc::new(Barrier::new(3));
                let worker_cm = Arc::clone(&cm);
                let worker_barrier = Arc::clone(&barrier);
                let worker = thread::spawn(move || {
                    worker_barrier.wait();
                    worker_cm.apply_assignment_delta(&expected, &[("a1", "b2"), ("a2", "b1")])
                });
                let reader_cm = Arc::clone(&cm);
                let reader_barrier = Arc::clone(&barrier);
                let reader = thread::spawn(move || {
                    reader_barrier.wait();
                    for _ in 0..50 {
                        let mut pairs = reader_cm.pairs();
                        pairs.sort();
                        let pairs: Vec<_> = pairs
                            .iter()
                            .map(|(a, b)| (a.as_str(), b.as_str()))
                            .collect();
                        let before = if free_claim {
                            vec![]
                        } else {
                            vec![("a1", "b1")]
                        };
                        let mut claimed = before.clone();
                        claimed.push(("a2", "b3"));
                        assert!(
                            pairs == before
                                || pairs == claimed
                                || pairs == vec![("a1", "b2"), ("a2", "b1")],
                            "reader must see a serial snapshot, not a partial delta: {pairs:?}"
                        );
                    }
                });
                barrier.wait();
                let claimed = cm.claim("a2", "b3");
                let applied = worker
                    .join()
                    .expect("delta worker must not panic")
                    .expect("delta storage must succeed");
                reader.join().expect("reader must not see partial state");
                assert_ne!(claimed, applied, "exactly one conflicting claim must win");
                assert_eq!(
                    cm.len(),
                    if free_claim && claimed { 1 } else { 2 },
                    "serial final pair count"
                );
                for (a, b) in cm.pairs() {
                    assert_eq!(cm.get_a(&b), Some(a), "race must preserve reverse mapping");
                }
            }
        }
    }

    pub(crate) fn check_replace_pairs(cm: &dyn CrossMapOps) {
        assert!(cm.claim("a1", "b1"), "seed pair must be free");
        let original = [("a1", "b1")];
        let split = [("a1", "b2"), ("a2", "b1")];
        assert_eq!(
            cm.replace_pairs(&original, &split).ok(),
            Some(true),
            "1->2 must succeed"
        );
        assert_eq!(cm.len(), 2, "both replacements must be installed");
        for (a, b) in split {
            assert_eq!(cm.get_b(a).as_deref(), Some(b), "forward replacement");
            assert_eq!(cm.get_a(b).as_deref(), Some(a), "reverse replacement");
        }
        assert_eq!(
            cm.replace_pairs(&split, &original).ok(),
            Some(true),
            "2->1 rollback must succeed"
        );
        assert_eq!(cm.len(), 1, "inverse must remove both split pairs");
        assert_eq!(cm.get_a("b2"), None, "inverse must free b2");
        assert_eq!(cm.get_b("a2"), None, "inverse must free a2");
        assert!(cm.claim("other-a", "other-b"), "seed unrelated owner");

        let invalid: Vec<PairDelta<'_>> = vec![
            (&original, &[("a1", "b2"), ("other-a", "b1")]),
            (&original, &[("a1", "b2"), ("a2", "other-b")]),
            (&[("a1", "stale")], &split),
            (&[("a1", "b1"), ("missing", "b9")], &split),
            (&[], &split),
            (&original, &[]),
            (&[("a1", "b1"), ("a2", "b2"), ("a3", "b3")], &split),
            (&original, &[("a1", "b1"), ("a2", "b2"), ("a3", "b3")]),
            (&[("a1", "b1"), ("a1", "b2")], &split),
            (&[("a1", "b1"), ("a2", "b1")], &split),
            (&original, &[("a1", "b2"), ("a1", "b1")]),
            (&original, &[("a1", "b1"), ("a2", "b1")]),
            (&original, &[("a1", "b1"), ("a1", "b1")]),
        ];
        let mut before = cm.pairs();
        before.sort();
        for (expected, replacements) in invalid {
            assert_eq!(
                cm.replace_pairs(expected, replacements).ok(),
                Some(false),
                "must reject {expected:?} -> {replacements:?}"
            );
            let mut after = cm.pairs();
            after.sort();
            assert_eq!(after, before, "rejection must not partially mutate pairs");
            for (a, b) in &before {
                assert_eq!(
                    cm.get_a(b).as_ref(),
                    Some(a),
                    "reverse mapping must be unchanged"
                );
            }
        }
    }

    pub(crate) fn check_replace_pairs_conflicts(make: impl Fn() -> Arc<dyn CrossMapOps>) {
        for confirm in [false, true] {
            for _ in 0..20 {
                let cm = make();
                assert!(cm.claim("a1", "b1"), "seed pair must be free");
                let barrier = Arc::new(Barrier::new(2));
                let worker_cm = Arc::clone(&cm);
                let worker_barrier = Arc::clone(&barrier);
                let worker = thread::spawn(move || {
                    worker_barrier.wait();
                    worker_cm.replace_pairs(&[("a1", "b1")], &[("a1", "b2"), ("a2", "b1")])
                });
                barrier.wait();
                let claimed = if confirm {
                    assert!(cm.confirm("a1", "b3").is_ok(), "confirm must succeed");
                    false
                } else {
                    cm.claim("a2", "b3")
                };
                let replaced = worker
                    .join()
                    .expect("worker must not panic")
                    .expect("replace must not fail");
                if confirm {
                    assert_eq!(cm.get_b("a1").as_deref(), Some("b3"), "confirm must win a1");
                    assert_eq!(
                        cm.has_a("a2"),
                        replaced,
                        "split exists only if replacement ran first"
                    );
                    assert_eq!(
                        cm.len(),
                        if replaced { 2 } else { 1 },
                        "only a serial outcome is allowed"
                    );
                } else {
                    assert_ne!(
                        claimed, replaced,
                        "exactly one conflicting operation must win"
                    );
                    assert_eq!(
                        cm.get_b("a1").as_deref(),
                        Some(if replaced { "b2" } else { "b1" }),
                        "no partial replacement"
                    );
                    assert_eq!(cm.len(), 2, "either serial outcome has two pairs");
                }
                for (a, b) in cm.pairs() {
                    assert_eq!(
                        cm.get_a(&b),
                        Some(a),
                        "concurrent operations must preserve bijection"
                    );
                }
            }
        }
    }
}
