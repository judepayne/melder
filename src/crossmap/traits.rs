//! `CrossMapOps` trait: runtime interface for bidirectional record-pair mapping.
//!
//! Persistence details differ between backends: `MemoryCrossMap` flushes to
//! CSV via `flush()`, while `SqliteCrossMap` is write-through (flush is a
//! no-op). The trait's `flush()` method abstracts this.

use crate::error::CrossMapError;

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
    fn take_a(&self, a_id: &str) -> Option<String>;

    /// Atomically remove the pair keyed by `b_id` and return the paired A-id.
    fn take_b(&self, b_id: &str) -> Option<String>;

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
