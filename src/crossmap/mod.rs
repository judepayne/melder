//! CrossMap: bidirectional record-pair mapping with CSV persistence.
//!
//! Internally backed by a single `RwLock` over two `HashMap`s, so every
//! multi-step operation (check A, check B, write both) is atomic under one
//! lock acquisition.  This gives us cross-map atomicity that a per-map scheme
//! (e.g. two DashMaps) cannot provide without deadlock risk.
//!
//! ## Why not two DashMaps?
//!
//! DashMap's shard locks are scoped to a single map.  To atomically check
//! *both* `a_to_b` and `b_to_a` before inserting, you would need to hold a
//! shard lock from each map simultaneously — which risks deadlock if two
//! threads acquire them in opposite order.  A single `RwLock<CrossMapInner>`
//! avoids this entirely.
//!
//! ## Performance
//!
//! Benchmarking showed CrossMap operations are nanosecond-level and not on
//! the critical path (encoding dominates at ~4–6 ms per request).  The coarser
//! `RwLock` has no measurable throughput impact.

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::error::CrossMapError;

/// Cross-platform rename that replaces the destination if it exists.
///
/// On Unix `fs::rename` atomically replaces the target.  On Windows it fails
/// if the destination already exists, so we remove-then-rename (tiny window
/// of non-atomicity, acceptable for crossmap flush).
fn rename_replacing(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        std::fs::rename(from, to)
    }
    #[cfg(not(unix))]
    {
        let _ = std::fs::remove_file(to);
        std::fs::rename(from, to)
    }
}

// ── private inner state ──────────────────────────────────────────────────────

#[derive(Debug)]
struct CrossMapInner {
    a_to_b: HashMap<String, String>,
    b_to_a: HashMap<String, String>,
}

impl CrossMapInner {
    fn new() -> Self {
        Self {
            a_to_b: HashMap::new(),
            b_to_a: HashMap::new(),
        }
    }
}

// ── public API ───────────────────────────────────────────────────────────────

/// Bidirectional mapping between A and B record IDs.
///
/// Thread-safe: all methods take `&self`. No external lock needed.
#[derive(Debug)]
pub struct CrossMap {
    inner: RwLock<CrossMapInner>,
}

impl CrossMap {
    /// Create an empty CrossMap.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(CrossMapInner::new()),
        }
    }

    // ── writes ────────────────────────────────────────────────────────────────

    /// Insert a pair unconditionally in both directions.
    ///
    /// Overwrites any existing entry for either ID.  Use `claim()` in the
    /// live scoring pipeline when you want to back off if either side is taken.
    pub fn add(&self, a_id: &str, b_id: &str) {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.insert(a_id.to_string(), b_id.to_string());
        g.b_to_a.insert(b_id.to_string(), a_id.to_string());
    }

    /// Remove a pair unconditionally in both directions.  No-op if absent.
    pub fn remove(&self, a_id: &str, b_id: &str) {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.remove(a_id);
        g.b_to_a.remove(b_id);
    }

    /// Remove the pair only if `a_id` is currently mapped to exactly `b_id`.
    ///
    /// Returns `true` and removes both directions if the pair matched.
    /// Returns `false` (no-op) if the pair was absent or mapped differently.
    pub fn remove_if_exact(&self, a_id: &str, b_id: &str) -> bool {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if g.a_to_b.get(a_id).map(String::as_str) == Some(b_id) {
            g.a_to_b.remove(a_id);
            g.b_to_a.remove(b_id);
            true
        } else {
            false
        }
    }

    /// Atomically remove the pair keyed by `a_id` and return the paired B-id.
    ///
    /// The read and remove happen under a single write lock, eliminating the
    /// TOCTOU race of `get_b` followed by a separate `remove`.
    /// Returns `None` if `a_id` is not currently mapped.
    pub fn take_a(&self, a_id: &str) -> Option<String> {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(b_id) = g.a_to_b.remove(a_id) {
            g.b_to_a.remove(&b_id);
            Some(b_id)
        } else {
            None
        }
    }

    /// Atomically remove the pair keyed by `b_id` and return the paired A-id.
    ///
    /// Symmetric to `take_a`.  Returns `None` if `b_id` is not currently mapped.
    pub fn take_b(&self, b_id: &str) -> Option<String> {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(a_id) = g.b_to_a.remove(b_id) {
            g.a_to_b.remove(&a_id);
            Some(a_id)
        } else {
            None
        }
    }

    /// Attempt to atomically claim the `(a_id, b_id)` pair.
    ///
    /// Under a single write lock: checks that *neither* `a_id` nor `b_id` is
    /// already mapped, then inserts both directions.
    ///
    /// Returns `true` on success; `false` if either side was already taken.
    pub fn claim(&self, a_id: &str, b_id: &str) -> bool {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if g.a_to_b.contains_key(a_id) || g.b_to_a.contains_key(b_id) {
            return false;
        }
        g.a_to_b.insert(a_id.to_string(), b_id.to_string());
        g.b_to_a.insert(b_id.to_string(), a_id.to_string());
        true
    }

    // ── reads ─────────────────────────────────────────────────────────────────

    /// A→B lookup.
    pub fn get_b(&self, a_id: &str) -> Option<String> {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.get(a_id).cloned()
    }

    /// B→A lookup.
    pub fn get_a(&self, b_id: &str) -> Option<String> {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.b_to_a.get(b_id).cloned()
    }

    /// Whether `a_id` is currently mapped.
    pub fn has_a(&self, a_id: &str) -> bool {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.contains_key(a_id)
    }

    /// Whether `b_id` is currently mapped.
    pub fn has_b(&self, b_id: &str) -> bool {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.b_to_a.contains_key(b_id)
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.is_empty()
    }

    /// Collect all pairs as owned `(a_id, b_id)` tuples.
    pub fn pairs(&self) -> Vec<(String, String)> {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b
            .iter()
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect()
    }

    // ── persistence ───────────────────────────────────────────────────────────

    /// Load a CrossMap from a CSV file.
    ///
    /// The CSV must have headers containing `a_field` and `b_field` column
    /// names.  A missing file or header-only file produces an empty CrossMap.
    pub fn load(path: &Path, a_field: &str, b_field: &str) -> Result<Self, CrossMapError> {
        let cm = Self::new();

        if !path.exists() {
            return Ok(cm);
        }

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;

        let headers: Vec<String> = rdr
            .headers()?
            .iter()
            .map(|h| h.trim().to_string())
            .collect();

        let a_idx = headers.iter().position(|h| h == a_field);
        let b_idx = headers.iter().position(|h| h == b_field);

        let (a_idx, b_idx) = match (a_idx, b_idx) {
            (Some(a), Some(b)) => (a, b),
            _ => return Ok(cm),
        };

        for result in rdr.records() {
            let row = result?;
            let a_id = row.get(a_idx).unwrap_or("").trim().to_string();
            let b_id = row.get(b_idx).unwrap_or("").trim().to_string();
            if !a_id.is_empty() && !b_id.is_empty() {
                cm.add(&a_id, &b_id);
            }
        }

        Ok(cm)
    }

    /// Save the CrossMap to a CSV file (atomic write via temp + rename).
    pub fn save(&self, path: &Path, a_field: &str, b_field: &str) -> Result<(), CrossMapError> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let temp_path = path.with_extension("tmp");
        {
            let mut wtr = csv::Writer::from_path(&temp_path)?;
            wtr.write_record([a_field, b_field])?;
            for (a_id, b_id) in self.pairs() {
                wtr.write_record([a_id.as_str(), b_id.as_str()])?;
            }
            wtr.flush()?;
        }

        rename_replacing(&temp_path, path)?;
        Ok(())
    }
}

impl Default for CrossMap {
    fn default() -> Self {
        Self::new()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn add_and_lookup() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
        assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
        assert_eq!(cm.get_a("B-2"), Some("A-2".to_string()));
        assert!(cm.has_a("A-1"));
        assert!(cm.has_b("B-1"));
        assert!(!cm.has_a("A-3"));
        assert_eq!(cm.len(), 2);
    }

    #[test]
    fn remove_pair() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        cm.remove("A-1", "B-1");
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 1);
        assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
    }

    #[test]
    fn remove_if_exact_matches() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        assert!(cm.remove_if_exact("A-1", "B-1"));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
    }

    #[test]
    fn remove_if_exact_wrong_pair_is_noop() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        // Wrong b_id — should be a no-op
        assert!(!cm.remove_if_exact("A-1", "B-99"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn take_a_removes_both_directions() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        assert_eq!(cm.take_a("A-1"), Some("B-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_b_removes_both_directions() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        assert_eq!(cm.take_b("B-1"), Some("A-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_a_missing_returns_none() {
        let cm = CrossMap::new();
        assert_eq!(cm.take_a("A-99"), None);
    }

    #[test]
    fn claim_vacant_succeeds() {
        let cm = CrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
    }

    #[test]
    fn claim_b_occupied_fails() {
        let cm = CrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        // Different A, same B — B is already taken
        assert!(!cm.claim("A-2", "B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn claim_a_occupied_fails() {
        let cm = CrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        // Same A, different B — A is already mapped
        assert!(!cm.claim("A-1", "B-2"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert!(cm.get_a("B-2").is_none());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn claim_then_break_then_reclaim() {
        let cm = CrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        cm.remove("A-1", "B-1");
        assert!(cm.claim("A-2", "B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-2".to_string()));
    }

    #[test]
    fn concurrent_claim_only_one_wins() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let cm = Arc::new(CrossMap::new());
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        for i in 0..10 {
            let cm = Arc::clone(&cm);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                cm.claim(&format!("A-{}", i), "B-contested")
            }));
        }

        let wins: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            wins.iter().filter(|&&w| w).count(),
            1,
            "exactly one thread should win the claim"
        );
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn concurrent_claim_same_a_only_one_wins() {
        // Regression test for the A-side race that the old DashMap design
        // could not prevent: two threads both try to claim "A-1" for different
        // B records simultaneously.
        use std::sync::{Arc, Barrier};
        use std::thread;

        let cm = Arc::new(CrossMap::new());
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        for i in 0..10 {
            let cm = Arc::clone(&cm);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                cm.claim("A-contested", &format!("B-{}", i))
            }));
        }

        let wins: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            wins.iter().filter(|&&w| w).count(),
            1,
            "exactly one thread should win when A is contested"
        );
        assert_eq!(cm.len(), 1);
        // Both directions must be consistent
        let b = cm.get_b("A-contested").expect("A-contested must be mapped");
        assert_eq!(cm.get_a(&b), Some("A-contested".to_string()));
    }

    #[test]
    fn pairs_returns_all() {
        let cm = CrossMap::new();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        let mut pairs = cm.pairs();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("A-1".to_string(), "B-1".to_string()),
                ("A-2".to_string(), "B-2".to_string()),
            ]
        );
    }

    #[test]
    fn save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("crossmap.csv");

        let cm = CrossMap::new();
        for i in 0..5 {
            cm.add(&format!("A-{}", i), &format!("B-{}", i));
        }
        cm.save(&path, "entity_id", "counterparty_id").unwrap();

        let loaded = CrossMap::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(loaded.len(), 5);
        for i in 0..5 {
            assert_eq!(loaded.get_b(&format!("A-{}", i)), Some(format!("B-{}", i)));
        }
    }

    #[test]
    fn load_missing_file() {
        let cm = CrossMap::load(
            &Path::new("nonexistent_dir_for_test").join("crossmap.csv"),
            "a",
            "b",
        )
        .unwrap();
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn load_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.csv");
        std::fs::write(&path, "entity_id,counterparty_id\n").unwrap();

        let cm = CrossMap::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(cm.len(), 0);
    }
}
