//! `MemoryCrossMap`: in-memory bidirectional record-pair mapping with CSV persistence.
//!
//! Internally backed by a single `RwLock` over two `HashMap`s, so every
//! multi-step operation (check A, check B, write both) is atomic under one
//! lock acquisition.
//!
//! ## Why not two DashMaps?
//!
//! DashMap's shard locks are scoped to a single map.  To atomically check
//! *both* `a_to_b` and `b_to_a` before inserting, you would need to hold a
//! shard lock from each map simultaneously — which risks deadlock if two
//! threads acquire them in opposite order.  A single `RwLock<Inner>` avoids
//! this entirely.
//!
//! ## Performance
//!
//! Benchmarking showed CrossMap operations are nanosecond-level and not on
//! the critical path (encoding dominates at ~4–6 ms per request).  The coarser
//! `RwLock` has no measurable throughput impact.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::error::CrossMapError;

use super::CrossMapOps;

use crate::util::rename_replacing;

// ── private inner state ──────────────────────────────────────────────────────

#[derive(Debug)]
struct Inner {
    a_to_b: HashMap<String, String>,
    b_to_a: HashMap<String, String>,
}

impl Inner {
    fn new() -> Self {
        Self {
            a_to_b: HashMap::new(),
            b_to_a: HashMap::new(),
        }
    }
}

// ── public API ───────────────────────────────────────────────────────────────

/// In-memory bidirectional mapping between A and B record IDs.
///
/// Thread-safe: all methods take `&self`. No external lock needed.
/// Persistence via `load()` / `save()` (CSV format).
/// Persistence config for CSV flush. Set via `set_flush_path()` after
/// construction. When `None`, `flush()` is a no-op (batch mode).
#[derive(Debug, Clone)]
struct FlushConfig {
    path: PathBuf,
    a_field: String,
    b_field: String,
}

#[derive(Debug)]
pub struct MemoryCrossMap {
    inner: RwLock<Inner>,
    flush_config: RwLock<Option<FlushConfig>>,
}

impl MemoryCrossMap {
    /// Create an empty MemoryCrossMap.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner::new()),
            flush_config: RwLock::new(None),
        }
    }

    /// Configure the CSV flush path and field names.
    ///
    /// Must be called before `flush()` will write anything. This is set
    /// by the live-mode startup code after loading or creating the crossmap.
    pub fn set_flush_path(&self, path: &Path, a_field: &str, b_field: &str) {
        let mut cfg = self.flush_config.write().unwrap_or_else(|e| e.into_inner());
        *cfg = Some(FlushConfig {
            path: path.to_path_buf(),
            a_field: a_field.to_string(),
            b_field: b_field.to_string(),
        });
    }

    // ── persistence (inherent, not in trait) ──────────────────────────────────

    /// Load a MemoryCrossMap from a CSV file.
    ///
    /// The CSV must have headers containing `a_field` and `b_field` column
    /// names.  A missing file or header-only file produces an empty map.
    pub fn load(path: &Path, a_field: &str, b_field: &str) -> Result<Self, CrossMapError> {
        let cm = Self::new();

        // Clean up stale temp file from a prior interrupted flush.
        let temp_path = path.with_extension("tmp");
        if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }

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

    /// Save the MemoryCrossMap to a CSV file (atomic write via temp + rename).
    pub fn save(&self, path: &Path, a_field: &str, b_field: &str) -> Result<(), CrossMapError> {
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
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

impl Default for MemoryCrossMap {
    fn default() -> Self {
        Self::new()
    }
}

// ── CrossMapOps implementation ───────────────────────────────────────────────

impl CrossMapOps for MemoryCrossMap {
    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let cfg = self.flush_config.read().unwrap_or_else(|e| e.into_inner());
        if let Some(ref fc) = *cfg {
            self.save(&fc.path, &fc.a_field, &fc.b_field)?;
        }
        Ok(())
    }

    fn add(&self, a_id: &str, b_id: &str) {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.insert(a_id.to_string(), b_id.to_string());
        g.b_to_a.insert(b_id.to_string(), a_id.to_string());
    }

    fn remove(&self, a_id: &str, b_id: &str) {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        // Only remove if the pair actually matches — prevents corrupting
        // two unrelated mappings when called with a mismatched pair.
        // Consistent with SqliteCrossMap::remove which uses
        // `WHERE a_id = ?1 AND b_id = ?2`.
        if g.a_to_b.get(a_id).map(String::as_str) == Some(b_id) {
            g.a_to_b.remove(a_id);
            g.b_to_a.remove(b_id);
        }
    }

    fn remove_if_exact(&self, a_id: &str, b_id: &str) -> bool {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if g.a_to_b.get(a_id).map(String::as_str) == Some(b_id) {
            g.a_to_b.remove(a_id);
            g.b_to_a.remove(b_id);
            true
        } else {
            false
        }
    }

    fn take_a(&self, a_id: &str) -> Option<String> {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(b_id) = g.a_to_b.remove(a_id) {
            g.b_to_a.remove(&b_id);
            Some(b_id)
        } else {
            None
        }
    }

    fn take_b(&self, b_id: &str) -> Option<String> {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(a_id) = g.b_to_a.remove(b_id) {
            g.a_to_b.remove(&a_id);
            Some(a_id)
        } else {
            None
        }
    }

    fn claim(&self, a_id: &str, b_id: &str) -> bool {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if g.a_to_b.contains_key(a_id) || g.b_to_a.contains_key(b_id) {
            return false;
        }
        g.a_to_b.insert(a_id.to_string(), b_id.to_string());
        g.b_to_a.insert(b_id.to_string(), a_id.to_string());
        true
    }

    fn get_b(&self, a_id: &str) -> Option<String> {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.get(a_id).cloned()
    }

    fn get_a(&self, b_id: &str) -> Option<String> {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.b_to_a.get(b_id).cloned()
    }

    fn has_a(&self, a_id: &str) -> bool {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.contains_key(a_id)
    }

    fn has_b(&self, b_id: &str) -> bool {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.b_to_a.contains_key(b_id)
    }

    fn len(&self) -> usize {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.len()
    }

    fn is_empty(&self) -> bool {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b.is_empty()
    }

    fn pairs(&self) -> Vec<(String, String)> {
        let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
        g.a_to_b
            .iter()
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn add_and_lookup() {
        let cm = MemoryCrossMap::new();
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
        let cm = MemoryCrossMap::new();
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
        let cm = MemoryCrossMap::new();
        cm.add("A-1", "B-1");
        assert!(cm.remove_if_exact("A-1", "B-1"));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
    }

    #[test]
    fn remove_if_exact_wrong_pair_is_noop() {
        let cm = MemoryCrossMap::new();
        cm.add("A-1", "B-1");
        // Wrong b_id — should be a no-op
        assert!(!cm.remove_if_exact("A-1", "B-99"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn take_a_removes_both_directions() {
        let cm = MemoryCrossMap::new();
        cm.add("A-1", "B-1");
        assert_eq!(cm.take_a("A-1"), Some("B-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_b_removes_both_directions() {
        let cm = MemoryCrossMap::new();
        cm.add("A-1", "B-1");
        assert_eq!(cm.take_b("B-1"), Some("A-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_a_missing_returns_none() {
        let cm = MemoryCrossMap::new();
        assert_eq!(cm.take_a("A-99"), None);
    }

    #[test]
    fn claim_vacant_succeeds() {
        let cm = MemoryCrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
    }

    #[test]
    fn claim_b_occupied_fails() {
        let cm = MemoryCrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        // Different A, same B — B is already taken
        assert!(!cm.claim("A-2", "B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn claim_a_occupied_fails() {
        let cm = MemoryCrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        // Same A, different B — A is already mapped
        assert!(!cm.claim("A-1", "B-2"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert!(cm.get_a("B-2").is_none());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn claim_then_break_then_reclaim() {
        let cm = MemoryCrossMap::new();
        assert!(cm.claim("A-1", "B-1"));
        cm.remove("A-1", "B-1");
        assert!(cm.claim("A-2", "B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-2".to_string()));
    }

    #[test]
    fn concurrent_claim_only_one_wins() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let cm = Arc::new(MemoryCrossMap::new());
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
        use std::sync::{Arc, Barrier};
        use std::thread;

        let cm = Arc::new(MemoryCrossMap::new());
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
        let cm = MemoryCrossMap::new();
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

        let cm = MemoryCrossMap::new();
        for i in 0..5 {
            cm.add(&format!("A-{}", i), &format!("B-{}", i));
        }
        cm.save(&path, "entity_id", "counterparty_id").unwrap();

        let loaded = MemoryCrossMap::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(loaded.len(), 5);
        for i in 0..5 {
            assert_eq!(loaded.get_b(&format!("A-{}", i)), Some(format!("B-{}", i)));
        }
    }

    #[test]
    fn load_missing_file() {
        let cm = MemoryCrossMap::load(
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

        let cm = MemoryCrossMap::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(cm.len(), 0);
    }
}
