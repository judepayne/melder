//! Known non-matching pairs excluded from scoring.
//!
//! Pairs are loaded from CSV at startup and can be added/removed at runtime.
//! Both orderings of each pair are stored so lookups work regardless of
//! which side is the query. Persistence follows the same pattern as CrossMap:
//! load from CSV at startup, flush to CSV on shutdown via atomic temp+rename.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use tracing::info;

use crate::error::CrossMapError;
use crate::util::rename_replacing;

// ── flush config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct FlushConfig {
    path: PathBuf,
    a_field: String,
    b_field: String,
}

// ── public API ───────────────────────────────────────────────────────────────

/// Thread-safe set of excluded (known non-matching) record pairs.
///
/// Internally stores both orderings `(a, b)` and `(b, a)` so `contains()`
/// works regardless of query direction. All methods take `&self`.
#[derive(Debug)]
pub struct Exclusions {
    pairs: RwLock<HashSet<(String, String)>>,
    flush_config: RwLock<Option<FlushConfig>>,
}

impl Exclusions {
    /// Create an empty exclusions set.
    pub fn new() -> Self {
        Self {
            pairs: RwLock::new(HashSet::new()),
            flush_config: RwLock::new(None),
        }
    }

    /// Load exclusions from a CSV file.
    ///
    /// The CSV must have headers containing `a_field` and `b_field` column
    /// names. A missing file or header-only file produces an empty set.
    pub fn load(path: &Path, a_field: &str, b_field: &str) -> Result<Self, CrossMapError> {
        let ex = Self::new();

        if !path.exists() {
            return Ok(ex);
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
            _ => return Ok(ex),
        };

        let mut count = 0usize;
        for result in rdr.records() {
            let row = result?;
            let a_id = row.get(a_idx).unwrap_or("").trim().to_string();
            let b_id = row.get(b_idx).unwrap_or("").trim().to_string();
            if !a_id.is_empty() && !b_id.is_empty() {
                ex.add(&a_id, &b_id);
                count += 1;
            }
        }

        if count > 0 {
            info!(pairs = count, "loaded exclusions");
        }

        Ok(ex)
    }

    /// Configure the CSV flush path and field names.
    ///
    /// Must be called before `flush()` will write anything.
    pub fn set_flush_path(&self, path: &Path, a_field: &str, b_field: &str) {
        let mut cfg = self.flush_config.write().unwrap_or_else(|e| e.into_inner());
        *cfg = Some(FlushConfig {
            path: path.to_path_buf(),
            a_field: a_field.to_string(),
            b_field: b_field.to_string(),
        });
    }

    /// Save exclusions to a CSV file (atomic write via temp + rename).
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

    /// Flush to the configured path. No-op if no flush path is set.
    pub fn flush(&self) -> Result<(), CrossMapError> {
        let cfg = self.flush_config.read().unwrap_or_else(|e| e.into_inner());
        if let Some(ref fc) = *cfg {
            self.save(&fc.path, &fc.a_field, &fc.b_field)?;
        }
        Ok(())
    }

    /// Add an excluded pair. Inserts both orderings.
    pub fn add(&self, id_1: &str, id_2: &str) {
        let mut g = self.pairs.write().unwrap_or_else(|e| e.into_inner());
        g.insert((id_1.to_string(), id_2.to_string()));
        g.insert((id_2.to_string(), id_1.to_string()));
    }

    /// Remove an excluded pair. Removes both orderings.
    pub fn remove(&self, id_1: &str, id_2: &str) {
        let mut g = self.pairs.write().unwrap_or_else(|e| e.into_inner());
        g.remove(&(id_1.to_string(), id_2.to_string()));
        g.remove(&(id_2.to_string(), id_1.to_string()));
    }

    /// Check whether a pair is excluded.
    pub fn contains(&self, id_1: &str, id_2: &str) -> bool {
        let g = self.pairs.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return false;
        }
        g.contains(&(id_1.to_string(), id_2.to_string()))
    }

    /// Number of unique excluded pairs (not counting both orderings).
    pub fn len(&self) -> usize {
        let g = self.pairs.read().unwrap_or_else(|e| e.into_inner());
        g.len() / 2
    }

    /// Whether the exclusion set is empty.
    pub fn is_empty(&self) -> bool {
        let g = self.pairs.read().unwrap_or_else(|e| e.into_inner());
        g.is_empty()
    }

    /// Return all unique pairs in canonical (sorted) order for serialization.
    ///
    /// Each pair appears once — the ordering where `a < b` lexicographically.
    pub fn pairs(&self) -> Vec<(String, String)> {
        let g = self.pairs.read().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<(String, String)> = g.iter().filter(|(a, b)| a <= b).cloned().collect();
        out.sort();
        out
    }
}

impl Default for Exclusions {
    fn default() -> Self {
        Self::new()
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn add_and_contains() {
        let ex = Exclusions::new();
        ex.add("A-1", "B-1");
        assert!(ex.contains("A-1", "B-1"), "forward lookup");
        assert!(ex.contains("B-1", "A-1"), "reverse lookup");
        assert!(!ex.contains("A-1", "B-2"), "unrelated pair");
    }

    #[test]
    fn remove_pair() {
        let ex = Exclusions::new();
        ex.add("A-1", "B-1");
        ex.remove("A-1", "B-1");
        assert!(!ex.contains("A-1", "B-1"), "should be removed");
        assert!(!ex.contains("B-1", "A-1"), "reverse should be removed");
    }

    #[test]
    fn len_counts_unique_pairs() {
        let ex = Exclusions::new();
        ex.add("A-1", "B-1");
        ex.add("A-2", "B-2");
        ex.add("A-3", "B-3");
        assert_eq!(ex.len(), 3, "3 unique pairs");
    }

    #[test]
    fn pairs_deduplicates() {
        let ex = Exclusions::new();
        ex.add("A-1", "B-1");
        ex.add("A-2", "B-2");
        let pairs = ex.pairs();
        assert_eq!(pairs.len(), 2, "each pair appears once");
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("exclusions.csv");

        let ex = Exclusions::new();
        ex.add("A-1", "B-1");
        ex.add("A-2", "B-2");
        ex.add("A-3", "B-3");
        ex.save(&path, "entity_id", "counterparty_id").unwrap();

        let loaded = Exclusions::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(loaded.len(), 3, "all pairs loaded");
        assert!(loaded.contains("A-1", "B-1"));
        assert!(loaded.contains("B-2", "A-2"), "reverse lookup works");
    }

    #[test]
    fn load_missing_file() {
        let ex = Exclusions::load(
            &Path::new("nonexistent_dir_for_test").join("exclusions.csv"),
            "a",
            "b",
        )
        .unwrap();
        assert_eq!(ex.len(), 0);
    }

    #[test]
    fn load_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.csv");
        std::fs::write(&path, "entity_id,counterparty_id\n").unwrap();

        let ex = Exclusions::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(ex.len(), 0);
    }

    #[test]
    fn load_missing_columns() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.csv");
        std::fs::write(&path, "foo,bar\na,b\n").unwrap();

        let ex = Exclusions::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(ex.len(), 0, "missing columns produce empty set");
    }

    #[test]
    fn flush_with_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("exclusions.csv");

        let ex = Exclusions::new();
        ex.set_flush_path(&path, "a_id", "b_id");
        ex.add("X", "Y");
        ex.flush().unwrap();

        let loaded = Exclusions::load(&path, "a_id", "b_id").unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains("X", "Y"));
    }

    #[test]
    fn flush_noop_without_config() {
        let ex = Exclusions::new();
        ex.add("X", "Y");
        // Should not panic or error
        ex.flush().unwrap();
    }
}
