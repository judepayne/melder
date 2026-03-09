//! CrossMap: bidirectional record-pair mapping with CSV persistence.
//!
//! Maintains `a_to_b` and `b_to_a` maps for O(1) bidirectional lookup.

use std::collections::HashMap;
use std::path::Path;

use crate::error::CrossMapError;

/// Bidirectional mapping between A and B record IDs.
#[derive(Debug, Clone)]
pub struct CrossMap {
    a_to_b: HashMap<String, String>,
    b_to_a: HashMap<String, String>,
}

impl CrossMap {
    /// Create an empty CrossMap.
    pub fn new() -> Self {
        Self {
            a_to_b: HashMap::new(),
            b_to_a: HashMap::new(),
        }
    }

    /// Add a pair (both directions).
    pub fn add(&mut self, a_id: &str, b_id: &str) {
        self.a_to_b.insert(a_id.to_string(), b_id.to_string());
        self.b_to_a.insert(b_id.to_string(), a_id.to_string());
    }

    /// Remove a pair (both directions).
    pub fn remove(&mut self, a_id: &str, b_id: &str) {
        self.a_to_b.remove(a_id);
        self.b_to_a.remove(b_id);
    }

    /// A→B lookup.
    pub fn get_b(&self, a_id: &str) -> Option<&str> {
        self.a_to_b.get(a_id).map(|s| s.as_str())
    }

    /// B→A lookup.
    pub fn get_a(&self, b_id: &str) -> Option<&str> {
        self.b_to_a.get(b_id).map(|s| s.as_str())
    }

    /// Check if an A-side ID is mapped.
    pub fn has_a(&self, a_id: &str) -> bool {
        self.a_to_b.contains_key(a_id)
    }

    /// Check if a B-side ID is mapped.
    pub fn has_b(&self, b_id: &str) -> bool {
        self.b_to_a.contains_key(b_id)
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        self.a_to_b.len()
    }

    /// Whether the crossmap is empty.
    pub fn is_empty(&self) -> bool {
        self.a_to_b.is_empty()
    }

    /// Iterate over all pairs as (a_id, b_id).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.a_to_b.iter().map(|(a, b)| (a.as_str(), b.as_str()))
    }

    /// Load a CrossMap from a CSV file.
    ///
    /// The CSV must have headers containing `a_field` and `b_field` column names.
    /// An empty file (header-only) is valid and produces an empty CrossMap.
    pub fn load(path: &Path, a_field: &str, b_field: &str) -> Result<Self, CrossMapError> {
        let mut cm = Self::new();

        if !path.exists() {
            // File doesn't exist — return empty CrossMap
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
            _ => {
                // Headers don't match — return empty (might be empty file)
                return Ok(cm);
            }
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
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Write to temp file
        let temp_path = path.with_extension("tmp");
        {
            let mut wtr = csv::Writer::from_path(&temp_path)?;
            wtr.write_record([a_field, b_field])?;
            for (a_id, b_id) in &self.a_to_b {
                wtr.write_record([a_id.as_str(), b_id.as_str()])?;
            }
            wtr.flush()?;
        }

        // Atomic rename
        std::fs::rename(&temp_path, path)?;

        Ok(())
    }
}

impl Default for CrossMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn add_and_lookup() {
        let mut cm = CrossMap::new();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        assert_eq!(cm.get_b("A-1"), Some("B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-1"));
        assert_eq!(cm.get_b("A-2"), Some("B-2"));
        assert_eq!(cm.get_a("B-2"), Some("A-2"));
        assert!(cm.has_a("A-1"));
        assert!(cm.has_b("B-1"));
        assert!(!cm.has_a("A-3"));
        assert_eq!(cm.len(), 2);
    }

    #[test]
    fn remove_pair() {
        let mut cm = CrossMap::new();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        cm.remove("A-1", "B-1");
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 1);
        assert_eq!(cm.get_b("A-2"), Some("B-2"));
    }

    #[test]
    fn save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("crossmap.csv");

        let mut cm = CrossMap::new();
        for i in 0..5 {
            cm.add(&format!("A-{}", i), &format!("B-{}", i));
        }
        cm.save(&path, "entity_id", "counterparty_id").unwrap();

        let loaded = CrossMap::load(&path, "entity_id", "counterparty_id").unwrap();
        assert_eq!(loaded.len(), 5);
        for i in 0..5 {
            assert_eq!(
                loaded.get_b(&format!("A-{}", i)),
                Some(format!("B-{}", i)).as_deref()
            );
        }
    }

    #[test]
    fn load_missing_file() {
        let cm = CrossMap::load(Path::new("/nonexistent/crossmap.csv"), "a", "b").unwrap();
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

    #[test]
    fn iter_pairs() {
        let mut cm = CrossMap::new();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        let pairs: Vec<_> = cm.iter().collect();
        assert_eq!(pairs.len(), 2);
    }
}
