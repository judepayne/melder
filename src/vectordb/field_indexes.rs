//! Per-field vector index collection.
//!
//! Wraps a `HashMap<String, Box<dyn VectorDB>>` keyed by field_key
//! (e.g. `"name_a/name_b"`). Each VectorDB instance stores vectors for
//! one embedding match field, for one side (A or B). Block routing is
//! handled internally by the VectorDB implementation (usearch) or
//! ignored (flat).

use std::collections::HashMap;
use std::path::Path;

use crate::models::{Record, Side};

use super::{VectorDB, VectorDBError};

/// Collection of per-field vector indexes for one side of the match.
///
/// Each entry maps a field_key (`"field_a/field_b"`) to a `Box<dyn VectorDB>`
/// holding all records' embedding vectors for that field. The backend
/// (flat or usearch) is determined at construction time.
pub struct FieldIndexes {
    indexes: HashMap<String, Box<dyn VectorDB>>,
    dim: usize,
}

impl std::fmt::Debug for FieldIndexes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldIndexes")
            .field("dim", &self.dim)
            .field("fields", &self.indexes.keys().collect::<Vec<_>>())
            .field("total_vecs", &self.len())
            .finish()
    }
}

// Safety: VectorDB is Send + Sync, HashMap is Send + Sync when values are.
unsafe impl Send for FieldIndexes {}
unsafe impl Sync for FieldIndexes {}

impl FieldIndexes {
    /// Create an empty collection with no field indexes.
    pub fn new(dim: usize) -> Self {
        Self {
            indexes: HashMap::new(),
            dim,
        }
    }

    /// Add a VectorDB instance for a field.
    pub fn insert_index(&mut self, field_key: String, index: Box<dyn VectorDB>) {
        self.indexes.insert(field_key, index);
    }

    /// Insert or replace a vector for a record in a specific field's index.
    ///
    /// `record` and `side` are passed through to the VectorDB for
    /// block-aware backends (usearch).
    pub fn upsert(
        &self,
        record_id: &str,
        field_key: &str,
        vec: &[f32],
        record: &Record,
        side: Side,
    ) -> Result<(), VectorDBError> {
        if let Some(index) = self.indexes.get(field_key) {
            index.upsert(record_id, vec, record, side)
        } else {
            // No index for this field key — silently skip (field may not
            // be an embedding field).
            Ok(())
        }
    }

    /// Retrieve the vector for a record from a specific field's index.
    pub fn get_vec(&self, record_id: &str, field_key: &str) -> Option<Vec<f32>> {
        self.indexes
            .get(field_key)
            .and_then(|idx| idx.get(record_id).ok().flatten())
    }

    /// Remove a record from all field indexes.
    pub fn remove_record(&self, record_id: &str) {
        for index in self.indexes.values() {
            let _ = index.remove(record_id);
        }
    }

    /// Total number of vectors across all field indexes.
    pub fn len(&self) -> usize {
        self.indexes.values().map(|idx| idx.len()).sum()
    }

    /// Whether all field indexes are empty.
    pub fn is_empty(&self) -> bool {
        self.indexes.values().all(|idx| idx.is_empty())
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the field keys in this collection.
    pub fn field_keys(&self) -> Vec<&str> {
        self.indexes.keys().map(|k| k.as_str()).collect()
    }

    /// Get a reference to the underlying VectorDB for a field.
    pub fn get_index(&self, field_key: &str) -> Option<&dyn VectorDB> {
        self.indexes.get(field_key).map(|b| &**b)
    }

    /// Save all field indexes to disk.
    ///
    /// Each field index is saved to a path derived from `base_path` with
    /// the field_key appended (slash replaced with `__` for filesystem
    /// safety). For example, base_path `cache/a_10000` and field_key
    /// `name_a/name_b` saves to `cache/a_10000.name_a__name_b.index`.
    pub fn save_all(&self, base_path: &str) -> Result<(), VectorDBError> {
        for (field_key, index) in &self.indexes {
            let path = field_cache_path(base_path, field_key);
            if let Some(parent) = Path::new(&path).parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).ok();
                }
            }
            index.save(Path::new(&path))?;
        }
        Ok(())
    }
}

/// Derive a per-field cache path from a base path and field key.
///
/// Replaces `/` in field_key with `__` for filesystem safety.
/// Example: `("cache/a_10000", "name_a/name_b")` → `"cache/a_10000.name_a__name_b.index"`
pub fn field_cache_path(base_path: &str, field_key: &str) -> String {
    let safe_key = field_key.replace('/', "__");
    format!("{}.{}.index", base_path, safe_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectordb::flat::FlatVectorDB;

    fn make_record() -> Record {
        let mut r = Record::new();
        r.insert("id".into(), "test".into());
        r
    }

    #[test]
    fn insert_and_get_vec() {
        let dim = 4;
        let mut fi = FieldIndexes::new(dim);
        fi.insert_index("f_a/f_b".into(), Box::new(FlatVectorDB::new(dim)));

        let record = make_record();
        fi.upsert("r1", "f_a/f_b", &[1.0, 0.0, 0.0, 0.0], &record, Side::A)
            .unwrap();

        let vec = fi.get_vec("r1", "f_a/f_b").unwrap();
        assert_eq!(vec.len(), 4);
        assert!((vec[0] - 1.0).abs() < f32::EPSILON);

        assert!(fi.get_vec("r1", "other/field").is_none());
        assert!(fi.get_vec("r999", "f_a/f_b").is_none());
    }

    #[test]
    fn remove_record_all_fields() {
        let dim = 4;
        let mut fi = FieldIndexes::new(dim);
        fi.insert_index("f1_a/f1_b".into(), Box::new(FlatVectorDB::new(dim)));
        fi.insert_index("f2_a/f2_b".into(), Box::new(FlatVectorDB::new(dim)));

        let record = make_record();
        fi.upsert("r1", "f1_a/f1_b", &[1.0, 0.0, 0.0, 0.0], &record, Side::A)
            .unwrap();
        fi.upsert("r1", "f2_a/f2_b", &[0.0, 1.0, 0.0, 0.0], &record, Side::A)
            .unwrap();

        assert_eq!(fi.len(), 2);
        fi.remove_record("r1");
        assert_eq!(fi.len(), 0);
        assert!(fi.get_vec("r1", "f1_a/f1_b").is_none());
        assert!(fi.get_vec("r1", "f2_a/f2_b").is_none());
    }

    #[test]
    fn len_across_fields() {
        let dim = 4;
        let mut fi = FieldIndexes::new(dim);
        fi.insert_index("f1".into(), Box::new(FlatVectorDB::new(dim)));
        fi.insert_index("f2".into(), Box::new(FlatVectorDB::new(dim)));

        let record = make_record();
        fi.upsert("r1", "f1", &[1.0, 0.0, 0.0, 0.0], &record, Side::A)
            .unwrap();
        fi.upsert("r2", "f1", &[0.0, 1.0, 0.0, 0.0], &record, Side::A)
            .unwrap();
        fi.upsert("r1", "f2", &[0.0, 0.0, 1.0, 0.0], &record, Side::A)
            .unwrap();

        assert_eq!(fi.len(), 3); // 2 in f1, 1 in f2
    }

    #[test]
    fn field_cache_path_replaces_slash() {
        assert_eq!(
            field_cache_path("cache/a_10000", "name_a/name_b"),
            "cache/a_10000.name_a__name_b.index"
        );
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("test_base");
        let base_str = base.to_str().unwrap();

        let dim = 4;
        let mut fi = FieldIndexes::new(dim);
        fi.insert_index("f_a/f_b".into(), Box::new(FlatVectorDB::new(dim)));

        let record = make_record();
        fi.upsert("r1", "f_a/f_b", &[1.0, 0.0, 0.0, 0.0], &record, Side::A)
            .unwrap();
        fi.upsert("r2", "f_a/f_b", &[0.0, 1.0, 0.0, 0.0], &record, Side::A)
            .unwrap();

        fi.save_all(base_str).unwrap();

        // Load back
        let path = field_cache_path(base_str, "f_a/f_b");
        let loaded = FlatVectorDB::load(std::path::Path::new(&path)).unwrap();
        assert_eq!(loaded.len(), 2);

        let v = loaded.get("r1").unwrap().unwrap();
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
    }
}
