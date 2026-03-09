//! Per-field vector storage for embedding match fields.
//!
//! Stores one vector per (record_id, field_key) pair. Field keys follow the
//! convention `"field_a/field_b"` matching the match_fields config entries.
//!
//! Used during full scoring to compute per-field cosine similarity between
//! a query record and candidate records. NOT used for nearest-neighbour
//! search — that is handled by VectorDB (if candidates.method=embedding).
//!
//! Thread-safe: backed by DashMap for concurrent reads/writes.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use dashmap::DashMap;

/// Per-field vector store.
///
/// Keys are `(record_id, field_key)` where `field_key` is `"field_a/field_b"`.
/// Each entry stores an L2-normalized embedding vector for that record's field.
#[derive(Debug)]
pub struct FieldVectors {
    /// (record_id, field_key) -> vector
    vecs: DashMap<(String, String), Vec<f32>>,
    dim: usize,
}

impl FieldVectors {
    /// Create a new empty FieldVectors store.
    pub fn new(dim: usize) -> Self {
        Self {
            vecs: DashMap::new(),
            dim,
        }
    }

    /// Insert or replace a vector for the given (record_id, field_key).
    pub fn insert(&self, record_id: &str, field_key: &str, vec: Vec<f32>) {
        self.vecs
            .insert((record_id.to_string(), field_key.to_string()), vec);
    }

    /// Get the vector for a given (record_id, field_key).
    pub fn get(&self, record_id: &str, field_key: &str) -> Option<Vec<f32>> {
        let key = (record_id.to_string(), field_key.to_string());
        self.vecs.get(&key).map(|entry| entry.value().clone())
    }

    /// Remove all vectors for a given record_id (across all field keys).
    pub fn remove_record(&self, record_id: &str) {
        // Collect keys to remove (can't remove during iteration).
        let keys_to_remove: Vec<(String, String)> = self
            .vecs
            .iter()
            .filter(|entry| entry.key().0 == record_id)
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys_to_remove {
            self.vecs.remove(&key);
        }
    }

    /// Number of stored vectors (total across all records and fields).
    pub fn len(&self) -> usize {
        self.vecs.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vecs.is_empty()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Save to disk in a simple binary format.
    ///
    /// Format: line-delimited text header + binary vectors.
    /// Header line: `dim={dim} count={count}`
    /// Per entry: `{record_id}\t{field_key}\n` followed by `dim * 4` bytes (f32 LE).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

        let count = self.vecs.len();
        writeln!(f, "dim={} count={}", self.dim, count)?;

        for entry in self.vecs.iter() {
            let (record_id, field_key) = entry.key();
            let vec = entry.value();
            writeln!(f, "{}\t{}", record_id, field_key)?;
            for &v in vec.iter() {
                f.write_all(&v.to_le_bytes())?;
            }
        }

        f.flush()?;
        Ok(())
    }

    /// Load from disk.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        // Parse header
        let mut header = String::new();
        reader.read_line(&mut header)?;
        let header = header.trim();

        let mut dim = 0usize;
        let mut count = 0usize;
        for part in header.split_whitespace() {
            if let Some(val) = part.strip_prefix("dim=") {
                dim = val
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            } else if let Some(val) = part.strip_prefix("count=") {
                count = val
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            }
        }

        if dim == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing or zero dim in header",
            ));
        }

        let vecs = DashMap::with_capacity(count);
        let vec_bytes = dim * 4;

        for _ in 0..count {
            // Read key line
            let mut key_line = String::new();
            reader.read_line(&mut key_line)?;
            let key_line = key_line.trim_end_matches('\n').trim_end_matches('\r');

            let (record_id, field_key) = key_line.split_once('\t').ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing tab in key line")
            })?;

            // Read vector bytes
            let mut buf = vec![0u8; vec_bytes];
            reader.read_exact(&mut buf)?;

            let vec: Vec<f32> = buf
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            vecs.insert((record_id.to_string(), field_key.to_string()), vec);
        }

        Ok(Self { vecs, dim })
    }

    /// Check if a cached file is stale compared to expected count.
    ///
    /// Returns `true` if the cache should be rebuilt.
    pub fn is_stale(path: &Path, expected_count: usize) -> bool {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return true,
        };
        let mut reader = BufReader::new(file);
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() {
            return true;
        }

        // Parse count from header
        let mut count = 0usize;
        for part in header.trim().split_whitespace() {
            if let Some(val) = part.strip_prefix("count=") {
                if let Ok(c) = val.parse::<usize>() {
                    count = c;
                }
            }
        }

        count != expected_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let fv = FieldVectors::new(3);
        fv.insert("rec1", "name_a/name_b", vec![1.0, 0.0, 0.0]);
        fv.insert("rec1", "addr_a/addr_b", vec![0.0, 1.0, 0.0]);
        fv.insert("rec2", "name_a/name_b", vec![0.0, 0.0, 1.0]);

        assert_eq!(fv.len(), 3);
        assert_eq!(fv.get("rec1", "name_a/name_b"), Some(vec![1.0, 0.0, 0.0]));
        assert_eq!(fv.get("rec1", "addr_a/addr_b"), Some(vec![0.0, 1.0, 0.0]));
        assert!(fv.get("rec1", "missing").is_none());
        assert!(fv.get("missing", "name_a/name_b").is_none());
    }

    #[test]
    fn remove_record() {
        let fv = FieldVectors::new(3);
        fv.insert("rec1", "name_a/name_b", vec![1.0, 0.0, 0.0]);
        fv.insert("rec1", "addr_a/addr_b", vec![0.0, 1.0, 0.0]);
        fv.insert("rec2", "name_a/name_b", vec![0.0, 0.0, 1.0]);

        fv.remove_record("rec1");
        assert_eq!(fv.len(), 1);
        assert!(fv.get("rec1", "name_a/name_b").is_none());
        assert!(fv.get("rec1", "addr_a/addr_b").is_none());
        assert!(fv.get("rec2", "name_a/name_b").is_some());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.fieldvecs");

        let fv = FieldVectors::new(4);
        fv.insert("rec1", "name_a/name_b", vec![1.0, 2.0, 3.0, 4.0]);
        fv.insert("rec2", "name_a/name_b", vec![5.0, 6.0, 7.0, 8.0]);
        fv.insert("rec1", "addr_a/addr_b", vec![0.1, 0.2, 0.3, 0.4]);

        fv.save(&path).unwrap();

        let loaded = FieldVectors::load(&path).unwrap();
        assert_eq!(loaded.dim(), 4);
        assert_eq!(loaded.len(), 3);
        assert_eq!(
            loaded.get("rec1", "name_a/name_b"),
            Some(vec![1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(
            loaded.get("rec2", "name_a/name_b"),
            Some(vec![5.0, 6.0, 7.0, 8.0])
        );
        assert_eq!(
            loaded.get("rec1", "addr_a/addr_b"),
            Some(vec![0.1, 0.2, 0.3, 0.4])
        );
    }

    #[test]
    fn staleness_check() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.fieldvecs");

        // Missing file is stale
        assert!(FieldVectors::is_stale(&path, 3));

        // Save with count=3
        let fv = FieldVectors::new(2);
        fv.insert("a", "f", vec![1.0, 0.0]);
        fv.insert("b", "f", vec![0.0, 1.0]);
        fv.insert("c", "f", vec![1.0, 1.0]);
        fv.save(&path).unwrap();

        // Matching count is fresh
        assert!(!FieldVectors::is_stale(&path, 3));
        // Different count is stale
        assert!(FieldVectors::is_stale(&path, 5));
    }
}
