//! FlatVectorDB: wraps the existing brute-force VecIndex behind the VectorDB trait.
//!
//! This is a thin adapter. All operations delegate to `VecIndex` under an
//! `RwLock` for concurrent access. Persistence uses the existing binary
//! cache format from `crate::index::cache`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::RwLock;

use crate::index::cache;
use crate::index::VecIndex;
use crate::models::{Record, Side};

use super::{SearchResult, VectorDB, VectorDBError};

/// Brute-force flat vector index implementing `VectorDB`.
///
/// O(N*D) search via dot product. Suitable for datasets up to ~100K vectors.
/// Uses `RwLock<VecIndex>` internally for `Send + Sync`.
pub struct FlatVectorDB {
    inner: RwLock<VecIndex>,
    dim: usize,
}

impl std::fmt::Debug for FlatVectorDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.inner.read().map(|idx| idx.len()).unwrap_or(0);
        f.debug_struct("FlatVectorDB")
            .field("dim", &self.dim)
            .field("len", &len)
            .finish()
    }
}

impl FlatVectorDB {
    /// Create an empty FlatVectorDB with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            inner: RwLock::new(VecIndex::new(dim)),
            dim,
        }
    }

    /// Create from a pre-existing VecIndex.
    pub fn from_vec_index(index: VecIndex) -> Self {
        let dim = index.dim();
        Self {
            inner: RwLock::new(index),
            dim,
        }
    }

    /// Load from a binary cache file.
    pub fn load(path: &Path) -> Result<Self, VectorDBError> {
        let index = cache::load_index(path).map_err(|e| VectorDBError::Backend(e.to_string()))?;
        let dim = index.dim();
        Ok(Self {
            inner: RwLock::new(index),
            dim,
        })
    }
}

impl VectorDB for FlatVectorDB {
    fn upsert(
        &self,
        id: &str,
        vec: &[f32],
        _record: &Record,
        _side: Side,
    ) -> Result<(), VectorDBError> {
        if vec.len() != self.dim {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dim,
                got: vec.len(),
            });
        }
        let mut idx = self.inner.write().unwrap();
        idx.upsert(id, vec);
        Ok(())
    }

    fn remove(&self, id: &str) -> Result<bool, VectorDBError> {
        let mut idx = self.inner.write().unwrap();
        Ok(idx.remove(id))
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        _record: &Record,
        _side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError> {
        if query.len() != self.dim {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let idx = self.inner.read().unwrap();
        let results = idx.search(query, k);
        Ok(results
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect())
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &HashSet<String>,
        _record: &Record,
        _side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError> {
        if query.len() != self.dim {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let idx = self.inner.read().unwrap();
        let results = idx.search_filtered(query, k, allowed);
        Ok(results
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect())
    }

    fn get(&self, id: &str) -> Result<Option<Vec<f32>>, VectorDBError> {
        let idx = self.inner.read().unwrap();
        Ok(idx.get(id).map(|slice| slice.to_vec()))
    }

    fn contains(&self, id: &str) -> bool {
        let idx = self.inner.read().unwrap();
        idx.contains(id)
    }

    fn len(&self) -> usize {
        let idx = self.inner.read().unwrap();
        idx.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn save(&self, path: &Path) -> Result<(), VectorDBError> {
        let idx = self.inner.read().unwrap();
        cache::save_index(path, &idx).map_err(|e| VectorDBError::Backend(e.to_string()))
    }

    fn is_stale(path: &Path, expected_count: usize) -> Result<bool, VectorDBError> {
        Ok(cache::is_cache_stale(path, expected_count))
    }
}
