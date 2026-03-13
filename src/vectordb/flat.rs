//! Flat brute-force vector index with incremental CRUD and binary cache.
//!
//! Contains:
//! - `VecIndex`: the core flat vector store (O(N*D) search via dot product)
//! - `FlatVectorDB`: wraps `VecIndex` behind the `VectorDB` trait with `RwLock`
//! - Cache serialization: save/load `VecIndex` to/from binary files

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::RwLock;

use crate::error::IndexError;
use crate::models::{Record, Side};

use super::texthash::TextHashStore;
use super::{SearchResult, VectorDB, VectorDBError};

// ===========================================================================
// VecIndex — flat brute-force vector store
// ===========================================================================

/// Flat vector index with O(N*D) brute-force search.
///
/// Vectors are assumed L2-normalized; dot product = cosine similarity.
/// Stores vectors in a flat `Vec<f32>` matrix (row-major). Search is
/// brute-force dot product (LLVM auto-vectorizes the inner loop on
/// ARM NEON / x86 SSE/AVX).
#[derive(Clone)]
pub struct VecIndex {
    /// Dense matrix: N rows x D columns, row-major.
    vectors: Vec<f32>,
    /// Embedding dimension.
    dim: usize,
    /// Parallel array: `vectors[i*dim..(i+1)*dim]` belongs to `ids[i]`.
    ids: Vec<String>,
    /// Reverse lookup: id -> position in ids/vectors.
    id_to_pos: HashMap<String, usize>,
}

impl VecIndex {
    /// Create an empty index with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
            ids: Vec::new(),
            id_to_pos: HashMap::new(),
        }
    }

    /// Create an index from pre-existing data (used by cache loading).
    pub fn from_parts(vectors: Vec<f32>, dim: usize, ids: Vec<String>) -> Self {
        assert_eq!(
            vectors.len(),
            ids.len() * dim,
            "vectors length must equal ids.len() * dim"
        );
        let id_to_pos: HashMap<String, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        Self {
            vectors,
            dim,
            ids,
            id_to_pos,
        }
    }

    /// Insert or replace a vector.
    ///
    /// If the id already exists, the vector is overwritten in-place.
    /// If new, appended to the end.
    ///
    /// # Panics
    /// Panics if `vec.len() != self.dim`.
    pub fn upsert(&mut self, id: &str, vec: &[f32]) {
        assert_eq!(
            vec.len(),
            self.dim,
            "vector dimension mismatch: expected {}, got {}",
            self.dim,
            vec.len()
        );

        if let Some(&pos) = self.id_to_pos.get(id) {
            // Overwrite existing vector
            let start = pos * self.dim;
            self.vectors[start..start + self.dim].copy_from_slice(vec);
        } else {
            // Append new
            let pos = self.ids.len();
            self.ids.push(id.to_string());
            self.vectors.extend_from_slice(vec);
            self.id_to_pos.insert(id.to_string(), pos);
        }
    }

    /// Remove a vector by id (swap-remove for O(1)).
    ///
    /// Returns `true` if the id was found and removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(&pos) = self.id_to_pos.get(id) else {
            return false;
        };

        let last = self.ids.len() - 1;

        if pos != last {
            // Swap the last element into the removed position
            let last_id = self.ids[last].clone();

            // Copy last vector row into removed position
            let src_start = last * self.dim;
            let dst_start = pos * self.dim;
            for i in 0..self.dim {
                self.vectors[dst_start + i] = self.vectors[src_start + i];
            }

            // Update ids
            self.ids[pos] = last_id.clone();
            self.id_to_pos.insert(last_id, pos);
        }

        // Truncate
        self.ids.pop();
        self.vectors.truncate(self.ids.len() * self.dim);
        self.id_to_pos.remove(id);

        true
    }

    /// Find top-K nearest by dot product (brute-force full scan).
    ///
    /// Returns `Vec<(id, score)>` sorted by score descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        assert_eq!(query.len(), self.dim);
        if self.ids.is_empty() || k == 0 {
            return vec![];
        }

        let n = self.ids.len();
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            let start = i * self.dim;
            let row = &self.vectors[start..start + self.dim];
            let dot = dot_product_f32(query, row);
            scores.push((i, dot));
        }

        // Partial sort: select top-K
        let k = k.min(n);
        scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .map(|(i, s)| (self.ids[i].clone(), s))
            .collect()
    }

    /// Find top-K nearest, considering only IDs in `allowed`.
    ///
    /// Skips dot-product computation for IDs not in the set.
    /// Returns `Vec<(id, score)>` sorted by score descending.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &HashSet<String>,
    ) -> Vec<(String, f32)> {
        assert_eq!(query.len(), self.dim);
        if self.ids.is_empty() || k == 0 || allowed.is_empty() {
            return vec![];
        }

        let mut scores: Vec<(usize, f32)> = Vec::new();

        for i in 0..self.ids.len() {
            if !allowed.contains(&self.ids[i]) {
                continue;
            }
            let start = i * self.dim;
            let row = &self.vectors[start..start + self.dim];
            let dot = dot_product_f32(query, row);
            scores.push((i, dot));
        }

        if scores.is_empty() {
            return vec![];
        }

        let k = k.min(scores.len());
        scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .map(|(i, s)| (self.ids[i].clone(), s))
            .collect()
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Check if an id exists in the index.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_pos.contains_key(id)
    }

    /// Get the vector for an id (as a slice).
    pub fn get(&self, id: &str) -> Option<&[f32]> {
        self.id_to_pos
            .get(id)
            .map(|&pos| &self.vectors[pos * self.dim..(pos + 1) * self.dim])
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get all ids (in internal order).
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Get the raw vectors (for serialization).
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }
}

impl std::fmt::Debug for VecIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecIndex")
            .field("len", &self.ids.len())
            .field("dim", &self.dim)
            .finish()
    }
}

/// Dot product of two f32 slices. LLVM auto-vectorizes this.
#[inline]
fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ===========================================================================
// Cache serialization
// ===========================================================================

/// Binary format:
/// ```text
/// [4 bytes] N (u32 little-endian) — number of vectors
/// [4 bytes] D (u32 little-endian) — dimension
/// [N*D*4 bytes] vectors (f32 little-endian, row-major)
/// [variable] N newline-separated ID strings (UTF-8)
/// ```

/// Save a VecIndex to a binary cache file.
pub fn save_index(path: &Path, index: &VecIndex) -> Result<(), IndexError> {
    let n = index.len() as u32;
    let d = index.dim() as u32;

    let mut file = std::fs::File::create(path)?;

    // Write header
    file.write_all(&n.to_le_bytes())?;
    file.write_all(&d.to_le_bytes())?;

    // Write vectors as raw f32 bytes (little-endian)
    let vectors = index.vectors();
    let byte_slice =
        unsafe { std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4) };
    file.write_all(byte_slice)?;

    // Write IDs as newline-separated strings
    for id in index.ids() {
        file.write_all(id.as_bytes())?;
        file.write_all(b"\n")?;
    }

    file.flush()?;
    Ok(())
}

/// Load a VecIndex from a binary cache file.
pub fn load_index(path: &Path) -> Result<VecIndex, IndexError> {
    let mut file = std::fs::File::open(path)?;

    // Read header
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    file.read_exact(&mut buf4)?;
    let d = u32::from_le_bytes(buf4) as usize;

    // Read vectors
    let vec_bytes = n * d * 4;
    let mut vec_buf = vec![0u8; vec_bytes];
    file.read_exact(&mut vec_buf)?;

    // Convert bytes to f32 (assumes little-endian platform — true for x86/ARM)
    let vectors: Vec<f32> = vec_buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Read IDs
    let mut id_buf = String::new();
    file.read_to_string(&mut id_buf)?;
    let ids: Vec<String> = id_buf
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();

    if ids.len() != n {
        return Err(IndexError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cache file corrupt: header says {} vectors but found {} IDs",
                n,
                ids.len()
            ),
        )));
    }

    Ok(VecIndex::from_parts(vectors, d, ids))
}

/// Check whether a cache file is stale relative to current data.
///
/// Returns `true` if the cache should be rebuilt:
/// - File doesn't exist
/// - Record count doesn't match
pub fn is_cache_stale(path: &Path, record_count: usize) -> bool {
    if !path.exists() {
        return true;
    }

    // Quick check: read just the header to get N
    let Ok(mut file) = std::fs::File::open(path) else {
        return true;
    };

    let mut buf4 = [0u8; 4];
    if file.read_exact(&mut buf4).is_err() {
        return true;
    }
    let cached_n = u32::from_le_bytes(buf4) as usize;

    cached_n != record_count
}

// ===========================================================================
// FlatVectorDB — VectorDB trait implementation
// ===========================================================================

/// Brute-force flat vector index implementing `VectorDB`.
///
/// O(N*D) search via dot product. Suitable for datasets up to ~100K vectors.
/// Uses `RwLock` internally for `Send + Sync`.
pub struct FlatVectorDB {
    inner: RwLock<VecIndex>,
    text_hashes: RwLock<TextHashStore>,
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
    ///
    /// Used in tests and wherever emb_specs are not needed. The text-hash
    /// store will be empty (no-op on upsert).
    pub fn new(dim: usize) -> Self {
        Self {
            inner: RwLock::new(VecIndex::new(dim)),
            text_hashes: RwLock::new(TextHashStore::empty()),
            dim,
        }
    }

    /// Create an empty FlatVectorDB with embedding specs for text hashing.
    ///
    /// Called by `new_index()` in mod.rs when building the combined index.
    pub fn new_with_emb_specs(dim: usize, emb_specs: Vec<(String, String, f64)>) -> Self {
        Self {
            inner: RwLock::new(VecIndex::new(dim)),
            text_hashes: RwLock::new(TextHashStore::new(emb_specs)),
            dim,
        }
    }

    /// Create from a pre-existing VecIndex (used in tests).
    pub fn from_vec_index(index: VecIndex) -> Self {
        let dim = index.dim();
        Self {
            inner: RwLock::new(index),
            text_hashes: RwLock::new(TextHashStore::empty()),
            dim,
        }
    }

    /// Load from a binary cache file, restoring the text-hash sidecar if present.
    pub fn load(path: &Path) -> Result<Self, VectorDBError> {
        let index = load_index(path).map_err(|e| VectorDBError::Backend(e.to_string()))?;
        let dim = index.dim();
        let text_hashes = TextHashStore::load(path)
            .map_err(|e| VectorDBError::Backend(format!("texthash load: {}", e)))?;
        Ok(Self {
            inner: RwLock::new(index),
            text_hashes: RwLock::new(text_hashes),
            dim,
        })
    }
}

impl VectorDB for FlatVectorDB {
    fn upsert(
        &self,
        id: &str,
        vec: &[f32],
        record: &Record,
        side: Side,
    ) -> Result<(), VectorDBError> {
        if vec.len() != self.dim {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dim,
                got: vec.len(),
            });
        }
        let mut idx = self.inner.write().unwrap();
        idx.upsert(id, vec);
        drop(idx);
        // Update text hash (no-op if emb_specs is empty).
        let mut th = self.text_hashes.write().unwrap();
        th.update(id, record, side);
        Ok(())
    }

    fn remove(&self, id: &str) -> Result<bool, VectorDBError> {
        let mut idx = self.inner.write().unwrap();
        let removed = idx.remove(id);
        drop(idx);
        if removed {
            let mut th = self.text_hashes.write().unwrap();
            th.remove(id);
        }
        Ok(removed)
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
        save_index(path, &idx).map_err(|e| VectorDBError::Backend(e.to_string()))?;
        drop(idx);
        let th = self.text_hashes.read().unwrap();
        th.save(path)
            .map_err(|e| VectorDBError::Backend(format!("texthash save: {}", e)))?;
        Ok(())
    }

    fn stored_text_hashes(&self) -> HashMap<String, u64> {
        self.text_hashes.read().unwrap().all().clone()
    }

    fn text_hash_for(&self, id: &str) -> Option<u64> {
        self.text_hashes.read().unwrap().get(id)
    }

    fn is_stale(path: &Path, expected_count: usize) -> Result<bool, VectorDBError> {
        Ok(is_cache_stale(path, expected_count))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod vec_index_tests {
    use super::*;

    fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as i32 as f32) / (i32::MAX as f32)
            })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn insert_and_search() {
        let dim = 384;
        let mut index = VecIndex::new(dim);

        for i in 0..100 {
            let id = format!("id_{}", i);
            let vec = random_unit_vec(dim, i as u64);
            index.upsert(&id, &vec);
        }

        assert_eq!(index.len(), 100);

        let query = random_unit_vec(dim, 0);
        let results = index.search(&query, 5);

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, "id_0");
        assert!(
            (results[0].1 - 1.0).abs() < 0.01,
            "self-similarity = {}",
            results[0].1
        );

        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted: {} < {}", w[0].1, w[1].1);
        }
    }

    #[test]
    fn upsert_replaces() {
        let dim = 4;
        let mut index = VecIndex::new(dim);

        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("a", &[0.0, 1.0, 0.0, 0.0]);

        assert_eq!(index.len(), 1);
        let v = index.get("a").unwrap();
        assert!((v[0]).abs() < f32::EPSILON);
        assert!((v[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn remove_basic() {
        let dim = 4;
        let mut index = VecIndex::new(dim);

        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);
        index.upsert("c", &[0.0, 0.0, 1.0, 0.0]);

        assert_eq!(index.len(), 3);
        assert!(index.remove("b"));
        assert_eq!(index.len(), 2);
        assert!(!index.contains("b"));
        assert!(index.contains("a"));
        assert!(index.contains("c"));

        let va = index.get("a").unwrap();
        assert!((va[0] - 1.0).abs() < f32::EPSILON);
        let vc = index.get("c").unwrap();
        assert!((vc[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn remove_last() {
        let dim = 4;
        let mut index = VecIndex::new(dim);

        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);

        assert!(index.remove("b"));
        assert_eq!(index.len(), 1);
        assert!(index.contains("a"));
        assert!(!index.contains("b"));
    }

    #[test]
    fn remove_nonexistent() {
        let mut index = VecIndex::new(4);
        assert!(!index.remove("xyz"));
    }

    #[test]
    fn remove_middle_preserves_mapping() {
        let dim = 4;
        let mut index = VecIndex::new(dim);

        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);
        index.upsert("c", &[0.0, 0.0, 1.0, 0.0]);
        index.upsert("d", &[0.0, 0.0, 0.0, 1.0]);
        index.upsert("e", &[0.5, 0.5, 0.0, 0.0]);

        assert!(index.remove("b"));
        assert_eq!(index.len(), 4);

        let va = index.get("a").unwrap();
        assert!((va[0] - 1.0).abs() < f32::EPSILON);

        let vc = index.get("c").unwrap();
        assert!((vc[2] - 1.0).abs() < f32::EPSILON);

        let vd = index.get("d").unwrap();
        assert!((vd[3] - 1.0).abs() < f32::EPSILON);

        let ve = index.get("e").unwrap();
        assert!((ve[0] - 0.5).abs() < f32::EPSILON);
        assert!((ve[1] - 0.5).abs() < f32::EPSILON);

        let query = [1.0_f32, 0.0, 0.0, 0.0];
        let results = index.search(&query, 2);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn search_filtered_basic() {
        let dim = 4;
        let mut index = VecIndex::new(dim);

        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.9, 0.1, 0.0, 0.0]);
        index.upsert("c", &[0.0, 1.0, 0.0, 0.0]);
        index.upsert("d", &[0.0, 0.0, 1.0, 0.0]);

        let query = [1.0_f32, 0.0, 0.0, 0.0];
        let allowed: HashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();

        let results = index.search_filtered(&query, 5, &allowed);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(id, _)| allowed.contains(id)));
    }

    #[test]
    fn search_empty_index() {
        let index = VecIndex::new(4);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_larger_than_n() {
        let dim = 4;
        let mut index = VecIndex::new(dim);
        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn contains_and_get() {
        let dim = 4;
        let mut index = VecIndex::new(dim);
        index.upsert("a", &[1.0, 2.0, 3.0, 4.0]);

        assert!(index.contains("a"));
        assert!(!index.contains("b"));

        let v = index.get("a").unwrap();
        assert_eq!(v.len(), 4);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
        assert!(index.get("b").is_none());
    }

    #[test]
    fn manual_verification_against_dot_product_sort() {
        let dim = 384;
        let mut index = VecIndex::new(dim);

        let n: usize = 100;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_unit_vec(dim, i as u64)).collect();
        for (i, v) in vecs.iter().enumerate() {
            index.upsert(&format!("id_{}", i), v);
        }

        let query = random_unit_vec(dim, 999);
        let k = 5;
        let results = index.search(&query, k);

        // Manual sort
        let mut manual: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, dot_product_f32(&query, &vecs[i])))
            .collect();
        manual.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (j, (id, score)) in results.iter().enumerate() {
            let expected_id = format!("id_{}", manual[j].0);
            assert_eq!(id, &expected_id, "position {}", j);
            assert!(
                (score - manual[j].1).abs() < 0.001,
                "score mismatch at position {}",
                j
            );
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use tempfile::tempdir;

    fn make_test_index() -> VecIndex {
        let dim = 4;
        let mut index = VecIndex::new(dim);
        for i in 0..100 {
            let id = format!("id_{}", i);
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
            index.upsert(&id, &vec);
        }
        index
    }

    #[test]
    fn save_and_load_roundtrip() {
        let index = make_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.index");

        save_index(&path, &index).unwrap();
        let loaded = load_index(&path).unwrap();

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.dim(), index.dim());

        for id in index.ids() {
            let orig = index.get(id).unwrap();
            let loaded_v = loaded.get(id).unwrap();
            assert_eq!(orig.len(), loaded_v.len());
            for (a, b) in orig.iter().zip(loaded_v.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "vector mismatch for id {}", id);
            }
        }
    }

    #[test]
    fn save_empty_index() {
        let index = VecIndex::new(384);
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.index");

        save_index(&path, &index).unwrap();
        let loaded = load_index(&path).unwrap();

        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.dim(), 384);
    }

    #[test]
    fn staleness_missing_file() {
        assert!(is_cache_stale(
            &Path::new("nonexistent_dir_for_test").join("file.index"),
            100,
        ));
    }

    #[test]
    fn staleness_count_mismatch() {
        let index = make_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.index");

        save_index(&path, &index).unwrap();

        assert!(!is_cache_stale(&path, 100));
        assert!(is_cache_stale(&path, 101));
        assert!(is_cache_stale(&path, 99));
    }

    #[test]
    fn search_after_load() {
        let dim = 4;
        let mut index = VecIndex::new(dim);
        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);
        index.upsert("c", &[0.707, 0.707, 0.0, 0.0]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.index");

        save_index(&path, &index).unwrap();
        let loaded = load_index(&path).unwrap();

        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > 0.99);
    }
}
