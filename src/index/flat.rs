//! Flat brute-force vector index with incremental CRUD.
//!
//! Stores vectors in a flat `Vec<f32>` matrix (row-major). Search is brute-force
//! dot product (LLVM auto-vectorizes the inner loop on ARM NEON / x86 SSE/AVX).

use std::collections::{HashMap, HashSet};

/// Flat vector index with O(N*D) brute-force search.
///
/// Vectors are assumed L2-normalized; dot product = cosine similarity.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random unit vector using LCG
        let mut state = seed.wrapping_add(1); // avoid zero state
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Convert to float in [-1, 1]
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

        // Insert 100 random unit vectors
        for i in 0..100 {
            let id = format!("id_{}", i);
            let vec = random_unit_vec(dim, i as u64);
            index.upsert(&id, &vec);
        }

        assert_eq!(index.len(), 100);

        // Search top-5 with the first vector as query
        let query = random_unit_vec(dim, 0);
        let results = index.search(&query, 5);

        assert_eq!(results.len(), 5);
        // First result should be "id_0" (self-match)
        assert_eq!(results[0].0, "id_0");
        assert!(
            (results[0].1 - 1.0).abs() < 0.01,
            "self-similarity = {}",
            results[0].1
        );

        // Results should be sorted descending
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
        assert!((v[0]).abs() < f32::EPSILON); // was 1.0, now 0.0
        assert!((v[1] - 1.0).abs() < f32::EPSILON); // was 0.0, now 1.0
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

        // Verify vectors are correct after swap-remove
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

        assert!(index.remove("b")); // remove last element (no swap needed)
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

        // Insert a, b, c, d, e
        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);
        index.upsert("c", &[0.0, 0.0, 1.0, 0.0]);
        index.upsert("d", &[0.0, 0.0, 0.0, 1.0]);
        index.upsert("e", &[0.5, 0.5, 0.0, 0.0]);

        // Remove "b" (middle) — "e" should be swapped into its position
        assert!(index.remove("b"));
        assert_eq!(index.len(), 4);

        // All remaining IDs should have correct vectors
        let va = index.get("a").unwrap();
        assert!((va[0] - 1.0).abs() < f32::EPSILON);

        let vc = index.get("c").unwrap();
        assert!((vc[2] - 1.0).abs() < f32::EPSILON);

        let vd = index.get("d").unwrap();
        assert!((vd[3] - 1.0).abs() < f32::EPSILON);

        let ve = index.get("e").unwrap();
        assert!((ve[0] - 0.5).abs() < f32::EPSILON);
        assert!((ve[1] - 0.5).abs() < f32::EPSILON);

        // Search should still work correctly
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
        // Only "c" and "d" should be returned; "a" and "b" are excluded
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
