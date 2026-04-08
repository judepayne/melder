//! Candidate generation: ANN search or flat-scan to select top-N pool records
//! for full scoring.
//!
//! This module is one of several independent candidate generators in the
//! pipeline. ANN candidates are unioned with candidates from other sources
//! (BM25, synonym, etc.) before full scoring.
//!
//! Two paths depending on backend:
//!
//! **usearch** — O(log N):
//!   `pool_combined_index.search(query_combined_vec, top_n, query_record, query_side)`
//!   Block routing is handled internally by the index.
//!
//! **flat** — O(N):
//!   Iterate `blocked_ids`, call `get` on the combined index, compute dot
//!   products manually, sort descending, truncate to `top_n`.
//!
//! If there are no embedding fields (`query_combined_vec` is empty), returns
//! an empty vec — other candidate generators (BM25, etc.) provide candidates
//! independently.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::models::{Record, Side};
use crate::scoring::embedding::dot_product_f32;
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// A candidate record from the opposite pool, with its combined embedding
/// dot product from Stage 1.
///
/// Owns a clone of the pool record. Since we only keep `top_n` candidates
/// the cloning cost is negligible.
#[derive(Debug)]
pub struct Candidate {
    pub id: String,
    pub record: Record,
    /// Dot product of the combined embedding vectors:
    ///   `dot(query_combined, pool_combined) = Σᵢ wᵢ · cosine_sim(aᵢ, bᵢ)`
    /// Zero when no embedding fields are configured.
    pub combined_dot: f64,
}

/// Select the top-N ANN candidates from the blocked pool.
///
/// Returns an empty vec when no embedding fields are configured — other
/// candidate generators (BM25, synonym, etc.) provide candidates
/// independently via the pipeline union.
///
/// Parameters:
/// - `query_combined_vec`: pre-encoded combined embedding vector for the
///   query record (empty slice if no embedding fields).
/// - `top_n`: maximum candidates to return. 0 = no limit (flat path only).
/// - `pool_combined_index`: combined embedding index for the pool side.
///   `None` if no embedding fields are configured.
/// - `blocked_ids`: IDs pre-selected by the blocking filter. Used by the
///   flat path. Ignored by the usearch path (blocking is internal).
/// - `pool_store`: record store for looking up pool-side records.
/// - `pool_side`: which side of the store the pool records are on.
/// - `query_record`: query record, used by usearch for block routing.
/// - `query_side`: query side, used by usearch for block routing.
/// - `backend`: "usearch" or "flat".
#[allow(clippy::too_many_arguments)]
pub fn select_candidates(
    query_combined_vec: &[f32],
    top_n: usize,
    pool_combined_index: Option<&dyn VectorDB>,
    blocked_ids: &[String],
    pool_store: &dyn RecordStore,
    pool_side: Side,
    query_record: &Record,
    query_side: Side,
    backend: &str,
) -> Vec<Candidate> {
    let has_embeddings = !query_combined_vec.is_empty();

    // usearch path: block-aware ANN search — O(log N).
    #[cfg(feature = "usearch")]
    if backend == "usearch"
        && has_embeddings
        && let Some(idx) = pool_combined_index
    {
        let k = if top_n > 0 { top_n } else { usize::MAX };
        if let Ok(results) = idx.search(query_combined_vec, k, query_record, query_side) {
            return results
                .into_iter()
                .filter_map(|r| {
                    pool_store
                        .get(pool_side, &r.id)
                        .ok()
                        .flatten()
                        .map(|record| Candidate {
                            id: r.id.clone(),
                            record,
                            combined_dot: r.score as f64,
                        })
                })
                .collect();
        }
    }

    // flat path (also fallback for usearch when no embedding fields):
    // iterate blocked_ids, compute dot products, sort, truncate.
    let _ = backend; // suppress unused warning in non-usearch builds

    if !has_embeddings || pool_combined_index.is_none() {
        // No embedding fields configured — return empty. Other candidate
        // generators (BM25, synonym, etc.) provide candidates independently
        // and are unioned by the pipeline.
        return Vec::new();
    }

    let idx = pool_combined_index.unwrap();

    let dim_warned = AtomicBool::new(false);
    let mut scored: Vec<Candidate> = blocked_ids
        .par_iter()
        .filter_map(|id| {
            let record = pool_store.get(pool_side, id).ok().flatten()?;
            let pool_vec = idx.get(id).ok().flatten().unwrap_or_default();
            let dot: f32 = if pool_vec.len() == query_combined_vec.len() {
                dot_product_f32(query_combined_vec, &pool_vec)
            } else {
                if !dim_warned.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        id,
                        expected = query_combined_vec.len(),
                        got = pool_vec.len(),
                        "embedding dimension mismatch in flat candidate scan — scoring as 0.0"
                    );
                }
                0.0
            };
            Some(Candidate {
                id: id.clone(),
                record,
                combined_dot: dot as f64,
            })
        })
        .collect();

    scored.sort_by(|a, b| {
        b.combined_dot
            .partial_cmp(&a.combined_dot)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if top_n > 0 {
        scored.truncate(top_n);
    }

    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingConfig;
    use crate::store::memory::MemoryStore;
    use crate::vectordb::flat::FlatVectorDB;

    fn make_record(id: &str) -> Record {
        let mut r = Record::new();
        r.insert("id".into(), id.into());
        r
    }

    fn make_store() -> MemoryStore {
        MemoryStore::new(&BlockingConfig::default())
    }

    fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
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
    fn no_embeddings_returns_empty() {
        let store = make_store();
        store.insert(Side::A, "r1", &make_record("r1")).unwrap();
        store.insert(Side::A, "r2", &make_record("r2")).unwrap();

        let blocked_ids = vec!["r1".to_string(), "r2".to_string()];
        let query_record = make_record("query");

        let cands = select_candidates(
            &[], // empty → no embeddings
            5,
            None,
            &blocked_ids,
            &store,
            Side::A,
            &query_record,
            Side::B,
            "flat",
        );

        assert!(
            cands.is_empty(),
            "expected empty when no embeddings — other generators provide candidates"
        );
    }

    #[test]
    fn flat_path_with_combined_index_sorts_by_dot_product() {
        let dim = 16usize;
        let idx = FlatVectorDB::new(dim);

        // Insert 3 vecs; query is identical to seed 0
        let v0 = unit_vec(dim, 0);
        let v1 = unit_vec(dim, 1);
        let v2 = unit_vec(dim, 2);

        let dummy = make_record("x");
        idx.upsert("r0", &v0, &dummy, Side::A).unwrap();
        idx.upsert("r1", &v1, &dummy, Side::A).unwrap();
        idx.upsert("r2", &v2, &dummy, Side::A).unwrap();

        let store = make_store();
        store.insert(Side::A, "r0", &make_record("r0")).unwrap();
        store.insert(Side::A, "r1", &make_record("r1")).unwrap();
        store.insert(Side::A, "r2", &make_record("r2")).unwrap();

        let blocked_ids = vec!["r0".to_string(), "r1".to_string(), "r2".to_string()];
        let query_record = make_record("query");

        let cands = select_candidates(
            &v0, // identical to r0's vec → r0 should be first
            2,
            Some(&idx as &dyn VectorDB),
            &blocked_ids,
            &store,
            Side::A,
            &query_record,
            Side::B,
            "flat",
        );

        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].id, "r0");
        // combined_dot for self-match should be ~1.0
        assert!(
            cands[0].combined_dot > 0.95,
            "expected ~1.0, got {}",
            cands[0].combined_dot
        );
    }

    #[test]
    fn flat_path_top_n_zero_returns_all() {
        let dim = 16usize;
        let idx = FlatVectorDB::new(dim);
        let dummy = make_record("x");
        let store = make_store();
        let mut blocked_ids = Vec::new();

        for i in 0..10 {
            let id = format!("r{}", i);
            let v = unit_vec(dim, i as u64);
            idx.upsert(&id, &v, &dummy, Side::A).unwrap();
            store.insert(Side::A, &id, &make_record(&id)).unwrap();
            blocked_ids.push(id);
        }

        let query = unit_vec(dim, 99);
        let query_record = make_record("query");

        let cands = select_candidates(
            &query,
            0, // no limit
            Some(&idx as &dyn VectorDB),
            &blocked_ids,
            &store,
            Side::A,
            &query_record,
            Side::B,
            "flat",
        );

        assert_eq!(cands.len(), 10);
    }
}
