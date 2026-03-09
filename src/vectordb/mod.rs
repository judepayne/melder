//! VectorDB abstraction layer.
//!
//! Defines a backend-agnostic trait for vector storage and search.
//! Implementations:
//! - `FlatVectorDB`: wraps the existing brute-force `VecIndex` (O(N) search)
//! - `UsearchVectorDB`: per-block HNSW indices via usearch (O(log N) search)
//!
//! The rest of melder interacts only with the `VectorDB` trait.
//! Implementations handle their own persistence format, key mapping,
//! block routing, and concurrency internally.

pub mod field_indexes;
pub mod flat;
#[cfg(feature = "usearch")]
pub mod usearch_backend;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::path::Path;
use std::time::Instant;

use crate::config::schema::BlockingConfig;
use crate::config::Config;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::models::{Record, Side};

/// Errors from vector database operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorDBError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("config mismatch: {0}")]
    ConfigMismatch(String),
}

/// Search result: (id, score) pair.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

/// Backend-agnostic vector storage and search.
///
/// All implementations must be `Send + Sync` for use in concurrent
/// server contexts. Implementations handle their own internal locking.
///
/// Vectors are assumed to be L2-normalized. Search returns results
/// ranked by dot product (= cosine similarity for normalized vectors),
/// highest first.
///
/// Methods that accept `record` and `side` allow block-aware backends
/// (e.g. `UsearchVectorDB`) to compute block keys from the record's
/// fields. Flat backends ignore these parameters.
pub trait VectorDB: Send + Sync + Debug {
    /// Insert or replace a vector for the given ID.
    ///
    /// If the ID already exists, the vector is overwritten.
    /// Returns an error if `vec.len() != self.dim()`.
    ///
    /// `record` and `side` are used by block-aware backends to determine
    /// which block this vector belongs to. Flat backends ignore them.
    fn upsert(
        &self,
        id: &str,
        vec: &[f32],
        record: &Record,
        side: Side,
    ) -> Result<(), VectorDBError>;

    /// Remove a vector by ID.
    ///
    /// Returns `true` if the ID was found and removed, `false` if not found.
    /// Implementations may retain the vector internally for reuse (orphan
    /// retention) — the caller should not assume the vector is freed.
    fn remove(&self, id: &str) -> Result<bool, VectorDBError>;

    /// Find the top-K nearest vectors by dot product.
    ///
    /// Returns results sorted by score descending. If the index contains
    /// fewer than `k` vectors, returns all of them.
    ///
    /// `record` and `side` are used by block-aware backends to determine
    /// which block to search. Flat backends ignore them.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        record: &Record,
        side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError>;

    /// Find the top-K nearest vectors, considering only IDs in `allowed`.
    ///
    /// This is the primary search method used during the scoring pipeline,
    /// where blocking pre-filters the candidate set.
    ///
    /// `record` and `side` are used by block-aware backends to determine
    /// which block to search. Flat backends ignore them.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &HashSet<String>,
        record: &Record,
        side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError>;

    /// Retrieve the vector for a given ID.
    ///
    /// Returns `None` if the ID is not present (or has been removed).
    fn get(&self, id: &str) -> Result<Option<Vec<f32>>, VectorDBError>;

    /// Check whether an ID exists in the index.
    fn contains(&self, id: &str) -> bool;

    /// Number of active vectors (excludes orphans/tombstones).
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Embedding dimension.
    fn dim(&self) -> usize;

    /// Persist the index to disk.
    ///
    /// The format is implementation-specific. The path should be treated
    /// as a base path — implementations may create sidecar files
    /// (e.g. `path.keys` alongside `path`) or directory structures.
    fn save(&self, path: &Path) -> Result<(), VectorDBError>;

    /// Check whether a saved index at `path` is stale relative to the
    /// expected record count. Returns `true` if the index should be rebuilt.
    fn is_stale(path: &Path, expected_count: usize) -> Result<bool, VectorDBError>
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// Backend dispatch: factory functions that select flat vs usearch at runtime
// ---------------------------------------------------------------------------

/// Create a new empty vector index for the given backend.
///
/// `blocking_config` is only used by the usearch backend (for per-block
/// routing). The flat backend ignores it.
pub fn new_index(
    backend: &str,
    dim: usize,
    #[allow(unused_variables)] blocking_config: Option<&BlockingConfig>,
) -> Box<dyn VectorDB> {
    match backend {
        #[cfg(feature = "usearch")]
        "usearch" => Box::new(usearch_backend::UsearchVectorDB::new(dim, blocking_config)),
        "flat" | _ => Box::new(flat::FlatVectorDB::new(dim)),
    }
}

/// Load a vector index from a cache file.
pub fn load_index(backend: &str, path: &Path) -> Result<Box<dyn VectorDB>, VectorDBError> {
    match backend {
        #[cfg(feature = "usearch")]
        "usearch" => Ok(Box::new(usearch_backend::UsearchVectorDB::load(path)?)),
        "flat" | _ => Ok(Box::new(flat::FlatVectorDB::load(path)?)),
    }
}

/// Check whether a cached index is stale (needs rebuilding).
pub fn is_index_stale(
    backend: &str,
    path: &Path,
    expected_count: usize,
) -> Result<bool, VectorDBError> {
    match backend {
        #[cfg(feature = "usearch")]
        "usearch" => usearch_backend::UsearchVectorDB::is_stale(path, expected_count),
        "flat" | _ => flat::FlatVectorDB::is_stale(path, expected_count),
    }
}

// ---------------------------------------------------------------------------
// Per-field embedding vectors
// ---------------------------------------------------------------------------

/// Collect the embedding field keys from config.
///
/// Returns a vec of `(field_key, field_a_name, field_b_name)` for each
/// embedding field pair. Includes both match fields with `method: "embedding"`
/// and the candidates field pair when `candidates.method == "embedding"`.
/// Field key is `"field_a/field_b"`. Duplicates are suppressed (if candidates
/// uses the same pair as a match field, it appears only once).
pub fn embedding_field_keys(config: &Config) -> Vec<(String, String, String)> {
    let mut result: Vec<(String, String, String)> = config
        .match_fields
        .iter()
        .filter(|mf| mf.method == "embedding")
        .map(|mf| {
            let key = format!("{}/{}", mf.field_a, mf.field_b);
            (key, mf.field_a.clone(), mf.field_b.clone())
        })
        .collect();

    // Include candidates field pair if method is embedding.
    let cand = &config.candidates;
    let cand_enabled = cand
        .enabled
        .unwrap_or(cand.field_a.is_some() || cand.field_b.is_some() || cand.method.is_some());
    if cand_enabled {
        if let (Some(method), Some(fa), Some(fb)) = (
            cand.method.as_deref(),
            cand.field_a.as_deref(),
            cand.field_b.as_deref(),
        ) {
            if method == "embedding" {
                let key = format!("{}/{}", fa, fb);
                if !result.iter().any(|(k, _, _)| k == &key) {
                    result.push((key, fa.to_string(), fb.to_string()));
                }
            }
        }
    }

    result
}

/// Build or load per-field vector indexes for one side.
///
/// For each embedding match field, creates a VectorDB instance (flat or
/// usearch, depending on `backend`), encodes field values, and upserts
/// vectors into it. Each per-field index handles its own block routing
/// (usearch) or stores all vectors flat.
///
/// `cache_dir`: if `Some`, attempts to load from / save to per-field
/// cache paths in this directory. If `None`, builds without caching.
pub fn build_or_load_field_indexes(
    backend: &str,
    cache_dir: Option<&str>,
    records: &HashMap<String, Record>,
    ids: &[String],
    config: &Config,
    is_a_side: bool,
    encoder_pool: &EncoderPool,
    dim: usize,
) -> Result<field_indexes::FieldIndexes, MelderError> {
    let emb_fields = embedding_field_keys(config);
    if emb_fields.is_empty() {
        return Ok(field_indexes::FieldIndexes::new(dim));
    }

    let side = if is_a_side { "A" } else { "B" };
    let side_prefix = if is_a_side { "a" } else { "b" };
    let side_enum = if is_a_side { Side::A } else { Side::B };
    let blocking_config = if config.blocking.enabled {
        Some(&config.blocking)
    } else {
        None
    };

    let mut fi = field_indexes::FieldIndexes::new(dim);

    for (field_key, field_a_name, field_b_name) in &emb_fields {
        let field_name = if is_a_side {
            field_a_name
        } else {
            field_b_name
        };

        // Try loading from cache
        if let Some(dir) = cache_dir {
            let cache_path = field_indexes::field_cache_path(dir, side_prefix, field_key);
            let path = Path::new(&cache_path);
            if !is_index_stale(backend, path, ids.len()).unwrap_or(true) {
                let load_start = Instant::now();
                match load_index(backend, path) {
                    Ok(index) => {
                        eprintln!(
                            "Loaded {} field index [{}] from cache: {} vecs in {:.1}ms",
                            side,
                            field_key,
                            index.len(),
                            load_start.elapsed().as_secs_f64() * 1000.0
                        );
                        fi.insert_index(field_key.clone(), index);
                        continue;
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: {} field index [{}] cache load failed ({}), rebuilding...",
                            side, field_key, e
                        );
                    }
                }
            }
        }

        // Build from scratch
        eprintln!(
            "Building {} field index [{}] ({} records)...",
            side,
            field_key,
            ids.len()
        );
        let build_start = Instant::now();
        let index = new_index(backend, dim, blocking_config);
        let batch_size = 256;

        for (batch_idx, chunk) in ids.chunks(batch_size).enumerate() {
            let texts: Vec<String> = chunk
                .iter()
                .map(|id| {
                    let record = records
                        .get(id)
                        .expect("id from ids vec must exist in records");
                    record
                        .get(field_name)
                        .map(|v| v.trim().to_string())
                        .unwrap_or_default()
                })
                .collect();

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let vecs = encoder_pool
                .encode(&text_refs)
                .map_err(MelderError::Encoder)?;

            for (i, vec) in vecs.into_iter().enumerate() {
                let id = &chunk[i];
                let record = records.get(id).unwrap();
                index
                    .upsert(id, &vec, record, side_enum)
                    .map_err(|e| MelderError::Other(anyhow::anyhow!("{}", e)))?;
            }

            let done = (batch_idx + 1) * batch_size;
            if done % 2000 == 0 || done >= ids.len() {
                eprintln!(
                    "  {} field [{}]: encoded {}/{}",
                    side,
                    field_key,
                    done.min(ids.len()),
                    ids.len()
                );
            }
        }

        let build_elapsed = build_start.elapsed();
        eprintln!(
            "{} field index [{}] built: {} vecs in {:.1}s ({:.0} records/sec)",
            side,
            field_key,
            index.len(),
            build_elapsed.as_secs_f64(),
            ids.len() as f64 / build_elapsed.as_secs_f64()
        );

        // Save to cache
        if let Some(dir) = cache_dir {
            let cache_path = field_indexes::field_cache_path(dir, side_prefix, field_key);
            let path = Path::new(&cache_path);
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).ok();
                }
            }
            if let Err(e) = index.save(path) {
                eprintln!(
                    "Warning: failed to save {} field index [{}] cache: {}",
                    side, field_key, e
                );
            } else {
                eprintln!(
                    "Saved {} field index [{}] cache to {}",
                    side, field_key, cache_path
                );
            }
        }

        fi.insert_index(field_key.clone(), index);
    }

    eprintln!(
        "{} field indexes ready: {} total vecs across {} fields",
        side,
        fi.len(),
        emb_fields.len()
    );

    Ok(fi)
}
