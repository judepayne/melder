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

pub mod flat;
pub mod manifest;
pub mod texthash;
#[cfg(feature = "usearch")]
pub mod usearch_backend;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::time::Instant;

use manifest::{StaleReason, blocking_hash, check_manifest, make_manifest, write_manifest};
use texthash::compute_text_hash;

use crate::config::Config;
use crate::config::schema::BlockingConfig;
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

    /// Return a snapshot of all stored text hashes.
    ///
    /// Used by `build_or_load_combined_index` to diff against current records
    /// and determine which ones need re-encoding. Returns an empty map if the
    /// backend was created without emb_specs (e.g. in tests).
    fn stored_text_hashes(&self) -> HashMap<String, u64>;

    /// Look up the stored text hash for a single record.
    ///
    /// Returns `None` if the ID has no stored hash (new record or backend
    /// created without emb_specs). Used by the live-mode encoding-skip
    /// optimisation to avoid re-encoding records whose embedding fields
    /// have not changed.
    fn text_hash_for(&self, id: &str) -> Option<u64>;
}

// ---------------------------------------------------------------------------
// Backend dispatch: factory functions that select flat vs usearch at runtime
// ---------------------------------------------------------------------------

/// Create a new empty vector index for the given backend.
///
/// - `blocking_config` is only used by the usearch backend (for per-block
///   routing). The flat backend ignores it.
/// - `emb_specs` are stored inside the index's `TextHashStore` so that
///   subsequent upserts can record text hashes for incremental re-encoding.
pub fn new_index(
    backend: &str,
    dim: usize,
    #[allow(unused_variables)] blocking_config: Option<&BlockingConfig>,
    emb_specs: &[(String, String, f64)],
    #[allow(unused_variables)] quantization: &str,
) -> Box<dyn VectorDB> {
    match backend {
        #[cfg(feature = "usearch")]
        "usearch" => Box::new(usearch_backend::UsearchVectorDB::new_with_emb_specs(
            dim,
            blocking_config,
            emb_specs.to_vec(),
            quantization,
        )),
        _ => Box::new(flat::FlatVectorDB::new_with_emb_specs(
            dim,
            emb_specs.to_vec(),
        )),
    }
}

/// Load a vector index from a cache file.
///
/// `vector_index_mode` controls whether the usearch backend loads the index
/// fully into memory (`"load"`, default) or memory-maps it (`"mmap"`).
/// The flat backend ignores this parameter.
pub fn load_index(
    backend: &str,
    path: &Path,
    #[allow(unused_variables)] quantization: &str,
    #[allow(unused_variables)] vector_index_mode: &str,
) -> Result<Box<dyn VectorDB>, VectorDBError> {
    match backend {
        #[cfg(feature = "usearch")]
        "usearch" => Ok(Box::new(usearch_backend::UsearchVectorDB::load(
            path,
            quantization,
            vector_index_mode,
        )?)),
        _ => Ok(Box::new(flat::FlatVectorDB::load(path)?)),
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
        _ => flat::FlatVectorDB::is_stale(path, expected_count),
    }
}

// ---------------------------------------------------------------------------
// Combined embedding index
// ---------------------------------------------------------------------------

/// Compute an 8-character FNV-1a hex hash of the embedding field spec string
/// and vector quantization setting.
///
/// The input is the ordered, semicolon-separated list of
/// `"field_a/field_b/weight"` tuples for all embedding match fields, plus
/// `";q=<quantization>"` suffix. Any change to field identity, order, weight,
/// or quantization produces a different hash, making the old cache path
/// unreachable and forcing a cold rebuild.
pub fn spec_hash(emb_specs: &[(String, String, f64)], vector_quantization: &str) -> String {
    let mut s: String = emb_specs
        .iter()
        .map(|(fa, fb, w)| format!("{}/{}/{:.6}", fa, fb, w))
        .collect::<Vec<_>>()
        .join(";");
    // Append quantization so that changing f32→f16 invalidates the cache.
    // "f32" is the default; we always include it for consistency.
    s.push_str(";q=");
    s.push_str(vector_quantization);
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    format!("{:08x}", h as u32)
}

/// Returns `(field_a, field_b, weight)` for every `method: embedding` match
/// field, in config order.
pub fn embedding_field_specs(config: &Config) -> Vec<(String, String, f64)> {
    config
        .match_fields
        .iter()
        .filter(|mf| mf.method == "embedding")
        .map(|mf| (mf.field_a.clone(), mf.field_b.clone(), mf.weight))
        .collect()
}

/// Derive the cache path for the combined embedding index.
///
/// Pattern: `{cache_dir}/{side_prefix}.combined_embedding_{hash}.index`
///
/// Example: `bench/cache/a.combined_embedding_a3f7c2b1.index`
pub fn combined_cache_path(cache_dir: &str, side_prefix: &str, hash: &str) -> PathBuf {
    Path::new(cache_dir).join(format!("{}.combined_embedding_{}.index", side_prefix, hash))
}

/// Encode a record's embedding fields into a single combined vector.
///
/// For each embedding field (in config order):
///   1. Extract the field value (empty string if field is missing).
///   2. Encode → 384-dim L2-normalised unit vector.
///   3. Scale every component by √weight.
///   4. Append to the output buffer.
///
/// Returns a vector of dimension `encoder_dim × N_embedding_fields`.
/// Returns an empty `Vec` if `emb_specs` is empty.
///
/// Mathematical property: for two records A and B with unit per-field vecs,
///   `dot(combined_A, combined_B) = Σᵢ wᵢ · cosine_sim(aᵢ, bᵢ)`
/// which equals the raw weighted embedding contribution used by `score_pair`.
pub fn encode_combined_vector(
    record: &Record,
    emb_specs: &[(String, String, f64)],
    encoder_pool: &EncoderPool,
    is_a_side: bool,
) -> Result<Vec<f32>, MelderError> {
    if emb_specs.is_empty() {
        return Ok(Vec::new());
    }

    let field_dim = encoder_pool.dim();
    let mut combined = Vec::with_capacity(field_dim * emb_specs.len());

    for (field_a, field_b, weight) in emb_specs {
        let field_name = if is_a_side { field_a } else { field_b };
        let text = record
            .get(field_name)
            .map(|v| v.trim().to_string())
            .unwrap_or_default();

        let mut unit_vec = encoder_pool
            .encode_one(&text)
            .map_err(MelderError::Encoder)?;

        // Scale by √weight so dot products equal the weighted cosine sum.
        let sqrt_w = weight.sqrt() as f32;
        for v in &mut unit_vec {
            *v *= sqrt_w;
        }

        combined.extend_from_slice(&unit_vec);
    }

    Ok(combined)
}

/// Build or load the single combined embedding index for one side.
///
/// Cache path: `{cache_dir}/{side_prefix}.combined_embedding_{hash}.index`
/// Index dimension: `encoder_pool.dim() × N_embedding_fields`
///
/// Returns `None` if there are no embedding fields in config.
///
/// ## Cache decision logic (7 steps)
///
/// 1. **Manifest check** — any config mismatch → cold build with a log message.
/// 2. **Load** — load vectors + TextHashStore; failure → warn → cold build.
/// 3. **Diff** — FNV-hash each record's text; compare to stored hashes.
/// 4. **Full hit** — nothing to do → update manifest, return loaded index.
/// 5. **Threshold** — >90% of records changed → cold build is more efficient.
/// 6. **Incremental** — encode + upsert changed records, remove deleted ones.
/// 7. **Cold build** — encode everything from scratch.
///
/// When `skip_deletes` is true (live mode), records present in the cache but
/// absent from the dataset are retained rather than removed.  This preserves
/// vectors that were added via the API and persisted at shutdown — the WAL
/// replay will confirm their presence.  Batch and tune modes pass `false` so
/// stale records are cleaned up normally.
#[allow(clippy::too_many_arguments)]
pub fn build_or_load_combined_index(
    backend: &str,
    cache_dir: Option<&str>,
    records: &HashMap<String, Record>,
    ids: &[String],
    config: &Config,
    is_a_side: bool,
    encoder_pool: &EncoderPool,
    skip_deletes: bool,
) -> Result<Option<Box<dyn VectorDB>>, MelderError> {
    let emb_specs = embedding_field_specs(config);
    if emb_specs.is_empty() {
        return Ok(None);
    }

    let side = if is_a_side { "A" } else { "B" };
    let side_prefix = if is_a_side { "a" } else { "b" };
    let side_enum = if is_a_side { Side::A } else { Side::B };
    let blocking_config = if config.blocking.enabled {
        Some(&config.blocking)
    } else {
        None
    };

    let field_dim = encoder_pool.dim();
    let combined_dim = field_dim * emb_specs.len();
    let vq = config
        .performance
        .vector_quantization
        .as_deref()
        .unwrap_or("f32");
    let vim = config
        .performance
        .vector_index_mode
        .as_deref()
        .unwrap_or("load");
    let current_spec_hash = spec_hash(&emb_specs, vq);
    let current_blocking_hash = blocking_hash(&config.blocking);
    let current_model = &config.embeddings.model;

    // -----------------------------------------------------------------------
    // Cache path
    // -----------------------------------------------------------------------
    let cache_path_opt: Option<PathBuf> =
        cache_dir.map(|dir| combined_cache_path(dir, side_prefix, &current_spec_hash));

    // -----------------------------------------------------------------------
    // Steps 1–6: attempt to use / incrementally update existing cache
    // -----------------------------------------------------------------------
    if let Some(ref cache_path) = cache_path_opt {
        // Step 1: Manifest check
        let stale_reason = check_manifest(
            cache_path,
            &current_spec_hash,
            &current_blocking_hash,
            current_model,
        );

        let proceed_to_load = match &stale_reason {
            StaleReason::Fresh | StaleReason::Missing => true,
            reason => {
                eprintln!(
                    "Warning: {} combined index cache invalidated ({}), rebuilding from scratch.",
                    side, reason
                );
                false
            }
        };

        // For flat backend the cache is a single `.index` file; for usearch
        // it is a `.usearchdb` directory next to the `.index` path.  Check
        // whichever is appropriate so we don't miss an existing usearch cache.
        let cache_exists = cache_path.exists() || cache_path.with_extension("usearchdb").is_dir();

        if proceed_to_load && cache_exists {
            // Step 2: Load existing index
            let load_start = Instant::now();
            match load_index(backend, cache_path, vq, vim) {
                Err(e) => {
                    eprintln!(
                        "Warning: {} combined index cache load failed ({}), rebuilding...",
                        side, e
                    );
                    // fall through to cold build
                }
                Ok(index) => {
                    // Step 3: Diff — O(N) with no ONNX calls
                    let stored = index.stored_text_hashes();
                    let current_ids_set: HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();

                    let to_encode: Vec<&String> = ids
                        .iter()
                        .filter(|id| {
                            let record = records.get(id.as_str()).unwrap();
                            let current_hash = compute_text_hash(record, &emb_specs, side_enum);
                            stored.get(id.as_str()) != Some(&current_hash)
                        })
                        .collect();

                    let to_delete: Vec<String> = if skip_deletes {
                        // Live mode: extra records in cache are WAL additions
                        // that will be confirmed during replay — do not remove.
                        vec![]
                    } else {
                        stored
                            .keys()
                            .filter(|id| !current_ids_set.contains(id.as_str()))
                            .cloned()
                            .collect()
                    };

                    // Step 4: Full cache hit
                    if to_encode.is_empty() && to_delete.is_empty() {
                        let elapsed_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                        eprintln!(
                            "Loaded {} combined embedding index from cache: {} vecs, \
                             all fresh in {:.1}ms",
                            side,
                            index.len(),
                            elapsed_ms
                        );
                        // Refresh the manifest with the current record count.
                        let m = make_manifest(
                            current_spec_hash.clone(),
                            current_blocking_hash.clone(),
                            current_model.to_string(),
                            ids.len(),
                        );
                        if let Err(e) = write_manifest(cache_path, &m) {
                            eprintln!("Warning: could not update {} manifest: {}", side, e);
                        }
                        return Ok(Some(index));
                    }

                    // Step 5: Threshold check — if >90% changed, cold build
                    // is cheaper (better batching, no incremental overhead).
                    let total_changes = to_encode.len() + to_delete.len();
                    let total = ids.len().max(1);
                    if total_changes * 10 > total * 9 {
                        eprintln!(
                            "{} combined index: {}/{} records changed — cold rebuild \
                             is more efficient.",
                            side, total_changes, total
                        );
                        // fall through to cold build
                    } else {
                        // Step 6: Incremental path
                        eprintln!(
                            "{} combined index: {} to encode, {} to remove (incremental)",
                            side,
                            to_encode.len(),
                            to_delete.len()
                        );
                        let incr_start = Instant::now();

                        // Remove deleted records first.
                        for id in &to_delete {
                            index
                                .remove(id)
                                .map_err(|e| MelderError::Other(anyhow::anyhow!("{}", e)))?;
                        }

                        // Encode and upsert changed records in batches.
                        encode_and_upsert(
                            &*index,
                            &to_encode,
                            records,
                            &emb_specs,
                            encoder_pool,
                            combined_dim,
                            is_a_side,
                            side_enum,
                            side,
                        )?;

                        let incr_elapsed = incr_start.elapsed();
                        eprintln!(
                            "{} combined index updated: {} vecs total in {:.1}s",
                            side,
                            index.len(),
                            incr_elapsed.as_secs_f64()
                        );

                        // Save updated index + sidecars.
                        ensure_parent(cache_path);
                        if let Err(e) = index.save(cache_path) {
                            eprintln!(
                                "Warning: failed to save {} combined index cache: {}",
                                side, e
                            );
                        } else {
                            let m = make_manifest(
                                current_spec_hash.clone(),
                                current_blocking_hash.clone(),
                                current_model.to_string(),
                                ids.len(),
                            );
                            if let Err(e) = write_manifest(cache_path, &m) {
                                eprintln!("Warning: could not write {} manifest: {}", side, e);
                            }
                            eprintln!(
                                "Saved {} combined index cache to {}",
                                side,
                                cache_path.display()
                            );
                        }

                        return Ok(Some(index));
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 7: Cold build
    // -----------------------------------------------------------------------
    eprintln!(
        "Building {} combined embedding index ({} records, dim={}, {} field(s))...",
        side,
        ids.len(),
        combined_dim,
        emb_specs.len(),
    );
    let build_start = Instant::now();
    let index = new_index(backend, combined_dim, blocking_config, &emb_specs, vq);
    let all_ids: Vec<&String> = ids.iter().collect();

    encode_and_upsert(
        &*index,
        &all_ids,
        records,
        &emb_specs,
        encoder_pool,
        combined_dim,
        is_a_side,
        side_enum,
        side,
    )?;

    let build_elapsed = build_start.elapsed();
    eprintln!(
        "{} combined embedding index built: {} vecs in {:.1}s ({:.0} records/sec)",
        side,
        index.len(),
        build_elapsed.as_secs_f64(),
        ids.len() as f64 / build_elapsed.as_secs_f64()
    );

    // Save to cache.
    if let Some(ref cache_path) = cache_path_opt {
        ensure_parent(cache_path);
        if let Err(e) = index.save(cache_path) {
            eprintln!(
                "Warning: failed to save {} combined index cache: {}",
                side, e
            );
        } else {
            let m = make_manifest(
                current_spec_hash,
                current_blocking_hash,
                current_model.to_string(),
                ids.len(),
            );
            if let Err(e) = write_manifest(cache_path, &m) {
                eprintln!("Warning: could not write {} manifest: {}", side, e);
            }
            eprintln!(
                "Saved {} combined index cache to {}",
                side,
                cache_path.display()
            );
        }
    }

    Ok(Some(index))
}

// ---------------------------------------------------------------------------
// Internal helpers for build_or_load_combined_index
// ---------------------------------------------------------------------------

/// Encode `ids` in batches and upsert combined vectors into `index`.
///
/// One ONNX call per field per batch; fields are interleaved into combined
/// vectors before upserting. This is shared between cold build and incremental
/// encoding.
#[allow(clippy::too_many_arguments)]
fn encode_and_upsert(
    index: &dyn VectorDB,
    ids: &[&String],
    records: &HashMap<String, Record>,
    emb_specs: &[(String, String, f64)],
    encoder_pool: &EncoderPool,
    combined_dim: usize,
    is_a_side: bool,
    side_enum: Side,
    side_label: &str,
) -> Result<(), MelderError> {
    if ids.is_empty() {
        return Ok(());
    }
    let batch_size = 256;
    let total = ids.len();

    for (batch_idx, chunk) in ids.chunks(batch_size).enumerate() {
        let mut combined_vecs: Vec<Vec<f32>> = vec![Vec::with_capacity(combined_dim); chunk.len()];

        for (field_a, field_b, weight) in emb_specs {
            let field_name = if is_a_side { field_a } else { field_b };
            let texts: Vec<String> = chunk
                .iter()
                .map(|id| {
                    records
                        .get(id.as_str())
                        .expect("id must exist in records")
                        .get(field_name)
                        .map(|v| v.trim().to_string())
                        .unwrap_or_default()
                })
                .collect();

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let vecs = encoder_pool
                .encode(&text_refs)
                .map_err(MelderError::Encoder)?;

            let sqrt_w = weight.sqrt() as f32;
            for (i, mut vec) in vecs.into_iter().enumerate() {
                for v in &mut vec {
                    *v *= sqrt_w;
                }
                combined_vecs[i].extend_from_slice(&vec);
            }
        }

        for (i, id) in chunk.iter().enumerate() {
            let record = records.get(id.as_str()).unwrap();
            index
                .upsert(id, &combined_vecs[i], record, side_enum)
                .map_err(|e| MelderError::Other(anyhow::anyhow!("{}", e)))?;
        }

        let done = ((batch_idx + 1) * batch_size).min(total);
        if done % 1024 < batch_size || done >= total {
            eprint!(
                "\r  {} combined index: encoded {}/{} ({:.0}%)",
                side_label,
                done,
                total,
                done as f64 / total as f64 * 100.0,
            );
            if done >= total {
                eprintln!();
            }
        }
    }

    Ok(())
}

/// Create parent directories for `path` if they don't exist.
fn ensure_parent(path: &Path) {
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).ok();
    }
}

// ---------------------------------------------------------------------------
// Tests for new combined-index functions
// ---------------------------------------------------------------------------

#[cfg(test)]
mod combined_tests {
    use super::*;

    #[test]
    fn spec_hash_deterministic() {
        let specs = vec![
            (
                "legal_name".to_string(),
                "counterparty_name".to_string(),
                0.55_f64,
            ),
            (
                "short_name".to_string(),
                "counterparty_name".to_string(),
                0.20_f64,
            ),
        ];
        let h1 = spec_hash(&specs, "f32");
        let h2 = spec_hash(&specs, "f32");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 8);
    }

    #[test]
    fn spec_hash_changes_on_weight_change() {
        let specs1 = vec![("f_a".to_string(), "f_b".to_string(), 0.55_f64)];
        let specs2 = vec![("f_a".to_string(), "f_b".to_string(), 0.60_f64)];
        assert_ne!(spec_hash(&specs1, "f32"), spec_hash(&specs2, "f32"));
    }

    #[test]
    fn spec_hash_changes_on_field_change() {
        let specs1 = vec![("name_a".to_string(), "name_b".to_string(), 0.55_f64)];
        let specs2 = vec![("other_a".to_string(), "name_b".to_string(), 0.55_f64)];
        assert_ne!(spec_hash(&specs1, "f32"), spec_hash(&specs2, "f32"));
    }

    #[test]
    fn spec_hash_empty() {
        let h = spec_hash(&[], "f32");
        assert_eq!(h.len(), 8);
    }

    #[test]
    fn spec_hash_changes_on_quantization() {
        let specs = vec![("f_a".to_string(), "f_b".to_string(), 0.55_f64)];
        let h_f32 = spec_hash(&specs, "f32");
        let h_f16 = spec_hash(&specs, "f16");
        let h_bf16 = spec_hash(&specs, "bf16");
        assert_ne!(h_f32, h_f16, "f32 and f16 should produce different hashes");
        assert_ne!(
            h_f32, h_bf16,
            "f32 and bf16 should produce different hashes"
        );
        assert_ne!(
            h_f16, h_bf16,
            "f16 and bf16 should produce different hashes"
        );
    }

    #[test]
    fn combined_cache_path_format() {
        let path = combined_cache_path("bench/cache", "a", "a3f7c2b1");
        let expected = Path::new("bench/cache").join("a.combined_embedding_a3f7c2b1.index");
        assert_eq!(path, expected);
    }

    #[test]
    fn encode_combined_vector_empty_specs() {
        let pool =
            crate::encoder::EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("encoder init");
        let record: Record = std::collections::HashMap::new();
        let vec = encode_combined_vector(&record, &[], &pool, true).unwrap();
        assert!(vec.is_empty());
    }

    #[test]
    fn encode_combined_vector_single_field_dimension() {
        let pool =
            crate::encoder::EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("encoder init");
        let mut record: Record = std::collections::HashMap::new();
        record.insert("legal_name".to_string(), "Acme Corp".to_string());
        let specs = vec![(
            "legal_name".to_string(),
            "counterparty_name".to_string(),
            1.0_f64,
        )];
        let vec = encode_combined_vector(&record, &specs, &pool, true).unwrap();
        assert_eq!(vec.len(), pool.dim());
    }

    #[test]
    fn encode_combined_vector_two_fields_dimension() {
        let pool =
            crate::encoder::EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("encoder init");
        let mut record: Record = std::collections::HashMap::new();
        record.insert("legal_name".to_string(), "Acme Corp".to_string());
        record.insert("short_name".to_string(), "Acme".to_string());
        let specs = vec![
            (
                "legal_name".to_string(),
                "counterparty_name".to_string(),
                0.55_f64,
            ),
            (
                "short_name".to_string(),
                "counterparty_name".to_string(),
                0.20_f64,
            ),
        ];
        let vec = encode_combined_vector(&record, &specs, &pool, true).unwrap();
        assert_eq!(vec.len(), pool.dim() * 2);
    }

    #[test]
    fn encode_combined_vector_missing_field_no_panic() {
        let pool =
            crate::encoder::EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("encoder init");
        let record: Record = std::collections::HashMap::new(); // no fields
        let specs = vec![(
            "legal_name".to_string(),
            "counterparty_name".to_string(),
            0.55_f64,
        )];
        let vec = encode_combined_vector(&record, &specs, &pool, true).unwrap();
        // Should succeed with empty string encoded
        assert_eq!(vec.len(), pool.dim());
    }

    #[test]
    fn encode_combined_vector_dot_product_equals_weighted_cosine_sum() {
        // Mathematical property: dot(combined_A, combined_B) = Σᵢ wᵢ·cosᵢ
        let pool =
            crate::encoder::EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("encoder init");

        let w1 = 0.55_f64;
        let w2 = 0.20_f64;

        let specs = vec![
            ("f1_a".to_string(), "f1_b".to_string(), w1),
            ("f2_a".to_string(), "f2_b".to_string(), w2),
        ];

        // Build A record
        let mut rec_a: Record = std::collections::HashMap::new();
        rec_a.insert("f1_a".to_string(), "Goldman Sachs".to_string());
        rec_a.insert("f2_a".to_string(), "GS".to_string());

        // Build B record (for is_a_side=false, uses field_b names)
        let mut rec_b: Record = std::collections::HashMap::new();
        rec_b.insert("f1_b".to_string(), "Goldman Sachs Group".to_string());
        rec_b.insert("f2_b".to_string(), "Goldman Sachs".to_string());

        let combined_a = encode_combined_vector(&rec_a, &specs, &pool, true).unwrap();
        let combined_b = encode_combined_vector(&rec_b, &specs, &pool, false).unwrap();

        // dot(combined_A, combined_B) should ≈ w1·cos1 + w2·cos2
        let combined_dot: f32 = combined_a
            .iter()
            .zip(combined_b.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Per-field unit vecs
        let dim = pool.dim();
        let a1 = pool.encode_one("Goldman Sachs").unwrap();
        let b1 = pool.encode_one("Goldman Sachs Group").unwrap();
        let cos1: f32 = a1.iter().zip(b1.iter()).map(|(a, b)| a * b).sum();

        let a2 = pool.encode_one("GS").unwrap();
        let b2 = pool.encode_one("Goldman Sachs").unwrap();
        let cos2: f32 = a2.iter().zip(b2.iter()).map(|(a, b)| a * b).sum();

        let expected = w1 as f32 * cos1 + w2 as f32 * cos2;

        assert!(
            (combined_dot - expected).abs() < 0.001,
            "dot(combined_A, combined_B)={:.4} expected Σwᵢ·cosᵢ={:.4} (dim={})",
            combined_dot,
            expected,
            dim
        );
    }
}
