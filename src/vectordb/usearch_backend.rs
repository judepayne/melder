//! UsearchVectorDB: per-block HNSW indices via usearch behind the VectorDB trait.
//!
//! Each "block" is determined by a record's blocking field values (composite key).
//! Records with identical blocking keys share an HNSW index. Blocks are created
//! lazily on first insert. Blocking-disabled configs use a single default block.
//!
//! Key mapping: melder uses `String` IDs; usearch uses `u64` keys. A monotonic
//! counter assigns u64 keys, with bidirectional `HashMap`s for translation.
//!
//! Concurrency: each block has its own `RwLock<BlockState>`, so writes to
//! different blocks never contend and reads are fully concurrent.
//!
//! Orphan retention: `remove()` removes a record from the key mappings and
//! marks it removed in usearch, but the vector data persists in the HNSW graph.
//! This is intentional — search is scale-insensitive (O(log N)) and orphan
//! vectors are harmless noise.
//!
//! ## Lock ordering
//!
//! When acquiring multiple locks, always follow this order to prevent
//! deadlocks. Never acquire a higher-numbered lock while holding a
//! lower-numbered one.
//!
//! 1. `block_router` — maps block keys to block indices
//! 2. `blocks` — the `Vec` of per-block `RwLock<BlockState>`s
//! 3. `blocks[i]` — individual block state (index + key maps)
//! 4. `record_block` — maps record IDs to their block index
//! 5. `text_hashes` — text-hash sidecar store
//!
//! The `next_key` field is an `AtomicU64` (lock-free) and can be accessed
//! at any point regardless of which locks are held.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{
    LazyLock, RwLock,
    atomic::{AtomicU64, Ordering},
};

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::config::schema::{BlockingConfig, BlockingFieldPair};
use crate::models::{Record, Side};

use super::texthash::TextHashStore;
use super::{SearchResult, VectorDB, VectorDBError};

// ---------------------------------------------------------------------------
// ScalarKind parsing helper
// ---------------------------------------------------------------------------

/// Map a config string to a usearch `ScalarKind`.
///
/// Allowed values: `"f32"`, `"f16"`, `"bf16"`. Everything else falls back to
/// `F32` (the caller — `loader.rs` — already validates the input).
fn parse_scalar_kind(s: &str) -> ScalarKind {
    match s {
        "f16" => ScalarKind::F16,
        "bf16" => ScalarKind::BF16,
        _ => ScalarKind::F32,
    }
}

// ---------------------------------------------------------------------------
// Block key: composite of lowercased/trimmed blocking field values
// ---------------------------------------------------------------------------

/// A block key is a vector of lowercased, trimmed field values.
/// For AND blocking with fields [(country_a, country_b)], a record
/// with country="US" produces block key ["us"].
type BlockKey = Vec<String>;

/// The default block key for records when blocking is disabled or
/// all blocking field values are empty.
static DEFAULT_BLOCK_KEY: LazyLock<BlockKey> = LazyLock::new(|| vec!["__default__".to_string()]);

fn default_block_key() -> BlockKey {
    DEFAULT_BLOCK_KEY.clone()
}

/// Compute the block key for a record given blocking field pairs and its side.
fn compute_block_key(record: &Record, fields: &[BlockingFieldPair], side: Side) -> BlockKey {
    if fields.is_empty() {
        return default_block_key();
    }
    let key: BlockKey = fields
        .iter()
        .map(|pair| {
            let field_name = match side {
                Side::A => &pair.field_a,
                Side::B => &pair.field_b,
            };
            record
                .get(field_name)
                .map(|v| v.trim().to_lowercase())
                .unwrap_or_default()
        })
        .collect();
    // If all values are empty, use the default block to avoid creating
    // degenerate single-record blocks for records with missing fields.
    if key.iter().all(|v| v.is_empty()) {
        default_block_key()
    } else {
        key
    }
}

// ---------------------------------------------------------------------------
// Per-block state
// ---------------------------------------------------------------------------

/// State for a single block: one HNSW index + bidirectional key mappings.
struct BlockState {
    /// The usearch HNSW index for this block.
    index: Index,
    /// String ID → u64 key.
    id_to_key: HashMap<String, u64>,
    /// u64 key → String ID.
    key_to_id: HashMap<u64, String>,
}

impl BlockState {
    fn new(
        dim: usize,
        quantization: ScalarKind,
        expansion_search: usize,
    ) -> Result<Self, VectorDBError> {
        let opts = IndexOptions {
            dimensions: dim,
            metric: MetricKind::IP,
            quantization,
            connectivity: 0,  // usearch default (typically 16)
            expansion_add: 0, // usearch default
            expansion_search,
            multi: false,
        };
        let index = Index::new(&opts).map_err(|e| VectorDBError::Backend(e.to_string()))?;
        Ok(Self {
            index,
            id_to_key: HashMap::new(),
            key_to_id: HashMap::new(),
        })
    }

    /// Number of active (non-removed) records.
    fn active_count(&self) -> usize {
        self.id_to_key.len()
    }
}

// ---------------------------------------------------------------------------
// UsearchVectorDB
// ---------------------------------------------------------------------------

/// Per-block HNSW vector index implementing `VectorDB`.
///
/// O(log N) search within each block. Suitable for any dataset size.
/// Blocks are determined by blocking config fields and created lazily.
pub struct UsearchVectorDB {
    dim: usize,
    /// Usearch vector quantization level (F32, F16, or BF16).
    quantization: ScalarKind,
    /// HNSW search beam width. 0 = usearch default.
    expansion_search: usize,
    /// Blocking field pairs for computing block keys.
    blocking_fields: Vec<BlockingFieldPair>,
    /// Whether blocking is enabled.
    blocking_enabled: bool,
    /// Block key → block index in `blocks` vec.
    block_router: RwLock<HashMap<BlockKey, usize>>,
    /// Per-block state, each behind its own RwLock.
    blocks: RwLock<Vec<RwLock<BlockState>>>,
    /// Global record ID → block index, for O(1) lookup on get/remove/contains.
    record_block: RwLock<HashMap<String, usize>>,
    /// Monotonic counter for assigning u64 keys to usearch.
    next_key: AtomicU64,
    /// Text-hash store (global, not per-block) for incremental re-encoding.
    text_hashes: RwLock<TextHashStore>,
}

impl std::fmt::Debug for UsearchVectorDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_len = self.len();
        let num_blocks = self.blocks.read().map(|b| b.len()).unwrap_or(0);
        f.debug_struct("UsearchVectorDB")
            .field("dim", &self.dim)
            .field("len", &total_len)
            .field("blocks", &num_blocks)
            .finish()
    }
}

impl UsearchVectorDB {
    /// Create a new UsearchVectorDB.
    ///
    /// - `dim`: embedding dimension (e.g. 384 for MiniLM-L6-v2).
    /// - `blocking_config`: the job's blocking configuration. If `None` or
    ///   disabled, all records go into a single default block.
    /// - `quantization`: vector quantization level (`"f32"`, `"f16"`, or
    ///   `"bf16"`). Defaults to `"f32"` if unrecognised.
    ///
    /// Used in tests and wherever emb_specs are not needed. The text-hash
    /// store will be empty (no-op on upsert).
    pub fn new(dim: usize, blocking_config: Option<&BlockingConfig>) -> Self {
        Self::new_with_emb_specs(dim, blocking_config, Vec::new(), "f32", 0)
    }

    /// Create a new UsearchVectorDB with embedding specs for text hashing.
    ///
    /// Called by `new_index()` in mod.rs when building the combined index.
    /// `quantization` controls the usearch scalar kind for storage/search.
    /// `expansion_search` sets the HNSW search beam width (0 = usearch default).
    pub fn new_with_emb_specs(
        dim: usize,
        blocking_config: Option<&BlockingConfig>,
        emb_specs: Vec<(String, String, f64)>,
        quantization: &str,
        expansion_search: usize,
    ) -> Self {
        let (blocking_enabled, blocking_fields) = match blocking_config {
            Some(cfg) if cfg.enabled && !cfg.fields.is_empty() => (true, cfg.fields.clone()),
            _ => (false, Vec::new()),
        };
        Self {
            dim,
            quantization: parse_scalar_kind(quantization),
            expansion_search,
            blocking_fields,
            blocking_enabled,
            block_router: RwLock::new(HashMap::new()),
            blocks: RwLock::new(Vec::new()),
            record_block: RwLock::new(HashMap::new()),
            next_key: AtomicU64::new(1), // start at 1; 0 reserved
            text_hashes: RwLock::new(TextHashStore::new(emb_specs)),
        }
    }

    /// Get or create the block index for a given block key.
    /// Returns the index into the `blocks` vec.
    fn get_or_create_block(&self, block_key: &BlockKey) -> Result<usize, VectorDBError> {
        // Fast path: block already exists (read lock only).
        {
            let router = self.block_router.read().unwrap_or_else(|e| e.into_inner());
            if let Some(&idx) = router.get(block_key) {
                return Ok(idx);
            }
        }
        // Slow path: need to create a new block.
        let mut router = self.block_router.write().unwrap_or_else(|e| e.into_inner());
        // Double-check after acquiring write lock (another thread may have created it).
        if let Some(&idx) = router.get(block_key) {
            return Ok(idx);
        }
        let state = BlockState::new(self.dim, self.quantization, self.expansion_search)?;
        let mut blocks = self.blocks.write().unwrap_or_else(|e| e.into_inner());
        let idx = blocks.len();
        blocks.push(RwLock::new(state));
        router.insert(block_key.clone(), idx);
        Ok(idx)
    }

    /// Look up the block index for a record's block key.
    /// Returns `None` if the block doesn't exist (no records from this key yet).
    fn find_block(&self, block_key: &BlockKey) -> Option<usize> {
        let router = self.block_router.read().unwrap_or_else(|e| e.into_inner());
        router.get(block_key).copied()
    }

    /// Compute the block key for a record.
    fn block_key_for(&self, record: &Record, side: Side) -> BlockKey {
        if !self.blocking_enabled {
            return default_block_key();
        }
        compute_block_key(record, &self.blocking_fields, side)
    }

    /// Allocate a new u64 key for usearch.
    fn alloc_key(&self) -> u64 {
        self.next_key.fetch_add(1, Ordering::Relaxed)
    }
}

impl VectorDB for UsearchVectorDB {
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

        let block_key = self.block_key_for(record, side);
        let block_idx = self.get_or_create_block(&block_key)?;

        // If the record already exists in a *different* block, remove it there first.
        {
            let rb = self.record_block.read().unwrap_or_else(|e| e.into_inner());
            if let Some(&old_block_idx) = rb.get(id)
                && old_block_idx != block_idx
            {
                drop(rb);
                // Remove from old block.
                let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
                let mut old_state = blocks[old_block_idx]
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(old_key) = old_state.id_to_key.remove(id) {
                    old_state.key_to_id.remove(&old_key);
                    // Mark removed in usearch (orphan retention — vector stays).
                    let _ = old_state.index.remove(old_key);
                }
                drop(old_state);
                let mut rb = self.record_block.write().unwrap_or_else(|e| e.into_inner());
                rb.remove(id);
            }
        }

        // Insert/replace in the target block.
        let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
        let mut state = blocks[block_idx].write().unwrap_or_else(|e| e.into_inner());

        // If the ID already exists in this block, remove the old usearch key
        // so the new vector gets a fresh HNSW node.
        if let Some(old_key) = state.id_to_key.remove(id) {
            state.key_to_id.remove(&old_key);
            let _ = state.index.remove(old_key);
        }

        let usearch_key = self.alloc_key();

        // Ensure capacity (usearch requires pre-reservation).
        let needed = state.index.size() + 1;
        if needed > state.index.capacity() {
            state
                .index
                .reserve(needed.max(state.index.capacity() * 2).max(64))
                .map_err(|e| VectorDBError::Backend(e.to_string()))?;
        }

        state
            .index
            .add(usearch_key, vec)
            .map_err(|e| VectorDBError::Backend(e.to_string()))?;

        let id_owned = id.to_string();
        state.id_to_key.insert(id_owned.clone(), usearch_key);
        state.key_to_id.insert(usearch_key, id_owned.clone());

        drop(state);
        drop(blocks);

        // Update global record→block mapping.
        let mut rb = self.record_block.write().unwrap_or_else(|e| e.into_inner());
        rb.insert(id_owned, block_idx);
        drop(rb);

        // Update text hash (no-op if emb_specs is empty).
        let mut th = self.text_hashes.write().unwrap_or_else(|e| e.into_inner());
        th.update(id, record, side);

        Ok(())
    }

    fn remove(&self, id: &str) -> Result<bool, VectorDBError> {
        let block_idx = {
            let rb = self.record_block.read().unwrap_or_else(|e| e.into_inner());
            match rb.get(id) {
                Some(&idx) => idx,
                None => return Ok(false),
            }
        };

        let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
        let mut state = blocks[block_idx].write().unwrap_or_else(|e| e.into_inner());

        if let Some(usearch_key) = state.id_to_key.remove(id) {
            state.key_to_id.remove(&usearch_key);
            // Mark removed in usearch (vector stays as orphan).
            let _ = state.index.remove(usearch_key);
        }

        drop(state);
        drop(blocks);

        let mut rb = self.record_block.write().unwrap_or_else(|e| e.into_inner());
        rb.remove(id);
        drop(rb);

        // Remove text hash entry.
        let mut th = self.text_hashes.write().unwrap_or_else(|e| e.into_inner());
        th.remove(id);

        Ok(true)
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        record: &Record,
        side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError> {
        if query.len() != self.dim {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let block_key = self.block_key_for(record, side);
        let block_idx = match self.find_block(&block_key) {
            Some(idx) => idx,
            None => return Ok(Vec::new()), // no block = no results
        };

        let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
        let state = blocks[block_idx].read().unwrap_or_else(|e| e.into_inner());

        if state.active_count() == 0 {
            return Ok(Vec::new());
        }

        // Search with extra margin since usearch may return removed keys
        // that haven't been fully purged from the graph.
        let search_k = k.min(state.index.size());
        if search_k == 0 {
            return Ok(Vec::new());
        }

        let matches = state
            .index
            .search(query, search_k)
            .map_err(|e| VectorDBError::Backend(e.to_string()))?;

        let mut results: Vec<SearchResult> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&usearch_key, &distance)| {
                let id = state.key_to_id.get(&usearch_key)?;
                Some(SearchResult {
                    id: id.clone(),
                    score: 1.0 - distance, // IP distance = 1 - dot_product
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &HashSet<String>,
        record: &Record,
        side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError> {
        if query.len() != self.dim {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        if allowed.is_empty() {
            return Ok(Vec::new());
        }

        let block_key = self.block_key_for(record, side);
        let block_idx = match self.find_block(&block_key) {
            Some(idx) => idx,
            None => return Ok(Vec::new()),
        };

        let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
        let state = blocks[block_idx].read().unwrap_or_else(|e| e.into_inner());

        if state.active_count() == 0 {
            return Ok(Vec::new());
        }

        // For filtered search on an HNSW index, we search unfiltered with a
        // larger K and post-filter. This is much faster than usearch's native
        // filtered_search (which is 12× slower at low filter acceptance).
        let expanded_k = (k * 10).min(state.index.size()).max(k);
        if expanded_k == 0 {
            return Ok(Vec::new());
        }

        let matches = state
            .index
            .search(query, expanded_k)
            .map_err(|e| VectorDBError::Backend(e.to_string()))?;

        let mut results: Vec<SearchResult> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&usearch_key, &distance)| {
                let id = state.key_to_id.get(&usearch_key)?;
                if !allowed.contains(id.as_str()) {
                    return None;
                }
                Some(SearchResult {
                    id: id.clone(),
                    score: 1.0 - distance,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    fn get(&self, id: &str) -> Result<Option<Vec<f32>>, VectorDBError> {
        let block_idx = {
            let rb = self.record_block.read().unwrap_or_else(|e| e.into_inner());
            match rb.get(id) {
                Some(&idx) => idx,
                None => return Ok(None),
            }
        };

        let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
        let state = blocks[block_idx].read().unwrap_or_else(|e| e.into_inner());

        let usearch_key = match state.id_to_key.get(id) {
            Some(&k) => k,
            None => return Ok(None),
        };

        let mut buffer = vec![0.0f32; self.dim];
        let found = state
            .index
            .get(usearch_key, &mut buffer)
            .map_err(|e| VectorDBError::Backend(e.to_string()))?;

        if found == 0 {
            return Ok(None);
        }

        Ok(Some(buffer))
    }

    fn contains(&self, id: &str) -> bool {
        let rb = self.record_block.read().unwrap_or_else(|e| e.into_inner());
        rb.contains_key(id)
    }

    fn len(&self) -> usize {
        let rb = self.record_block.read().unwrap_or_else(|e| e.into_inner());
        rb.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn save(&self, path: &Path) -> Result<(), VectorDBError> {
        // Save format: directory with one `.usearch` file per block plus
        // a `manifest.json` with key mappings and block metadata.
        let dir = path.with_extension("usearchdb");
        std::fs::create_dir_all(&dir)?;

        let blocks = self.blocks.read().unwrap_or_else(|e| e.into_inner());
        let router = self.block_router.read().unwrap_or_else(|e| e.into_inner());
        let rb = self.record_block.read().unwrap_or_else(|e| e.into_inner());

        // Build manifest.
        let mut manifest = ManifestData {
            dim: self.dim,
            blocking_enabled: self.blocking_enabled,
            blocking_fields: self.blocking_fields.clone(),
            next_key: self.next_key.load(Ordering::Relaxed),
            blocks: Vec::new(),
            record_blocks: rb.clone(),
        };

        for (block_key, &block_idx) in router.iter() {
            let state = blocks[block_idx].read().unwrap_or_else(|e| e.into_inner());

            // Save the usearch index.
            let index_path = dir.join(format!("block_{}.usearch", block_idx));
            state
                .index
                .save(
                    index_path
                        .to_str()
                        .ok_or_else(|| VectorDBError::Backend("non-UTF8 path".to_string()))?,
                )
                .map_err(|e| VectorDBError::Backend(e.to_string()))?;

            manifest.blocks.push(BlockManifest {
                id: block_idx,
                key: block_key.clone(),
                id_to_key: state.id_to_key.clone(),
                key_to_id: state.key_to_id.clone(),
            });
        }

        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| VectorDBError::Serialization(e.to_string()))?;
        std::fs::write(&manifest_path, json)?;

        // Save text-hash sidecar alongside the .usearchdb directory.
        // The sidecar path is derived from `path` (the .index base path),
        // not from `dir` (the .usearchdb directory).
        let th = self.text_hashes.read().unwrap_or_else(|e| e.into_inner());
        th.save(path)
            .map_err(|e| VectorDBError::Backend(format!("texthash save: {}", e)))?;

        Ok(())
    }

    fn stored_text_hashes(&self) -> HashMap<String, u64> {
        self.text_hashes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .all()
            .clone()
    }

    fn text_hash_for(&self, id: &str) -> Option<u64> {
        self.text_hashes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
    }

    fn is_stale(path: &Path, expected_count: usize) -> Result<bool, VectorDBError>
    where
        Self: Sized,
    {
        let dir = path.with_extension("usearchdb");
        let manifest_path = dir.join("manifest.json");
        if !manifest_path.exists() {
            return Ok(true);
        }
        let data = std::fs::read_to_string(&manifest_path)?;
        let manifest: ManifestData =
            serde_json::from_str(&data).map_err(|e| VectorDBError::Serialization(e.to_string()))?;
        let actual_count = manifest.record_blocks.len();
        Ok(actual_count != expected_count)
    }
}

// ---------------------------------------------------------------------------
// Loading from disk
// ---------------------------------------------------------------------------

impl UsearchVectorDB {
    /// Load a previously saved UsearchVectorDB from disk.
    ///
    /// `quantization` controls the scalar kind used when re-creating the
    /// usearch `IndexOptions`. It must match whatever was used when the index
    /// was originally built (the cache path already encodes the quantization
    /// via `spec_hash`, so mismatches are prevented at a higher level).
    ///
    /// `vector_index_mode` controls how each block index is loaded:
    /// - `"load"` (default): full in-memory load via `index.load()`.
    /// - `"mmap"`: memory-mapped via `index.view()`. Lower peak RAM but
    ///   read-only — any subsequent `add()` call will return an error.
    pub fn load(
        path: &Path,
        quantization: &str,
        vector_index_mode: &str,
        expansion_search: usize,
    ) -> Result<Self, VectorDBError> {
        let scalar_kind = parse_scalar_kind(quantization);
        let use_mmap = vector_index_mode == "mmap";
        let dir = path.with_extension("usearchdb");
        let manifest_path = dir.join("manifest.json");
        let data = std::fs::read_to_string(&manifest_path)?;
        let manifest: ManifestData =
            serde_json::from_str(&data).map_err(|e| VectorDBError::Serialization(e.to_string()))?;

        let dim = manifest.dim;
        let mut block_router = HashMap::new();
        let mut blocks_vec: Vec<RwLock<BlockState>> = Vec::new();
        // Map from old block ID (saved in manifest) to new Vec position.
        let mut id_remap: HashMap<usize, usize> = HashMap::new();

        for bm in &manifest.blocks {
            let opts = IndexOptions {
                dimensions: dim,
                metric: MetricKind::IP,
                quantization: scalar_kind,
                connectivity: 0,
                expansion_add: 0,
                expansion_search,
                multi: false,
            };
            let index = Index::new(&opts).map_err(|e| VectorDBError::Backend(e.to_string()))?;
            let index_path = dir.join(format!("block_{}.usearch", bm.id));
            let index_path_str = index_path
                .to_str()
                .ok_or_else(|| VectorDBError::Backend("non-UTF8 path".to_string()))?;
            if use_mmap {
                index
                    .view(index_path_str)
                    .map_err(|e| VectorDBError::Backend(e.to_string()))?;
            } else {
                index
                    .load(index_path_str)
                    .map_err(|e| VectorDBError::Backend(e.to_string()))?;
            }

            let state = BlockState {
                index,
                id_to_key: bm.id_to_key.clone(),
                key_to_id: bm.key_to_id.clone(),
            };

            let new_idx = blocks_vec.len();
            id_remap.insert(bm.id, new_idx);
            blocks_vec.push(RwLock::new(state));
            block_router.insert(bm.key.clone(), new_idx);
        }

        // Remap record_blocks from old block IDs to new Vec positions.
        let record_blocks: HashMap<String, usize> = manifest
            .record_blocks
            .into_iter()
            .filter_map(|(record_id, old_block_id)| {
                id_remap
                    .get(&old_block_id)
                    .map(|&new_idx| (record_id, new_idx))
            })
            .collect();

        // Restore text-hash sidecar (empty store if sidecar is absent).
        let text_hashes = TextHashStore::load(path)
            .map_err(|e| VectorDBError::Backend(format!("texthash load: {}", e)))?;

        Ok(Self {
            dim,
            quantization: scalar_kind,
            expansion_search,
            blocking_fields: manifest.blocking_fields,
            blocking_enabled: manifest.blocking_enabled,
            block_router: RwLock::new(block_router),
            blocks: RwLock::new(blocks_vec),
            record_block: RwLock::new(record_blocks),
            next_key: AtomicU64::new(manifest.next_key),
            text_hashes: RwLock::new(text_hashes),
        })
    }
}

// ---------------------------------------------------------------------------
// Manifest serialization types
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct ManifestData {
    dim: usize,
    blocking_enabled: bool,
    blocking_fields: Vec<BlockingFieldPair>,
    next_key: u64,
    blocks: Vec<BlockManifest>,
    record_blocks: HashMap<String, usize>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BlockManifest {
    id: usize,
    key: BlockKey,
    id_to_key: HashMap<String, u64>,
    key_to_id: HashMap<u64, String>,
}
