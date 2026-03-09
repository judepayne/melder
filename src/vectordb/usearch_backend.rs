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

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    RwLock,
};

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::config::schema::{BlockingConfig, BlockingFieldPair};
use crate::models::{Record, Side};

use super::{SearchResult, VectorDB, VectorDBError};

// ---------------------------------------------------------------------------
// Block key: composite of lowercased/trimmed blocking field values
// ---------------------------------------------------------------------------

/// A block key is a vector of lowercased, trimmed field values.
/// For AND blocking with fields [(country_a, country_b)], a record
/// with country="US" produces block key ["us"].
type BlockKey = Vec<String>;

/// The default block key for records when blocking is disabled or
/// all blocking field values are empty.
fn default_block_key() -> BlockKey {
    vec!["__default__".to_string()]
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
    /// Tracks which side each record belongs to (for post-filtering).
    id_to_side: HashMap<String, Side>,
}

impl BlockState {
    fn new(dim: usize) -> Result<Self, VectorDBError> {
        let opts = IndexOptions {
            dimensions: dim,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            connectivity: 0,  // usearch default (typically 16)
            expansion_add: 0, // usearch default
            expansion_search: 0,
            multi: false,
        };
        let index = Index::new(&opts).map_err(|e| VectorDBError::Backend(e.to_string()))?;
        Ok(Self {
            index,
            id_to_key: HashMap::new(),
            key_to_id: HashMap::new(),
            id_to_side: HashMap::new(),
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
    pub fn new(dim: usize, blocking_config: Option<&BlockingConfig>) -> Self {
        let (blocking_enabled, blocking_fields) = match blocking_config {
            Some(cfg) if cfg.enabled && !cfg.fields.is_empty() => (true, cfg.fields.clone()),
            _ => (false, Vec::new()),
        };
        Self {
            dim,
            blocking_fields,
            blocking_enabled,
            block_router: RwLock::new(HashMap::new()),
            blocks: RwLock::new(Vec::new()),
            record_block: RwLock::new(HashMap::new()),
            next_key: AtomicU64::new(1), // start at 1; 0 reserved
        }
    }

    /// Get or create the block index for a given block key.
    /// Returns the index into the `blocks` vec.
    fn get_or_create_block(&self, block_key: &BlockKey) -> Result<usize, VectorDBError> {
        // Fast path: block already exists (read lock only).
        {
            let router = self.block_router.read().unwrap();
            if let Some(&idx) = router.get(block_key) {
                return Ok(idx);
            }
        }
        // Slow path: need to create a new block.
        let mut router = self.block_router.write().unwrap();
        // Double-check after acquiring write lock (another thread may have created it).
        if let Some(&idx) = router.get(block_key) {
            return Ok(idx);
        }
        let state = BlockState::new(self.dim)?;
        let mut blocks = self.blocks.write().unwrap();
        let idx = blocks.len();
        blocks.push(RwLock::new(state));
        router.insert(block_key.clone(), idx);
        Ok(idx)
    }

    /// Look up the block index for a record's block key.
    /// Returns `None` if the block doesn't exist (no records from this key yet).
    fn find_block(&self, block_key: &BlockKey) -> Option<usize> {
        let router = self.block_router.read().unwrap();
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
            let rb = self.record_block.read().unwrap();
            if let Some(&old_block_idx) = rb.get(id) {
                if old_block_idx != block_idx {
                    drop(rb);
                    // Remove from old block.
                    let blocks = self.blocks.read().unwrap();
                    let mut old_state = blocks[old_block_idx].write().unwrap();
                    if let Some(old_key) = old_state.id_to_key.remove(id) {
                        old_state.key_to_id.remove(&old_key);
                        old_state.id_to_side.remove(id);
                        // Mark removed in usearch (orphan retention — vector stays).
                        let _ = old_state.index.remove(old_key);
                    }
                    drop(old_state);
                    let mut rb = self.record_block.write().unwrap();
                    rb.remove(id);
                }
            }
        }

        // Insert/replace in the target block.
        let blocks = self.blocks.read().unwrap();
        let mut state = blocks[block_idx].write().unwrap();

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

        state.id_to_key.insert(id.to_string(), usearch_key);
        state.key_to_id.insert(usearch_key, id.to_string());
        state.id_to_side.insert(id.to_string(), side);

        drop(state);
        drop(blocks);

        // Update global record→block mapping.
        let mut rb = self.record_block.write().unwrap();
        rb.insert(id.to_string(), block_idx);

        Ok(())
    }

    fn remove(&self, id: &str) -> Result<bool, VectorDBError> {
        let block_idx = {
            let rb = self.record_block.read().unwrap();
            match rb.get(id) {
                Some(&idx) => idx,
                None => return Ok(false),
            }
        };

        let blocks = self.blocks.read().unwrap();
        let mut state = blocks[block_idx].write().unwrap();

        if let Some(usearch_key) = state.id_to_key.remove(id) {
            state.key_to_id.remove(&usearch_key);
            state.id_to_side.remove(id);
            // Mark removed in usearch (vector stays as orphan).
            let _ = state.index.remove(usearch_key);
        }

        drop(state);
        drop(blocks);

        let mut rb = self.record_block.write().unwrap();
        rb.remove(id);

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

        let blocks = self.blocks.read().unwrap();
        let state = blocks[block_idx].read().unwrap();

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

        let blocks = self.blocks.read().unwrap();
        let state = blocks[block_idx].read().unwrap();

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
            let rb = self.record_block.read().unwrap();
            match rb.get(id) {
                Some(&idx) => idx,
                None => return Ok(None),
            }
        };

        let blocks = self.blocks.read().unwrap();
        let state = blocks[block_idx].read().unwrap();

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
        let rb = self.record_block.read().unwrap();
        if let Some(&block_idx) = rb.get(id) {
            let blocks = self.blocks.read().unwrap();
            let state = blocks[block_idx].read().unwrap();
            state.id_to_key.contains_key(id)
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        let rb = self.record_block.read().unwrap();
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

        let blocks = self.blocks.read().unwrap();
        let router = self.block_router.read().unwrap();
        let rb = self.record_block.read().unwrap();

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
            let state = blocks[block_idx].read().unwrap();

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
                id_to_side: state
                    .id_to_side
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            match v {
                                Side::A => "a".to_string(),
                                Side::B => "b".to_string(),
                            },
                        )
                    })
                    .collect(),
            });
        }

        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| VectorDBError::Serialization(e.to_string()))?;
        std::fs::write(&manifest_path, json)?;

        // Write a small marker at the original path so callers can check
        // existence without knowing the backend's directory layout.
        std::fs::write(path, b"usearchdb")?;

        Ok(())
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
    pub fn load(path: &Path) -> Result<Self, VectorDBError> {
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
                quantization: ScalarKind::F32,
                connectivity: 0,
                expansion_add: 0,
                expansion_search: 0,
                multi: false,
            };
            let index = Index::new(&opts).map_err(|e| VectorDBError::Backend(e.to_string()))?;
            let index_path = dir.join(format!("block_{}.usearch", bm.id));
            index
                .load(
                    index_path
                        .to_str()
                        .ok_or_else(|| VectorDBError::Backend("non-UTF8 path".to_string()))?,
                )
                .map_err(|e| VectorDBError::Backend(e.to_string()))?;

            let id_to_side: HashMap<String, Side> = bm
                .id_to_side
                .iter()
                .map(|(k, v)| {
                    let side = if v == "b" { Side::B } else { Side::A };
                    (k.clone(), side)
                })
                .collect();

            let state = BlockState {
                index,
                id_to_key: bm.id_to_key.clone(),
                key_to_id: bm.key_to_id.clone(),
                id_to_side,
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

        Ok(Self {
            dim,
            blocking_fields: manifest.blocking_fields,
            blocking_enabled: manifest.blocking_enabled,
            block_router: RwLock::new(block_router),
            blocks: RwLock::new(blocks_vec),
            record_block: RwLock::new(record_blocks),
            next_key: AtomicU64::new(manifest.next_key),
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
    id_to_side: HashMap<String, String>,
}
