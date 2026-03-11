//! Live match state: concurrent data structures for the live HTTP server.
//!
//! Both A and B sides are fully symmetrical: each has a DashMap of records,
//! a combined embedding index, a BlockingIndex, and an unmatched set.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use dashmap::{DashMap, DashSet};

use crate::config::Config;
use crate::crossmap::CrossMap;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::matching::blocking::BlockingIndex;
use crate::models::{Record, Side};
use crate::state::upsert_log::{UpsertLog, WalEvent};
use crate::vectordb::{self, combined_cache_path, spec_hash, VectorDB};

/// Per-side live state: records, combined index, blocking, unmatched set.
pub struct LiveSideState {
    pub records: DashMap<String, Record>,
    /// Single combined embedding index for this side.
    /// `None` if no embedding fields are configured in the job config.
    pub combined_index: Option<Box<dyn VectorDB>>,
    pub blocking_index: RwLock<BlockingIndex>,
    pub unmatched: DashSet<String>,
    /// Reverse index: common_id_value -> record_id. Only populated when
    /// `common_id_field` is configured for this side.
    pub common_id_index: DashMap<String, String>,
}

impl std::fmt::Debug for LiveSideState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveSideState")
            .field("records", &self.records.len())
            .field(
                "combined_index_len",
                &self.combined_index.as_ref().map(|i| i.len()).unwrap_or(0),
            )
            .field("unmatched", &self.unmatched.len())
            .finish()
    }
}

/// Composite live match state with both sides + shared resources.
pub struct LiveMatchState {
    pub config: Config,
    pub a: LiveSideState,
    pub b: LiveSideState,
    pub crossmap: RwLock<CrossMap>,
    pub encoder_pool: EncoderPool,
    pub wal: UpsertLog,
    pub crossmap_dirty: AtomicBool,
}

impl std::fmt::Debug for LiveMatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveMatchState")
            .field("a", &self.a)
            .field("b", &self.b)
            .field(
                "crossmap_len",
                &self.crossmap.read().map(|c| c.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl LiveMatchState {
    /// Load live match state from config: full startup sequence.
    ///
    /// 1. Init encoder pool
    /// 2. Load A dataset
    /// 3. Load B dataset
    /// 4. Build/load A-side combined embedding index
    /// 5. Build/load B-side combined embedding index
    /// 6. Build BlockingIndex for A
    /// 7. Build BlockingIndex for B
    /// 8. Load CrossMap
    /// 9. Build unmatched sets
    /// 10. Open WAL, replay events
    /// 11. Rebuild unmatched sets after WAL replay
    /// 12. Build common_id_index
    /// 13. Log startup summary
    pub fn load(config: Config) -> Result<Arc<Self>, MelderError> {
        let start = Instant::now();

        // 1. Init encoder pool
        let pool_size = config.performance.encoder_pool_size.unwrap_or(1);
        eprintln!(
            "Initializing encoder pool (model={}, pool_size={})...",
            config.embeddings.model, pool_size
        );
        let encoder_pool = EncoderPool::new(
            &config.embeddings.model,
            pool_size,
            config.performance.quantized,
        )
        .map_err(MelderError::Encoder)?;
        let dim = encoder_pool.dim();
        eprintln!(
            "Encoder ready (dim={}), took {:.1}s",
            dim,
            start.elapsed().as_secs_f64()
        );

        // 2. Load A dataset
        let a_start = Instant::now();
        let (records_a_map, ids_a) = data::load_dataset(
            Path::new(&config.datasets.a.path),
            &config.datasets.a.id_field,
            &config.required_fields_a,
            config.datasets.a.format.as_deref(),
        )
        .map_err(MelderError::Data)?;
        eprintln!(
            "Loaded dataset A: {} records in {:.1}s",
            records_a_map.len(),
            a_start.elapsed().as_secs_f64()
        );

        // 3. Load B dataset
        let b_start = Instant::now();
        let (records_b_map, ids_b) = data::load_dataset(
            Path::new(&config.datasets.b.path),
            &config.datasets.b.id_field,
            &config.required_fields_b,
            config.datasets.b.format.as_deref(),
        )
        .map_err(MelderError::Data)?;
        eprintln!(
            "Loaded dataset B: {} records in {:.1}s",
            records_b_map.len(),
            b_start.elapsed().as_secs_f64()
        );

        // 4. Build/load A-side combined embedding index
        let combined_index_a = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            Some(&config.embeddings.a_cache_dir),
            &records_a_map,
            &ids_a,
            &config,
            true,
            &encoder_pool,
        )?;

        // 5. Build/load B-side combined embedding index
        let combined_index_b = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            config.embeddings.b_cache_dir.as_deref(),
            &records_b_map,
            &ids_b,
            &config,
            false,
            &encoder_pool,
        )?;

        // 6 & 7. Build BlockingIndex for A and B
        let bi_start = Instant::now();
        let mut blocking_a = BlockingIndex::from_config(&config.blocking);
        for (id, record) in &records_a_map {
            blocking_a.insert(id, record, Side::A);
        }
        let mut blocking_b = BlockingIndex::from_config(&config.blocking);
        for (id, record) in &records_b_map {
            blocking_b.insert(id, record, Side::B);
        }
        eprintln!(
            "Built blocking indices in {:.1}ms",
            bi_start.elapsed().as_secs_f64() * 1000.0
        );

        // 8. Load CrossMap
        let crossmap_path = config.cross_map.path.as_deref().unwrap_or("crossmap.csv");
        let crossmap = match CrossMap::load(
            Path::new(crossmap_path),
            &config.cross_map.a_id_field,
            &config.cross_map.b_id_field,
        ) {
            Ok(cm) => {
                if cm.len() > 0 {
                    eprintln!("Loaded crossmap: {} pairs", cm.len());
                }
                cm
            }
            Err(e) => {
                eprintln!("Warning: failed to load crossmap ({}), starting fresh", e);
                CrossMap::new()
            }
        };

        // Move records into DashMaps
        let records_a: DashMap<String, Record> = DashMap::with_capacity(records_a_map.len());
        for (id, rec) in &records_a_map {
            records_a.insert(id.clone(), rec.clone());
        }
        let records_b: DashMap<String, Record> = DashMap::with_capacity(records_b_map.len());
        for (id, rec) in &records_b_map {
            records_b.insert(id.clone(), rec.clone());
        }

        // 9. Build unmatched sets
        let unmatched_a = DashSet::new();
        let unmatched_b = DashSet::new();
        for entry in records_a.iter() {
            if !crossmap.has_a(entry.key()) {
                unmatched_a.insert(entry.key().clone());
            }
        }
        for entry in records_b.iter() {
            if !crossmap.has_b(entry.key()) {
                unmatched_b.insert(entry.key().clone());
            }
        }

        // 10. Open WAL and replay events
        let wal_path = config
            .live
            .upsert_log
            .as_deref()
            .unwrap_or("bench/upsert.wal");
        let wal = UpsertLog::open(Path::new(wal_path))
            .map_err(|e| MelderError::Other(anyhow::anyhow!("WAL open failed: {}", e)))?;

        let wal_events = UpsertLog::replay(Path::new(wal_path))
            .map_err(|e| MelderError::Other(anyhow::anyhow!("WAL replay failed: {}", e)))?;

        let emb_specs = vectordb::embedding_field_specs(&config);

        let mut crossmap = crossmap;
        if !wal_events.is_empty() {
            eprintln!("Replaying {} WAL events...", wal_events.len());
            for event in &wal_events {
                match event {
                    WalEvent::UpsertRecord { side, record } => {
                        let is_a = *side == Side::A;
                        let (id_field, side_records, side_idx) = match side {
                            Side::A => (
                                &config.datasets.a.id_field,
                                &records_a,
                                combined_index_a.as_deref(),
                            ),
                            Side::B => (
                                &config.datasets.b.id_field,
                                &records_b,
                                combined_index_b.as_deref(),
                            ),
                        };
                        if let Some(id) = record.get(id_field) {
                            side_records.insert(id.clone(), record.clone());

                            // Re-encode combined vector and upsert into index
                            if let Some(idx) = side_idx {
                                if let Ok(combined_vec) = vectordb::encode_combined_vector(
                                    record,
                                    &emb_specs,
                                    &encoder_pool,
                                    is_a,
                                ) {
                                    if !combined_vec.is_empty() {
                                        let _ = idx.upsert(id, &combined_vec, record, *side);
                                    }
                                }
                            }
                        }
                    }
                    WalEvent::CrossMapConfirm { a_id, b_id } => {
                        crossmap.add(a_id, b_id);
                    }
                    WalEvent::CrossMapBreak { a_id, b_id } => {
                        crossmap.remove(a_id, b_id);
                    }
                    WalEvent::RemoveRecord { side, id } => {
                        let side_records = match side {
                            Side::A => &records_a,
                            Side::B => &records_b,
                        };
                        side_records.remove(id);
                        // Remove from combined index
                        let side_idx = match side {
                            Side::A => combined_index_a.as_deref(),
                            Side::B => combined_index_b.as_deref(),
                        };
                        if let Some(idx) = side_idx {
                            let _ = idx.remove(id);
                        }
                        // Break any crossmap pair involving this ID
                        match side {
                            Side::A => {
                                if let Some(b_id) = crossmap.get_b(id).map(|s| s.to_string()) {
                                    crossmap.remove(id, &b_id);
                                }
                            }
                            Side::B => {
                                if let Some(a_id) = crossmap.get_a(id).map(|s| s.to_string()) {
                                    crossmap.remove(&a_id, id);
                                }
                            }
                        }
                    }
                }
            }

            // 11. Rebuild unmatched sets after WAL replay
            unmatched_a.clear();
            unmatched_b.clear();
            for entry in records_a.iter() {
                if !crossmap.has_a(entry.key()) {
                    unmatched_a.insert(entry.key().clone());
                }
            }
            for entry in records_b.iter() {
                if !crossmap.has_b(entry.key()) {
                    unmatched_b.insert(entry.key().clone());
                }
            }
            eprintln!("WAL replay complete");
        }

        // 12. Build common_id_index for each side if configured
        let common_id_a = DashMap::new();
        if let Some(ref cid_field) = config.datasets.a.common_id_field {
            for entry in records_a.iter() {
                if let Some(val) = entry.value().get(cid_field) {
                    let val = val.trim();
                    if !val.is_empty() {
                        common_id_a.insert(val.to_string(), entry.key().clone());
                    }
                }
            }
        }
        let common_id_b = DashMap::new();
        if let Some(ref cid_field) = config.datasets.b.common_id_field {
            for entry in records_b.iter() {
                if let Some(val) = entry.value().get(cid_field) {
                    let val = val.trim();
                    if !val.is_empty() {
                        common_id_b.insert(val.to_string(), entry.key().clone());
                    }
                }
            }
        }

        let a_side = LiveSideState {
            records: records_a,
            combined_index: combined_index_a,
            blocking_index: RwLock::new(blocking_a),
            unmatched: unmatched_a,
            common_id_index: common_id_a,
        };

        let b_side = LiveSideState {
            records: records_b,
            combined_index: combined_index_b,
            blocking_index: RwLock::new(blocking_b),
            unmatched: unmatched_b,
            common_id_index: common_id_b,
        };

        let total = start.elapsed();
        eprintln!(
            "Live state loaded in {:.1}s (A: {} records/{} unmatched, B: {} records/{} unmatched, crossmap: {} pairs)",
            total.as_secs_f64(),
            a_side.records.len(),
            a_side.unmatched.len(),
            b_side.records.len(),
            b_side.unmatched.len(),
            crossmap.len(),
        );

        Ok(Arc::new(Self {
            config,
            a: a_side,
            b: b_side,
            crossmap: RwLock::new(crossmap),
            encoder_pool,
            wal,
            crossmap_dirty: AtomicBool::new(false),
        }))
    }

    /// Get the side state for a given side.
    pub fn side(&self, side: Side) -> &LiveSideState {
        match side {
            Side::A => &self.a,
            Side::B => &self.b,
        }
    }

    /// Get the opposite side state.
    pub fn opposite_side(&self, side: Side) -> &LiveSideState {
        self.side(side.opposite())
    }

    /// Mark the crossmap as dirty (needs flushing to disk).
    pub fn mark_crossmap_dirty(&self) {
        self.crossmap_dirty.store(true, Ordering::Relaxed);
    }

    /// Check and clear the crossmap dirty flag.
    pub fn take_crossmap_dirty(&self) -> bool {
        self.crossmap_dirty.swap(false, Ordering::Relaxed)
    }

    /// Get the crossmap path from config.
    pub fn crossmap_path(&self) -> &str {
        self.config
            .cross_map
            .path
            .as_deref()
            .unwrap_or("crossmap.csv")
    }

    /// Flush crossmap to disk if dirty.
    pub fn flush_crossmap(&self) -> Result<(), MelderError> {
        if !self.take_crossmap_dirty() {
            return Ok(());
        }
        let cm = self
            .crossmap
            .read()
            .map_err(|e| MelderError::Other(anyhow::anyhow!("crossmap lock: {}", e)))?;
        cm.save(
            Path::new(self.crossmap_path()),
            &self.config.cross_map.a_id_field,
            &self.config.cross_map.b_id_field,
        )
        .map_err(MelderError::CrossMap)?;
        Ok(())
    }

    /// Save combined embedding index caches to disk.
    ///
    /// The cache path encodes a hash of the embedding field spec so that any
    /// config change (field names, order, weights) produces a new path and
    /// forces a cold rebuild on next startup.
    pub fn save_combined_index_caches(&self) -> Result<(), MelderError> {
        let emb_specs = vectordb::embedding_field_specs(&self.config);
        if emb_specs.is_empty() {
            return Ok(());
        }
        let hash = spec_hash(&emb_specs);

        // A-side (always configured)
        if let Some(ref idx) = self.a.combined_index {
            let path = combined_cache_path(&self.config.embeddings.a_cache_dir, "a", &hash);
            if let Some(parent) = Path::new(&path).parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).ok();
                }
            }
            if let Err(e) = idx.save(Path::new(&path)) {
                eprintln!("Warning: failed to save A combined index cache: {}", e);
            }
        }

        // B-side (only if b_cache_dir configured)
        if let Some(ref b_dir) = self.config.embeddings.b_cache_dir {
            if let Some(ref idx) = self.b.combined_index {
                let path = combined_cache_path(b_dir, "b", &hash);
                if let Some(parent) = Path::new(&path).parent() {
                    if !parent.exists() {
                        std::fs::create_dir_all(parent).ok();
                    }
                }
                if let Err(e) = idx.save(Path::new(&path)) {
                    eprintln!("Warning: failed to save B combined index cache: {}", e);
                }
            }
        }

        Ok(())
    }
}
