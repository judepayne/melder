//! Live match state: concurrent data structures for the live HTTP server.
//!
//! Both A and B sides are fully symmetrical: each has a combined embedding
//! index and (optionally) a BM25 index. Records, blocking, unmatched sets,
//! and common ID indices are held by the shared `RecordStore` on
//! `LiveMatchState`.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use dashmap::DashMap;

use crate::config::Config;
use crate::crossmap::CrossMap;
use crate::data;
use crate::encoder::EncoderPool;
use crate::encoder::coordinator::EncoderCoordinator;
use crate::error::MelderError;
use crate::models::Side;
use crate::state::upsert_log::{UpsertLog, WalEvent};
use crate::store::RecordStore;
use crate::store::memory::MemoryStore;
use crate::vectordb::{self, VectorDB, combined_cache_path, spec_hash};

/// Per-side live state: combined embedding index and BM25 index.
///
/// Records, blocking, unmatched, and common_id are in the shared
/// `RecordStore` on `LiveMatchState`.
pub struct LiveSideState {
    /// Single combined embedding index for this side.
    /// `None` if no embedding fields are configured in the job config.
    pub combined_index: Option<Box<dyn VectorDB>>,
    /// BM25 full-text index for this side. Built from fuzzy/embedding
    /// text fields. `None` if BM25 not configured or feature not enabled.
    #[cfg(feature = "bm25")]
    pub bm25_index: Option<std::sync::Mutex<crate::bm25::index::BM25Index>>,
}

impl std::fmt::Debug for LiveSideState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("LiveSideState");
        s.field(
            "combined_index_len",
            &self.combined_index.as_ref().map(|i| i.len()).unwrap_or(0),
        );
        #[cfg(feature = "bm25")]
        s.field("bm25_index", &self.bm25_index.is_some());
        s.finish()
    }
}

/// Build the composite key for a review queue entry.
pub fn review_queue_key(side: Side, id: &str, candidate_id: &str) -> String {
    let s = match side {
        Side::A => "a",
        Side::B => "b",
    };
    format!("{}:{}:{}", s, id, candidate_id)
}

/// A pending review-band match awaiting human resolution.
#[derive(Debug, Clone)]
pub struct ReviewEntry {
    pub id: String,
    pub side: Side,
    pub candidate_id: String,
    pub score: f64,
}

/// Composite live match state with both sides + shared resources.
pub struct LiveMatchState {
    pub config: Config,
    pub store: Arc<MemoryStore>,
    pub a: LiveSideState,
    pub b: LiveSideState,
    pub crossmap: CrossMap,
    pub encoder_pool: Arc<EncoderPool>,
    /// Optional batching coordinator for concurrent encoding.
    /// Created when `performance.encoder_batch_wait_ms > 0`.
    pub coordinator: Option<EncoderCoordinator>,
    pub wal: UpsertLog,
    pub crossmap_dirty: AtomicBool,
    /// Pending review-band matches. Keyed by `"{side}:{id}:{candidate_id}"`.
    /// Populated on `ReviewMatch` WAL events, drained on crossmap
    /// confirm/break and record re-upsert.
    pub review_queue: DashMap<String, ReviewEntry>,
}

impl std::fmt::Debug for LiveMatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveMatchState")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("crossmap_len", &self.crossmap.len())
            .field("review_queue_len", &self.review_queue.len())
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
    /// 6. Build MemoryStore with records + blocking
    /// 7. Load CrossMap
    /// 8. Build unmatched sets
    /// 9. Open WAL, replay events
    /// 10. Rebuild unmatched sets after WAL replay
    /// 11. Build common_id_index
    /// 12. Build BM25 indices
    /// 13. Log startup summary
    pub fn load(config: Config) -> Result<Arc<Self>, MelderError> {
        let start = Instant::now();

        // 1. Init encoder pool
        let pool_size = config.performance.encoder_pool_size.unwrap_or(1);
        eprintln!(
            "Initializing encoder pool (model={}, pool_size={})...",
            config.embeddings.model, pool_size
        );
        let encoder_pool = Arc::new(
            EncoderPool::new(
                &config.embeddings.model,
                pool_size,
                config.performance.quantized,
            )
            .map_err(MelderError::Encoder)?,
        );
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
        //    skip_deletes=true: WAL-added records may be in the cache and should
        //    be retained until replay confirms their presence.
        let combined_index_a = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            Some(&config.embeddings.a_cache_dir),
            &records_a_map,
            &ids_a,
            &config,
            true,
            &encoder_pool,
            true,
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
            true,
        )?;

        // 6. Build MemoryStore from loaded records (includes blocking indices)
        let store = Arc::new(MemoryStore::from_records(
            records_a_map,
            records_b_map,
            &config.blocking,
        ));
        eprintln!(
            "Built blocking indices in {:.1}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        // 7. Load CrossMap
        let crossmap_path = config.cross_map.path.as_deref().unwrap_or("crossmap.csv");
        let crossmap = match CrossMap::load(
            Path::new(crossmap_path),
            &config.cross_map.a_id_field,
            &config.cross_map.b_id_field,
        ) {
            Ok(cm) => {
                if !cm.is_empty() {
                    eprintln!("Loaded crossmap: {} pairs", cm.len());
                }
                cm
            }
            Err(e) => {
                eprintln!("Warning: failed to load crossmap ({}), starting fresh", e);
                CrossMap::new()
            }
        };

        // 8. Build unmatched sets
        for id in store.ids(Side::A) {
            if !crossmap.has_a(&id) {
                store.mark_unmatched(Side::A, &id);
            }
        }
        for id in store.ids(Side::B) {
            if !crossmap.has_b(&id) {
                store.mark_unmatched(Side::B, &id);
            }
        }

        // 9. Open WAL and replay events
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

        let mut wal_skipped = 0usize;
        let mut wal_encoded = 0usize;
        if !wal_events.is_empty() {
            eprintln!("Replaying {} WAL events...", wal_events.len());
            for event in &wal_events {
                match event {
                    WalEvent::UpsertRecord { side, record } => {
                        let is_a = *side == Side::A;
                        let id_field = match side {
                            Side::A => &config.datasets.a.id_field,
                            Side::B => &config.datasets.b.id_field,
                        };
                        let side_idx = match side {
                            Side::A => combined_index_a.as_deref(),
                            Side::B => combined_index_b.as_deref(),
                        };
                        if let Some(id) = record.get(id_field) {
                            // Remove old record from blocking index before overwrite
                            if let Some(old_rec) = store.get(*side, id) {
                                store.blocking_remove(*side, id, &old_rec);
                            }

                            store.insert(*side, id, record);

                            // Update blocking index with new record
                            store.blocking_insert(*side, id, record);

                            // Re-encode combined vector only if not already
                            // present in the cached index (saved at shutdown).
                            if let Some(idx) = side_idx {
                                if idx.contains(id) {
                                    wal_skipped += 1;
                                } else if let Ok(combined_vec) = vectordb::encode_combined_vector(
                                    record,
                                    &emb_specs,
                                    &encoder_pool,
                                    is_a,
                                ) {
                                    if !combined_vec.is_empty() {
                                        let _ = idx.upsert(id, &combined_vec, record, *side);
                                    }
                                    wal_encoded += 1;
                                }
                            }
                        }
                    }
                    WalEvent::CrossMapConfirm { a_id, b_id, .. } => {
                        crossmap.add(a_id, b_id);
                    }
                    WalEvent::ReviewMatch { .. } => {
                        // Informational — no state change on replay.
                    }
                    WalEvent::CrossMapBreak { a_id, b_id } => {
                        crossmap.remove(a_id, b_id);
                    }
                    WalEvent::RemoveRecord { side, id } => {
                        let side_idx = match side {
                            Side::A => combined_index_a.as_deref(),
                            Side::B => combined_index_b.as_deref(),
                        };
                        // Remove from blocking index (needs record data)
                        if let Some(rec) = store.get(*side, id) {
                            store.blocking_remove(*side, id, &rec);
                        }
                        store.remove(*side, id);
                        // Remove from combined index
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
            if wal_skipped > 0 || wal_encoded > 0 {
                eprintln!(
                    "WAL vector index: {} skipped (cached), {} re-encoded",
                    wal_skipped, wal_encoded
                );
            }

            // 10. Rebuild unmatched sets after WAL replay
            // Clear and rebuild from current records vs crossmap
            for id in store.unmatched_ids(Side::A) {
                store.mark_matched(Side::A, &id);
            }
            for id in store.unmatched_ids(Side::B) {
                store.mark_matched(Side::B, &id);
            }
            for id in store.ids(Side::A) {
                if !crossmap.has_a(&id) {
                    store.mark_unmatched(Side::A, &id);
                }
            }
            for id in store.ids(Side::B) {
                if !crossmap.has_b(&id) {
                    store.mark_unmatched(Side::B, &id);
                }
            }
            eprintln!("WAL replay complete");
        }

        // 11. Build common_id_index for each side if configured
        if let Some(ref cid_field) = config.datasets.a.common_id_field {
            for id in store.ids(Side::A) {
                if let Some(rec) = store.get(Side::A, &id)
                    && let Some(val) = rec.get(cid_field)
                {
                    let val = val.trim();
                    if !val.is_empty() {
                        store.common_id_insert(Side::A, val, &id);
                    }
                }
            }
        }
        if let Some(ref cid_field) = config.datasets.b.common_id_field {
            for id in store.ids(Side::B) {
                if let Some(rec) = store.get(Side::B, &id)
                    && let Some(val) = rec.get(cid_field)
                {
                    let val = val.trim();
                    if !val.is_empty() {
                        store.common_id_insert(Side::B, val, &id);
                    }
                }
            }
        }

        // 12. Build BM25 indices if configured
        #[cfg(feature = "bm25")]
        let (bm25_index_a, bm25_index_b) = {
            let has_bm25 = config.match_fields.iter().any(|mf| mf.method == "bm25");
            if has_bm25 && !config.bm25_fields.is_empty() {
                let bm25_start = Instant::now();
                let idx_a = crate::bm25::index::BM25Index::build(
                    store.as_ref(),
                    Side::A,
                    &config.bm25_fields,
                )?;
                let idx_b = crate::bm25::index::BM25Index::build(
                    store.as_ref(),
                    Side::B,
                    &config.bm25_fields,
                )?;
                eprintln!(
                    "Built BM25 indices (A: {}, B: {}) in {:.1}ms",
                    store.len(Side::A),
                    store.len(Side::B),
                    bm25_start.elapsed().as_secs_f64() * 1000.0
                );
                (
                    Some(std::sync::Mutex::new(idx_a)),
                    Some(std::sync::Mutex::new(idx_b)),
                )
            } else {
                (None, None)
            }
        };

        let a_side = LiveSideState {
            combined_index: combined_index_a,
            #[cfg(feature = "bm25")]
            bm25_index: bm25_index_a,
        };

        let b_side = LiveSideState {
            combined_index: combined_index_b,
            #[cfg(feature = "bm25")]
            bm25_index: bm25_index_b,
        };

        let total = start.elapsed();
        eprintln!(
            "Live state loaded in {:.1}s (A: {} records/{} unmatched, B: {} records/{} unmatched, crossmap: {} pairs)",
            total.as_secs_f64(),
            store.len(Side::A),
            store.unmatched_count(Side::A),
            store.len(Side::B),
            store.unmatched_count(Side::B),
            crossmap.len(),
        );

        // 13. Build review queue from WAL replay
        let review_queue: DashMap<String, ReviewEntry> = DashMap::new();
        for event in &wal_events {
            match event {
                WalEvent::ReviewMatch {
                    id,
                    side,
                    candidate_id,
                    score,
                } => {
                    let key = review_queue_key(*side, id, candidate_id);
                    review_queue.insert(
                        key,
                        ReviewEntry {
                            id: id.clone(),
                            side: *side,
                            candidate_id: candidate_id.clone(),
                            score: *score,
                        },
                    );
                }
                WalEvent::CrossMapConfirm { a_id, b_id, .. } => {
                    review_queue.retain(|_, v| {
                        !((v.id == *a_id || v.candidate_id == *a_id)
                            || (v.id == *b_id || v.candidate_id == *b_id))
                    });
                }
                WalEvent::CrossMapBreak { a_id, b_id } => {
                    review_queue.retain(|_, v| {
                        !((v.id == *a_id || v.candidate_id == *a_id)
                            || (v.id == *b_id || v.candidate_id == *b_id))
                    });
                }
                WalEvent::RemoveRecord { id, .. } => {
                    review_queue.retain(|_, v| v.id != *id && v.candidate_id != *id);
                }
                _ => {}
            }
        }
        if !review_queue.is_empty() {
            eprintln!("Review queue: {} pending reviews", review_queue.len());
        }

        Ok(Arc::new(Self {
            config,
            store,
            a: a_side,
            b: b_side,
            crossmap,
            encoder_pool,
            coordinator: None,
            wal,
            crossmap_dirty: AtomicBool::new(false),
            review_queue,
        }))
    }

    /// Initialise the encoding coordinator if `encoder_batch_wait_ms > 0`.
    ///
    /// **Must be called from within a tokio runtime** (the coordinator
    /// spawns a background task). Call this via `Arc::get_mut` before
    /// the `Arc<LiveMatchState>` is shared with the session / handlers.
    pub fn init_coordinator(&mut self) {
        let batch_wait_ms = self.config.performance.encoder_batch_wait_ms.unwrap_or(0);
        if batch_wait_ms > 0 {
            eprintln!(
                "Encoding coordinator enabled (batch_wait={}ms)",
                batch_wait_ms
            );
            self.coordinator = Some(EncoderCoordinator::new(
                Arc::clone(&self.encoder_pool),
                std::time::Duration::from_millis(batch_wait_ms),
            ));
        }
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
        self.crossmap
            .save(
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
        let vq = self
            .config
            .performance
            .vector_quantization
            .as_deref()
            .unwrap_or("f32");
        let hash = spec_hash(&emb_specs, vq);

        // A-side (always configured)
        if let Some(ref idx) = self.a.combined_index {
            let path = combined_cache_path(&self.config.embeddings.a_cache_dir, "a", &hash);
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                std::fs::create_dir_all(parent).ok();
            }
            if let Err(e) = idx.save(&path) {
                eprintln!("Warning: failed to save A combined index cache: {}", e);
            }
        }

        // B-side (only if b_cache_dir configured)
        if let Some(ref b_dir) = self.config.embeddings.b_cache_dir
            && let Some(ref idx) = self.b.combined_index
        {
            let path = combined_cache_path(b_dir, "b", &hash);
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                std::fs::create_dir_all(parent).ok();
            }
            if let Err(e) = idx.save(&path) {
                eprintln!("Warning: failed to save B combined index cache: {}", e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_queue_key_format() {
        let key = review_queue_key(Side::A, "ENT-1", "CP-2");
        assert_eq!(key, "a:ENT-1:CP-2");
        let key = review_queue_key(Side::B, "X", "Y");
        assert_eq!(key, "b:X:Y");
    }

    #[test]
    fn review_queue_insert_and_drain_on_confirm() {
        let queue: DashMap<String, ReviewEntry> = DashMap::new();

        // Simulate two reviews
        let k1 = review_queue_key(Side::A, "A-1", "B-5");
        queue.insert(
            k1,
            ReviewEntry {
                id: "A-1".into(),
                side: Side::A,
                candidate_id: "B-5".into(),
                score: 0.75,
            },
        );
        let k2 = review_queue_key(Side::B, "B-2", "A-3");
        queue.insert(
            k2,
            ReviewEntry {
                id: "B-2".into(),
                side: Side::B,
                candidate_id: "A-3".into(),
                score: 0.65,
            },
        );
        assert_eq!(queue.len(), 2);

        // Confirm A-1 <-> B-5: should drain the first review
        queue.retain(|_, v| {
            !((v.id == "A-1" || v.candidate_id == "A-1")
                || (v.id == "B-5" || v.candidate_id == "B-5"))
        });
        assert_eq!(queue.len(), 1, "only the unrelated review should remain");
    }

    #[test]
    fn review_queue_drain_on_re_upsert() {
        let queue: DashMap<String, ReviewEntry> = DashMap::new();

        let k = review_queue_key(Side::A, "A-1", "B-5");
        queue.insert(
            k,
            ReviewEntry {
                id: "A-1".into(),
                side: Side::A,
                candidate_id: "B-5".into(),
                score: 0.75,
            },
        );

        // Re-upsert A-1: clears its stale review
        let id = "A-1";
        queue.retain(|_, v| v.id != id && v.candidate_id != id);
        assert_eq!(queue.len(), 0, "review should be drained on re-upsert");
    }

    #[test]
    fn review_queue_drain_on_remove_candidate() {
        let queue: DashMap<String, ReviewEntry> = DashMap::new();

        let k = review_queue_key(Side::A, "A-1", "B-5");
        queue.insert(
            k,
            ReviewEntry {
                id: "A-1".into(),
                side: Side::A,
                candidate_id: "B-5".into(),
                score: 0.75,
            },
        );

        // Remove B-5: should drain the review referencing it
        let id = "B-5";
        queue.retain(|_, v| v.id != id && v.candidate_id != id);
        assert_eq!(
            queue.len(),
            0,
            "review should be drained when candidate is removed"
        );
    }

    #[test]
    fn review_queue_unrelated_entries_preserved() {
        let queue: DashMap<String, ReviewEntry> = DashMap::new();

        let k1 = review_queue_key(Side::A, "A-1", "B-5");
        queue.insert(
            k1,
            ReviewEntry {
                id: "A-1".into(),
                side: Side::A,
                candidate_id: "B-5".into(),
                score: 0.75,
            },
        );
        let k2 = review_queue_key(Side::A, "A-2", "B-6");
        queue.insert(
            k2,
            ReviewEntry {
                id: "A-2".into(),
                side: Side::A,
                candidate_id: "B-6".into(),
                score: 0.70,
            },
        );

        // Confirm A-1 <-> B-5: should keep A-2 <-> B-6
        queue.retain(|_, v| {
            !((v.id == "A-1" || v.candidate_id == "A-1")
                || (v.id == "B-5" || v.candidate_id == "B-5"))
        });
        assert_eq!(queue.len(), 1);
        assert!(
            queue.iter().any(|e| e.value().id == "A-2"),
            "A-2 review should be preserved"
        );
    }
}
