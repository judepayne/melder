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
use tracing::{info, warn};

use crate::config::{Config, MatchMethod};
use crate::crossmap::{CrossMapOps, MemoryCrossMap};
use crate::data;
use crate::encoder::Encoder;
use crate::encoder::coordinator::EncoderCoordinator;
use crate::error::MelderError;
use crate::models::Side;
use crate::state::match_log::{MatchLog, MatchLogEvent};
use crate::store::RecordStore;
use crate::store::memory::MemoryStore;
use crate::store::sqlite::open_sqlite;
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
    /// text fields. `None` if BM25 not configured.
    pub bm25_index: Option<crate::bm25::simple::SimpleBm25>,
    /// Synonym index for this side. Built from synonym_fields config.
    /// `None` if no synonym matching configured.
    pub synonym_index: Option<std::sync::RwLock<crate::synonym::index::SynonymIndex>>,
}

impl std::fmt::Debug for LiveSideState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("LiveSideState");
        s.field(
            "combined_index_len",
            &self.combined_index.as_ref().map(|i| i.len()).unwrap_or(0),
        );
        s.field("bm25_index", &self.bm25_index.is_some());
        s.field("synonym_index", &self.synonym_index.is_some());
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
    pub store: Arc<dyn RecordStore>,
    pub a: LiveSideState,
    pub b: LiveSideState,
    pub crossmap: Box<dyn CrossMapOps>,
    pub exclusions: crate::matching::exclusions::Exclusions,
    pub encoder_pool: Arc<dyn Encoder>,
    /// Optional batching coordinator for concurrent encoding.
    /// Created when `performance.encoder_batch_wait_ms > 0`.
    pub coordinator: Option<EncoderCoordinator>,
    pub wal: MatchLog,
    pub crossmap_dirty: AtomicBool,
    /// Pending review-band matches. Keyed by `"{side}:{id}:{candidate_id}"`.
    /// Populated on upsert/initial-match when a score falls in the review band,
    /// drained on crossmap confirm/break and record re-upsert/remove.
    ///
    /// Serves a single consumer: `GET /api/v1/review/list`. The queue is an
    /// in-memory read cache over the store (SQLite `persist_review`); the store
    /// is the source of truth and survives restarts.
    ///
    /// Unbounded by design: the queue naturally drains as records are re-upserted,
    /// confirmed, broken, or removed. Only a workload that inserts once and never
    /// resolves would see unbounded growth — if that becomes a real concern,
    /// add a configurable cap with eviction (data remains in the store).
    pub review_queue: DashMap<String, ReviewEntry>,
    /// Optional user-provided synonym dictionary shared across both sides.
    pub synonym_dictionary: Option<Arc<crate::synonym::dictionary::SynonymDictionary>>,
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
    /// Two startup paths depending on `live.db_path`:
    ///
    /// **Memory path** (no `db_path`): Load datasets from CSV, build
    /// MemoryStore, load MemoryCrossMap from CSV, replay WAL. Current
    /// behavior — backward compatible.
    ///
    /// **SQLite path** (`db_path` set):
    /// - Cold start (no DB file): create DB, load datasets from CSV,
    ///   populate SqliteStore + SqliteCrossMap.
    /// - Warm start (DB file exists): open existing DB. Records, blocking,
    ///   unmatched, common_id, and crossmap are already populated. Skip CSV
    ///   loading and WAL replay.
    ///
    pub fn load(config: Config) -> Result<Arc<Self>, MelderError> {
        let start = Instant::now();

        // 1. Init encoder (local or remote — branch on config).
        //
        // Live mode note: GPU is silently ignored (doesn't help per-record
        // latency) and remote encoders work but are bounded by round-trip
        // latency. See docs/remote-encoder.md.
        if config.performance.encoder_device.as_deref() == Some("gpu")
            && config.embeddings.remote_encoder_cmd.is_none()
        {
            eprintln!(
                "NOTE: encoder_device: gpu is ignored in live mode — GPU encoding \
                 does not improve single-record latency. Using CPU."
            );
        }
        // Live mode always forces CPU — GPU encoding does not improve
        // per-record latency for single-record upserts.
        let encoder_pool = crate::state::state::build_encoder(&config, true)?;
        let dim = encoder_pool.dim();
        info!(
            dim,
            elapsed_s = format!("{:.1}", start.elapsed().as_secs_f64()),
            "encoder ready"
        );

        // Determine startup mode
        if config.is_enroll_mode() {
            return Self::load_enroll(config, start, encoder_pool);
        }

        let sqlite_pool_config = Some(crate::store::sqlite::SqlitePoolConfig {
            writer_cache_kb: config.live.sqlite_cache_mb.unwrap_or(64) * 1024,
            read_pool_size: config.live.sqlite_read_pool_size.unwrap_or(4),
            reader_cache_kb: config.live.sqlite_pool_worker_cache_mb.unwrap_or(128) * 1024,
        });
        if let Some(db_path) = config.live.db_path.clone() {
            Self::load_sqlite(config, &db_path, start, encoder_pool, sqlite_pool_config)
        } else {
            Self::load_memory(config, start, encoder_pool)
        }
    }

    /// Load both datasets from CSV files.
    #[allow(clippy::type_complexity)]
    fn load_datasets(
        config: &Config,
    ) -> Result<
        (
            std::collections::HashMap<String, crate::models::Record>,
            Vec<String>,
            std::collections::HashMap<String, crate::models::Record>,
            Vec<String>,
        ),
        MelderError,
    > {
        let a_start = Instant::now();
        let (records_a_map, ids_a) = data::load_dataset(
            Path::new(&config.datasets.a.path),
            &config.datasets.a.id_field,
            &config.required_fields_a,
            config.datasets.a.format.as_deref(),
        )
        .map_err(MelderError::Data)?;
        info!(
            side = "A",
            records = records_a_map.len(),
            elapsed_s = format!("{:.1}", a_start.elapsed().as_secs_f64()),
            "loaded dataset"
        );

        let b_start = Instant::now();
        let (records_b_map, ids_b) = data::load_dataset(
            Path::new(&config.datasets.b.path),
            &config.datasets.b.id_field,
            &config.required_fields_b,
            config.datasets.b.format.as_deref(),
        )
        .map_err(MelderError::Data)?;
        info!(
            side = "B",
            records = records_b_map.len(),
            elapsed_s = format!("{:.1}", b_start.elapsed().as_secs_f64()),
            "loaded dataset"
        );

        Ok((records_a_map, ids_a, records_b_map, ids_b))
    }

    /// Finish construction: build BM25, open WAL, load reviews, assemble
    /// the `LiveMatchState`.
    ///
    /// Shared by both `load_memory` and `load_sqlite`. Each caller builds
    /// the store, crossmap, and embedding indices its own way, then
    /// delegates to this method for the common tail.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        config: Config,
        start: Instant,
        encoder_pool: Arc<dyn Encoder>,
        store: Arc<dyn RecordStore>,
        crossmap: Box<dyn CrossMapOps>,
        exclusions: crate::matching::exclusions::Exclusions,
        combined_index_a: Option<Box<dyn VectorDB>>,
        combined_index_b: Option<Box<dyn VectorDB>>,
        label: &str,
    ) -> Result<Arc<Self>, MelderError> {
        // Build BM25 indices if configured
        let (bm25_index_a, bm25_index_b) = Self::build_bm25_indices(&store, &config, start)?;

        // Load synonym dictionary if configured
        let synonym_dict: Option<Arc<crate::synonym::dictionary::SynonymDictionary>> =
            if let Some(ref sd_cfg) = config.synonym_dictionary {
                let dict = crate::synonym::dictionary::SynonymDictionary::load(
                    std::path::Path::new(&sd_cfg.path),
                )?;
                info!(groups = dict.len(), "loaded synonym dictionary");
                Some(Arc::new(dict))
            } else {
                None
            };

        // Build synonym indices if configured
        let (synonym_index_a, synonym_index_b) =
            Self::build_synonym_indices(&store, &config, synonym_dict.clone());

        let a_side = LiveSideState {
            combined_index: combined_index_a,
            bm25_index: bm25_index_a,
            synonym_index: synonym_index_a,
        };

        let b_side = LiveSideState {
            combined_index: combined_index_b,
            bm25_index: bm25_index_b,
            synonym_index: synonym_index_b,
        };

        // Open WAL for append-only writes
        let wal_path = config
            .live
            .match_log_path
            .as_deref()
            .unwrap_or("bench/upsert.wal");
        let wal = MatchLog::open(Path::new(wal_path))
            .map_err(|e| MelderError::Other(anyhow::anyhow!("WAL open failed: {}", e)))?;

        let total = start.elapsed();
        info!(
            label,
            a_records = store.len(Side::A).unwrap_or(0),
            a_unmatched = store.unmatched_count(Side::A).unwrap_or(0),
            b_records = store.len(Side::B).unwrap_or(0),
            b_unmatched = store.unmatched_count(Side::B).unwrap_or(0),
            crossmap_pairs = crossmap.len(),
            elapsed_s = format!("{:.1}", total.as_secs_f64()),
            "live state loaded"
        );

        // Load review queue from the store (WAL replay or SQLite read)
        let review_queue: DashMap<String, ReviewEntry> = DashMap::new();
        for (key, id, side, candidate_id, score) in store.load_reviews()? {
            review_queue.insert(
                key,
                ReviewEntry {
                    id,
                    side,
                    candidate_id,
                    score,
                },
            );
        }
        if !review_queue.is_empty() {
            info!(pending = review_queue.len(), "review queue loaded");
        }

        Ok(Arc::new(Self {
            config,
            store,
            a: a_side,
            b: b_side,
            crossmap,
            exclusions,
            encoder_pool,
            coordinator: None,
            wal,
            crossmap_dirty: AtomicBool::new(false),
            review_queue,
            synonym_dictionary: synonym_dict,
        }))
    }

    /// Memory-backed startup: load datasets, MemoryStore, WAL replay.
    fn load_memory(
        config: Config,
        start: Instant,
        encoder_pool: Arc<dyn Encoder>,
    ) -> Result<Arc<Self>, MelderError> {
        let (records_a_map, ids_a, records_b_map, ids_b) = Self::load_datasets(&config)?;

        // Build embedding indices (skip_deletes=true: WAL-added records may
        // be in the cache and should be retained until replay confirms them).
        let combined_index_a = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            Some(&config.embeddings.a_cache_dir),
            &records_a_map,
            &ids_a,
            &config,
            true,
            encoder_pool.as_ref(),
            true,
            Some(Path::new(&config.datasets.a.path)),
        )?;

        let combined_index_b = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            config.embeddings.b_cache_dir.as_deref(),
            &records_b_map,
            &ids_b,
            &config,
            false,
            encoder_pool.as_ref(),
            true,
            Some(Path::new(&config.datasets.b.path)),
        )?;

        // Build MemoryStore from loaded records (includes blocking indices)
        let store: Arc<dyn RecordStore> = Arc::new(MemoryStore::from_records(
            records_a_map,
            records_b_map,
            &config.blocking,
        ));
        info!(
            elapsed_ms = format!("{:.1}", start.elapsed().as_secs_f64() * 1000.0),
            "built blocking indices"
        );

        // Load CrossMap
        let crossmap_path = config.cross_map.path.as_deref().unwrap_or("crossmap.csv");
        let crossmap: Box<dyn CrossMapOps> = match MemoryCrossMap::load(
            Path::new(crossmap_path),
            &config.cross_map.a_id_field,
            &config.cross_map.b_id_field,
        ) {
            Ok(cm) => {
                if !cm.is_empty() {
                    info!(pairs = cm.len(), "loaded crossmap");
                }
                cm.set_flush_path(
                    Path::new(crossmap_path),
                    &config.cross_map.a_id_field,
                    &config.cross_map.b_id_field,
                );
                Box::new(cm)
            }
            Err(e) => {
                warn!(error = %e, "failed to load crossmap, starting fresh");
                let cm = MemoryCrossMap::new();
                cm.set_flush_path(
                    Path::new(crossmap_path),
                    &config.cross_map.a_id_field,
                    &config.cross_map.b_id_field,
                );
                Box::new(cm)
            }
        };

        // Load exclusions
        let exclusions = Self::load_exclusions(&config);

        // Build unmatched sets
        for id in store.ids(Side::A).unwrap_or_default() {
            if !crossmap.has_a(&id) {
                let _ = store.mark_unmatched(Side::A, &id);
            }
        }
        for id in store.ids(Side::B).unwrap_or_default() {
            if !crossmap.has_b(&id) {
                let _ = store.mark_unmatched(Side::B, &id);
            }
        }

        // WAL replay (updates store, crossmap, exclusions, and vector indices)
        let wal_path = config
            .live
            .match_log_path
            .as_deref()
            .unwrap_or("bench/upsert.wal");
        let wal_events = MatchLog::replay(Path::new(wal_path))
            .map_err(|e| MelderError::Other(anyhow::anyhow!("WAL replay failed: {}", e)))?;

        if !wal_events.is_empty() {
            Self::replay_wal(
                &wal_events,
                &config,
                &store,
                crossmap.as_ref(),
                &exclusions,
                &encoder_pool,
                combined_index_a.as_deref(),
                combined_index_b.as_deref(),
            )?;
        }

        // Build common_id_index for each side if configured
        Self::build_common_id_index(&store, &config);

        Self::finish(
            config,
            start,
            encoder_pool,
            store,
            crossmap,
            exclusions,
            combined_index_a,
            combined_index_b,
            "",
        )
    }

    /// Enroll-mode startup: single-pool, no B-side, no crossmap.
    fn load_enroll(
        config: Config,
        start: Instant,
        encoder_pool: Arc<dyn Encoder>,
    ) -> Result<Arc<Self>, MelderError> {
        // Load A-side dataset if configured (optional — pool can start empty)
        let (records_a_map, ids_a) = if !config.datasets.a.path.is_empty() {
            let a_start = Instant::now();
            let (map, ids) = data::load_dataset(
                Path::new(&config.datasets.a.path),
                &config.datasets.a.id_field,
                &config.required_fields_a,
                config.datasets.a.format.as_deref(),
            )
            .map_err(MelderError::Data)?;
            info!(
                records = map.len(),
                elapsed_s = format!("{:.1}", a_start.elapsed().as_secs_f64()),
                "loaded pool dataset"
            );
            (map, ids)
        } else {
            info!("no initial dataset, starting with empty pool");
            (std::collections::HashMap::new(), Vec::new())
        };

        // Build A-side embedding index
        let combined_index_a = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            Some(&config.embeddings.a_cache_dir),
            &records_a_map,
            &ids_a,
            &config,
            true,
            encoder_pool.as_ref(),
            true,
            if config.datasets.a.path.is_empty() {
                None
            } else {
                Some(Path::new(&config.datasets.a.path))
            },
        )?;

        // Build MemoryStore with A records only, B is empty
        let store: Arc<dyn RecordStore> = Arc::new(MemoryStore::from_records(
            records_a_map,
            std::collections::HashMap::new(),
            &config.blocking,
        ));

        // Empty crossmap (not used in enroll mode)
        let crossmap: Box<dyn CrossMapOps> = Box::new(MemoryCrossMap::new());

        // Load exclusions
        let exclusions = Self::load_exclusions(&config);

        // Mark all A records as unmatched (no crossmap)
        for id in store.ids(Side::A).unwrap_or_default() {
            let _ = store.mark_unmatched(Side::A, &id);
        }

        // WAL replay (enroll events are stored as A-side upserts)
        let wal_path = config
            .live
            .match_log_path
            .as_deref()
            .unwrap_or("bench/upsert.wal");
        let wal_events = MatchLog::replay(Path::new(wal_path))
            .map_err(|e| MelderError::Other(anyhow::anyhow!("WAL replay failed: {}", e)))?;

        if !wal_events.is_empty() {
            Self::replay_wal(
                &wal_events,
                &config,
                &store,
                crossmap.as_ref(),
                &exclusions,
                &encoder_pool,
                combined_index_a.as_deref(),
                None, // No B-side index
            )?;
        }

        // Build common_id_index if configured
        Self::build_common_id_index(&store, &config);

        Self::finish(
            config,
            start,
            encoder_pool,
            store,
            crossmap,
            exclusions,
            combined_index_a,
            None, // No B-side index
            "enroll ",
        )
    }

    /// Replay WAL events into the store, crossmap, exclusions, and vector indices.
    ///
    /// Used only by the memory-backed startup path. The SQLite path skips
    /// WAL replay entirely (state is already durable in the DB).
    #[allow(clippy::too_many_arguments)]
    fn replay_wal(
        wal_events: &[MatchLogEvent],
        config: &Config,
        store: &Arc<dyn RecordStore>,
        crossmap: &dyn CrossMapOps,
        exclusions: &crate::matching::exclusions::Exclusions,
        encoder_pool: &Arc<dyn Encoder>,
        combined_index_a: Option<&dyn VectorDB>,
        combined_index_b: Option<&dyn VectorDB>,
    ) -> Result<(), MelderError> {
        let emb_specs = vectordb::embedding_field_specs(config);
        let mut wal_skipped = 0usize;
        let mut wal_encoded = 0usize;

        info!(events = wal_events.len(), "replaying WAL");
        for event in wal_events {
            match event {
                MatchLogEvent::UpsertRecord { side, record } => {
                    let is_a = *side == Side::A;
                    let id_field = match side {
                        Side::A => &config.datasets.a.id_field,
                        Side::B => &config.datasets.b.id_field,
                    };
                    let side_idx = match side {
                        Side::A => combined_index_a,
                        Side::B => combined_index_b,
                    };
                    if let Some(id) = record.get(id_field) {
                        // Remove old record from blocking index before overwrite
                        if let Some(old_rec) = store.get(*side, id).ok().flatten() {
                            let _ = store.blocking_remove(*side, id, &old_rec);
                        }

                        let _ = store.insert(*side, id, record);
                        let _ = store.blocking_insert(*side, id, record);

                        // Re-encode combined vector only if not already
                        // present in the cached index (saved at shutdown).
                        if let Some(idx) = side_idx {
                            if idx.contains(id) {
                                wal_skipped += 1;
                            } else if let Ok(combined_vec) = vectordb::encode_combined_vector(
                                record,
                                &emb_specs,
                                encoder_pool.as_ref(),
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
                MatchLogEvent::CrossMapConfirm { a_id, b_id, .. } => {
                    crossmap.add(a_id, b_id);
                }
                MatchLogEvent::ReviewMatch { .. } => {
                    // Informational — no state change on replay.
                }
                MatchLogEvent::CrossMapBreak { a_id, b_id } => {
                    crossmap.remove(a_id, b_id);
                }
                MatchLogEvent::RemoveRecord { side, id } => {
                    let side_idx = match side {
                        Side::A => combined_index_a,
                        Side::B => combined_index_b,
                    };
                    // Remove from blocking index (needs record data)
                    if let Some(rec) = store.get(*side, id).ok().flatten() {
                        let _ = store.blocking_remove(*side, id, &rec);
                    }
                    let _ = store.remove(*side, id);
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
                MatchLogEvent::NoMatchBelow { .. } => {
                    // Informational — no state change on replay.
                }
                MatchLogEvent::Exclude { a_id, b_id } => {
                    exclusions.add(a_id, b_id);
                }
                MatchLogEvent::Unexclude { a_id, b_id } => {
                    exclusions.remove(a_id, b_id);
                }
            }
        }
        if wal_skipped > 0 || wal_encoded > 0 {
            info!(
                skipped = wal_skipped,
                re_encoded = wal_encoded,
                "WAL vector index"
            );
        }

        // Rebuild unmatched sets after WAL replay
        for id in store.unmatched_ids(Side::A).unwrap_or_default() {
            let _ = store.mark_matched(Side::A, &id);
        }
        for id in store.unmatched_ids(Side::B).unwrap_or_default() {
            let _ = store.mark_matched(Side::B, &id);
        }
        for id in store.ids(Side::A).unwrap_or_default() {
            if !crossmap.has_a(&id) {
                let _ = store.mark_unmatched(Side::A, &id);
            }
        }
        for id in store.ids(Side::B).unwrap_or_default() {
            if !crossmap.has_b(&id) {
                let _ = store.mark_unmatched(Side::B, &id);
            }
        }
        info!("WAL replay complete");
        Ok(())
    }

    /// SQLite-backed startup: cold start (create + populate) or warm start
    /// (open existing DB).
    fn load_sqlite(
        config: Config,
        db_path: &str,
        start: Instant,
        encoder_pool: Arc<dyn Encoder>,
        pool_config: Option<crate::store::sqlite::SqlitePoolConfig>,
    ) -> Result<Arc<Self>, MelderError> {
        let db_exists = Path::new(db_path).exists();
        let mode = if db_exists { "warm" } else { "cold" };
        info!(mode, path = %db_path, "sqlite startup");

        let (sqlite_store, sqlite_crossmap, _conn) = open_sqlite(
            Path::new(db_path),
            &config.blocking,
            pool_config,
            &config.required_fields_a,
            &config.required_fields_b,
        )
        .map_err(|e| MelderError::Other(anyhow::anyhow!("SQLite open failed: {}", e)))?;

        let store: Arc<dyn RecordStore> = Arc::new(sqlite_store);
        let crossmap: Box<dyn CrossMapOps> = Box::new(sqlite_crossmap);

        if !db_exists {
            // --- Cold start: load datasets from CSV into SQLite ---
            let (records_a_map, _ids_a, records_b_map, _ids_b) = Self::load_datasets(&config)?;

            // Populate SqliteStore with records + blocking
            let pop_start = Instant::now();
            for (id, record) in &records_a_map {
                let _ = store.insert(Side::A, id, record);
                let _ = store.blocking_insert(Side::A, id, record);
            }
            for (id, record) in &records_b_map {
                let _ = store.insert(Side::B, id, record);
                let _ = store.blocking_insert(Side::B, id, record);
            }
            info!(
                elapsed_s = format!("{:.1}", pop_start.elapsed().as_secs_f64()),
                "populated sqlite store"
            );

            // Load existing crossmap from CSV into SQLite
            let crossmap_path = config.cross_map.path.as_deref().unwrap_or("crossmap.csv");
            if let Ok(mem_cm) = MemoryCrossMap::load(
                Path::new(crossmap_path),
                &config.cross_map.a_id_field,
                &config.cross_map.b_id_field,
            ) && !mem_cm.is_empty()
            {
                let pairs = mem_cm.pairs();
                for (a_id, b_id) in &pairs {
                    crossmap.add(a_id, b_id);
                }
                info!(pairs = pairs.len(), "imported crossmap into sqlite");
            }

            // Build unmatched sets
            for id in store.ids(Side::A).unwrap_or_default() {
                if crossmap.has_a(&id) {
                    let _ = store.mark_matched(Side::A, &id);
                } else {
                    let _ = store.mark_unmatched(Side::A, &id);
                }
            }
            for id in store.ids(Side::B).unwrap_or_default() {
                if crossmap.has_b(&id) {
                    let _ = store.mark_matched(Side::B, &id);
                } else {
                    let _ = store.mark_unmatched(Side::B, &id);
                }
            }

            // Build common_id_index
            Self::build_common_id_index(&store, &config);
        } else {
            // --- Warm start: everything is already in SQLite ---
            info!(
                a_records = store.len(Side::A).unwrap_or(0),
                b_records = store.len(Side::B).unwrap_or(0),
                crossmap_pairs = crossmap.len(),
                "sqlite warm start"
            );
        }

        // Collect records from the store for the vector cache builder.
        let ids_a = store.ids(Side::A).unwrap_or_default();
        let ids_b = store.ids(Side::B).unwrap_or_default();
        let records_a_map: std::collections::HashMap<String, crate::models::Record> = ids_a
            .iter()
            .filter_map(|id| {
                store
                    .get(Side::A, id)
                    .ok()
                    .flatten()
                    .map(|r| (id.clone(), r))
            })
            .collect();
        let records_b_map: std::collections::HashMap<String, crate::models::Record> = ids_b
            .iter()
            .filter_map(|id| {
                store
                    .get(Side::B, id)
                    .ok()
                    .flatten()
                    .map(|r| (id.clone(), r))
            })
            .collect();

        // Build embedding indices (source is SQLite, not a file — no fingerprint)
        let combined_index_a = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            Some(&config.embeddings.a_cache_dir),
            &records_a_map,
            &ids_a,
            &config,
            true,
            encoder_pool.as_ref(),
            true,
            None,
        )?;

        let combined_index_b = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            config.embeddings.b_cache_dir.as_deref(),
            &records_b_map,
            &ids_b,
            &config,
            false,
            encoder_pool.as_ref(),
            true,
            None,
        )?;

        // Load exclusions
        let exclusions = Self::load_exclusions(&config);

        Self::finish(
            config,
            start,
            encoder_pool,
            store,
            crossmap,
            exclusions,
            combined_index_a,
            combined_index_b,
            "SQLite, ",
        )
    }

    /// Build common_id_index from the store for each side (if configured).
    fn build_common_id_index(store: &Arc<dyn RecordStore>, config: &Config) {
        if let Some(ref cid_field) = config.datasets.a.common_id_field {
            for id in store.ids(Side::A).unwrap_or_default() {
                if let Some(rec) = store.get(Side::A, &id).ok().flatten()
                    && let Some(val) = rec.get(cid_field)
                {
                    let val = val.trim();
                    if !val.is_empty() {
                        let _ = store.common_id_insert(Side::A, val, &id);
                    }
                }
            }
        }
        if let Some(ref cid_field) = config.datasets.b.common_id_field {
            for id in store.ids(Side::B).unwrap_or_default() {
                if let Some(rec) = store.get(Side::B, &id).ok().flatten()
                    && let Some(val) = rec.get(cid_field)
                {
                    let val = val.trim();
                    if !val.is_empty() {
                        let _ = store.common_id_insert(Side::B, val, &id);
                    }
                }
            }
        }
    }

    /// Load exclusions from CSV if configured. Returns an empty set if not configured
    /// or the file is missing.
    fn load_exclusions(config: &Config) -> crate::matching::exclusions::Exclusions {
        use crate::matching::exclusions::Exclusions;

        if let Some(ref p) = config.exclusions.path
            && !p.is_empty()
        {
            match Exclusions::load(
                Path::new(p),
                &config.exclusions.a_id_field,
                &config.exclusions.b_id_field,
            ) {
                Ok(ex) => {
                    if !ex.is_empty() {
                        ex.set_flush_path(
                            Path::new(p),
                            &config.exclusions.a_id_field,
                            &config.exclusions.b_id_field,
                        );
                    }
                    return ex;
                }
                Err(e) => {
                    warn!(error = %e, "failed to load exclusions, starting fresh");
                }
            }
        }
        Exclusions::new()
    }

    /// Build BM25 indices from the store (if BM25 is configured and enabled).
    ///
    /// Uses `SimpleBm25` — a lock-free DashMap-based scorer. No RwLock
    /// wrapping, no commit step, no self-score pre-warming needed (uses
    /// analytical self-score at query time).
    fn build_bm25_indices(
        store: &Arc<dyn RecordStore>,
        config: &Config,
        start: Instant,
    ) -> Result<
        (
            Option<crate::bm25::simple::SimpleBm25>,
            Option<crate::bm25::simple::SimpleBm25>,
        ),
        MelderError,
    > {
        let has_bm25 = config
            .match_fields
            .iter()
            .any(|mf| mf.method == MatchMethod::Bm25);
        if has_bm25 && !config.bm25_fields.is_empty() {
            let bm25_start = Instant::now();
            let idx_a = crate::bm25::simple::SimpleBm25::build(
                store.as_ref(),
                Side::A,
                &config.bm25_fields,
            );
            let idx_b = crate::bm25::simple::SimpleBm25::build(
                store.as_ref(),
                Side::B,
                &config.bm25_fields,
            );
            info!(
                a_records = store.len(Side::A).unwrap_or(0),
                b_records = store.len(Side::B).unwrap_or(0),
                elapsed_ms = format!("{:.1}", bm25_start.elapsed().as_secs_f64() * 1000.0),
                "built BM25 indices"
            );

            let _ = start; // suppress unused warning when bm25 timing is used
            Ok((Some(idx_a), Some(idx_b)))
        } else {
            Ok((None, None))
        }
    }

    /// Build synonym indices for both sides if `synonym_fields` is configured.
    fn build_synonym_indices(
        store: &Arc<dyn RecordStore>,
        config: &Config,
        dictionary: Option<std::sync::Arc<crate::synonym::dictionary::SynonymDictionary>>,
    ) -> (
        Option<std::sync::RwLock<crate::synonym::index::SynonymIndex>>,
        Option<std::sync::RwLock<crate::synonym::index::SynonymIndex>>,
    ) {
        if config.synonym_fields.is_empty() {
            return (None, None);
        }
        let syn_start = std::time::Instant::now();
        let idx_a = crate::synonym::index::SynonymIndex::build(
            store.as_ref(),
            Side::A,
            &config.synonym_fields,
            dictionary.clone(),
        );
        let idx_b = crate::synonym::index::SynonymIndex::build(
            store.as_ref(),
            Side::B,
            &config.synonym_fields,
            dictionary,
        );
        info!(
            a_keys = idx_a.len(),
            b_keys = idx_b.len(),
            elapsed_ms = format!("{:.1}", syn_start.elapsed().as_secs_f64() * 1000.0),
            "built synonym indices"
        );
        (
            Some(std::sync::RwLock::new(idx_a)),
            Some(std::sync::RwLock::new(idx_b)),
        )
    }

    /// Initialise the encoding coordinator if `encoder_batch_wait_ms > 0`.
    ///
    /// The coordinator uses a plain OS thread (no tokio runtime required).
    /// Call this via `Arc::get_mut` before the `Arc<LiveMatchState>` is
    /// shared with the session / handlers.
    pub fn init_coordinator(&mut self) {
        let batch_wait_ms = self.config.performance.encoder_batch_wait_ms.unwrap_or(0);
        if batch_wait_ms > 0 {
            info!(batch_wait_ms, "encoding coordinator enabled");
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

    /// Get the pool-side state for scoring.
    ///
    /// In match mode, the pool is the opposite side. In enroll mode, the
    /// pool is always Side::A (the single pool).
    pub fn pool_side(&self, query_side: Side) -> &LiveSideState {
        if self.config.is_enroll_mode() {
            &self.a
        } else {
            self.opposite_side(query_side)
        }
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
    ///
    /// Delegates to `CrossMapOps::flush()` — saves to CSV for
    /// `MemoryCrossMap`, no-op for `SqliteCrossMap` (write-through).
    pub fn flush_crossmap(&self) -> Result<(), MelderError> {
        if !self.take_crossmap_dirty() {
            return Ok(());
        }
        self.crossmap
            .flush()
            .map_err(|e| MelderError::Other(anyhow::anyhow!("crossmap flush: {}", e)))?;
        Ok(())
    }

    // -------------------------------------------------------------------
    // Review queue write-through methods
    // -------------------------------------------------------------------

    /// Insert a review entry into the queue (and persist via RecordStore).
    pub fn insert_review(&self, key: String, entry: ReviewEntry) {
        if let Err(e) = self.store.persist_review(
            &key,
            &entry.id,
            entry.side,
            &entry.candidate_id,
            entry.score,
        ) {
            warn!(error = %e, key, "review persist failed — in-memory only until restart");
        }
        self.review_queue.insert(key, entry);
    }

    /// Drain review entries for a given record ID (on re-upsert or remove).
    pub fn drain_reviews_for_id(&self, id: &str) {
        self.review_queue
            .retain(|_, v| v.id != id && v.candidate_id != id);
        if let Err(e) = self.store.remove_reviews_for_id(id) {
            warn!(error = %e, id, "review drain (by id) persist failed");
        }
    }

    /// Drain review entries involving either ID (on crossmap confirm/break).
    pub fn drain_reviews_for_pair(&self, a_id: &str, b_id: &str) {
        self.review_queue.retain(|_, v| {
            !((v.id == a_id || v.candidate_id == a_id) || (v.id == b_id || v.candidate_id == b_id))
        });
        if let Err(e) = self.store.remove_reviews_for_pair(a_id, b_id) {
            warn!(error = %e, a_id, b_id, "review drain (by pair) persist failed");
        }
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
                warn!(side = "A", error = %e, "failed to save combined index cache");
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
                warn!(side = "B", error = %e, "failed to save combined index cache");
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
