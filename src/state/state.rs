//! MatchState: composite struct holding datasets, combined index, and encoder pool.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::Config;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::models::Side;
use crate::store::RecordStore;
use crate::store::memory::MemoryStore;
use crate::store::sqlite::open_sqlite;
use crate::vectordb::{self, VectorDB};

/// Options controlling what to load.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Whether to load dataset B (false for batch mode where B is streamed).
    pub load_b: bool,
    /// Whether this is batch mode (affects cache paths used).
    pub batch_mode: bool,
}

/// Composite state holding everything needed for matching.
pub struct MatchState {
    pub config: Config,
    /// Record store (in-memory DashMap or SQLite depending on memory_budget).
    pub store: Arc<dyn RecordStore>,
    pub ids_a: Vec<String>,
    /// Combined embedding index for side A. None if no embedding fields configured.
    pub combined_index_a: Option<Box<dyn VectorDB>>,
    pub ids_b: Option<Vec<String>>,
    /// Combined embedding index for side B. None if not loaded or no embedding fields.
    pub combined_index_b: Option<Box<dyn VectorDB>>,
    pub encoder_pool: EncoderPool,
    /// Temporary directory holding the batch-mode SQLite database (if
    /// `memory_budget` triggered SQLite). Kept alive for the run lifetime;
    /// auto-deleted on drop.
    pub _batch_sqlite_dir: Option<tempfile::TempDir>,
}

impl std::fmt::Debug for MatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchState")
            .field("records_a", &self.store.len(Side::A))
            .field(
                "combined_index_a",
                &self.combined_index_a.as_ref().map(|i| i.len()),
            )
            .field(
                "records_b",
                &if self.ids_b.is_some() {
                    Some(self.store.len(Side::B))
                } else {
                    None
                },
            )
            .field(
                "combined_index_b",
                &self.combined_index_b.as_ref().map(|i| i.len()),
            )
            .finish()
    }
}

/// Load the full match state: datasets, caches, encoder pool.
pub fn load_state(mut config: Config, opts: &LoadOptions) -> Result<MatchState, MelderError> {
    let start = Instant::now();

    // 1. Create encoder pool
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
    eprintln!(
        "Encoder ready (dim={}), took {:.1}s",
        encoder_pool.dim(),
        start.elapsed().as_secs_f64()
    );

    // 1b. Compute memory budget decision (if memory_budget is configured).
    //
    // Runs before data loading so that:
    //  - mmap mode is applied to the vector index build/load step.
    //  - SQLite vs in-memory store choice is made before records are loaded.
    let budget_decision = if let Some(ref budget_str) = config.memory_budget.clone() {
        let budget_bytes = crate::budget::parse_budget(budget_str)
            .unwrap_or_else(|_| crate::budget::available_ram());
        let record_count = crate::budget::estimate_record_count(
            &config.embeddings.a_cache_dir,
            &config.datasets.a.path,
            config.datasets.a.format.as_deref(),
        );
        let n_embedding_fields = config
            .match_fields
            .iter()
            .filter(|mf| mf.method == "embedding")
            .count();
        let use_f16 = matches!(
            config.performance.vector_quantization.as_deref(),
            Some("f16") | Some("bf16")
        );
        let decision = crate::budget::decide(
            budget_bytes,
            record_count,
            encoder_pool.dim(),
            n_embedding_fields,
            use_f16,
        );
        eprintln!(
            "memory_budget: {} → record_store={}, vector_index={}{}",
            budget_str,
            if decision.use_sqlite {
                "sqlite"
            } else {
                "memory"
            },
            if decision.use_mmap { "mmap" } else { "load" },
            if decision.use_sqlite {
                format!(
                    ", sqlite_cache={}MB",
                    decision.sqlite_cache_bytes / (1024 * 1024)
                )
            } else {
                String::new()
            },
        );
        // Auto-select mmap before building the vector index (if not already set).
        if decision.use_mmap && config.performance.vector_index_mode.is_none() {
            config.performance.vector_index_mode = Some("mmap".into());
        }
        Some(decision)
    } else {
        None
    };

    // 2. Load dataset A
    let a_start = Instant::now();
    let (records_a, ids_a) = data::load_dataset(
        Path::new(&config.datasets.a.path),
        &config.datasets.a.id_field,
        &config.required_fields_a,
        config.datasets.a.format.as_deref(),
    )
    .map_err(MelderError::Data)?;
    eprintln!(
        "Loaded dataset A: {} records in {:.1}s",
        records_a.len(),
        a_start.elapsed().as_secs_f64()
    );

    // 3. Build/load A-side combined embedding index
    let combined_index_a = vectordb::build_or_load_combined_index(
        &config.vector_backend,
        Some(&config.embeddings.a_cache_dir),
        &records_a,
        &ids_a,
        &config,
        true,
        &encoder_pool,
        false,
    )?;

    // 4. Optionally load dataset B
    let (records_b, ids_b, combined_index_b) = if opts.load_b {
        let b_start = Instant::now();
        let (recs, ids) = data::load_dataset(
            Path::new(&config.datasets.b.path),
            &config.datasets.b.id_field,
            &config.required_fields_b,
            config.datasets.b.format.as_deref(),
        )
        .map_err(MelderError::Data)?;
        eprintln!(
            "Loaded dataset B: {} records in {:.1}s",
            recs.len(),
            b_start.elapsed().as_secs_f64()
        );

        // Build/load B-side combined embedding index
        let idx = vectordb::build_or_load_combined_index(
            &config.vector_backend,
            config.embeddings.b_cache_dir.as_deref(),
            &recs,
            &ids,
            &config,
            false,
            &encoder_pool,
            false,
        )?;

        (Some(recs), Some(ids), idx)
    } else {
        (None, None, None)
    };

    // 5. Build store from loaded records.
    //
    // When memory_budget triggers SQLite: create a temporary on-disk database,
    // populate it with A-side (and B-side if loaded) records, and use it as the
    // record store. The TempDir is held in `_batch_sqlite_dir` and auto-deleted
    // when `MatchState` drops.
    //
    // Otherwise: build the default in-memory DashMap store.
    let use_sqlite = budget_decision.as_ref().is_some_and(|d| d.use_sqlite);

    let (store, _batch_sqlite_dir): (Arc<dyn RecordStore>, Option<tempfile::TempDir>) =
        if use_sqlite {
            let d = budget_decision.as_ref().unwrap();
            let tmp_dir = tempfile::TempDir::new().map_err(|e| {
                MelderError::Other(anyhow::anyhow!("failed to create temp SQLite dir: {}", e))
            })?;
            let db_path = tmp_dir.path().join("batch.db");
            let cache_kb = d.sqlite_cache_bytes / 1024;
            let (sqlite_store, _, _) = open_sqlite(&db_path, &config.blocking, Some(cache_kb))
                .map_err(|e| MelderError::Other(anyhow::anyhow!("SQLite open failed: {}", e)))?;

            // Populate A-side records and blocking index.
            for (id, record) in &records_a {
                sqlite_store.insert(Side::A, id, record);
                sqlite_store.blocking_insert(Side::A, id, record);
            }

            // Populate B-side records and blocking index (if loaded).
            if let Some(ref recs_b) = records_b {
                for (id, record) in recs_b {
                    sqlite_store.insert(Side::B, id, record);
                    sqlite_store.blocking_insert(Side::B, id, record);
                }
            }

            (Arc::new(sqlite_store), Some(tmp_dir))
        } else {
            let store = Arc::new(MemoryStore::from_records(
                records_a,
                records_b.unwrap_or_default(),
                &config.blocking,
            ));
            (store, None)
        };

    let total = start.elapsed();
    eprintln!(
        "State loaded in {:.1}s (A: {} records{})",
        total.as_secs_f64(),
        store.len(Side::A),
        if ids_b.is_some() {
            format!(", B: {} records", store.len(Side::B))
        } else {
            String::new()
        },
    );

    Ok(MatchState {
        config,
        store,
        ids_a,
        combined_index_a,
        ids_b,
        combined_index_b,
        encoder_pool,
        _batch_sqlite_dir,
    })
}
