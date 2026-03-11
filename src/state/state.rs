//! MatchState: composite struct holding datasets, combined index, and encoder pool.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use dashmap::DashMap;

use crate::config::Config;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::models::Record;
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
    pub records_a: DashMap<String, Record>,
    pub ids_a: Vec<String>,
    /// Combined embedding index for side A. None if no embedding fields configured.
    pub combined_index_a: Option<Box<dyn VectorDB>>,
    pub records_b: Option<DashMap<String, Record>>,
    pub ids_b: Option<Vec<String>>,
    /// Combined embedding index for side B. None if not loaded or no embedding fields.
    pub combined_index_b: Option<Box<dyn VectorDB>>,
    pub encoder_pool: EncoderPool,
}

impl std::fmt::Debug for MatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchState")
            .field("records_a", &self.records_a.len())
            .field(
                "combined_index_a",
                &self.combined_index_a.as_ref().map(|i| i.len()),
            )
            .field("records_b", &self.records_b.as_ref().map(|r| r.len()))
            .field(
                "combined_index_b",
                &self.combined_index_b.as_ref().map(|i| i.len()),
            )
            .finish()
    }
}

/// Convert a HashMap into a DashMap (one-time move during state loading).
fn into_dashmap(map: HashMap<String, Record>) -> DashMap<String, Record> {
    let dm = DashMap::with_capacity(map.len());
    for (k, v) in map {
        dm.insert(k, v);
    }
    dm
}

/// Load the full match state: datasets, caches, encoder pool.
pub fn load_state(config: Config, opts: &LoadOptions) -> Result<MatchState, MelderError> {
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
        )?;

        (Some(recs), Some(ids), idx)
    } else {
        (None, None, None)
    };

    let total = start.elapsed();
    eprintln!(
        "State loaded in {:.1}s (A: {} records{})",
        total.as_secs_f64(),
        records_a.len(),
        if let Some(ref rb) = records_b {
            format!(", B: {} records", rb.len())
        } else {
            String::new()
        },
    );

    Ok(MatchState {
        config,
        records_a: into_dashmap(records_a),
        ids_a,
        combined_index_a,
        records_b: records_b.map(into_dashmap),
        ids_b,
        combined_index_b,
        encoder_pool,
    })
}
