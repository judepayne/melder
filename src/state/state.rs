//! MatchState: composite struct holding datasets, field indexes, and encoder pool.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use dashmap::DashMap;

use crate::config::Config;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::models::Record;
use crate::vectordb;
use crate::vectordb::field_indexes::FieldIndexes;

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
    pub field_indexes_a: FieldIndexes,
    pub records_b: Option<DashMap<String, Record>>,
    pub ids_b: Option<Vec<String>>,
    pub field_indexes_b: Option<FieldIndexes>,
    pub encoder_pool: EncoderPool,
}

impl std::fmt::Debug for MatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchState")
            .field("records_a", &self.records_a.len())
            .field("field_indexes_a", &self.field_indexes_a.len())
            .field("records_b", &self.records_b.as_ref().map(|r| r.len()))
            .field(
                "field_indexes_b",
                &self.field_indexes_b.as_ref().map(|fi| fi.len()),
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

/// Derive a field index cache base path from an index cache path.
///
/// Strips the extension to produce a base path that `FieldIndexes::save_all()`
/// will append field-specific suffixes to. For example:
/// `cache/a_10000.index` → `cache/a_10000`
pub fn field_index_cache_base(index_cache_path: &str) -> String {
    let p = Path::new(index_cache_path);
    p.with_extension("").to_string_lossy().to_string()
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
    let encoder_pool =
        EncoderPool::new(&config.embeddings.model, pool_size).map_err(MelderError::Encoder)?;
    let dim = encoder_pool.dim();
    eprintln!(
        "Encoder ready (dim={}), took {:.1}s",
        dim,
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

    // 3. Build/load A-side field indexes
    let a_cache_base = field_index_cache_base(&config.embeddings.a_index_cache);
    let field_indexes_a = vectordb::build_or_load_field_indexes(
        &config.vector_backend,
        Some(&a_cache_base),
        &records_a,
        &ids_a,
        &config,
        true,
        &encoder_pool,
        dim,
    )?;

    // 4. Optionally load dataset B
    let (records_b, ids_b, field_indexes_b) = if opts.load_b {
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

        // Build/load B-side field indexes
        let b_cache_base = config
            .embeddings
            .b_index_cache
            .as_deref()
            .map(field_index_cache_base);
        let fi = vectordb::build_or_load_field_indexes(
            &config.vector_backend,
            b_cache_base.as_deref(),
            &recs,
            &ids,
            &config,
            false,
            &encoder_pool,
            dim,
        )?;

        (Some(recs), Some(ids), Some(fi))
    } else {
        (None, None, None)
    };

    let total = start.elapsed();
    eprintln!(
        "State loaded in {:.1}s (A: {} records, field indexes: {} vecs{})",
        total.as_secs_f64(),
        records_a.len(),
        field_indexes_a.len(),
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
        field_indexes_a,
        records_b: records_b.map(into_dashmap),
        ids_b,
        field_indexes_b,
        encoder_pool,
    })
}
