//! MatchState: composite struct holding datasets, indices, and encoder pool.

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
use crate::vectordb::field_vectors::FieldVectors;

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
    pub field_vecs_a: FieldVectors,
    pub records_b: Option<DashMap<String, Record>>,
    pub ids_b: Option<Vec<String>>,
    pub field_vecs_b: Option<FieldVectors>,
    pub encoder_pool: EncoderPool,
}

impl std::fmt::Debug for MatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchState")
            .field("records_a", &self.records_a.len())
            .field("field_vecs_a", &self.field_vecs_a.len())
            .field("records_b", &self.records_b.as_ref().map(|r| r.len()))
            .field(
                "field_vecs_b",
                &self.field_vecs_b.as_ref().map(|fv| fv.len()),
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

/// Build the embedding text for a record by concatenating values of all
/// embedding-method match fields.
///
/// For A-side records, uses `field_a` names; for B-side, uses `field_b` names.
/// Falls back to the first fuzzy field, then the ID field.
pub fn primary_embedding_text(record: &Record, config: &Config, is_a_side: bool) -> String {
    let mut parts: Vec<&str> = Vec::new();

    // Collect values from embedding fields
    for mf in &config.match_fields {
        if mf.method == "embedding" {
            let field_name = if is_a_side { &mf.field_a } else { &mf.field_b };
            if let Some(val) = record.get(field_name) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
            }
        }
    }

    if !parts.is_empty() {
        return parts.join(" ");
    }

    // Fallback: first fuzzy field
    for mf in &config.match_fields {
        if mf.method == "fuzzy" {
            let field_name = if is_a_side { &mf.field_a } else { &mf.field_b };
            if let Some(val) = record.get(field_name) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }

    // Fallback: ID field value
    let id_field = if is_a_side {
        &config.datasets.a.id_field
    } else {
        &config.datasets.b.id_field
    };
    record
        .get(id_field)
        .map(|v| v.trim().to_string())
        .unwrap_or_default()
}

/// Derive the field vectors cache path from an index cache path.
///
/// Replaces the extension with `.fieldvecs`. For example:
/// `cache/a_10000.index` → `cache/a_10000.fieldvecs`
pub fn field_vecs_cache_path(index_cache_path: &str) -> String {
    let p = Path::new(index_cache_path);
    p.with_extension("fieldvecs").to_string_lossy().to_string()
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

    // 3. Build/load A-side field vectors
    let a_fv_cache = field_vecs_cache_path(&config.embeddings.a_index_cache);
    let field_vecs_a = vectordb::build_or_load_field_vectors(
        Some(&a_fv_cache),
        &records_a,
        &ids_a,
        &config,
        true,
        &encoder_pool,
        dim,
    )?;

    // 4. Optionally load dataset B
    let (records_b, ids_b, field_vecs_b) = if opts.load_b {
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

        // Build/load B-side field vectors
        let b_fv_cache = config
            .embeddings
            .b_index_cache
            .as_deref()
            .map(field_vecs_cache_path);
        let fv = vectordb::build_or_load_field_vectors(
            b_fv_cache.as_deref(),
            &recs,
            &ids,
            &config,
            false,
            &encoder_pool,
            dim,
        )?;

        (Some(recs), Some(ids), Some(fv))
    } else {
        (None, None, None)
    };

    let total = start.elapsed();
    eprintln!(
        "State loaded in {:.1}s (A: {} records, field_vecs: {} entries{})",
        total.as_secs_f64(),
        records_a.len(),
        field_vecs_a.len(),
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
        field_vecs_a,
        records_b: records_b.map(into_dashmap),
        ids_b,
        field_vecs_b,
        encoder_pool,
    })
}
