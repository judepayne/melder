//! MatchState: composite struct holding datasets, indices, and encoder pool.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use dashmap::DashMap;

use crate::config::Config;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::index::cache;
use crate::index::VecIndex;
use crate::models::Record;

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
    pub index_a: VecIndex,
    pub records_b: Option<DashMap<String, Record>>,
    pub ids_b: Option<Vec<String>>,
    pub index_b: Option<VecIndex>,
    pub encoder_pool: EncoderPool,
}

impl std::fmt::Debug for MatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchState")
            .field("records_a", &self.records_a.len())
            .field("index_a", &self.index_a.len())
            .field("records_b", &self.records_b.as_ref().map(|r| r.len()))
            .field("index_b", &self.index_b.as_ref().map(|i| i.len()))
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

    // 3. Build/load A-side index
    let index_a = build_or_load_index(
        &config.embeddings.a_index_cache,
        &records_a,
        &ids_a,
        &config,
        true, // is_a_side
        &encoder_pool,
        dim,
    )?;

    // 4. Optionally load dataset B
    let (records_b, ids_b, index_b) = if opts.load_b {
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

        // Build/load B-side index if b_index_cache is configured
        let b_index = if let Some(ref b_cache_path) = config.embeddings.b_index_cache {
            Some(build_or_load_index(
                b_cache_path,
                &recs,
                &ids,
                &config,
                false, // is_b_side
                &encoder_pool,
                dim,
            )?)
        } else {
            None
        };

        (Some(recs), Some(ids), b_index)
    } else {
        (None, None, None)
    };

    let total = start.elapsed();
    eprintln!(
        "State loaded in {:.1}s (A: {} records, index: {} vecs{})",
        total.as_secs_f64(),
        records_a.len(),
        index_a.len(),
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
        index_a,
        records_b: records_b.map(into_dashmap),
        ids_b,
        index_b,
        encoder_pool,
    })
}

/// Build a VecIndex from scratch by encoding all records, or load from cache
/// if fresh.
fn build_or_load_index(
    cache_path: &str,
    records: &HashMap<String, Record>,
    ids: &[String],
    config: &Config,
    is_a_side: bool,
    encoder_pool: &EncoderPool,
    dim: usize,
) -> Result<VecIndex, MelderError> {
    let path = Path::new(cache_path);

    // Check cache staleness
    if !cache::is_cache_stale(path, records.len()) {
        let load_start = Instant::now();
        match cache::load_index(path) {
            Ok(index) => {
                let side = if is_a_side { "A" } else { "B" };
                eprintln!(
                    "Loaded {} index from cache: {} vecs in {:.1}ms",
                    side,
                    index.len(),
                    load_start.elapsed().as_secs_f64() * 1000.0
                );
                return Ok(index);
            }
            Err(e) => {
                eprintln!("Warning: cache load failed ({}), rebuilding...", e);
            }
        }
    }

    // Build from scratch
    let side = if is_a_side { "A" } else { "B" };
    eprintln!("Building {} index ({} records)...", side, records.len());
    let build_start = Instant::now();

    let mut index = VecIndex::new(dim);
    let batch_size = 256;

    // Process in batches
    for (batch_idx, chunk) in ids.chunks(batch_size).enumerate() {
        let texts: Vec<String> = chunk
            .iter()
            .map(|id| {
                let record = records
                    .get(id)
                    .expect("id from ids vec must exist in records");
                primary_embedding_text(record, config, is_a_side)
            })
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let vecs = encoder_pool
            .encode(&text_refs)
            .map_err(MelderError::Encoder)?;

        for (id, vec) in chunk.iter().zip(vecs.into_iter()) {
            index.upsert(id, &vec);
        }

        let done = (batch_idx + 1) * batch_size;
        if done % 1000 == 0 || done >= ids.len() {
            eprintln!(
                "  encoded {}/{} {} records...",
                done.min(ids.len()),
                ids.len(),
                side
            );
        }
    }

    let build_elapsed = build_start.elapsed();
    eprintln!(
        "{} index built: {} vecs in {:.1}s ({:.0} records/sec)",
        side,
        index.len(),
        build_elapsed.as_secs_f64(),
        index.len() as f64 / build_elapsed.as_secs_f64()
    );

    // Save to cache
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    if let Err(e) = cache::save_index(path, &index) {
        eprintln!("Warning: failed to save {} index cache: {}", side, e);
    } else {
        eprintln!("Saved {} index cache to {}", side, cache_path);
    }

    Ok(index)
}
