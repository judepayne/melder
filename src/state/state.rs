//! MatchState: composite struct holding datasets, combined index, and encoder pool.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::Config;
use crate::data;
use crate::encoder::{Encoder, EncoderOptions, EncoderPool};
use crate::error::MelderError;
use crate::models::Side;
use crate::store::RecordStore;
use crate::store::memory::MemoryStore;
use crate::vectordb::{self, VectorDB};

/// Construct the encoder for the given config.
///
/// Branches on `embeddings.remote_encoder_cmd`: if set, spawns a
/// `SubprocessEncoder` pool; otherwise constructs the local ONNX
/// `EncoderPool`. Both return `Arc<dyn Encoder>` so downstream code is
/// oblivious to the backend.
///
/// `force_cpu` is honoured only for the local ONNX path (live mode sets
/// this to `true` — GPU does not improve single-record latency).
pub(crate) fn build_encoder(
    config: &Config,
    force_cpu: bool,
) -> Result<Arc<dyn Encoder>, MelderError> {
    if let Some(cmd) = config
        .embeddings
        .remote_encoder_cmd
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        // Remote path: validation guarantees pool_size is set.
        let pool_size = config
            .performance
            .encoder_pool_size
            .expect("validated: encoder_pool_size required with remote_encoder_cmd");
        let batch_size = config.performance.encoder_batch_size.unwrap_or(256);
        let timeout_ms = config.performance.encoder_call_timeout_ms.unwrap_or(60_000);
        eprintln!(
            "Initializing remote encoder (cmd={:?}, pool_size={}, timeout_ms={})...",
            cmd, pool_size, timeout_ms
        );
        let enc = crate::encoder::subprocess::SubprocessEncoder::new(
            cmd.to_string(),
            pool_size,
            batch_size,
            std::time::Duration::from_millis(timeout_ms),
        )
        .map_err(MelderError::Encoder)?;
        eprintln!(
            "Remote encoder ready (dim={}, model_id={})",
            enc.dim(),
            enc.model_id()
        );
        Ok(Arc::new(enc))
    } else {
        let gpu = !force_cpu && config.performance.encoder_device.as_deref() == Some("gpu");
        let pool_size = config.performance.encoder_pool_size.unwrap_or_else(|| {
            if gpu {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let default = ((cores as f64 * 0.6).round() as usize).max(2);
                eprintln!(
                    "NOTE: encoder_pool_size not set — defaulting to {} for GPU \
                     (~60% of {} cores). Set explicitly to tune.",
                    default, cores
                );
                default
            } else {
                1
            }
        });
        eprintln!(
            "Initializing encoder pool (model={}, pool_size={}, device={})...",
            config.embeddings.model,
            pool_size,
            if gpu { "gpu" } else { "cpu" }
        );
        let pool = EncoderPool::new(EncoderOptions {
            model_name: config.embeddings.model.clone(),
            pool_size,
            quantized: config.performance.quantized,
            gpu,
            encode_batch_size: config.performance.encoder_batch_size,
        })
        .map_err(MelderError::Encoder)?;
        Ok(Arc::new(pool))
    }
}

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
    /// Record store (in-memory DashMap).
    pub store: Arc<dyn RecordStore>,
    pub ids_a: Vec<String>,
    /// Combined embedding index for side A. None if no embedding fields configured.
    pub combined_index_a: Option<Box<dyn VectorDB>>,
    pub ids_b: Option<Vec<String>>,
    /// Combined embedding index for side B. None if not loaded or no embedding fields.
    pub combined_index_b: Option<Box<dyn VectorDB>>,
    /// The encoder: local `EncoderPool` (ONNX via fastembed) or a
    /// `SubprocessEncoder` (user-supplied remote script), selected at
    /// load time based on `embeddings.remote_encoder_cmd`.
    pub encoder_pool: Arc<dyn Encoder>,
}

impl std::fmt::Debug for MatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchState")
            .field("records_a", &self.store.len(Side::A).unwrap_or(0))
            .field(
                "combined_index_a",
                &self.combined_index_a.as_ref().map(|i| i.len()),
            )
            .field(
                "records_b",
                &if self.ids_b.is_some() {
                    Some(self.store.len(Side::B).unwrap_or(0))
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
pub fn load_state(config: Config, opts: &LoadOptions) -> Result<MatchState, MelderError> {
    let start = Instant::now();

    // 1. Create encoder (local or remote — branch on config). Batch mode
    //    honours `encoder_device: gpu` when set.
    let encoder_pool = build_encoder(&config, false)?;
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
        encoder_pool.as_ref(),
        false,
        Some(Path::new(&config.datasets.a.path)),
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
            encoder_pool.as_ref(),
            false,
            Some(Path::new(&config.datasets.b.path)),
        )?;

        (Some(recs), Some(ids), idx)
    } else {
        (None, None, None)
    };

    // 5. Build in-memory store from loaded records.
    let store = Arc::new(MemoryStore::from_records(
        records_a,
        records_b.unwrap_or_default(),
        &config.blocking,
    ));

    let total = start.elapsed();
    eprintln!(
        "State loaded in {:.1}s (A: {} records{})",
        total.as_secs_f64(),
        store.len(Side::A).unwrap_or(0),
        if ids_b.is_some() {
            format!(", B: {} records", store.len(Side::B).unwrap_or(0))
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
    })
}
