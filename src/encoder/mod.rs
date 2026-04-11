//! Encoder pool for embedding inference via `fastembed`.
//!
//! Wraps one or more `TextEmbedding` instances behind `Mutex` locks.
//! Each instance is an independent ONNX session with its own tokenizer
//! state. Model weights are memory-mapped by the OS, so multiple instances
//! share the same physical pages.
//!
//! Supports three model sources:
//! - Named models (e.g. `"all-MiniLM-L6-v2"`) — downloaded by fastembed on
//!   first use and cached on disk.
//! - Local paths (e.g. `"./models/round_1"` or `"/abs/path/model.onnx"`) —
//!   loaded directly from disk using fastembed's `UserDefinedEmbeddingModel`
//!   API. The directory must contain `model.onnx` plus the four standard
//!   HuggingFace tokenizer files.
//! - Builtin model (`"builtin"`) — compiled into the binary when built with
//!   `--features builtin-model`. Zero network access, zero disk dependency.
//!   The model is specified at build time via `MELDER_BUILTIN_MODEL` env var
//!   (defaults to `themelder/arctic-embed-xs-entity-resolution`).

pub mod coordinator;
pub mod subprocess;

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fastembed::{
    EmbeddingModel, ExecutionProviderDispatch, InitOptions, InitOptionsUserDefined, Pooling,
    TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use tracing::{info, info_span};

use crate::error::EncoderError;

// ---------------------------------------------------------------------------
// Encoder trait
// ---------------------------------------------------------------------------

/// Per-record encoding outcome.
///
/// Used by `Encoder::encode_detailed` to surface per-record errors
/// (e.g. a remote embedding service rejecting a single text for content
/// policy reasons) without failing the whole batch at the trait level.
/// Phase 1's fail-fast policy still collapses any error here into a
/// whole-batch failure at the call site, but preserves per-record context
/// in tracing events before doing so.
#[derive(Debug, Clone)]
pub enum EncodeResult {
    Vector(Vec<f32>),
    Error(String),
}

/// Trait for text embedding encoders.
///
/// The default implementation is `EncoderPool` (local ONNX inference).
/// Alternative implementations might call a remote embedding service
/// via subprocess (`SubprocessEncoder`), use a different model runtime, etc.
pub trait Encoder: Send + Sync + std::fmt::Debug {
    /// Encode a batch of texts into embedding vectors.
    ///
    /// Local-encoder simple path: all-or-nothing. Remote encoders can
    /// emit per-record errors; callers that want to distinguish per-record
    /// from whole-batch failures should call `encode_detailed` instead.
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError>;

    /// Per-record-aware encoding. The default impl wraps `encode()` so
    /// existing impls don't need to change; `SubprocessEncoder` overrides
    /// this to return per-record `EncodeResult::Error` variants for records
    /// the remote service rejected.
    fn encode_detailed(&self, texts: &[&str]) -> Result<Vec<EncodeResult>, EncoderError> {
        self.encode(texts)
            .map(|vs| vs.into_iter().map(EncodeResult::Vector).collect())
    }

    /// Encode a single text, returning a single vector.
    fn encode_one(&self, text: &str) -> Result<Vec<f32>, EncoderError> {
        let results = self.encode(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EncoderError::Inference("empty result from encoder".into()))
    }

    /// The dimensionality of the embedding vectors produced by this encoder.
    fn dim(&self) -> usize;

    /// Number of concurrent encoding slots available.
    fn pool_size(&self) -> usize;

    /// Maximum batch size for a single encode call.
    fn encode_batch_size(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Stderr suppression for CoreML framework noise
// ---------------------------------------------------------------------------

/// Temporarily suppress stderr output. Returns a guard that restores stderr
/// on drop. Used to silence CoreML's "Context leak detected" messages during
/// ONNX session creation.
#[cfg(all(feature = "gpu-encode", target_os = "macos"))]
struct SuppressStderr {
    saved_fd: i32,
}

#[cfg(all(feature = "gpu-encode", target_os = "macos"))]
impl SuppressStderr {
    fn new() -> Option<Self> {
        unsafe {
            let saved = libc::dup(2);
            if saved < 0 {
                return None;
            }
            let devnull = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if devnull < 0 {
                libc::close(saved);
                return None;
            }
            libc::dup2(devnull, 2);
            libc::close(devnull);
            Some(Self { saved_fd: saved })
        }
    }
}

#[cfg(all(feature = "gpu-encode", target_os = "macos"))]
impl Drop for SuppressStderr {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_fd, 2);
            libc::close(self.saved_fd);
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder options
// ---------------------------------------------------------------------------

/// Options for constructing an `EncoderPool`.
pub struct EncoderOptions {
    pub model_name: String,
    pub pool_size: usize,
    pub quantized: bool,
    /// Use GPU (CoreML on macOS, CUDA on Linux) for ONNX inference.
    pub gpu: bool,
    /// ONNX batch size override. None = auto (64 for CPU, 256 for GPU).
    pub encode_batch_size: Option<usize>,
}

// ---------------------------------------------------------------------------
// Named model resolution
// ---------------------------------------------------------------------------

/// Map a model name string to a `fastembed::EmbeddingModel` enum variant.
///
/// When `quantized` is true the INT8 quantised variant is selected where
/// available. Returns an error if the model has no quantised variant.
fn resolve_model(model_name: &str, quantized: bool) -> Result<EmbeddingModel, EncoderError> {
    let base = match model_name {
        "all-MiniLM-L6-v2" | "sentence-transformers/all-MiniLM-L6-v2" => {
            if quantized {
                EmbeddingModel::AllMiniLML6V2Q
            } else {
                EmbeddingModel::AllMiniLML6V2
            }
        }
        "all-MiniLM-L12-v2" | "sentence-transformers/all-MiniLM-L12-v2" => {
            if quantized {
                EmbeddingModel::AllMiniLML12V2Q
            } else {
                EmbeddingModel::AllMiniLML12V2
            }
        }
        "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => {
            if quantized {
                return Err(EncoderError::ModelNotFound {
                    model: format!("{} (no quantised variant available)", model_name),
                });
            }
            EmbeddingModel::BGESmallENV15
        }
        "bge-base-en-v1.5" | "BAAI/bge-base-en-v1.5" => {
            if quantized {
                return Err(EncoderError::ModelNotFound {
                    model: format!("{} (no quantised variant available)", model_name),
                });
            }
            EmbeddingModel::BGEBaseENV15
        }
        "bge-large-en-v1.5" | "BAAI/bge-large-en-v1.5" => {
            if quantized {
                return Err(EncoderError::ModelNotFound {
                    model: format!("{} (no quantised variant available)", model_name),
                });
            }
            EmbeddingModel::BGELargeENV15
        }
        _ => {
            return Err(EncoderError::ModelNotFound {
                model: model_name.to_string(),
            });
        }
    };
    Ok(base)
}

/// Dimension of the embedding vector for a given named model.
fn model_dim(model: &EmbeddingModel) -> usize {
    match model {
        EmbeddingModel::AllMiniLML6V2 => 384,
        EmbeddingModel::AllMiniLML6V2Q => 384,
        EmbeddingModel::AllMiniLML12V2 => 384,
        EmbeddingModel::AllMiniLML12V2Q => 384,
        EmbeddingModel::BGESmallENV15 => 384,
        EmbeddingModel::BGEBaseENV15 => 768,
        EmbeddingModel::BGELargeENV15 => 1024,
        other => {
            tracing::warn!(model = ?other, "unknown model variant, defaulting to dim=384");
            384
        }
    }
}

// ---------------------------------------------------------------------------
// Builtin model (compiled into the binary)
// ---------------------------------------------------------------------------

#[cfg(feature = "builtin-model")]
mod builtin {
    pub const MODEL_ONNX: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/builtin_model/model.onnx"));
    pub const TOKENIZER: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/builtin_model/tokenizer.json"));
    pub const CONFIG: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/builtin_model/config.json"));
    pub const SPECIAL_TOKENS: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/builtin_model/special_tokens_map.json"
    ));
    pub const TOKENIZER_CONFIG: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/builtin_model/tokenizer_config.json"
    ));
}

// ---------------------------------------------------------------------------
// Local path detection and loading
// ---------------------------------------------------------------------------

/// Returns true when `model_name` looks like a filesystem path rather than
/// a HuggingFace model name.
///
/// Triggers on absolute paths, relative paths starting with `./` or `../`,
/// strings ending in `.onnx`, and any name that resolves to an existing
/// file or directory on disk.
fn is_local_model_path(model_name: &str) -> bool {
    if model_name.starts_with('/')
        || model_name.starts_with("./")
        || model_name.starts_with("../")
        || model_name.ends_with(".onnx")
    {
        return true;
    }
    Path::new(model_name).exists()
}

/// Read the embedding output dimension from a HuggingFace `config.json`.
///
/// Looks for `"hidden_size": N` with a minimal text scan — no full JSON
/// parse required. Returns `None` if the file is absent or the key is missing.
fn detect_dim_from_config(config_path: impl AsRef<Path>) -> Option<usize> {
    let bytes = std::fs::read(config_path).ok()?;
    detect_dim_from_bytes(&bytes)
}

/// Read the embedding output dimension from `config.json` bytes.
///
/// Looks for `"hidden_size": N` with a minimal text scan.
fn detect_dim_from_bytes(bytes: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(bytes).ok()?;
    let key = "\"hidden_size\"";
    let pos = text.find(key)?;
    let after = text[pos + key.len()..].trim_start();
    let after = after.strip_prefix(':')?.trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

// ---------------------------------------------------------------------------
// EncoderPool
// ---------------------------------------------------------------------------

/// Pool of `TextEmbedding` instances for concurrent encoding.
///
/// Each slot is behind a `std::sync::Mutex`. Callers acquire a slot,
/// run the (blocking) `embed()` call, and release the slot.
pub struct EncoderPool {
    encoders: Vec<Mutex<TextEmbedding>>,
    dim: usize,
    /// Round-robin counter for fallback slot selection when all slots are
    /// busy via try_lock. Prevents thundering herd on slot 0.
    next_fallback: AtomicUsize,
    /// ONNX batch size for fastembed encode calls.
    encode_batch_size: usize,
}

impl std::fmt::Debug for EncoderPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderPool")
            .field("pool_size", &self.encoders.len())
            .field("dim", &self.dim)
            .finish()
    }
}

impl EncoderPool {
    /// Create a new encoder pool.
    ///
    /// `model_name` resolution order:
    /// 1. `"builtin"` — load from bytes compiled into the binary (requires
    ///    `--features builtin-model`).
    /// 2. Local path (absolute, `./`, `../`, `.onnx` suffix, or resolves on
    ///    disk) — loaded directly from the filesystem.
    /// 3. HuggingFace repo ID (contains `/`) — downloaded from the Hub.
    /// 4. Named fastembed model (e.g. `"all-MiniLM-L6-v2"`).
    ///
    /// For named models: `pool_size` instances are created; the first run may
    /// download the model (~90 MB fp32 / ~23 MB quantised).
    ///
    /// For local paths: `quantized` is ignored (use the `.onnx` file you
    /// want directly); `pool_size` instances share the loaded bytes.
    pub fn new(opts: EncoderOptions) -> Result<Self, EncoderError> {
        let encode_batch_size = opts
            .encode_batch_size
            .unwrap_or(if opts.gpu { 256 } else { 64 });

        // With gpu-encode, ort uses load-dynamic for ALL sessions (even CPU).
        // Ensure the dylib is discoverable before creating any sessions.
        #[cfg(feature = "gpu-encode")]
        Self::ensure_ort_dylib()?;

        let eps = Self::build_execution_providers(opts.gpu)?;

        // Suppress CoreML "Context leak detected" stderr noise during
        // session creation. The guard restores stderr on drop.
        #[cfg(all(feature = "gpu-encode", target_os = "macos"))]
        let _stderr_guard = if opts.gpu {
            SuppressStderr::new()
        } else {
            None
        };

        // 1. Builtin model — compiled into the binary.
        if opts.model_name == "builtin" {
            #[cfg(feature = "builtin-model")]
            {
                return Self::new_from_builtin(opts.pool_size, &eps, encode_batch_size);
            }
            #[cfg(not(feature = "builtin-model"))]
            {
                return Err(EncoderError::ModelNotFound {
                    model: "builtin model requested but binary was not built with \
                            --features builtin-model"
                        .to_string(),
                });
            }
        }

        // 2. Local filesystem path.
        if is_local_model_path(&opts.model_name) {
            return Self::new_from_local_path(
                &opts.model_name,
                opts.pool_size,
                &eps,
                encode_batch_size,
            );
        }
        // 3. HuggingFace repo ID (contains `/` but isn't a local path).
        if opts.model_name.contains('/') {
            return Self::new_from_hub(&opts.model_name, opts.pool_size, &eps, encode_batch_size);
        }
        let model = resolve_model(&opts.model_name, opts.quantized)?;
        let dim = model_dim(&model);
        let pool_size = opts.pool_size.max(1);

        let mut encoders = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let show_progress = i == 0;
            let te = TextEmbedding::try_new(
                InitOptions::new(model.clone())
                    .with_show_download_progress(show_progress)
                    .with_execution_providers(eps.clone()),
            )
            .map_err(|e| EncoderError::Inference(e.to_string()))?;
            encoders.push(Mutex::new(te));
        }

        Ok(Self {
            encoders,
            dim,
            next_fallback: AtomicUsize::new(0),
            encode_batch_size,
        })
    }

    /// Build the ONNX execution provider list.
    #[allow(clippy::needless_return)]
    fn build_execution_providers(
        gpu: bool,
    ) -> Result<Vec<ExecutionProviderDispatch>, EncoderError> {
        if !gpu {
            return Ok(vec![]);
        }

        #[cfg(feature = "gpu-encode")]
        {
            #[cfg(target_os = "macos")]
            {
                use ort::ep::CoreML;
                use ort::ep::coreml::{ComputeUnits, ModelFormat};
                info!("GPU encoding enabled: CoreML (MLProgram, CPUAndGPU)");
                return Ok(vec![
                    CoreML::default()
                        .with_compute_units(ComputeUnits::CPUAndGPU)
                        .with_model_format(ModelFormat::MLProgram)
                        .build(),
                ]);
            }
            #[cfg(target_os = "linux")]
            {
                use ort::ep::CUDA;
                info!("GPU encoding enabled: CUDA");
                return Ok(vec![CUDA::default().build()]);
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            {
                eprintln!(
                    "WARNING: encoder_device: gpu is not supported on this platform, \
                     falling back to CPU"
                );
                return Ok(vec![]);
            }
        }

        #[cfg(not(feature = "gpu-encode"))]
        {
            Err(EncoderError::Inference(
                "GPU encoding requires building with --features gpu-encode".to_string(),
            ))
        }
    }

    /// Ensure `ORT_DYLIB_PATH` is set before creating ONNX sessions.
    ///
    /// # Safety
    /// Uses `std::env::set_var` which is not thread-safe. Must be called
    /// during single-threaded startup, before the HTTP server or any
    /// worker threads are spawned. This is guaranteed because
    /// `EncoderPool::new()` is invoked during session/state construction,
    /// which completes before the Axum server binds.
    #[cfg(feature = "gpu-encode")]
    #[allow(clippy::needless_return)]
    fn ensure_ort_dylib() -> Result<(), EncoderError> {
        if std::env::var("ORT_DYLIB_PATH")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        {
            let brew_path = "/opt/homebrew/lib/libonnxruntime.dylib";
            if std::path::Path::new(brew_path).exists() {
                info!(path = brew_path, "auto-detected onnxruntime via Homebrew");
                unsafe { std::env::set_var("ORT_DYLIB_PATH", brew_path) };
                return Ok(());
            }
            let intel_path = "/usr/local/lib/libonnxruntime.dylib";
            if std::path::Path::new(intel_path).exists() {
                info!(path = intel_path, "auto-detected onnxruntime via Homebrew");
                unsafe { std::env::set_var("ORT_DYLIB_PATH", intel_path) };
                return Ok(());
            }
            Err(EncoderError::Inference(
                "onnxruntime not found. Install via `brew install onnxruntime` \
                 or set ORT_DYLIB_PATH to the libonnxruntime.dylib location"
                    .to_string(),
            ))
        }

        #[cfg(target_os = "linux")]
        {
            // Check common system locations where libonnxruntime.so might be.
            let candidates = [
                "/usr/lib/libonnxruntime.so",
                "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
                "/usr/local/lib/libonnxruntime.so",
            ];
            for path in &candidates {
                if std::path::Path::new(path).exists() {
                    info!(path, "auto-detected onnxruntime");
                    unsafe { std::env::set_var("ORT_DYLIB_PATH", path) };
                    return Ok(());
                }
            }
            Err(EncoderError::Inference(
                "onnxruntime not found. Download the CUDA-enabled ONNX Runtime build from \
                 https://github.com/microsoft/onnxruntime/releases and set ORT_DYLIB_PATH \
                 to the libonnxruntime.so location"
                    .to_string(),
            ))
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Err(EncoderError::Inference(
                "gpu-encode requires macOS or Linux. Set ORT_DYLIB_PATH to the \
                 libonnxruntime shared library location"
                    .to_string(),
            ))
        }
    }

    /// Download a model from HuggingFace Hub and load it.
    ///
    /// Downloads `model.onnx`, `tokenizer.json`, `config.json`,
    /// `special_tokens_map.json`, and `tokenizer_config.json` into the
    /// hf-hub cache directory, then delegates to [`new_from_local_path`].
    fn new_from_hub(
        repo_id: &str,
        pool_size: usize,
        eps: &[ExecutionProviderDispatch],
        encode_batch_size: usize,
    ) -> Result<Self, EncoderError> {
        use hf_hub::api::sync::Api;

        info!(repo_id = repo_id, "downloading model from huggingface hub");
        let api = Api::new().map_err(|e| EncoderError::ModelNotFound {
            model: format!("failed to init HuggingFace API: {}", e),
        })?;
        let repo = api.model(repo_id.to_string());

        let required_files = [
            "model.onnx",
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ];

        let mut dir_path: Option<std::path::PathBuf> = None;
        for fname in &required_files {
            let path = repo.get(fname).map_err(|e| EncoderError::ModelNotFound {
                model: format!("failed to download {}/{}: {}", repo_id, fname, e),
            })?;
            if dir_path.is_none() {
                dir_path = path.parent().map(|p| p.to_path_buf());
            }
        }

        let dir = dir_path.ok_or_else(|| EncoderError::ModelNotFound {
            model: format!("no files downloaded from {}", repo_id),
        })?;

        info!(path = %dir.display(), "model cached");

        Self::new_from_local_path(
            dir.to_str().unwrap_or(repo_id),
            pool_size,
            eps,
            encode_batch_size,
        )
    }

    /// Build an `EncoderPool` from model bytes embedded in the binary.
    ///
    /// The model files are compiled in via `include_bytes!()` when the
    /// `builtin-model` feature is enabled. No filesystem or network access.
    #[cfg(feature = "builtin-model")]
    fn new_from_builtin(
        pool_size: usize,
        eps: &[ExecutionProviderDispatch],
        encode_batch_size: usize,
    ) -> Result<Self, EncoderError> {
        info!("loading builtin embedding model");

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: builtin::TOKENIZER.to_vec(),
            config_file: builtin::CONFIG.to_vec(),
            special_tokens_map_file: builtin::SPECIAL_TOKENS.to_vec(),
            tokenizer_config_file: builtin::TOKENIZER_CONFIG.to_vec(),
        };

        let dim = detect_dim_from_bytes(builtin::CONFIG).unwrap_or(384);

        let user_model =
            UserDefinedEmbeddingModel::new(builtin::MODEL_ONNX.to_vec(), tokenizer_files)
                .with_pooling(Pooling::Mean);

        let pool_size = pool_size.max(1);
        let mut encoders = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let te = TextEmbedding::try_new_from_user_defined(
                user_model.clone(),
                InitOptionsUserDefined::new().with_execution_providers(eps.to_vec()),
            )
            .map_err(|e| EncoderError::Inference(e.to_string()))?;
            encoders.push(Mutex::new(te));
        }

        info!(dim, pool_size, "builtin model loaded");

        Ok(Self {
            encoders,
            dim,
            next_fallback: AtomicUsize::new(0),
            encode_batch_size,
        })
    }

    /// Build an `EncoderPool` from a local directory or `.onnx` file.
    ///
    /// Expected directory layout (standard `optimum-cli export onnx` output
    /// combined with `sentence-transformers` tokenizer files):
    ///
    /// ```text
    /// model.onnx
    /// tokenizer.json
    /// config.json
    /// special_tokens_map.json
    /// tokenizer_config.json
    /// ```
    ///
    /// When `path` points to a file, its parent directory is used for the
    /// tokenizer files. Mean pooling is always applied — suitable for all
    /// fine-tuned MiniLM / BERT-family models.
    fn new_from_local_path(
        path: &str,
        pool_size: usize,
        eps: &[ExecutionProviderDispatch],
        encode_batch_size: usize,
    ) -> Result<Self, EncoderError> {
        let model_path = Path::new(path);

        let (dir, onnx_file) = if model_path.is_dir() {
            (model_path, model_path.join("model.onnx"))
        } else {
            let dir = model_path.parent().unwrap_or(Path::new("."));
            (dir, model_path.to_path_buf())
        };

        if !onnx_file.exists() {
            return Err(EncoderError::ModelNotFound {
                model: format!("ONNX file not found: {}", onnx_file.display()),
            });
        }

        let read_file = |name: &str| -> Result<Vec<u8>, EncoderError> {
            std::fs::read(dir.join(name)).map_err(|e| EncoderError::ModelNotFound {
                model: format!("tokenizer file '{}' not found: {}", name, e),
            })
        };

        let onnx_bytes = std::fs::read(&onnx_file).map_err(|e| EncoderError::ModelNotFound {
            model: format!("failed to read {}: {}", onnx_file.display(), e),
        })?;

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: read_file("tokenizer.json")?,
            config_file: read_file("config.json")?,
            special_tokens_map_file: read_file("special_tokens_map.json")?,
            tokenizer_config_file: read_file("tokenizer_config.json")?,
        };

        // Detect output dim from config.json; default to 384 (MiniLM).
        let dim = detect_dim_from_config(dir.join("config.json")).unwrap_or(384);

        let user_model =
            UserDefinedEmbeddingModel::new(onnx_bytes, tokenizer_files).with_pooling(Pooling::Mean);

        let pool_size = pool_size.max(1);
        let mut encoders = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let te = TextEmbedding::try_new_from_user_defined(
                user_model.clone(),
                InitOptionsUserDefined::new().with_execution_providers(eps.to_vec()),
            )
            .map_err(|e| EncoderError::Inference(e.to_string()))?;
            encoders.push(Mutex::new(te));
        }

        Ok(Self {
            encoders,
            dim,
            next_fallback: AtomicUsize::new(0),
            encode_batch_size,
        })
    }
}

impl Encoder for EncoderPool {
    fn encode_batch_size(&self) -> usize {
        self.encode_batch_size
    }

    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        let _span = info_span!("onnx_encode", n = texts.len()).entered();
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let text_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let batch_size = Some(self.encode_batch_size);

        // Try round-robin
        for encoder in &self.encoders {
            if let Ok(mut guard) = encoder.try_lock() {
                return guard
                    .embed(text_vec, batch_size)
                    .map_err(|e| EncoderError::Inference(e.to_string()));
            }
        }

        // All busy — block on a round-robin slot to distribute contention
        // evenly instead of thundering-herding on slot 0.
        let n = self.encoders.len();
        let slot = self.next_fallback.fetch_add(1, Ordering::Relaxed) % n;
        let mut guard = self.encoders[slot]
            .lock()
            .map_err(|e| EncoderError::Inference(format!("mutex poisoned: {}", e)))?;
        guard
            .embed(text_vec, batch_size)
            .map_err(|e| EncoderError::Inference(e.to_string()))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn pool_size(&self) -> usize {
        self.encoders.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> EncoderPool {
        EncoderPool::new(EncoderOptions {
            model_name: "all-MiniLM-L6-v2".to_string(),
            pool_size: 1,
            quantized: false,
            gpu: false,
            encode_batch_size: None,
        })
        .expect("failed to create pool")
    }

    #[test]
    fn create_pool_and_encode() {
        let pool = make_pool();
        assert_eq!(pool.dim(), 384);
        assert_eq!(pool.pool_size(), 1);

        let vecs = pool
            .encode(&["hello world", "test sentence"])
            .expect("encode failed");
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 384);
        assert_eq!(vecs[1].len(), 384);
    }

    #[test]
    fn encode_one() {
        let pool = make_pool();
        let vec = pool.encode_one("hello world").expect("encode_one failed");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn self_similarity_high() {
        let pool = make_pool();
        let vecs = pool
            .encode(&["hello world", "hello world"])
            .expect("encode failed");

        // Dot product of identical normalized vectors should be ~1.0
        let dot: f32 = vecs[0].iter().zip(vecs[1].iter()).map(|(a, b)| a * b).sum();
        assert!(
            (dot - 1.0).abs() < 0.01,
            "self-similarity dot product = {}, expected ~1.0",
            dot
        );
    }

    #[test]
    fn different_sentences_lower_similarity() {
        let pool = make_pool();
        let vecs = pool
            .encode(&["hello world", "quantum physics equations"])
            .expect("encode failed");

        let dot: f32 = vecs[0].iter().zip(vecs[1].iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot < 1.0,
            "different sentences should have dot product < 1.0, got {}",
            dot
        );
    }

    #[test]
    fn empty_input() {
        let pool = make_pool();
        let vecs = pool.encode(&[]).expect("encode empty failed");
        assert!(vecs.is_empty());
    }

    #[test]
    fn invalid_model() {
        let err = EncoderPool::new(EncoderOptions {
            model_name: "nonexistent-model-xyz".to_string(),
            pool_size: 1,
            quantized: false,
            gpu: false,
            encode_batch_size: None,
        })
        .unwrap_err();
        assert!(matches!(err, EncoderError::ModelNotFound { .. }));
    }

    #[test]
    fn detect_dim_from_valid_config() {
        let json = br#"{"architectures":["BertModel"],"hidden_size":384,"num_attention_heads":12}"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json).unwrap();
        assert_eq!(detect_dim_from_config(tmp.path()), Some(384));
    }

    #[test]
    fn detect_dim_missing_file() {
        assert_eq!(detect_dim_from_config("/nonexistent/config.json"), None);
    }

    #[test]
    fn detect_dim_missing_key() {
        let json = br#"{"architectures":["BertModel"],"vocab_size":30522}"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json).unwrap();
        assert_eq!(detect_dim_from_config(tmp.path()), None);
    }

    #[test]
    fn local_path_nonexistent_onnx_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = EncoderPool::new(EncoderOptions {
            model_name: dir.path().to_string_lossy().to_string(),
            pool_size: 1,
            quantized: false,
            gpu: false,
            encode_batch_size: None,
        })
        .unwrap_err();
        assert!(
            matches!(err, EncoderError::ModelNotFound { .. }),
            "expected ModelNotFound, got {:?}",
            err
        );
    }
}
