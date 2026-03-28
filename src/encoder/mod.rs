//! Encoder pool for embedding inference via `fastembed`.
//!
//! Wraps one or more `TextEmbedding` instances behind `Mutex` locks.
//! Each instance is an independent ONNX session with its own tokenizer
//! state. Model weights are memory-mapped by the OS, so multiple instances
//! share the same physical pages.
//!
//! Supports two model sources:
//! - Named models (e.g. `"all-MiniLM-L6-v2"`) — downloaded by fastembed on
//!   first use and cached on disk.
//! - Local paths (e.g. `"./models/round_1"` or `"/abs/path/model.onnx"`) —
//!   loaded directly from disk using fastembed's `UserDefinedEmbeddingModel`
//!   API. The directory must contain `model.onnx` plus the four standard
//!   HuggingFace tokenizer files.

pub mod coordinator;

use std::path::Path;
use std::sync::Mutex;

use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use tracing::{info, info_span};

use crate::error::EncoderError;

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
        _ => 384, // fallback
    }
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
    let text = std::str::from_utf8(&bytes).ok()?;
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
    /// `model_name` accepts either a HuggingFace model name (e.g.
    /// `"all-MiniLM-L6-v2"`) or a local path to a directory or `.onnx` file
    /// produced by `optimum-cli export onnx`. Local paths are detected
    /// automatically — see [`is_local_model_path`].
    ///
    /// For named models: `pool_size` instances are created; the first run may
    /// download the model (~90 MB fp32 / ~23 MB quantised).
    ///
    /// For local paths: `quantized` is ignored (use the `.onnx` file you
    /// want directly); `pool_size` instances share the loaded bytes.
    pub fn new(model_name: &str, pool_size: usize, quantized: bool) -> Result<Self, EncoderError> {
        if is_local_model_path(model_name) {
            return Self::new_from_local_path(model_name, pool_size);
        }
        // If the name looks like a HuggingFace repo ID (contains `/` but
        // isn't a local path), download it from the Hub and load locally.
        if model_name.contains('/') {
            return Self::new_from_hub(model_name, pool_size);
        }
        let model = resolve_model(model_name, quantized)?;
        let dim = model_dim(&model);
        let pool_size = pool_size.max(1);

        let mut encoders = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let show_progress = i == 0; // only show download progress for first instance
            let te = TextEmbedding::try_new(
                InitOptions::new(model.clone()).with_show_download_progress(show_progress),
            )
            .map_err(|e| EncoderError::Inference(e.to_string()))?;
            encoders.push(Mutex::new(te));
        }

        Ok(Self { encoders, dim })
    }

    /// Download a model from HuggingFace Hub and load it.
    ///
    /// Downloads `model.onnx`, `tokenizer.json`, `config.json`,
    /// `special_tokens_map.json`, and `tokenizer_config.json` into the
    /// hf-hub cache directory, then delegates to [`new_from_local_path`].
    fn new_from_hub(repo_id: &str, pool_size: usize) -> Result<Self, EncoderError> {
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

        Self::new_from_local_path(dir.to_str().unwrap_or(repo_id), pool_size)
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
    fn new_from_local_path(path: &str, pool_size: usize) -> Result<Self, EncoderError> {
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
                InitOptionsUserDefined::new(),
            )
            .map_err(|e| EncoderError::Inference(e.to_string()))?;
            encoders.push(Mutex::new(te));
        }

        Ok(Self { encoders, dim })
    }

    /// ONNX batch size for fastembed. Smaller batches improve CPU cache
    /// locality on Apple Silicon and similar architectures. Benchmarked at
    /// ~9% faster than fastembed's default of 256 on M3.
    const ENCODE_BATCH_SIZE: usize = 64;

    /// Encode texts synchronously. Acquires the first available encoder slot.
    ///
    /// Tries each slot via `try_lock` (round-robin). If all busy, blocks on
    /// slot 0.
    pub fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        let _span = info_span!("onnx_encode", n = texts.len()).entered();
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Convert &[&str] to Vec<String> as required by fastembed
        let text_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Try round-robin
        for encoder in &self.encoders {
            if let Ok(mut guard) = encoder.try_lock() {
                return guard
                    .embed(text_vec, Some(Self::ENCODE_BATCH_SIZE))
                    .map_err(|e| EncoderError::Inference(e.to_string()));
            }
        }

        // All busy — block on slot 0
        let mut guard = self.encoders[0]
            .lock()
            .map_err(|e| EncoderError::Inference(format!("mutex poisoned: {}", e)))?;
        guard
            .embed(text_vec, Some(Self::ENCODE_BATCH_SIZE))
            .map_err(|e| EncoderError::Inference(e.to_string()))
    }

    /// Encode a single text, returning a single vector.
    pub fn encode_one(&self, text: &str) -> Result<Vec<f32>, EncoderError> {
        let results = self.encode(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EncoderError::Inference("empty result from encoder".into()))
    }

    /// Returns the embedding dimension for this pool's model.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of encoder instances in the pool.
    pub fn pool_size(&self) -> usize {
        self.encoders.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_pool_and_encode() {
        let pool = EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("failed to create pool");
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
        let pool = EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("failed to create pool");
        let vec = pool.encode_one("hello world").expect("encode_one failed");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn self_similarity_high() {
        let pool = EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("failed to create pool");
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
        let pool = EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("failed to create pool");
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
        let pool = EncoderPool::new("all-MiniLM-L6-v2", 1, false).expect("failed to create pool");
        let vecs = pool.encode(&[]).expect("encode empty failed");
        assert!(vecs.is_empty());
    }

    #[test]
    fn invalid_model() {
        let err = EncoderPool::new("nonexistent-model-xyz", 1, false).unwrap_err();
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
        let err = EncoderPool::new(&dir.path().to_string_lossy(), 1, false).unwrap_err();
        assert!(
            matches!(err, EncoderError::ModelNotFound { .. }),
            "expected ModelNotFound, got {:?}",
            err
        );
    }
}
