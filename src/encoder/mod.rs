//! Encoder pool for embedding inference via `fastembed`.
//!
//! Wraps one or more `TextEmbedding` instances behind `Mutex` locks.
//! Each instance is an independent ONNX session with its own tokenizer
//! state. Model weights are memory-mapped by the OS, so multiple instances
//! share the same physical pages.

pub mod coordinator;

use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::EncoderError;

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

/// Dimension of the embedding vector for a given model.
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
    /// `model_name` is resolved to a `fastembed::EmbeddingModel` enum variant.
    /// `pool_size` instances of `TextEmbedding` are created (first run may
    /// download the model, ~90MB for fp32 or ~23MB for quantised).
    /// Set `quantized` to use the INT8 quantised model variant (~2x faster,
    /// negligible quality loss).
    pub fn new(model_name: &str, pool_size: usize, quantized: bool) -> Result<Self, EncoderError> {
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

    /// Encode texts synchronously. Acquires the first available encoder slot.
    ///
    /// Tries each slot via `try_lock` (round-robin). If all busy, blocks on
    /// slot 0.
    pub fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Convert &[&str] to Vec<String> as required by fastembed
        let text_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Try round-robin
        for encoder in &self.encoders {
            if let Ok(mut guard) = encoder.try_lock() {
                return guard
                    .embed(text_vec, None)
                    .map_err(|e| EncoderError::Inference(e.to_string()));
            }
        }

        // All busy — block on slot 0
        let mut guard = self.encoders[0]
            .lock()
            .map_err(|e| EncoderError::Inference(format!("mutex poisoned: {}", e)))?;
        guard
            .embed(text_vec, None)
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
}
