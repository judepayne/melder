//! Error types for melder.
//!
//! Uses `thiserror` for typed errors within modules. Application boundaries
//! (CLI main, HTTP handlers) use `anyhow` for ad-hoc context.

/// Top-level error enum. Each module has its own error type that converts
/// into this via `From` impls.
#[derive(Debug, thiserror::Error)]
pub enum MelderError {
    #[error("config error: {0}")]
    Config(#[from] ConfigError),

    #[error("data error: {0}")]
    Data(#[from] DataError),

    #[error("encoder error: {0}")]
    Encoder(#[from] EncoderError),

    #[error("index error: {0}")]
    Index(#[from] IndexError),

    #[error("crossmap error: {0}")]
    CrossMap(#[from] CrossMapError),

    #[error("session error: {0}")]
    Session(#[from] SessionError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing required field: {field}")]
    MissingField { field: String },

    #[error("invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    #[error("weights sum to {sum}, expected 1.0")]
    WeightSum { sum: f64 },

    #[error("parse error: {0}")]
    Parse(#[from] serde_yaml::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("file not found: {path}")]
    NotFound { path: String },

    #[error("missing id field '{field}' in {path}")]
    MissingIdField { field: String, path: String },

    #[error("duplicate id '{id}' in {path}")]
    DuplicateId { id: String, path: String },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// Stub error types for Phase 2+ modules. Variants added when those modules
// are implemented.

#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("model not found: {model}")]
    ModelNotFound { model: String },

    #[error("encoding failed: {0}")]
    Inference(String),

    #[error("pool exhausted (all {pool_size} encoders busy)")]
    PoolExhausted { pool_size: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum CrossMapError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("missing required field '{field}' in record")]
    MissingField { field: String },

    #[error("empty id in record")]
    EmptyId,

    #[error("encoder error: {0}")]
    Encoder(#[from] EncoderError),
}
