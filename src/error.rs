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

    #[error("store error: {0}")]
    Store(#[from] StoreError),

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

// Module-level error types for encoder, index, crossmap, and session.

#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("model not found: {model}")]
    ModelNotFound { model: String },

    #[error("encoding failed: {0}")]
    Inference(String),

    #[error("pool exhausted (all {pool_size} encoders busy)")]
    PoolExhausted { pool_size: usize },

    // --- Remote encoder variants (SubprocessEncoder) ---
    /// Subprocess failed to emit a valid handshake within the 30s window,
    /// or emitted an invalid one (bad version, zero dim, empty model_id).
    #[error("remote encoder handshake failed (slot {slot}): {reason}")]
    HandshakeFailed { slot: usize, reason: String },

    /// A call exceeded `encoder_call_timeout_ms`. The slot has been killed
    /// and will be respawned.
    #[error("remote encoder call timed out after {elapsed_ms}ms (slot {slot})")]
    Timeout { slot: usize, elapsed_ms: u64 },

    /// Subprocess died mid-call (crash, OOM, non-zero exit, broken pipe).
    #[error("remote encoder subprocess died (slot {slot}): {reason}")]
    SubprocessDied { slot: usize, reason: String },

    /// Subprocess produced output that did not parse as a valid protocol
    /// message (malformed JSON, truncated trailer, unknown message type).
    #[error("remote encoder protocol violation (slot {slot}): {reason}")]
    ProtocolViolation { slot: usize, reason: String },

    /// Subprocess explicitly reported a whole-batch error (rate limit after
    /// internal retries, auth failure, etc.). Final — melder does not retry.
    #[error("remote encoder batch error (slot {slot}): {message}")]
    BatchError { slot: usize, message: String },

    /// Slot crossed the consecutive-failure threshold and is no longer
    /// dispatched to.
    #[error("remote encoder slot {slot} marked unhealthy")]
    SlotUnhealthy { slot: usize },

    /// Remote encoder spawn failed at startup — every slot failed to
    /// handshake within the initial respawn cycle. Fail-loud: melder
    /// will not start with a degraded remote encoder pool.
    #[error("remote encoder spawn failed ({command}): {reason}")]
    RemoteSpawnFailed { command: String, reason: String },
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
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(String),
}

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        StoreError::Sqlite(e.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("missing required field '{field}' in record")]
    MissingField { field: String },

    #[error("empty id in record")]
    EmptyId,

    #[error("not found: {message}")]
    NotFound { message: String },

    #[error("batch validation error: {message}")]
    BatchValidation { message: String },

    #[error("encoder error: {0}")]
    Encoder(#[from] EncoderError),

    #[error("store error: {0}")]
    Store(#[from] StoreError),

    #[error("WAL write failed: {0}")]
    Wal(#[from] std::io::Error),
}

impl SessionError {
    /// Suggested HTTP status code for this error variant.
    pub fn status_code(&self) -> u16 {
        match self {
            SessionError::MissingField { .. } | SessionError::EmptyId => 400,
            SessionError::NotFound { .. } => 404,
            SessionError::BatchValidation { .. } => 422,
            SessionError::Encoder(_) | SessionError::Store(_) | SessionError::Wal(_) => 500,
        }
    }
}
