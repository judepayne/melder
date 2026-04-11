//! `SubprocessEncoder` — remote encoding via a user-supplied subprocess.
//!
//! This module implements the Phase 1 `RemoteEncoder` described in
//! `remote_operation.md`. A `SubprocessEncoder` spawns one or more
//! long-lived subprocess slots at construction time, each one talking to
//! melder via a newline-delimited JSON envelope + binary trailer protocol
//! on stdin/stdout. See `docs/remote-encoder.md` for the user-facing
//! contract.
//!
//! **Phase 1 stub:** this module currently declares the type surface and
//! always errors from `new()`. The full protocol + lifecycle implementation
//! lands in Phase 2.

use std::time::Duration;

use crate::encoder::{EncodeResult, Encoder};
use crate::error::EncoderError;

/// Remote encoder: a pool of long-lived subprocess slots, each owning one
/// user-supplied script speaking the melder remote-encoder protocol.
#[derive(Debug)]
#[allow(dead_code)] // Fields wired up in Phase 2 — stub for now.
pub struct SubprocessEncoder {
    command: String,
    pool_size: usize,
    encode_batch_size: usize,
    call_timeout: Duration,
    dim: usize,
    model_id: String,
}

impl SubprocessEncoder {
    /// Spawn the pool, handshake every slot, and return a ready encoder.
    ///
    /// Fails loudly if any slot cannot complete its handshake within the
    /// initial respawn cycle.
    pub fn new(
        command: String,
        pool_size: usize,
        encode_batch_size: usize,
        call_timeout: Duration,
    ) -> Result<Self, EncoderError> {
        // Phase 1 stub: reject until the real implementation lands.
        let _ = (command, pool_size, encode_batch_size, call_timeout);
        Err(EncoderError::RemoteSpawnFailed {
            command: String::new(),
            reason: "SubprocessEncoder is not yet implemented (Phase 2)".into(),
        })
    }

    /// Handshake-reported `model_id` from the first slot. Used for
    /// tracing and, downstream, for billing reconciliation.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl Encoder for SubprocessEncoder {
    fn encode(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        Err(EncoderError::Inference(
            "SubprocessEncoder is not yet implemented (Phase 2)".into(),
        ))
    }

    fn encode_detailed(&self, _texts: &[&str]) -> Result<Vec<EncodeResult>, EncoderError> {
        Err(EncoderError::Inference(
            "SubprocessEncoder is not yet implemented (Phase 2)".into(),
        ))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn pool_size(&self) -> usize {
        self.pool_size
    }

    fn encode_batch_size(&self) -> usize {
        self.encode_batch_size
    }
}
