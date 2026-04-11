//! `SubprocessEncoder` — remote encoding via a user-supplied subprocess.
//!
//! Implements the Phase 1 `RemoteEncoder` described in `remote_operation.md`.
//! A `SubprocessEncoder` spawns one or more long-lived subprocess slots at
//! construction time, each running the user-supplied command and talking to
//! melder via a newline-delimited JSON envelope + binary trailer protocol on
//! stdin/stdout. See `docs/remote-encoder.md` for the user-facing contract.
//!
//! ## Architecture
//!
//! ```text
//!   SubprocessEncoder
//!     ├── Arc<Mutex<Slot>>  ← one per pool slot
//!     │     └── subprocess + reader thread + stderr drain thread
//!     │     └── per-slot: respawn loop, health tracking
//!     └── encode()/encode_detailed(): try_lock round-robin
//! ```
//!
//! Thread budget for a pool of size N:
//!
//! - N reader threads (one per slot)
//! - N stderr drain threads (one per slot)
//! - 0 respawn threads — respawns happen inline inside the `Mutex<Slot>` guard
//!
//! All fully synchronous (stdlib `mpsc` + `std::process`) — no tokio runtime.
//!
//! ## Failure policy
//!
//! Phase 1 is fail-fast-loud:
//!
//! - Per-record errors → logged, then the batch fails with `EncoderError::BatchError`.
//! - Whole-batch errors → propagated immediately.
//! - Subprocess death / hang / protocol violation → slot killed, respawned
//!   (with exponential backoff) on the next call. Slot marked Unhealthy after
//!   5 consecutive failures within a 60s window.
//! - All slots unhealthy → `EncoderError::PoolExhausted`.

mod protocol;
mod slot;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tracing::{info, warn};

use self::slot::{Slot, SlotState};
use crate::encoder::{EncodeResult, Encoder};
use crate::error::EncoderError;

/// Remote encoder: a pool of long-lived subprocess slots.
pub struct SubprocessEncoder {
    slots: Vec<Arc<Mutex<Slot>>>,
    command: String,
    encode_batch_size: usize,
    call_timeout: Duration,
    dim: usize,
    model_id: String,
    /// Round-robin counter for fallback slot selection when all slots are
    /// busy via try_lock. Mirrors `EncoderPool::next_fallback`.
    next_fallback: AtomicUsize,
}

impl std::fmt::Debug for SubprocessEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubprocessEncoder")
            .field("command", &self.command)
            .field("pool_size", &self.slots.len())
            .field("dim", &self.dim)
            .field("model_id", &self.model_id)
            .field("call_timeout_ms", &self.call_timeout.as_millis())
            .finish()
    }
}

impl SubprocessEncoder {
    /// Spawn the pool and complete every slot's handshake before returning.
    ///
    /// Fails loudly if any slot cannot handshake within the hardcoded 30s
    /// window. Surviving slots are cleanly shut down before the error
    /// propagates. Startup is sequential for slot 0 (so we can learn the
    /// vector dim from its handshake), then parallel for the rest.
    pub fn new(
        command: String,
        pool_size: usize,
        encode_batch_size: usize,
        call_timeout: Duration,
    ) -> Result<Self, EncoderError> {
        if pool_size == 0 {
            return Err(EncoderError::RemoteSpawnFailed {
                command,
                reason: "pool_size must be >= 1".into(),
            });
        }

        info!(
            command = %command,
            pool_size,
            call_timeout_ms = call_timeout.as_millis() as u64,
            "spawning remote encoder pool"
        );

        // Slot 0 goes first so we can capture the dim and validate the
        // rest of the pool against it.
        let slot0 = Slot::spawn(0, &command, None).map_err(|e| {
            // Map the handshake failure to RemoteSpawnFailed at startup
            // so the CLI can show operators a single clear error message.
            EncoderError::RemoteSpawnFailed {
                command: command.clone(),
                reason: format!("slot 0: {e}"),
            }
        })?;
        let dim = slot0.vector_dim;
        let model_id = slot0.model_id.clone();

        let mut slots: Vec<Arc<Mutex<Slot>>> = Vec::with_capacity(pool_size);
        slots.push(Arc::new(Mutex::new(slot0)));

        if pool_size > 1 {
            // Parallel spawn of the remaining slots via scoped threads.
            // Any failure tears the whole pool down.
            let command_ref = command.as_str();
            let results: Vec<Result<Slot, EncoderError>> = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(pool_size - 1);
                for i in 1..pool_size {
                    handles.push(scope.spawn(move || Slot::spawn(i, command_ref, Some(dim))));
                }
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(EncoderError::RemoteSpawnFailed {
                                command: String::new(),
                                reason: "slot spawn thread panicked".into(),
                            })
                        })
                    })
                    .collect()
            });

            for res in results {
                match res {
                    Ok(s) => slots.push(Arc::new(Mutex::new(s))),
                    Err(e) => {
                        // Tear down already-spawned slots.
                        for s in &slots {
                            if let Ok(mut guard) = s.lock() {
                                guard.shutdown_graceful();
                            }
                        }
                        return Err(EncoderError::RemoteSpawnFailed {
                            command: command.clone(),
                            reason: format!("slot spawn failed: {e}"),
                        });
                    }
                }
            }
        }

        info!(
            pool_size,
            dim,
            model_id = %model_id,
            "remote encoder pool ready"
        );

        Ok(SubprocessEncoder {
            slots,
            command,
            encode_batch_size,
            call_timeout,
            dim,
            model_id,
            next_fallback: AtomicUsize::new(0),
        })
    }

    /// Handshake-reported `model_id` from slot 0. Useful for tracing and
    /// billing reconciliation against metered remote services.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Try-lock round-robin slot acquisition; mirrors `EncoderPool::encode`.
    /// Blocks on a fallback slot if all try_lock attempts fail.
    fn with_healthy_slot<F, T>(&self, mut f: F) -> Result<T, EncoderError>
    where
        F: FnMut(&mut Slot) -> Result<T, EncoderError>,
    {
        // Fast path: try_lock every slot in order, skipping unhealthy ones.
        for s in &self.slots {
            if let Ok(mut guard) = s.try_lock() {
                if guard.state == SlotState::Unhealthy {
                    continue;
                }
                return self.run_with_slot(&mut guard, &mut f);
            }
        }

        // All slots busy — fall back to a blocking round-robin lock.
        let n = self.slots.len();
        for _ in 0..n {
            let slot_idx = self.next_fallback.fetch_add(1, Ordering::Relaxed) % n;
            let mut guard = self.slots[slot_idx]
                .lock()
                .map_err(|e| EncoderError::Inference(format!("slot mutex poisoned: {e}")))?;
            if guard.state == SlotState::Unhealthy {
                continue;
            }
            return self.run_with_slot(&mut guard, &mut f);
        }

        Err(EncoderError::PoolExhausted {
            pool_size: self.slots.len(),
        })
    }

    /// Run `f` against `slot`, respawning once on subprocess-lifecycle
    /// failures (crash, hang, protocol violation). Batch errors are
    /// returned verbatim without a respawn attempt.
    fn run_with_slot<F, T>(&self, slot: &mut Slot, f: &mut F) -> Result<T, EncoderError>
    where
        F: FnMut(&mut Slot) -> Result<T, EncoderError>,
    {
        match f(slot) {
            Ok(t) => Ok(t),
            Err(err) => {
                // Check if the error is something we should try to recover from.
                let should_respawn = matches!(
                    err,
                    EncoderError::SubprocessDied { .. }
                        | EncoderError::Timeout { .. }
                        | EncoderError::ProtocolViolation { .. }
                );
                if !should_respawn {
                    return Err(err);
                }
                if slot.state == SlotState::Unhealthy {
                    return Err(err);
                }
                // Try one respawn. If that succeeds, retry the call once.
                match slot.try_respawn(self.dim) {
                    Ok(()) => match f(slot) {
                        Ok(t) => Ok(t),
                        Err(e) => Err(e),
                    },
                    Err(respawn_err) => {
                        warn!(
                            slot = slot.index,
                            error = %respawn_err,
                            "slot respawn failed; propagating original error"
                        );
                        Err(err)
                    }
                }
            }
        }
    }
}

impl Encoder for SubprocessEncoder {
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // Collapse per-record errors into a whole-batch error with
        // structured tracing per failed record — Phase 1 fail-fast policy.
        let detailed = self.encode_detailed(texts)?;
        let mut out = Vec::with_capacity(detailed.len());
        for (i, r) in detailed.into_iter().enumerate() {
            match r {
                EncodeResult::Vector(v) => out.push(v),
                EncodeResult::Error(msg) => {
                    let text_prefix: String = texts[i].chars().take(200).collect();
                    tracing::error!(
                        record_index = i,
                        text = %text_prefix,
                        error = %msg,
                        model_id = %self.model_id,
                        command = %self.command,
                        "remote encoder per-record error (fail-fast)"
                    );
                    return Err(EncoderError::BatchError {
                        slot: 0,
                        message: format!("per-record error at index {i}: {msg}"),
                    });
                }
            }
        }
        Ok(out)
    }

    fn encode_detailed(&self, texts: &[&str]) -> Result<Vec<EncodeResult>, EncoderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.with_healthy_slot(|slot| slot.encode(texts, self.call_timeout))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn pool_size(&self) -> usize {
        self.slots.len()
    }

    fn encode_batch_size(&self) -> usize {
        self.encode_batch_size
    }
}

impl Drop for SubprocessEncoder {
    fn drop(&mut self) {
        // Shut down all slots in parallel.
        std::thread::scope(|scope| {
            for s in &self.slots {
                let s = Arc::clone(s);
                scope.spawn(move || {
                    if let Ok(mut guard) = s.lock() {
                        guard.shutdown_graceful();
                    }
                });
            }
        });
    }
}
