//! Encoding coordinator: batches concurrent encoding requests for throughput.
//!
//! The coordinator sits between HTTP handlers and the `EncoderPool`. Instead
//! of each request independently acquiring an encoder slot and running a
//! single-text ONNX call, the coordinator collects requests that arrive
//! within a configurable time window (`batch_wait`) and dispatches them as
//! a single batched `encode()` call. ONNX transformers are batch-efficient:
//! encoding N texts in one call is significantly faster than N sequential
//! single-text calls because self-attention computation is shared.
//!
//! ## Architecture
//!
//! ```text
//!   caller 1 ─► submit(text) ──► mpsc ──► background task ──► EncoderPool.encode(batch)
//!   caller 2 ─► submit(text) ──┘                              │
//!   caller 3 ─► submit(text) ──┘                              ▼
//!                  ◄── oneshot ◄── scatter results back to each caller
//! ```
//!
//! Each caller awaits a `tokio::sync::oneshot` receiver. The background
//! task collects requests for up to `batch_wait` after the first arrival,
//! then encodes them all in one `EncoderPool::encode()` call and sends
//! each result back via the caller's oneshot channel.
//!
//! ## When to use
//!
//! Enable via `performance.encoder_batch_wait_ms` in config. Only useful
//! for live mode under concurrency >= 4. Adds `batch_wait` latency to
//! each request but increases throughput by amortising ONNX overhead.
//! Sequential workloads (c=1) should leave this disabled (default: 0).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::encoder::EncoderPool;
use crate::error::EncoderError;

/// A single text submitted for encoding, with a channel to return the result.
struct EncodeRequest {
    text: String,
    tx: oneshot::Sender<Result<Vec<f32>, EncoderError>>,
}

/// Batching coordinator that wraps an `EncoderPool`.
///
/// Submit texts via [`encode_one`] or [`encode_many`]; the coordinator
/// collects them into batches and dispatches to the underlying pool.
pub struct EncoderCoordinator {
    submit_tx: mpsc::UnboundedSender<EncodeRequest>,
    // Hold the task handle so it lives as long as the coordinator.
    _task: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for EncoderCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderCoordinator")
            .field("active", &!self.submit_tx.is_closed())
            .finish()
    }
}

impl EncoderCoordinator {
    /// Create a new coordinator.
    ///
    /// - `encoder_pool`: the underlying pool of ONNX sessions.
    /// - `batch_wait`: how long to collect requests after the first arrival
    ///   before dispatching. Typical values: 2–10ms.
    pub fn new(encoder_pool: Arc<EncoderPool>, batch_wait: Duration) -> Self {
        let (submit_tx, submit_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(coordinator_loop(encoder_pool, submit_rx, batch_wait));
        Self {
            submit_tx,
            _task: task,
        }
    }

    /// Submit a single text for encoding.
    ///
    /// The text is batched with other concurrent requests. Returns the
    /// embedding vector once the batch completes. The added latency is
    /// at most `batch_wait` from the plan.
    pub async fn encode_one(&self, text: String) -> Result<Vec<f32>, EncoderError> {
        let (tx, rx) = oneshot::channel();
        self.submit_tx
            .send(EncodeRequest { text, tx })
            .map_err(|_| EncoderError::Inference("coordinator channel closed".into()))?;
        rx.await
            .map_err(|_| EncoderError::Inference("coordinator dropped result".into()))?
    }

    /// Submit multiple texts for encoding in a single batch.
    ///
    /// All texts are submitted to the coordinator and will be included in
    /// the same (or consecutive) batch. Returns vectors in input order.
    pub async fn encode_many(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EncoderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut receivers = Vec::with_capacity(texts.len());
        for text in texts {
            let (tx, rx) = oneshot::channel();
            self.submit_tx
                .send(EncodeRequest { text, tx })
                .map_err(|_| EncoderError::Inference("coordinator channel closed".into()))?;
            receivers.push(rx);
        }

        let mut results = Vec::with_capacity(receivers.len());
        for rx in receivers {
            let vec = rx
                .await
                .map_err(|_| EncoderError::Inference("coordinator dropped result".into()))??;
            results.push(vec);
        }
        Ok(results)
    }
}

/// Background task that collects requests and dispatches batches.
async fn coordinator_loop(
    encoder_pool: Arc<EncoderPool>,
    mut rx: mpsc::UnboundedReceiver<EncodeRequest>,
    batch_wait: Duration,
) {
    loop {
        // Wait for the first request (blocks indefinitely until one arrives
        // or the channel closes).
        let first = match rx.recv().await {
            Some(req) => req,
            None => return, // channel closed, coordinator shutting down
        };

        // Collect more requests for up to `batch_wait`.
        let mut batch = vec![first];
        let deadline = Instant::now() + batch_wait;

        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                Ok(None) => return, // channel closed
                Err(_) => break,    // timeout — dispatch what we have
            }
        }

        // Dispatch the batch on a blocking thread (ONNX is CPU-bound).
        let pool = Arc::clone(&encoder_pool);
        let batch_len = batch.len();

        // Move batch into the blocking closure, encode, scatter results.
        tokio::task::spawn_blocking(move || {
            let texts: Vec<&str> = batch.iter().map(|r| r.text.as_str()).collect();

            match pool.encode(&texts) {
                Ok(vecs) => {
                    debug_assert_eq!(vecs.len(), batch_len);
                    for (req, vec) in batch.into_iter().zip(vecs) {
                        let _ = req.tx.send(Ok(vec));
                    }
                }
                Err(e) => {
                    // Send the error to all callers. We can't clone EncoderError
                    // directly, so we create new instances with the same message.
                    let msg = e.to_string();
                    for req in batch {
                        let _ = req.tx.send(Err(EncoderError::Inference(msg.clone())));
                    }
                }
            }
        })
        .await
        .ok(); // If spawn_blocking panics, the oneshot senders are dropped
               // and callers get a "coordinator dropped result" error.
    }
}
