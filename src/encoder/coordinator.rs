//! Encoding coordinator: batches concurrent encoding requests for throughput.
//!
//! The coordinator sits between request handlers and the underlying
//! `Encoder` (local ONNX pool or remote subprocess). Instead of each request
//! independently acquiring an encoder slot and running a single-text call,
//! the coordinator collects requests that arrive within a configurable time
//! window (`batch_wait`) and dispatches them as a single batched `encode()`
//! call. ONNX transformers are batch-efficient: encoding N texts in one call
//! is significantly faster than N sequential single-text calls because
//! self-attention computation is shared. Remote encoders benefit too, since
//! one subprocess round-trip carries many texts.
//!
//! ## Architecture
//!
//! ```text
//!   caller 1 ─► submit(texts) ──► crossbeam ──► background thread ──► Encoder.encode(batch)
//!   caller 2 ─► submit(texts) ──┘                                      │
//!   caller 3 ─► submit(texts) ──┘                                      ▼
//!                     ◄── std::sync::mpsc ◄── scatter results back to each caller
//! ```
//!
//! Callers are fully synchronous — no tokio runtime context required. This
//! allows the coordinator to be called from rayon threads, spawn_blocking
//! threads, or any other synchronous context.
//!
//! ## When to use
//!
//! Enable via `performance.encoder_batch_wait_ms` in config. Only useful
//! for live mode under concurrency >= 4. Adds `batch_wait` latency to
//! each request but increases throughput by amortising ONNX overhead.
//! Sequential workloads (c=1) should leave this disabled (default: 0).

use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel as cbc;

use crate::encoder::Encoder;
use crate::error::EncoderError;

/// A batch of texts submitted by one caller, with a channel to return results.
struct EncodeRequest {
    /// Texts to encode.
    texts: Vec<String>,
    /// Channel to send results back to the caller.
    tx: std_mpsc::Sender<Result<Vec<Vec<f32>>, EncoderError>>,
}

/// Batching coordinator that wraps an `Encoder`.
///
/// Submit texts via [`encode_many`]; the coordinator collects them into
/// batches and dispatches to the underlying encoder (local ONNX pool or
/// remote subprocess encoder). All public methods are synchronous — safe
/// to call from any thread.
pub struct EncoderCoordinator {
    submit_tx: cbc::Sender<EncodeRequest>,
    /// Hold the thread handle so we can join on drop.
    _thread: Option<thread::JoinHandle<()>>,
}

impl std::fmt::Debug for EncoderCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderCoordinator")
            .field("active", &!self.submit_tx.is_empty())
            .finish()
    }
}

impl EncoderCoordinator {
    /// Create a new coordinator.
    ///
    /// Spawns a background OS thread that collects requests and dispatches
    /// batched encode calls. No tokio runtime required.
    ///
    /// - `encoder`: the underlying encoder (local pool or remote subprocess).
    /// - `batch_wait`: how long to collect requests after the first arrival
    ///   before dispatching. Typical values: 2–10ms.
    pub fn new(encoder: Arc<dyn Encoder>, batch_wait: Duration) -> Self {
        let (submit_tx, submit_rx) = cbc::unbounded();
        let handle = thread::Builder::new()
            .name("encoder-coordinator".into())
            .spawn(move || coordinator_loop(encoder, submit_rx, batch_wait))
            .expect("failed to spawn encoder coordinator thread");
        Self {
            submit_tx,
            _thread: Some(handle),
        }
    }

    /// Submit multiple texts for encoding. Blocks until the batch completes.
    ///
    /// The texts are batched with other concurrent callers' texts. Returns
    /// vectors in the same order as the input texts. The added latency is
    /// at most `batch_wait`.
    ///
    /// Safe to call from any thread (rayon, spawn_blocking, OS thread).
    pub fn encode_many(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EncoderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let (tx, rx) = std_mpsc::channel();
        self.submit_tx
            .send(EncodeRequest { texts, tx })
            .map_err(|_| EncoderError::Inference("coordinator channel closed".into()))?;
        rx.recv()
            .map_err(|_| EncoderError::Inference("coordinator dropped result".into()))?
    }
}

/// Background thread that collects requests and dispatches batches.
fn coordinator_loop(
    encoder: Arc<dyn Encoder>,
    rx: cbc::Receiver<EncodeRequest>,
    batch_wait: Duration,
) {
    loop {
        // Wait for the first request (blocks indefinitely until one arrives
        // or the channel closes).
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => return, // channel closed, coordinator shutting down
        };

        // Collect more requests for up to `batch_wait`, capped at
        // MAX_BATCH_REQUESTS to prevent unbounded accumulation. Very
        // large batches can OOM due to O(N²) transformer self-attention.
        const MAX_BATCH_REQUESTS: usize = 64;

        let mut batch = vec![first];
        let deadline = Instant::now() + batch_wait;

        loop {
            if batch.len() >= MAX_BATCH_REQUESTS {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(cbc::RecvTimeoutError::Timeout) => break,
                Err(cbc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, but process what we have first.
                    break;
                }
            }
        }

        // Flatten all texts from all requests into a single batch.
        let mut all_texts: Vec<String> = Vec::new();
        let mut boundaries: Vec<usize> = Vec::new(); // end index for each request
        for req in &batch {
            all_texts.extend_from_slice(&req.texts);
            boundaries.push(all_texts.len());
        }

        // Encode the full batch.
        let text_refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();
        let encode_result = encoder.encode(&text_refs);

        // Scatter results back to each caller.
        match encode_result {
            Ok(all_vecs) => {
                let mut start = 0;
                for (req, &end) in batch.into_iter().zip(boundaries.iter()) {
                    let vecs = all_vecs[start..end].to_vec();
                    let _ = req.tx.send(Ok(vecs));
                    start = end;
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for req in batch {
                    let _ = req.tx.send(Err(EncoderError::Inference(msg.clone())));
                }
            }
        }
    }
}
