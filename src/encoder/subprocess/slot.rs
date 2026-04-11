//! One slot in the `SubprocessEncoder` pool: a long-lived subprocess plus
//! the per-slot thread infrastructure needed to talk to it synchronously.
//!
//! Each slot owns:
//! - a spawned child process (stdin/stdout/stderr piped),
//! - one reader thread that parses response frames from stdout,
//! - one stderr-drain thread that forwards log lines to `tracing`,
//! - an mpsc receiver the encode path blocks on with `recv_timeout`.
//!
//! Respawn and health tracking happen in-band inside `Slot::encode` —
//! there is no dedicated respawn worker.

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{debug, info, info_span, warn};

use super::protocol::{self, Frame, HandshakeInfo, ProtocolError};
use crate::encoder::EncodeResult;
use crate::error::EncoderError;

/// Maximum handshake wait. Hardcoded for Phase 1 per the design doc.
pub(crate) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// Respawn backoff cap.
const BACKOFF_CAP_SECS: u64 = 60;

/// Slot-unhealthy threshold: N consecutive failures within this window → unhealthy.
pub(crate) const SLOT_UNHEALTHY_FAILURES: u32 = 5;
pub(crate) const SLOT_UNHEALTHY_WINDOW: Duration = Duration::from_secs(60);

/// Shutdown grace periods.
const SHUTDOWN_GRACE_BEFORE_SIGTERM: Duration = Duration::from_secs(5);
const SHUTDOWN_GRACE_BEFORE_SIGKILL: Duration = Duration::from_secs(5);

/// Slot health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Healthy: can accept encode calls.
    Healthy,
    /// Permanently unhealthy: no more calls dispatched.
    Unhealthy,
}

/// One subprocess slot + its reader and stderr threads.
pub(crate) struct Slot {
    pub(crate) index: usize,
    command: String,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    /// Parsed frames come from the reader thread through this channel.
    frame_rx: Option<mpsc::Receiver<Result<Frame, ProtocolError>>>,
    reader_handle: Option<JoinHandle<()>>,
    stderr_handle: Option<JoinHandle<()>>,
    pub(crate) model_id: String,
    pub(crate) vector_dim: usize,
    #[allow(dead_code)] // Informational — logged at startup, not enforced.
    pub(crate) max_batch_size: Option<usize>,
    pub(crate) state: SlotState,
    /// Failures inside the slot-unhealthy window.
    consecutive_failures: u32,
    /// When the slot last became `Healthy`. Used to decay
    /// `consecutive_failures` after a stable runtime window.
    last_healthy_at: Instant,
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("index", &self.index)
            .field("state", &self.state)
            .field("model_id", &self.model_id)
            .field("vector_dim", &self.vector_dim)
            .field("consecutive_failures", &self.consecutive_failures)
            .finish()
    }
}

impl Slot {
    /// Spawn a new slot and perform the initial handshake.
    ///
    /// Returns a fully-initialised `Slot` in the `Healthy` state, or an
    /// error variant that callers map to the right `EncoderError`.
    ///
    /// If `expected_dim` is `Some`, the handshake must report a matching
    /// dim (used when we spawn slots 2..N after slot 1 has established the
    /// contract).
    pub(crate) fn spawn(
        index: usize,
        command: &str,
        expected_dim: Option<usize>,
    ) -> Result<Self, EncoderError> {
        let span = info_span!("remote_encoder_handshake", slot = index).entered();
        let handshake_start = Instant::now();

        let (mut child, stdin, stdout, stderr) =
            spawn_process(command).map_err(|e| EncoderError::HandshakeFailed {
                slot: index,
                reason: format!("spawn: {e}"),
            })?;

        // Spawn the stderr drain thread first — any errors during handshake
        // should already have their stderr context captured.
        let stderr_handle = spawn_stderr_drain(index, stderr);

        // Spawn the reader thread. It owns stdout and pushes parsed frames
        // onto `frame_rx`. For the handshake we read exactly one envelope
        // inline via a one-shot channel, then hand stdout to the reader.
        let (hs_tx, hs_rx) = mpsc::channel::<Result<HandshakeInfo, ProtocolError>>();
        let (frame_tx, frame_rx) = mpsc::channel::<Result<Frame, ProtocolError>>();
        let dim_for_reader = expected_dim;
        let reader_handle = thread::Builder::new()
            .name(format!("remote-encoder-reader-{index}"))
            .spawn(move || reader_loop(stdout, hs_tx, frame_tx, dim_for_reader))
            .expect("spawn reader thread");

        // Wait for the handshake with a hard timeout.
        let hs_result = match hs_rx.recv_timeout(HANDSHAKE_TIMEOUT) {
            Ok(Ok(info)) => info,
            Ok(Err(ProtocolError::Eof)) => {
                // Process died before handshake.
                let reason = "stdout closed before handshake".to_string();
                // Best-effort reap.
                let _ = child.kill();
                let _ = child.wait();
                drop(stdin);
                let _ = reader_handle.join();
                let _ = stderr_handle.join();
                return Err(EncoderError::HandshakeFailed {
                    slot: index,
                    reason,
                });
            }
            Ok(Err(e)) => {
                let reason = format!("{e}");
                let _ = child.kill();
                let _ = child.wait();
                drop(stdin);
                let _ = reader_handle.join();
                let _ = stderr_handle.join();
                return Err(EncoderError::HandshakeFailed {
                    slot: index,
                    reason,
                });
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                warn!(slot = index, "handshake timeout after 30s");
                let _ = child.kill();
                let _ = child.wait();
                drop(stdin);
                let _ = reader_handle.join();
                let _ = stderr_handle.join();
                return Err(EncoderError::HandshakeFailed {
                    slot: index,
                    reason: "handshake did not arrive within 30s".into(),
                });
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                let _ = child.wait();
                drop(stdin);
                let _ = reader_handle.join();
                let _ = stderr_handle.join();
                return Err(EncoderError::HandshakeFailed {
                    slot: index,
                    reason: "reader thread died before handshake".into(),
                });
            }
        };

        // Validate handshake.
        if hs_result.protocol_version != 1 {
            let reason = format!(
                "unsupported protocol_version {}; only v1 is supported",
                hs_result.protocol_version
            );
            let _ = child.kill();
            let _ = child.wait();
            return Err(EncoderError::HandshakeFailed {
                slot: index,
                reason,
            });
        }
        if hs_result.vector_dim == 0 {
            let _ = child.kill();
            let _ = child.wait();
            return Err(EncoderError::HandshakeFailed {
                slot: index,
                reason: "vector_dim must be > 0".into(),
            });
        }
        if hs_result.model_id.is_empty() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(EncoderError::HandshakeFailed {
                slot: index,
                reason: "model_id must be non-empty".into(),
            });
        }
        if let Some(expected) = expected_dim
            && expected != hs_result.vector_dim
        {
            let reason = format!(
                "vector_dim mismatch: slot 0 declared {expected}, this slot declared {}",
                hs_result.vector_dim
            );
            let _ = child.kill();
            let _ = child.wait();
            return Err(EncoderError::HandshakeFailed {
                slot: index,
                reason,
            });
        }

        info!(
            slot = index,
            vector_dim = hs_result.vector_dim,
            model_id = %hs_result.model_id,
            max_batch_size = ?hs_result.max_batch_size,
            latency_ms = handshake_start.elapsed().as_millis() as u64,
            "remote encoder slot ready"
        );
        drop(span);

        Ok(Slot {
            index,
            command: command.to_string(),
            child: Some(child),
            stdin: Some(stdin),
            frame_rx: Some(frame_rx),
            reader_handle: Some(reader_handle),
            stderr_handle: Some(stderr_handle),
            model_id: hs_result.model_id,
            vector_dim: hs_result.vector_dim,
            max_batch_size: hs_result.max_batch_size,
            state: SlotState::Healthy,
            consecutive_failures: 0,
            last_healthy_at: Instant::now(),
        })
    }

    /// Send an encode request and await the response with `call_timeout`.
    ///
    /// On success: returns the per-record `EncodeResult` list from Phase 1's
    /// fail-fast detailed path.
    ///
    /// On failure: the slot's state may transition — IO errors and timeouts
    /// kill and mark the slot (caller should respawn or promote to unhealthy);
    /// batch errors are final but leave the slot alive.
    pub(crate) fn encode(
        &mut self,
        texts: &[&str],
        call_timeout: Duration,
    ) -> Result<Vec<EncodeResult>, EncoderError> {
        let span = info_span!(
            "remote_encoder_call",
            slot = self.index,
            batch_size = texts.len()
        )
        .entered();
        let call_start = Instant::now();

        if self.state != SlotState::Healthy {
            return Err(EncoderError::SlotUnhealthy { slot: self.index });
        }

        // Write the request.
        let stdin = match self.stdin.as_mut() {
            Some(s) => s,
            None => {
                self.mark_failed();
                return Err(EncoderError::SubprocessDied {
                    slot: self.index,
                    reason: "stdin missing".into(),
                });
            }
        };
        if let Err(e) = protocol::write_encode_request(stdin, texts) {
            warn!(slot = self.index, error = %e, "stdin write failed");
            self.kill_and_reap(format!("stdin write: {e}"));
            return Err(EncoderError::SubprocessDied {
                slot: self.index,
                reason: format!("stdin write: {e}"),
            });
        }

        // Wait for the response with per-call timeout.
        let rx = match self.frame_rx.as_ref() {
            Some(rx) => rx,
            None => {
                self.mark_failed();
                return Err(EncoderError::SubprocessDied {
                    slot: self.index,
                    reason: "frame channel missing".into(),
                });
            }
        };
        let frame = match rx.recv_timeout(call_timeout) {
            Ok(Ok(f)) => f,
            Ok(Err(ProtocolError::Eof)) => {
                let reason = "stdout closed mid-call".to_string();
                self.kill_and_reap(reason.clone());
                return Err(EncoderError::SubprocessDied {
                    slot: self.index,
                    reason,
                });
            }
            Ok(Err(e)) => {
                let reason = format!("{e}");
                self.kill_and_reap(reason.clone());
                return Err(EncoderError::ProtocolViolation {
                    slot: self.index,
                    reason,
                });
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let elapsed_ms = call_start.elapsed().as_millis() as u64;
                warn!(
                    slot = self.index,
                    elapsed_ms, "remote encoder call timed out"
                );
                self.kill_and_reap(format!("call timeout after {elapsed_ms}ms"));
                return Err(EncoderError::Timeout {
                    slot: self.index,
                    elapsed_ms,
                });
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let reason = "reader thread disconnected".to_string();
                self.kill_and_reap(reason.clone());
                return Err(EncoderError::SubprocessDied {
                    slot: self.index,
                    reason,
                });
            }
        };

        // Convert the frame to the caller's EncodeResult list.
        match frame {
            Frame::EncodeResponse { results, vectors } => {
                let out: Vec<EncodeResult> = results
                    .into_iter()
                    .map(|per| match per {
                        super::protocol::PerRecord::Ok(idx) => {
                            EncodeResult::Vector(vectors[idx].clone())
                        }
                        super::protocol::PerRecord::Err(msg) => EncodeResult::Error(msg),
                    })
                    .collect();
                // Successful call: reset stability tracking.
                self.on_successful_call();
                drop(span);
                Ok(out)
            }
            Frame::BatchError(message) => {
                // Whole-batch errors are final but the slot stays alive —
                // the subprocess can handle further requests. Don't kill it.
                warn!(
                    slot = self.index,
                    message = %message,
                    "remote encoder batch error"
                );
                Err(EncoderError::BatchError {
                    slot: self.index,
                    message,
                })
            }
        }
    }

    /// Graceful shutdown: close stdin, wait, SIGTERM, wait, SIGKILL.
    pub(crate) fn shutdown_graceful(&mut self) {
        // Drop stdin → EOF → subprocess should exit cleanly.
        let _ = self.stdin.take();
        let start = Instant::now();
        if let Some(mut child) = self.child.take() {
            while start.elapsed() < SHUTDOWN_GRACE_BEFORE_SIGTERM {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        self.join_threads();
                        return;
                    }
                    Ok(None) => thread::sleep(Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            // SIGTERM via kill (std doesn't distinguish; on Unix it's SIGKILL,
            // on Windows TerminateProcess). Phase 1 documents the gap —
            // portable SIGTERM would require nix, and we promised no new deps.
            let _ = child.kill();
            let term_start = Instant::now();
            while term_start.elapsed() < SHUTDOWN_GRACE_BEFORE_SIGKILL {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => thread::sleep(Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            let _ = child.wait();
        }
        self.join_threads();
    }

    fn join_threads(&mut self) {
        if let Some(h) = self.reader_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.stderr_handle.take() {
            let _ = h.join();
        }
    }

    /// Kill the child, reap, join threads. Marks the slot as failed.
    pub(crate) fn kill_and_reap(&mut self, reason: String) {
        debug!(slot = self.index, reason = %reason, "killing slot");
        let _ = self.stdin.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.join_threads();
        self.mark_failed();
    }

    fn mark_failed(&mut self) {
        // Decay failure count if the slot was stable for long enough.
        if self.last_healthy_at.elapsed() > SLOT_UNHEALTHY_WINDOW {
            self.consecutive_failures = 0;
        }
        self.consecutive_failures += 1;
        if self.consecutive_failures >= SLOT_UNHEALTHY_FAILURES {
            self.state = SlotState::Unhealthy;
            info!(slot = self.index, "slot promoted to Unhealthy");
        }
    }

    fn on_successful_call(&mut self) {
        if self.last_healthy_at.elapsed() > SLOT_UNHEALTHY_WINDOW {
            self.consecutive_failures = 0;
        }
    }

    /// Attempt a single respawn on this slot. Used by the pool when a
    /// slot's subprocess dies mid-run. Exponential backoff is honoured.
    pub(crate) fn try_respawn(&mut self, expected_dim: usize) -> Result<(), EncoderError> {
        if self.state == SlotState::Unhealthy {
            return Err(EncoderError::SlotUnhealthy { slot: self.index });
        }

        let backoff = backoff_duration(self.consecutive_failures);
        info!(
            slot = self.index,
            backoff_ms = backoff.as_millis() as u64,
            attempt = self.consecutive_failures,
            "respawning slot"
        );
        thread::sleep(backoff);

        match Slot::spawn(self.index, &self.command, Some(expected_dim)) {
            Ok(new_slot) => {
                // Preserve this slot's health counters across the respawn
                // so the unhealthy promotion rule still fires correctly.
                let keep_failures = self.consecutive_failures;
                let keep_healthy_at = self.last_healthy_at;
                *self = new_slot;
                self.consecutive_failures = keep_failures;
                self.last_healthy_at = keep_healthy_at;
                Ok(())
            }
            Err(e) => {
                self.mark_failed();
                Err(e)
            }
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.child.is_some() {
            self.shutdown_graceful();
        }
    }
}

// ---------------------------------------------------------------------------
// Process spawning & reader threads
// ---------------------------------------------------------------------------

type Pipes = (
    Child,
    ChildStdin,
    std::process::ChildStdout,
    std::process::ChildStderr,
);

fn spawn_process(command: &str) -> std::io::Result<Pipes> {
    #[cfg(unix)]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    };

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("failed to capture stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("failed to capture stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("failed to capture stderr"))?;
    Ok((child, stdin, stdout, stderr))
}

/// Reader thread body. Reads the handshake once, then serves encode-response
/// frames for the rest of the slot's life.
fn reader_loop(
    stdout: std::process::ChildStdout,
    hs_tx: mpsc::Sender<Result<HandshakeInfo, ProtocolError>>,
    frame_tx: mpsc::Sender<Result<Frame, ProtocolError>>,
    expected_dim: Option<usize>,
) {
    let mut r = BufReader::new(stdout);

    // Step 1: read the handshake envelope.
    let hs = match protocol::read_envelope(&mut r) {
        Ok(env) => protocol::parse_handshake(&env),
        Err(e) => Err(e),
    };
    match hs {
        Ok(info) => {
            // If we haven't been told what dim to expect yet (first slot),
            // accept whatever the handshake declares.
            let dim_for_frames = expected_dim.unwrap_or(info.vector_dim);
            if hs_tx.send(Ok(info)).is_err() {
                return; // caller abandoned the slot
            }
            // Step 2: continuous encode-response loop.
            loop {
                match protocol::read_encode_response(&mut r, dim_for_frames) {
                    Ok(frame) => {
                        if frame_tx.send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let is_eof = matches!(e, ProtocolError::Eof);
                        let _ = frame_tx.send(Err(e));
                        if is_eof {
                            return;
                        }
                    }
                }
            }
        }
        Err(e) => {
            let _ = hs_tx.send(Err(e));
        }
    }
}

/// Stderr drain: line-buffered reader that forwards each line to tracing.
fn spawn_stderr_drain(index: usize, stderr: std::process::ChildStderr) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("remote-encoder-stderr-{index}"))
        .spawn(move || {
            let r = BufReader::new(stderr);
            for line in r.lines() {
                match line {
                    Ok(line) => {
                        debug!(slot = index, line = %line, "remote encoder stderr");
                    }
                    Err(_) => return,
                }
            }
        })
        .expect("spawn stderr drain thread")
}

/// Respawn backoff: 1s, 2s, 4s, …, capped at 60s. Mirrors
/// `src/hooks/writer.rs::backoff_duration`.
fn backoff_duration(consecutive_failures: u32) -> Duration {
    let secs = (1u64 << consecutive_failures.min(6)).min(BACKOFF_CAP_SECS);
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_progression() {
        assert_eq!(backoff_duration(0), Duration::from_secs(1));
        assert_eq!(backoff_duration(1), Duration::from_secs(2));
        assert_eq!(backoff_duration(2), Duration::from_secs(4));
        assert_eq!(backoff_duration(3), Duration::from_secs(8));
        assert_eq!(backoff_duration(4), Duration::from_secs(16));
        assert_eq!(backoff_duration(5), Duration::from_secs(32));
        assert_eq!(backoff_duration(6), Duration::from_secs(60));
        assert_eq!(backoff_duration(10), Duration::from_secs(60));
    }
}
