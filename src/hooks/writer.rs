//! Hook writer task — manages the long-running hook subprocess.
//!
//! Receives `HookEvent` values from an mpsc channel, serializes them to
//! JSON, and writes them to the subprocess stdin pipe. Handles process
//! death with exponential backoff and automatic respawn.

use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::HookEvent;

/// Maximum consecutive failures before hooks are disabled.
const MAX_FAILURES: u32 = 5;

/// Process must be alive this long (seconds) to reset the failure counter.
const STABLE_THRESHOLD_SECS: u64 = 60;

/// Maximum backoff between respawn attempts (seconds).
const MAX_BACKOFF_SECS: u64 = 60;

// ---------------------------------------------------------------------------
// Subprocess spawning
// ---------------------------------------------------------------------------

/// Spawn the hook subprocess with stdin piped.
fn spawn_hook_process(command: &str) -> Result<(Child, ChildStdin), String> {
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
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let mut child = cmd.spawn().map_err(|e| format!("{}", e))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to capture subprocess stdin".to_string())?;

    Ok((child, stdin))
}

// ---------------------------------------------------------------------------
// Writer task
// ---------------------------------------------------------------------------

/// Run the hook writer loop. This is the entry point spawned by `tokio::spawn`.
///
/// Reads events from `rx`, serializes them, and writes to the hook process
/// stdin. Respawns the process on failure with exponential backoff. Shuts
/// down cleanly when the channel is closed (sender dropped).
pub async fn run(command: String, mut rx: mpsc::Receiver<HookEvent>) {
    let mut consecutive_failures: u32 = 0;
    let mut hook_disabled = false;

    'outer: loop {
        if hook_disabled {
            // Drain remaining events silently.
            while rx.recv().await.is_some() {}
            return;
        }

        // Spawn the subprocess.
        let (mut child, mut stdin) = match spawn_hook_process(&command) {
            Ok(pair) => {
                info!(command = %command, "hook process started");
                pair
            }
            Err(e) => {
                consecutive_failures += 1;
                warn!(
                    error = %e,
                    consecutive_failures,
                    "hook process failed to start"
                );
                if consecutive_failures >= MAX_FAILURES {
                    warn!("hook disabled after repeated failures");
                    hook_disabled = true;
                    continue 'outer;
                }
                let backoff = backoff_duration(consecutive_failures);
                tokio::time::sleep(backoff).await;
                continue 'outer;
            }
        };

        let spawn_time = Instant::now();

        // Event loop: read from channel, write to pipe.
        loop {
            let event = match rx.recv().await {
                Some(ev) => ev,
                None => {
                    // Channel closed — server shutting down.
                    break 'outer;
                }
            };

            let json_line = event.to_json_line();
            if let Err(e) = stdin.write_all(json_line.as_bytes()).await {
                warn!(error = %e, "hook pipe write failed");
                // Process is dead or pipe broken — break inner loop to respawn.
                break;
            }
            // Flush to ensure the subprocess sees the event promptly.
            if let Err(e) = stdin.flush().await {
                warn!(error = %e, "hook pipe flush failed");
                break;
            }

            // Reset failure counter if process has been stable.
            if consecutive_failures > 0
                && spawn_time.elapsed() > Duration::from_secs(STABLE_THRESHOLD_SECS)
            {
                consecutive_failures = 0;
            }
        }

        // Inner loop broke — process died or pipe failed.
        drop(stdin);

        // Reap the child process.
        match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
            Ok(Ok(status)) => {
                consecutive_failures += 1;
                warn!(
                    exit_code = status.code(),
                    consecutive_failures, "hook process exited"
                );
            }
            Ok(Err(e)) => {
                consecutive_failures += 1;
                warn!(error = %e, consecutive_failures, "hook process wait failed");
            }
            Err(_) => {
                // Timed out waiting — kill it.
                let _ = child.kill().await;
                consecutive_failures += 1;
                warn!(consecutive_failures, "hook process killed after timeout");
            }
        }

        if consecutive_failures >= MAX_FAILURES {
            warn!("hook disabled after repeated failures");
            hook_disabled = true;
            continue 'outer;
        }

        let backoff = backoff_duration(consecutive_failures);
        info!(
            backoff_ms = backoff.as_millis() as u64,
            "respawning hook process"
        );
        tokio::time::sleep(backoff).await;
    }

    // Shutdown: close stdin and wait for process to exit.
    // (stdin already dropped if we broke out of inner loop; this handles
    // the clean channel-closed path.)
    info!("hook writer shutting down");
}

/// Exponential backoff: 1s, 2s, 4s, 8s, ..., capped at MAX_BACKOFF_SECS.
fn backoff_duration(consecutive_failures: u32) -> Duration {
    let secs = (1u64 << consecutive_failures.min(6)).min(MAX_BACKOFF_SECS);
    Duration::from_secs(secs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        assert_eq!(backoff_duration(6), Duration::from_secs(60)); // capped
        assert_eq!(backoff_duration(10), Duration::from_secs(60)); // still capped
    }

    #[tokio::test]
    async fn channel_full_drops_event() {
        let (tx, _rx) = mpsc::channel(1);
        // Fill the channel
        tx.try_send(HookEvent::Break {
            a_id: "A".into(),
            b_id: "B".into(),
        })
        .unwrap();
        // Second send should fail (channel full)
        let result = tx.try_send(HookEvent::Break {
            a_id: "A2".into(),
            b_id: "B2".into(),
        });
        assert!(result.is_err(), "expected channel full error");
    }
}
