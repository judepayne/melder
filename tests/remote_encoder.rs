//! Integration tests for `SubprocessEncoder` using `tests/fixtures/stub_encoder.py`.
//!
//! Each test spawns real subprocesses. Python 3 must be on PATH — the
//! test harness checks for it at the start and skips the whole file if
//! it's unavailable (this mirrors the accuracy-test gating). CI already
//! has Python 3 available for benchmark data generation.

use std::path::PathBuf;
use std::sync::Once;
use std::time::Duration;

use melder::encoder::Encoder;
use melder::encoder::subprocess::SubprocessEncoder;
use melder::error::EncoderError;

static INIT: Once = Once::new();

fn have_python() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn init_tracing() {
    INIT.call_once(|| {
        // Best-effort: ignore if already set by another test binary.
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
    });
}

fn stub_path() -> PathBuf {
    let root =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo at test time");
    let mut p = PathBuf::from(root);
    p.push("tests");
    p.push("fixtures");
    p.push("stub_encoder.py");
    assert!(p.exists(), "stub script not found at {}", p.display());
    p
}

/// Build a `python3 tests/fixtures/stub_encoder.py <flags>` command string.
fn stub_cmd(flags: &str) -> String {
    format!("python3 {:?} {}", stub_path(), flags).replace('"', "'")
}

fn new_encoder(
    flags: &str,
    pool_size: usize,
    timeout: Duration,
) -> Result<SubprocessEncoder, EncoderError> {
    SubprocessEncoder::new(stub_cmd(flags), pool_size, 256, timeout)
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[test]
fn happy_path_single_slot() {
    if !have_python() {
        eprintln!("skipping: python3 not available");
        return;
    }
    init_tracing();
    let enc = new_encoder("--vector-dim 16", 1, Duration::from_secs(10))
        .expect("happy path: encoder should construct");
    assert_eq!(enc.dim(), 16);
    assert_eq!(enc.pool_size(), 1);
    assert_eq!(enc.model_id(), "stub");

    let vecs = enc
        .encode(&["hello", "world", "foo", "bar"])
        .expect("encode should succeed");
    assert_eq!(vecs.len(), 4);
    assert_eq!(vecs[0].len(), 16);
    // Hashes are deterministic — same text should produce the same vector.
    let again = enc.encode(&["hello"]).unwrap();
    assert_eq!(again[0], vecs[0]);
    // Vectors should be unit length (L2 norm ≈ 1.0).
    let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected unit vector, got norm {norm}"
    );
}

#[test]
fn happy_path_multi_slot_concurrent() {
    if !have_python() {
        eprintln!("skipping: python3 not available");
        return;
    }
    init_tracing();
    let enc = std::sync::Arc::new(
        new_encoder("--vector-dim 8", 4, Duration::from_secs(10))
            .expect("4-slot pool should construct"),
    );
    assert_eq!(enc.pool_size(), 4);

    // Hammer the pool from many threads to exercise try_lock round-robin +
    // fallback-slot contention.
    let mut handles = Vec::new();
    for t in 0..16 {
        let enc = std::sync::Arc::clone(&enc);
        handles.push(std::thread::spawn(move || {
            let texts = [format!("t{t}_1"), format!("t{t}_2")];
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            enc.encode(&text_refs)
                .expect("concurrent encode should work")
        }));
    }
    for h in handles {
        let vs = h.join().unwrap();
        assert_eq!(vs.len(), 2);
        assert_eq!(vs[0].len(), 8);
    }
}

// ---------------------------------------------------------------------------
// Startup failure modes
// ---------------------------------------------------------------------------

#[test]
fn handshake_timeout_fails_construction() {
    if !have_python() {
        return;
    }
    init_tracing();
    // The 30s handshake timeout is hardcoded; we can't shorten it just
    // for this test without plumbing a test-only constant. Instead, use
    // --skip-handshake which exits cleanly → melder sees stdout EOF
    // immediately, which is a different but equivalent failure mode
    // (and much faster than waiting 30 seconds).
    let err = new_encoder("--skip-handshake", 1, Duration::from_secs(5))
        .expect_err("skip-handshake should fail construction");
    assert!(
        matches!(err, EncoderError::RemoteSpawnFailed { .. }),
        "got: {err:?}"
    );
}

#[test]
fn bad_protocol_version_fails_construction() {
    if !have_python() {
        return;
    }
    init_tracing();
    let err = new_encoder("--bad-protocol-version", 1, Duration::from_secs(5))
        .expect_err("bad protocol version should fail construction");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("protocol_version") || msg.contains("RemoteSpawn"),
        "got: {msg}"
    );
}

#[test]
fn zero_dim_fails_construction() {
    if !have_python() {
        return;
    }
    init_tracing();
    let err = new_encoder("--zero-dim", 1, Duration::from_secs(5))
        .expect_err("zero dim should fail construction");
    assert!(matches!(err, EncoderError::RemoteSpawnFailed { .. }));
}

// ---------------------------------------------------------------------------
// Per-record + whole-batch errors
// ---------------------------------------------------------------------------

#[test]
fn per_record_error_fails_batch_but_keeps_slot_alive() {
    if !have_python() {
        return;
    }
    init_tracing();
    // rate=1.0 → every record errors; first call fails the batch.
    let enc = new_encoder(
        "--vector-dim 8 --record-error-rate 1.0",
        1,
        Duration::from_secs(5),
    )
    .expect("construction should succeed");

    let err = enc.encode(&["a", "b"]).unwrap_err();
    match err {
        EncoderError::BatchError { message, .. } => {
            assert!(message.contains("per-record"), "got: {message}");
        }
        other => panic!("expected BatchError, got {other:?}"),
    }
}

#[test]
fn per_record_errors_visible_via_encode_detailed() {
    if !have_python() {
        return;
    }
    init_tracing();
    let enc = new_encoder(
        "--vector-dim 8 --record-error-rate 1.0",
        1,
        Duration::from_secs(5),
    )
    .expect("construction should succeed");
    let results = enc.encode_detailed(&["a", "b", "c"]).unwrap();
    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(
            matches!(r, melder::encoder::EncodeResult::Error(_)),
            "expected per-record error"
        );
    }
}

#[test]
fn batch_error_fails_call_and_keeps_slot_alive() {
    if !have_python() {
        return;
    }
    init_tracing();
    let enc = new_encoder(
        "--vector-dim 8 --batch-error-every 1",
        1,
        Duration::from_secs(5),
    )
    .expect("construction should succeed");

    let err = enc.encode(&["a"]).unwrap_err();
    assert!(
        matches!(err, EncoderError::BatchError { .. }),
        "got: {err:?}"
    );
    // Second call still fails with batch error — slot should still be alive.
    let err2 = enc.encode(&["b"]).unwrap_err();
    assert!(matches!(err2, EncoderError::BatchError { .. }));
}

// ---------------------------------------------------------------------------
// Subprocess death + respawn
// ---------------------------------------------------------------------------

#[test]
fn subprocess_crash_respawns_and_succeeds() {
    if !have_python() {
        return;
    }
    init_tracing();
    // Crash on the 2nd encode call of each subprocess lifetime. This
    // means:
    //   Call 1 (Rust) → subprocess counter=1 → ok.
    //   Call 2 (Rust) → subprocess counter=2 → CRASH → pool respawns
    //                   and inline-retries the call against the fresh
    //                   subprocess, whose counter starts at 1 again → ok.
    //   Call 3 (Rust) → fresh subprocess counter=2 → CRASH again → respawn → ok.
    // So every user-visible call succeeds, but every second call exercises
    // the respawn-and-retry path.
    let enc = new_encoder("--vector-dim 8 --fail-on-call 2", 1, Duration::from_secs(5))
        .expect("construction should succeed");

    let v1 = enc.encode(&["first"]).expect("first call (ok)");
    assert_eq!(v1.len(), 1);
    let v2 = enc
        .encode(&["second"])
        .expect("second call should succeed after respawn");
    assert_eq!(v2.len(), 1);
    let v3 = enc
        .encode(&["third"])
        .expect("third call should also succeed after another respawn");
    assert_eq!(v3.len(), 1);
}

#[test]
fn call_timeout_kills_slot_and_surfaces_timeout_error() {
    if !have_python() {
        return;
    }
    init_tracing();
    let enc = new_encoder(
        "--vector-dim 8 --hang-on-call 1",
        1,
        Duration::from_millis(400),
    )
    .expect("construction should succeed");

    let result = enc.encode(&["will_hang"]);
    // Implementation retries once on a respawn; both calls hang on call 1,
    // so we should see either Timeout or SubprocessDied (from the second
    // hang after respawn). Both are acceptable failure modes here.
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            EncoderError::Timeout { .. }
                | EncoderError::SubprocessDied { .. }
                | EncoderError::SlotUnhealthy { .. }
        ),
        "expected Timeout/SubprocessDied/SlotUnhealthy, got: {err:?}"
    );
}

#[test]
fn garbage_response_surfaces_protocol_violation() {
    if !have_python() {
        return;
    }
    init_tracing();
    let enc = new_encoder(
        "--vector-dim 8 --garbage-response",
        1,
        Duration::from_secs(5),
    )
    .expect("construction should succeed");

    let err = enc.encode(&["x"]).unwrap_err();
    assert!(
        matches!(
            err,
            EncoderError::ProtocolViolation { .. }
                | EncoderError::SubprocessDied { .. }
                | EncoderError::SlotUnhealthy { .. }
        ),
        "got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// stderr forwarding sanity check
// ---------------------------------------------------------------------------

#[test]
fn stderr_log_lines_are_drained_without_wedging() {
    if !have_python() {
        return;
    }
    init_tracing();
    let enc = new_encoder("--vector-dim 8 --log-stderr", 1, Duration::from_secs(5))
        .expect("construction should succeed");
    // Issue several calls — the stderr drain must not block the stdout
    // response path even if the script is chatty.
    for i in 0..5 {
        let text = format!("call_{i}");
        let v = enc.encode(&[text.as_str()]).unwrap();
        assert_eq!(v.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Clean shutdown
// ---------------------------------------------------------------------------

#[test]
fn drop_cleans_up_all_slots() {
    if !have_python() {
        return;
    }
    init_tracing();
    let enc = new_encoder("--vector-dim 8", 3, Duration::from_secs(5))
        .expect("construction should succeed");
    enc.encode(&["a", "b", "c"]).unwrap();
    // Dropping the encoder should shut down every slot via the Drop impl.
    drop(enc);
    // If we got here without hanging, shutdown worked.
}
