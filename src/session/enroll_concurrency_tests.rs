//! Enrollment mutations and output snapshots share one operation-boundary gate.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

use crate::config::Mode;
use crate::encoder::Encoder;
use crate::error::EncoderError;
use crate::models::Side;

use super::{AngleEncoder, Session, TestResult, make_fixture, make_record};

#[derive(Debug)]
struct PausingEncoder {
    armed: AtomicBool,
    entered: Barrier,
    release: Barrier,
}

impl Encoder for PausingEncoder {
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.entered.wait();
            self.release.wait();
        }
        AngleEncoder.encode(texts)
    }

    fn dim(&self) -> usize {
        2
    }

    fn pool_size(&self) -> usize {
        1
    }

    fn encode_batch_size(&self) -> usize {
        32
    }
}

#[test]
fn enrollment_serializes_reenroll_remove_exclude_and_admin_snapshots() -> TestResult {
    for sqlite in [false, true] {
        for operation in ["reenroll", "remove", "exclude", "unexclude", "snapshot"] {
            let fixture = make_fixture(sqlite, None)?;
            let old = Arc::try_unwrap(fixture.session)
                .map_err(|_| "fixture session unexpectedly shared")?;
            let mut state = old.state;
            let mutable = Arc::get_mut(&mut state).ok_or("fixture state unexpectedly shared")?;
            mutable.config.mode = Mode::Enroll;
            let encoder = Arc::new(PausingEncoder {
                armed: AtomicBool::new(false),
                entered: Barrier::new(2),
                release: Barrier::new(2),
            });
            mutable.encoder_pool = encoder.clone();
            let session = Arc::new(Session::new(state, None, None));
            session.enroll(make_record(Side::A, "endpoint", "0"))?;

            encoder.armed.store(true, Ordering::SeqCst);
            let enrolling = session.clone();
            let enrollment =
                std::thread::spawn(move || enrolling.enroll(make_record(Side::A, "query", "0")));
            encoder.entered.wait(); // The enrollment is in flight, before evidence publication.

            let competing = session.clone();
            let (started_tx, started_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();
            let competitor = std::thread::spawn(move || {
                started_tx.send(()).expect("test receiver exists");
                let result = match operation {
                    "reenroll" => competing
                        .enroll(make_record(Side::A, "query", "180"))
                        .map(|_| ()),
                    "remove" => competing.remove_record(Side::A, "endpoint").map(|_| ()),
                    "exclude" => competing.exclude("query", "endpoint").map(|_| ()),
                    "unexclude" => competing.unexclude("query", "endpoint").map(|_| ()),
                    "snapshot" => (|| {
                        // This is the same boundary used by the admin output builder.
                        let _guard = competing.assignment_guard()?;
                        competing.flush_assignment_log()?;
                        assert!(
                            competing.state.store.get(Side::A, "query")?.is_some(),
                            "snapshot must observe the completed enrollment"
                        );
                        Ok(())
                    })(),
                    _ => unreachable!("fixed test operations"),
                };
                done_tx.send(result).expect("test receiver exists");
            });
            started_rx.recv_timeout(Duration::from_secs(5))?;
            let premature = done_rx.recv_timeout(Duration::from_millis(50));
            // Always release the worker before asserting, even if serialization regresses.
            encoder.release.wait();
            let enrolled = enrollment.join().map_err(|_| "enrollment panicked")??;
            competitor
                .join()
                .map_err(|_| "competing operation panicked")?;
            assert!(
                matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)),
                "{operation} must wait for enrollment evidence publication (sqlite={sqlite})"
            );
            done_rx.recv_timeout(Duration::from_secs(5))??;
            assert_eq!(
                enrolled.edges.len(),
                1,
                "enrollment retains its scored edge"
            );
            assert!(
                session.state.crossmap.is_empty(),
                "enroll never claims pairs"
            );
        }
    }
    Ok(())
}

#[test]
fn enrollment_waits_for_admin_snapshot_boundary() -> TestResult {
    let fixture = make_fixture(false, None)?;
    let old =
        Arc::try_unwrap(fixture.session).map_err(|_| "fixture session unexpectedly shared")?;
    let mut state = old.state;
    Arc::get_mut(&mut state)
        .ok_or("fixture state unexpectedly shared")?
        .config
        .mode = Mode::Enroll;
    let session = Arc::new(Session::new(state, None, None));
    let guard = session.assignment_guard()?;
    assert!(guard.is_some(), "enroll snapshots must acquire a guard");
    let worker = session.clone();
    let (tx, rx) = mpsc::channel();
    let enrollment =
        std::thread::spawn(move || tx.send(worker.enroll(make_record(Side::A, "query", "0"))));
    let premature = rx.recv_timeout(Duration::from_millis(50));
    drop(guard);
    enrollment.join().map_err(|_| "enrollment panicked")??;
    assert!(
        matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)),
        "enrollment cannot mutate the pool during an admin snapshot"
    );
    rx.recv_timeout(Duration::from_secs(5))??;
    Ok(())
}
