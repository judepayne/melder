//! Enrollment mutations and output snapshots share one operation-boundary gate.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

use crate::encoder::Encoder;
use crate::error::EncoderError;
use crate::models::Side;
use crate::store::RecordCollection;

use super::{AngleEncoder, TestResult, make_record};

fn typed_fixture(
    encoder: Arc<dyn Encoder>,
) -> Result<
    (
        tempfile::TempDir,
        Arc<crate::session::enroll::EnrollSession>,
    ),
    Box<dyn std::error::Error>,
> {
    let dir = tempfile::tempdir()?;
    let config = native_config(dir.path())?;
    let state = crate::state::enroll::EnrollState::load_job_with_encoder(config, encoder)?;
    Ok((
        dir,
        Arc::new(crate::session::enroll::EnrollSession::new(state, None)),
    ))
}

fn native_config(
    dir: &std::path::Path,
) -> Result<crate::config::enroll_schema::EnrollJobConfig, Box<dyn std::error::Error>> {
    let path = dir.join("job.yaml");
    std::fs::write(
        &path,
        format!(
            r#"
job: {{name: enrollment_test}}
vector_backend: flat
dataset: {{path: "", id_field: a_id}}
embeddings: {{model: test, cache_dir: {:?}}}
match_fields:
  - {{field: name, method: embedding, weight: 1.0}}
thresholds: {{auto_match: 0.8, review_floor: 0.1}}
live: {{match_log_path: {:?}}}
"#,
            dir.join("vectors"),
            dir.join("enroll.wal")
        ),
    )?;
    Ok(crate::config::load_enroll_job_config(&path)?)
}

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
fn native_enrollment_preserves_multiple_edges_and_isolated_evidence() -> TestResult {
    let (_dir, session) = typed_fixture(Arc::new(AngleEncoder))?;
    session.enroll(make_record(Side::A, "one", "0"))?;
    session.enroll(make_record(Side::A, "two", "27"))?;
    let result = session.enroll(make_record(Side::A, "three", "0"))?;
    assert_eq!(
        result.edges.len(),
        2,
        "enrollment retains multiple qualifying edges"
    );
    let (_, events) = session.state.snapshot()?;
    let outcomes: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            crate::state::match_log::MatchLogEvent::EnrollOutcome { query_id, edges } => {
                Some((query_id, edges))
            }
            _ => None,
        })
        .collect();
    assert_eq!(outcomes.len(), 3, "every enrollment publishes evidence");
    assert!(
        outcomes[0].1.is_empty(),
        "isolated enrollment persists empty evidence"
    );
    assert_eq!(
        outcomes[2].1.len(),
        2,
        "all returned edges persist without paired claims"
    );
    assert_eq!(
        outcomes[2].1[0].id, result.edges[0].id,
        "evidence preserves ranking"
    );
    Ok(())
}

#[test]
fn typed_enrollment_recovers_single_collection_and_ordered_evidence() -> TestResult {
    use crate::store::RecordCollection;

    let dir = tempfile::tempdir()?;
    let config = native_config(dir.path())?;
    let state =
        crate::state::enroll::EnrollState::load_job_with_encoder(config, Arc::new(AngleEncoder))?;
    let session = crate::session::enroll::EnrollSession::new(state.clone(), None);
    let result = session.enroll_batch(vec![
        make_record(Side::A, "one", "0"),
        make_record(Side::A, "two", "0"),
        make_record(Side::A, "one", "0"),
    ])?;
    assert_eq!(result.results.len(), 3, "bulk preserves repeated input IDs");
    assert!(
        result.results[1].edges.iter().any(|edge| edge.id == "one"),
        "later entries see earlier records"
    );
    assert!(
        result.results[2].edges.iter().all(|edge| edge.id != "one"),
        "self is excluded"
    );
    session.remove_record("two")?;
    state.flush()?;
    drop(session);
    let config = Arc::try_unwrap(state)
        .map_err(|_| "shared enrollment state")?
        .config;
    let recovered =
        crate::state::enroll::EnrollState::load_job_with_encoder(config, Arc::new(AngleEncoder))?;
    assert_eq!(
        recovered.collection.len()?,
        1,
        "WAL removal survives restart"
    );
    let (records, events) = recovered.snapshot()?;
    assert_eq!(
        records.len(),
        1,
        "snapshot uses current authoritative records"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            crate::state::match_log::MatchLogEvent::EnrollOutcome { .. }
        )),
        "snapshot retains directed enrollment evidence"
    );
    Ok(())
}

#[test]
fn enrollment_preparation_allows_mutations_and_admin_snapshots() -> TestResult {
    {
        for operation in ["reenroll", "remove", "exclude", "unexclude", "snapshot"] {
            let encoder = Arc::new(PausingEncoder {
                armed: AtomicBool::new(false),
                entered: Barrier::new(2),
                release: Barrier::new(2),
            });
            let (_dir, session) = typed_fixture(encoder.clone())?;
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
                    "remove" => competing.remove_record("endpoint").map(|_| ()),
                    "exclude" => competing.exclude("query", "endpoint").map(|_| ()),
                    "unexclude" => competing.unexclude("query", "endpoint").map(|_| ()),
                    "snapshot" => (|| {
                        // This is the same boundary used by the admin output builder.
                        let _guard = competing.assignment_guard()?;
                        competing.flush_assignment_log()?;
                        assert!(
                            competing.state.collection.get("query")?.is_none(),
                            "pure preparation must not expose an uncommitted record"
                        );
                        Ok(())
                    })(),
                    _ => unreachable!("fixed test operations"),
                };
                done_tx.send(result).expect("test receiver exists");
            });
            started_rx.recv_timeout(Duration::from_secs(5))?;
            let completed = done_rx.recv_timeout(Duration::from_secs(5));
            // Always release the worker before asserting, even if serialization regresses.
            encoder.release.wait();
            let _enrolled = enrollment.join().map_err(|_| "enrollment panicked")??;
            competitor
                .join()
                .map_err(|_| "competing operation panicked")?;
            completed
                .map_err(|_| format!("{operation} blocked on pure encoding (typed enroll)"))??;
            assert!(session.count()? <= 2, "enroll never claims pairs");
        }
    }
    Ok(())
}

#[test]
fn enrollment_batch_preserves_order_and_invalid_items() -> TestResult {
    let (_dir, session) = typed_fixture(Arc::new(AngleEncoder))?;
    let response = session.enroll_batch(vec![
        make_record(Side::A, "first", "0"),
        make_record(Side::A, "second", "0"),
        make_record(Side::A, "", "0"),
        make_record(Side::A, "first", "0"),
        make_record(Side::A, "third", "0"),
    ])?;
    assert_eq!(response.results.len(), 5, "one result per input");
    assert!(
        !response.results[2].enrolled,
        "invalid ID remains per-item error"
    );
    assert!(
        response.results[0].edges.is_empty(),
        "first sees empty pool"
    );
    assert_eq!(
        response.results[1].edges[0].id, "first",
        "second sees first"
    );
    assert_eq!(
        response.results[3].edges[0].id, "second",
        "duplicate excludes self"
    );
    assert_eq!(
        response.results[4].edges.len(),
        2,
        "later record sees both IDs"
    );
    Ok(())
}

#[derive(Debug)]
struct RejectBatchEncoder;

impl Encoder for RejectBatchEncoder {
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        if texts.len() > 1 {
            return Err(EncoderError::Inference("test batch rejection".into()));
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
fn enrollment_batch_retries_individual_encoding_on_batch_failure() -> TestResult {
    let (_dir, session) = typed_fixture(Arc::new(RejectBatchEncoder))?;
    let response = session.enroll_batch(vec![
        make_record(Side::A, "first", "0"),
        make_record(Side::A, "second", "0"),
    ])?;
    assert!(
        response.results.iter().all(|r| r.enrolled),
        "batch failure must not fail healthy individual records"
    );
    assert_eq!(
        response.results[1].edges[0].id, "first",
        "fallback preserves incremental visibility"
    );
    Ok(())
}

#[test]
fn enrollment_waits_for_admin_snapshot_boundary() -> TestResult {
    let (_dir, session) = typed_fixture(Arc::new(AngleEncoder))?;
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
