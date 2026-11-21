//! Offline live proposal regressions through the real embedding scoring pipeline.

use std::error::Error;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tempfile::TempDir;

use crate::config::load_config;
use crate::crossmap::MemoryCrossMap;
use crate::encoder::Encoder;
use crate::error::EncoderError;
use crate::matching::exclusions::Exclusions;
use crate::models::{Record, Side};
use crate::state::live::{LiveMatchState, LiveSideState};
use crate::state::match_log::{AssignmentDelta, MatchLog, MatchLogEvent};
use crate::store::memory::MemoryStore;
use crate::vectordb::flat::FlatVectorDB;

use super::Session;

#[path = "enroll_concurrency_tests.rs"]
mod enroll_concurrency_tests;

#[path = "assignment_policy_tests.rs"]
mod policy_tests;

#[path = "startup_input_tests.rs"]
mod startup_input_tests;

#[path = "assignment_cid_tests.rs"]
mod cid_tests;
#[path = "assignment_take_tests.rs"]
mod take_tests;

type TestResult = Result<(), Box<dyn Error>>;

#[derive(Debug)]
struct AngleEncoder;

impl Encoder for AngleEncoder {
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        texts
            .iter()
            .map(|text| {
                let degrees = text
                    .parse::<f32>()
                    .map_err(|error| EncoderError::Inference(error.to_string()))?;
                let angle = degrees.to_radians();
                Ok(vec![angle.cos(), angle.sin()])
            })
            .collect()
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

struct Fixture {
    session: Arc<Session>,
    _dir: TempDir,
}

fn make_fixture(sqlite: bool, cap: Option<usize>) -> Result<Fixture, Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let session = make_session(dir.path(), sqlite, cap)?;
    Ok(Fixture { session, _dir: dir })
}

fn make_session(
    root: &Path,
    sqlite: bool,
    cap: Option<usize>,
) -> Result<Arc<Session>, Box<dyn Error>> {
    let storage = if sqlite { "  db_path: live.db\n" } else { "" };
    let assignment = cap.map_or_else(String::new, |cap| {
        format!("  assignment:\n    strategy: deferred_acceptance\n    max_displacements: {cap}\n")
    });
    let path = root.join("config.yaml");
    std::fs::write(
        &path,
        format!(
            r#"
job:
  name: assignment_test
datasets:
  a:
    path: a.csv
    id_field: a_id
    common_id_field: common
  b:
    path: b.csv
    id_field: b_id
    common_id_field: common
cross_map:
  backend: local
  path: crossmap.csv
  a_id_field: a_id
  b_id_field: b_id
embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: cache/a
  b_cache_dir: cache/b
match_fields:
  - field_a: name
    field_b: name
    method: embedding
    weight: 1.0
thresholds:
  auto_match: 0.6
  review_floor: 0.5
output:
  csv_dir_path: output/
vector_backend: flat
top_n: 20
live:
  skip_initial_match: true
  match_log_path: matches.ndjson
{storage}{assignment}"#
        ),
    )?;
    let config = load_config(&path)?;
    let specs = crate::vectordb::embedding_field_specs(&config);
    let make_side = || LiveSideState {
        combined_index: Some(Box::new(FlatVectorDB::new_with_emb_specs(2, specs.clone()))),
        bm25_index: None,
        synonym_index: None,
    };
    let (store, crossmap): (
        Arc<dyn crate::store::RecordStore>,
        Box<dyn crate::crossmap::CrossMapOps>,
    ) = if sqlite {
        let (store, map, _) = crate::store::sqlite::open_sqlite(
            &root.join("live.db"),
            &config.blocking,
            None,
            &config.required_fields_a,
            &config.required_fields_b,
        )?;
        (Arc::new(store), Box::new(map))
    } else {
        (
            Arc::new(MemoryStore::new(&config.blocking)),
            Box::new(MemoryCrossMap::new()),
        )
    };
    let state = LiveMatchState {
        store,
        a: make_side(),
        b: make_side(),
        crossmap,
        exclusions: Exclusions::new(),
        encoder_pool: Arc::new(AngleEncoder),
        coordinator: None,
        wal: MatchLog::open(&root.join("matches.ndjson"))?,
        crossmap_dirty: AtomicBool::new(false),
        review_queue: DashMap::new(),
        synonym_dictionary: None,
        config,
    };
    Ok(Arc::new(Session::new(Arc::new(state), None, None)))
}

fn make_record(side: Side, id: &str, angle: &str) -> Record {
    Record::from([
        (
            if side == Side::A { "a_id" } else { "b_id" }.into(),
            id.into(),
        ),
        ("name".into(), angle.into()),
    ])
}

#[test]
fn deleting_endpoint_preserves_reviews_for_namesake() -> TestResult {
    use crate::state::live::{ReviewEntry, review_queue_key};
    for sqlite in [false, true] {
        for removed in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(4))?;
            let session = &fixture.session;
            session.upsert_record(removed, make_record(removed, "shared", "0"))?;
            let other = removed.opposite();
            for (side, id, candidate) in [
                (removed, "shared", "x"),
                (other, "y", "shared"),
                (other, "shared", "x"),
                (removed, "y", "shared"),
            ] {
                let key = review_queue_key(side, id, candidate);
                session
                    .state
                    .store
                    .persist_review(&key, id, side, candidate, 0.55)?;
                session.state.review_queue.insert(
                    key,
                    ReviewEntry {
                        side,
                        id: id.into(),
                        candidate_id: candidate.into(),
                        score: 0.55,
                    },
                );
            }
            session.remove_record(removed, "shared")?;
            let mut expected = vec![
                review_queue_key(other, "shared", "x"),
                review_queue_key(removed, "y", "shared"),
            ];
            expected.sort();
            let mut actual: Vec<_> = session
                .state
                .review_queue
                .iter()
                .map(|e| e.key().clone())
                .collect();
            actual.sort();
            assert_eq!(actual, expected);
            if sqlite {
                let mut stored: Vec<_> = session
                    .state
                    .store
                    .load_reviews()?
                    .into_iter()
                    .map(|e| e.0)
                    .collect();
                stored.sort();
                assert_eq!(stored, expected);
            }
        }
    }
    Ok(())
}

#[test]
fn assignment_append_failure_rolls_back_and_stops_mutations() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(4))?;
            let session = &fixture.session;
            session.upsert_record(side.opposite(), make_record(side.opposite(), "pool", "0"))?;
            session.upsert_record(side, make_record(side, "old", "20"))?;
            assert_partner(session, side, "old", Some("pool"));
            session.fail_next_assignment_append();
            let error = session.upsert_record(side, make_record(side, "new", "0"));
            assert!(
                matches!(error, Err(crate::error::SessionError::Wal(_))),
                "report append failure"
            );
            assert_partner(session, side, "old", Some("pool"));
            assert_partner(session, side, "new", None);
            assert_metadata(session)?;
            assert_eq!(
                read_deltas(session)?.len(),
                1,
                "failed delta never entered WAL"
            );
            assert!(
                matches!(
                    session.remove_record(side, "old"),
                    Err(crate::error::SessionError::AssignmentUnhealthy)
                ),
                "further writes require recovery"
            );
        }
    }
    Ok(())
}

#[test]
fn assignment_bulk_delete_stops_after_failed_orphan_claim() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(4))?;
            let session = &fixture.session;
            let delete_side = side.opposite();
            session.upsert_record(side, make_record(side, "owner", "0"))?;
            session.upsert_record(delete_side, make_record(delete_side, "first", "0"))?;
            session.upsert_record(delete_side, make_record(delete_side, "next", "20"))?;
            assert_partner(session, side, "owner", Some("first"));
            session.fail_next_assignment_append();
            let result = session.remove_batch(delete_side, vec!["first".into(), "next".into()]);
            assert!(
                matches!(result, Err(crate::error::SessionError::Wal(_))),
                "bulk must report the proposal failure"
            );
            assert!(
                !session.state.store.contains(delete_side, "first")?,
                "the completed deletion is not rolled back"
            );
            assert!(
                session.state.store.contains(delete_side, "next")?,
                "later bulk items must not be touched"
            );
            assert_partner(session, side, "owner", None);
            assert_metadata(session)?;
            assert_eq!(
                read_deltas(session)?.len(),
                1,
                "failed rematch adds no assignment event"
            );
        }
    }
    Ok(())
}

#[test]
fn directional_bm25_uses_receivers_current_preference_not_old_claim_score() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let mut fixture = make_fixture(sqlite, Some(4))?;
            let session = Arc::get_mut(&mut fixture.session).expect("unique session");
            session.emb_specs.clear();
            let state = Arc::get_mut(&mut session.state).expect("unique live state");
            state.config.match_fields[0].method = crate::config::MatchMethod::Bm25;
            let fields = [crate::config::Bm25FieldPair {
                field_a: "name".into(),
                field_b: "name".into(),
            }];
            state.a.combined_index = None;
            state.b.combined_index = None;
            state.a.bm25_index = Some(crate::bm25::simple::SimpleBm25::new(&fields, Side::A));
            state.b.bm25_index = Some(crate::bm25::simple::SimpleBm25::new(&fields, Side::B));
            session.upsert_record(
                side.opposite(),
                make_record(side.opposite(), "receiver", "alpha"),
            )?;
            session.upsert_record(side, make_record(side, "old", "alpha beta"))?;
            assert_partner(session, side, "old", Some("receiver"));
            let old = read_deltas(session)?.remove(0).replacements.remove(0);
            let (proposer_score, receiver_score) = match side {
                Side::A => (old.score_a_to_b, old.score_b_to_a),
                Side::B => (old.score_b_to_a, old.score_a_to_b),
            };
            assert!(
                proposer_score > receiver_score,
                "fixture must produce reciprocal score asymmetry"
            );
            let report = session
                .upsert_record(side, make_record(side, "new", "alpha"))?
                .assignment
                .expect("assignment");
            assert_eq!(
                report.displacements, 1,
                "receiver prefers short exact lexical match"
            );
            assert_partner(session, side, "new", Some("receiver"));
            assert_metadata(session)?;
        }
    }
    Ok(())
}

#[test]
fn assignment_delete_rematches_the_surviving_orphan() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(4))?;
            let session = &fixture.session;
            session.upsert_record(side, make_record(side, "receiver", "0"))?;
            session.upsert_record(side.opposite(), make_record(side.opposite(), "old", "0"))?;
            session.upsert_record(side.opposite(), make_record(side.opposite(), "other", "20"))?;
            assert_partner(session, side, "receiver", Some("old"));
            let result = session.remove_record(side.opposite(), "old")?;
            let summary = result.assignment.expect("deferred deletion summary");
            assert_eq!(
                summary.changes.len(),
                1,
                "deletion seeds the surviving owner"
            );
            assert_eq!(
                summary.displacements, 0,
                "free claim does not consume budget"
            );
            assert_partner(session, side, "receiver", Some("other"));
            assert_metadata(session)?;
        }
    }
    Ok(())
}

fn assert_partner(session: &Session, side: Side, id: &str, expected: Option<&str>) {
    let actual = match side {
        Side::A => session.state.crossmap.get_b(id),
        Side::B => session.state.crossmap.get_a(id),
    };
    assert_eq!(actual.as_deref(), expected, "partner of {side:?}:{id}");
    if let Some(other) = expected {
        let reverse = match side {
            Side::A => session.state.crossmap.get_a(other),
            Side::B => session.state.crossmap.get_b(other),
        };
        assert_eq!(reverse.as_deref(), Some(id), "reverse membership");
    }
}

fn assert_metadata(session: &Session) -> TestResult {
    for side in [Side::A, Side::B] {
        for id in session.state.store.ids(side)? {
            let paired = match side {
                Side::A => session.state.crossmap.has_a(&id),
                Side::B => session.state.crossmap.has_b(&id),
            };
            assert_eq!(
                session.state.store.is_unmatched(side, &id)?,
                !paired,
                "matched flag of {side:?}:{id}"
            );
            if paired {
                assert!(
                    !session
                        .state
                        .review_queue
                        .iter()
                        .any(|entry| entry.side == side && entry.id == id),
                    "accepted owner has no stale review"
                );
            }
        }
    }
    for (a, b) in session.state.crossmap.pairs() {
        assert_partner(session, Side::A, &a, Some(&b));
    }
    Ok(())
}

fn read_deltas(session: &Session) -> Result<Vec<AssignmentDelta>, Box<dyn Error>> {
    session.state.wal.flush()?;
    let mut deltas = vec![];
    for line in std::fs::read_to_string(session.state.wal.path())?.lines() {
        if let MatchLogEvent::AssignmentDelta(delta) = serde_json::from_str(line)? {
            deltas.push(delta);
        }
    }
    Ok(deltas)
}

// Decreasing chooser preferences along the displacement chain:
// q3-p1=1 > q1-p1=cos20 > q1-p2=cos30 > q2-p2=cos40 > q2-p3=cos50.
// All edges used pass both directions, with no model downloads or fabricated scores.
fn make_pool(session: &Session, side: Side) -> TestResult {
    for (id, angle) in [("p1", "0"), ("p2", "50"), ("p3", "140")] {
        session.upsert_record(side.opposite(), make_record(side.opposite(), id, angle))?;
    }
    Ok(())
}

fn make_chain(session: &Session, side: Side) -> TestResult {
    make_pool(session, side)?;
    session.upsert_record(side, make_record(side, "q1", "20"))?;
    session.upsert_record(side, make_record(side, "q2", "90"))?;
    assert_partner(session, side, "q1", Some("p1"));
    assert_partner(session, side, "q2", Some("p2"));
    Ok(())
}

// A separate process is needed so the killed attempt cannot run destructors or
// unwind the in-flight chain. No failpoint is present in the `meld` binary.
#[test]
fn sqlite_recovery_child() -> TestResult {
    let Some(operation) = std::env::var_os("MELDER_TEST_SQLITE_RECOVERY_CHILD") else {
        return Ok(());
    };
    match operation.to_str() {
        Some("a" | "b") => {
            let side = if operation == "a" { Side::A } else { Side::B };
            let root = std::env::current_dir()?;
            let session = make_session(&root, true, Some(3))?;
            make_chain(&session, side)?;
            session.pause_after_assignment_displacement(root.join("first-committed"));
            session.upsert_record(side, make_record(side, "q3", "0"))?;
            panic!("fixture must pause after its first successful displacement");
        }
        Some("export") => {
            crate::cli::export::cmd_export(Path::new("config.yaml"), Path::new("export"));
        }
        _ => return Err("unknown SQLite recovery test operation".into()),
    }
    Ok(())
}

struct RecoveryChild(Child);

impl Drop for RecoveryChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start_recovery_child(root: &Path, operation: &str) -> std::io::Result<RecoveryChild> {
    Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "session::assignment_tests::sqlite_recovery_child",
            "--nocapture",
        ])
        .env("MELDER_TEST_SQLITE_RECOVERY_CHILD", operation)
        .current_dir(root)
        .stdin(Stdio::null())
        .spawn()
        .map(RecoveryChild)
}

#[test]
fn sqlite_restart_recovers_only_the_committed_displacement_prefix() -> TestResult {
    for side in [Side::A, Side::B] {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let mut child = start_recovery_child(root, if side == Side::A { "a" } else { "b" })?;
        let ready = root.join("first-committed");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if std::fs::read(&ready).is_ok_and(|bytes| bytes == b"committed") {
                break;
            }
            assert!(
                child.0.try_wait()?.is_none(),
                "chain exited before the first commit marker"
            );
            assert!(
                Instant::now() < deadline,
                "timed out waiting for first commit marker"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        child.0.kill()?;
        assert!(
            !child.0.wait()?.success(),
            "the child must be killed, not shut down gracefully"
        );

        let wal = root.join("matches.ndjson");
        let events = MatchLog::replay(&wal)?;
        let displacements: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                MatchLogEvent::AssignmentDelta(delta) if !delta.expected.is_empty() => Some(delta),
                _ => None,
            })
            .collect();
        assert_eq!(
            displacements.len(),
            1,
            "no second displacement ran before the crash"
        );
        assert_eq!(displacements[0].newly_unmatched[0].id, "q1");
        // Remove the application's WAL from discovery, without deleting the
        // evidence. Startup and export must use the database as the authority.
        let saved = root.join("old-audit");
        std::fs::create_dir(&saved)?;
        for file in MatchLog::find_wal_files(&wal) {
            let name = file.file_name().ok_or("WAL missing filename")?;
            std::fs::rename(&file, saved.join(name))?;
        }
        let mut config = load_config(&root.join("config.yaml"))?;
        config.live.match_log_path = Some(wal.to_string_lossy().into_owned());
        config.embeddings.a_cache_dir = root.join("cache-a").to_string_lossy().into_owned();
        config.embeddings.b_cache_dir = Some(root.join("cache-b").to_string_lossy().into_owned());
        let db = root.join("live.db");
        let restarted = Session::new(
            LiveMatchState::restart_sqlite_with_encoder(
                config,
                &db.to_string_lossy(),
                Arc::new(AngleEncoder),
            )?,
            None,
            None,
        );
        for (id, expected) in [("q1", None), ("q2", Some("p2")), ("q3", Some("p1"))] {
            assert_eq!(
                restarted.lookup_crossmap(id, side)?.paired_id.as_deref(),
                expected,
                "lookup for {id} after restart"
            );
        }
        assert_metadata(&restarted)?;
        assert!(
            read_deltas(&restarted)?.is_empty(),
            "startup must not resume the chain"
        );
        let result = restarted.upsert_record(side, make_record(side, "q4", "50"))?;
        assert_eq!(
            result
                .assignment
                .ok_or("missing assignment summary")?
                .displacements,
            0,
            "restored p2 link is protected from stronger q4"
        );
        assert_partner(&restarted, side, "q2", Some("p2"));
        assert_metadata(&restarted)?;

        let mut export = start_recovery_child(root, "export")?;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(exit) = export.0.try_wait()? {
                assert!(exit.success(), "SQLite CLI export failed: {exit}");
                break;
            }
            assert!(Instant::now() < deadline, "SQLite CLI export timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut exported: Vec<(String, String)> =
            csv::Reader::from_path(root.join("export/results.csv"))?
                .deserialize()
                .collect::<Result<_, _>>()?;
        let mut actual = restarted.state.crossmap.pairs();
        actual.sort_unstable();
        exported.sort_unstable();
        assert_eq!(
            exported, actual,
            "export exactly matches recovered database membership"
        );
    }
    Ok(())
}

#[test]
fn sequential_chain_caps_and_audit_are_mirrored() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            for cap in [0, 1, 2, 3] {
                let fixture = make_fixture(sqlite, Some(cap))?;
                let session = &fixture.session;
                make_chain(session, side)?;
                let before = read_deltas(session)?.len();
                let response = session.upsert_record(side, make_record(side, "q3", "0"))?;
                let summary = response.assignment.as_ref().ok_or("missing summary")?;
                assert_eq!(
                    summary.displacements,
                    cap.min(2),
                    "successful displacements only"
                );
                assert_eq!(
                    summary.limit_reached,
                    cap < 2,
                    "cap reached only on suppressed improvement"
                );
                assert_eq!(
                    summary.partner.as_deref(),
                    if cap == 0 { None } else { Some("p1") }
                );
                assert_partner(
                    session,
                    side,
                    "q3",
                    if cap == 0 { None } else { Some("p1") },
                );
                assert_partner(
                    session,
                    side,
                    "q1",
                    match cap {
                        0 => Some("p1"),
                        1 => None,
                        _ => Some("p2"),
                    },
                );
                assert_partner(
                    session,
                    side,
                    "q2",
                    if cap < 2 { Some("p2") } else { Some("p3") },
                );
                let deltas = read_deltas(session)?;
                let steps = &deltas[before..];
                assert_eq!(
                    steps.len(),
                    match cap {
                        0 => 0,
                        1 => 1,
                        _ => 3,
                    },
                    "free accept is a separate audited step"
                );
                assert_eq!(
                    summary.changes.len(),
                    steps.len(),
                    "response and WAL steps agree"
                );
                for (index, step) in steps.iter().enumerate() {
                    step.validate()?;
                    assert_eq!(step.version, 1, "event schema version");
                    assert_eq!(step.step_index, index, "ordered attempt steps");
                    assert_eq!(step.attempt_id, steps[0].attempt_id, "one request attempt");
                    assert_eq!(step.proposer.side, side, "orphan remains proposer side");
                    let pair = &step.replacements[0];
                    assert!(
                        pair.score_b_to_a >= 0.6 && pair.score_a_to_b >= 0.6,
                        "both current directional scores qualify"
                    );
                    let new_receiver = if side == Side::A {
                        pair.score_b_to_a
                    } else {
                        pair.score_a_to_b
                    };
                    if let Some(old) = step.previous_receiver_score {
                        assert!(new_receiver > old, "receiver strictly improves");
                        assert_eq!(step.expected.len(), 1, "one displaced incumbent");
                        assert_eq!(step.newly_unmatched.len(), 1, "audited orphan");
                    } else {
                        assert!(step.expected.is_empty(), "free claim has no old pair");
                    }
                }
                if cap == 1 {
                    assert!(
                        summary
                            .unmatched
                            .iter()
                            .any(|e| e.side == side && e.id == "q1"),
                        "orphan reported"
                    );
                    assert!(
                        session
                            .state
                            .review_queue
                            .iter()
                            .any(|e| e.side == side && e.id == "q1"),
                        "orphan owns review"
                    );
                    session.upsert_record(side, make_record(side, "q1", "140"))?;
                    assert_partner(session, side, "q1", Some("p3"));
                }
                assert_metadata(session)?;
            }
        }
    }
    Ok(())
}

#[test]
fn cap_zero_still_scans_a_free_lower_ranked_alternative() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(0))?;
            let session = &fixture.session;
            session.upsert_record(side.opposite(), make_record(side.opposite(), "p1", "0"))?;
            session.upsert_record(side.opposite(), make_record(side.opposite(), "p2", "50"))?;
            session.upsert_record(side, make_record(side, "q1", "20"))?;
            let response = session.upsert_record(side, make_record(side, "q2", "0"))?;
            let summary = response.assignment.ok_or("missing summary")?;
            assert_eq!(
                summary.displacements, 0,
                "free claims do not spend displacement budget"
            );
            assert!(
                summary.limit_reached,
                "better occupied receiver was suppressed"
            );
            assert_eq!(
                summary.partner.as_deref(),
                Some("p2"),
                "continue after capped proposal"
            );
            assert_partner(session, side, "q1", Some("p1"));
            assert_partner(session, side, "q2", Some("p2"));
            assert_metadata(session)?;
        }
    }
    Ok(())
}

#[test]
fn receiver_ties_and_worse_proposals_keep_the_incumbent() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            for newcomer in ["20", "30"] {
                let fixture = make_fixture(sqlite, Some(8))?;
                let session = &fixture.session;
                session.upsert_record(side.opposite(), make_record(side.opposite(), "p1", "0"))?;
                session.upsert_record(side, make_record(side, "q1", "20"))?;
                let before = read_deltas(session)?.len();
                let response = session.upsert_record(side, make_record(side, "q2", newcomer))?;
                let summary = response.assignment.ok_or("missing summary")?;
                assert_eq!(
                    summary.displacements, 0,
                    "receiver does not strictly improve"
                );
                assert!(!summary.limit_reached, "rejection is not budget exhaustion");
                assert_eq!(
                    read_deltas(session)?.len(),
                    before,
                    "no successful transition event"
                );
                assert_partner(session, side, "q1", Some("p1"));
                assert_partner(session, side, "q2", None);
                assert_metadata(session)?;
            }
        }
    }
    Ok(())
}

#[test]
fn manual_and_restored_membership_is_protected() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            for restored in [false, true] {
                let fixture = make_fixture(sqlite, Some(8))?;
                let original = &fixture.session;
                original.upsert_record(side.opposite(), make_record(side.opposite(), "p1", "0"))?;
                original.upsert_record(side, make_record(side, "q1", "20"))?;
                // A fresh Session over already loaded membership deliberately has no
                // automatic provenance, just like membership restored at startup.
                let fresh;
                let session = if restored {
                    fresh = Session::new(Arc::clone(&original.state), None, None);
                    &fresh
                } else {
                    let (a, b) = if side == Side::A {
                        ("q1", "p1")
                    } else {
                        ("p1", "q1")
                    };
                    original.confirm_match(a, b)?;
                    original.as_ref()
                };
                let response = session.upsert_record(side, make_record(side, "q2", "0"))?;
                let summary = response.assignment.ok_or("missing summary")?;
                assert_eq!(
                    summary.displacements, 0,
                    "protected membership cannot be displaced"
                );
                assert!(!summary.limit_reached, "protection is not a cap rejection");
                assert_partner(session, side, "q1", Some("p1"));
                assert_partner(session, side, "q2", None);
                assert_metadata(session)?;
            }
        }
    }
    Ok(())
}

#[test]
fn common_id_confirmation_is_protected_from_proposals() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(8))?;
            let session = &fixture.session;
            let mut pool = make_record(side.opposite(), "p1", "0");
            pool.insert("common".into(), "canonical".into());
            session.upsert_record(side.opposite(), pool)?;
            let mut query = make_record(side, "q1", "20");
            query.insert("common".into(), "canonical".into());
            let canonical = session.upsert_record(side, query)?;
            assert_eq!(
                canonical.classification, "auto",
                "common-ID confirmation is accepted"
            );
            assert!(
                read_deltas(session)?.is_empty(),
                "canonical confirmation is not an ordinary proposal"
            );
            assert_partner(session, side, "q1", Some("p1"));
            let response = session.upsert_record(side, make_record(side, "q2", "0"))?;
            assert_eq!(
                response.assignment.ok_or("missing summary")?.displacements,
                0,
                "better cosine cannot displace canonical membership"
            );
            let mut duplicate = make_record(side, "q3", "0");
            duplicate.insert("common".into(), "canonical".into());
            session.upsert_record(side, duplicate)?;
            assert_partner(session, side, "q3", None);
            assert_partner(session, side, "q1", Some("p1"));
            assert_partner(session, side, "q2", None);
            assert_metadata(session)?;
        }
    }
    Ok(())
}

#[test]
fn bulk_returns_final_not_temporary_assignments() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(1))?;
            let session = &fixture.session;
            make_pool(session, side)?;
            let response = session.upsert_batch(
                side,
                vec![
                    make_record(side, "q1", "20"),
                    make_record(side, "q2", "90"),
                    make_record(side, "q3", "0"),
                ],
            )?;
            let json = serde_json::to_value(response)?;
            let results = json
                .get("results")
                .and_then(serde_json::Value::as_array)
                .ok_or("bulk results missing")?;
            assert_eq!(results.len(), 3, "all bulk items returned");
            for item in results {
                let id = item
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("bulk id missing")?;
                let expected = match id {
                    "q1" => None,
                    "q2" => Some("p2"),
                    "q3" => Some("p1"),
                    _ => return Err("unexpected bulk id".into()),
                };
                let partner = item
                    .pointer("/assignment/partner")
                    .and_then(serde_json::Value::as_str);
                assert_eq!(
                    partner, expected,
                    "bulk summary describes final CrossMap for {id}"
                );
                if expected.is_none() {
                    assert_ne!(
                        item.get("classification")
                            .and_then(serde_json::Value::as_str),
                        Some("auto"),
                        "later-displaced item cannot remain Auto"
                    );
                }
                assert_partner(session, side, id, expected);
            }
            assert_metadata(session)?;
        }
    }
    Ok(())
}

#[test]
fn greedy_default_allocates_no_assignment_state_or_events() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, None)?;
            let session = &fixture.session;
            assert!(
                session.assignment.is_none(),
                "default has no gate or provenance allocation"
            );
            make_chain(session, side)?;
            let response = session.upsert_record(side, make_record(side, "q3", "0"))?;
            assert!(
                response.assignment.is_none(),
                "greedy response has no optimizer summary"
            );
            assert!(
                serde_json::to_value(response)?.get("assignment").is_none(),
                "optional field omitted on wire"
            );
            assert!(
                read_deltas(session)?.is_empty(),
                "greedy never emits assignment transitions"
            );
            assert_partner(session, side, "q1", Some("p1"));
            assert_partner(session, side, "q2", Some("p2"));
            assert_partner(session, side, "q3", None);
            assert_metadata(session)?;
        }
    }
    Ok(())
}
