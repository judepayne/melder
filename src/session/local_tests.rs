//! Offline regression tests for live bounded local reassignment.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use dashmap::DashMap;
use tempfile::TempDir;

use crate::config::load_config;
use crate::crossmap::MemoryCrossMap;
use crate::encoder::Encoder;
use crate::error::EncoderError;
use crate::matching::exclusions::Exclusions;
use crate::models::{Record, Side};
use crate::state::live::{LiveMatchState, LiveSideState};
use crate::state::match_log::{MatchLog, MatchLogEvent};
use crate::store::memory::MemoryStore;
use crate::vectordb::flat::FlatVectorDB;

use super::Session;

type TestResult = Result<(), Box<dyn Error>>;

#[derive(Debug)]
struct LocalEncoder;

impl Encoder for LocalEncoder {
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        texts
            .iter()
            .map(|text| match *text {
                "axis0" => Ok(vec![1.0, 0.0]),
                "axis60" => Ok(vec![0.5, 0.75_f32.sqrt()]),
                angle if angle.starts_with("angle") => {
                    let degrees = angle[5..]
                        .parse::<f32>()
                        .map_err(|error| EncoderError::Inference(error.to_string()))?;
                    let angle = degrees.to_radians();
                    Ok(vec![angle.cos(), angle.sin()])
                }
                "opposite" => Ok(vec![-1.0, 0.0]),
                other => Err(EncoderError::Inference(format!(
                    "unexpected test text: {other}"
                ))),
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

fn make_fixture(threshold: f64) -> Result<Fixture, Box<dyn Error>> {
    make_fixture_with_backend(threshold, false)
}

fn make_fixture_with_backend(threshold: f64, sqlite: bool) -> Result<Fixture, Box<dyn Error>> {
    make_fixture_with_top_n(threshold, sqlite, 20)
}

fn make_fixture_with_top_n(
    threshold: f64,
    sqlite: bool,
    top_n: usize,
) -> Result<Fixture, Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let config_path = dir.path().join("config.yaml");
    std::fs::write(
        &config_path,
        format!(
            r#"
job:
  name: local_reassignment_test
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
  auto_match: {threshold}
  review_floor: 0.6
output:
  csv_dir_path: output/
vector_backend: flat
top_n: {top_n}
live:
  skip_initial_match: true
"#,
        ),
    )?;
    let config = load_config(&config_path)?;
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
        let (store, crossmap, _) = crate::store::sqlite::open_sqlite(
            &dir.path().join("live.db"),
            &config.blocking,
            None,
            &config.required_fields_a,
            &config.required_fields_b,
        )?;
        (Arc::new(store), Box::new(crossmap))
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
        encoder_pool: Arc::new(LocalEncoder),
        coordinator: None,
        wal: MatchLog::open(&dir.path().join("matches.ndjson"))?,
        crossmap_dirty: AtomicBool::new(false),
        review_queue: DashMap::new(),
        synonym_dictionary: None,
        config,
    };
    Ok(Fixture {
        session: Arc::new(Session::new(Arc::new(state), None, None)),
        _dir: dir,
    })
}

fn make_record(side: Side, id: &str, name: &str) -> Record {
    Record::from([
        (
            match side {
                Side::A => "a_id",
                Side::B => "b_id",
            }
            .into(),
            id.into(),
        ),
        ("name".into(), name.into()),
    ])
}

fn make_pair(side: Side, query: &str, pool: &str) -> (String, String) {
    match side {
        Side::A => (query.into(), pool.into()),
        Side::B => (pool.into(), query.into()),
    }
}

fn make_incumbent(session: &Session, side: Side) -> TestResult {
    session.upsert_record(side.opposite(), make_record(side.opposite(), "p1", "axis0"))?;
    session.upsert_record(
        side.opposite(),
        make_record(side.opposite(), "p2", "axis60"),
    )?;
    session.upsert_record(side, make_record(side, "q1", "angle27"))?;
    assert_pair(session, &make_pair(side, "q1", "p1"));
    assert!(
        session
            .local_automatic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&make_pair(side, "q1", "p1")),
        "fixture must establish an eligible ordinary automatic pair: {:?}",
        read_events(session)?
    );
    Ok(())
}

fn assert_pair(session: &Session, pair: &(String, String)) {
    let (a, b) = pair;
    assert_eq!(
        session.state.crossmap.get_b(a).as_ref(),
        Some(b),
        "forward pair must exist"
    );
    assert_eq!(
        session.state.crossmap.get_a(b).as_ref(),
        Some(a),
        "reverse pair must exist"
    );
}

fn assert_consistent(session: &Session, side: Side) -> TestResult {
    for query in ["q1", "q2"] {
        let paired = match side {
            Side::A => session.state.crossmap.has_a(query),
            Side::B => session.state.crossmap.has_b(query),
        };
        assert_eq!(
            session.state.store.is_unmatched(side, query)?,
            !paired,
            "query unmatched flag: {query}"
        );
    }
    for pool in ["p1", "p2"] {
        let paired = match side {
            Side::A => session.state.crossmap.has_b(pool),
            Side::B => session.state.crossmap.has_a(pool),
        };
        assert_eq!(
            session.state.store.is_unmatched(side.opposite(), pool)?,
            !paired,
            "pool unmatched flag: {pool}"
        );
    }
    for pair in session.state.crossmap.pairs() {
        assert_pair(session, &pair);
    }
    Ok(())
}

fn read_events(session: &Session) -> Result<Vec<MatchLogEvent>, Box<dyn Error>> {
    session.state.wal.flush()?;
    std::fs::read_to_string(session.state.wal.path())?
        .lines()
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn assert_not_reassigned(session: &Session, side: Side) -> TestResult {
    assert_eq!(
        session.state.crossmap.len(),
        1,
        "protected incumbent must not move"
    );
    assert_pair(session, &make_pair(side, "q1", "p1"));
    assert_consistent(session, side)?;
    assert!(
        !read_events(session)?
            .iter()
            .any(|event| matches!(event, MatchLogEvent::CrossMapReassign { .. })),
        "no reassignment WAL event expected"
    );
    Ok(())
}

#[test]
fn live_local_augmentation_is_symmetric_and_logs_one_atomic_reassignment() -> TestResult {
    for (side, sqlite) in [
        (Side::A, false),
        (Side::B, false),
        (Side::A, true),
        (Side::B, true),
    ] {
        let fixture = make_fixture_with_backend(0.8, sqlite)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        let before = read_events(session)?.len();
        let response = session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_eq!(
            response.classification,
            "auto",
            "incoming query should now auto-match; side={side:?}, response={response:?}, eligible={:?}",
            session
                .local_automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
        assert_eq!(
            session.state.crossmap.len(),
            2,
            "augmentation must increase cardinality"
        );
        assert_pair(session, &make_pair(side, "q1", "p2"));
        assert_pair(session, &make_pair(side, "q2", "p1"));
        assert_consistent(session, side)?;
        let events = read_events(session)?;
        let changes = &events[before..];
        let reassignments: Vec<_> = changes
            .iter()
            .filter(|event| matches!(event, MatchLogEvent::CrossMapReassign { .. }))
            .collect();
        assert_eq!(
            reassignments.len(),
            1,
            "one WAL transaction must describe both pairs"
        );
        assert!(
            !changes.iter().any(|event| matches!(
                event,
                MatchLogEvent::CrossMapBreak { .. } | MatchLogEvent::CrossMapConfirm { .. }
            )),
            "repair must not log intermediate half-pairs"
        );
        if let MatchLogEvent::CrossMapReassign {
            old_a_id,
            old_b_id,
            replacements,
        } = reassignments[0]
        {
            assert_eq!(
                (old_a_id.clone(), old_b_id.clone()),
                make_pair(side, "q1", "p1"),
                "WAL names expected incumbent"
            );
            let pairs: std::collections::HashSet<_> = replacements
                .iter()
                .map(|pair| (pair.a_id.clone(), pair.b_id.clone()))
                .collect();
            assert_eq!(
                pairs,
                [make_pair(side, "q1", "p2"), make_pair(side, "q2", "p1")]
                    .into_iter()
                    .collect(),
                "WAL must contain the complete replacement"
            );
            assert!(
                replacements.iter().all(|pair| pair.score >= 0.8),
                "both replacement scores must pass auto threshold"
            );
        }
    }
    Ok(())
}

#[test]
fn live_local_repair_uses_fourth_shortlist_entry_on_either_side_and_backend() -> TestResult {
    for incoming_fourth in [true, false] {
        // In the first case only incoming rank four is movable (preflight).
        // In the second case only incumbent rank four is free (alternative search).
        let (target, incumbent, blockers, incoming_order, incumbent_order) = if incoming_fourth {
            (
                "angle30",
                "angle40",
                vec!["axis0", "angle5", "angle10"],
                vec!["block0", "block1", "block2", "p1"],
                vec!["p1", "p2", "block2", "block1"],
            )
        } else {
            (
                "axis0",
                "angle27",
                vec!["angle10", "angle20"],
                vec!["p1", "block0", "block1", "p2"],
                vec!["block1", "block0", "p1", "p2"],
            )
        };
        for side in [Side::A, Side::B] {
            for sqlite in [false, true] {
                for top_n in [3, 4] {
                    let fixture = make_fixture_with_top_n(0.8, sqlite, top_n)?;
                    let session = &fixture.session;
                    for (id, name) in [("p1", target), ("p2", "axis60")] {
                        session.upsert_record(
                            side.opposite(),
                            make_record(side.opposite(), id, name),
                        )?;
                    }
                    for (i, name) in blockers.iter().enumerate() {
                        let id = format!("block{i}");
                        session.upsert_record(
                            side.opposite(),
                            make_record(side.opposite(), &id, name),
                        )?;
                    }
                    for (i, name) in blockers.iter().enumerate() {
                        let owner = format!("owner{i}");
                        let id = format!("block{i}");
                        session.upsert_record(side, make_record(side, &owner, name))?;
                        let (a, b) = make_pair(side, &owner, &id);
                        session.confirm_match(&a, &b)?;
                    }
                    session.upsert_record(side, make_record(side, "q1", incumbent))?;
                    assert_pair(session, &make_pair(side, "q1", "p1"));
                    assert!(
                        session
                            .local_automatic
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .contains(&make_pair(side, "q1", "p1")),
                        "incumbent must be an ordinary movable automatic pair"
                    );
                    let before = read_events(session)?.len();
                    let response = session.upsert_record(side, make_record(side, "q2", "axis0"))?;
                    for (query, expected) in [("q1", &incumbent_order), ("q2", &incoming_order)] {
                        let scored = session.score_stored(side, query)?;
                        assert_eq!(
                            scored
                                .iter()
                                .map(|r| r.matched_id.as_str())
                                .collect::<Vec<_>>(),
                            expected[..top_n],
                            "real score_pool shortlist: query={query}, incoming_fourth={incoming_fourth}, side={side:?}, sqlite={sqlite}"
                        );
                    }
                    let events = read_events(session)?;
                    let changes = &events[before..];
                    let reassignments: Vec<_> = changes
                        .iter()
                        .filter(|event| matches!(event, MatchLogEvent::CrossMapReassign { .. }))
                        .collect();
                    assert_eq!(
                        reassignments.len(),
                        usize::from(top_n == 4),
                        "repair requires rank four: incoming_fourth={incoming_fourth}, side={side:?}, sqlite={sqlite}, top_n={top_n}"
                    );
                    assert_eq!(
                        session.state.crossmap.len(),
                        blockers.len() + 1 + usize::from(top_n == 4),
                        "only the fourth-entry repair may increase cardinality"
                    );
                    if top_n == 3 {
                        assert_ne!(
                            response.classification, "auto",
                            "truncated shortlist cannot repair"
                        );
                        assert_pair(session, &make_pair(side, "q1", "p1"));
                    } else {
                        assert_eq!(
                            response.classification, "auto",
                            "full shortlist must repair"
                        );
                        assert_pair(session, &make_pair(side, "q1", "p2"));
                        assert_pair(session, &make_pair(side, "q2", "p1"));
                        if let MatchLogEvent::CrossMapReassign {
                            old_a_id,
                            old_b_id,
                            replacements,
                        } = reassignments[0]
                        {
                            assert_eq!(
                                (old_a_id.clone(), old_b_id.clone()),
                                make_pair(side, "q1", "p1"),
                                "WAL identifies incumbent"
                            );
                            let pairs: std::collections::HashSet<_> = replacements
                                .iter()
                                .map(|pair| (pair.a_id.clone(), pair.b_id.clone()))
                                .collect();
                            assert_eq!(
                                pairs,
                                [make_pair(side, "q1", "p2"), make_pair(side, "q2", "p1")]
                                    .into_iter()
                                    .collect(),
                                "WAL describes both replacements"
                            );
                            assert!(
                                replacements.iter().all(|pair| pair.score >= 0.8),
                                "replacement scores pass threshold"
                            );
                        }
                    }
                    assert!(
                        !changes.iter().any(|event| matches!(
                            event,
                            MatchLogEvent::CrossMapBreak { .. }
                                | MatchLogEvent::CrossMapConfirm { .. }
                        )),
                        "repair must never emit intermediate half-pairs"
                    );
                    for i in 0..blockers.len() {
                        let pair = make_pair(side, &format!("owner{i}"), &format!("block{i}"));
                        assert_pair(session, &pair);
                        assert!(
                            !session
                                .local_automatic
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .contains(&pair),
                            "manual blockers remain protected"
                        );
                    }
                    assert_consistent(session, side)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn live_local_manual_confirmation_protects_an_existing_automatic_pair() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        let (a, b) = make_pair(side, "q1", "p1");
        session.confirm_match(&a, &b)?;
        session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_not_reassigned(session, side)?;
    }
    Ok(())
}

#[test]
fn live_local_preloaded_crossmap_pair_is_protected() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        // A new Session sees fully indexed records and a pre-existing mapping,
        // but has no process-local evidence that this mapping is movable.
        let restored = Session::new(Arc::clone(&session.state), None, None);
        let (a, b) = make_pair(side, "q1", "p1");
        restored.state.crossmap.remove(&a, &b);
        restored.state.crossmap.add(&a, &b)?;
        restored.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_not_reassigned(&restored, side)?;
    }
    Ok(())
}

#[test]
fn live_local_common_id_pair_is_protected() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        let mut pool = make_record(side.opposite(), "p1", "axis0");
        pool.insert("common".into(), "canonical".into());
        session.upsert_record(side.opposite(), pool)?;
        session.upsert_record(
            side.opposite(),
            make_record(side.opposite(), "p2", "axis60"),
        )?;
        let mut query = make_record(side, "q1", "angle27");
        query.insert("common".into(), "canonical".into());
        session.upsert_record(side, query)?;
        assert!(read_events(session)?.iter().any(|event| matches!(event, MatchLogEvent::CrossMapConfirm { reason: Some(reason), .. } if reason == "canonical")), "fixture must take canonical fast path");
        session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_not_reassigned(session, side)?;
    }
    Ok(())
}

#[test]
fn live_local_excluded_alternative_prevents_augmentation() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        let (a, b) = make_pair(side, "q1", "p2");
        session.exclude(&a, &b)?;
        session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_not_reassigned(session, side)?;
    }
    Ok(())
}

#[test]
fn live_local_below_threshold_alternative_prevents_augmentation() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.85)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_not_reassigned(session, side)?;
    }
    Ok(())
}

#[test]
fn live_local_busy_mutation_gate_skips_without_waiting() -> TestResult {
    let fixture = make_fixture(0.8)?;
    let session = &fixture.session;
    make_incumbent(session, Side::B)?;
    let guard = session
        .local_mutations
        .read()
        .unwrap_or_else(|e| e.into_inner());
    let worker = Arc::clone(session);
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let result = worker.upsert_record(Side::B, make_record(Side::B, "q2", "axis0"));
        let _ = tx.send(result.is_ok());
    });
    let result = rx.recv_timeout(Duration::from_secs(5));
    drop(guard);
    assert!(handle.join().is_ok(), "upsert worker must not panic");
    assert_eq!(
        result.ok(),
        Some(true),
        "upsert must finish while another mutation reader is active"
    );
    assert_not_reassigned(session, Side::B)
}

#[test]
fn live_local_record_update_does_not_reuse_stale_eligibility() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        session.upsert_record(side, make_record(side, "q1", "opposite"))?;
        let pair = make_pair(side, "q1", "p1");
        assert!(
            !session
                .local_automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&pair),
            "updating an incumbent must revoke its old eligibility"
        );
        session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_eq!(
            session.state.crossmap.len(),
            1,
            "updated below-threshold incumbent must remain unmatched"
        );
        assert_pair(session, &make_pair(side, "q2", "p1"));
        assert_consistent(session, side)?;
        assert!(
            !read_events(session)?
                .iter()
                .any(|event| matches!(event, MatchLogEvent::CrossMapReassign { .. })),
            "stale incumbent must never be reassigned"
        );
    }
    Ok(())
}

#[test]
fn live_local_record_removal_revokes_eligibility() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        session.remove_record(side, "q1")?;
        let pair = make_pair(side, "q1", "p1");
        assert!(
            !session
                .local_automatic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&pair),
            "removal must revoke incumbent eligibility"
        );
        session.upsert_record(side, make_record(side, "q2", "axis0"))?;
        assert_pair(session, &make_pair(side, "q2", "p1"));
        assert_eq!(
            session.state.crossmap.len(),
            1,
            "removed record cannot participate in repair"
        );
        assert!(
            !read_events(session)?
                .iter()
                .any(|event| matches!(event, MatchLogEvent::CrossMapReassign { .. })),
            "removal must not leave a repair candidate"
        );
    }
    Ok(())
}

#[test]
fn live_local_free_automatic_partner_keeps_ordinary_claim_fast_path() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        make_incumbent(session, side)?;
        session.upsert_record(side, make_record(side, "q2", "axis60"))?;
        assert_pair(session, &make_pair(side, "q1", "p1"));
        assert_pair(session, &make_pair(side, "q2", "p2"));
        assert_consistent(session, side)?;
        assert!(
            !read_events(session)?
                .iter()
                .any(|event| matches!(event, MatchLogEvent::CrossMapReassign { .. })),
            "ordinary free candidate must avoid repair"
        );
    }
    Ok(())
}

#[test]
fn live_local_delayed_automatic_registration_cannot_unprotect_manual_confirmation() -> TestResult {
    use std::sync::atomic::Ordering;

    let fixture = make_fixture(0.8)?;
    let session = &fixture.session;
    make_incumbent(session, Side::B)?;
    let provenance = session.local_protection_epoch.load(Ordering::Acquire);
    let pending_result = session
        .score_stored(Side::B, "q1")?
        .into_iter()
        .find(|r| r.matched_id == "p1")
        .ok_or("missing ordinary candidate")?;
    session.confirm_match("p1", "q1")?;
    // Simulate an earlier claim finishing its WAL append after confirmation.
    session.remember_automatic_at(&pending_result, provenance);
    session.upsert_record(Side::B, make_record(Side::B, "q2", "axis0"))?;
    assert_not_reassigned(session, Side::B)?;
    Ok(())
}

#[test]
fn live_local_bulk_upserts_repair_only_after_all_records_are_installed() -> TestResult {
    for side in [Side::A, Side::B] {
        let fixture = make_fixture(0.8)?;
        let session = &fixture.session;
        session.upsert_record(side.opposite(), make_record(side.opposite(), "p1", "axis0"))?;
        session.upsert_record(
            side.opposite(),
            make_record(side.opposite(), "p2", "axis60"),
        )?;
        let response = session.upsert_batch(
            side,
            vec![
                make_record(side, "q1", "angle27"),
                make_record(side, "q2", "axis0"),
            ],
        )?;
        assert!(
            response.results.iter().all(|r| r.classification == "auto"),
            "both batch queries are matched"
        );
        assert_pair(session, &make_pair(side, "q1", "p2"));
        assert_pair(session, &make_pair(side, "q2", "p1"));
        assert_consistent(session, side)?;
    }
    Ok(())
}

#[test]
fn live_local_stale_query_vector_is_not_reused_for_repair() -> TestResult {
    let fixture = make_fixture(0.8)?;
    let session = &fixture.session;
    make_incumbent(session, Side::B)?;
    let index = session
        .state
        .b
        .combined_index
        .as_ref()
        .ok_or("missing test index")?;
    // A read-only query can replace a cached vector without replacing its record.
    index.upsert(
        "q1",
        &[1.0, 0.0],
        &make_record(Side::B, "q1", "axis0"),
        Side::B,
    )?;
    assert!(
        session.score_stored(Side::B, "q1")?.is_empty(),
        "stale vector is ineligible for local repair"
    );
    session.upsert_record(Side::B, make_record(Side::B, "q2", "axis0"))?;
    assert_not_reassigned(session, Side::B)?;
    Ok(())
}

#[test]
fn live_local_repair_preserves_reviews_owned_by_other_queries() -> TestResult {
    use crate::state::live::ReviewEntry;

    for sqlite in [false, true] {
        let fixture = make_fixture_with_backend(0.8, sqlite)?;
        let session = &fixture.session;
        make_incumbent(session, Side::B)?;
        for (key, side, id, candidate) in [
            ("b:outside:p2", Side::B, "outside", "p2"),
            ("b:p2:p1", Side::B, "p2", "p1"),
            ("a:p2:else", Side::A, "p2", "else"),
        ] {
            session.state.insert_review(
                key.into(),
                ReviewEntry {
                    id: id.into(),
                    side,
                    candidate_id: candidate.into(),
                    score: 0.7,
                },
            );
        }
        session.upsert_record(Side::B, make_record(Side::B, "q2", "axis0"))?;
        assert_pair(session, &make_pair(Side::B, "q1", "p2"));
        assert!(
            session.state.review_queue.contains_key("b:outside:p2"),
            "another query's review of the newly claimed target stays pending"
        );
        assert!(
            session.state.review_queue.contains_key("b:p2:p1"),
            "A/B identifier namespaces remain distinct"
        );
        assert!(
            !session.state.review_queue.contains_key("a:p2:else"),
            "resolved A query is cleared"
        );
        if sqlite {
            let reviews = session.state.store.load_reviews()?;
            assert!(
                reviews.iter().any(|r| r.0 == "b:outside:p2"),
                "SQLite retains unrelated pending review"
            );
            assert!(
                reviews.iter().any(|r| r.0 == "b:p2:p1"),
                "SQLite retains other namespace"
            );
            assert!(
                !reviews.iter().any(|r| r.0 == "a:p2:else"),
                "SQLite clears only resolved query"
            );
        }
    }
    Ok(())
}

#[test]
fn local_trial_does_not_change_enroll_multi_edge_semantics() -> TestResult {
    let fixture = make_fixture(0.8)?;
    let session = &fixture.session;
    session.enroll(make_record(Side::A, "one", "axis0"))?;
    session.enroll(make_record(Side::A, "two", "angle27"))?;
    let result = session.enroll(make_record(Side::A, "three", "axis0"))?;
    assert_eq!(
        result.edges.len(),
        2,
        "enroll still returns multiple qualifying edges"
    );
    assert!(
        session.state.crossmap.is_empty(),
        "enroll must not install 1:1 pairs"
    );
    assert!(
        session
            .local_automatic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty(),
        "enroll must not populate the optimizer hints"
    );
    Ok(())
}
