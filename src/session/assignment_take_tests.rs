//! Storage refusal regressions for pair removal during record mutations.

use super::{TestResult, make_fixture, make_record};
use crate::error::{CrossMapError, SessionError};
use crate::models::Side;

#[test]
fn sqlite_take_refusal_aborts_record_mutation_on_both_sides() -> TestResult {
    for side in [Side::A, Side::B] {
        for update in [false, true] {
            for bulk in [false, true] {
                for cap in [None, Some(4)] {
                    let fixture = make_fixture(true, cap)?;
                    let session = &fixture.session;
                    session.upsert_record(
                        side.opposite(),
                        make_record(side.opposite(), "receiver", "0"),
                    )?;
                    session.upsert_record(side, make_record(side, "owner", "0"))?;
                    assert_eq!(
                        session.assigned_partner(side, "owner").as_deref(),
                        Some("receiver"),
                        "fixture must have an incumbent pair",
                    );
                    let original = session.state.store.get(side, "owner")?;
                    let conn = rusqlite::Connection::open(fixture._dir.path().join("live.db"))?;
                    conn.execute_batch(
                        "CREATE TRIGGER reject_pair_delete BEFORE DELETE ON crossmap
                         BEGIN SELECT RAISE(ABORT, 'injected pair delete refusal'); END;",
                    )?;
                    let error = if update {
                        if bulk {
                            session
                                .upsert_batch(
                                    side,
                                    vec![
                                        make_record(side, "owner", "90"),
                                        make_record(side, "later", "180"),
                                    ],
                                )
                                .map(|r| {
                                    assert!(
                                        cap.is_none() && r.results[0].status.starts_with("error:"),
                                        "greedy bulk must report the storage error per item",
                                    );
                                })
                                .err()
                        } else {
                            session
                                .upsert_record(side, make_record(side, "owner", "90"))
                                .err()
                        }
                    } else if bulk {
                        session
                            .remove_batch(side, vec!["owner".into(), "receiver".into()])
                            .map(|r| {
                                assert!(
                                    cap.is_none() && r.results[0].status.starts_with("error:"),
                                    "greedy bulk must report the storage error per item",
                                );
                            })
                            .err()
                    } else {
                        session.remove_record(side, "owner").err()
                    };
                    if !bulk || cap.is_some() {
                        assert!(
                            matches!(
                                error,
                                Some(SessionError::CrossMap(CrossMapError::Sqlite(_)))
                            ),
                            "take refusal must be a typed error: {error:?}",
                        );
                    }
                    assert_eq!(
                        session.state.store.get(side, "owner")?,
                        original,
                        "record must not be deleted or overwritten after failed take",
                    );
                    assert_eq!(
                        session.assigned_partner(side, "owner").as_deref(),
                        Some("receiver"),
                        "failed DELETE must preserve the incumbent pair",
                    );
                    if cap.is_some() {
                        assert!(
                            !session.state.store.contains(side, "later")?,
                            "enabled bulk must stop before the next mutation",
                        );
                        assert!(
                            matches!(
                                session.upsert_record(side, make_record(side, "probe", "180")),
                                Err(SessionError::AssignmentUnhealthy)
                            ),
                            "storage refusal must fail enabled assignment health",
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
