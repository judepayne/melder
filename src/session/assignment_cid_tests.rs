//! Canonical-ID index ownership and stale-entry regressions.

use super::{TestResult, assert_partner, make_fixture, make_record};
use crate::models::{Record, Side};

fn make_cid_record(side: Side, id: &str, common: &str, angle: &str) -> Record {
    let mut record = make_record(side, id, angle);
    record.insert("common".into(), common.into());
    record
}

#[test]
fn changed_or_cleared_common_id_cannot_confirm_old_identity() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            for new_common in ["NEW", "", "   ", "MISSING_FIELD"] {
                let fixture = make_fixture(sqlite, Some(4))?;
                let session = &fixture.session;
                session.upsert_record(side, make_cid_record(side, "changed", " OLD ", "0"))?;
                let mut updated = make_cid_record(side, "changed", new_common, "0");
                if new_common == "MISSING_FIELD" {
                    updated.remove("common");
                }
                session.upsert_record(side, updated)?;
                assert_eq!(
                    session.state.store.common_id_lookup(side, "OLD")?,
                    None,
                    "old common ID must be retired on {side:?}, sqlite={sqlite}"
                );
                // Orthogonal vectors cannot auto-match through ordinary scoring.
                session.upsert_record(
                    side.opposite(),
                    make_cid_record(side.opposite(), "probe", "OLD", "90"),
                )?;
                assert_partner(session, side, "changed", None);
                assert_partner(session, side.opposite(), "probe", None);
            }
        }
    }
    Ok(())
}

#[test]
fn stale_index_entry_must_be_verified_against_current_record() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(4))?;
            let session = &fixture.session;
            session.upsert_record(side, make_cid_record(side, "changed", "NEW", "0"))?;
            // Simulate an old persisted/stale index entry; lookup alone is insufficient.
            session
                .state
                .store
                .common_id_insert(side, "OLD", "changed")?;
            session.upsert_record(
                side.opposite(),
                make_cid_record(side.opposite(), "probe", "OLD", "90"),
            )?;
            assert_partner(session, side, "changed", None);
            assert_partner(session, side.opposite(), "probe", None);
        }
    }
    Ok(())
}

#[test]
fn updating_or_deleting_duplicate_does_not_erase_current_common_id_owner() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            for delete in [false, true] {
                let fixture = make_fixture(sqlite, Some(4))?;
                let session = &fixture.session;
                session.upsert_record(side, make_cid_record(side, "older", "shared", "0"))?;
                session.upsert_record(side, make_cid_record(side, "owner", "shared", "0"))?;
                if delete {
                    session.remove_record(side, "older")?;
                } else {
                    session
                        .upsert_record(side, make_cid_record(side, "older", "different", "0"))?;
                }
                assert_eq!(
                    session.state.store.common_id_lookup(side, "shared")?,
                    Some("owner".into()),
                    "a non-owner must not erase the owner's entry"
                );
                // Canonical matching ignores the intentionally orthogonal embedding.
                session.upsert_record(
                    side.opposite(),
                    make_cid_record(side.opposite(), "probe", " shared ", "90"),
                )?;
                assert_partner(session, side, "owner", Some("probe"));
            }
        }
    }
    Ok(())
}

#[test]
fn deleting_owner_retires_its_common_id_entry() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_fixture(sqlite, Some(4))?;
            let session = &fixture.session;
            session.upsert_record(side, make_cid_record(side, "owner", "shared", "0"))?;
            session.remove_record(side, "owner")?;
            assert_eq!(
                session.state.store.common_id_lookup(side, "shared")?,
                None,
                "deleted owner's index must not survive"
            );
        }
    }
    Ok(())
}
