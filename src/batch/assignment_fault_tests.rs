//! Fault coverage for the graph publisher against both CrossMap implementations.

use std::io::{self, Write};

use crate::config::BlockingConfig;
use crate::crossmap::MemoryCrossMap;
use crate::error::CrossMapError;
use crate::matching::assignment::SCALE;
use crate::state::match_log::validate_batch_assignment_framing;

use super::*;

#[derive(Clone, Copy, Debug)]
enum Fault {
    PublishRefused,
    PublishError,
    OutcomeAppend,
    CompletionWrite,
    CompletionSync,
    RollbackRefused,
    RollbackError,
}

#[test]
fn failed_publication_preserves_fixed_pairs_and_cannot_be_exported()
-> Result<(), Box<dyn std::error::Error>> {
    for sqlite in [false, true] {
        for fault in [
            Fault::PublishRefused,
            Fault::PublishError,
            Fault::OutcomeAppend,
            Fault::CompletionWrite,
            Fault::CompletionSync,
            Fault::RollbackRefused,
            Fault::RollbackError,
        ] {
            let dir = tempfile::tempdir()?;
            let cm: Box<dyn CrossMapOps> = if sqlite {
                let (_, cm, _) = crate::store::sqlite::open_sqlite(
                    &dir.path().join("state.db"),
                    &BlockingConfig::default(),
                    None,
                    &["aid".into()],
                    &["bid".into()],
                )?;
                Box::new(cm)
            } else {
                Box::new(MemoryCrossMap::new())
            };
            cm.add("fixed-a", "fixed-b")?;
            let log = MatchLog::open(&dir.path().join("batch.ndjson"))?;
            log.append(&MatchLogEvent::BatchAssignmentStarted {
                mode: BatchAssignmentMode::Exact,
            })?;
            log.flush()?;
            let mut writes = 0;
            let err = publish_changes(
                &[("new-a", "new-b"), ("new-a2", "new-b2")],
                |expected, replacements| match (expected.is_empty(), fault) {
                    (true, Fault::PublishRefused) | (false, Fault::RollbackRefused) => Ok(false),
                    (true, Fault::PublishError) | (false, Fault::RollbackError) => Err(
                        CrossMapError::Io(io::Error::other("injected backend error")),
                    ),
                    _ => cm.apply_assignment_delta(expected, replacements),
                },
                || {
                    writes += 1;
                    log.append(&MatchLogEvent::CrossMapConfirm {
                        a_id: "new-a".into(),
                        b_id: "new-b".into(),
                        score: Some(1.0),
                        rank: Some(1),
                        reason: Some("graph_assignment".into()),
                    })?;
                    match fault {
                        Fault::CompletionWrite | Fault::CompletionSync => log
                            .seal_batch_assignment(
                                &MatchLogEvent::BatchAssignmentCompleted {
                                    retained_edges: 2,
                                    fixed_pairs: 1,
                                    scale: SCALE,
                                    exact_groups: 2,
                                    approximate_groups: 0,
                                    largest_group: 2,
                                    objective_units: 2 * i128::from(SCALE),
                                    upper_units: 2 * i128::from(SCALE),
                                },
                                |mut file, bytes| {
                                    // Model a partial write as well as a sync error after the entire marker is visible.
                                    file.write_all(if matches!(fault, Fault::CompletionWrite) {
                                        &bytes[..bytes.len() / 2]
                                    } else {
                                        bytes
                                    })?;
                                    Err(io::Error::other("injected completion failure"))
                                },
                            )?,
                        _ => {
                            return Err(io::Error::other(
                                "injected failure appending another outcome",
                            )
                            .into());
                        }
                    }
                    Ok(())
                },
            )
            .expect_err("all injected paths must fail");
            let publish_failed = matches!(fault, Fault::PublishRefused | Fault::PublishError);
            let rollback_failed = matches!(fault, Fault::RollbackRefused | Fault::RollbackError);
            assert_eq!(
                writes,
                usize::from(!publish_failed),
                "no log publication after backend refusal: {fault:?}"
            );
            assert_eq!(
                matches!(
                    err,
                    MelderError::BatchAssignment(BatchAssignmentError::Rollback { .. })
                ),
                rollback_failed,
                "both errors must be reported: {err}"
            );
            assert_eq!(
                cm.get_b("fixed-a").as_deref(),
                Some("fixed-b"),
                "fixed membership survives: {fault:?}"
            );
            assert_eq!(
                cm.len(),
                if rollback_failed { 3 } else { 1 },
                "unpublished links rolled back when possible: {fault:?}"
            );
            log.flush()?;
            let events = MatchLog::replay(log.path())?;
            assert_eq!(events.len(), if publish_failed { 1 } else { 2 });
            assert!(
                validate_batch_assignment_framing(&events).is_err(),
                "failed publication cannot be exported: {fault:?}"
            );
        }
    }
    Ok(())
}
