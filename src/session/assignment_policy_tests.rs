//! Negative deferred-acceptance policy regressions using ordinary production candidates.

use crate::models::{Classification, MatchResult};

use super::{
    Arc, Session, Side, TestResult, assert_metadata, assert_partner, make_bm25_fixture,
    make_fixture, make_record, read_deltas,
};

// Re-score stored records through the ordinary live pipeline, without injecting
// verification IDs, candidates, normalized scores, or classification overrides.
fn make_scores(
    session: &Session,
    side: Side,
    id: &str,
) -> Result<Vec<MatchResult>, Box<dyn std::error::Error>> {
    let state = &session.state;
    let config = &state.config;
    let record = state.store.get(side, id)?.ok_or("missing query record")?;
    let pool_side = side.opposite();
    let blocked = if config.blocking.enabled {
        state.store.blocking_query(&record, side, pool_side)?
    } else {
        state.store.ids(pool_side)?
    };
    let vector = match &state.side(side).combined_index {
        Some(index) => index.get(id)?.ok_or("missing query vector")?,
        None => vec![],
    };
    Ok(session.score_against_pool(
        id,
        &record,
        side,
        &vector,
        &blocked,
        state.side(pool_side),
        pool_side,
        config,
        config.ann_candidates.unwrap_or(50),
        config.bm25_candidates.unwrap_or(10),
        config.top_n.unwrap_or(5),
        &config.synonym_fields,
    ))
}

#[test]
fn one_direction_auto_cannot_claim_a_reciprocally_ineligible_receiver() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let fixture = make_bm25_fixture(sqlite, 0.9)?;
            let session = &fixture.session;
            session.upsert_record(
                side.opposite(),
                make_record(side.opposite(), "receiver", "alpha"),
            )?;
            let response = session.upsert_record(side, make_record(side, "query", "alpha beta"))?;
            let forward = make_scores(session, side, "query")?;
            let reverse = make_scores(session, side.opposite(), "receiver")?;
            assert_eq!(forward.len(), 1, "ordinary BM25 retrieval finds receiver");
            assert_eq!(reverse.len(), 1, "reverse retrieval really finds proposer");
            assert_eq!(forward[0].matched_id, "receiver", "forward candidate");
            assert_eq!(reverse[0].matched_id, "query", "reverse candidate");
            let threshold = session.state.config.thresholds.auto_match;
            assert!(forward[0].score >= threshold, "forward threshold passes");
            assert_eq!(
                forward[0].classification,
                Classification::Auto,
                "forward Auto"
            );
            assert!(
                reverse[0].score < threshold,
                "actual reverse BM25 score {} misses {threshold}; forward={}",
                reverse[0].score,
                forward[0].score,
            );
            assert_eq!(
                reverse[0].classification,
                Classification::Review,
                "reverse review"
            );
            let summary = response.assignment.ok_or("missing assignment summary")?;
            assert!(summary.partner.is_none(), "one-way Auto is not membership");
            assert!(summary.changes.is_empty(), "no assignment transition");
            assert_eq!(summary.displacements, 0, "no displacement");
            assert!(
                !summary.limit_reached,
                "reciprocal rejection is not a cap hit"
            );
            assert_eq!(response.classification, "review", "no false Auto response");
            assert_partner(session, side, "query", None);
            assert_partner(session, side.opposite(), "receiver", None);
            assert!(
                read_deltas(session)?.is_empty(),
                "no accepted-pair WAL event"
            );
            assert_metadata(session)?;
        }
    }
    Ok(())
}

#[test]
fn whole_list_gap_demoted_top_cannot_be_bypassed_by_lower_auto() -> TestResult {
    for sqlite in [false, true] {
        for side in [Side::A, Side::B] {
            let mut fixture = make_fixture(sqlite, Some(4))?;
            let session = Arc::get_mut(&mut fixture.session).expect("unique session");
            Arc::get_mut(&mut session.state)
                .expect("unique live state")
                .config
                .thresholds
                .min_score_gap = Some(0.1);
            for (id, angle) in [("top", "0"), ("lower", "10")] {
                session.upsert_record(side.opposite(), make_record(side.opposite(), id, angle))?;
            }
            let response = session.upsert_record(side, make_record(side, "query", "0"))?;
            let scores = make_scores(session, side, "query")?;
            assert_eq!(scores.len(), 2, "real ANN retrieval retains the whole list");
            assert_eq!(scores[0].matched_id, "top", "highest cosine first");
            assert_eq!(scores[1].matched_id, "lower", "lower cosine second");
            assert!(
                scores[0].score > scores[1].score
                    && scores[1].score >= session.state.config.thresholds.auto_match,
                "both raw scores qualify with a strictly ordered top",
            );
            let gap = session
                .state
                .config
                .thresholds
                .min_score_gap
                .ok_or("missing gap")?;
            assert!(
                scores[0].score - scores[1].score < gap,
                "actual cosine gap is ambiguous"
            );
            assert_eq!(
                scores[0].classification,
                Classification::Review,
                "top demoted"
            );
            assert_eq!(
                scores[0].reason.as_deref(),
                Some("downgraded"),
                "gap evidence"
            );
            assert_eq!(
                scores[1].classification,
                Classification::Auto,
                "lower remains Auto"
            );
            // Both free receivers would accept in isolation: rejection must be
            // the proposer's whole-list gap, not reverse eligibility or occupancy.
            for id in ["top", "lower"] {
                let reverse = make_scores(session, side.opposite(), id)?;
                assert_eq!(reverse.len(), 1, "receiver retrieves the sole proposer");
                assert_eq!(reverse[0].matched_id, "query", "reverse candidate");
                assert_eq!(
                    reverse[0].classification,
                    Classification::Auto,
                    "reverse accepts"
                );
                assert_partner(session, side.opposite(), id, None);
            }
            let summary = response.assignment.ok_or("missing assignment summary")?;
            assert!(
                summary.partner.is_none(),
                "cannot fall through to lower Auto"
            );
            assert!(
                summary.changes.is_empty(),
                "ambiguity produces no accepted step"
            );
            assert_eq!(summary.displacements, 0, "no displacement");
            assert!(!summary.limit_reached, "gap rejection is not a cap hit");
            assert_eq!(
                response.classification, "review",
                "ambiguous response stays review"
            );
            assert_partner(session, side, "query", None);
            assert!(
                read_deltas(session)?.is_empty(),
                "no accepted-pair WAL event"
            );
            assert!(
                session.state.review_queue.iter().any(|entry| {
                    entry.side == side && entry.id == "query" && entry.candidate_id == "top"
                }),
                "review remains owned by the ambiguous top candidate",
            );
            assert_metadata(session)?;
        }
    }
    Ok(())
}
