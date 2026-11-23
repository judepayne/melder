//! Build pipeline: reads match log + optional scoring log, produces CSVs,
//! Parquet, and/or SQLite DB.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use tracing::{info, warn};

use crate::batch::BatchResult;
use crate::crossmap::CrossMapOps;
use crate::models::{Record, Side};
use crate::state::match_log::{MatchLog, MatchLogEvent};
use crate::store::RecordStore;

use super::manifest::{BuildReport, OutputManifest};

/// A relationship row for the output.
pub struct Relationship {
    pub a_id: String,
    pub b_id: String,
    pub score: Option<f64>,
    pub rank: Option<u8>,
    pub relationship_type: String, // "match" | "review" | "broken" | "candidate" | "enroll_edge"
    pub reason: Option<String>,
}

/// A field score row for the output DB.
pub struct FieldScoreRow {
    pub a_id: String,
    pub b_id: String,
    pub field_a: String,
    pub field_b: String,
    pub method: String,
    pub score: f64,
    pub weight: f64,
}

/// A candidate row for candidates.csv.
pub struct CandidateRow {
    pub b_id: String,
    pub rank: u8,
    pub a_id: String,
    pub score: f64,
}

/// State for tracking unmatched B records with best candidate info.
struct NoMatchInfo {
    best_score: Option<f64>,
    best_a_id: Option<String>,
}

/// Original dataset records and a possibly newer crossmap snapshot for replay.
#[derive(Default)]
pub(crate) struct BuildInput {
    pub a_records: HashMap<String, Record>,
    pub b_records: HashMap<String, Record>,
    pub snapshot_pairs: Vec<(String, String)>,
    /// Present for an in-process snapshot taken under the live mutation gate.
    /// In that case membership, records and reviews are authoritative.
    pub current_reviews: Option<Vec<crate::state::live::ReviewEntry>>,
}

/// Whether enrollment records are historical seeds or a complete current pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnrollRecordAuthority {
    SeedAndReplay,
    Snapshot,
}

/// Prevent a live export from replacing its authoritative SQLite database.
pub(crate) fn validate_live_output_path(cfg: &crate::config::Config) -> std::io::Result<()> {
    if let (Some(live), Some(output)) = (&cfg.live.db_path, &cfg.output.db_path) {
        let live = Path::new(live);
        let output = Path::new(output);
        if live == output
            || (live.exists()
                && output.exists()
                && std::fs::canonicalize(live)? == std::fs::canonicalize(output)?)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "output.db_path must not replace live.db_path",
            ));
        }
        #[cfg(unix)]
        if let (Ok(a), Ok(b)) = (std::fs::metadata(live), std::fs::metadata(output)) {
            use std::os::unix::fs::MetadataExt;
            if a.dev() == b.dev() && a.ino() == b.ino() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "output.db_path must not replace live.db_path",
                ));
            }
        }
    }
    Ok(())
}

/// Build output files from a match log.
///
/// Reads the match log, replays events to build in-memory state, then writes
/// CSVs and/or SQLite DB. This is the single build function called from four
/// places: batch end-of-run, `meld export`, `/admin/flush`, `/admin/shutdown`.
pub fn build_outputs(
    match_log_path: &Path,
    scoring_log_path: Option<&Path>,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    build_outputs_with_input(
        match_log_path,
        scoring_log_path,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        BuildInput::default(),
    )
}

/// Build graph batch outputs directly from the store and final CrossMap membership.
///
/// No persistent match/scoring log is required. This materializes all A records,
/// processed B records, and output rows in memory to reuse the same serializers
/// and schema as log-based builds; it is not a streaming SQLite export.
/// Protected CrossMap pairs are retained even when absent from `result.matched`.
pub fn build_batch_graph_outputs(
    store: &dyn RecordStore,
    crossmap: &dyn CrossMapOps,
    result: &BatchResult,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    if result.processed_graph_b.is_none() {
        return Err(
            "direct graph output requires processed_graph_b from a completed graph batch".into(),
        );
    }
    build_batch_outputs(
        store,
        crossmap,
        result,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
    )
}

/// Build batch outputs using final CrossMap membership, for either assignment strategy.
///
/// Greedy CSV callers may retain the legacy CSV writer while using this for other sinks.
pub fn build_batch_outputs(
    store: &dyn RecordStore,
    crossmap: &dyn CrossMapOps,
    result: &BatchResult,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    build_batch_outputs_with_scoring_evidence(
        store,
        crossmap,
        result,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        false,
    )
}

/// Build direct batch outputs with explicit run-associated scoring evidence.
/// When requested, export this run's retained field scores. Greedy CSV keeps its
/// legacy relationship writer; only its companion file is produced here.
#[allow(clippy::too_many_arguments)]
pub fn build_batch_outputs_with_scoring_evidence(
    store: &dyn RecordStore,
    crossmap: &dyn CrossMapOps,
    result: &BatchResult,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    scoring_evidence: bool,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let greedy_processed;
    let processed = if let Some(processed) = &result.processed_graph_b {
        processed
    } else {
        greedy_processed = result
            .matched
            .iter()
            .chain(&result.review)
            .map(|candidate| candidate.query_id.clone())
            .chain(result.unmatched.iter().map(|(id, _, _)| id.clone()))
            .chain(crossmap.pairs().into_iter().map(|(_, b_id)| b_id))
            .collect::<Vec<_>>();
        &greedy_processed
    };
    let processed: std::collections::HashSet<_> = processed.iter().map(String::as_str).collect();
    let mut builder = OutputBuilder {
        export_field_scores: scoring_evidence,
        ..OutputBuilder::default()
    };
    store.for_each_record(Side::A, &mut |id, record| {
        builder.a_records.insert(id.to_owned(), record.clone());
    })?;
    store.for_each_record(Side::B, &mut |id, record| {
        if processed.contains(id) {
            builder.b_records.insert(id.to_owned(), record.clone());
        }
    })?;

    // CrossMap, not the result vectors, is authoritative for accepted pairs.
    let mut relationships: HashMap<_, _> = crossmap
        .pairs()
        .into_iter()
        .map(|(a_id, b_id)| {
            (
                (a_id.clone(), b_id.clone()),
                Relationship {
                    a_id,
                    b_id,
                    score: None,
                    rank: None,
                    relationship_type: "match".into(),
                    reason: None,
                },
            )
        })
        .collect();
    let matched_b: std::collections::HashSet<_> =
        relationships.values().map(|rel| rel.b_id.clone()).collect();
    let mut review_b = std::collections::HashSet::new();
    for (matches, is_review) in [(&result.matched, false), (&result.review, true)] {
        for candidate in matches {
            if !processed.contains(candidate.query_id.as_str()) {
                continue;
            }
            let key = (candidate.matched_id.clone(), candidate.query_id.clone());
            if is_review {
                if matched_b.contains(&candidate.query_id) {
                    continue;
                }
                review_b.insert(candidate.query_id.clone());
                relationships.insert(
                    key.clone(),
                    Relationship {
                        a_id: key.0.clone(),
                        b_id: key.1.clone(),
                        score: None,
                        rank: None,
                        relationship_type: "review".into(),
                        reason: None,
                    },
                );
            }
            let Some(rel) = relationships.get_mut(&key) else {
                continue;
            };
            rel.score = Some(candidate.score);
            rel.rank = candidate.rank;
            rel.reason = candidate.reason.clone();
            builder
                .field_scores
                .extend(candidate.field_scores.iter().map(|field| FieldScoreRow {
                    a_id: key.0.clone(),
                    b_id: key.1.clone(),
                    field_a: field.field_a.clone(),
                    field_b: field.field_b.clone(),
                    method: field.method.clone(),
                    score: field.score,
                    weight: field.weight,
                }));
        }
    }
    let no_match_scores: HashMap<_, _> = result
        .unmatched
        .iter()
        .map(|(id, _, score)| (id.as_str(), *score))
        .collect();
    builder.unmatched = builder
        .b_records
        .iter()
        .filter(|(id, _)| !matched_b.contains(*id) && !review_b.contains(*id))
        .map(|(id, record)| {
            (
                id.clone(),
                record.clone(),
                no_match_scores.get(id.as_str()).copied().flatten(),
                // BatchResult preserves the best score, but not the best candidate ID.
                None,
            )
        })
        .collect();
    builder.unmatched.sort_by(|a, b| a.0.cmp(&b.0));
    builder.rel_vec = relationships.into_values().collect();
    builder
        .rel_vec
        .sort_by(|a, b| a.b_id.cmp(&b.b_id).then(a.a_id.cmp(&b.a_id)));
    // Greedy callers already wrote their legacy relationship CSVs.
    let csv_dir = if scoring_evidence && result.processed_graph_b.is_none() {
        if let Some(dir) = csv_dir {
            std::fs::create_dir_all(dir)?;
            super::csv::write_field_scores_csv(
                &dir.join("field_scores.csv"),
                &builder.field_scores,
            )?;
        }
        None
    } else {
        csv_dir
    };
    builder.write(csv_dir, parquet_dir, db_path, manifest, start)
}

// A simple batch log writes records first, then outcomes for separate B IDs.
// Identical confirmations can repeat (e.g. a pre-match followed by a crossmap
// skip); replay still applies each event to preserve its metadata precedence.
fn batch_events_need_no_relationship_scans(events: &[MatchLogEvent]) -> bool {
    let mut outcomes = HashMap::<&str, Option<&str>>::new();
    let mut matched_a = std::collections::HashSet::new();
    for event in events {
        match event {
            MatchLogEvent::UpsertRecord { .. } if outcomes.is_empty() => {}
            MatchLogEvent::CrossMapConfirm { a_id, b_id, .. } => {
                if let Some(previous) = outcomes.get(b_id.as_str()) {
                    if *previous != Some(a_id.as_str()) {
                        return false;
                    }
                } else {
                    if !matched_a.insert(a_id.as_str()) {
                        return false;
                    }
                    outcomes.insert(b_id, Some(a_id));
                }
            }
            MatchLogEvent::ReviewMatch {
                side: Side::B, id, ..
            }
            | MatchLogEvent::NoMatchBelow {
                query_side: Side::B,
                query_id: id,
                ..
            } => {
                if outcomes.insert(id, None).is_some() {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// Build full outputs from original records, a crossmap snapshot, and WAL history.
///
/// Snapshot pairs may be newer than the WAL's initial state. Assignment history
/// determines touched membership; unrelated snapshot pairs are retained.
pub(crate) fn build_outputs_with_input(
    match_log_path: &Path,
    scoring_log_path: Option<&Path>,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    input: BuildInput,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    build_wal_outputs(
        match_log_path,
        scoring_log_path,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        input,
        EnrollRecordAuthority::SeedAndReplay,
    )
}

/// Build from a specific source of records and the corresponding WAL history.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_wal_outputs(
    match_log_path: &Path,
    scoring_log_path: Option<&Path>,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    input: BuildInput,
    authority: EnrollRecordAuthority,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    build_wal_outputs_with_scoring_evidence(
        match_log_path,
        scoring_log_path,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        input,
        authority,
        false,
    )
}

/// Build a WAL export with explicit batch-only, run-associated scoring evidence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_wal_outputs_with_scoring_evidence(
    match_log_path: &Path,
    scoring_log_path: Option<&Path>,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    input: BuildInput,
    authority: EnrollRecordAuthority,
    batch_scoring_evidence: bool,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    let (events, _) = MatchLog::replay_for_output(match_log_path)?;
    build_events_with_scoring_evidence(
        &events,
        scoring_log_path,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        input,
        authority,
        batch_scoring_evidence,
    )
}

/// Build outputs from an authoritative live snapshot without requiring WAL history.
pub(crate) fn build_live_snapshot_outputs(
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    input: BuildInput,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    build_outputs_from_events_with_authority(
        &[],
        None,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        input,
        EnrollRecordAuthority::Snapshot,
    )
}

/// Build captured output events with explicit enrollment record authority.
///
/// Snapshot records remain authoritative, while events still supply ordered
/// enrollment evidence and invalidations. Seed records instead evolve by replay.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_outputs_from_events_with_authority(
    events: &[MatchLogEvent],
    scoring_log_path: Option<&Path>,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    input: BuildInput,
    authority: EnrollRecordAuthority,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    build_events_with_scoring_evidence(
        events,
        scoring_log_path,
        csv_dir,
        parquet_dir,
        db_path,
        manifest,
        input,
        authority,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_events_with_scoring_evidence(
    events: &[MatchLogEvent],
    scoring_log_path: Option<&Path>,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    input: BuildInput,
    authority: EnrollRecordAuthority,
    batch_scoring_evidence: bool,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    let start = Instant::now();

    if manifest.mode == crate::config::Mode::Enroll {
        return build_enroll_outputs(
            events,
            input,
            authority,
            csv_dir,
            parquet_dir,
            db_path,
            manifest,
            start,
        );
    }

    // Validate assignment ownership before creating any output files.
    let final_pairs = if input.current_reviews.is_some() {
        Some(input.snapshot_pairs.clone())
    } else if events
        .iter()
        .any(|event| matches!(event, MatchLogEvent::AssignmentDelta(_)))
    {
        Some(crate::state::match_log::reconstruct_assignment_history(
            events,
            &input.snapshot_pairs,
        )?)
    } else {
        None
    };

    // Build in-memory state from events.
    let mut a_records = input.a_records;
    let mut b_records = input.b_records;
    // Keyed by (a_id, b_id) for dedup — later events overwrite earlier.
    let mut relationships: HashMap<(String, String), Relationship> = HashMap::new();
    let mut review_queries: HashMap<(String, String), Side> = HashMap::new();
    let mut match_metadata = HashMap::new();
    let mut no_match_map: HashMap<String, NoMatchInfo> = HashMap::new();
    let warnings = Vec::new();

    for (a_id, b_id) in input.snapshot_pairs {
        relationships.insert(
            (a_id.clone(), b_id.clone()),
            Relationship {
                a_id,
                b_id,
                score: None,
                rank: None,
                relationship_type: "match".to_string(),
                reason: None,
            },
        );
    }

    // A graph run is a snapshot, not a sequence of live mutations. Avoid a
    // full relationship scan per terminal event on this validated path.
    let graph_snapshot = input.current_reviews.is_none()
        && relationships.is_empty()
        && matches!(
            events.first(),
            Some(MatchLogEvent::BatchAssignmentStarted { .. })
        )
        && matches!(
            events.last(),
            Some(MatchLogEvent::BatchAssignmentCompleted { .. })
        );
    let skip_relationship_scans = graph_snapshot
        || (relationships.is_empty()
            && input.current_reviews.is_none()
            && batch_events_need_no_relationship_scans(events));
    let mut graph_owned_a = std::collections::HashSet::new();
    let mut graph_terminal_b = std::collections::HashSet::new();
    for event in events {
        if graph_snapshot {
            let terminal_b = match event {
                MatchLogEvent::CrossMapConfirm { a_id, b_id, .. } => {
                    if !graph_owned_a.insert(a_id.as_str()) {
                        return Err("graph assignment log has conflicting A owners".into());
                    }
                    Some(b_id)
                }
                MatchLogEvent::ReviewMatch {
                    side: Side::B, id, ..
                } => Some(id),
                MatchLogEvent::NoMatchBelow {
                    query_side: Side::B,
                    query_id,
                    ..
                } => Some(query_id),
                MatchLogEvent::UpsertRecord { .. }
                | MatchLogEvent::BatchAssignmentStarted { .. }
                | MatchLogEvent::BatchAssignmentCompleted { .. } => None,
                _ => return Err("unexpected event in graph assignment log".into()),
            };
            if let Some(id) = terminal_b
                && !graph_terminal_b.insert(id.as_str())
            {
                return Err("graph assignment log has multiple outcomes for a B record".into());
            }
        }
        match event {
            MatchLogEvent::BatchAssignmentStarted { .. }
            | MatchLogEvent::BatchAssignmentCompleted { .. }
            | MatchLogEvent::EnrollOutcome { .. } => {}
            MatchLogEvent::UpsertRecord { side, record } => {
                let id_field = match side {
                    Side::A => &manifest.a_id_field,
                    Side::B => &manifest.b_id_field,
                };
                let id = record.get(id_field).cloned().unwrap_or_default();
                if id.is_empty() {
                    warn!(
                        id_field,
                        side = ?side,
                        "record missing id field — skipped in output"
                    );
                    continue;
                }
                if !skip_relationship_scans {
                    relationships.retain(|key, rel| {
                        rel.relationship_type != "review"
                            || match review_queries.get(key) {
                                Some(owner) if owner == side => match side {
                                    Side::A => rel.a_id != id,
                                    Side::B => rel.b_id != id,
                                },
                                _ => true,
                            }
                    });
                }
                if *side == Side::B {
                    no_match_map.remove(&id);
                }
                if input.current_reviews.is_some() {
                    continue;
                }
                match side {
                    Side::A => {
                        a_records.insert(id, record.clone());
                    }
                    Side::B => {
                        b_records.insert(id, record.clone());
                    }
                }
            }
            MatchLogEvent::CrossMapConfirm {
                a_id,
                b_id,
                score,
                rank,
                reason,
            } => {
                match_metadata.insert(
                    (a_id.clone(), b_id.clone()),
                    (*score, *rank, reason.clone()),
                );
                if !skip_relationship_scans {
                    relationships.retain(|key, rel| {
                        rel.relationship_type != "review"
                            || match review_queries.get(key) {
                                Some(Side::A) => rel.a_id != *a_id,
                                _ => rel.b_id != *b_id,
                            }
                    });
                    for rel in relationships.values_mut() {
                        if rel.relationship_type == "match"
                            && (rel.a_id == *a_id || rel.b_id == *b_id)
                        {
                            rel.relationship_type = "broken".to_string();
                        }
                    }
                }
                // Remove any prior "broken" entry for this pair.
                relationships.insert(
                    (a_id.clone(), b_id.clone()),
                    Relationship {
                        a_id: a_id.clone(),
                        b_id: b_id.clone(),
                        score: *score,
                        rank: *rank,
                        relationship_type: "match".to_string(),
                        reason: reason.clone(),
                    },
                );
                // A confirmed pair clears any no-match status for this B.
                no_match_map.remove(b_id);
            }
            MatchLogEvent::AssignmentDelta(delta) => {
                for pair in &delta.expected {
                    if let Some(rel) = relationships
                        .get_mut(&(pair.a_id.clone(), pair.b_id.clone()))
                        .filter(|rel| rel.relationship_type == "match")
                    {
                        rel.relationship_type = "broken".to_string();
                    }
                }
                relationships.retain(|key, rel| {
                    rel.relationship_type != "review"
                        || !delta
                            .replacements
                            .iter()
                            .any(|pair| match review_queries.get(key) {
                                Some(Side::A) => rel.a_id == pair.a_id,
                                _ => rel.b_id == pair.b_id,
                            })
                });
                for endpoint in &delta.newly_unmatched {
                    if endpoint.side == Side::B {
                        no_match_map.remove(&endpoint.id);
                    }
                }
                for pair in &delta.replacements {
                    match_metadata.insert(
                        (pair.a_id.clone(), pair.b_id.clone()),
                        (Some(pair.score_b_to_a), pair.rank, pair.reason.clone()),
                    );
                    no_match_map.remove(&pair.b_id);
                    relationships.insert(
                        (pair.a_id.clone(), pair.b_id.clone()),
                        Relationship {
                            a_id: pair.a_id.clone(),
                            b_id: pair.b_id.clone(),
                            score: Some(pair.score_b_to_a),
                            rank: pair.rank,
                            relationship_type: "match".to_string(),
                            reason: pair.reason.clone(),
                        },
                    );
                }
            }
            MatchLogEvent::CrossMapReassign {
                old_a_id,
                old_b_id,
                replacements,
            } => {
                if let Some(rel) = relationships.get_mut(&(old_a_id.clone(), old_b_id.clone())) {
                    rel.relationship_type = "broken".to_string();
                }
                // Resolve reviews owned by now-matched queries. Other queries
                // may still need review even if their candidate just got claimed.
                relationships.retain(|key, rel| {
                    rel.relationship_type != "review"
                        || !replacements
                            .iter()
                            .any(|pair| match review_queries.get(key) {
                                Some(Side::A) => rel.a_id == pair.a_id,
                                _ => rel.b_id == pair.b_id,
                            })
                });
                // This map is keyed by B IDs; A/B identifier namespaces can overlap.
                no_match_map.remove(old_b_id);
                for pair in replacements {
                    match_metadata.insert(
                        (pair.a_id.clone(), pair.b_id.clone()),
                        (Some(pair.score), pair.rank, pair.reason.clone()),
                    );
                    no_match_map.remove(&pair.b_id);
                    relationships.insert(
                        (pair.a_id.clone(), pair.b_id.clone()),
                        Relationship {
                            a_id: pair.a_id.clone(),
                            b_id: pair.b_id.clone(),
                            score: Some(pair.score),
                            rank: pair.rank,
                            relationship_type: "match".to_string(),
                            reason: pair.reason.clone(),
                        },
                    );
                }
            }
            MatchLogEvent::ReviewMatch {
                id,
                side,
                candidate_id,
                score,
                rank,
                reason,
            } => {
                let (a_id, b_id) = match side {
                    Side::A => (id.clone(), candidate_id.clone()),
                    Side::B => (candidate_id.clone(), id.clone()),
                };
                review_queries.insert((a_id.clone(), b_id.clone()), *side);
                relationships.insert(
                    (a_id.clone(), b_id.clone()),
                    Relationship {
                        a_id,
                        b_id: b_id.clone(),
                        score: Some(*score),
                        rank: *rank,
                        relationship_type: "review".to_string(),
                        reason: reason.clone(),
                    },
                );
                // Only a B-owned review clears that query's no-match status.
                if *side == Side::B {
                    no_match_map.remove(&b_id);
                }
            }
            MatchLogEvent::CrossMapBreak { a_id, b_id } => {
                // Replace the match with a "broken" record if it existed.
                if let Some(rel) = relationships.get_mut(&(a_id.clone(), b_id.clone())) {
                    rel.relationship_type = "broken".to_string();
                }
            }
            MatchLogEvent::NoMatchBelow {
                query_id,
                query_side,
                best_candidate_id,
                best_score,
            } => {
                if !skip_relationship_scans {
                    relationships.retain(|key, rel| {
                        rel.relationship_type != "review"
                            || match review_queries.get(key) {
                                Some(owner) if owner == query_side => match query_side {
                                    Side::A => rel.a_id != *query_id,
                                    Side::B => rel.b_id != *query_id,
                                },
                                _ => true,
                            }
                    });
                }
                if *query_side != Side::B {
                    continue;
                }
                no_match_map.insert(
                    query_id.clone(),
                    NoMatchInfo {
                        best_score: *best_score,
                        best_a_id: best_candidate_id.clone(),
                    },
                );
            }
            MatchLogEvent::RemoveRecord { side, id } => {
                relationships.retain(|_, rel| match side {
                    Side::A => rel.a_id != *id,
                    Side::B => rel.b_id != *id,
                });
                if input.current_reviews.is_some() {
                    continue;
                }
                match side {
                    Side::A => {
                        a_records.remove(id);
                    }
                    Side::B => {
                        b_records.remove(id);
                        no_match_map.remove(id);
                    }
                }
            }
            MatchLogEvent::Exclude { .. } | MatchLogEvent::Unexclude { .. } => {
                // Exclusions don't affect output.
            }
        }
    }

    if graph_snapshot
        && b_records
            .keys()
            .any(|id| !graph_terminal_b.contains(id.as_str()))
    {
        return Err("graph assignment log is missing an outcome for a B record".into());
    }

    if let Some(reviews) = input.current_reviews {
        relationships.retain(|_, rel| rel.relationship_type != "review");
        review_queries.clear();
        for review in reviews {
            let (a_id, b_id) = match review.side {
                Side::A => (review.id, review.candidate_id),
                Side::B => (review.candidate_id, review.id),
            };
            let key = (a_id.clone(), b_id.clone());
            review_queries.insert(key.clone(), review.side);
            relationships.insert(
                key,
                Relationship {
                    a_id,
                    b_id,
                    score: Some(review.score),
                    rank: None,
                    relationship_type: "review".into(),
                    reason: None,
                },
            );
        }
    }

    if let Some(pairs) = final_pairs {
        let pairs: std::collections::HashSet<_> = pairs.into_iter().collect();
        let matched_a: std::collections::HashSet<_> =
            pairs.iter().map(|(a, _)| a.as_str()).collect();
        let matched_b: std::collections::HashSet<_> =
            pairs.iter().map(|(_, b)| b.as_str()).collect();
        relationships.retain(|key, rel| {
            rel.relationship_type != "review"
                || match review_queries.get(key) {
                    Some(Side::A) => !matched_a.contains(rel.a_id.as_str()),
                    _ => !matched_b.contains(rel.b_id.as_str()),
                }
        });
        for b_id in &matched_b {
            no_match_map.remove(*b_id);
        }
        for (key, rel) in &mut relationships {
            if rel.relationship_type == "match" && !pairs.contains(key) {
                rel.relationship_type = "broken".to_string();
            }
        }
        // A review or later stale snapshot must not hide authoritative membership.
        // Preserve canonical B-to-A metadata from replay whenever it is available.
        for (a_id, b_id) in pairs {
            let (score, rank, reason) = match_metadata
                .remove(&(a_id.clone(), b_id.clone()))
                .unwrap_or_default();
            let rel = relationships
                .entry((a_id.clone(), b_id.clone()))
                .or_insert(Relationship {
                    a_id,
                    b_id,
                    score: None,
                    rank: None,
                    relationship_type: "match".to_string(),
                    reason: None,
                });
            rel.relationship_type = "match".to_string();
            rel.score = score;
            rel.rank = rank;
            rel.reason = reason;
        }
    }

    // Collect relationships into a sorted vec, tracking seen pairs.
    let relationships_seen: std::collections::HashSet<(String, String)> =
        relationships.keys().cloned().collect();
    let mut rel_vec: Vec<Relationship> = relationships.into_values().collect();
    rel_vec.sort_by(|a, b| a.b_id.cmp(&b.b_id).then(a.a_id.cmp(&b.a_id)));

    // Derive unmatched B records: those not in any match/review relationship.
    let matched_b_ids: std::collections::HashSet<&str> = rel_vec
        .iter()
        .filter(|r| {
            r.relationship_type == "match"
                || (r.relationship_type == "review"
                    && review_queries.get(&(r.a_id.clone(), r.b_id.clone())) != Some(&Side::A))
        })
        .map(|r| r.b_id.as_str())
        .collect();

    let mut unmatched: Vec<(String, Record, Option<f64>, Option<String>)> = b_records
        .iter()
        .filter(|(id, _)| !matched_b_ids.contains(id.as_str()))
        .map(|(id, rec)| {
            let info = no_match_map.get(id);
            (
                id.clone(),
                rec.clone(),
                info.and_then(|i| i.best_score),
                info.and_then(|i| i.best_a_id.clone()),
            )
        })
        .collect();
    unmatched.sort_by(|a, b| a.0.cmp(&b.0));

    // Read scoring log if present — enriches output with field_scores and candidates.
    let (field_scores, candidate_rows) =
        if let Some(sl_path) = scoring_log_path.filter(|p| p.exists()) {
            read_scoring_log_with_fields(sl_path, !(batch_scoring_evidence || graph_snapshot))?
        } else {
            (Vec::new(), Vec::new())
        };

    // Add candidate relationships from scoring log (ranks 2..N).
    for cr in &candidate_rows {
        let key = (cr.a_id.clone(), cr.b_id.clone());
        if !relationships_seen.contains(&key) {
            rel_vec.push(Relationship {
                a_id: cr.a_id.clone(),
                b_id: cr.b_id.clone(),
                score: Some(cr.score),
                rank: Some(cr.rank),
                relationship_type: "candidate".to_string(),
                reason: None,
            });
        }
    }

    OutputBuilder {
        a_records,
        b_records,
        rel_vec,
        unmatched,
        field_scores,
        candidate_rows,
        warnings,
        // Batch evidence files are written together by the streaming exporter below.
        export_field_scores: false,
    }
    .write(csv_dir, parquet_dir, db_path, manifest, start)
    .and_then(|report| {
        if (batch_scoring_evidence || graph_snapshot)
            && let Some(path) = scoring_log_path
        {
            super::shortlist::export(path, csv_dir, parquet_dir, db_path)?;
        }
        Ok(report)
    })
}

// Enrollment has one ID namespace and directed historical evidence, not a
// bijection. Replay it separately so live confirmation/review rules cannot
// accidentally turn qualifying edges into accepted CrossMap membership.
#[allow(clippy::too_many_arguments)]
fn build_enroll_outputs(
    events: &[MatchLogEvent],
    input: BuildInput,
    authority: EnrollRecordAuthority,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
    start: Instant,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    let mut records = input.a_records;
    let mut outcomes = HashMap::new();
    let mut last_change = HashMap::new();
    let mut last_excluded = HashMap::new();
    let authoritative_records = authority == EnrollRecordAuthority::Snapshot;
    for (sequence, event) in events.iter().enumerate() {
        match event {
            MatchLogEvent::UpsertRecord {
                side: Side::A,
                record,
            } => {
                if let Some(id) = record.get(&manifest.a_id_field) {
                    // Check endpoint generations once per retained edge, not once
                    // per previous query for each new record.
                    last_change.insert(id, sequence);
                    outcomes.remove(id);
                    if !authoritative_records {
                        records.insert(id.clone(), record.clone());
                    }
                }
            }
            MatchLogEvent::EnrollOutcome { query_id, edges } => {
                outcomes.insert(query_id.clone(), (sequence, edges));
            }
            MatchLogEvent::RemoveRecord { side: Side::A, id } => {
                last_change.insert(id, sequence);
                outcomes.remove(id);
                if !authoritative_records {
                    records.remove(id);
                }
            }
            MatchLogEvent::Exclude { a_id, b_id } => {
                last_excluded.insert((a_id.as_str(), b_id.as_str()), sequence);
                last_excluded.insert((b_id.as_str(), a_id.as_str()), sequence);
            }
            _ => {}
        }
    }
    let mut builder = OutputBuilder {
        // Both output table namespaces refer to the same single pool.
        a_records: records.clone(),
        b_records: records,
        ..OutputBuilder::default()
    };
    let mut connected = std::collections::HashSet::new();
    for (query_id, (sequence, edges)) in outcomes {
        if !builder.a_records.contains_key(&query_id) {
            continue;
        }
        for (index, edge) in edges.iter().enumerate() {
            if edge.id == query_id
                || !builder.a_records.contains_key(&edge.id)
                || last_change
                    .get(&edge.id)
                    .is_some_and(|last| *last > sequence)
                || last_excluded
                    .get(&(query_id.as_str(), edge.id.as_str()))
                    .is_some_and(|last| *last > sequence)
            {
                continue;
            }
            let rank = (index + 1).min(255) as u8;
            connected.insert(query_id.clone());
            connected.insert(edge.id.clone());
            builder.rel_vec.push(Relationship {
                a_id: edge.id.clone(),
                b_id: query_id.clone(),
                score: Some(edge.score),
                rank: Some(rank),
                relationship_type: "enroll_edge".into(),
                reason: None,
            });
            builder.candidate_rows.push(CandidateRow {
                a_id: edge.id.clone(),
                b_id: query_id.clone(),
                rank,
                score: edge.score,
            });
            builder
                .field_scores
                .extend(edge.field_scores.iter().map(|field| FieldScoreRow {
                    a_id: edge.id.clone(),
                    b_id: query_id.clone(),
                    field_a: field.field.clone(),
                    field_b: field.field.clone(),
                    method: field.method.clone(),
                    score: field.score,
                    weight: field.weight,
                }));
        }
    }
    builder.unmatched = builder
        .b_records
        .iter()
        .filter(|(id, _)| !connected.contains(*id))
        .map(|(id, record)| (id.clone(), record.clone(), None, None))
        .collect();
    builder.unmatched.sort_by(|a, b| a.0.cmp(&b.0));
    builder
        .rel_vec
        .sort_by(|a, b| a.b_id.cmp(&b.b_id).then(a.rank.cmp(&b.rank)));
    builder
        .candidate_rows
        .sort_by(|a, b| a.b_id.cmp(&b.b_id).then(a.rank.cmp(&b.rank)));
    builder.write(csv_dir, parquet_dir, db_path, manifest, start)
}

/// Prepared output state shared by log replay and direct batch graph builds.
#[derive(Default)]
struct OutputBuilder {
    a_records: HashMap<String, Record>,
    b_records: HashMap<String, Record>,
    rel_vec: Vec<Relationship>,
    unmatched: Vec<(String, Record, Option<f64>, Option<String>)>,
    field_scores: Vec<FieldScoreRow>,
    // Explicit provenance: enabled empty runs must still write a schema-only file.
    export_field_scores: bool,
    candidate_rows: Vec<CandidateRow>,
    warnings: Vec<String>,
}

impl OutputBuilder {
    fn write(
        self,
        csv_dir: Option<&Path>,
        parquet_dir: Option<&Path>,
        db_path: Option<&Path>,
        manifest: &OutputManifest,
        start: Instant,
    ) -> Result<BuildReport, Box<dyn std::error::Error>> {
        let Self {
            a_records,
            b_records,
            rel_vec,
            unmatched,
            field_scores,
            export_field_scores,
            candidate_rows,
            warnings,
        } = self;
        let match_count = rel_vec
            .iter()
            .filter(|r| r.relationship_type == "match")
            .count();
        let review_count = rel_vec
            .iter()
            .filter(|r| r.relationship_type == "review")
            .count();
        let broken_count = rel_vec
            .iter()
            .filter(|r| r.relationship_type == "broken")
            .count();

        // Write CSV outputs.
        if let Some(dir) = csv_dir {
            std::fs::create_dir_all(dir)?;

            super::csv::write_relationships_csv(
                &dir.join("relationships.csv"),
                &rel_vec,
                &a_records,
                &manifest.a_fields,
                &manifest.a_id_field,
                &manifest.b_id_field,
            )
            .map_err(|e| format!("relationships.csv: {}", e))?;

            super::csv::write_unmatched_csv(
                &dir.join("unmatched.csv"),
                &unmatched,
                &manifest.b_fields,
                &manifest.b_id_field,
            )
            .map_err(|e| format!("unmatched.csv: {}", e))?;

            // Leave existing files untouched when disabled: a reused directory may
            // contain unrelated artifacts, which are not evidence for this run.
            if export_field_scores && manifest.mode != crate::config::Mode::Enroll {
                super::csv::write_field_scores_csv(&dir.join("field_scores.csv"), &field_scores)
                    .map_err(|e| format!("field_scores.csv: {e}"))?;
            }

            // Write candidates.csv if scoring log provided data.
            if !candidate_rows.is_empty() || manifest.mode == crate::config::Mode::Enroll {
                super::csv::write_candidates_csv(
                    &dir.join("candidates.csv"),
                    &candidate_rows,
                    &manifest.b_id_field,
                    &manifest.a_id_field,
                )
                .map_err(|e| format!("candidates.csv: {}", e))?;
            }

            info!(
                relationships = match_count + review_count,
                unmatched = unmatched.len(),
                field_scores = field_scores.len(),
                dir = %dir.display(),
                "CSV output written"
            );
        }

        // Write Parquet outputs.
        #[cfg(feature = "parquet-format")]
        if let Some(dir) = parquet_dir {
            std::fs::create_dir_all(dir)?;

            super::parquet::write_relationships_parquet(
                &dir.join("relationships.parquet"),
                &rel_vec,
                &a_records,
                &manifest.a_fields,
                &manifest.a_id_field,
                &manifest.b_id_field,
            )
            .map_err(|e| format!("relationships.parquet: {}", e))?;

            super::parquet::write_unmatched_parquet(
                &dir.join("unmatched.parquet"),
                &unmatched,
                &manifest.b_fields,
                &manifest.b_id_field,
            )
            .map_err(|e| format!("unmatched.parquet: {}", e))?;

            if export_field_scores && manifest.mode != crate::config::Mode::Enroll {
                super::parquet::write_field_scores_parquet(
                    &dir.join("field_scores.parquet"),
                    &field_scores,
                )
                .map_err(|e| format!("field_scores.parquet: {e}"))?;
            }

            if !candidate_rows.is_empty() || manifest.mode == crate::config::Mode::Enroll {
                super::parquet::write_candidates_parquet(
                    &dir.join("candidates.parquet"),
                    &candidate_rows,
                    &manifest.b_id_field,
                    &manifest.a_id_field,
                )
                .map_err(|e| format!("candidates.parquet: {}", e))?;
            }

            info!(
                relationships = match_count + review_count,
                unmatched = unmatched.len(),
                field_scores = field_scores.len(),
                dir = %dir.display(),
                "Parquet output written"
            );
        }

        #[cfg(not(feature = "parquet-format"))]
        if parquet_dir.is_some() {
            return Err("parquet output requires the 'parquet-format' feature: cargo build --features parquet-format".into());
        }

        // Write SQLite DB.
        if let Some(db) = db_path {
            if let Some(parent) = db.parent() {
                std::fs::create_dir_all(parent)?;
            }
            super::db::build_db(
                db,
                &a_records,
                &b_records,
                &rel_vec,
                &field_scores,
                manifest,
            )?;

            info!(
                path = %db.display(),
                a_records = a_records.len(),
                b_records = b_records.len(),
                relationships = rel_vec.len(),
                "SQLite DB written"
            );
        }

        let elapsed = start.elapsed().as_secs_f64();
        if !warnings.is_empty() {
            for w in &warnings {
                tracing::warn!(warning = %w, "build warning");
            }
        }

        Ok(BuildReport {
            a_record_count: a_records.len(),
            b_record_count: b_records.len(),
            match_count,
            enroll_edge_count: rel_vec
                .iter()
                .filter(|r| r.relationship_type == "enroll_edge")
                .count(),
            review_count,
            no_match_count: unmatched.len(),
            broken_count,
            warnings,
            elapsed_secs: elapsed,
        })
    }
}

/// Read the scoring log and extract field_scores + candidate rows.
///
/// Handles both plain ndjson and zstd-compressed (.ndjson.zst) files.
fn read_scoring_log_with_fields(
    path: &Path,
    include_fields: bool,
) -> Result<(Vec<FieldScoreRow>, Vec<CandidateRow>), Box<dyn std::error::Error>> {
    let mut field_scores = Vec::new();
    let mut candidates = Vec::new();

    let file = std::fs::File::open(path)?;
    let is_zstd = path.to_string_lossy().ends_with(".zst");

    let reader: Box<dyn std::io::Read> = if is_zstd {
        Box::new(zstd::Decoder::new(file)?)
    } else {
        Box::new(file)
    };
    let buf = BufReader::new(reader);

    for line in buf.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse as generic JSON value to check type field.
        let val: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let ty = val.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if ty == "header" {
            continue; // Skip the header line
        }
        if ty != "scored" {
            continue;
        }

        let query_id = val.get("query_id").and_then(|v| v.as_str()).unwrap_or("");
        let query_side = val
            .get("query_side")
            .and_then(|v| v.as_str())
            .unwrap_or("b");

        if let Some(cands) = val.get("candidates").and_then(|v| v.as_array()) {
            for cand in cands {
                let rank = cand
                    .get("rank")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
                    .min(255) as u8;
                let matched_id = cand
                    .get("matched_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let score = cand.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);

                // Determine a_id and b_id based on query side
                let (a_id, b_id) = match query_side {
                    "a" => (query_id.to_string(), matched_id.to_string()),
                    _ => (matched_id.to_string(), query_id.to_string()),
                };

                // Candidates at rank 2+ go into candidates.csv
                if rank >= 2 {
                    candidates.push(CandidateRow {
                        b_id: b_id.clone(),
                        rank,
                        a_id: a_id.clone(),
                        score,
                    });
                }

                // All candidates get field_scores
                if let Some(fscores) = cand
                    .get("field_scores")
                    .and_then(|v| v.as_array())
                    .filter(|_| include_fields)
                {
                    for fs in fscores {
                        field_scores.push(FieldScoreRow {
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                            field_a: fs
                                .get("field_a")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            field_b: fs
                                .get("field_b")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            method: fs
                                .get("method")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            score: fs.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            weight: fs.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        });
                    }
                }
            }
        }
    }

    info!(
        path = %path.display(),
        field_scores = field_scores.len(),
        candidates = candidates.len(),
        "scoring log read"
    );
    Ok((field_scores, candidates))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enroll_explicit_authority_is_independent_of_reviews()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut manifest = make_graph_manifest();
        manifest.mode = crate::config::Mode::Enroll;
        manifest.b_id_field = manifest.a_id_field.clone();
        manifest.b_fields = manifest.a_fields.clone();
        let events = [serde_json::from_str::<MatchLogEvent>(
            r#"{"type":"upsert_record","side":"a","record":{"aid":"historical","name":"old"}}"#,
        )?];
        for authority in [
            EnrollRecordAuthority::Snapshot,
            EnrollRecordAuthority::SeedAndReplay,
        ] {
            let dir = tempfile::tempdir()?;
            let db = dir.path().join("output.sqlite");
            // Deliberately oppose the legacy signal: explicit authority wins.
            let input = BuildInput {
                a_records: HashMap::from([(
                    "current".into(),
                    Record::from([
                        ("aid".into(), "current".into()),
                        ("name".into(), "new".into()),
                    ]),
                )]),
                current_reviews: (authority == EnrollRecordAuthority::SeedAndReplay).then(Vec::new),
                ..BuildInput::default()
            };
            let report = build_outputs_from_events_with_authority(
                &events,
                None,
                Some(dir.path()),
                None,
                Some(&db),
                &manifest,
                input,
                authority,
            )?;
            let expected = if authority == EnrollRecordAuthority::Snapshot {
                1
            } else {
                2
            };
            assert_eq!(
                report.a_record_count, expected,
                "explicit authority controls replay"
            );
            assert_eq!(report.no_match_count, expected, "isolates remain visible");
            let conn = rusqlite::Connection::open(db)?;
            let count: usize =
                conn.query_row("SELECT count(*) FROM a_records", [], |r| r.get(0))?;
            assert_eq!(count, expected, "SQL uses the same authoritative pool");
        }
        Ok(())
    }

    #[test]
    fn enroll_outputs_preserve_all_edges_and_isolated_records_through_compaction()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::state::match_log::{EnrollEvidence, EnrollEvidenceField};
        let dir = tempfile::tempdir()?;
        let mut manifest = make_graph_manifest();
        manifest.mode = crate::config::Mode::Enroll;
        manifest.b_id_field = manifest.a_id_field.clone();
        manifest.b_fields = manifest.a_fields.clone();
        for detail in [false, true] {
            let log = MatchLog::open(&dir.path().join(format!("enroll-{detail}.ndjson")))?;
            let records: HashMap<_, _> = ["seed", "other", "new", "isolated"]
                .into_iter()
                .map(|id| {
                    (
                        id.to_owned(),
                        Record::from([("aid".into(), id.into()), ("name".into(), id.into())]),
                    )
                })
                .collect();
            // Seed records are supplied by the caller, not manufactured by scoring.
            for id in ["new", "isolated"] {
                log.append_upsert(Side::A, &records[id])?;
                let edges = if id == "new" {
                    ["seed", "other"]
                        .into_iter()
                        .map(|id| EnrollEvidence {
                            id: id.into(),
                            score: 0.8,
                            field_scores: if detail {
                                vec![EnrollEvidenceField {
                                    field: "name".into(),
                                    method: "exact".into(),
                                    score: 0.8,
                                    weight: 1.0,
                                }]
                            } else {
                                Vec::new()
                            },
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                log.append(&MatchLogEvent::EnrollOutcome {
                    query_id: id.into(),
                    edges,
                })?;
            }
            log.flush()?;
            for compact in [false, true] {
                if compact {
                    log.compact("aid", "aid")?;
                }
                let csv_dir = dir.path().join(format!("csv-{detail}-{compact}"));
                let db = dir.path().join(format!("out-{detail}-{compact}.db"));
                let parquet_dir = dir.path().join(format!("parquet-{detail}-{compact}"));
                let report = build_outputs_with_input(
                    log.path(),
                    None,
                    Some(&csv_dir),
                    cfg!(feature = "parquet-format").then_some(parquet_dir.as_path()),
                    Some(&db),
                    &manifest,
                    BuildInput {
                        a_records: records.clone(),
                        ..BuildInput::default()
                    },
                )?;
                assert_eq!(
                    (
                        report.a_record_count,
                        report.b_record_count,
                        report.match_count,
                        report.no_match_count
                    ),
                    (4, 4, 0, 1),
                    "single pool records, no bijective matches, only isolated unmatched"
                );
                #[cfg(feature = "parquet-format")]
                {
                    use parquet::file::reader::{FileReader, SerializedFileReader};
                    for (file, expected) in
                        [("relationships", 2), ("candidates", 2), ("unmatched", 1)]
                    {
                        let reader = SerializedFileReader::new(std::fs::File::open(
                            parquet_dir.join(format!("{file}.parquet")),
                        )?)?;
                        assert_eq!(
                            reader.metadata().file_metadata().num_rows(),
                            expected,
                            "{file} Parquet agrees with single-pool evidence"
                        );
                    }
                }
                let relationships = std::fs::read_to_string(csv_dir.join("relationships.csv"))?;
                assert_eq!(
                    relationships.lines().count(),
                    3,
                    "both qualifying edges including rank 1 survive"
                );
                assert_eq!(
                    relationships.matches("enroll_edge").count(),
                    2,
                    "all relationships are non-bijective enrollment evidence"
                );
                let candidates = std::fs::read_to_string(csv_dir.join("candidates.csv"))?;
                assert_eq!(
                    candidates.lines().count(),
                    3,
                    "candidates includes rank 1 even without scoring log"
                );
                let unmatched = std::fs::read_to_string(csv_dir.join("unmatched.csv"))?;
                assert!(
                    unmatched.contains("isolated") && !unmatched.contains("seed"),
                    "incoming edges connect seeds too"
                );
                let conn = rusqlite::Connection::open(&db)?;
                let fields: i64 =
                    conn.query_row("SELECT COUNT(*) FROM field_scores", [], |row| row.get(0))?;
                assert_eq!(
                    fields,
                    if detail { 2 } else { 0 },
                    "field evidence reflects scoring logging setting"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn enroll_updates_invalidate_evidence_and_never_emit_self_edges()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut manifest = make_graph_manifest();
        manifest.mode = crate::config::Mode::Enroll;
        manifest.b_id_field = "aid".into();
        let log = MatchLog::open(&dir.path().join("enroll.ndjson"))?;
        for id in ["one", "two"] {
            log.append_upsert(Side::A, &Record::from([("aid".into(), id.into())]))?;
        }
        log.append(&serde_json::from_str::<MatchLogEvent>(r#"{"type":"enroll_outcome","query_id":"two","edges":[{"id":"one","score":1.0,"field_scores":[]},{"id":"two","score":1.0,"field_scores":[]}]}"#)?)?;
        log.append_upsert(Side::A, &Record::from([("aid".into(), "one".into())]))?;
        log.flush()?;
        let csv_dir = dir.path().join("csv");
        let report = build_outputs(log.path(), None, Some(&csv_dir), None, None, &manifest)?;
        assert_eq!(
            report.no_match_count, 2,
            "changing either endpoint invalidates stale evidence; self edges never connect"
        );
        assert_eq!(
            std::fs::read_to_string(csv_dir.join("relationships.csv"))?
                .lines()
                .count(),
            1,
            "no stale or self relationship rows"
        );
        Ok(())
    }

    #[test]
    fn batch_fast_replay_preserves_legacy_output_and_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let manifest = make_graph_manifest();
        let events: Vec<MatchLogEvent> = [
            r#"{"type":"upsert_record","side":"a","record":{"aid":"1","name":"first"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"bid":"1","name":"match"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"bid":"2","name":"review"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"bid":"3","name":"below"}}"#,
            r#"{"type":"crossmap_confirm","a_id":"1","b_id":"1","score":1.0,"reason":"exact"}"#,
            r#"{"type":"review_match","side":"b","id":"2","candidate_id":"1","score":0.7,"rank":2}"#,
            r#"{"type":"crossmap_confirm","a_id":"1","b_id":"1","reason":"crossmap"}"#,
            r#"{"type":"no_match_below","query_side":"b","query_id":"3","best_candidate_id":"1","best_score":0.2}"#,
        ].iter().map(|line| serde_json::from_str(line)).collect::<Result<_, _>>()?;
        assert!(batch_events_need_no_relationship_scans(&events));
        let noop = MatchLogEvent::Exclude {
            a_id: "irrelevant".into(),
            b_id: "irrelevant".into(),
        };
        for (name, fallback) in [("fast", false), ("legacy", true)] {
            let log = MatchLog::open(&dir.path().join(format!("{name}.ndjson")))?;
            for event in &events {
                log.append(event)?;
            }
            if fallback {
                log.append(&noop)?;
            }
            log.flush()?;
            let report = build_outputs(
                log.path(),
                None,
                Some(&dir.path().join(name)),
                None,
                None,
                &manifest,
            )?;
            assert_eq!(
                (
                    report.match_count,
                    report.review_count,
                    report.no_match_count,
                    report.broken_count
                ),
                (1, 1, 1, 0)
            );
        }
        for file in ["relationships.csv", "unmatched.csv"] {
            assert_eq!(
                std::fs::read(dir.path().join("fast").join(file))?,
                std::fs::read(dir.path().join("legacy").join(file))?,
                "{file}"
            );
        }
        let csv = std::fs::read_to_string(dir.path().join("fast/relationships.csv"))?;
        assert!(
            csv.contains("crossmap") && !csv.contains(",exact,"),
            "the second confirmation still wins"
        );
        Ok(())
    }

    #[test]
    fn fast_replay_rejects_stateful_or_conflicting_sequences()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = r#"{"type":"crossmap_confirm","a_id":"a","b_id":"b","score":0.9}"#;
        for next in [
            r#"{"type":"crossmap_confirm","a_id":"a","b_id":"other"}"#,
            r#"{"type":"crossmap_confirm","a_id":"other","b_id":"b"}"#,
            r#"{"type":"review_match","side":"a","id":"other","candidate_id":"b","score":0.7}"#,
            r#"{"type":"review_match","side":"b","id":"b","candidate_id":"other","score":0.7}"#,
            r#"{"type":"no_match_below","query_side":"b","query_id":"b"}"#,
            r#"{"type":"crossmap_break","a_id":"a","b_id":"b"}"#,
            r#"{"type":"upsert_record","side":"b","record":{"bid":"b"}}"#,
            r#"{"type":"remove_record","side":"b","id":"b"}"#,
        ] {
            let events = [serde_json::from_str(first)?, serde_json::from_str(next)?];
            assert!(
                !batch_events_need_no_relationship_scans(&events),
                "must replay statefully: {next}"
            );
        }
        Ok(())
    }

    fn make_graph_result() -> BatchResult {
        let candidate = |a: &str, b: &str, score, classification| crate::models::MatchResult {
            query_id: b.into(),
            matched_id: a.into(),
            query_side: Side::B,
            score,
            field_scores: vec![crate::models::FieldScore {
                field_a: "name".into(),
                field_b: "name".into(),
                method: "exact".into(),
                score,
                weight: 1.0,
            }],
            classification,
            matched_record: None,
            from_crossmap: false,
            rank: Some(2),
            reason: Some("graph_assignment".into()),
        };
        BatchResult {
            processed_graph_b: Some(
                ["selected", "protected", "review", "below", "empty"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            ),
            matched: vec![
                candidate("A1", "selected", 0.91, crate::models::Classification::Auto),
                // A stale/non-member result must never become an accepted link.
                candidate("A2", "empty", 0.92, crate::models::Classification::Auto),
            ],
            review: vec![
                candidate("A2", "review", 0.7, crate::models::Classification::Review),
                // Authoritative protected membership suppresses stale reviews.
                candidate(
                    "A2",
                    "protected",
                    0.71,
                    crate::models::Classification::Review,
                ),
                candidate(
                    "A2",
                    "unprocessed",
                    0.72,
                    crate::models::Classification::Review,
                ),
            ],
            unmatched: vec![("below".into(), Record::new(), Some(0.3))],
            stats: crate::batch::BatchStats {
                assignment: None,
                total_b: 5,
                auto_matched: 1,
                review_count: 1,
                no_match: 2,
                skipped: 1,
                elapsed_secs: 0.0,
                scoring_elapsed_secs: 0.0,
            },
        }
    }

    #[test]
    fn batch_field_score_files_require_evidence_and_preserve_unowned_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let csv_dir = dir.path().join("csv");
        let db = dir.path().join("output.sqlite");
        let manifest = make_graph_manifest();
        let make_builder = |enabled, empty| OutputBuilder {
            export_field_scores: enabled,
            field_scores: if empty {
                vec![]
            } else {
                vec![FieldScoreRow {
                    a_id: "a,quoted".into(),
                    b_id: "b".into(),
                    field_a: "name".into(),
                    field_b: "label".into(),
                    method: "exact".into(),
                    score: 0.987654321,
                    weight: 2.5,
                }]
            },
            ..OutputBuilder::default()
        };
        make_builder(false, false).write(
            Some(&csv_dir),
            None,
            Some(&db),
            &manifest,
            Instant::now(),
        )?;
        let path = csv_dir.join("field_scores.csv");
        assert!(
            !path.exists(),
            "nonempty SQLite detail is not opt-in evidence"
        );
        make_builder(true, false).write(
            Some(&csv_dir),
            None,
            Some(&db),
            &manifest,
            Instant::now(),
        )?;
        let rows = csv::Reader::from_path(&path)?
            .records()
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(rows.len(), 1, "one field score exported");
        let conn = rusqlite::Connection::open(&db)?;
        let expected: (String, String, String, String, String, f64, f64) = conn.query_row(
            "SELECT a_id,b_id,field_a,field_b,method,score,weight FROM field_scores",
            [],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            },
        )?;
        assert_eq!(
            (
                &rows[0][0],
                &rows[0][1],
                &rows[0][2],
                &rows[0][3],
                &rows[0][4]
            ),
            (
                expected.0.as_str(),
                expected.1.as_str(),
                expected.2.as_str(),
                expected.3.as_str(),
                expected.4.as_str()
            ),
            "CSV identity matches SQLite"
        );
        assert_eq!(
            rows[0][5].parse::<f64>()?,
            expected.5,
            "score precision matches SQLite"
        );
        assert_eq!(
            rows[0][6].parse::<f64>()?,
            expected.6,
            "weight matches SQLite"
        );
        std::fs::write(&path, "unrelated artifact")?;
        make_builder(false, true).write(Some(&csv_dir), None, None, &manifest, Instant::now())?;
        assert_eq!(
            std::fs::read_to_string(&path)?,
            "unrelated artifact",
            "disabled runs do not delete or rewrite unrelated files"
        );
        make_builder(true, true).write(Some(&csv_dir), None, None, &manifest, Instant::now())?;
        assert_eq!(
            csv::Reader::from_path(&path)?.records().count(),
            0,
            "enabled empty run replaces stale rows with schema-only output"
        );
        Ok(())
    }

    #[cfg(feature = "parquet-format")]
    #[test]
    fn empty_batch_parquet_uses_evidence_not_rows() -> Result<(), Box<dyn std::error::Error>> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let dir = tempfile::tempdir()?;
        let manifest = make_graph_manifest();
        let path = dir.path().join("field_scores.parquet");
        OutputBuilder::default().write(None, Some(dir.path()), None, &manifest, Instant::now())?;
        assert!(!path.exists(), "disabled empty run creates no field file");
        OutputBuilder {
            export_field_scores: true,
            ..OutputBuilder::default()
        }
        .write(None, Some(dir.path()), None, &manifest, Instant::now())?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path)?)?;
        assert_eq!(
            reader.schema().fields().len(),
            7,
            "empty file retains field score schema"
        );
        assert_eq!(
            reader.metadata().file_metadata().num_rows(),
            0,
            "empty evidence has no rows"
        );
        Ok(())
    }

    #[test]
    fn direct_greedy_field_scores_leave_legacy_csvs_untouched()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let store =
            crate::store::memory::MemoryStore::new(&crate::config::BlockingConfig::default());
        let crossmap = crate::crossmap::memory::MemoryCrossMap::new();
        let mut result = make_graph_result();
        result.processed_graph_b = None;
        let legacy = dir.path().join("relationships.csv");
        std::fs::write(&legacy, "legacy output")?;
        build_batch_outputs_with_scoring_evidence(
            &store,
            &crossmap,
            &result,
            Some(dir.path()),
            None,
            None,
            &make_graph_manifest(),
            true,
        )?;
        assert!(
            dir.path().join("field_scores.csv").exists(),
            "greedy evidence creates field file"
        );
        assert_eq!(
            std::fs::read_to_string(legacy)?,
            "legacy output",
            "legacy relationships retained"
        );
        Ok(())
    }

    #[test]
    fn wal_field_score_export_needs_run_evidence_even_with_scoring_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let scoring = dir.path().join("scoring.ndjson");
        std::fs::write(&scoring, "{\"type\":\"header\"}\n")?;
        let manifest = make_graph_manifest();
        for (evidence, scoring_path, expected) in [
            (false, Some(scoring.as_path()), false),
            (true, None, false),
            (true, Some(scoring.as_path()), true),
        ] {
            let out = tempfile::tempdir()?;
            build_events_with_scoring_evidence(
                &[],
                scoring_path,
                Some(out.path()),
                None,
                None,
                &manifest,
                BuildInput::default(),
                EnrollRecordAuthority::SeedAndReplay,
                evidence,
            )?;
            assert_eq!(
                out.path().join("field_scores.csv").exists(),
                expected,
                "file presence follows associated evidence, not scoring rows"
            );
        }
        Ok(())
    }

    fn make_graph_manifest() -> OutputManifest {
        OutputManifest {
            mode: crate::config::Mode::Match,
            job_name: "direct graph".into(),
            a_id_field: "aid".into(),
            b_id_field: "bid".into(),
            a_fields: vec!["name".into()],
            b_fields: vec!["name".into()],
            auto_match: 0.85,
            review_floor: 0.6,
            min_score_gap: None,
            top_n: 5,
            model: String::new(),
        }
    }

    #[test]
    fn direct_graph_outputs_share_schema_and_authoritative_pairs_for_both_stores() {
        for sqlite in [false, true] {
            let dir = tempfile::tempdir().expect("tempdir");
            let blocking = crate::config::BlockingConfig::default();
            let (store, crossmap): (Box<dyn RecordStore>, Box<dyn CrossMapOps>) = if sqlite {
                let (store, crossmap, _) = crate::store::sqlite::open_sqlite(
                    &dir.path().join("state.sqlite"),
                    &blocking,
                    None,
                    &["aid".into(), "name".into()],
                    &["bid".into(), "name".into()],
                )
                .expect("SQLite input");
                (Box::new(store), Box::new(crossmap))
            } else {
                (
                    Box::new(crate::store::memory::MemoryStore::new(&blocking)),
                    Box::new(crate::crossmap::MemoryCrossMap::new()),
                )
            };
            for (side, id_field, ids) in [
                (Side::A, "aid", vec!["A1", "A2", "A3", "A4"]),
                (
                    Side::B,
                    "bid",
                    vec![
                        "selected",
                        "protected",
                        "review",
                        "below",
                        "empty",
                        "unprocessed",
                        "outside-protected",
                    ],
                ),
            ] {
                for id in ids {
                    store
                        .insert(
                            side,
                            id,
                            &Record::from([
                                (id_field.into(), id.into()),
                                ("name".into(), format!("record {id}")),
                            ]),
                        )
                        .expect("insert record");
                }
            }
            for (a, b) in [
                ("A1", "selected"),
                ("A3", "protected"),
                ("A4", "outside-protected"),
            ] {
                assert!(crossmap.claim(a, b), "install accepted pair");
            }
            let csv_dir = dir.path().join("csv");
            let db = dir.path().join("output.sqlite");
            let parquet_dir = cfg!(feature = "parquet-format").then(|| dir.path().join("parquet"));
            let report = build_batch_outputs_with_scoring_evidence(
                store.as_ref(),
                crossmap.as_ref(),
                &make_graph_result(),
                Some(&csv_dir),
                parquet_dir.as_deref(),
                Some(&db),
                &make_graph_manifest(),
                true,
            )
            .expect("direct output without logs");
            assert_eq!(
                (report.a_record_count, report.b_record_count),
                (4, 5),
                "only processed B records are materialized"
            );
            assert_eq!(
                (
                    report.match_count,
                    report.review_count,
                    report.no_match_count,
                    report.broken_count
                ),
                (3, 1, 2, 0),
                "membership and final outcomes"
            );
            let conn = rusqlite::Connection::open(&db).expect("output DB");
            let accepted: Vec<(String, String)> = conn.prepare(
                "SELECT a_id, b_id FROM relationships WHERE relationship_type = 'match' ORDER BY a_id"
            ).expect("prepare").query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .expect("query").collect::<Result<_, _>>().expect("rows");
            let mut expected = crossmap.pairs();
            expected.sort();
            assert_eq!(
                accepted, expected,
                "all accepted and protected pairs match CrossMap"
            );
            let selected: (f64, u8, String) = conn
                .query_row(
                    "SELECT score, rank, reason FROM relationships WHERE b_id = 'selected'",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .expect("selected metadata");
            assert_eq!(
                selected,
                (0.91, 2, "graph_assignment".into()),
                "selected score/rank/reason preserved"
            );
            let fields: Vec<(String, String, String, String, String, f64, f64)> = conn.prepare(
                "SELECT a_id, b_id, field_a, field_b, method, score, weight FROM field_scores ORDER BY b_id"
            ).expect("prepare").query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)))
                .expect("query").collect::<Result<_, _>>().expect("rows");
            assert_eq!(
                fields,
                vec![
                    (
                        "A2".into(),
                        "review".into(),
                        "name".into(),
                        "name".into(),
                        "exact".into(),
                        0.7,
                        1.0
                    ),
                    (
                        "A1".into(),
                        "selected".into(),
                        "name".into(),
                        "name".into(),
                        "exact".into(),
                        0.91,
                        1.0
                    ),
                ],
                "available scores for final selected and review pairs only"
            );
            let exported = csv::Reader::from_path(csv_dir.join("field_scores.csv"))
                .expect("field scores CSV")
                .records()
                .collect::<Result<Vec<_>, _>>()
                .expect("field rows");
            let mut csv_fields = exported
                .iter()
                .map(|r| {
                    (
                        r[0].to_owned(),
                        r[1].to_owned(),
                        r[2].to_owned(),
                        r[3].to_owned(),
                        r[4].to_owned(),
                        r[5].parse::<f64>().expect("score"),
                        r[6].parse::<f64>().expect("weight"),
                    )
                })
                .collect::<Vec<_>>();
            csv_fields.sort_by(|a, b| a.1.cmp(&b.1));
            assert_eq!(
                csv_fields, fields,
                "CSV contains exactly SQLite field detail"
            );
            #[cfg(feature = "parquet-format")]
            {
                use arrow::array::{Float64Array, StringArray};
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
                let file = std::fs::File::open(
                    parquet_dir
                        .as_ref()
                        .expect("parquet dir")
                        .join("field_scores.parquet"),
                )
                .expect("field parquet");
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                    .expect("reader")
                    .build()
                    .expect("batches");
                let mut parquet_fields = Vec::new();
                for batch in reader {
                    let batch = batch.expect("batch");
                    for row in 0..batch.num_rows() {
                        let text = |col| {
                            batch
                                .column(col)
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .expect("text")
                                .value(row)
                                .to_owned()
                        };
                        let number = |col| {
                            batch
                                .column(col)
                                .as_any()
                                .downcast_ref::<Float64Array>()
                                .expect("number")
                                .value(row)
                        };
                        parquet_fields.push((
                            text(0),
                            text(1),
                            text(2),
                            text(3),
                            text(4),
                            number(5),
                            number(6),
                        ));
                    }
                }
                parquet_fields.sort_by(|a, b| a.1.cmp(&b.1));
                assert_eq!(
                    parquet_fields, fields,
                    "Parquet matches SQLite field detail"
                );
            }
            let b_ids: Vec<String> = conn
                .prepare("SELECT id FROM b_records ORDER BY id")
                .expect("prepare")
                .query_map([], |r| r.get(0))
                .expect("query")
                .collect::<Result<_, _>>()
                .expect("rows");
            assert_eq!(
                b_ids,
                ["below", "empty", "protected", "review", "selected"],
                "limit never leaks extra B records"
            );
            let below_name: String = conn
                .query_row("SELECT name FROM b_records WHERE id = 'below'", [], |r| {
                    r.get(0)
                })
                .expect("B record");
            assert_eq!(
                below_name, "record below",
                "records come from authoritative store"
            );
            let mut reader =
                csv::Reader::from_path(csv_dir.join("relationships.csv")).expect("CSV");
            assert_eq!(
                reader.headers().expect("headers"),
                &csv::StringRecord::from(vec![
                    "bid",
                    "aid",
                    "score",
                    "relationship_type",
                    "reason",
                    "rank",
                    "name"
                ]),
                "shared CSV schema"
            );
            let rows = reader
                .records()
                .collect::<Result<Vec<_>, _>>()
                .expect("CSV rows");
            let mut accepted_csv: Vec<_> = rows
                .iter()
                .filter(|r| &r[3] == "match")
                .map(|r| (r[1].to_string(), r[0].to_string()))
                .collect();
            accepted_csv.sort();
            assert_eq!(accepted_csv, expected, "CSV accepted pairs match CrossMap");
            let unmatched = csv::Reader::from_path(csv_dir.join("unmatched.csv"))
                .expect("CSV")
                .records()
                .collect::<Result<Vec<_>, _>>()
                .expect("unmatched rows");
            assert_eq!(
                unmatched,
                vec![
                    csv::StringRecord::from(vec!["below", "record below", "0.3000", ""]),
                    csv::StringRecord::from(vec!["empty", "record empty", "", ""]),
                ],
                "no-match scores and empty candidate outcomes survive without logs"
            );
            #[cfg(feature = "parquet-format")]
            {
                use arrow::array::StringArray;
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
                let file = std::fs::File::open(
                    parquet_dir
                        .as_ref()
                        .expect("Parquet dir")
                        .join("relationships.parquet"),
                )
                .expect("Parquet");
                let batches = ParquetRecordBatchReaderBuilder::try_new(file)
                    .expect("reader")
                    .build()
                    .expect("build reader");
                let mut accepted_parquet = Vec::new();
                for batch in batches {
                    let batch = batch.expect("batch");
                    let strings = |index| {
                        batch
                            .column(index)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .expect("string column")
                    };
                    for row in 0..batch.num_rows() {
                        if strings(3).value(row) == "match" {
                            accepted_parquet.push((
                                strings(1).value(row).to_owned(),
                                strings(0).value(row).to_owned(),
                            ));
                        }
                    }
                }
                accepted_parquet.sort();
                assert_eq!(
                    accepted_parquet, expected,
                    "Parquet accepted pairs match CrossMap"
                );
            }
        }
    }

    #[test]
    fn direct_graph_outputs_require_explicit_processed_b_scope() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store =
            crate::store::memory::MemoryStore::new(&crate::config::BlockingConfig::default());
        let crossmap = crate::crossmap::MemoryCrossMap::new();
        let mut result = make_graph_result();
        result.processed_graph_b = None;
        let output = dir.path().join("outputs");
        assert!(
            build_batch_graph_outputs(
                &store,
                &crossmap,
                &result,
                Some(&output),
                None,
                None,
                &make_graph_manifest()
            )
            .is_err(),
            "missing graph scope must fail rather than export unprocessed B"
        );
        assert!(!output.exists(), "invalid graph scope creates no output");
        result.processed_graph_b = Some(Vec::new());
        let report = build_batch_graph_outputs(
            &store,
            &crossmap,
            &result,
            Some(&output),
            None,
            None,
            &make_graph_manifest(),
        )
        .expect("empty graph batch");
        assert_eq!(
            (
                report.b_record_count,
                report.match_count,
                report.no_match_count
            ),
            (0, 0, 0),
            "empty graph scope remains empty"
        );
    }

    #[test]
    fn invalid_graph_input_never_creates_output_targets() {
        for contents in [
            None,
            Some(""),
            Some("{\"type\":\"batch_assignment_sta"),
            Some("{\"type\":\"no_match_below\",\"query_side\":\"b\",\"query_id\":\"B\"}\n"),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("job.graph.match_log.ndjson");
            if let Some(contents) = contents {
                std::fs::write(&path, contents).expect("input");
            }
            let csv = dir.path().join("csv");
            let parquet = dir.path().join("parquet");
            let db = dir.path().join("db/output.sqlite");
            assert!(
                build_outputs(
                    &path,
                    None,
                    Some(&csv),
                    Some(&parquet),
                    Some(&db),
                    &make_graph_manifest()
                )
                .is_err(),
                "invalid graph input: {contents:?}"
            );
            assert!(
                !csv.exists() && !parquet.exists() && !db.exists(),
                "validation precedes all output creation"
            );
        }
    }

    #[test]
    fn batch_assignment_build_requires_completion_before_any_output() {
        let manifest = OutputManifest {
            mode: crate::config::Mode::Match,
            job_name: "test".into(),
            a_id_field: "id".into(),
            b_id_field: "id".into(),
            a_fields: vec![],
            b_fields: vec![],
            auto_match: 0.85,
            review_floor: 0.6,
            min_score_gap: None,
            top_n: 5,
            model: String::new(),
        };
        for snapshot in [false, true] {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("matches.ndjson");
            let log = MatchLog::open(&path).expect("log");
            for line in [
                r#"{"type":"batch_assignment_started","mode":"approximate"}"#,
                r#"{"type":"upsert_record","side":"b","record":{"id":"B1"}}"#,
                r#"{"type":"upsert_record","side":"b","record":{"id":"B2"}}"#,
                r#"{"type":"crossmap_confirm","a_id":"A1","b_id":"B1","score":0.91,"rank":2}"#,
                r#"{"type":"no_match_below","query_side":"b","query_id":"B2","best_score":0.2}"#,
            ] {
                log.append(&serde_json::from_str(line).expect("event"))
                    .expect("append");
            }
            log.flush().expect("flush prefix");
            let csv_dir = dir.path().join("csv");
            let parquet_dir = dir.path().join("parquet");
            let db = dir.path().join("db/output.sqlite");
            let input = BuildInput {
                current_reviews: snapshot.then(Vec::new),
                snapshot_pairs: vec![("A1".into(), "B1".into())],
                ..BuildInput::default()
            };
            assert!(
                build_outputs_with_input(
                    &path,
                    None,
                    Some(&csv_dir),
                    Some(&parquet_dir),
                    Some(&db),
                    &manifest,
                    input,
                )
                .is_err(),
                "incomplete batch must fail even with current snapshot={snapshot}"
            );
            assert!(
                !csv_dir.exists() && !parquet_dir.exists() && !db.exists(),
                "no output target is created before framing validation"
            );
            log.append(&MatchLogEvent::BatchAssignmentCompleted {
                retained_edges: 1,
                fixed_pairs: 0,
                scale: crate::matching::assignment::SCALE,
                exact_groups: 1,
                approximate_groups: 0,
                largest_group: 2,
                objective_units: 910_000_000,
                upper_units: 910_000_000,
            })
            .expect("completion");
            log.flush().expect("flush completion");
            let report = build_outputs(&path, None, Some(&csv_dir), None, Some(&db), &manifest)
                .expect("complete build");
            assert_eq!(
                (report.match_count, report.no_match_count),
                (1, 1),
                "batch markers preserve confirm and no-match replay"
            );
            let conn = rusqlite::Connection::open(&db).expect("DB");
            let accepted: (String, String, f64, u8) = conn.query_row(
                "SELECT a_id, b_id, score, rank FROM relationships WHERE relationship_type = 'match'",
                [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            ).expect("accepted pair");
            assert_eq!(
                accepted,
                ("A1".into(), "B1".into(), 0.91, 2),
                "selected pair and score metadata survive framing"
            );
        }
    }

    #[test]
    fn greedy_admin_snapshot_is_authoritative_over_stale_wal() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("wal.ndjson");
        let log = MatchLog::open(&path).expect("WAL");
        for event in [
            MatchLogEvent::UpsertRecord {
                side: Side::B,
                record: Record::from([("bid".into(), "deleted".into())]),
            },
            MatchLogEvent::CrossMapConfirm {
                a_id: "A".into(),
                b_id: "deleted".into(),
                score: Some(0.99),
                rank: None,
                reason: None,
            },
            MatchLogEvent::RemoveRecord {
                side: Side::B,
                id: "current".into(),
            },
        ] {
            log.append(&event).expect("append");
        }
        log.flush().expect("flush");
        let input = BuildInput {
            a_records: HashMap::from([("A".into(), Record::from([("aid".into(), "A".into())]))]),
            b_records: HashMap::from([(
                "current".into(),
                Record::from([
                    ("bid".into(), "current".into()),
                    ("name".into(), "Latest".into()),
                ]),
            )]),
            snapshot_pairs: vec![("A".into(), "current".into())],
            current_reviews: Some(vec![]),
        };
        let db = dir.path().join("output.db");
        let report = build_outputs_with_input(
            &path,
            None,
            Some(dir.path()),
            None,
            Some(&db),
            &make_graph_manifest(),
            input,
        )
        .expect("admin snapshot build");
        assert_eq!(
            (
                report.a_record_count,
                report.b_record_count,
                report.match_count
            ),
            (1, 1, 1),
            "current snapshot records and pair are authoritative"
        );
        let conn = rusqlite::Connection::open(db).expect("output DB");
        let pair: String = conn
            .query_row(
                "SELECT b_id FROM relationships WHERE relationship_type='match'",
                [],
                |row| row.get(0),
            )
            .expect("accepted pair");
        assert_eq!(
            pair, "current",
            "historical confirmation cannot restore deleted membership"
        );
        let name: String = conn
            .query_row("SELECT name FROM b_records", [], |row| row.get(0))
            .expect("current record");
        assert_eq!(
            name, "Latest",
            "historical removal cannot erase current records"
        );
    }

    #[test]
    fn snapshot_build_retains_untouched_pairs_and_original_records() {
        for snapshot_b in ["B1", "B2"] {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("matches.ndjson");
            let log = MatchLog::open(&path).expect("log");
            for line in [
                r#"{"type":"no_match_below","query_side":"b","query_id":"shared","best_candidate_id":"old","best_score":0.4}"#,
                r#"{"type":"review_match","side":"a","id":"A3","candidate_id":"shared","score":0.7}"#,
                r#"{"type":"review_match","side":"b","id":"B3","candidate_id":"A3","score":0.7}"#,
                r#"{"type":"upsert_record","side":"b","record":{"id":"B3","name":"updated"}}"#,
                r#"{"type":"assignment_delta","version":1,"attempt_id":3,"step_index":0,"proposer":{"side":"b","id":"B2"},"expected":[{"a_id":"A1","b_id":"B1"}],"replacements":[{"a_id":"A1","b_id":"B2","score_b_to_a":0.91,"score_a_to_b":0.95,"rank":2,"reason":"embedding"}],"newly_unmatched":[{"side":"b","id":"B1"}],"previous_receiver_score":0.8}"#,
            ] {
                log.append(&serde_json::from_str(line).expect("event"))
                    .expect("append");
            }
            log.flush().expect("flush");
            let records = |ids: &[&str]| {
                ids.iter()
                    .map(|id| {
                        (
                            (*id).to_string(),
                            Record::from([
                                ("id".into(), (*id).into()),
                                ("name".into(), format!("original {id}")),
                            ]),
                        )
                    })
                    .collect()
            };
            let input = BuildInput {
                a_records: records(&["A1", "A3", "untouched-a"]),
                b_records: records(&["B1", "B2", "B3", "shared", "untouched-b"]),
                snapshot_pairs: vec![
                    ("A1".into(), snapshot_b.into()),
                    ("untouched-a".into(), "untouched-b".into()),
                ],
                current_reviews: None,
            };
            let manifest = OutputManifest {
                mode: crate::config::Mode::Match,
                job_name: "test".into(),
                a_id_field: "id".into(),
                b_id_field: "id".into(),
                a_fields: vec!["name".into()],
                b_fields: vec!["name".into()],
                auto_match: 0.85,
                review_floor: 0.6,
                min_score_gap: None,
                top_n: 5,
                model: String::new(),
            };
            let db = dir.path().join("output.db");
            let parquet = cfg!(feature = "parquet-format").then_some(dir.path());
            let report = build_outputs_with_input(
                &path,
                None,
                Some(dir.path()),
                parquet,
                Some(&db),
                &manifest,
                input,
            )
            .expect("snapshot-aware build");
            assert_eq!(
                (
                    report.a_record_count,
                    report.b_record_count,
                    report.match_count,
                    report.review_count,
                    report.no_match_count
                ),
                (3, 5, 2, 1, 3),
                "full records and untouched membership survive snapshots of either age"
            );
            let conn = rusqlite::Connection::open(&db).expect("DB");
            let score: (f64, u8, String) = conn.query_row(
                "SELECT score, rank, reason FROM relationships WHERE a_id='A1' AND b_id='B2' AND relationship_type='match'",
                [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            ).expect("canonical pair");
            assert_eq!(score, (0.91, 2, "embedding".into()), "B-to-A metadata");
            let untouched: i64 = conn.query_row(
                "SELECT count(*) FROM relationships WHERE a_id='untouched-a' AND b_id='untouched-b' AND relationship_type='match'",
                [], |row| row.get(0),
            ).expect("untouched pair");
            assert_eq!(untouched, 1, "unrelated preloaded pair included in DB");
            let mut csv = csv::Reader::from_path(dir.path().join("unmatched.csv")).expect("CSV");
            let rows: Vec<HashMap<String, String>> =
                csv.deserialize().map(|r| r.expect("row")).collect();
            let shared = rows
                .iter()
                .find(|r| r.get("id").map(String::as_str) == Some("shared"))
                .expect("A-owned review does not own B unmatched status");
            assert_eq!(
                shared.get("best_score").map(String::as_str),
                Some("0.4000"),
                "A review cannot clear B no-match metadata"
            );
            #[cfg(feature = "parquet-format")]
            {
                let reader =
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        std::fs::File::open(dir.path().join("relationships.parquet"))
                            .expect("Parquet"),
                    )
                    .expect("Parquet reader")
                    .build()
                    .expect("batches");
                let rows: usize = reader.map(|batch| batch.expect("batch").num_rows()).sum();
                assert_eq!(
                    rows, 3,
                    "Parquet includes both matches and the surviving review"
                );
            }
        }
    }

    #[test]
    fn assignment_build_handles_preloaded_orphan_reviews_and_removal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("matches.ndjson");
        let log = MatchLog::open(&path).expect("log");
        for line in [
            r#"{"type":"upsert_record","side":"a","record":{"id":"A1"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"id":"B1"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"id":"B2"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"id":"B3"}}"#,
            r#"{"type":"review_match","side":"b","id":"B1","candidate_id":"A1","score":0.7}"#,
            r#"{"type":"review_match","side":"b","id":"B2","candidate_id":"other","score":0.7}"#,
            r#"{"type":"assignment_delta","version":1,"attempt_id":3,"step_index":0,"proposer":{"side":"b","id":"B2"},"expected":[{"a_id":"A1","b_id":"B1"}],"replacements":[{"a_id":"A1","b_id":"B2","score_b_to_a":0.91,"score_a_to_b":0.95,"rank":null,"reason":null}],"newly_unmatched":[{"side":"b","id":"B1"}],"previous_receiver_score":0.8}"#,
            r#"{"type":"review_match","side":"b","id":"B3","candidate_id":"A1","score":0.7}"#,
        ] {
            log.append(&serde_json::from_str(line).expect("event"))
                .expect("append");
        }
        log.flush().expect("flush");
        let manifest = OutputManifest {
            mode: crate::config::Mode::Match,
            job_name: "test".into(),
            a_id_field: "id".into(),
            b_id_field: "id".into(),
            a_fields: vec![],
            b_fields: vec![],
            auto_match: 0.85,
            review_floor: 0.6,
            min_score_gap: None,
            top_n: 5,
            model: String::new(),
        };
        let db = dir.path().join("output.db");
        let report = build_outputs(&path, None, Some(dir.path()), None, Some(&db), &manifest)
            .expect("build");
        assert_eq!(
            (report.match_count, report.review_count),
            (1, 2),
            "retain orphan and unrelated candidate-pointing reviews"
        );
        let conn = rusqlite::Connection::open(&db).expect("output DB");
        let pair: (String, String, f64) = conn
            .query_row(
                "SELECT a_id, b_id, score FROM relationships WHERE relationship_type = 'match'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("accepted SQL pair");
        assert_eq!(
            pair,
            ("A1".into(), "B2".into(), 0.91),
            "SQLite output uses the same accepted pair and primary score"
        );
        let mut csv = csv::Reader::from_path(dir.path().join("relationships.csv")).expect("CSV");
        let rows: Vec<HashMap<String, String>> =
            csv.deserialize().map(|r| r.expect("row")).collect();
        let accepted = rows
            .iter()
            .find(|row| row.get("relationship_type").map(String::as_str) == Some("match"))
            .expect("match row");
        assert_eq!(
            accepted.get("score").map(String::as_str),
            Some("0.9100"),
            "output uses canonical B-to-A score"
        );
        assert_eq!(
            accepted.get("rank").map(String::as_str),
            Some(""),
            "verification-only rank remains absent"
        );
        log.append(&MatchLogEvent::RemoveRecord {
            side: Side::B,
            id: "B2".into(),
        })
        .expect("remove");
        log.flush().expect("flush remove");
        let removed =
            build_outputs(&path, None, None, None, None, &manifest).expect("rebuild after remove");
        assert_eq!(
            removed.match_count, 0,
            "record removal also removes accepted relationship"
        );
    }

    #[test]
    fn reassignment_build_preserves_metadata_and_clears_stale_status() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("matches.ndjson");
        let log = MatchLog::open(&path).expect("open log");
        for line in [
            r#"{"type":"upsert_record","side":"b","record":{"id":"B1"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"id":"B2"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"id":"A1"}}"#,
            r#"{"type":"upsert_record","side":"b","record":{"id":"B3"}}"#,
            r#"{"type":"no_match_below","query_side":"b","query_id":"A1","best_candidate_id":"A9","best_score":0.45}"#,
            r#"{"type":"crossmap_confirm","a_id":"A1","b_id":"B1","score":0.8}"#,
            r#"{"type":"review_match","side":"b","id":"B2","candidate_id":"A9","score":0.7}"#,
            r#"{"type":"review_match","side":"a","id":"A2","candidate_id":"B9","score":0.7}"#,
            r#"{"type":"review_match","side":"b","id":"B3","candidate_id":"A2","score":0.7}"#,
            r#"{"type":"no_match_below","query_side":"b","query_id":"B2","best_score":0.2}"#,
            r#"{"type":"crossmap_reassign","old_a_id":"A1","old_b_id":"B1","replacements":[{"a_id":"A1","b_id":"B2","score":0.91,"rank":2},{"a_id":"A2","b_id":"B1","score":0.87,"rank":3,"reason":"exact"}]}"#,
        ] {
            log.append(&serde_json::from_str(line).expect("event"))
                .expect("append");
        }
        log.flush().expect("flush");
        let manifest = OutputManifest {
            mode: crate::config::Mode::Match,
            job_name: "test".into(),
            a_id_field: "id".into(),
            b_id_field: "id".into(),
            a_fields: vec![],
            b_fields: vec![],
            auto_match: 0.85,
            review_floor: 0.6,
            min_score_gap: None,
            top_n: 5,
            model: String::new(),
        };
        let report =
            build_outputs(&path, None, Some(dir.path()), None, None, &manifest).expect("build");
        assert_eq!(
            (
                report.match_count,
                report.broken_count,
                report.review_count,
                report.no_match_count
            ),
            (2, 1, 1, 1),
            "resolve only matched queries; unrelated review and B-side A1 remain"
        );
        let mut unmatched =
            csv::Reader::from_path(dir.path().join("unmatched.csv")).expect("unmatched CSV");
        let unmatched_rows: Vec<_> = unmatched
            .deserialize::<std::collections::HashMap<String, String>>()
            .map(|r| r.expect("unmatched row"))
            .collect();
        assert_eq!(
            unmatched_rows.len(),
            1,
            "only unrelated B-side A1 is unmatched"
        );
        assert_eq!(
            unmatched_rows[0].get("best_score").map(String::as_str),
            Some("0.4500"),
            "A-side endpoint must not clear another B record's score"
        );
        let mut reader = csv::Reader::from_path(dir.path().join("relationships.csv")).expect("CSV");
        let headers = reader.headers().expect("headers").clone();
        let index = |name: &str| headers.iter().position(|h| h == name).expect("column");
        let rows: Vec<_> = reader.records().map(|r| r.expect("row")).collect();
        for (a, b, score, rank, reason) in [
            ("A1", "B2", 0.91, "2", ""),
            ("A2", "B1", 0.87, "3", "exact"),
        ] {
            let row = rows
                .iter()
                .find(|r| &r[0] == b && &r[1] == a)
                .expect("replacement row");
            assert_eq!(
                row[index("score")].parse::<f64>().expect("score"),
                score,
                "own score"
            );
            assert_eq!(&row[index("rank")], rank, "own rank");
            assert_eq!(&row[index("reason")], reason, "own reason");
        }
    }
}
