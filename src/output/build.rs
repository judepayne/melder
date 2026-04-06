//! Build pipeline: reads match log, produces CSVs and/or SQLite DB.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use tracing::info;

use crate::models::{Record, Side};
use crate::state::match_log::{MatchLog, MatchLogEvent};

use super::manifest::{BuildReport, OutputManifest};

/// A relationship row for the output.
pub struct Relationship {
    pub a_id: String,
    pub b_id: String,
    pub score: Option<f64>,
    pub rank: Option<u8>,
    pub relationship_type: String, // "match" | "review" | "broken"
    pub reason: Option<String>,
}

/// State for tracking unmatched B records with best candidate info.
struct NoMatchInfo {
    best_score: Option<f64>,
    best_a_id: Option<String>,
}

/// Build output files from a match log.
///
/// Reads the match log, replays events to build in-memory state, then writes
/// CSVs and/or SQLite DB. This is the single build function called from four
/// places: batch end-of-run, `meld export`, `/admin/flush`, `/admin/shutdown`.
pub fn build_outputs(
    match_log_path: &Path,
    _scoring_log_path: Option<&Path>, // reserved for Phase 5
    csv_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
) -> Result<BuildReport, Box<dyn std::error::Error>> {
    let start = Instant::now();

    // Replay match log events.
    let events = MatchLog::replay(match_log_path)?;

    // Build in-memory state from events.
    let mut a_records: HashMap<String, Record> = HashMap::new();
    let mut b_records: HashMap<String, Record> = HashMap::new();
    // Keyed by (a_id, b_id) for dedup — later events overwrite earlier.
    let mut relationships: HashMap<(String, String), Relationship> = HashMap::new();
    let mut no_match_map: HashMap<String, NoMatchInfo> = HashMap::new();
    let warnings = Vec::new();

    for event in &events {
        match event {
            MatchLogEvent::UpsertRecord { side, record } => {
                let id_field = match side {
                    Side::A => &manifest.a_id_field,
                    Side::B => &manifest.b_id_field,
                };
                let id = record.get(id_field).cloned().unwrap_or_default();
                if id.is_empty() {
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
                // A review clears no-match status for this B.
                no_match_map.remove(&b_id);
            }
            MatchLogEvent::CrossMapBreak { a_id, b_id } => {
                // Replace the match with a "broken" record if it existed.
                if let Some(rel) = relationships.get_mut(&(a_id.clone(), b_id.clone())) {
                    rel.relationship_type = "broken".to_string();
                }
            }
            MatchLogEvent::NoMatchBelow {
                query_id,
                best_candidate_id,
                best_score,
                ..
            } => {
                no_match_map.insert(
                    query_id.clone(),
                    NoMatchInfo {
                        best_score: *best_score,
                        best_a_id: best_candidate_id.clone(),
                    },
                );
            }
            MatchLogEvent::RemoveRecord { side, id } => match side {
                Side::A => {
                    a_records.remove(id);
                }
                Side::B => {
                    b_records.remove(id);
                    no_match_map.remove(id);
                }
            },
            MatchLogEvent::Exclude { .. } | MatchLogEvent::Unexclude { .. } => {
                // Exclusions don't affect output.
            }
        }
    }

    // Collect relationships into a sorted vec.
    let mut rel_vec: Vec<Relationship> = relationships.into_values().collect();
    rel_vec.sort_by(|a, b| a.b_id.cmp(&b.b_id).then(a.a_id.cmp(&b.a_id)));

    // Derive unmatched B records: those not in any match/review relationship.
    let matched_b_ids: std::collections::HashSet<&str> = rel_vec
        .iter()
        .filter(|r| r.relationship_type == "match" || r.relationship_type == "review")
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

    // Count stats.
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

        info!(
            relationships = match_count + review_count,
            unmatched = unmatched.len(),
            dir = %dir.display(),
            "CSV output written"
        );
    }

    // Write SQLite DB.
    if let Some(db) = db_path {
        if let Some(parent) = db.parent() {
            std::fs::create_dir_all(parent)?;
        }
        super::db::build_db(db, &a_records, &b_records, &rel_vec, manifest)?;

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
        review_count,
        no_match_count: unmatched.len(),
        broken_count,
        warnings,
        elapsed_secs: elapsed,
    })
}
