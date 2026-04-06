//! Build pipeline: reads match log + optional scoring log, produces CSVs
//! and/or SQLite DB.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
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

/// Build output files from a match log.
///
/// Reads the match log, replays events to build in-memory state, then writes
/// CSVs and/or SQLite DB. This is the single build function called from four
/// places: batch end-of-run, `meld export`, `/admin/flush`, `/admin/shutdown`.
pub fn build_outputs(
    match_log_path: &Path,
    scoring_log_path: Option<&Path>,
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

    // Collect relationships into a sorted vec, tracking seen pairs.
    let relationships_seen: std::collections::HashSet<(String, String)> =
        relationships.keys().cloned().collect();
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

    // Read scoring log if present — enriches output with field_scores and candidates.
    let (field_scores, candidate_rows) =
        if let Some(sl_path) = scoring_log_path.filter(|p| p.exists()) {
            read_scoring_log(sl_path)?
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

        // Write candidates.csv if scoring log provided data.
        if !candidate_rows.is_empty() {
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
        review_count,
        no_match_count: unmatched.len(),
        broken_count,
        warnings,
        elapsed_secs: elapsed,
    })
}

/// Read the scoring log and extract field_scores + candidate rows.
///
/// Handles both plain ndjson and zstd-compressed (.ndjson.zst) files.
fn read_scoring_log(
    path: &Path,
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
                let rank = cand.get("rank").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
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
                if let Some(fscores) = cand.get("field_scores").and_then(|v| v.as_array()) {
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
