//! Build pipeline: reads match log + optional scoring log, produces CSVs,
//! Parquet, and/or SQLite DB.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use tracing::{info, warn};

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
    let start = Instant::now();

    // Replay match log events.
    let events = MatchLog::replay(match_log_path)?;

    // Validate assignment ownership before creating any output files.
    let final_pairs = if input.current_reviews.is_some() {
        Some(input.snapshot_pairs.clone())
    } else if events
        .iter()
        .any(|event| matches!(event, MatchLogEvent::AssignmentDelta(_)))
    {
        Some(crate::state::match_log::reconstruct_assignment_history(
            &events,
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

    for event in &events {
        match event {
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
                relationships.retain(|key, rel| {
                    rel.relationship_type != "review"
                        || match review_queries.get(key) {
                            Some(Side::A) => rel.a_id != *a_id,
                            _ => rel.b_id != *b_id,
                        }
                });
                for rel in relationships.values_mut() {
                    if rel.relationship_type == "match" && (rel.a_id == *a_id || rel.b_id == *b_id)
                    {
                        rel.relationship_type = "broken".to_string();
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

        if !candidate_rows.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

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
