//! `meld export` — export live-mode state to configured output formats.
//!
//! Reads from SQLite (if `live.db_path` is set and the DB exists) or
//! reconstructs state from CSV datasets and WAL replay. Does not require
//! the server to be running.

use std::collections::HashMap;
use std::path::Path;
use std::process;

use crate::config::{Config, EnrollJobConfig};
use crate::crossmap::traits::CrossMapOps;
use crate::models::{Record, Side};
use crate::state::match_log::{MatchLog, MatchLogEvent};

// --- Entry point -------------------------------------------------------------

/// Export live-mode state to CSV files in `out_dir`.
///
/// Produces four files: results.csv, review.csv, unmatched_a.csv,
/// unmatched_b.csv. Reads from SQLite if `live.db_path` is set and the DB
/// file exists; otherwise reconstructs state from CSV datasets + WAL replay.
pub fn cmd_export(config_path: &Path, out_dir: &Path) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(match_error) => match crate::config::load_enroll_job_config(config_path) {
            Ok(c) => {
                export_enroll(&c, out_dir);
                return;
            }
            Err(enroll_error) => {
                eprintln!(
                    "Config error (match): {match_error}\nConfig error (enroll): {enroll_error}"
                );
                process::exit(1);
            }
        },
    };

    if let Err(e) = crate::output::build::validate_live_output_path(&cfg) {
        eprintln!("Export failed: {e}");
        process::exit(1);
    }

    if let Err(error) = crate::config::validation::validate_live_outputs(&cfg, Some(out_dir)) {
        eprintln!("Export failed: {error}");
        process::exit(1);
    }

    // Guard: batch configs have neither match_log_path nor db_path configured.
    if cfg.live.match_log_path.is_none() && cfg.live.db_path.is_none() {
        eprintln!(
            "Error: meld export is for live mode. \
             Batch mode already writes output files directly via meld run."
        );
        process::exit(1);
    }

    // Batch history is self-contained: never combine it with today's live DB,
    // datasets or CrossMap. Historical greedy logs use the batch filename namespace.
    let greedy_path_selected = cfg
        .live
        .match_log_path
        .as_deref()
        .is_some_and(|path| is_greedy_batch_log_path(Path::new(path)));
    let graph_path_selected = cfg
        .live
        .match_log_path
        .as_deref()
        .is_some_and(|path| crate::state::match_log::is_graph_match_log_path(Path::new(path)));
    let batch_path_selected = graph_path_selected || greedy_path_selected;
    if let Some(db_path) = cfg
        .live
        .db_path
        .as_deref()
        .map(Path::new)
        .filter(|p| p.exists() && !batch_path_selected)
    {
        create_export_dir(out_dir);
        export_sqlite(&cfg, db_path, out_dir);
        return;
    }

    // Try the new build pipeline if a match log path is configured.
    // Also produce a SQLite DB if output.db_path is set.
    if let Some(ref ml_path) = cfg.live.match_log_path {
        let ml = std::path::Path::new(ml_path);
        let (_, graph_segment) = MatchLog::replay_for_output(ml).unwrap_or_else(|e| {
            eprintln!("Export failed: {e}");
            process::exit(1);
        });
        let input = if graph_segment.is_none() && !greedy_path_selected {
            match load_build_input(&cfg) {
                Ok(input) => input,
                Err(e) => {
                    eprintln!("Failed to load export snapshot: {}", e);
                    process::exit(1);
                }
            }
        } else {
            crate::output::build::BuildInput::default()
        };
        let manifest = crate::output::OutputManifest::from_config(&cfg);
        let parquet_dir = cfg.output.parquet_dir_path.as_deref().map(Path::new);
        let db_path = cfg.output.db_path.as_deref().map(Path::new);
        let sl_path = if let Some(segment) = graph_segment.as_deref() {
            crate::output::scoring_log::graph_scoring_path(segment).unwrap_or_else(|e| {
                eprintln!("Failed to select graph scoring log: {e}");
                process::exit(1);
            })
        } else if cfg.scoring_log.enabled && !greedy_path_selected {
            let sl_dir = cfg.output.csv_dir_path.as_deref().unwrap_or(".");
            let sl_base = Path::new(sl_dir).join(format!("{}.scoring_log", cfg.job.name));
            let use_zstd = cfg.scoring_log.compression != "none";
            let actual = if use_zstd {
                std::path::PathBuf::from(format!("{}.ndjson.zst", sl_base.display()))
            } else {
                std::path::PathBuf::from(format!("{}.ndjson", sl_base.display()))
            };
            if actual.exists() { Some(actual) } else { None }
        } else {
            None
        };
        match crate::output::build::build_outputs_with_input(
            graph_segment.as_deref().unwrap_or(ml),
            sl_path.as_deref(),
            Some(out_dir),
            parquet_dir,
            db_path,
            &manifest,
            input,
        ) {
            Ok(report) => {
                println!("Export complete (build pipeline):");
                println!(
                    "  relationships.csv: {} matches + {} review",
                    report.match_count, report.review_count
                );
                println!("  unmatched.csv:     {} records", report.no_match_count);
                if let Some(db) = db_path {
                    println!("  database:          {}", db.display());
                }
                println!("  elapsed:           {:.1}s", report.elapsed_secs);
                // Retain the legacy four-CSV interface for live match exports.
                if graph_segment.is_none() && !greedy_path_selected {
                    export_memory(&cfg, out_dir);
                }
                return;
            }
            Err(e) => {
                eprintln!("Export failed: {}", e);
                process::exit(1);
            }
        }
    }

    // No WAL selected and no existing live SQLite database.
    create_export_dir(out_dir);
    let input = load_build_input(&cfg).unwrap_or_else(|e| {
        eprintln!("Failed to load export snapshot: {e}");
        process::exit(1);
    });
    crate::output::build::build_live_snapshot_outputs(
        Some(out_dir),
        cfg.output.parquet_dir_path.as_deref().map(Path::new),
        cfg.output.db_path.as_deref().map(Path::new),
        &crate::output::OutputManifest::from_config(&cfg),
        input,
    )
    .unwrap_or_else(|e| {
        eprintln!("Export failed: {e}");
        process::exit(1);
    });
    export_memory(&cfg, out_dir);
}

fn load_enroll_build_input(
    cfg: &EnrollJobConfig,
) -> Result<crate::output::build::BuildInput, Box<dyn std::error::Error>> {
    let a_records = if cfg.dataset.path.is_empty() {
        HashMap::new()
    } else {
        crate::data::load_dataset(
            Path::new(&cfg.dataset.path),
            &cfg.dataset.id_field,
            &cfg.required_fields,
            cfg.dataset.format.as_deref(),
        )?
        .0
    };
    Ok(crate::output::build::BuildInput {
        a_records,
        ..Default::default()
    })
}

fn export_enroll(cfg: &EnrollJobConfig, out_dir: &Path) {
    let Some(log_path) = cfg.live.match_log_path.as_deref() else {
        eprintln!("Error: enroll export requires live.match_log_path");
        process::exit(1);
    };
    let input = load_enroll_build_input(cfg).unwrap_or_else(|e| {
        eprintln!("Failed to load export snapshot: {e}");
        process::exit(1);
    });
    let manifest = crate::output::OutputManifest::from_enroll_job(cfg);
    let scoring = crate::output::manifest::enroll_scoring_log_path(cfg);
    let report = crate::output::build::build_outputs_with_input(
        Path::new(log_path),
        scoring.as_deref(),
        Some(out_dir),
        cfg.output.parquet_dir_path.as_deref().map(Path::new),
        cfg.output.db_path.as_deref().map(Path::new),
        &manifest,
        input,
    )
    .unwrap_or_else(|e| {
        eprintln!("Export failed: {e}");
        process::exit(1);
    });
    println!("Export complete (build pipeline):");
    println!(
        "  relationships.csv: {} matches + {} review",
        report.match_count, report.review_count
    );
    println!("  Enrollment edges: {}", report.enroll_edge_count);
    println!("  unmatched.csv:     {} records", report.no_match_count);
    if let Some(db) = &cfg.output.db_path {
        println!("  database:          {db}");
    }
    println!("  elapsed:           {:.1}s", report.elapsed_secs);
}

fn create_export_dir(out_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!(
            "Failed to create output directory {}: {e}",
            out_dir.display()
        );
        process::exit(1);
    }
}

// Batch run historically wrote <job>.match_log[_<timestamp>[_<sequence>]].ndjson.
// Do not infer batch intent from the modern assignment config (live uses it too).
fn is_greedy_batch_log_path(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    stem.ends_with(".match_log") || stem.contains(".match_log_")
}

fn load_build_input(
    cfg: &Config,
) -> Result<crate::output::build::BuildInput, Box<dyn std::error::Error>> {
    let a_records = crate::data::load_dataset(
        Path::new(&cfg.datasets.a.path),
        &cfg.datasets.a.id_field,
        &cfg.required_fields_a,
        cfg.datasets.a.format.as_deref(),
    )?
    .0;
    let b_records = crate::data::load_dataset(
        Path::new(&cfg.datasets.b.path),
        &cfg.datasets.b.id_field,
        &cfg.required_fields_b,
        cfg.datasets.b.format.as_deref(),
    )?
    .0;
    let crossmap_path = Path::new(cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv"));
    let snapshot_pairs = if crossmap_path.exists() {
        crate::crossmap::MemoryCrossMap::load(
            crossmap_path,
            &cfg.cross_map.a_id_field,
            &cfg.cross_map.b_id_field,
        )?
        .pairs()
    } else {
        Vec::new()
    };
    Ok(crate::output::build::BuildInput {
        a_records,
        b_records,
        snapshot_pairs,
        current_reviews: None,
    })
}

// --- SQLite path -------------------------------------------------------------

fn export_sqlite(cfg: &Config, db_path: &Path, out_dir: &Path) {
    let mut database = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Failed to open SQLite database {}: {}",
                db_path.display(),
                e
            );
            process::exit(1);
        }
    };

    // One read transaction keeps all four CSVs on the same committed snapshot.
    // Assignment membership commits before derived flags/reviews, so export must
    // use CrossMap directly rather than requiring a server restart to repair them.
    let conn = database.transaction().unwrap_or_else(|e| {
        eprintln!("Failed to begin SQLite export snapshot: {e}");
        process::exit(1);
    });

    // results.csv — confirmed crossmap pairs.
    let pairs: Vec<(String, String)> = {
        let mut stmt = match conn.prepare("SELECT a_id, b_id FROM crossmap ORDER BY a_id") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to query crossmap: {}", e);
                process::exit(1);
            }
        };
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap_or_else(|e| {
                eprintln!("Failed to query crossmap: {}", e);
                process::exit(1);
            })
            .filter_map(|r| r.ok())
            .collect()
    };
    write_results_or_exit(
        &out_dir.join("results.csv"),
        &pairs,
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    );

    // review.csv — pending review-band matches.
    let reviews: Vec<ReviewRow> = {
        let mut stmt = match conn.prepare(
            "SELECT id, side, candidate_id, score FROM reviews r \
                 WHERE (r.side = 'a' AND NOT EXISTS \
                    (SELECT 1 FROM crossmap c WHERE c.a_id = r.id)) \
                    OR (r.side = 'b' AND NOT EXISTS \
                    (SELECT 1 FROM crossmap c WHERE c.b_id = r.id)) \
                 ORDER BY score DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to query reviews: {}", e);
                process::exit(1);
            }
        };
        stmt.query_map([], |row| {
            Ok(ReviewRow {
                id: row.get(0)?,
                side: row.get(1)?,
                candidate_id: row.get(2)?,
                score: row.get(3)?,
            })
        })
        .unwrap_or_else(|e| {
            eprintln!("Failed to query reviews: {}", e);
            process::exit(1);
        })
        .filter_map(|r| r.ok())
        .collect()
    };
    write_review_or_exit(&out_dir.join("review.csv"), &reviews);

    // unmatched_a.csv — A-side records with no crossmap pair.
    let unmatched_a = query_unmatched(&conn, "a", &cfg.required_fields_a);
    write_unmatched_or_exit(
        &out_dir.join("unmatched_a.csv"),
        &unmatched_a,
        &cfg.datasets.a.id_field,
    );

    // unmatched_b.csv — B-side records with no crossmap pair.
    let unmatched_b = query_unmatched(&conn, "b", &cfg.required_fields_b);
    write_unmatched_or_exit(
        &out_dir.join("unmatched_b.csv"),
        &unmatched_b,
        &cfg.datasets.b.id_field,
    );

    // Read records from the same transaction as membership/reviews. Never replay
    // a potentially stale WAL over SQLite's authoritative committed state.
    let mut a_records: HashMap<_, _> =
        query_records_or_exit(&conn, "a", false, &cfg.required_fields_a)
            .into_iter()
            .collect();
    let mut b_records: HashMap<_, _> =
        query_records_or_exit(&conn, "b", false, &cfg.required_fields_b)
            .into_iter()
            .collect();
    for (id, record) in &mut a_records {
        record.insert(cfg.datasets.a.id_field.clone(), id.clone());
    }
    for (id, record) in &mut b_records {
        record.insert(cfg.datasets.b.id_field.clone(), id.clone());
    }
    let input = crate::output::build::BuildInput {
        a_records,
        b_records,
        snapshot_pairs: pairs.clone(),
        current_reviews: Some(
            reviews
                .iter()
                .map(|review| crate::state::live::ReviewEntry {
                    id: review.id.clone(),
                    side: if review.side == "a" { Side::A } else { Side::B },
                    candidate_id: review.candidate_id.clone(),
                    score: review.score,
                })
                .collect(),
        ),
    };
    let manifest = crate::output::OutputManifest::from_config(cfg);
    crate::output::build::build_live_snapshot_outputs(
        Some(out_dir),
        cfg.output.parquet_dir_path.as_deref().map(Path::new),
        cfg.output.db_path.as_deref().map(Path::new),
        &manifest,
        input,
    )
    .unwrap_or_else(|e| {
        eprintln!("Export failed: {e}");
        process::exit(1);
    });

    print_summary(
        "SQLite",
        out_dir,
        &pairs,
        &reviews,
        &unmatched_a,
        &unmatched_b,
    );
}

/// Query unmatched records for the given side ("a" or "b").
///
/// Reads the table schema dynamically (columnar storage — no JSON blob).
fn query_unmatched(
    conn: &rusqlite::Connection,
    side: &str,
    fields: &[String],
) -> Vec<(String, Record)> {
    query_records_or_exit(conn, side, true, fields)
}

fn query_records_or_exit(
    conn: &rusqlite::Connection,
    side: &str,
    unmatched: bool,
    fields: &[String],
) -> Vec<(String, Record)> {
    query_records(conn, side, unmatched, fields).unwrap_or_else(|e| {
        eprintln!("Failed to query {side} records: {e}");
        process::exit(1);
    })
}

fn query_records(
    conn: &rusqlite::Connection,
    side: &str,
    unmatched: bool,
    fields: &[String],
) -> rusqlite::Result<Vec<(String, Record)>> {
    use crate::store::sqlite::{quote_identifier, record_column_name};
    assert!(
        side == "a" || side == "b",
        "query_unmatched: side must be \"a\" or \"b\", got \"{side}\""
    );
    // Reverse the store's mapping using the logical schema. An alias may itself
    // be a source name, so decoding by prefix alone would lose payload fields.
    let mapping: HashMap<_, _> = fields
        .iter()
        .map(|field| (record_column_name(field, fields), field.clone()))
        .collect();
    if mapping.len() != fields.len() {
        return Err(rusqlite::Error::InvalidQuery);
    }
    let columns: Vec<String> = {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({side}_records)"))?;
        stmt.query_map([], |row| row.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .filter(|name| name != "id")
            .collect()
    };
    let logical_columns: Vec<_> = columns
        .iter()
        .map(|name| {
            mapping
                .get(name)
                .ok_or_else(|| rusqlite::Error::InvalidColumnName(name.clone()))
        })
        .collect::<rusqlite::Result<_>>()?;

    let col_list = if columns.is_empty() {
        "r.id".to_string()
    } else {
        format!(
            "r.id, {}",
            columns
                .iter()
                .map(|c| format!("r.{}", quote_identifier(c)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let filter = if unmatched {
        format!("WHERE NOT EXISTS (SELECT 1 FROM crossmap c WHERE c.{side}_id = r.id)")
    } else {
        String::new()
    };
    let sql = format!("SELECT {col_list} FROM {side}_records r {filter} ORDER BY r.id");
    let mut stmt = conn.prepare(&sql)?;
    stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let mut record = Record::new();
        for (i, col_name) in logical_columns.iter().enumerate() {
            let val: Option<String> = row.get(i + 1)?;
            record.insert((*col_name).clone(), val.unwrap_or_default());
        }
        Ok((id, record))
    })?
    .collect()
}

// --- In-memory WAL replay path -----------------------------------------------

fn export_memory(cfg: &Config, out_dir: &Path) {
    let a_id_field = &cfg.datasets.a.id_field;
    let b_id_field = &cfg.datasets.b.id_field;

    // Seed record maps from CSV datasets.
    let mut record_map_a: HashMap<String, Record> = match crate::data::load_dataset(
        Path::new(&cfg.datasets.a.path),
        a_id_field,
        &cfg.required_fields_a,
        cfg.datasets.a.format.as_deref(),
    ) {
        Ok((records, _)) => records,
        Err(e) => {
            eprintln!("Warning: failed to load A dataset: {}", e);
            HashMap::new()
        }
    };

    let mut record_map_b: HashMap<String, Record> = match crate::data::load_dataset(
        Path::new(&cfg.datasets.b.path),
        b_id_field,
        &cfg.required_fields_b,
        cfg.datasets.b.format.as_deref(),
    ) {
        Ok((records, _)) => records,
        Err(e) => {
            eprintln!("Warning: failed to load B dataset: {}", e);
            HashMap::new()
        }
    };

    // Seed crossmap from CSV.
    let crossmap_path = cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv");
    let mut crossmap_a_to_b: HashMap<String, String> = HashMap::new();
    let mut crossmap_b_to_a: HashMap<String, String> = HashMap::new();
    if let Ok(mem_cm) = crate::crossmap::MemoryCrossMap::load(
        Path::new(crossmap_path),
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        for (a_id, b_id) in mem_cm.pairs() {
            crossmap_b_to_a.insert(b_id.clone(), a_id.clone());
            crossmap_a_to_b.insert(a_id, b_id);
        }
    }

    // Review map: key = "{side}:{id}:{candidate_id}".
    let mut review_map: HashMap<String, ReviewRow> = HashMap::new();

    // Replay WAL events, applying them to all four maps.
    if let Some(wal_base) = &cfg.live.match_log_path {
        let events = match MatchLog::replay(Path::new(wal_base)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("WAL replay failed: {}", e);
                process::exit(1);
            }
        };

        crate::state::match_log::validate_batch_assignment_framing(&events).unwrap_or_else(|e| {
            eprintln!("Export failed: {e}");
            process::exit(1);
        });
        let final_pairs = if events
            .iter()
            .any(|event| matches!(event, MatchLogEvent::AssignmentDelta(_)))
        {
            let snapshot: Vec<_> = crossmap_a_to_b
                .iter()
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect();
            Some(
                crate::state::match_log::reconstruct_assignment_history(&events, &snapshot)
                    .unwrap_or_else(|e| {
                        eprintln!("WAL assignment replay failed: {}", e);
                        process::exit(1);
                    }),
            )
        } else {
            None
        };
        for event in events {
            match event {
                MatchLogEvent::UpsertRecord { side, record } => {
                    let id_field = match side {
                        Side::A => a_id_field.as_str(),
                        Side::B => b_id_field.as_str(),
                    };
                    let id = record.get(id_field).cloned().unwrap_or_default();
                    if id.is_empty() {
                        continue;
                    }
                    // Re-upsert clears any stale review entries for this id.
                    let owner_side = if side == Side::A { "a" } else { "b" };
                    review_map.retain(|_, row| row.side != owner_side || row.id != id);
                    match side {
                        Side::A => {
                            record_map_a.insert(id, record);
                        }
                        Side::B => {
                            record_map_b.insert(id, record);
                        }
                    }
                }
                MatchLogEvent::RemoveRecord { side, id } => {
                    let removed_side = if side == Side::A { "a" } else { "b" };
                    review_map.retain(|_, row| {
                        if row.side == removed_side {
                            row.id != id
                        } else {
                            row.candidate_id != id
                        }
                    });
                    match side {
                        Side::A => {
                            record_map_a.remove(&id);
                            if let Some(b) = crossmap_a_to_b.remove(&id) {
                                crossmap_b_to_a.remove(&b);
                            }
                        }
                        Side::B => {
                            record_map_b.remove(&id);
                            if let Some(a) = crossmap_b_to_a.remove(&id) {
                                crossmap_a_to_b.remove(&a);
                            }
                        }
                    }
                }
                MatchLogEvent::CrossMapConfirm { a_id, b_id, .. } => {
                    // Resolve owners, not unrelated queries pointing at a candidate.
                    review_map.retain(|_, row| {
                        !((row.side == "a" && row.id == a_id)
                            || (row.side == "b" && row.id == b_id))
                    });
                    if let Some(b) = crossmap_a_to_b.remove(&a_id) {
                        crossmap_b_to_a.remove(&b);
                    }
                    if let Some(a) = crossmap_b_to_a.remove(&b_id) {
                        crossmap_a_to_b.remove(&a);
                    }
                    crossmap_b_to_a.insert(b_id.clone(), a_id.clone());
                    crossmap_a_to_b.insert(a_id, b_id);
                }
                MatchLogEvent::AssignmentDelta(delta) => {
                    review_map.retain(|_, row| {
                        !delta.replacements.iter().any(|pair| {
                            (row.side == "a" && row.id == pair.a_id)
                                || (row.side == "b" && row.id == pair.b_id)
                        })
                    });
                }
                MatchLogEvent::CrossMapReassign {
                    old_a_id,
                    old_b_id,
                    replacements,
                } => {
                    crossmap_a_to_b.remove(&old_a_id);
                    crossmap_b_to_a.remove(&old_b_id);
                    review_map.retain(|_, row| {
                        !replacements.iter().any(|pair| {
                            (row.side == "a" && row.id == pair.a_id)
                                || (row.side == "b" && row.id == pair.b_id)
                        })
                    });
                    for pair in replacements {
                        // Remove stale reverse/forward entries before installation,
                        // including when replaying over an already-flushed snapshot.
                        if let Some(b_id) = crossmap_a_to_b.remove(&pair.a_id) {
                            crossmap_b_to_a.remove(&b_id);
                        }
                        if let Some(a_id) = crossmap_b_to_a.remove(&pair.b_id) {
                            crossmap_a_to_b.remove(&a_id);
                        }
                        crossmap_b_to_a.insert(pair.b_id.clone(), pair.a_id.clone());
                        crossmap_a_to_b.insert(pair.a_id, pair.b_id);
                    }
                }
                MatchLogEvent::CrossMapBreak { a_id, b_id } => {
                    if crossmap_a_to_b.get(&a_id) == Some(&b_id) {
                        crossmap_a_to_b.remove(&a_id);
                        crossmap_b_to_a.remove(&b_id);
                    }
                    // A break doesn't re-add to review — the match is simply gone.
                }
                MatchLogEvent::ReviewMatch {
                    id,
                    side,
                    candidate_id,
                    score,
                    ..
                } => {
                    let side_str = match side {
                        Side::A => "a",
                        Side::B => "b",
                    };
                    let key = format!("{}:{}:{}", side_str, id, candidate_id);
                    review_map.insert(
                        key,
                        ReviewRow {
                            id,
                            side: side_str.to_string(),
                            candidate_id,
                            score,
                        },
                    );
                }
                MatchLogEvent::BatchAssignmentStarted { .. }
                | MatchLogEvent::BatchAssignmentCompleted { .. }
                | MatchLogEvent::EnrollOutcome { .. } => {}
                MatchLogEvent::NoMatchBelow { .. } => {
                    // No-match events don't affect export state.
                }
                MatchLogEvent::Exclude { .. } | MatchLogEvent::Unexclude { .. } => {
                    // Exclusions are not part of the export state.
                }
            }
        }
        if let Some(pairs) = final_pairs {
            crossmap_a_to_b.clear();
            crossmap_b_to_a.clear();
            for (a, b) in pairs {
                crossmap_a_to_b.insert(a.clone(), b.clone());
                crossmap_b_to_a.insert(b, a);
            }
        }
        review_map.retain(|_, row| match row.side.as_str() {
            "a" => !crossmap_a_to_b.contains_key(&row.id),
            _ => !crossmap_b_to_a.contains_key(&row.id),
        });
    }

    // Derive unmatched: records not present in the crossmap.
    let mut unmatched_a: Vec<(String, Record)> = record_map_a
        .into_iter()
        .filter(|(id, _)| !crossmap_a_to_b.contains_key(id))
        .collect();
    unmatched_a.sort_by(|a, b| a.0.cmp(&b.0));

    let mut unmatched_b: Vec<(String, Record)> = record_map_b
        .into_iter()
        .filter(|(id, _)| !crossmap_b_to_a.contains_key(id))
        .collect();
    unmatched_b.sort_by(|a, b| a.0.cmp(&b.0));

    let mut pairs: Vec<(String, String)> = crossmap_a_to_b.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let mut reviews: Vec<ReviewRow> = review_map.into_values().collect();
    reviews.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    write_results_or_exit(
        &out_dir.join("results.csv"),
        &pairs,
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    );
    write_review_or_exit(&out_dir.join("review.csv"), &reviews);
    write_unmatched_or_exit(&out_dir.join("unmatched_a.csv"), &unmatched_a, a_id_field);
    write_unmatched_or_exit(&out_dir.join("unmatched_b.csv"), &unmatched_b, b_id_field);

    print_summary(
        "in-memory",
        out_dir,
        &pairs,
        &reviews,
        &unmatched_a,
        &unmatched_b,
    );
}

// --- Shared types and helpers ------------------------------------------------

/// A pending review-band match row.
struct ReviewRow {
    id: String,
    side: String,
    candidate_id: String,
    score: f64,
}

fn write_results_or_exit(path: &Path, pairs: &[(String, String)], a_field: &str, b_field: &str) {
    if let Err(e) = write_results_csv(path, pairs, a_field, b_field) {
        eprintln!("Failed to write {}: {}", path.display(), e);
        process::exit(1);
    }
}

fn write_review_or_exit(path: &Path, entries: &[ReviewRow]) {
    if let Err(e) = write_review_csv(path, entries) {
        eprintln!("Failed to write {}: {}", path.display(), e);
        process::exit(1);
    }
}

fn write_unmatched_or_exit(path: &Path, records: &[(String, Record)], id_field: &str) {
    // Score is not available in the export path (no live scoring), so use None.
    let with_scores: Vec<(String, Record, Option<f64>)> = records
        .iter()
        .map(|(id, rec)| (id.clone(), rec.clone(), None))
        .collect();
    if let Err(e) = crate::batch::writer::write_unmatched_csv(path, &with_scores, id_field) {
        eprintln!("Failed to write {}: {}", path.display(), e);
        process::exit(1);
    }
}

fn write_results_csv(
    path: &Path,
    pairs: &[(String, String)],
    a_field: &str,
    b_field: &str,
) -> Result<(), csv::Error> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([a_field, b_field])?;
    for (a_id, b_id) in pairs {
        wtr.write_record([a_id.as_str(), b_id.as_str()])?;
    }
    wtr.flush()?;
    Ok(())
}

fn write_review_csv(path: &Path, entries: &[ReviewRow]) -> Result<(), csv::Error> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["id", "side", "candidate_id", "score"])?;
    for r in entries {
        wtr.write_record([
            r.id.as_str(),
            r.side.as_str(),
            r.candidate_id.as_str(),
            &format!("{:.4}", r.score),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn print_summary(
    source: &str,
    out_dir: &Path,
    pairs: &[(String, String)],
    reviews: &[ReviewRow],
    unmatched_a: &[(String, Record)],
    unmatched_b: &[(String, Record)],
) {
    println!("Export complete ({}):", source);
    println!(
        "  results:     {} ({} pairs)",
        out_dir.join("results.csv").display(),
        pairs.len()
    );
    println!(
        "  review:      {} ({} entries)",
        out_dir.join("review.csv").display(),
        reviews.len()
    );
    println!(
        "  unmatched_a: {} ({} records)",
        out_dir.join("unmatched_a.csv").display(),
        unmatched_a.len()
    );
    println!(
        "  unmatched_b: {} ({} records)",
        out_dir.join("unmatched_b.csv").display(),
        unmatched_b.len()
    );
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingConfig;
    use crate::models::Side;
    use crate::state::match_log::MatchLog;
    use crate::store::RecordStore;
    use tempfile::{TempDir, tempdir};

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn sqlite_export_reverses_payload_mapping_and_quotes_identifiers()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let fields: Vec<String> = [
            "entity_key",
            "id",
            "__melder_source_id",
            "__MELDER_SOURCE_ID_",
            "quoted\"; DROP TABLE crossmap; --",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let record: Record = fields
            .iter()
            .enumerate()
            .map(|(i, field)| (field.clone(), format!("payload-{i}")))
            .collect();
        let (store, _, writer) = crate::store::sqlite::open_sqlite(
            &dir.path().join("records.db"),
            &BlockingConfig::default(),
            None,
            &fields,
            &fields,
        )?;
        for side in [Side::A, Side::B] {
            store.insert(side, "internal-key", &record)?;
        }
        let conn = writer.lock().unwrap_or_else(|e| e.into_inner());
        for side in ["a", "b"] {
            for unmatched in [false, true] {
                assert_eq!(
                    query_records(&conn, side, unmatched, &fields)?,
                    vec![("internal-key".into(), record.clone())],
                    "offline reads must preserve logical payload fields for {side}"
                );
            }
        }
        conn.execute(
            "INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?1)",
            ["internal-key"],
        )?;
        for side in ["a", "b"] {
            assert!(
                query_records(&conn, side, true, &fields)?.is_empty(),
                "unmatched filtering must still use the internal key"
            );
        }
        Ok(())
    }

    #[test]
    fn sqlite_export_rejects_unmapped_discovered_columns() -> rusqlite::Result<()> {
        let conn = rusqlite::Connection::open_in_memory()?;
        conn.execute_batch("CREATE TABLE a_records (id TEXT PRIMARY KEY, unexpected TEXT);")?;
        assert!(
            matches!(
                query_records(&conn, "a", false, &[]),
                Err(rusqlite::Error::InvalidColumnName(name)) if name == "unexpected"
            ),
            "schema mismatches must not silently rename or omit source fields"
        );
        Ok(())
    }

    /// Write a minimal live-mode config YAML to disk and load it.
    /// `a_csv` and `b_csv` are paths that will be written into the YAML.
    /// `wal_path` is written into `live.match_log_path` (may be non-existent).
    fn write_and_load_config(
        dir: &TempDir,
        a_csv: &Path,
        b_csv: &Path,
        wal_path: &Path,
        db_path: Option<&Path>,
    ) -> Config {
        let db_line = match db_path {
            Some(p) => format!("  db_path: {}", p.display()),
            None => String::new(),
        };
        let yaml = format!(
            r#"
job:
  name: test_export
vector_backend: flat
datasets:
  a:
    path: {a_csv}
    id_field: entity_id
  b:
    path: {b_csv}
    id_field: counterparty_id
cross_map:
  path: {crossmap}
  a_id_field: entity_id
  b_id_field: counterparty_id
embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: {cache}
match_fields:
  - field_a: name
    field_b: cpty_name
    method: exact
    weight: 1.0
thresholds:
  auto_match: 0.85
  review_floor: 0.60
output:
  csv_dir_path: {out}
live:
  match_log_path: {wal}
{db_line}
"#,
            a_csv = a_csv.display(),
            b_csv = b_csv.display(),
            crossmap = dir.path().join("crossmap.csv").display(),
            cache = dir.path().join("cache").display(),
            out = dir.path().display(),
            wal = wal_path.display(),
        );
        let config_path = dir.path().join("config.yaml");
        std::fs::write(&config_path, yaml).unwrap();
        crate::config::load_config(&config_path).expect("failed to load test config")
    }

    #[test]
    fn seedless_enroll_cli_exports_wal_without_loading_an_a_dataset() {
        let dir = tempdir().expect("fixture");
        let wal = dir.path().join("enroll.ndjson");
        let log = MatchLog::open(&wal).expect("WAL");
        log.append(&MatchLogEvent::UpsertRecord {
            side: Side::A,
            record: make_record(&[("id", "isolated"), ("name", "WAL only")]),
        })
        .expect("record");
        log.append(&MatchLogEvent::EnrollOutcome {
            query_id: "isolated".into(),
            edges: vec![],
        })
        .expect("outcome");
        log.flush().expect("flush");
        let config = dir.path().join("enroll.yaml");
        std::fs::write(
            &config,
            format!(
                "job: {{ name: enroll }}\nvector_backend: flat\nembeddings: {{ model: ignored, cache_dir: cache }}\nmatch_fields: [{{ field: name, method: exact, weight: 1.0 }}]\nthresholds: {{ auto_match: 0.9, review_floor: 0.5 }}\nlive: {{ match_log_path: {} }}\n",
                wal.display()
            ),
        )
        .expect("config");
        let cfg = crate::config::load_enroll_job_config(&config).expect("enroll config");
        let input = load_enroll_build_input(&cfg).expect("seedless input");
        assert!(input.a_records.is_empty(), "seedless A must start empty");
        let out = dir.path().join("export");
        cmd_export(&config, &out);
        let unmatched = std::fs::read_to_string(out.join("unmatched.csv")).expect("unmatched");
        assert!(
            unmatched.contains("isolated"),
            "WAL record retained: {unmatched}"
        );
        assert!(
            !out.join("results.csv").exists(),
            "enroll is not live matching"
        );
    }

    #[test]
    fn native_enroll_export_replays_seed_removal_and_keeps_isolated_seed() {
        let dir = tempdir().expect("fixture");
        let seed = dir.path().join("seed.csv");
        std::fs::write(&seed, "key,name\nkept,Isolated\nremoved,Deleted\n").expect("seed");
        let wal = dir.path().join("enroll.ndjson");
        let log = MatchLog::open(&wal).expect("WAL");
        log.append(&MatchLogEvent::RemoveRecord {
            side: Side::A,
            id: "removed".into(),
        })
        .expect("remove");
        log.flush().expect("flush");
        let config = dir.path().join("enroll.yaml");
        std::fs::write(&config, format!(
            "job: {{name: enroll}}\ndataset: {{path: '{}', id_field: key}}\nvector_backend: flat\nembeddings: {{model: ignored, cache_dir: cache}}\nmatch_fields: [{{field: name, method: exact, weight: 1.0}}]\nthresholds: {{auto_match: 0.9, review_floor: 0.5}}\nlive: {{match_log_path: '{}'}}\n",
            seed.display(), wal.display()
        )).expect("config");
        let out = dir.path().join("out");
        cmd_export(&config, &out);
        let unmatched = std::fs::read_to_string(out.join("unmatched.csv")).expect("unmatched");
        assert!(
            unmatched.contains("kept") && unmatched.contains("Isolated"),
            "isolated seed retained: {unmatched}"
        );
        assert!(
            !unmatched.contains("removed"),
            "historical removal respected: {unmatched}"
        );
        assert!(
            !out.join("results.csv").exists(),
            "no paired export artifacts"
        );
    }

    #[test]
    fn historical_greedy_batch_cli_never_loads_modern_snapshots() {
        for missing_inputs in [false, true] {
            let dir = tempdir().expect("fixture");
            let a = dir.path().join("a.csv");
            let b = dir.path().join("b.csv");
            if !missing_inputs {
                std::fs::write(&a, "entity_id,name\nMODERN-A,Modern\n").expect("A");
                std::fs::write(&b, "counterparty_id,cpty_name\nMODERN-B,Modern\n").expect("B");
            }
            std::fs::write(
                dir.path().join("crossmap.csv"),
                "entity_id,counterparty_id\nMODERN-A,MODERN-B\n",
            )
            .expect("modern crossmap");
            // Historical greedy runs have no graph framing or new identity event.
            let wal = dir.path().join("oldjob.match_log.ndjson");
            let log = MatchLog::open(&wal).expect("WAL");
            for (side, record) in [
                (
                    Side::A,
                    make_record(&[("entity_id", "OLD-A"), ("name", "Historical")]),
                ),
                (
                    Side::B,
                    make_record(&[("counterparty_id", "OLD-B"), ("cpty_name", "Historical")]),
                ),
            ] {
                log.append(&MatchLogEvent::UpsertRecord { side, record })
                    .expect("record");
            }
            log.append(&MatchLogEvent::CrossMapConfirm {
                a_id: "OLD-A".into(),
                b_id: "OLD-B".into(),
                score: Some(1.0),
                rank: None,
                reason: None,
            })
            .expect("pair");
            log.flush().expect("flush");
            // Select the timestamped segment, not merely the old base filename.
            let selected = log.path();
            assert!(
                is_greedy_batch_log_path(selected),
                "historical namespace recognized"
            );
            let live_db = dir.path().join("modern-live.db");
            std::fs::write(&live_db, "not a SQLite DB").expect("unrelated live DB");
            write_and_load_config(&dir, &a, &b, selected, Some(&live_db));
            let out = dir.path().join("export");
            cmd_export(&dir.path().join("config.yaml"), &out);
            let relationships =
                std::fs::read_to_string(out.join("relationships.csv")).expect("relationships");
            assert!(
                relationships.contains("OLD-A") && relationships.contains("OLD-B"),
                "historical pair retained: {relationships}"
            );
            assert!(
                !relationships.contains("MODERN"),
                "modern pairs must not leak: {relationships}"
            );
            let unmatched = std::fs::read_to_string(out.join("unmatched.csv")).expect("unmatched");
            assert!(
                !unmatched.contains("MODERN"),
                "modern inputs must not leak: {unmatched}"
            );
            assert!(
                !out.join("results.csv").exists(),
                "batch must not run legacy live replay"
            );
        }
    }

    #[test]
    fn greedy_cli_exports_seed_records_and_pairs_without_resurrecting_removals() {
        let dir = tempdir().expect("fixture");
        let a = dir.path().join("a.csv");
        let b = dir.path().join("b.csv");
        let wal = dir.path().join("wal.ndjson");
        std::fs::write(&a, "entity_id,name\nA1,Seed\nA2,Removed\n").expect("A seed");
        std::fs::write(
            &b,
            "counterparty_id,cpty_name\nB1,Seed\nB2,Removed\nB3,Untouched\n",
        )
        .expect("B seed");
        std::fs::write(
            dir.path().join("crossmap.csv"),
            "entity_id,counterparty_id\nA1,B1\nA2,B2\n",
        )
        .expect("pairs");
        let log = MatchLog::open(&wal).expect("WAL");
        log.append(&MatchLogEvent::RemoveRecord {
            side: Side::B,
            id: "B2".into(),
        })
        .expect("remove");
        log.flush().expect("flush");
        write_and_load_config(&dir, &a, &b, &wal, None);
        let out = dir.path().join("export");
        cmd_export(&dir.path().join("config.yaml"), &out);
        let rels = std::fs::read_to_string(out.join("relationships.csv")).expect("relationships");
        assert!(
            rels.contains("A1") && rels.contains("B1"),
            "seed pair retained: {rels}"
        );
        assert!(!rels.contains("B2"), "removed seed must not return: {rels}");
        let unmatched = std::fs::read_to_string(out.join("unmatched.csv")).expect("unmatched");
        assert!(
            unmatched.contains("B3") && unmatched.contains("Untouched"),
            "untouched B fields retained: {unmatched}"
        );
        assert!(
            out.join("results.csv").exists(),
            "legacy exports remain available"
        );
    }

    #[test]
    fn sqlite_snapshot_build_includes_records_absent_from_seeds() {
        let dir = tempdir().expect("fixture");
        let db = dir.path().join("live.db");
        let blocking = BlockingConfig {
            enabled: false,
            operator: "and".into(),
            fields: vec![],
            field_a: None,
            field_b: None,
        };
        let (store, crossmap, _) = crate::store::sqlite::open_sqlite(
            &db,
            &blocking,
            None,
            &["entity_id".into(), "name".into()],
            &["counterparty_id".into(), "cpty_name".into()],
        )
        .expect("store");
        store
            .insert(
                Side::A,
                "A-live",
                &Record::from([("name".into(), "Authoritative".into())]),
            )
            .expect("A record");
        store
            .insert(
                Side::B,
                "B-live",
                &Record::from([("cpty_name".into(), "Current".into())]),
            )
            .expect("B record");
        crossmap.add("A-live", "B-live").expect("pair");
        let a = dir.path().join("missing-a.csv");
        let b = dir.path().join("missing-b.csv");
        let mut cfg =
            write_and_load_config(&dir, &a, &b, &dir.path().join("missing-wal"), Some(&db));
        cfg.output.db_path = Some(dir.path().join("output.db").to_string_lossy().into_owned());
        #[cfg(feature = "parquet-format")]
        {
            cfg.output.parquet_dir_path =
                Some(dir.path().join("parquet").to_string_lossy().into_owned());
        }
        let out = dir.path().join("export");
        std::fs::create_dir_all(&out).expect("output dir");
        export_sqlite(&cfg, &db, &out);
        let rels = std::fs::read_to_string(out.join("relationships.csv")).expect("relationships");
        assert!(
            rels.contains("A-live") && rels.contains("B-live"),
            "SQLite pairs retained: {rels}"
        );
        let output = rusqlite::Connection::open(cfg.output.db_path.as_ref().expect("output path"))
            .expect("output DB");
        let count: i64 = output
            .query_row("SELECT count(*) FROM a_records", [], |row| row.get(0))
            .expect("record count");
        assert_eq!(
            count, 1,
            "output uses current SQLite records, not missing CSV seeds"
        );
        #[cfg(feature = "parquet-format")]
        assert!(
            dir.path().join("parquet/relationships.parquet").exists(),
            "configured parquet generated"
        );
        cfg.output.db_path = Some(db.to_string_lossy().into_owned());
        assert!(
            crate::output::build::validate_live_output_path(&cfg).is_err(),
            "live DB must not be an output target"
        );
    }

    #[test]
    fn assignment_cli_build_exports_snapshot_to_configured_database() {
        let dir = tempdir().expect("tempdir");
        let a = dir.path().join("a.csv");
        let b = dir.path().join("b.csv");
        let wal = dir.path().join("wal.ndjson");
        std::fs::write(&a, "entity_id,name\nA1,One\nAX,Untouched\n").expect("A");
        std::fs::write(
            &b,
            "counterparty_id,cpty_name\nB1,Old\nB2,New\nBX,Untouched\n",
        )
        .expect("B");
        std::fs::write(
            dir.path().join("crossmap.csv"),
            "entity_id,counterparty_id\nA1,B2\nAX,BX\n",
        )
        .expect("snapshot");
        let log = MatchLog::open(&wal).expect("log");
        log.append(&serde_json::from_str::<MatchLogEvent>(
            r#"{"type":"assignment_delta","version":1,"attempt_id":1,"step_index":0,"proposer":{"side":"b","id":"B2"},"expected":[{"a_id":"A1","b_id":"B1"}],"replacements":[{"a_id":"A1","b_id":"B2","score_b_to_a":0.91,"score_a_to_b":0.96,"rank":null,"reason":null}],"newly_unmatched":[{"side":"b","id":"B1"}],"previous_receiver_score":0.8}"#,
        ).expect("event")).expect("append");
        log.flush().expect("flush");
        let cfg = write_and_load_config(&dir, &a, &b, &wal, None);
        let input = load_build_input(&cfg).expect("input");
        assert_eq!(input.a_records.len(), 2, "load original A records");
        assert_eq!(input.b_records.len(), 3, "load original B records");
        assert_eq!(input.snapshot_pairs.len(), 2, "load all snapshot pairs");
        let config = dir.path().join("config.yaml");
        let db = dir.path().join("output.db");
        let yaml = std::fs::read_to_string(&config).expect("config");
        let output = format!("output:\n  db_path: {}\n", db.display());
        #[cfg(feature = "parquet-format")]
        let output = format!(
            "{output}  parquet_dir_path: {}\n",
            dir.path().join("parquet").display()
        );
        std::fs::write(&config, yaml.replace("output:\n", &output)).expect("configured outputs");
        let out = dir.path().join("export");
        cmd_export(&config, &out);
        let conn = rusqlite::Connection::open(&db).expect("output DB");
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM relationships WHERE relationship_type='match'",
                [],
                |row| row.get(0),
            )
            .expect("matches");
        assert_eq!(
            count, 2,
            "CLI builds unrelated snapshot pair as well as assignment"
        );
        let relationships = std::fs::read_to_string(out.join("relationships.csv")).expect("CSV");
        assert!(
            relationships.contains("AX") && relationships.contains("BX"),
            "CSV retains untouched pair"
        );
        let unmatched = std::fs::read_to_string(out.join("unmatched.csv")).expect("unmatched");
        assert!(
            unmatched.contains("B1") && unmatched.contains("Old"),
            "unmatched output retains original dataset fields"
        );
        #[cfg(feature = "parquet-format")]
        assert!(
            dir.path().join("parquet/relationships.parquet").exists(),
            "configured Parquet no longer rejected"
        );
    }

    #[test]
    fn sqlite_export_round_trip() {
        use crate::store::sqlite::open_sqlite;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        let out_dir = dir.path().join("export");

        let bc = BlockingConfig {
            enabled: false,
            operator: "and".to_string(),
            fields: vec![],
            field_a: None,
            field_b: None,
        };

        // Populate the DB in a scoped block so connections are dropped
        // before we open it again inside export_sqlite.
        {
            let cols_a = vec!["entity_id".to_string(), "name".to_string()];
            let cols_b = vec!["counterparty_id".to_string(), "cpty_name".to_string()];
            let (store, crossmap, conn) =
                open_sqlite(&db_path, &bc, None, &cols_a, &cols_b).unwrap();

            let rec_a1 = make_record(&[("entity_id", "A-1"), ("name", "Acme")]);
            let rec_a2 = make_record(&[("entity_id", "A-2"), ("name", "Globex")]);
            let rec_b1 = make_record(&[("counterparty_id", "B-1"), ("cpty_name", "Acme Corp")]);
            let rec_b2 = make_record(&[("counterparty_id", "B-2"), ("cpty_name", "Globex Inc")]);

            store.insert(Side::A, "A-1", &rec_a1).unwrap();
            store.insert(Side::A, "A-2", &rec_a2).unwrap();
            store.insert(Side::B, "B-1", &rec_b1).unwrap();
            store.insert(Side::B, "B-2", &rec_b2).unwrap();

            // A-1 ↔ B-1 confirmed; A-2 and B-2 are unmatched.
            crossmap.add("A-1", "B-1").unwrap();

            let db_conn = conn.lock().unwrap_or_else(|e| e.into_inner());
            db_conn
                .execute("INSERT INTO a_unmatched (id) VALUES (?1)", ["A-2"])
                .unwrap();
            db_conn
                .execute("INSERT INTO b_unmatched (id) VALUES (?1)", ["B-2"])
                .unwrap();
            // One pending review entry.
            db_conn
                .execute(
                    "INSERT INTO reviews \
                     (key, id, side, candidate_id, score) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params!["b:B-2:A-2", "B-2", "b", "A-2", 0.72f64],
                )
                .unwrap();
        }

        // Build a minimal config pointing at the DB.
        let a_csv = dir.path().join("a.csv");
        let b_csv = dir.path().join("b.csv");
        let wal = dir.path().join("wal.ndjson");
        std::fs::write(&a_csv, "entity_id,name\n").unwrap();
        std::fs::write(&b_csv, "counterparty_id,cpty_name\n").unwrap();
        let cfg = write_and_load_config(&dir, &a_csv, &b_csv, &wal, Some(&db_path));

        std::fs::create_dir_all(&out_dir).unwrap();
        export_sqlite(&cfg, &db_path, &out_dir);

        // results.csv: one pair, with correct id-field column headers.
        let results = std::fs::read_to_string(out_dir.join("results.csv")).unwrap();
        assert!(
            results.contains("entity_id,counterparty_id"),
            "results header wrong: {results}"
        );
        assert!(results.contains("A-1,B-1"), "pair missing: {results}");
        assert!(!results.contains("A-2"), "unmatched A-2 in results");

        // review.csv: one entry for B-2.
        let review = std::fs::read_to_string(out_dir.join("review.csv")).unwrap();
        assert!(
            review.contains("id,side,candidate_id,score"),
            "review header wrong: {review}"
        );
        assert!(
            review.contains("B-2,b,A-2"),
            "review entry missing: {review}"
        );

        // unmatched_a.csv: A-2 with its fields, not A-1.
        let ua = std::fs::read_to_string(out_dir.join("unmatched_a.csv")).unwrap();
        assert!(ua.contains("entity_id"), "unmatched_a header missing");
        assert!(ua.contains("A-2"), "A-2 not in unmatched_a: {ua}");
        assert!(!ua.contains("A-1"), "matched A-1 in unmatched_a");

        // unmatched_b.csv: B-2 with its fields, not B-1.
        let ub = std::fs::read_to_string(out_dir.join("unmatched_b.csv")).unwrap();
        assert!(ub.contains("counterparty_id"), "unmatched_b header missing");
        assert!(ub.contains("B-2"), "B-2 not in unmatched_b: {ub}");
        assert!(!ub.contains("B-1"), "matched B-1 in unmatched_b");
    }

    #[test]
    fn sqlite_export_uses_membership_before_restart_repairs_metadata() {
        use crate::store::sqlite::open_sqlite;

        for side in [Side::A, Side::B] {
            let dir = tempdir().expect("fixture directory");
            let db_path = dir.path().join("live.db");
            let out_dir = dir.path().join("export");
            let bc = BlockingConfig {
                enabled: false,
                operator: "and".into(),
                fields: vec![],
                field_a: None,
                field_b: None,
            };
            {
                let (store, crossmap, _) = open_sqlite(
                    &db_path,
                    &bc,
                    None,
                    &["entity_id".into()],
                    &["counterparty_id".into()],
                )
                .expect("open database");
                // Identical IDs on opposite sides must not share review ownership.
                for s in [Side::A, Side::B] {
                    for id in ["receiver", "old", "new"] {
                        store.insert(s, id, &Record::new()).expect("insert record");
                        store.mark_unmatched(s, id).expect("initial flags");
                    }
                }
                let (old, new) = if side == Side::A {
                    (("old", "receiver"), ("new", "receiver"))
                } else {
                    (("receiver", "old"), ("receiver", "new"))
                };
                crossmap.add(old.0, old.1).expect("initial pair");
                store.mark_matched(side, "old").expect("old flags");
                store
                    .mark_matched(side.opposite(), "receiver")
                    .expect("receiver flags");
                store
                    .persist_review("resolved", "new", side, "receiver", 0.8)
                    .expect("new endpoint's stale review");
                store
                    .persist_review("unrelated", "new", side.opposite(), "receiver", 0.7)
                    .expect("opposite side review");
                assert!(
                    crossmap
                        .apply_assignment_delta(&[old], &[new])
                        .expect("commit assignment"),
                    "fixture must commit the displacement"
                );
                // Simulate interruption here: deliberately do not update any
                // flags or reviews, and never call the warm server loader.
            }
            let a_csv = dir.path().join("a.csv");
            let b_csv = dir.path().join("b.csv");
            std::fs::write(&a_csv, "entity_id\n").expect("A fixture");
            std::fs::write(&b_csv, "counterparty_id\n").expect("B fixture");
            let cfg = write_and_load_config(
                &dir,
                &a_csv,
                &b_csv,
                &dir.path().join("wal.ndjson"),
                Some(&db_path),
            );
            std::fs::create_dir_all(&out_dir).expect("output directory");
            export_sqlite(&cfg, &db_path, &out_dir);
            let mut unmatched = csv::Reader::from_path(out_dir.join(if side == Side::A {
                "unmatched_a.csv"
            } else {
                "unmatched_b.csv"
            }))
            .expect("unmatched output");
            let ids: Vec<String> = unmatched
                .records()
                .map(|row| row.expect("CSV row")[0].to_owned())
                .collect();
            assert!(
                ids.contains(&"old".into()),
                "orphan must be exported: {ids:?}"
            );
            assert!(
                !ids.contains(&"new".into()),
                "matched endpoint excluded: {ids:?}"
            );
            let mut review =
                csv::Reader::from_path(out_dir.join("review.csv")).expect("review output");
            let rows: Vec<_> = review.records().map(|r| r.expect("review row")).collect();
            assert_eq!(rows.len(), 1, "only unmatched owner's review survives");
            assert_eq!(
                &rows[0][1],
                if side == Side::A { "b" } else { "a" },
                "review filtering must distinguish owner side"
            );
            let results =
                std::fs::read_to_string(out_dir.join("results.csv")).expect("pair output");
            assert!(
                results.contains(if side == Side::A {
                    "new,receiver"
                } else {
                    "receiver,new"
                }),
                "pair output must reflect committed membership: {results}"
            );
        }
    }

    #[test]
    fn memory_export_with_wal() {
        let dir = tempdir().unwrap();
        let out_dir = dir.path().join("export");
        std::fs::create_dir_all(&out_dir).unwrap();

        // Write CSV datasets.
        let a_csv = dir.path().join("a.csv");
        let b_csv = dir.path().join("b.csv");
        std::fs::write(&a_csv, "entity_id,name\nA-1,Acme\nA-2,Globex\n").unwrap();
        std::fs::write(
            &b_csv,
            "counterparty_id,cpty_name\nB-1,Acme Corp\nB-2,Globex Inc\n",
        )
        .unwrap();

        // Write WAL events: upsert both sides, confirm A-1↔B-1, review B-2→A-2.
        let wal_base = dir.path().join("wal.ndjson");
        {
            let log = MatchLog::open(&wal_base).unwrap();
            // These upserts add API-only record A-3 and update A-1.
            log.append(&MatchLogEvent::UpsertRecord {
                side: Side::A,
                record: make_record(&[("entity_id", "A-3"), ("name", "NewCo")]),
            })
            .unwrap();
            log.append(&MatchLogEvent::CrossMapConfirm {
                a_id: "A-1".into(),
                b_id: "B-1".into(),
                score: Some(0.95),
                rank: Some(1),
                reason: None,
            })
            .unwrap();
            log.append(&MatchLogEvent::ReviewMatch {
                id: "B-2".into(),
                side: Side::B,
                candidate_id: "A-2".into(),
                score: 0.73,
                rank: Some(1),
                reason: None,
            })
            .unwrap();
            // A-3 gets a review entry, then immediately removed — should not appear.
            log.append(&MatchLogEvent::ReviewMatch {
                id: "A-3".into(),
                side: Side::A,
                candidate_id: "B-2".into(),
                score: 0.65,
                rank: Some(1),
                reason: None,
            })
            .unwrap();
            log.append(&MatchLogEvent::RemoveRecord {
                side: Side::A,
                id: "A-3".into(),
            })
            .unwrap();
            log.flush().unwrap();
        }

        let cfg = write_and_load_config(&dir, &a_csv, &b_csv, &wal_base, None);
        export_memory(&cfg, &out_dir);

        // results.csv: A-1 ↔ B-1.
        let results = std::fs::read_to_string(out_dir.join("results.csv")).unwrap();
        assert!(results.contains("A-1,B-1"), "pair missing: {results}");
        assert!(!results.contains("A-2"), "A-2 should not be in results");

        // review.csv: B-2 → A-2 at 0.73; A-3's review must be gone (removed).
        let review = std::fs::read_to_string(out_dir.join("review.csv")).unwrap();
        assert!(
            review.contains("B-2,b,A-2"),
            "B-2 review entry missing: {review}"
        );
        assert!(!review.contains("A-3"), "removed A-3 review still present");

        // unmatched_a.csv: A-2 is unmatched; A-1 is matched; A-3 was removed.
        let ua = std::fs::read_to_string(out_dir.join("unmatched_a.csv")).unwrap();
        assert!(ua.contains("A-2"), "A-2 not in unmatched_a: {ua}");
        assert!(!ua.contains("A-1"), "matched A-1 in unmatched_a");
        assert!(!ua.contains("A-3"), "removed A-3 in unmatched_a");

        // unmatched_b.csv: B-2 is unmatched; B-1 is matched.
        let ub = std::fs::read_to_string(out_dir.join("unmatched_b.csv")).unwrap();
        assert!(ub.contains("B-2"), "B-2 not in unmatched_b: {ub}");
        assert!(!ub.contains("B-1"), "matched B-1 in unmatched_b");
    }

    #[test]
    fn memory_export_reassignment_updates_both_maps_and_reviews() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("export");
        std::fs::create_dir_all(&out).expect("output dir");
        let a = dir.path().join("a.csv");
        let b = dir.path().join("b.csv");
        let wal = dir.path().join("wal.ndjson");
        std::fs::write(&a, "entity_id,name\nA1,One\nA2,Two\n").expect("A CSV");
        std::fs::write(&b, "counterparty_id,cpty_name\nB1,One\nB2,Two\n").expect("B CSV");
        let log = MatchLog::open(&wal).expect("log");
        for line in [
            r#"{"type":"crossmap_confirm","a_id":"A1","b_id":"B1","score":0.8}"#,
            r#"{"type":"review_match","side":"b","id":"B2","candidate_id":"A2","score":0.7}"#,
            r#"{"type":"crossmap_reassign","old_a_id":"A1","old_b_id":"B1","replacements":[{"a_id":"A1","b_id":"B2","score":0.91,"rank":2},{"a_id":"A2","b_id":"B1","score":0.87}]}"#,
        ] {
            log.append(&serde_json::from_str(line).expect("event"))
                .expect("append");
        }
        log.flush().expect("flush");
        let cfg = write_and_load_config(&dir, &a, &b, &wal, None);
        export_memory(&cfg, &out);
        let results = std::fs::read_to_string(out.join("results.csv")).expect("results");
        assert!(
            results.contains("A1,B2") && results.contains("A2,B1"),
            "both replacement pairs: {results}"
        );
        assert!(!results.contains("A1,B1"), "old pair removed");
        for name in ["unmatched_a.csv", "unmatched_b.csv", "review.csv"] {
            let contents = std::fs::read_to_string(out.join(name)).expect("output");
            assert_eq!(
                contents.lines().count(),
                1,
                "{name} must have only header: {contents}"
            );
        }
    }

    #[test]
    fn empty_state_writes_headers_only() {
        let dir = tempdir().unwrap();
        let out_dir = dir.path().join("export");
        std::fs::create_dir_all(&out_dir).unwrap();

        let a_csv = dir.path().join("a.csv");
        let b_csv = dir.path().join("b.csv");
        let wal = dir.path().join("wal.ndjson");
        std::fs::write(&a_csv, "entity_id,name\n").unwrap();
        std::fs::write(&b_csv, "counterparty_id,cpty_name\n").unwrap();

        let cfg = write_and_load_config(&dir, &a_csv, &b_csv, &wal, None);
        export_memory(&cfg, &out_dir);

        // All files exist with headers but no data rows.
        let results = std::fs::read_to_string(out_dir.join("results.csv")).unwrap();
        assert_eq!(results.trim(), "entity_id,counterparty_id");

        let review = std::fs::read_to_string(out_dir.join("review.csv")).unwrap();
        assert_eq!(review.trim(), "id,side,candidate_id,score");

        let ua = std::fs::read_to_string(out_dir.join("unmatched_a.csv")).unwrap();
        assert_eq!(ua.trim(), "entity_id,score");

        let ub = std::fs::read_to_string(out_dir.join("unmatched_b.csv")).unwrap();
        assert_eq!(ub.trim(), "counterparty_id,score");
    }
}
