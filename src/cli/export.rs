//! `meld export` — export live-mode state to CSV files.
//!
//! Reads from SQLite (if `live.db_path` is set and the DB exists) or
//! reconstructs state from CSV datasets and WAL replay. Does not require
//! the server to be running.

use std::collections::HashMap;
use std::path::Path;
use std::process;

use crate::config::Config;
use crate::crossmap::traits::CrossMapOps;
use crate::models::{Record, Side};
use crate::state::upsert_log::{UpsertLog, WalEvent};

// --- Entry point -------------------------------------------------------------

/// Export live-mode state to CSV files in `out_dir`.
///
/// Produces four files: results.csv, review.csv, unmatched_a.csv,
/// unmatched_b.csv. Reads from SQLite if `live.db_path` is set and the DB
/// file exists; otherwise reconstructs state from CSV datasets + WAL replay.
pub fn cmd_export(config_path: &Path, out_dir: &Path) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // Guard: batch configs have neither upsert_log nor db_path configured.
    if cfg.live.upsert_log.is_none() && cfg.live.db_path.is_none() {
        eprintln!(
            "Error: meld export is for live mode. \
             Batch mode already writes output files directly via meld run."
        );
        process::exit(1);
    }

    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!(
            "Failed to create output directory {}: {}",
            out_dir.display(),
            e
        );
        process::exit(1);
    }

    // Use SQLite if the DB file already exists; fall back to WAL replay.
    let sqlite_db = cfg
        .live
        .db_path
        .as_deref()
        .map(Path::new)
        .filter(|p| p.exists());

    if let Some(db_path) = sqlite_db {
        export_sqlite(&cfg, db_path, out_dir);
    } else {
        export_memory(&cfg, out_dir);
    }
}

// --- SQLite path -------------------------------------------------------------

fn export_sqlite(cfg: &Config, db_path: &Path, out_dir: &Path) {
    let conn = match rusqlite::Connection::open(db_path) {
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

    // results.csv — confirmed crossmap pairs.
    let pairs: Vec<(String, String)> = {
        let mut stmt = conn
            .prepare("SELECT a_id, b_id FROM crossmap ORDER BY a_id")
            .expect("prepare crossmap query");
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .expect("query crossmap")
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
        let mut stmt = conn
            .prepare("SELECT id, side, candidate_id, score FROM reviews ORDER BY score DESC")
            .expect("prepare reviews query");
        stmt.query_map([], |row| {
            Ok(ReviewRow {
                id: row.get(0)?,
                side: row.get(1)?,
                candidate_id: row.get(2)?,
                score: row.get(3)?,
            })
        })
        .expect("query reviews")
        .filter_map(|r| r.ok())
        .collect()
    };
    write_review_or_exit(&out_dir.join("review.csv"), &reviews);

    // unmatched_a.csv — A-side records with no crossmap pair.
    let unmatched_a = query_unmatched(&conn, "a");
    write_unmatched_or_exit(
        &out_dir.join("unmatched_a.csv"),
        &unmatched_a,
        &cfg.datasets.a.id_field,
    );

    // unmatched_b.csv — B-side records with no crossmap pair.
    let unmatched_b = query_unmatched(&conn, "b");
    write_unmatched_or_exit(
        &out_dir.join("unmatched_b.csv"),
        &unmatched_b,
        &cfg.datasets.b.id_field,
    );

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
fn query_unmatched(conn: &rusqlite::Connection, side: &str) -> Vec<(String, Record)> {
    // Discover columns from the table schema (skip "id")
    let columns: Vec<String> = {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({side}_records)"))
            .expect("pragma table_info");
        stmt.query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .expect("query table_info")
        .filter_map(|r| r.ok())
        .filter(|name| name != "id")
        .collect()
    };

    let col_list = if columns.is_empty() {
        "r.id".to_string()
    } else {
        format!(
            "r.id, {}",
            columns
                .iter()
                .map(|c| format!("r.\"{}\"", c))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let sql = format!(
        "SELECT {} FROM {}_records r \
         INNER JOIN {}_unmatched u ON r.id = u.id \
         ORDER BY r.id",
        col_list, side, side
    );
    let mut stmt = conn.prepare(&sql).expect("prepare unmatched query");
    stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let mut record = Record::new();
        for (i, col_name) in columns.iter().enumerate() {
            let val: Option<String> = row.get(i + 1).unwrap_or(None);
            record.insert(col_name.clone(), val.unwrap_or_default());
        }
        Ok((id, record))
    })
    .expect("query unmatched")
    .filter_map(|r| r.ok())
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
    if let Some(wal_base) = &cfg.live.upsert_log {
        let events = match UpsertLog::replay(Path::new(wal_base)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Warning: WAL replay failed: {}", e);
                vec![]
            }
        };

        for event in events {
            match event {
                WalEvent::UpsertRecord { side, record } => {
                    let id_field = match side {
                        Side::A => a_id_field.as_str(),
                        Side::B => b_id_field.as_str(),
                    };
                    let id = record.get(id_field).cloned().unwrap_or_default();
                    if id.is_empty() {
                        continue;
                    }
                    // Re-upsert clears any stale review entries for this id.
                    drain_reviews_for_id(&mut review_map, &id);
                    match side {
                        Side::A => {
                            record_map_a.insert(id, record);
                        }
                        Side::B => {
                            record_map_b.insert(id, record);
                        }
                    }
                }
                WalEvent::RemoveRecord { side, id } => {
                    drain_reviews_for_id(&mut review_map, &id);
                    match side {
                        Side::A => {
                            record_map_a.remove(&id);
                        }
                        Side::B => {
                            record_map_b.remove(&id);
                        }
                    }
                }
                WalEvent::CrossMapConfirm { a_id, b_id, .. } => {
                    // A confirm resolves pending reviews for both ids.
                    drain_reviews_for_id(&mut review_map, &a_id);
                    drain_reviews_for_id(&mut review_map, &b_id);
                    crossmap_b_to_a.insert(b_id.clone(), a_id.clone());
                    crossmap_a_to_b.insert(a_id, b_id);
                }
                WalEvent::CrossMapBreak { a_id, b_id } => {
                    crossmap_a_to_b.remove(&a_id);
                    crossmap_b_to_a.remove(&b_id);
                    // A break doesn't re-add to review — the match is simply gone.
                }
                WalEvent::ReviewMatch {
                    id,
                    side,
                    candidate_id,
                    score,
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
            }
        }
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

/// Remove all review entries where the given id appears as either the
/// record id or the candidate id.
fn drain_reviews_for_id(map: &mut HashMap<String, ReviewRow>, id: &str) {
    map.retain(|_, v| v.id != id && v.candidate_id != id);
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
    if let Err(e) = crate::batch::writer::write_unmatched_csv(path, records, id_field) {
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
    use crate::state::upsert_log::UpsertLog;
    use crate::store::RecordStore;
    use tempfile::{tempdir, TempDir};

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Write a minimal live-mode config YAML to disk and load it.
    /// `a_csv` and `b_csv` are paths that will be written into the YAML.
    /// `wal_path` is written into `live.upsert_log` (may be non-existent).
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
  results_path: {out}/results.csv
  review_path: {out}/review.csv
  unmatched_path: {out}/unmatched.csv
live:
  upsert_log: {wal}
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

            store.insert(Side::A, "A-1", &rec_a1);
            store.insert(Side::A, "A-2", &rec_a2);
            store.insert(Side::B, "B-1", &rec_b1);
            store.insert(Side::B, "B-2", &rec_b2);

            // A-1 ↔ B-1 confirmed; A-2 and B-2 are unmatched.
            crossmap.add("A-1", "B-1");

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
            let log = UpsertLog::open(&wal_base).unwrap();
            // These upserts add API-only record A-3 and update A-1.
            log.append(&WalEvent::UpsertRecord {
                side: Side::A,
                record: make_record(&[("entity_id", "A-3"), ("name", "NewCo")]),
            })
            .unwrap();
            log.append(&WalEvent::CrossMapConfirm {
                a_id: "A-1".into(),
                b_id: "B-1".into(),
                score: Some(0.95),
            })
            .unwrap();
            log.append(&WalEvent::ReviewMatch {
                id: "B-2".into(),
                side: Side::B,
                candidate_id: "A-2".into(),
                score: 0.73,
            })
            .unwrap();
            // A-3 gets a review entry, then immediately removed — should not appear.
            log.append(&WalEvent::ReviewMatch {
                id: "A-3".into(),
                side: Side::A,
                candidate_id: "B-2".into(),
                score: 0.65,
            })
            .unwrap();
            log.append(&WalEvent::RemoveRecord {
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
        assert_eq!(ua.trim(), "entity_id");

        let ub = std::fs::read_to_string(out_dir.join("unmatched_b.csv")).unwrap();
        assert_eq!(ub.trim(), "counterparty_id");
    }
}
