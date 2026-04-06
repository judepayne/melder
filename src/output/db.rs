//! SQLite output DB builder.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::Connection;

use crate::models::Record;

use super::build::Relationship;
use super::manifest::OutputManifest;

/// Build the output SQLite database.
///
/// Creates tables (a_records, b_records, relationships, field_scores, metadata),
/// populates them from the replayed match log state, adds indices and views.
/// Written to a `.tmp` file and renamed atomically on success.
pub fn build_db(
    path: &Path,
    a_records: &HashMap<String, Record>,
    b_records: &HashMap<String, Record>,
    relationships: &[Relationship],
    manifest: &OutputManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = path.with_extension("db.tmp");
    if tmp.exists() {
        std::fs::remove_file(&tmp)?;
    }

    let conn = Connection::open(&tmp)?;
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;

    // --- a_records table (dynamic columns) ---
    create_records_table(&conn, "a_records", &manifest.a_id_field, &manifest.a_fields)?;
    insert_records(
        &conn,
        "a_records",
        &manifest.a_id_field,
        &manifest.a_fields,
        a_records,
    )?;

    // --- b_records table (dynamic columns) ---
    create_records_table(&conn, "b_records", &manifest.b_id_field, &manifest.b_fields)?;
    insert_records(
        &conn,
        "b_records",
        &manifest.b_id_field,
        &manifest.b_fields,
        b_records,
    )?;

    // --- relationships table ---
    conn.execute_batch(
        "CREATE TABLE relationships (
            a_id              TEXT    NOT NULL,
            b_id              TEXT    NOT NULL,
            score             REAL,
            rank              INTEGER,
            relationship_type TEXT    NOT NULL,
            reason            TEXT,
            PRIMARY KEY (a_id, b_id)
        )",
    )?;
    conn.execute_batch(
        "CREATE INDEX idx_relationships_a    ON relationships(a_id);
         CREATE INDEX idx_relationships_b    ON relationships(b_id);
         CREATE INDEX idx_relationships_type ON relationships(relationship_type);
         CREATE INDEX idx_relationships_b_rank ON relationships(b_id, rank);",
    )?;
    insert_relationships(&conn, relationships)?;

    // --- field_scores table (empty without scoring log) ---
    conn.execute_batch(
        "CREATE TABLE field_scores (
            a_id    TEXT NOT NULL,
            b_id    TEXT NOT NULL,
            field_a TEXT NOT NULL,
            field_b TEXT NOT NULL,
            method  TEXT NOT NULL,
            score   REAL NOT NULL,
            weight  REAL NOT NULL,
            FOREIGN KEY (a_id, b_id) REFERENCES relationships(a_id, b_id)
        );
        CREATE INDEX idx_fscores_relationship ON field_scores(a_id, b_id);",
    )?;

    // --- metadata table ---
    conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")?;
    insert_metadata(&conn, manifest, a_records.len(), b_records.len())?;

    // --- views ---
    let views_sql = include_str!("views.sql");
    conn.execute_batch(views_sql)?;

    // Flush and close
    drop(conn);

    // Atomic rename
    crate::util::rename_replacing(&tmp, path)?;
    Ok(())
}

fn create_records_table(
    conn: &Connection,
    table: &str,
    _id_field: &str,
    fields: &[String],
) -> rusqlite::Result<()> {
    // Always use "id" as the PK column so views work universally.
    // All other configured fields become additional columns.
    let mut cols = "id TEXT PRIMARY KEY".to_string();
    for f in fields {
        cols.push_str(&format!(", \"{}\" TEXT", f));
    }
    conn.execute_batch(&format!("CREATE TABLE {table} ({cols})"))?;
    Ok(())
}

fn insert_records(
    conn: &Connection,
    table: &str,
    _id_field: &str,
    fields: &[String],
    records: &HashMap<String, Record>,
) -> rusqlite::Result<()> {
    if records.is_empty() {
        return Ok(());
    }

    // Build column list: id + all configured fields.
    let mut col_names = vec!["id".to_string()];
    for f in fields {
        col_names.push(format!("\"{}\"", f));
    }
    let placeholders: Vec<&str> = col_names.iter().map(|_| "?").collect();
    let sql = format!(
        "INSERT OR REPLACE INTO {} ({}) VALUES ({})",
        table,
        col_names.join(", "),
        placeholders.join(", ")
    );

    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare_cached(&sql)?;
        for (id, rec) in records {
            let mut params: Vec<String> = vec![id.clone()];
            for f in fields {
                params.push(rec.get(f).cloned().unwrap_or_default());
            }
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            stmt.execute(param_refs.as_slice())?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn insert_relationships(conn: &Connection, relationships: &[Relationship]) -> rusqlite::Result<()> {
    if relationships.is_empty() {
        return Ok(());
    }

    let sql = "INSERT OR REPLACE INTO relationships \
               (a_id, b_id, score, rank, relationship_type, reason) \
               VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare_cached(sql)?;
        for rel in relationships {
            stmt.execute(rusqlite::params![
                rel.a_id,
                rel.b_id,
                rel.score,
                rel.rank,
                rel.relationship_type,
                rel.reason,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn insert_metadata(
    conn: &Connection,
    manifest: &OutputManifest,
    a_count: usize,
    b_count: usize,
) -> rusqlite::Result<()> {
    let sql = "INSERT INTO metadata (key, value) VALUES (?1, ?2)";
    let mut stmt = conn.prepare(sql)?;
    let pairs: Vec<(&str, String)> = vec![
        ("job_name", manifest.job_name.clone()),
        ("mode", format!("{:?}", manifest.mode)),
        ("auto_match_threshold", manifest.auto_match.to_string()),
        ("review_floor_threshold", manifest.review_floor.to_string()),
        (
            "min_score_gap",
            manifest
                .min_score_gap
                .map(|g| g.to_string())
                .unwrap_or_default(),
        ),
        ("model", manifest.model.clone()),
        ("top_n", manifest.top_n.to_string()),
        ("a_record_count", a_count.to_string()),
        ("b_record_count", b_count.to_string()),
        ("scoring_log_enabled", "false".to_string()),
        ("schema_version", "1".to_string()),
    ];
    for (k, v) in &pairs {
        stmt.execute(rusqlite::params![k, v])?;
    }
    Ok(())
}
