//! SQLite output DB builder.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::Connection;

use crate::models::Record;
use crate::store::sqlite::record_column;

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
    field_scores: &[super::build::FieldScoreRow],
    manifest: &OutputManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    // Write directly to final path (no atomic rename — this is a write-once
    // output file, and rename can fail on network/cloud filesystems).
    if path.exists() {
        std::fs::remove_file(path)?;
    }

    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode = DELETE; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = OFF;",
    )?;

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
    let has_field_scores = !field_scores.is_empty();
    insert_field_scores(&conn, field_scores)?;

    // --- metadata table ---
    conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")?;
    insert_metadata(
        &conn,
        manifest,
        a_records.len(),
        b_records.len(),
        has_field_scores,
    )?;

    // --- views ---
    let views_sql = include_str!("views.sql");
    conn.execute_batch(views_sql)?;

    // Flush and close
    drop(conn);
    Ok(())
}

fn create_records_table(
    conn: &Connection,
    table: &str,
    id_field: &str,
    fields: &[String],
) -> rusqlite::Result<()> {
    // Always use "id" as the PK column so views work universally.
    // All other configured fields become additional columns.
    let mut cols = "id TEXT PRIMARY KEY".to_string();
    for f in record_payload_fields(id_field, fields) {
        cols.push_str(&format!(", {} TEXT", record_column(f, fields)));
    }
    conn.execute_batch(&format!("CREATE TABLE {table} ({cols})"))?;
    Ok(())
}

// If the input ID is itself called "id", its value is already stored in the PK.
fn record_payload_fields<'a>(
    id_field: &'a str,
    fields: &'a [String],
) -> impl Iterator<Item = &'a String> {
    fields
        .iter()
        .filter(move |f| !(f.as_str() == id_field && f.eq_ignore_ascii_case("id")))
}

fn insert_records(
    conn: &Connection,
    table: &str,
    id_field: &str,
    fields: &[String],
    records: &HashMap<String, Record>,
) -> rusqlite::Result<()> {
    if records.is_empty() {
        return Ok(());
    }

    // Build column list: id + all configured fields.
    let mut col_names = vec!["id".to_string()];
    for f in record_payload_fields(id_field, fields) {
        col_names.push(record_column(f, fields));
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
            for f in record_payload_fields(id_field, fields) {
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
    has_field_scores: bool,
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
        (
            "scoring_log_enabled",
            if has_field_scores { "true" } else { "false" }.to_string(),
        ),
        ("schema_version", "1".to_string()),
    ];
    for (k, v) in &pairs {
        stmt.execute(rusqlite::params![k, v])?;
    }
    Ok(())
}

fn insert_field_scores(
    conn: &Connection,
    field_scores: &[super::build::FieldScoreRow],
) -> rusqlite::Result<()> {
    if field_scores.is_empty() {
        return Ok(());
    }
    let sql = "INSERT INTO field_scores (a_id, b_id, field_a, field_b, method, score, weight) \
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)";
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare_cached(sql)?;
        for fs in field_scores {
            stmt.execute(rusqlite::params![
                fs.a_id, fs.b_id, fs.field_a, fs.field_b, fs.method, fs.score, fs.weight,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::Mode;
    use crate::store::sqlite::record_column_name;

    #[test]
    fn output_db_preserves_distinct_payload_id_and_reserved_source_names()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        for source_id in ["id", "ID"] {
            let fields: Vec<String> = [
                "entity\"key",
                source_id,
                "__melder_source_id",
                "__MELDER_SOURCE_ID_",
                "quoted\"; DROP TABLE relationships; --",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            let record: Record = fields
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), format!("payload-{i}")))
                .collect();
            let records = HashMap::from([("internal-key".into(), record.clone())]);
            let manifest = OutputManifest {
                mode: Mode::Match,
                job_name: "mapping".into(),
                a_id_field: fields[0].clone(),
                b_id_field: fields[0].clone(),
                a_fields: fields.clone(),
                b_fields: fields.clone(),
                auto_match: 0.9,
                review_floor: 0.7,
                min_score_gap: None,
                top_n: 5,
                model: String::new(),
            };
            let path = dir.path().join(format!("{source_id}.db"));
            build_db(&path, &records, &records, &[], &[], &manifest)?;
            let conn = Connection::open(path)?;
            for table in ["a_records", "b_records"] {
                let projections = fields
                    .iter()
                    .map(|field| record_column(field, &fields))
                    .collect::<Vec<_>>()
                    .join(", ");
                let actual = conn.query_row(
                    &format!("SELECT id, {projections} FROM {table}"),
                    [],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            fields
                                .iter()
                                .enumerate()
                                .map(|(i, _)| row.get::<_, String>(i + 1))
                                .collect::<rusqlite::Result<Vec<_>>>()?,
                        ))
                    },
                )?;
                assert_eq!(actual.0, "internal-key", "output PK must stay internal");
                assert_eq!(
                    actual.1,
                    fields.iter().map(|f| record[f].clone()).collect::<Vec<_>>(),
                    "output payload must not be replaced by the internal key"
                );
                assert_ne!(
                    record_column_name(source_id, &fields),
                    "id",
                    "payload id must have a separate physical column"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn output_db_does_not_duplicate_id_when_it_is_the_configured_key() -> rusqlite::Result<()> {
        let conn = Connection::open_in_memory()?;
        let fields = vec!["id".to_owned(), "name".to_owned()];
        create_records_table(&conn, "a_records", "id", &fields)?;
        insert_records(
            &conn,
            "a_records",
            "id",
            &fields,
            &HashMap::from([(
                "key".into(),
                Record::from([("id".into(), "key".into()), ("name".into(), "Alice".into())]),
            )]),
        )?;
        let row: (String, String) = conn.query_row("SELECT * FROM a_records", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        assert_eq!(row, ("key".into(), "Alice".into()), "one key column only");
        Ok(())
    }
}
