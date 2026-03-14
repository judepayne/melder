//! SQLite-backed record store for durable live-mode persistence.
//!
//! Uses a single `Mutex<Connection>` for serialized access. SQLite WAL mode
//! ensures writes don't block reads at the filesystem level. The page cache
//! (`PRAGMA cache_size = -65536`) keeps hot index pages in memory.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, params};

use crate::config::{BlockingConfig, BlockingFieldPair};
use crate::models::{Record, Side};

use super::RecordStore;

/// SQLite-backed record store.
///
/// Both sides (A and B) live in the same database file with separate tables.
/// Blocking keys are stored in indexed tables for fast candidate lookups.
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    blocking_config: BlockingConfig,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore")
            .field("a_records", &self.len(Side::A))
            .field("b_records", &self.len(Side::B))
            .finish()
    }
}

// ── Schema + factory ─────────────────────────────────────────────────────────

/// SQL statements to create all tables and indices.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS a_records (
    id          TEXT PRIMARY KEY,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS b_records (
    id          TEXT PRIMARY KEY,
    record_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS a_blocking_keys (
    record_id   TEXT    NOT NULL,
    field_index INTEGER NOT NULL,
    value       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a_blocking ON a_blocking_keys(field_index, value);

CREATE TABLE IF NOT EXISTS b_blocking_keys (
    record_id   TEXT    NOT NULL,
    field_index INTEGER NOT NULL,
    value       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_b_blocking ON b_blocking_keys(field_index, value);

CREATE TABLE IF NOT EXISTS a_unmatched (id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS b_unmatched (id TEXT PRIMARY KEY);

CREATE TABLE IF NOT EXISTS a_common_ids (
    common_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS b_common_ids (
    common_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crossmap (
    a_id TEXT NOT NULL UNIQUE,
    b_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS reviews (
    key          TEXT PRIMARY KEY,
    id           TEXT NOT NULL,
    side         TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    score        REAL NOT NULL
);
";

/// Open a SQLite database and create all tables.
///
/// Returns the `SqliteStore`, `SqliteCrossMap`, and the shared
/// `Arc<Mutex<Connection>>` for use by other components (e.g., review
/// queue write-through).
pub fn open_sqlite(
    path: &Path,
    blocking_config: &BlockingConfig,
) -> Result<
    (
        SqliteStore,
        crate::crossmap::sqlite::SqliteCrossMap,
        Arc<Mutex<Connection>>,
    ),
    rusqlite::Error,
> {
    let conn = Connection::open(path)?;

    // Performance pragmas
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA cache_size = -65536;
         PRAGMA page_size = 8192;
         PRAGMA foreign_keys = OFF;
         PRAGMA synchronous = NORMAL;",
    )?;

    // Create schema
    conn.execute_batch(SCHEMA)?;

    let conn = Arc::new(Mutex::new(conn));

    let store = SqliteStore {
        conn: Arc::clone(&conn),
        blocking_config: blocking_config.clone(),
    };

    let crossmap = crate::crossmap::sqlite::SqliteCrossMap::from_conn(Arc::clone(&conn));

    Ok((store, crossmap, conn))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Table name prefix for a given side.
fn side_prefix(side: Side) -> &'static str {
    match side {
        Side::A => "a",
        Side::B => "b",
    }
}

/// Serialize a Record to JSON.
fn record_to_json(record: &Record) -> String {
    serde_json::to_string(record).expect("Record serialization cannot fail")
}

/// Deserialize a Record from JSON.
fn record_from_json(json: &str) -> Record {
    serde_json::from_str(json).expect("Record deserialization cannot fail")
}

/// Extract blocking key values from a record for a given side.
fn blocking_values(
    record: &Record,
    fields: &[BlockingFieldPair],
    side: Side,
) -> Vec<(usize, String)> {
    fields
        .iter()
        .enumerate()
        .filter_map(|(i, fp)| {
            let field = match side {
                Side::A => &fp.field_a,
                Side::B => &fp.field_b,
            };
            let val = record
                .get(field)
                .map(|s| s.trim().to_lowercase())
                .unwrap_or_default();
            if val.is_empty() { None } else { Some((i, val)) }
        })
        .collect()
}

// ── RecordStore implementation ───────────────────────────────────────────────

impl RecordStore for SqliteStore {
    // --- Records ---

    fn get(&self, side: Side, id: &str) -> Option<Record> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!(
            "SELECT record_json FROM {}_records WHERE id = ?1",
            side_prefix(side)
        );
        conn.query_row(&sql, params![id], |row| {
            let json: String = row.get(0)?;
            Ok(record_from_json(&json))
        })
        .ok()
    }

    fn insert(&self, side: Side, id: &str, record: &Record) -> Option<Record> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);

        // Fetch old record if it exists (for return value)
        let old: Option<Record> = conn
            .query_row(
                &format!("SELECT record_json FROM {}_records WHERE id = ?1", prefix),
                params![id],
                |row| {
                    let json: String = row.get(0)?;
                    Ok(record_from_json(&json))
                },
            )
            .ok();

        // Insert or replace
        let json = record_to_json(record);
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {}_records (id, record_json) VALUES (?1, ?2)",
                prefix
            ),
            params![id, json],
        )
        .expect("insert record");

        old
    }

    fn remove(&self, side: Side, id: &str) -> Option<Record> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);

        // Fetch old record
        let old: Option<Record> = conn
            .query_row(
                &format!("SELECT record_json FROM {}_records WHERE id = ?1", prefix),
                params![id],
                |row| {
                    let json: String = row.get(0)?;
                    Ok(record_from_json(&json))
                },
            )
            .ok();

        if old.is_some() {
            conn.execute(
                &format!("DELETE FROM {}_records WHERE id = ?1", prefix),
                params![id],
            )
            .expect("delete record");
        }

        old
    }

    fn contains(&self, side: Side, id: &str) -> bool {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!("SELECT 1 FROM {}_records WHERE id = ?1", side_prefix(side));
        conn.query_row(&sql, params![id], |_| Ok(())).is_ok()
    }

    fn len(&self, side: Side) -> usize {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!("SELECT COUNT(*) FROM {}_records", side_prefix(side));
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    fn ids(&self, side: Side) -> Vec<String> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!("SELECT id FROM {}_records", side_prefix(side));
        let mut stmt = conn.prepare(&sql).expect("prepare ids query");
        stmt.query_map([], |row| row.get(0))
            .expect("query ids")
            .filter_map(|r| r.ok())
            .collect()
    }

    // --- Blocking ---

    fn blocking_insert(&self, side: Side, id: &str, record: &Record) {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        let values = blocking_values(record, &self.blocking_config.fields, side);

        for (field_index, value) in values {
            conn.execute(
                &format!(
                    "INSERT INTO {}_blocking_keys (record_id, field_index, value) VALUES (?1, ?2, ?3)",
                    prefix
                ),
                params![id, field_index as i64, value],
            )
            .expect("insert blocking key");
        }
    }

    fn blocking_remove(&self, side: Side, id: &str, _record: &Record) {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_blocking_keys WHERE record_id = ?1", prefix),
            params![id],
        )
        .expect("delete blocking keys");
    }

    fn blocking_query(&self, record: &Record, query_side: Side) -> Vec<String> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let fields = &self.blocking_config.fields;
        let opp = query_side.opposite();
        let prefix = side_prefix(opp);
        let is_and = self.blocking_config.operator.eq_ignore_ascii_case("and");

        // Collect query values: for a B-side query, we use field_b from the
        // query record and look up in the A-side blocking table.
        let query_values: Vec<(usize, String)> = fields
            .iter()
            .enumerate()
            .filter_map(|(i, fp)| {
                let field = match query_side {
                    Side::A => &fp.field_a,
                    Side::B => &fp.field_b,
                };
                let val = record
                    .get(field)
                    .map(|s| s.trim().to_lowercase())
                    .unwrap_or_default();
                if val.is_empty() { None } else { Some((i, val)) }
            })
            .collect();

        // If no query values, return all IDs from the opposite side
        if query_values.is_empty() {
            let sql = format!("SELECT id FROM {}_records", prefix);
            let mut stmt = conn.prepare(&sql).expect("prepare all-ids query");
            return stmt
                .query_map([], |row| row.get(0))
                .expect("query all ids")
                .filter_map(|r| r.ok())
                .collect();
        }

        // Build the SQL query dynamically based on the number of non-empty fields
        let conditions: Vec<String> = query_values
            .iter()
            .map(|(i, _)| format!("(field_index = {} AND value = ?)", i))
            .collect();
        let where_clause = conditions.join(" OR ");

        let sql = if is_and {
            // AND mode: all fields must match
            format!(
                "SELECT record_id FROM {}_blocking_keys WHERE {} GROUP BY record_id HAVING COUNT(DISTINCT field_index) = {}",
                prefix,
                where_clause,
                query_values.len()
            )
        } else {
            // OR mode: any field match
            format!(
                "SELECT DISTINCT record_id FROM {}_blocking_keys WHERE {}",
                prefix, where_clause
            )
        };

        let mut stmt = conn.prepare(&sql).expect("prepare blocking query");
        let param_values: Vec<&dyn rusqlite::types::ToSql> = query_values
            .iter()
            .map(|(_, v)| v as &dyn rusqlite::types::ToSql)
            .collect();

        stmt.query_map(param_values.as_slice(), |row| row.get(0))
            .expect("execute blocking query")
            .filter_map(|r| r.ok())
            .collect()
    }

    // --- Unmatched ---

    fn mark_unmatched(&self, side: Side, id: &str) {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!(
                "INSERT OR IGNORE INTO {}_unmatched (id) VALUES (?1)",
                prefix
            ),
            params![id],
        )
        .expect("mark unmatched");
    }

    fn mark_matched(&self, side: Side, id: &str) {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_unmatched WHERE id = ?1", prefix),
            params![id],
        )
        .expect("mark matched");
    }

    fn is_unmatched(&self, side: Side, id: &str) -> bool {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!(
            "SELECT 1 FROM {}_unmatched WHERE id = ?1",
            side_prefix(side)
        );
        conn.query_row(&sql, params![id], |_| Ok(())).is_ok()
    }

    fn unmatched_count(&self, side: Side) -> usize {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!("SELECT COUNT(*) FROM {}_unmatched", side_prefix(side));
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    fn unmatched_ids(&self, side: Side) -> Vec<String> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!("SELECT id FROM {}_unmatched ORDER BY id", side_prefix(side));
        let mut stmt = conn.prepare(&sql).expect("prepare unmatched query");
        stmt.query_map([], |row| row.get(0))
            .expect("query unmatched")
            .filter_map(|r| r.ok())
            .collect()
    }

    // --- Common ID index ---

    fn common_id_insert(&self, side: Side, common_id: &str, record_id: &str) {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {}_common_ids (common_id, record_id) VALUES (?1, ?2)",
                prefix
            ),
            params![common_id, record_id],
        )
        .expect("insert common id");
    }

    fn common_id_lookup(&self, side: Side, common_id: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let sql = format!(
            "SELECT record_id FROM {}_common_ids WHERE common_id = ?1",
            side_prefix(side)
        );
        conn.query_row(&sql, params![common_id], |row| row.get(0))
            .ok()
    }

    fn common_id_remove(&self, side: Side, common_id: &str) {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_common_ids WHERE common_id = ?1", prefix),
            params![common_id],
        )
        .expect("remove common id");
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingFieldPair;
    use crate::store::RecordStore;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn make_store() -> SqliteStore {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let bc = BlockingConfig {
            enabled: true,
            operator: "and".to_string(),
            fields: vec![BlockingFieldPair {
                field_a: "country_a".to_string(),
                field_b: "country_b".to_string(),
            }],
            field_a: None,
            field_b: None,
        };
        let (store, _crossmap, _conn) = open_sqlite(&path, &bc).unwrap();
        // Keep tempdir alive by leaking it — tests are short-lived
        std::mem::forget(dir);
        store
    }

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<String, String>>()
    }

    #[test]
    fn insert_and_get() {
        let store = make_store();
        let rec = make_record(&[("name", "Alice")]);
        assert!(store.insert(Side::A, "1", &rec).is_none());
        let got = store.get(Side::A, "1").unwrap();
        assert_eq!(got.get("name").unwrap(), "Alice");
    }

    #[test]
    fn insert_replaces_and_returns_old() {
        let store = make_store();
        let r1 = make_record(&[("name", "Alice")]);
        let r2 = make_record(&[("name", "Bob")]);
        assert!(store.insert(Side::A, "1", &r1).is_none());
        let old = store.insert(Side::A, "1", &r2).unwrap();
        assert_eq!(old.get("name").unwrap(), "Alice");
        assert_eq!(store.get(Side::A, "1").unwrap().get("name").unwrap(), "Bob");
    }

    #[test]
    fn remove_returns_old() {
        let store = make_store();
        let rec = make_record(&[("name", "Alice")]);
        store.insert(Side::A, "1", &rec);
        let old = store.remove(Side::A, "1").unwrap();
        assert_eq!(old.get("name").unwrap(), "Alice");
        assert!(store.get(Side::A, "1").is_none());
        assert!(store.remove(Side::A, "1").is_none());
    }

    #[test]
    fn contains_and_len() {
        let store = make_store();
        assert!(!store.contains(Side::A, "1"));
        assert_eq!(store.len(Side::A), 0);
        let rec = make_record(&[("name", "Alice")]);
        store.insert(Side::A, "1", &rec);
        assert!(store.contains(Side::A, "1"));
        assert_eq!(store.len(Side::A), 1);
    }

    #[test]
    fn ids_returns_all() {
        let store = make_store();
        let rec = make_record(&[("name", "x")]);
        store.insert(Side::A, "2", &rec);
        store.insert(Side::A, "1", &rec);
        store.insert(Side::A, "3", &rec);
        let mut ids = store.ids(Side::A);
        ids.sort();
        assert_eq!(ids, vec!["1", "2", "3"]);
    }

    #[test]
    fn sides_are_independent() {
        let store = make_store();
        let rec = make_record(&[("name", "x")]);
        store.insert(Side::A, "1", &rec);
        assert!(store.contains(Side::A, "1"));
        assert!(!store.contains(Side::B, "1"));
        assert_eq!(store.len(Side::A), 1);
        assert_eq!(store.len(Side::B), 0);
    }

    #[test]
    fn from_records_populates_all() {
        // SqliteStore doesn't have from_records, so test insert + blocking
        let store = make_store();
        let rec = make_record(&[("name", "Alice"), ("country_a", "US")]);
        store.insert(Side::A, "1", &rec);
        store.blocking_insert(Side::A, "1", &rec);
        assert_eq!(store.len(Side::A), 1);
    }

    #[test]
    fn blocking_insert_query_remove() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let bc = BlockingConfig {
            enabled: true,
            operator: "and".to_string(),
            fields: vec![BlockingFieldPair {
                field_a: "country_a".to_string(),
                field_b: "country_b".to_string(),
            }],
            field_a: None,
            field_b: None,
        };
        let (store, _, _conn) = open_sqlite(&path, &bc).unwrap();

        // Insert A-side records with blocking keys
        let r1 = make_record(&[("name", "Alice"), ("country_a", "US")]);
        let r2 = make_record(&[("name", "Bob"), ("country_a", "UK")]);
        store.insert(Side::A, "1", &r1);
        store.blocking_insert(Side::A, "1", &r1);
        store.insert(Side::A, "2", &r2);
        store.blocking_insert(Side::A, "2", &r2);

        // Query from B-side: country_b = "us" should match record 1
        let query = make_record(&[("country_b", "US")]);
        let results = store.blocking_query(&query, Side::B);
        assert_eq!(results.len(), 1, "should find 1 match for US");
        assert_eq!(results[0], "1");

        // Remove blocking keys for record 1
        store.blocking_remove(Side::A, "1", &r1);
        let results2 = store.blocking_query(&query, Side::B);
        assert!(results2.is_empty(), "should find 0 after remove");

        std::mem::forget(dir);
    }

    #[test]
    fn unmatched_lifecycle() {
        let store = make_store();
        assert_eq!(store.unmatched_count(Side::A), 0);
        store.mark_unmatched(Side::A, "1");
        store.mark_unmatched(Side::A, "2");
        assert_eq!(store.unmatched_count(Side::A), 2);
        assert!(store.is_unmatched(Side::A, "1"));
        store.mark_matched(Side::A, "1");
        assert!(!store.is_unmatched(Side::A, "1"));
        assert_eq!(store.unmatched_count(Side::A), 1);
    }

    #[test]
    fn unmatched_ids_sorted() {
        let store = make_store();
        store.mark_unmatched(Side::A, "c");
        store.mark_unmatched(Side::A, "a");
        store.mark_unmatched(Side::A, "b");
        assert_eq!(store.unmatched_ids(Side::A), vec!["a", "b", "c"]);
    }

    #[test]
    fn common_id_lifecycle() {
        let store = make_store();
        store.common_id_insert(Side::A, "CID-1", "REC-1");
        assert_eq!(
            store.common_id_lookup(Side::A, "CID-1"),
            Some("REC-1".to_string())
        );
        store.common_id_remove(Side::A, "CID-1");
        assert!(store.common_id_lookup(Side::A, "CID-1").is_none());
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("persist.db");
        let bc = BlockingConfig {
            enabled: false,
            operator: "and".to_string(),
            fields: vec![],
            field_a: None,
            field_b: None,
        };

        // Insert data
        {
            let (store, _, _conn) = open_sqlite(&path, &bc).unwrap();
            let rec = make_record(&[("name", "Alice")]);
            store.insert(Side::A, "1", &rec);
            store.mark_unmatched(Side::A, "1");
            store.common_id_insert(Side::A, "CID-1", "1");
        }

        // Reopen and verify
        {
            let (store, _, _conn) = open_sqlite(&path, &bc).unwrap();
            assert_eq!(store.len(Side::A), 1);
            assert_eq!(
                store.get(Side::A, "1").unwrap().get("name").unwrap(),
                "Alice"
            );
            assert!(store.is_unmatched(Side::A, "1"));
            assert_eq!(
                store.common_id_lookup(Side::A, "CID-1"),
                Some("1".to_string())
            );
        }

        std::mem::forget(dir);
    }
}
