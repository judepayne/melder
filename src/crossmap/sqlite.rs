//! `SqliteCrossMap`: SQLite-backed bidirectional record-pair mapping.
//!
//! Uses the shared writer + reader pool architecture from `SqliteStore`.
//! Writes (add, remove, claim, take) go through the writer connection.
//! Reads (get, has, len, pairs) go through the reader pool.

use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection};

use super::CrossMapOps;
use crate::store::sqlite::SqliteReaderPool;

/// SQLite-backed bidirectional mapping between A and B record IDs.
///
/// Writes use a single `Mutex<Connection>`. Reads use the shared
/// `SqliteReaderPool` for concurrent access. Mutations are immediately
/// durable (no explicit flush needed).
pub struct SqliteCrossMap {
    writer: Arc<Mutex<Connection>>,
    reader_pool: Arc<SqliteReaderPool>,
}

impl std::fmt::Debug for SqliteCrossMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteCrossMap")
            .field("len", &self.len())
            .finish()
    }
}

impl SqliteCrossMap {
    /// Create from a shared writer connection and reader pool.
    ///
    /// The `crossmap` table must already exist (created by `open_sqlite()`).
    pub fn new(writer: Arc<Mutex<Connection>>, reader_pool: Arc<SqliteReaderPool>) -> Self {
        Self {
            writer,
            reader_pool,
        }
    }
}

impl CrossMapOps for SqliteCrossMap {
    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(()) // Write-through: mutations are immediately durable.
    }

    fn add(&self, a_id: &str, b_id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute("DELETE FROM crossmap WHERE a_id = ?1", params![a_id])
            .expect("delete old a_id");
        conn.execute("DELETE FROM crossmap WHERE b_id = ?1", params![b_id])
            .expect("delete old b_id");
        conn.execute(
            "INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?2)",
            params![a_id, b_id],
        )
        .expect("insert crossmap pair");
    }

    fn remove(&self, a_id: &str, b_id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "DELETE FROM crossmap WHERE a_id = ?1 AND b_id = ?2",
            params![a_id, b_id],
        )
        .expect("remove crossmap pair");
    }

    fn remove_if_exact(&self, a_id: &str, b_id: &str) -> bool {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let changed = conn
            .execute(
                "DELETE FROM crossmap WHERE a_id = ?1 AND b_id = ?2",
                params![a_id, b_id],
            )
            .expect("remove_if_exact");
        changed > 0
    }

    fn take_a(&self, a_id: &str) -> Option<String> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.query_row(
            "DELETE FROM crossmap WHERE a_id = ?1 RETURNING b_id",
            params![a_id],
            |row| row.get(0),
        )
        .ok()
    }

    fn take_b(&self, b_id: &str) -> Option<String> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.query_row(
            "DELETE FROM crossmap WHERE b_id = ?1 RETURNING a_id",
            params![b_id],
            |row| row.get(0),
        )
        .ok()
    }

    fn claim(&self, a_id: &str, b_id: &str) -> bool {
        // Must use writer: check-then-insert must be atomic on one connection.
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let a_taken: bool = conn
            .query_row(
                "SELECT 1 FROM crossmap WHERE a_id = ?1",
                params![a_id],
                |_| Ok(()),
            )
            .is_ok();
        if a_taken {
            return false;
        }
        let b_taken: bool = conn
            .query_row(
                "SELECT 1 FROM crossmap WHERE b_id = ?1",
                params![b_id],
                |_| Ok(()),
            )
            .is_ok();
        if b_taken {
            return false;
        }
        conn.execute(
            "INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?2)",
            params![a_id, b_id],
        )
        .expect("claim insert");
        true
    }

    fn get_b(&self, a_id: &str) -> Option<String> {
        let conn = self.reader_pool.acquire();
        conn.query_row(
            "SELECT b_id FROM crossmap WHERE a_id = ?1",
            params![a_id],
            |row: &rusqlite::Row| row.get(0),
        )
        .ok()
    }

    fn get_a(&self, b_id: &str) -> Option<String> {
        let conn = self.reader_pool.acquire();
        conn.query_row(
            "SELECT a_id FROM crossmap WHERE b_id = ?1",
            params![b_id],
            |row: &rusqlite::Row| row.get(0),
        )
        .ok()
    }

    fn has_a(&self, a_id: &str) -> bool {
        let conn = self.reader_pool.acquire();
        conn.query_row(
            "SELECT 1 FROM crossmap WHERE a_id = ?1",
            params![a_id],
            |_| Ok(()),
        )
        .is_ok()
    }

    fn has_b(&self, b_id: &str) -> bool {
        let conn = self.reader_pool.acquire();
        conn.query_row(
            "SELECT 1 FROM crossmap WHERE b_id = ?1",
            params![b_id],
            |_| Ok(()),
        )
        .is_ok()
    }

    fn len(&self) -> usize {
        let conn = self.reader_pool.acquire();
        conn.query_row(
            "SELECT COUNT(*) FROM crossmap",
            [],
            |row: &rusqlite::Row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as usize
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn pairs(&self) -> Vec<(String, String)> {
        let conn = self.reader_pool.acquire();
        let mut stmt = conn
            .prepare("SELECT a_id, b_id FROM crossmap")
            .expect("prepare pairs query");
        stmt.query_map([], |row: &rusqlite::Row| Ok((row.get(0)?, row.get(1)?)))
            .expect("query pairs")
            .filter_map(|r: Result<(String, String), _>| r.ok())
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingConfig;
    use crate::crossmap::CrossMapOps;
    use tempfile::tempdir;

    fn make_crossmap() -> SqliteCrossMap {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let bc = BlockingConfig {
            enabled: false,
            operator: "and".to_string(),
            fields: vec![],
            field_a: None,
            field_b: None,
        };
        let pool_cfg = crate::store::sqlite::SqlitePoolConfig {
            read_pool_size: 2,
            ..Default::default()
        };
        let (_store, crossmap, _conn) =
            crate::store::sqlite::open_sqlite(&path, &bc, Some(pool_cfg)).unwrap();
        std::mem::forget(dir);
        crossmap
    }

    #[test]
    fn add_and_lookup() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
        assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
        assert_eq!(cm.get_a("B-2"), Some("A-2".to_string()));
        assert!(cm.has_a("A-1"));
        assert!(cm.has_b("B-1"));
        assert!(!cm.has_a("A-3"));
        assert_eq!(cm.len(), 2);
    }

    #[test]
    fn remove_pair() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        cm.remove("A-1", "B-1");
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 1);
        assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
    }

    #[test]
    fn remove_if_exact_matches() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        assert!(cm.remove_if_exact("A-1", "B-1"));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
    }

    #[test]
    fn remove_if_exact_wrong_pair_is_noop() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        assert!(!cm.remove_if_exact("A-1", "B-99"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn take_a_removes_both_directions() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        assert_eq!(cm.take_a("A-1"), Some("B-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_b_removes_both_directions() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        assert_eq!(cm.take_b("B-1"), Some("A-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_a_missing_returns_none() {
        let cm = make_crossmap();
        assert_eq!(cm.take_a("A-99"), None);
    }

    #[test]
    fn claim_vacant_succeeds() {
        let cm = make_crossmap();
        assert!(cm.claim("A-1", "B-1"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
    }

    #[test]
    fn claim_b_occupied_fails() {
        let cm = make_crossmap();
        assert!(cm.claim("A-1", "B-1"));
        assert!(!cm.claim("A-2", "B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn claim_a_occupied_fails() {
        let cm = make_crossmap();
        assert!(cm.claim("A-1", "B-1"));
        assert!(!cm.claim("A-1", "B-2"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert!(cm.get_a("B-2").is_none());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn claim_then_break_then_reclaim() {
        let cm = make_crossmap();
        assert!(cm.claim("A-1", "B-1"));
        cm.remove("A-1", "B-1");
        assert!(cm.claim("A-2", "B-1"));
        assert_eq!(cm.get_a("B-1"), Some("A-2".to_string()));
    }

    #[test]
    fn pairs_returns_all() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1");
        cm.add("A-2", "B-2");

        let mut pairs = cm.pairs();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("A-1".to_string(), "B-1".to_string()),
                ("A-2".to_string(), "B-2".to_string()),
            ]
        );
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
            let (_store, cm, _conn) = crate::store::sqlite::open_sqlite(&path, &bc, None).unwrap();
            cm.add("A-1", "B-1");
            cm.add("A-2", "B-2");
        }

        // Reopen and verify
        {
            let (_store, cm, _conn) = crate::store::sqlite::open_sqlite(&path, &bc, None).unwrap();
            assert_eq!(cm.len(), 2);
            assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
            assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
        }

        std::mem::forget(dir);
    }
}
