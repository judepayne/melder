//! `SqliteCrossMap`: SQLite-backed bidirectional record-pair mapping.
//!
//! Uses the shared writer + reader pool architecture from `SqliteStore`.
//! Writes (add, remove, claim, take) go through the writer connection.
//! Reads (get, has, len, pairs) go through the reader pool.

use std::sync::{Arc, Mutex};

use rusqlite::{Connection, params};

use super::{ConfirmOutcome, CrossMapOps};
use crate::error::CrossMapError;
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

    fn add(&self, a_id: &str, b_id: &str) -> Result<(), CrossMapError> {
        let mut conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(e) => {
                tracing::warn!(error = %e, "crossmap add: failed to begin transaction");
                return Err(CrossMapError::from(e));
            }
        };
        if let Err(e) = tx.execute("DELETE FROM crossmap WHERE a_id = ?1", params![a_id]) {
            tracing::warn!(error = %e, a_id, "crossmap add: failed to delete old a_id");
            return Err(CrossMapError::from(e));
        }
        if let Err(e) = tx.execute("DELETE FROM crossmap WHERE b_id = ?1", params![b_id]) {
            tracing::warn!(error = %e, b_id, "crossmap add: failed to delete old b_id");
            return Err(CrossMapError::from(e));
        }
        if let Err(e) = tx.execute(
            "INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?2)",
            params![a_id, b_id],
        ) {
            tracing::warn!(error = %e, a_id, b_id, "crossmap add: failed to insert pair");
            return Err(CrossMapError::from(e));
        }
        if let Err(e) = tx.commit() {
            tracing::warn!(error = %e, a_id, b_id, "crossmap add: failed to commit");
            return Err(CrossMapError::from(e));
        }
        Ok(())
    }

    fn remove(&self, a_id: &str, b_id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = conn.execute(
            "DELETE FROM crossmap WHERE a_id = ?1 AND b_id = ?2",
            params![a_id, b_id],
        ) {
            tracing::warn!(error = %e, a_id, b_id, "crossmap remove failed");
        }
    }

    fn remove_if_exact(&self, a_id: &str, b_id: &str) -> bool {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        match conn.execute(
            "DELETE FROM crossmap WHERE a_id = ?1 AND b_id = ?2",
            params![a_id, b_id],
        ) {
            Ok(changed) => changed > 0,
            Err(e) => {
                tracing::warn!(error = %e, a_id, b_id, "crossmap remove_if_exact failed");
                false
            }
        }
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
        // Wrapped in a transaction so the two SELECTs + INSERT are atomic,
        // and any error triggers automatic rollback via Transaction::drop.
        let mut conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(e) => {
                tracing::warn!(error = %e, "crossmap claim: failed to begin transaction");
                return false;
            }
        };
        let a_taken: bool = tx
            .query_row(
                "SELECT 1 FROM crossmap WHERE a_id = ?1",
                params![a_id],
                |_| Ok(()),
            )
            .is_ok();
        if a_taken {
            return false;
        }
        let b_taken: bool = tx
            .query_row(
                "SELECT 1 FROM crossmap WHERE b_id = ?1",
                params![b_id],
                |_| Ok(()),
            )
            .is_ok();
        if b_taken {
            return false;
        }
        if let Err(e) = tx.execute(
            "INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?2)",
            params![a_id, b_id],
        ) {
            tracing::warn!(error = %e, a_id, b_id, "crossmap claim: failed to insert");
            return false;
        }
        match tx.commit() {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(error = %e, a_id, b_id, "crossmap claim: failed to commit");
                false
            }
        }
    }

    fn confirm(&self, a_id: &str, b_id: &str) -> Result<ConfirmOutcome, CrossMapError> {
        // Single transaction: take both prior partners (if any), then insert
        // the new pair. Bijection is preserved atomically.
        let mut conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(e) => {
                tracing::warn!(error = %e, "crossmap confirm: failed to begin transaction");
                return Err(CrossMapError::from(e));
            }
        };

        // Idempotent no-op: pair already installed.
        let existing_b: Option<String> = tx
            .query_row(
                "SELECT b_id FROM crossmap WHERE a_id = ?1",
                params![a_id],
                |row| row.get(0),
            )
            .ok();
        if existing_b.as_deref() == Some(b_id) {
            return Ok(ConfirmOutcome::default());
        }

        // Take a_id's prior partner (if any). RETURNING gives us the b_id
        // that was deleted in the same statement.
        let displaced_b: Option<String> = tx
            .query_row(
                "DELETE FROM crossmap WHERE a_id = ?1 RETURNING b_id",
                params![a_id],
                |row| row.get(0),
            )
            .ok();

        // Take b_id's prior partner (if any).
        let displaced_a: Option<String> = tx
            .query_row(
                "DELETE FROM crossmap WHERE b_id = ?1 RETURNING a_id",
                params![b_id],
                |row| row.get(0),
            )
            .ok();

        if let Err(e) = tx.execute(
            "INSERT INTO crossmap (a_id, b_id) VALUES (?1, ?2)",
            params![a_id, b_id],
        ) {
            tracing::warn!(error = %e, a_id, b_id, "crossmap confirm: failed to insert pair");
            return Err(CrossMapError::from(e));
        }

        if let Err(e) = tx.commit() {
            tracing::warn!(error = %e, a_id, b_id, "crossmap confirm: failed to commit");
            return Err(CrossMapError::from(e));
        }

        Ok(ConfirmOutcome {
            displaced_b,
            displaced_a,
        })
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
        let mut stmt = match conn.prepare("SELECT a_id, b_id FROM crossmap") {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "pairs: failed to prepare query");
                return Vec::new();
            }
        };
        match stmt.query_map([], |row: &rusqlite::Row| Ok((row.get(0)?, row.get(1)?))) {
            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                tracing::warn!(error = %e, "pairs: failed to execute query");
                Vec::new()
            }
        }
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
            crate::store::sqlite::open_sqlite(&path, &bc, Some(pool_cfg), &[], &[]).unwrap();
        std::mem::forget(dir);
        crossmap
    }

    #[test]
    fn add_and_lookup() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
        cm.add("A-2", "B-2").unwrap();

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
        cm.add("A-1", "B-1").unwrap();
        cm.add("A-2", "B-2").unwrap();

        cm.remove("A-1", "B-1");
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 1);
        assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
    }

    #[test]
    fn remove_if_exact_matches() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
        assert!(cm.remove_if_exact("A-1", "B-1"));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
    }

    #[test]
    fn remove_if_exact_wrong_pair_is_noop() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
        assert!(!cm.remove_if_exact("A-1", "B-99"));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn take_a_removes_both_directions() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
        assert_eq!(cm.take_a("A-1"), Some("B-1".to_string()));
        assert!(cm.get_b("A-1").is_none());
        assert!(cm.get_a("B-1").is_none());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn take_b_removes_both_directions() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
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
    fn confirm_into_vacant_returns_no_displacements() {
        let cm = make_crossmap();
        let outcome = cm.confirm("A-1", "B-1").unwrap();
        assert_eq!(outcome.displaced_a, None);
        assert_eq!(outcome.displaced_b, None);
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert_eq!(cm.get_a("B-1"), Some("A-1".to_string()));
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn confirm_idempotent_pair_is_noop() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
        let outcome = cm.confirm("A-1", "B-1").unwrap();
        assert_eq!(outcome, ConfirmOutcome::default());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn confirm_displaces_old_b_when_a_was_paired() {
        let cm = make_crossmap();
        cm.add("A-1", "B-old").unwrap();
        let outcome = cm.confirm("A-1", "B-new").unwrap();
        assert_eq!(outcome.displaced_b, Some("B-old".to_string()));
        assert_eq!(outcome.displaced_a, None);
        assert_eq!(cm.get_b("A-1"), Some("B-new".to_string()));
        assert!(cm.get_a("B-old").is_none());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn confirm_displaces_old_a_when_b_was_paired() {
        let cm = make_crossmap();
        cm.add("A-old", "B-1").unwrap();
        let outcome = cm.confirm("A-new", "B-1").unwrap();
        assert_eq!(outcome.displaced_a, Some("A-old".to_string()));
        assert_eq!(outcome.displaced_b, None);
        assert_eq!(cm.get_b("A-new"), Some("B-1".to_string()));
        assert!(cm.get_b("A-old").is_none());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn confirm_displaces_both_sides() {
        let cm = make_crossmap();
        cm.add("A-1", "B-old").unwrap();
        cm.add("A-old", "B-1").unwrap();
        let outcome = cm.confirm("A-1", "B-1").unwrap();
        assert_eq!(outcome.displaced_b, Some("B-old".to_string()));
        assert_eq!(outcome.displaced_a, Some("A-old".to_string()));
        assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
        assert!(cm.get_a("B-old").is_none());
        assert!(cm.get_b("A-old").is_none());
        assert_eq!(cm.len(), 1);
    }

    #[test]
    fn add_returns_error_when_table_missing() {
        let cm = make_crossmap();
        {
            let conn = cm.writer.lock().unwrap_or_else(|e| e.into_inner());
            conn.execute("DROP TABLE crossmap", []).unwrap();
        }

        let err = cm.add("A-1", "B-1").unwrap_err();
        assert!(
            matches!(err, CrossMapError::Sqlite(_)),
            "expected sqlite error, got {err:?}"
        );
    }

    #[test]
    fn confirm_returns_error_when_table_missing() {
        let cm = make_crossmap();
        {
            let conn = cm.writer.lock().unwrap_or_else(|e| e.into_inner());
            conn.execute("DROP TABLE crossmap", []).unwrap();
        }

        let err = cm.confirm("A-1", "B-1").unwrap_err();
        assert!(
            matches!(err, CrossMapError::Sqlite(_)),
            "expected sqlite error, got {err:?}"
        );
    }

    #[test]
    fn pairs_returns_all() {
        let cm = make_crossmap();
        cm.add("A-1", "B-1").unwrap();
        cm.add("A-2", "B-2").unwrap();

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
            let (_store, cm, _conn) =
                crate::store::sqlite::open_sqlite(&path, &bc, None, &[], &[]).unwrap();
            cm.add("A-1", "B-1").unwrap();
            cm.add("A-2", "B-2").unwrap();
        }

        // Reopen and verify
        {
            let (_store, cm, _conn) =
                crate::store::sqlite::open_sqlite(&path, &bc, None, &[], &[]).unwrap();
            assert_eq!(cm.len(), 2);
            assert_eq!(cm.get_b("A-1"), Some("B-1".to_string()));
            assert_eq!(cm.get_b("A-2"), Some("B-2".to_string()));
        }

        std::mem::forget(dir);
    }
}
