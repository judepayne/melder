//! SQLite-backed record store for durable live-mode persistence.
//!
//! Uses a writer + reader pool architecture. One `Mutex<Connection>` handles
//! all writes (SQLite allows only one writer). A pool of read-only connections
//! serves concurrent reads via round-robin `try_lock`, matching the
//! `EncoderPool` pattern. SQLite WAL mode enables concurrent readers.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, params};

use crate::config::{BlockingConfig, BlockingFieldPair};
use crate::error::StoreError;
use crate::models::{Record, Side};

use super::RecordStore;

/// Pool of read-only SQLite connections with round-robin acquisition.
///
/// Each connection has its own `Mutex` (required because `rusqlite::Connection`
/// is `Send` but not `Sync`). Concurrency comes from having multiple
/// connections, not from sharing one. Shared between `SqliteStore` and
/// `SqliteCrossMap`.
pub struct SqliteReaderPool {
    readers: Vec<Mutex<Connection>>,
}

impl SqliteReaderPool {
    /// Acquire a reader connection via round-robin try_lock.
    ///
    /// Tries each slot in order; falls back to blocking on slot 0 if all
    /// are busy. Same pattern as `EncoderPool`.
    pub fn acquire(&self) -> std::sync::MutexGuard<'_, Connection> {
        for reader in &self.readers {
            if let Ok(guard) = reader.try_lock() {
                return guard;
            }
        }
        // All busy — block on slot 0
        self.readers[0].lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// SQLite-backed record store.
///
/// Both sides (A and B) live in the same database file with separate tables.
/// Records are stored in columnar format (one column per field) rather than
/// as JSON blobs — ~2.3× faster for candidate lookups during scoring.
/// Blocking keys are stored in indexed tables for fast candidate lookups.
/// Reads are served by a shared `SqliteReaderPool`; writes go through a
/// single dedicated connection.
pub struct SqliteStore {
    writer: Arc<Mutex<Connection>>,
    reader_pool: Arc<SqliteReaderPool>,
    blocking_config: BlockingConfig,
    /// Column names for A-side records (from required_fields_a).
    columns_a: Vec<String>,
    /// Column names for B-side records (from required_fields_b).
    columns_b: Vec<String>,
}

impl SqliteStore {
    /// Get the column names for a given side.
    fn columns(&self, side: Side) -> &[String] {
        match side {
            Side::A => &self.columns_a,
            Side::B => &self.columns_b,
        }
    }
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore")
            .field("a_records", &self.len(Side::A).unwrap_or(0))
            .field("b_records", &self.len(Side::B).unwrap_or(0))
            .field("reader_pool_size", &self.reader_pool.readers.len())
            .finish()
    }
}

// ── Schema + factory ─────────────────────────────────────────────────────────

/// SQL statements for non-record tables (blocking, crossmap, etc).
/// Record tables are created dynamically based on config fields.
const SCHEMA_FIXED: &str = "
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

/// Generate CREATE TABLE DDL for a side's record table with columnar fields.
fn record_table_ddl(side: Side, columns: &[String]) -> String {
    let prefix = side_prefix(side);
    let col_defs: Vec<String> = columns
        .iter()
        .map(|c| format!("    \"{}\" TEXT NOT NULL DEFAULT ''", c))
        .collect();
    let cols = if col_defs.is_empty() {
        String::new()
    } else {
        format!(",\n{}", col_defs.join(",\n"))
    };
    format!("CREATE TABLE IF NOT EXISTS {prefix}_records (\n    id TEXT PRIMARY KEY{cols}\n);")
}

/// Configuration for the SQLite connection pool.
pub struct SqlitePoolConfig {
    /// Page cache for the write connection in KiB. Default: 65536 (64 MB).
    pub writer_cache_kb: u64,
    /// Number of read-only connections. Default: 4.
    pub read_pool_size: u32,
    /// Page cache per read connection in KiB. Default: 131072 (128 MB).
    pub reader_cache_kb: u64,
}

impl Default for SqlitePoolConfig {
    fn default() -> Self {
        Self {
            writer_cache_kb: 65536, // 64 MB
            read_pool_size: 4,
            reader_cache_kb: 131072, // 128 MB
        }
    }
}

/// Apply performance pragmas to a connection.
fn apply_pragmas(
    conn: &Connection,
    cache_kb: u64,
    query_only: bool,
) -> Result<(), rusqlite::Error> {
    let mut pragmas = format!(
        "PRAGMA journal_mode = WAL;
         PRAGMA cache_size = -{cache_kb};
         PRAGMA page_size = 8192;
         PRAGMA foreign_keys = OFF;
         PRAGMA synchronous = NORMAL;"
    );
    if query_only {
        pragmas.push_str("\nPRAGMA query_only = ON;");
    }
    conn.execute_batch(&pragmas)
}

/// Open a SQLite database with a writer + reader pool.
///
/// Returns the `SqliteStore`, `SqliteCrossMap`, and the writer connection
/// `Arc<Mutex<Connection>>` (for review write-through or other components).
///
/// `columns_a` / `columns_b` specify the field names for each side's
/// record table (columnar storage). Pass empty slices to create tables
/// with only the `id` column (useful for tests or when fields aren't
/// known at open time).
///
/// Pass `None` for `pool_config` to use defaults.
pub fn open_sqlite(
    path: &Path,
    blocking_config: &BlockingConfig,
    pool_config: Option<SqlitePoolConfig>,
    columns_a: &[String],
    columns_b: &[String],
) -> Result<
    (
        SqliteStore,
        crate::crossmap::sqlite::SqliteCrossMap,
        Arc<Mutex<Connection>>,
    ),
    rusqlite::Error,
> {
    let cfg = pool_config.unwrap_or_default();

    // --- Writer connection ---
    let writer_conn = Connection::open(path)?;
    apply_pragmas(&writer_conn, cfg.writer_cache_kb, false)?;
    writer_conn.execute_batch(SCHEMA_FIXED)?;
    writer_conn.execute_batch(&record_table_ddl(Side::A, columns_a))?;
    writer_conn.execute_batch(&record_table_ddl(Side::B, columns_b))?;

    let writer = Arc::new(Mutex::new(writer_conn));

    // --- Reader pool ---
    let mut readers = Vec::with_capacity(cfg.read_pool_size as usize);
    for _ in 0..cfg.read_pool_size {
        let rconn = Connection::open(path)?;
        apply_pragmas(&rconn, cfg.reader_cache_kb, true)?;
        readers.push(Mutex::new(rconn));
    }
    let reader_pool = Arc::new(SqliteReaderPool { readers });

    let store = SqliteStore {
        writer: Arc::clone(&writer),
        reader_pool: Arc::clone(&reader_pool),
        blocking_config: blocking_config.clone(),
        columns_a: columns_a.to_vec(),
        columns_b: columns_b.to_vec(),
    };

    let crossmap =
        crate::crossmap::sqlite::SqliteCrossMap::new(Arc::clone(&writer), Arc::clone(&reader_pool));

    Ok((store, crossmap, writer))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Table name prefix for a given side.
fn side_prefix(side: Side) -> &'static str {
    match side {
        Side::A => "a",
        Side::B => "b",
    }
}

/// Build a Record from column values.
fn record_from_row(columns: &[String], row: &rusqlite::Row, offset: usize) -> Record {
    let mut record = Record::new();
    for (i, col_name) in columns.iter().enumerate() {
        let val: Option<String> = row.get(offset + i).unwrap_or(None);
        record.insert(col_name.clone(), val.unwrap_or_default());
    }
    record
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
    // --- Records (reads → reader pool, writes → writer) ---

    fn get(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError> {
        let conn = self.reader_pool.acquire();
        let columns = self.columns(side);
        let col_list = if columns.is_empty() {
            "id".to_string()
        } else {
            format!(
                "id, {}",
                columns
                    .iter()
                    .map(|c| format!("\"{}\"", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let sql = format!(
            "SELECT {} FROM {}_records WHERE id = ?1",
            col_list,
            side_prefix(side)
        );
        match conn.query_row(&sql, params![id], |row| {
            Ok(record_from_row(columns, row, 1))
        }) {
            Ok(rec) => Ok(Some(rec)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StoreError::from(e)),
        }
    }

    fn insert(&self, side: Side, id: &str, record: &Record) -> Result<Option<Record>, StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        let columns = self.columns(side);

        // Fetch old record via the writer (sees its own pending writes)
        let col_list = if columns.is_empty() {
            "id".to_string()
        } else {
            format!(
                "id, {}",
                columns
                    .iter()
                    .map(|c| format!("\"{}\"", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let old: Option<Record> = conn
            .query_row(
                &format!("SELECT {} FROM {}_records WHERE id = ?1", col_list, prefix),
                params![id],
                |row| Ok(record_from_row(columns, row, 1)),
            )
            .ok();

        // Insert or replace — build column list and values dynamically
        let col_names: Vec<String> = std::iter::once("id".to_string())
            .chain(columns.iter().map(|c| format!("\"{}\"", c)))
            .collect();
        let placeholders: Vec<String> =
            (1..=columns.len() + 1).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "INSERT OR REPLACE INTO {}_records ({}) VALUES ({})",
            prefix,
            col_names.join(", "),
            placeholders.join(", ")
        );

        let mut param_values: Vec<String> = vec![id.to_string()];
        for col in columns {
            param_values.push(record.get(col).cloned().unwrap_or_default());
        }
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values
            .iter()
            .map(|v| v as &dyn rusqlite::types::ToSql)
            .collect();
        conn.execute(&sql, param_refs.as_slice())?;

        Ok(old)
    }

    fn remove(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        let columns = self.columns(side);

        // Fetch old record via writer
        let col_list = if columns.is_empty() {
            "id".to_string()
        } else {
            format!(
                "id, {}",
                columns
                    .iter()
                    .map(|c| format!("\"{}\"", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let old: Option<Record> = conn
            .query_row(
                &format!("SELECT {} FROM {}_records WHERE id = ?1", col_list, prefix),
                params![id],
                |row| Ok(record_from_row(columns, row, 1)),
            )
            .ok();

        if old.is_some() {
            conn.execute(
                &format!("DELETE FROM {}_records WHERE id = ?1", prefix),
                params![id],
            )?;
        }

        Ok(old)
    }

    fn contains(&self, side: Side, id: &str) -> Result<bool, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT 1 FROM {}_records WHERE id = ?1", side_prefix(side));
        Ok(conn.query_row(&sql, params![id], |_| Ok(())).is_ok())
    }

    fn len(&self, side: Side) -> Result<usize, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT COUNT(*) FROM {}_records", side_prefix(side));
        let count = conn.query_row(&sql, [], |row| row.get::<_, i64>(0))?;
        Ok(count as usize)
    }

    fn ids(&self, side: Side) -> Result<Vec<String>, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT id FROM {}_records", side_prefix(side));
        let mut stmt = conn.prepare(&sql)?;
        let ids = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| match r {
                Ok(id) => Some(id),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed row in id query");
                    None
                }
            })
            .collect();
        Ok(ids)
    }

    fn get_many(&self, side: Side, ids: &[String]) -> Result<Vec<(String, Record)>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.reader_pool.acquire();
        let prefix = side_prefix(side);
        let columns = self.columns(side);
        let col_list = if columns.is_empty() {
            "id".to_string()
        } else {
            format!(
                "id, {}",
                columns
                    .iter()
                    .map(|c| format!("\"{}\"", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let mut results = Vec::with_capacity(ids.len());

        for chunk in ids.chunks(900) {
            let placeholders: Vec<&str> = chunk.iter().map(|_| "?").collect();
            let sql = format!(
                "SELECT {} FROM {}_records WHERE id IN ({})",
                col_list,
                prefix,
                placeholders.join(",")
            );
            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            let mut rows = stmt.query(params.as_slice())?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                results.push((id, record_from_row(columns, row, 1)));
            }
        }
        Ok(results)
    }

    fn get_many_fields(
        &self,
        side: Side,
        ids: &[String],
        fields: &[String],
    ) -> Result<Vec<(String, Record)>, StoreError> {
        if ids.is_empty() || fields.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.reader_pool.acquire();
        let prefix = side_prefix(side);
        let mut results = Vec::with_capacity(ids.len());

        // With columnar storage, just SELECT the requested columns directly.
        let col_list = fields
            .iter()
            .map(|f| format!("\"{}\"", f))
            .collect::<Vec<_>>()
            .join(", ");

        for chunk in ids.chunks(900) {
            let placeholders: Vec<&str> = chunk.iter().map(|_| "?").collect();
            let sql = format!(
                "SELECT id, {} FROM {}_records WHERE id IN ({})",
                col_list,
                prefix,
                placeholders.join(",")
            );
            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            let mut rows = stmt.query(params.as_slice())?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                results.push((id, record_from_row(fields, row, 1)));
            }
        }
        Ok(results)
    }

    fn for_each_record(
        &self,
        side: Side,
        f: &mut dyn FnMut(&str, &Record),
    ) -> Result<(), StoreError> {
        let conn = self.reader_pool.acquire();
        let columns = self.columns(side);
        let col_list = if columns.is_empty() {
            "id".to_string()
        } else {
            format!(
                "id, {}",
                columns
                    .iter()
                    .map(|c| format!("\"{}\"", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let sql = format!("SELECT {} FROM {}_records", col_list, side_prefix(side));
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let record = record_from_row(columns, row, 1);
            f(&id, &record);
        }
        Ok(())
    }

    // --- Blocking (reads → reader pool, writes → writer) ---

    fn blocking_insert(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        let values = blocking_values(record, &self.blocking_config.fields, side);

        for (field_index, value) in values {
            conn.execute(
                &format!(
                    "INSERT INTO {}_blocking_keys (record_id, field_index, value) VALUES (?1, ?2, ?3)",
                    prefix
                ),
                params![id, field_index as i64, value],
            )?;
        }
        Ok(())
    }

    fn blocking_remove(&self, side: Side, id: &str, _record: &Record) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_blocking_keys WHERE record_id = ?1", prefix),
            params![id],
        )?;
        Ok(())
    }

    fn blocking_query(
        &self,
        record: &Record,
        query_side: Side,
        pool_side: Side,
    ) -> Result<Vec<String>, StoreError> {
        let conn = self.reader_pool.acquire();
        let fields = &self.blocking_config.fields;
        let prefix = side_prefix(pool_side);

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

        if query_values.is_empty() {
            let sql = format!("SELECT id FROM {}_records", prefix);
            let mut stmt = conn.prepare(&sql)?;
            let ids = stmt
                .query_map([], |row| row.get(0))?
                .filter_map(|r| match r {
                    Ok(id) => Some(id),
                    Err(e) => {
                        tracing::warn!(error = %e, "skipping malformed row in id query");
                        None
                    }
                })
                .collect();
            return Ok(ids);
        }

        let conditions: Vec<String> = query_values
            .iter()
            .map(|(i, _)| format!("(field_index = {} AND value = ?)", i))
            .collect();
        let where_clause = conditions.join(" OR ");

        // AND blocking: all field pairs must match.
        let sql = format!(
            "SELECT record_id FROM {}_blocking_keys WHERE {} GROUP BY record_id HAVING COUNT(DISTINCT field_index) = {}",
            prefix,
            where_clause,
            query_values.len()
        );

        let mut stmt = conn.prepare(&sql)?;
        let param_values: Vec<&dyn rusqlite::types::ToSql> = query_values
            .iter()
            .map(|(_, v)| v as &dyn rusqlite::types::ToSql)
            .collect();

        let ids = stmt
            .query_map(param_values.as_slice(), |row| row.get(0))?
            .filter_map(|r| match r {
                Ok(id) => Some(id),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed row in id query");
                    None
                }
            })
            .collect();
        Ok(ids)
    }

    // --- Unmatched (reads → reader pool, writes → writer) ---

    fn mark_unmatched(&self, side: Side, id: &str) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!(
                "INSERT OR IGNORE INTO {}_unmatched (id) VALUES (?1)",
                prefix
            ),
            params![id],
        )?;
        Ok(())
    }

    fn mark_matched(&self, side: Side, id: &str) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_unmatched WHERE id = ?1", prefix),
            params![id],
        )?;
        Ok(())
    }

    fn is_unmatched(&self, side: Side, id: &str) -> Result<bool, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!(
            "SELECT 1 FROM {}_unmatched WHERE id = ?1",
            side_prefix(side)
        );
        Ok(conn.query_row(&sql, params![id], |_| Ok(())).is_ok())
    }

    fn unmatched_count(&self, side: Side) -> Result<usize, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT COUNT(*) FROM {}_unmatched", side_prefix(side));
        let count = conn.query_row(&sql, [], |row| row.get::<_, i64>(0))?;
        Ok(count as usize)
    }

    fn unmatched_ids(&self, side: Side) -> Result<Vec<String>, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT id FROM {}_unmatched ORDER BY id", side_prefix(side));
        let mut stmt = conn.prepare(&sql)?;
        let ids = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| match r {
                Ok(id) => Some(id),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed row in unmatched query");
                    None
                }
            })
            .collect();
        Ok(ids)
    }

    // --- Common ID index (reads → reader pool, writes → writer) ---

    fn common_id_insert(
        &self,
        side: Side,
        common_id: &str,
        record_id: &str,
    ) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {}_common_ids (common_id, record_id) VALUES (?1, ?2)",
                prefix
            ),
            params![common_id, record_id],
        )?;
        Ok(())
    }

    fn common_id_lookup(&self, side: Side, common_id: &str) -> Result<Option<String>, StoreError> {
        let conn = self.reader_pool.acquire();
        let sql = format!(
            "SELECT record_id FROM {}_common_ids WHERE common_id = ?1",
            side_prefix(side)
        );
        Ok(conn
            .query_row(&sql, params![common_id], |row| row.get(0))
            .ok())
    }

    fn common_id_remove(&self, side: Side, common_id: &str) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_common_ids WHERE common_id = ?1", prefix),
            params![common_id],
        )?;
        Ok(())
    }

    // --- Exact prefilter ---

    fn build_exact_index(&self, side: Side, field_names: &[String]) -> Result<(), StoreError> {
        if field_names.is_empty() {
            return Ok(());
        }
        let prefix = side_prefix(side);
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        // Create a composite index on the exact prefilter fields.
        // Column names are quoted to handle any reserved words.
        let cols: Vec<String> = field_names.iter().map(|f| format!("\"{}\"", f)).collect();
        let idx_name = format!("idx_{}_exact_prefilter", prefix);
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS {} ON {}_records ({})",
            idx_name,
            prefix,
            cols.join(", ")
        );
        if let Err(e) = conn.execute_batch(&sql) {
            tracing::warn!(error = %e, "failed to build exact prefilter index");
        }
        Ok(())
    }

    fn exact_lookup(
        &self,
        side: Side,
        kvs: &[(String, String)],
    ) -> Result<Option<String>, StoreError> {
        if kvs.is_empty() {
            return Ok(None);
        }
        // Any empty value means no match (AND semantics).
        let vals: Vec<&str> = kvs.iter().map(|(_, v)| v.trim()).collect();
        if vals.iter().any(|v| v.is_empty()) {
            return Ok(None);
        }

        let prefix = side_prefix(side);
        let where_clauses: Vec<String> = kvs
            .iter()
            .map(|(f, _)| format!("lower(\"{}\") = lower(?)", f))
            .collect();
        let sql = format!(
            "SELECT id FROM {}_records WHERE {} LIMIT 1",
            prefix,
            where_clauses.join(" AND ")
        );

        let conn = self.reader_pool.acquire();
        let result = match kvs.len() {
            1 => conn.query_row(&sql, params![vals[0]], |row| row.get::<_, String>(0)),
            2 => conn.query_row(&sql, params![vals[0], vals[1]], |row| {
                row.get::<_, String>(0)
            }),
            3 => conn.query_row(&sql, params![vals[0], vals[1], vals[2]], |row| {
                row.get::<_, String>(0)
            }),
            _ => {
                // Fallback for >3 fields: use rusqlite's params_from_iter
                use rusqlite::types::ToSql;
                let boxed: Vec<Box<dyn ToSql>> = vals
                    .iter()
                    .map(|v| Box::new(v.to_string()) as Box<dyn ToSql>)
                    .collect();
                conn.query_row(&sql, rusqlite::params_from_iter(boxed.iter()), |row| {
                    row.get::<_, String>(0)
                })
            }
        };
        Ok(result.ok())
    }

    // --- Review persistence (reads → reader pool, writes → writer) ---

    fn persist_review(
        &self,
        key: &str,
        id: &str,
        side: Side,
        candidate_id: &str,
        score: f64,
    ) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let side_str = match side {
            Side::A => "a",
            Side::B => "b",
        };
        conn.execute(
            "INSERT OR REPLACE INTO reviews (key, id, side, candidate_id, score) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![key, id, side_str, candidate_id, score],
        )?;
        Ok(())
    }

    fn remove_reviews_for_id(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "DELETE FROM reviews WHERE id = ?1 OR candidate_id = ?1",
            params![id],
        )?;
        Ok(())
    }

    fn remove_reviews_for_pair(&self, a_id: &str, b_id: &str) -> Result<(), StoreError> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "DELETE FROM reviews WHERE id = ?1 OR candidate_id = ?1 OR id = ?2 OR candidate_id = ?2",
            params![a_id, b_id],
        )?;
        Ok(())
    }

    fn load_reviews(&self) -> Result<Vec<super::ReviewEntry>, StoreError> {
        let conn = self.reader_pool.acquire();
        let mut stmt = conn.prepare("SELECT key, id, side, candidate_id, score FROM reviews")?;
        let reviews = stmt
            .query_map([], |row| {
                let key: String = row.get(0)?;
                let id: String = row.get(1)?;
                let side_str: String = row.get(2)?;
                let candidate_id: String = row.get(3)?;
                let score: f64 = row.get(4)?;
                let side = if side_str == "a" { Side::A } else { Side::B };
                Ok((key, id, side, candidate_id, score))
            })?
            .filter_map(|r| match r {
                Ok(entry) => Some(entry),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed row in review query");
                    None
                }
            })
            .collect();
        Ok(reviews)
    }
}

// ── Bulk loading ─────────────────────────────────────────────────────────────

impl SqliteStore {
    /// Bulk-load records into SQLite by streaming from a data source.
    ///
    /// Calls `stream_fn` which should call `stream_dataset()`. Each chunk
    /// is inserted immediately, so only one chunk (~10K records, ~15MB)
    /// is in memory at any time. The entire load is wrapped in a single
    /// transaction with deferred index creation.
    ///
    /// The writer lock is held for the entire operation (acceptable during
    /// single-threaded startup). Progress is printed periodically.
    pub fn bulk_load<F>(&self, side: Side, stream_fn: F) -> Result<usize, crate::error::DataError>
    where
        F: FnOnce(&mut dyn FnMut(Vec<(String, Record)>)) -> Result<usize, crate::error::DataError>,
    {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);

        // Drop blocking index for faster inserts (recreated after)
        let _ = conn.execute_batch(&format!("DROP INDEX IF EXISTS idx_{prefix}_blocking"));

        // Begin single transaction for entire load
        if let Err(e) = conn.execute_batch("BEGIN") {
            tracing::warn!(error = %e, "bulk_load: failed to begin transaction");
            return Ok(0);
        }

        // Build columnar INSERT SQL: INSERT INTO {prefix}_records (id, "col1", "col2", ...) VALUES (?1, ?2, ...)
        let columns = self.columns(side).to_vec();
        let col_names: Vec<String> = std::iter::once("id".to_string())
            .chain(columns.iter().map(|c| format!("\"{}\"", c)))
            .collect();
        let placeholders: Vec<String> =
            (1..=columns.len() + 1).map(|i| format!("?{}", i)).collect();
        let rec_sql = format!(
            "INSERT OR REPLACE INTO {prefix}_records ({}) VALUES ({})",
            col_names.join(", "),
            placeholders.join(", ")
        );
        let blk_sql = format!(
            "INSERT INTO {prefix}_blocking_keys (record_id, field_index, value) VALUES (?1, ?2, ?3)"
        );
        let unm_sql = format!("INSERT OR IGNORE INTO {prefix}_unmatched (id) VALUES (?1)");

        let blocking_fields = self.blocking_config.fields.clone();
        let mut total = 0usize;
        let load_start = std::time::Instant::now();

        // Stream chunks via callback — only one chunk in memory at a time
        let mut insert_chunk = |chunk: Vec<(String, Record)>| {
            for (id, record) in &chunk {
                // Build column values: id, then each field in order
                let mut param_values: Vec<String> = vec![id.clone()];
                for col in &columns {
                    param_values.push(record.get(col).cloned().unwrap_or_default());
                }
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values
                    .iter()
                    .map(|v| v as &dyn rusqlite::types::ToSql)
                    .collect();
                conn.execute(&rec_sql, param_refs.as_slice())
                    .unwrap_or_else(|e| {
                        tracing::warn!(error = %e, id, "bulk insert record failed");
                        0
                    });

                let values = blocking_values(record, &blocking_fields, side);
                for (field_index, value) in values {
                    conn.execute(&blk_sql, params![id, field_index as i64, value])
                        .unwrap_or_else(|e| {
                            tracing::warn!(error = %e, id, "bulk insert blocking key failed");
                            0
                        });
                }

                conn.execute(&unm_sql, params![id]).unwrap_or_else(|e| {
                    tracing::warn!(error = %e, id, "bulk mark unmatched failed");
                    0
                });
            }

            total += chunk.len();
            if total.is_multiple_of(100_000) || (total < 100_000 && total.is_multiple_of(10_000)) {
                let elapsed = load_start.elapsed().as_secs_f64();
                let rate = total as f64 / elapsed;
                tracing::info!(
                    side = prefix,
                    records = total,
                    rate = format!("{:.0}", rate),
                    "bulk_load progress"
                );
            }
        };

        stream_fn(&mut insert_chunk)?;

        if let Err(e) = conn.execute_batch("COMMIT") {
            tracing::warn!(error = %e, "bulk_load commit failed");
        }

        // Recreate blocking index
        let _ = conn.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS idx_{prefix}_blocking ON {prefix}_blocking_keys(field_index, value)"
        ));

        tracing::info!(
            side = prefix,
            records = total,
            elapsed_secs = format!("{:.1}", load_start.elapsed().as_secs_f64()),
            "bulk_load complete"
        );

        Ok(total)
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
        let pool_cfg = SqlitePoolConfig {
            read_pool_size: 2,
            ..Default::default()
        };
        // Test columns cover all fields used in test records
        let cols_a = vec![
            "name".to_string(),
            "country_a".to_string(),
            "city".to_string(),
        ];
        let cols_b = vec![
            "name".to_string(),
            "country_b".to_string(),
            "city".to_string(),
        ];
        let (store, _crossmap, _conn) =
            open_sqlite(&path, &bc, Some(pool_cfg), &cols_a, &cols_b).unwrap();
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
        assert!(store.insert(Side::A, "1", &rec).unwrap().is_none());
        let got = store.get(Side::A, "1").unwrap().unwrap();
        assert_eq!(got.get("name").unwrap(), "Alice");
    }

    #[test]
    fn insert_replaces_and_returns_old() {
        let store = make_store();
        let r1 = make_record(&[("name", "Alice")]);
        let r2 = make_record(&[("name", "Bob")]);
        assert!(store.insert(Side::A, "1", &r1).unwrap().is_none());
        let old = store.insert(Side::A, "1", &r2).unwrap().unwrap();
        assert_eq!(old.get("name").unwrap(), "Alice");
        assert_eq!(
            store
                .get(Side::A, "1")
                .unwrap()
                .unwrap()
                .get("name")
                .unwrap(),
            "Bob"
        );
    }

    #[test]
    fn remove_returns_old() {
        let store = make_store();
        let rec = make_record(&[("name", "Alice")]);
        store.insert(Side::A, "1", &rec).unwrap();
        let old = store.remove(Side::A, "1").unwrap().unwrap();
        assert_eq!(old.get("name").unwrap(), "Alice");
        assert!(store.get(Side::A, "1").unwrap().is_none());
        assert!(store.remove(Side::A, "1").unwrap().is_none());
    }

    #[test]
    fn contains_and_len() {
        let store = make_store();
        assert!(!store.contains(Side::A, "1").unwrap());
        assert_eq!(store.len(Side::A).unwrap(), 0);
        let rec = make_record(&[("name", "Alice")]);
        store.insert(Side::A, "1", &rec).unwrap();
        assert!(store.contains(Side::A, "1").unwrap());
        assert_eq!(store.len(Side::A).unwrap(), 1);
    }

    #[test]
    fn ids_returns_all() {
        let store = make_store();
        let rec = make_record(&[("name", "x")]);
        store.insert(Side::A, "2", &rec).unwrap();
        store.insert(Side::A, "1", &rec).unwrap();
        store.insert(Side::A, "3", &rec).unwrap();
        let mut ids = store.ids(Side::A).unwrap();
        ids.sort();
        assert_eq!(ids, vec!["1", "2", "3"]);
    }

    #[test]
    fn sides_are_independent() {
        let store = make_store();
        let rec = make_record(&[("name", "x")]);
        store.insert(Side::A, "1", &rec).unwrap();
        assert!(store.contains(Side::A, "1").unwrap());
        assert!(!store.contains(Side::B, "1").unwrap());
        assert_eq!(store.len(Side::A).unwrap(), 1);
        assert_eq!(store.len(Side::B).unwrap(), 0);
    }

    #[test]
    fn from_records_populates_all() {
        // SqliteStore doesn't have from_records, so test insert + blocking
        let store = make_store();
        let rec = make_record(&[("name", "Alice"), ("country_a", "US")]);
        store.insert(Side::A, "1", &rec).unwrap();
        store.blocking_insert(Side::A, "1", &rec).unwrap();
        assert_eq!(store.len(Side::A).unwrap(), 1);
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
        let cols_a = vec!["name".to_string(), "country_a".to_string()];
        let cols_b = vec!["name".to_string(), "country_b".to_string()];
        let (store, _, _conn) = open_sqlite(&path, &bc, None, &cols_a, &cols_b).unwrap();

        // Insert A-side records with blocking keys
        let r1 = make_record(&[("name", "Alice"), ("country_a", "US")]);
        let r2 = make_record(&[("name", "Bob"), ("country_a", "UK")]);
        store.insert(Side::A, "1", &r1).unwrap();
        store.blocking_insert(Side::A, "1", &r1).unwrap();
        store.insert(Side::A, "2", &r2).unwrap();
        store.blocking_insert(Side::A, "2", &r2).unwrap();

        // Query from B-side: country_b = "us" should match record 1
        let query = make_record(&[("country_b", "US")]);
        let results = store.blocking_query(&query, Side::B, Side::A).unwrap();
        assert_eq!(results.len(), 1, "should find 1 match for US");
        assert_eq!(results[0], "1");

        // Remove blocking keys for record 1
        store.blocking_remove(Side::A, "1", &r1).unwrap();
        let results2 = store.blocking_query(&query, Side::B, Side::A).unwrap();
        assert!(results2.is_empty(), "should find 0 after remove");

        std::mem::forget(dir);
    }

    #[test]
    fn unmatched_lifecycle() {
        let store = make_store();
        assert_eq!(store.unmatched_count(Side::A).unwrap(), 0);
        store.mark_unmatched(Side::A, "1").unwrap();
        store.mark_unmatched(Side::A, "2").unwrap();
        assert_eq!(store.unmatched_count(Side::A).unwrap(), 2);
        assert!(store.is_unmatched(Side::A, "1").unwrap());
        store.mark_matched(Side::A, "1").unwrap();
        assert!(!store.is_unmatched(Side::A, "1").unwrap());
        assert_eq!(store.unmatched_count(Side::A).unwrap(), 1);
    }

    #[test]
    fn unmatched_ids_sorted() {
        let store = make_store();
        store.mark_unmatched(Side::A, "c").unwrap();
        store.mark_unmatched(Side::A, "a").unwrap();
        store.mark_unmatched(Side::A, "b").unwrap();
        assert_eq!(store.unmatched_ids(Side::A).unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn common_id_lifecycle() {
        let store = make_store();
        store.common_id_insert(Side::A, "CID-1", "REC-1").unwrap();
        assert_eq!(
            store.common_id_lookup(Side::A, "CID-1").unwrap(),
            Some("REC-1".to_string())
        );
        store.common_id_remove(Side::A, "CID-1").unwrap();
        assert!(store.common_id_lookup(Side::A, "CID-1").unwrap().is_none());
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
        let cols = vec!["name".to_string()];

        // Insert data
        {
            let (store, _, _conn) = open_sqlite(&path, &bc, None, &cols, &[]).unwrap();
            let rec = make_record(&[("name", "Alice")]);
            store.insert(Side::A, "1", &rec).unwrap();
            store.mark_unmatched(Side::A, "1").unwrap();
            store.common_id_insert(Side::A, "CID-1", "1").unwrap();
        }

        // Reopen and verify
        {
            let (store, _, _conn) = open_sqlite(&path, &bc, None, &cols, &[]).unwrap();
            assert_eq!(store.len(Side::A).unwrap(), 1);
            assert_eq!(
                store
                    .get(Side::A, "1")
                    .unwrap()
                    .unwrap()
                    .get("name")
                    .unwrap(),
                "Alice"
            );
            assert!(store.is_unmatched(Side::A, "1").unwrap());
            assert_eq!(
                store.common_id_lookup(Side::A, "CID-1").unwrap(),
                Some("1".to_string())
            );
        }

        std::mem::forget(dir);
    }
}
