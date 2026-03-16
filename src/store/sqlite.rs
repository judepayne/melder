//! SQLite-backed record store for durable live-mode persistence.
//!
//! Uses a writer + reader pool architecture. One `Mutex<Connection>` handles
//! all writes (SQLite allows only one writer). A pool of read-only connections
//! serves concurrent reads via round-robin `try_lock`, matching the
//! `EncoderPool` pattern. SQLite WAL mode enables concurrent readers.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection};

use crate::config::{BlockingConfig, BlockingFieldPair};
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
/// Blocking keys are stored in indexed tables for fast candidate lookups.
/// Reads are served by a shared `SqliteReaderPool`; writes go through a
/// single dedicated connection.
pub struct SqliteStore {
    writer: Arc<Mutex<Connection>>,
    reader_pool: Arc<SqliteReaderPool>,
    blocking_config: BlockingConfig,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore")
            .field("a_records", &self.len(Side::A))
            .field("b_records", &self.len(Side::B))
            .field("reader_pool_size", &self.reader_pool.readers.len())
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
/// Pass `None` for `pool_config` to use defaults.
pub fn open_sqlite(
    path: &Path,
    blocking_config: &BlockingConfig,
    pool_config: Option<SqlitePoolConfig>,
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
    writer_conn.execute_batch(SCHEMA)?;

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
            if val.is_empty() {
                None
            } else {
                Some((i, val))
            }
        })
        .collect()
}

// ── RecordStore implementation ───────────────────────────────────────────────

impl RecordStore for SqliteStore {
    // --- Records (reads → reader pool, writes → writer) ---

    fn get(&self, side: Side, id: &str) -> Option<Record> {
        let conn = self.reader_pool.acquire();
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);

        // Fetch old record via the writer (sees its own pending writes)
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);

        // Fetch old record via writer
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
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT 1 FROM {}_records WHERE id = ?1", side_prefix(side));
        conn.query_row(&sql, params![id], |_| Ok(())).is_ok()
    }

    fn len(&self, side: Side) -> usize {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT COUNT(*) FROM {}_records", side_prefix(side));
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    fn ids(&self, side: Side) -> Vec<String> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT id FROM {}_records", side_prefix(side));
        let mut stmt = conn.prepare(&sql).expect("prepare ids query");
        stmt.query_map([], |row| row.get(0))
            .expect("query ids")
            .filter_map(|r| r.ok())
            .collect()
    }

    fn get_many(&self, side: Side, ids: &[String]) -> Vec<(String, Record)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let conn = self.reader_pool.acquire();
        let prefix = side_prefix(side);
        let mut results = Vec::with_capacity(ids.len());

        // Chunk to stay within SQLite's variable limit (999 is the safe
        // minimum across builds; most modern builds allow 32766).
        for chunk in ids.chunks(900) {
            let placeholders: Vec<&str> = chunk.iter().map(|_| "?").collect();
            let sql = format!(
                "SELECT id, record_json FROM {}_records WHERE id IN ({})",
                prefix,
                placeholders.join(",")
            );
            let mut stmt = conn.prepare(&sql).expect("prepare get_many query");
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            let mut rows = stmt.query(params.as_slice()).expect("execute get_many");
            while let Some(row) = rows.next().expect("next row") {
                let id: String = row.get(0).expect("get id");
                let json: String = row.get(1).expect("get record_json");
                results.push((id, record_from_json(&json)));
            }
        }
        results
    }

    fn get_many_fields(
        &self,
        side: Side,
        ids: &[String],
        fields: &[String],
    ) -> Vec<(String, Record)> {
        if ids.is_empty() || fields.is_empty() {
            return Vec::new();
        }
        let conn = self.reader_pool.acquire();
        let prefix = side_prefix(side);
        let mut results = Vec::with_capacity(ids.len());

        // Build SELECT with json_extract() for each requested field.
        // This avoids full JSON deserialization in Rust — SQLite's C-based
        // JSON parser extracts only the needed fields.
        let field_exprs: Vec<String> = fields
            .iter()
            .map(|f| format!("json_extract(record_json, '$.{}') AS \"{}\"", f, f))
            .collect();
        let field_list = field_exprs.join(", ");

        for chunk in ids.chunks(900) {
            let placeholders: Vec<&str> = chunk.iter().map(|_| "?").collect();
            let sql = format!(
                "SELECT id, {} FROM {}_records WHERE id IN ({})",
                field_list,
                prefix,
                placeholders.join(",")
            );
            let mut stmt = conn.prepare(&sql).expect("prepare get_many_fields query");
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            let mut rows = stmt
                .query(params.as_slice())
                .expect("execute get_many_fields");
            while let Some(row) = rows.next().expect("next row") {
                let id: String = row.get(0).expect("get id");
                let mut record = Record::new();
                for (i, field_name) in fields.iter().enumerate() {
                    let val: Option<String> = row.get(i + 1).unwrap_or(None);
                    record.insert(field_name.clone(), val.unwrap_or_default());
                }
                results.push((id, record));
            }
        }
        results
    }

    fn for_each_record(&self, side: Side, f: &mut dyn FnMut(&str, &Record)) {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT id, record_json FROM {}_records", side_prefix(side));
        let mut stmt = conn.prepare(&sql).expect("prepare for_each_record query");
        let mut rows = stmt.query([]).expect("query for_each_record");
        while let Some(row) = rows.next().expect("next row") {
            let id: String = row.get(0).expect("get id");
            let json: String = row.get(1).expect("get record_json");
            let record = record_from_json(&json);
            f(&id, &record);
        }
    }

    // --- Blocking (reads → reader pool, writes → writer) ---

    fn blocking_insert(&self, side: Side, id: &str, record: &Record) {
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
            )
            .expect("insert blocking key");
        }
    }

    fn blocking_remove(&self, side: Side, id: &str, _record: &Record) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_blocking_keys WHERE record_id = ?1", prefix),
            params![id],
        )
        .expect("delete blocking keys");
    }

    fn blocking_query(&self, record: &Record, query_side: Side) -> Vec<String> {
        let conn = self.reader_pool.acquire();
        let fields = &self.blocking_config.fields;
        let opp = query_side.opposite();
        let prefix = side_prefix(opp);
        let is_and = self.blocking_config.operator.eq_ignore_ascii_case("and");

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
                if val.is_empty() {
                    None
                } else {
                    Some((i, val))
                }
            })
            .collect();

        if query_values.is_empty() {
            let sql = format!("SELECT id FROM {}_records", prefix);
            let mut stmt = conn.prepare(&sql).expect("prepare all-ids query");
            return stmt
                .query_map([], |row| row.get(0))
                .expect("query all ids")
                .filter_map(|r| r.ok())
                .collect();
        }

        let conditions: Vec<String> = query_values
            .iter()
            .map(|(i, _)| format!("(field_index = {} AND value = ?)", i))
            .collect();
        let where_clause = conditions.join(" OR ");

        let sql = if is_and {
            format!(
                "SELECT record_id FROM {}_blocking_keys WHERE {} GROUP BY record_id HAVING COUNT(DISTINCT field_index) = {}",
                prefix,
                where_clause,
                query_values.len()
            )
        } else {
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

    // --- Unmatched (reads → reader pool, writes → writer) ---

    fn mark_unmatched(&self, side: Side, id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_unmatched WHERE id = ?1", prefix),
            params![id],
        )
        .expect("mark matched");
    }

    fn is_unmatched(&self, side: Side, id: &str) -> bool {
        let conn = self.reader_pool.acquire();
        let sql = format!(
            "SELECT 1 FROM {}_unmatched WHERE id = ?1",
            side_prefix(side)
        );
        conn.query_row(&sql, params![id], |_| Ok(())).is_ok()
    }

    fn unmatched_count(&self, side: Side) -> usize {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT COUNT(*) FROM {}_unmatched", side_prefix(side));
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    fn unmatched_ids(&self, side: Side) -> Vec<String> {
        let conn = self.reader_pool.acquire();
        let sql = format!("SELECT id FROM {}_unmatched ORDER BY id", side_prefix(side));
        let mut stmt = conn.prepare(&sql).expect("prepare unmatched query");
        stmt.query_map([], |row| row.get(0))
            .expect("query unmatched")
            .filter_map(|r| r.ok())
            .collect()
    }

    // --- Common ID index (reads → reader pool, writes → writer) ---

    fn common_id_insert(&self, side: Side, common_id: &str, record_id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
        let conn = self.reader_pool.acquire();
        let sql = format!(
            "SELECT record_id FROM {}_common_ids WHERE common_id = ?1",
            side_prefix(side)
        );
        conn.query_row(&sql, params![common_id], |row| row.get(0))
            .ok()
    }

    fn common_id_remove(&self, side: Side, common_id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let prefix = side_prefix(side);
        conn.execute(
            &format!("DELETE FROM {}_common_ids WHERE common_id = ?1", prefix),
            params![common_id],
        )
        .expect("remove common id");
    }

    // --- Review persistence (reads → reader pool, writes → writer) ---

    fn persist_review(&self, key: &str, id: &str, side: Side, candidate_id: &str, score: f64) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let side_str = match side {
            Side::A => "a",
            Side::B => "b",
        };
        let _ = conn.execute(
            "INSERT OR REPLACE INTO reviews (key, id, side, candidate_id, score) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![key, id, side_str, candidate_id, score],
        );
    }

    fn remove_reviews_for_id(&self, id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let _ = conn.execute(
            "DELETE FROM reviews WHERE id = ?1 OR candidate_id = ?1",
            params![id],
        );
    }

    fn remove_reviews_for_pair(&self, a_id: &str, b_id: &str) {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let _ = conn.execute(
            "DELETE FROM reviews WHERE id = ?1 OR candidate_id = ?1 OR id = ?2 OR candidate_id = ?2",
            params![a_id, b_id],
        );
    }

    fn load_reviews(&self) -> Vec<(String, String, Side, String, f64)> {
        let conn = self.reader_pool.acquire();
        let mut stmt = conn
            .prepare("SELECT key, id, side, candidate_id, score FROM reviews")
            .expect("prepare reviews query");
        stmt.query_map([], |row| {
            let key: String = row.get(0)?;
            let id: String = row.get(1)?;
            let side_str: String = row.get(2)?;
            let candidate_id: String = row.get(3)?;
            let score: f64 = row.get(4)?;
            let side = if side_str == "a" { Side::A } else { Side::B };
            Ok((key, id, side, candidate_id, score))
        })
        .expect("query reviews")
        .filter_map(|r| r.ok())
        .collect()
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
        conn.execute_batch("BEGIN").expect("begin bulk transaction");

        let rec_sql =
            format!("INSERT OR REPLACE INTO {prefix}_records (id, record_json) VALUES (?1, ?2)");
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
                let json = record_to_json(record);
                conn.execute(&rec_sql, params![id, json])
                    .expect("bulk insert record");

                let values = blocking_values(record, &blocking_fields, side);
                for (field_index, value) in values {
                    conn.execute(&blk_sql, params![id, field_index as i64, value])
                        .expect("bulk insert blocking key");
                }

                conn.execute(&unm_sql, params![id])
                    .expect("bulk mark unmatched");
            }

            total += chunk.len();
            if total % 100_000 == 0 || (total < 100_000 && total % 10_000 == 0) {
                let elapsed = load_start.elapsed().as_secs_f64();
                let rate = total as f64 / elapsed;
                eprintln!(
                    "  bulk_load {}: {} records ({:.0} rec/s)",
                    prefix, total, rate
                );
            }
        };

        stream_fn(&mut insert_chunk)?;

        conn.execute_batch("COMMIT")
            .expect("commit bulk transaction");

        // Recreate blocking index
        let _ = conn.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS idx_{prefix}_blocking ON {prefix}_blocking_keys(field_index, value)"
        ));

        eprintln!(
            "  bulk_load {} complete: {} records in {:.1}s",
            prefix,
            total,
            load_start.elapsed().as_secs_f64()
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
        let (store, _crossmap, _conn) = open_sqlite(&path, &bc, Some(pool_cfg)).unwrap();
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
        let (store, _, _conn) = open_sqlite(&path, &bc, None).unwrap();

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
            let (store, _, _conn) = open_sqlite(&path, &bc, None).unwrap();
            let rec = make_record(&[("name", "Alice")]);
            store.insert(Side::A, "1", &rec);
            store.mark_unmatched(Side::A, "1");
            store.common_id_insert(Side::A, "CID-1", "1");
        }

        // Reopen and verify
        {
            let (store, _, _conn) = open_sqlite(&path, &bc, None).unwrap();
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
