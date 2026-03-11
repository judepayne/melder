//! Write-Ahead Log (WAL) for live mode.
//!
//! Records upsert events and crossmap changes as newline-delimited JSON.
//! Provides replay for crash recovery and compaction for space reclamation.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::models::{Record, Side};

/// WAL event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WalEvent {
    #[serde(rename = "upsert_record")]
    UpsertRecord { side: Side, record: Record },
    #[serde(rename = "crossmap_confirm")]
    CrossMapConfirm { a_id: String, b_id: String },
    #[serde(rename = "crossmap_break")]
    CrossMapBreak { a_id: String, b_id: String },
    #[serde(rename = "remove_record")]
    RemoveRecord { side: Side, id: String },
}

/// Write-ahead log for crash recovery.
///
/// Events are JSON-serialized, one per line. The log is append-only.
/// A `Mutex<BufWriter>` protects concurrent writes.
pub struct UpsertLog {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl UpsertLog {
    /// Open (or create) a WAL file for appending.
    pub fn open(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let writer = Mutex::new(BufWriter::new(file));

        Ok(Self {
            path: path.to_path_buf(),
            writer,
        })
    }

    /// Append an event to the WAL.
    ///
    /// Serializes to JSON + newline, writes to buffer. Does not fsync —
    /// background flush handles that periodically.
    pub fn append(&self, event: &WalEvent) -> io::Result<()> {
        let line = serde_json::to_string(event)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    }

    /// Append an upsert event without cloning the record.
    ///
    /// Uses a borrowing struct for serialization to avoid the clone that
    /// `WalEvent::UpsertRecord { record: record.clone() }` would require.
    pub fn append_upsert(&self, side: Side, record: &Record) -> io::Result<()> {
        #[derive(serde::Serialize)]
        struct UpsertRef<'a> {
            #[serde(rename = "type")]
            ty: &'static str,
            side: Side,
            record: &'a Record,
        }
        let event = UpsertRef {
            ty: "upsert_record",
            side,
            record,
        };
        let line = serde_json::to_string(&event)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    }

    /// Flush the buffered writer to the OS.
    pub fn flush(&self) -> io::Result<()> {
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        w.flush()
    }

    /// Replay all events from the WAL file.
    ///
    /// Tolerates a truncated last line (logs a warning, skips it).
    /// Returns all successfully parsed events.
    pub fn replay(path: &Path) -> io::Result<Vec<WalEvent>> {
        if !path.exists() {
            return Ok(vec![]);
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        let mut line_num = 0;
        let mut truncated = 0;

        for line_result in reader.lines() {
            line_num += 1;
            let line = match line_result {
                Ok(l) => l,
                Err(_) => {
                    // Truncated line at end of file — expected after crash
                    truncated += 1;
                    continue;
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<WalEvent>(trimmed) {
                Ok(event) => events.push(event),
                Err(e) => {
                    truncated += 1;
                    eprintln!(
                        "WAL: skipping malformed line {} ({}): {}",
                        line_num,
                        e,
                        if trimmed.len() > 80 {
                            format!("{}...", &trimmed[..80])
                        } else {
                            trimmed.to_string()
                        }
                    );
                }
            }
        }

        if truncated > 0 {
            eprintln!(
                "WAL: recovered {} events, skipped {} malformed/truncated lines",
                events.len(),
                truncated
            );
        }

        Ok(events)
    }

    /// Compact the WAL: deduplicate UpsertRecord events (last-write-wins
    /// per side+id), keep all CrossMap events in order.
    ///
    /// Rewrites the file atomically (temp + rename).
    pub fn compact(&self, a_id_field: &str, b_id_field: &str) -> io::Result<()> {
        // Flush current buffer first
        self.flush()?;

        // Read all events
        let events = Self::replay(&self.path)?;

        // Deduplicate: for UpsertRecord/RemoveRecord, keep only the latest per (side, id).
        // A RemoveRecord supersedes any prior UpsertRecord for the same key.
        // For CrossMap events, keep all in order.
        use std::collections::{HashMap, HashSet};
        let mut record_latest: HashMap<(String, String), usize> = HashMap::new(); // (side_str, id) -> index
        let mut removed_ids: HashSet<(String, String)> = HashSet::new();
        for (idx, event) in events.iter().enumerate() {
            match event {
                WalEvent::UpsertRecord { side, record } => {
                    let id_field = match side {
                        Side::A => a_id_field,
                        Side::B => b_id_field,
                    };
                    let id = record.get(id_field).cloned().unwrap_or_default();
                    let side_str = match side {
                        Side::A => "a".to_string(),
                        Side::B => "b".to_string(),
                    };
                    let key = (side_str, id);
                    removed_ids.remove(&key);
                    record_latest.insert(key, idx);
                }
                WalEvent::RemoveRecord { side, id } => {
                    let side_str = match side {
                        Side::A => "a".to_string(),
                        Side::B => "b".to_string(),
                    };
                    let key = (side_str, id.clone());
                    // Remove supersedes any prior upsert
                    record_latest.remove(&key);
                    removed_ids.insert(key);
                }
                _ => {}
            }
        }

        // Build compacted event list
        let mut compacted: Vec<&WalEvent> = Vec::new();
        for (idx, event) in events.iter().enumerate() {
            match event {
                WalEvent::UpsertRecord { side, record } => {
                    let id_field = match side {
                        Side::A => a_id_field,
                        Side::B => b_id_field,
                    };
                    let id = record.get(id_field).cloned().unwrap_or_default();
                    let side_str = match side {
                        Side::A => "a".to_string(),
                        Side::B => "b".to_string(),
                    };
                    // Only keep if this is the latest for this (side, id)
                    // and not superseded by a later RemoveRecord
                    if record_latest.get(&(side_str, id)) == Some(&idx) {
                        compacted.push(event);
                    }
                }
                WalEvent::RemoveRecord { side, id } => {
                    let side_str = match side {
                        Side::A => "a".to_string(),
                        Side::B => "b".to_string(),
                    };
                    // Keep remove events for IDs that are currently removed
                    if removed_ids.contains(&(side_str, id.clone())) {
                        compacted.push(event);
                    }
                }
                WalEvent::CrossMapConfirm { .. } | WalEvent::CrossMapBreak { .. } => {
                    compacted.push(event);
                }
            }
        }

        // Write to temp file
        let temp_path = self.path.with_extension("wal.tmp");
        {
            let file = File::create(&temp_path)?;
            let mut writer = BufWriter::new(file);
            for event in &compacted {
                let line = serde_json::to_string(event)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                writer.write_all(line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }

        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;

        // Reopen the file for appending (the old file handle is now stale)
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        *w = BufWriter::new(file);

        eprintln!(
            "WAL compacted: {} events -> {} events",
            events.len(),
            compacted.len()
        );

        Ok(())
    }

    /// Returns the path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for UpsertLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpsertLog")
            .field("path", &self.path)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn append_and_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Append 100 UpsertRecord + 5 CrossMapConfirm
        {
            let log = UpsertLog::open(&path).unwrap();
            for i in 0..100 {
                let rec = make_record(&[
                    ("entity_id", &format!("A-{}", i)),
                    ("legal_name", &format!("Company {}", i)),
                ]);
                log.append(&WalEvent::UpsertRecord {
                    side: Side::A,
                    record: rec,
                })
                .unwrap();
            }
            for i in 0..5 {
                log.append(&WalEvent::CrossMapConfirm {
                    a_id: format!("A-{}", i),
                    b_id: format!("B-{}", i),
                })
                .unwrap();
            }
            log.flush().unwrap();
        }

        // Replay
        let events = UpsertLog::replay(&path).unwrap();
        assert_eq!(events.len(), 105);

        let upserts = events
            .iter()
            .filter(|e| matches!(e, WalEvent::UpsertRecord { .. }))
            .count();
        let confirms = events
            .iter()
            .filter(|e| matches!(e, WalEvent::CrossMapConfirm { .. }))
            .count();
        assert_eq!(upserts, 100);
        assert_eq!(confirms, 5);
    }

    #[test]
    fn replay_tolerates_truncated_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncated.wal");

        // Write valid events
        {
            let log = UpsertLog::open(&path).unwrap();
            for i in 0..10 {
                let rec = make_record(&[("entity_id", &format!("A-{}", i))]);
                log.append(&WalEvent::UpsertRecord {
                    side: Side::A,
                    record: rec,
                })
                .unwrap();
            }
            log.flush().unwrap();
        }

        // Append truncated JSON
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(b"{\"type\":\"upsert_record\",\"si").unwrap();
            // No newline — simulates crash mid-write
        }

        // Replay should recover 10 events and warn about truncated line
        let events = UpsertLog::replay(&path).unwrap();
        assert_eq!(events.len(), 10);
    }

    #[test]
    fn compact_deduplicates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("compact.wal");

        {
            let log = UpsertLog::open(&path).unwrap();

            // Write 50 records, then overwrite them (50 duplicates)
            for i in 0..50 {
                let rec = make_record(&[
                    ("entity_id", &format!("A-{}", i)),
                    ("legal_name", &format!("Company {}", i)),
                ]);
                log.append(&WalEvent::UpsertRecord {
                    side: Side::A,
                    record: rec,
                })
                .unwrap();
            }
            for i in 0..50 {
                let rec = make_record(&[
                    ("entity_id", &format!("A-{}", i)),
                    ("legal_name", &format!("Company {} Updated", i)),
                ]);
                log.append(&WalEvent::UpsertRecord {
                    side: Side::A,
                    record: rec,
                })
                .unwrap();
            }

            // Add crossmap events
            for i in 0..3 {
                log.append(&WalEvent::CrossMapConfirm {
                    a_id: format!("A-{}", i),
                    b_id: format!("B-{}", i),
                })
                .unwrap();
            }
            log.append(&WalEvent::CrossMapBreak {
                a_id: "A-0".into(),
                b_id: "B-0".into(),
            })
            .unwrap();

            log.compact("entity_id", "counterparty_id").unwrap();
        }

        // Replay compacted file
        let events = UpsertLog::replay(&path).unwrap();

        // Should have: 50 unique upserts (latest version) + 3 confirms + 1 break = 54
        let upserts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WalEvent::UpsertRecord { .. }))
            .collect();
        assert_eq!(upserts.len(), 50, "expected 50 deduplicated upserts");

        // Verify latest version was kept
        if let WalEvent::UpsertRecord { record, .. } = &upserts[0] {
            let name = record.get("legal_name").unwrap();
            assert!(
                name.contains("Updated"),
                "expected updated version, got: {}",
                name
            );
        }

        let crossmap_events = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    WalEvent::CrossMapConfirm { .. } | WalEvent::CrossMapBreak { .. }
                )
            })
            .count();
        assert_eq!(
            crossmap_events, 4,
            "all crossmap events should be preserved"
        );
    }

    #[test]
    fn replay_missing_file() {
        let events = UpsertLog::replay(Path::new("/nonexistent/wal.log")).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn replay_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.wal");
        File::create(&path).unwrap();
        let events = UpsertLog::replay(&path).unwrap();
        assert!(events.is_empty());
    }
}
