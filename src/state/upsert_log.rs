//! Write-Ahead Log (WAL) for live mode.
//!
//! Records upsert events and crossmap changes as newline-delimited JSON.
//! Provides replay for crash recovery and compaction for space reclamation.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

/// Cross-platform rename that replaces the destination if it exists.
///
/// On Unix `fs::rename` atomically replaces the target.  On Windows it fails
/// if the destination already exists, so we remove-then-rename (tiny window
/// of non-atomicity, acceptable for WAL compaction).
fn rename_replacing(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        fs::rename(from, to)
    }
    #[cfg(not(unix))]
    {
        let _ = fs::remove_file(to);
        fs::rename(from, to)
    }
}

use serde::{Deserialize, Serialize};

use crate::models::{Record, Side};

/// Format a `SystemTime` as ISO-8601 UTC (e.g. `2026-03-11T18:42:07Z`).
fn iso8601_now() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Split into date/time components
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;

    // Civil date from days since epoch (algorithm from Howard Hinnant)
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if mon <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, mon, d, h, m, s
    )
}

/// Generate a compact timestamp for filenames (e.g. `20260311T184207Z`).
fn filename_timestamp() -> String {
    iso8601_now().replace([':', '-'], "")
}

/// Generate a WAL filename with a startup timestamp.
///
/// Given a configured path like `bench/live_upserts.ndjson`, produces
/// `bench/live_upserts_20260311T184207Z.ndjson`.
fn timestamped_wal_path(base: &Path) -> PathBuf {
    let ts = filename_timestamp();
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base.extension().unwrap_or_default().to_string_lossy();
    let new_name = if ext.is_empty() {
        format!("{}_{}", stem, ts)
    } else {
        format!("{}_{}.{}", stem, ts, ext)
    };
    base.with_file_name(new_name)
}

/// WAL event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WalEvent {
    #[serde(rename = "upsert_record")]
    UpsertRecord { side: Side, record: Record },
    #[serde(rename = "crossmap_confirm")]
    CrossMapConfirm {
        a_id: String,
        b_id: String,
        /// Match score that triggered the confirm. `None` for manual confirms
        /// via the `/crossmap/confirm` API and for backwards-compatible replay
        /// of old WAL files that lack this field.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        score: Option<f64>,
    },
    #[serde(rename = "review_match")]
    ReviewMatch {
        /// The ID of the record that was upserted.
        id: String,
        side: Side,
        /// The best candidate's ID on the opposite side.
        candidate_id: String,
        /// The best candidate's composite score.
        score: f64,
    },
    #[serde(rename = "crossmap_break")]
    CrossMapBreak { a_id: String, b_id: String },
    #[serde(rename = "remove_record")]
    RemoveRecord { side: Side, id: String },
}

/// Write-ahead log for crash recovery.
///
/// Events are JSON-serialized, one per line. The log is append-only.
/// A `Mutex<BufWriter>` protects concurrent writes.
///
/// Each server run creates a new timestamped WAL file. On startup,
/// all WAL files matching the configured base path are replayed in
/// chronological order.
pub struct UpsertLog {
    /// The timestamped path for the current run's WAL file.
    path: PathBuf,
    /// The configured base path (without timestamp) for glob matching.
    #[allow(dead_code)]
    base_path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl UpsertLog {
    /// Open a new timestamped WAL file for the current server run.
    ///
    /// The `base_path` is the configured path (e.g. `bench/live_upserts.ndjson`).
    /// The actual file created will be timestamped (e.g.
    /// `bench/live_upserts_20260311T184207Z.ndjson`).
    pub fn open(base_path: &Path) -> io::Result<Self> {
        let path = timestamped_wal_path(base_path);
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent)?;
        }

        eprintln!("WAL: writing to {}", path.display());
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let writer = Mutex::new(BufWriter::new(file));

        Ok(Self {
            path,
            base_path: base_path.to_path_buf(),
            writer,
        })
    }

    /// Append an event to the WAL, with an ISO-8601 UTC timestamp.
    ///
    /// Serializes to JSON + newline, writes to buffer. Does not fsync —
    /// background flush handles that periodically.
    pub fn append(&self, event: &WalEvent) -> io::Result<()> {
        #[derive(serde::Serialize)]
        struct Timestamped<'a> {
            ts: String,
            #[serde(flatten)]
            event: &'a WalEvent,
        }
        let wrapped = Timestamped {
            ts: iso8601_now(),
            event,
        };
        let line = serde_json::to_string(&wrapped)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    }

    /// Append an upsert event without cloning the record.
    ///
    /// Uses a borrowing struct for serialization to avoid the clone that
    /// `WalEvent::UpsertRecord { record: record.clone() }` would require.
    /// Includes an ISO-8601 UTC timestamp.
    pub fn append_upsert(&self, side: Side, record: &Record) -> io::Result<()> {
        #[derive(serde::Serialize)]
        struct UpsertRef<'a> {
            ts: String,
            #[serde(rename = "type")]
            ty: &'static str,
            side: Side,
            record: &'a Record,
        }
        let event = UpsertRef {
            ts: iso8601_now(),
            ty: "upsert_record",
            side,
            record,
        };
        let line = serde_json::to_string(&event)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    }

    /// Flush the buffered writer to the OS.
    pub fn flush(&self) -> io::Result<()> {
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        w.flush()
    }

    /// Find all WAL files matching the base path pattern.
    ///
    /// Given base path `dir/name.ext`, finds all files matching
    /// `dir/name*.ext` (includes both the exact path and timestamped
    /// variants like `dir/name_20260311T184207Z.ext`).
    /// Returns paths sorted lexicographically (which is chronological
    /// for our timestamp format).
    pub fn find_wal_files(base_path: &Path) -> Vec<PathBuf> {
        let parent = base_path.parent().unwrap_or(Path::new("."));
        let stem = base_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let ext = base_path
            .extension()
            .map(|e| e.to_string_lossy().to_string());

        let entries = match fs::read_dir(parent) {
            Ok(e) => e,
            Err(_) => return vec![],
        };

        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let name_stem = Path::new(&name)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let name_ext = Path::new(&name)
                    .extension()
                    .map(|x| x.to_string_lossy().to_string());
                // Must start with the base stem and have the same extension
                name_stem.starts_with(&stem) && name_ext == ext
            })
            .map(|e| e.path())
            .collect();

        paths.sort();
        paths
    }

    /// Replay all events from a single WAL file.
    ///
    /// Tolerates a truncated last line (logs a warning, skips it).
    /// The `ts` field in events is ignored during deserialization
    /// (serde skips unknown fields by default with the internally
    /// tagged enum).
    fn replay_file(path: &Path) -> io::Result<Vec<WalEvent>> {
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
                        "WAL: skipping malformed line {} in {} ({}): {}",
                        line_num,
                        path.display(),
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
                "WAL: {} — recovered {} events, skipped {} malformed/truncated lines",
                path.display(),
                events.len(),
                truncated
            );
        }

        Ok(events)
    }

    /// Replay all events from all WAL files matching the base path.
    ///
    /// Files are replayed in lexicographic (chronological) order.
    /// Also replays the exact base path for backwards compatibility
    /// with pre-timestamp WAL files.
    pub fn replay(base_path: &Path) -> io::Result<Vec<WalEvent>> {
        let mut all_events = Vec::new();

        // First replay the exact base path if it exists (old-style WAL)
        if base_path.exists() {
            let events = Self::replay_file(base_path)?;
            if !events.is_empty() {
                eprintln!(
                    "WAL: replayed {} events from {} (legacy)",
                    events.len(),
                    base_path.display()
                );
                all_events.extend(events);
            }
        }

        // Then replay all timestamped WAL files
        for path in Self::find_wal_files(base_path) {
            // Skip the base path itself (already replayed above)
            if path == base_path {
                continue;
            }
            let events = Self::replay_file(&path)?;
            if !events.is_empty() {
                eprintln!(
                    "WAL: replayed {} events from {}",
                    events.len(),
                    path.display()
                );
                all_events.extend(events);
            }
        }

        Ok(all_events)
    }

    /// Compact the WAL: deduplicate UpsertRecord events (last-write-wins
    /// per side+id), keep all CrossMap events in order.
    ///
    /// Rewrites the file atomically (temp + rename).
    pub fn compact(&self, a_id_field: &str, b_id_field: &str) -> io::Result<()> {
        // Flush current buffer first
        self.flush()?;

        // Read all events from the current WAL file only
        let events = Self::replay_file(&self.path)?;

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
                WalEvent::CrossMapConfirm { .. }
                | WalEvent::CrossMapBreak { .. }
                | WalEvent::ReviewMatch { .. } => {
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

        // Atomic rename (cross-platform: see rename_replacing)
        rename_replacing(&temp_path, &self.path)?;

        // Reopen the file for appending (the old file handle is now stale)
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
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
                    score: Some(0.9),
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
        let base_path = dir.path().join("truncated.wal");

        // Write valid events, capture the actual timestamped path
        let actual_path;
        {
            let log = UpsertLog::open(&base_path).unwrap();
            actual_path = log.path().to_path_buf();
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

        // Append truncated JSON to the actual file
        {
            let mut file = OpenOptions::new().append(true).open(&actual_path).unwrap();
            file.write_all(b"{\"type\":\"upsert_record\",\"si").unwrap();
            // No newline — simulates crash mid-write
        }

        // Replay should recover 10 events and warn about truncated line
        let events = UpsertLog::replay(&base_path).unwrap();
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
                    score: Some(0.88),
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
        let events =
            UpsertLog::replay(&Path::new("nonexistent_dir_for_test").join("wal.log")).unwrap();
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
