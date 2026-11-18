//! Match log (formerly "upsert log" / WAL) for all modes.
//!
//! Records matching events as newline-delimited JSON: upserts, confirms,
//! reviews, breaks, removes, excludes. Canonical input for output generation.
//! Provides replay for crash recovery and compaction for space reclamation.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config::BatchAssignmentMode;
use crate::util::rename_replacing;

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

// Fixed-width sequences sort after legacy unsuffixed files for the same second.
fn timestamped_wal_path(base: &Path, ts: &str, sequence: u32) -> PathBuf {
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base.extension().unwrap_or_default().to_string_lossy();
    let new_name = if ext.is_empty() {
        format!("{}_{}_{:06}", stem, ts, sequence)
    } else {
        format!("{}_{}_{:06}.{}", stem, ts, sequence, ext)
    };
    base.with_file_name(new_name)
}

/// One scored replacement pair in an atomic reassignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReassignmentMatch {
    pub a_id: String,
    pub b_id: String,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Canonical A/B pair expected by an assignment transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentPair {
    pub a_id: String,
    pub b_id: String,
}

/// Side-qualified endpoint (A and B IDs have separate namespaces).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentEndpoint {
    pub side: Side,
    pub id: String,
}

/// Installed pair with current scores in both directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentMatch {
    pub a_id: String,
    pub b_id: String,
    pub score_b_to_a: f64,
    pub score_a_to_b: f64,
    pub rank: Option<u8>,
    pub reason: Option<String>,
}

/// One committed live deferred-acceptance step. Version 1 implies the strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentDelta {
    pub version: u8,
    pub attempt_id: u64,
    pub step_index: usize,
    pub proposer: AssignmentEndpoint,
    pub expected: Vec<AssignmentPair>,
    pub replacements: Vec<AssignmentMatch>,
    pub newly_unmatched: Vec<AssignmentEndpoint>,
    pub previous_receiver_score: Option<f64>,
}

impl AssignmentDelta {
    /// Reject unsupported schemas and internally inconsistent live transitions.
    pub fn validate(&self) -> io::Result<()> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid assignment_delta v1");
        if self.version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported assignment_delta version {}", self.version),
            ));
        }
        // V1 is one free acceptance or one displacement, not batch publication.
        if self.expected.len() > 1 || self.replacements.len() != 1 {
            return Err(invalid());
        }
        let replacement = &self.replacements[0];
        if !replacement.score_b_to_a.is_finite()
            || !replacement.score_a_to_b.is_finite()
            || self
                .previous_receiver_score
                .is_some_and(|score| !score.is_finite())
            || self.expected.is_empty() != self.previous_receiver_score.is_none()
        {
            return Err(invalid());
        }
        let (proposer, receiver) = match self.proposer.side {
            Side::A => (replacement.a_id.as_str(), replacement.b_id.as_str()),
            Side::B => (replacement.b_id.as_str(), replacement.a_id.as_str()),
        };
        if self.proposer.id != proposer {
            return Err(invalid());
        }
        let mut unmatched = HashSet::new();
        for pair in &self.expected {
            let (old_proposer, old_receiver) = match self.proposer.side {
                Side::A => (pair.a_id.as_str(), pair.b_id.as_str()),
                Side::B => (pair.b_id.as_str(), pair.a_id.as_str()),
            };
            if old_receiver != receiver || old_proposer == proposer {
                return Err(invalid());
            }
            unmatched.insert((self.proposer.side, old_proposer));
        }
        let recorded: HashSet<_> = self
            .newly_unmatched
            .iter()
            .map(|e| (e.side, e.id.as_str()))
            .collect();
        if recorded.len() != self.newly_unmatched.len() || recorded != unmatched {
            return Err(invalid());
        }
        Ok(())
    }
}

/// Match log event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MatchLogEvent {
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
        /// Position in the ranked candidate list. `None` for pre-score paths
        /// (canonical, exact, crossmap) and for backwards-compatible replay.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rank: Option<u8>,
        /// How the match was confirmed. `None` for normal scored matches.
        /// Values: "canonical" | "exact" | "crossmap" | "downgraded".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    #[serde(rename = "crossmap_reassign")]
    CrossMapReassign {
        old_a_id: String,
        old_b_id: String,
        replacements: [ReassignmentMatch; 2],
    },
    #[serde(rename = "assignment_delta")]
    AssignmentDelta(AssignmentDelta),
    #[serde(rename = "batch_assignment_started")]
    BatchAssignmentStarted { mode: BatchAssignmentMode },
    #[serde(rename = "batch_assignment_completed")]
    BatchAssignmentCompleted {
        retained_edges: usize,
        fixed_pairs: usize,
        scale: i64,
        exact_groups: usize,
        approximate_groups: usize,
        largest_group: usize,
        /// Decimal strings on the wire: internally tagged serde enums buffer
        /// numbers in a representation that cannot deserialize arbitrary i128.
        #[serde(with = "decimal_i128")]
        objective_units: i128,
        #[serde(with = "decimal_i128")]
        upper_units: i128,
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
        /// Position in the ranked candidate list. `None` for backwards compat.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rank: Option<u8>,
        /// How the review was triggered. `None` for normal; "downgraded" for
        /// min_score_gap demotion.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    #[serde(rename = "no_match_below")]
    NoMatchBelow {
        query_id: String,
        query_side: Side,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        best_candidate_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        best_score: Option<f64>,
    },
    #[serde(rename = "crossmap_break")]
    CrossMapBreak { a_id: String, b_id: String },
    #[serde(rename = "remove_record")]
    RemoveRecord { side: Side, id: String },
    #[serde(rename = "exclude")]
    Exclude { a_id: String, b_id: String },
    #[serde(rename = "unexclude")]
    Unexclude { a_id: String, b_id: String },
}

// Keep the Rust API checked/integer-valued without lossy JSON number conversion.
mod decimal_i128 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &i128, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i128, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Require every batch optimization start to have exactly one completion.
///
/// Replay itself remains usable for recovery of unfinished WAL prefixes. Output
/// consumers must call this before creating files, even with an authoritative
/// in-process snapshot. Legacy logs and live assignment deltas need no markers.
pub fn validate_batch_assignment_framing(events: &[MatchLogEvent]) -> io::Result<()> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut active = None;
    for (position, event) in events.iter().enumerate() {
        match event {
            MatchLogEvent::BatchAssignmentStarted { mode } => {
                if position != 0 || active.replace(*mode).is_some() {
                    return Err(invalid(
                        "batch assignment export requires a single run log segment",
                    ));
                }
            }
            MatchLogEvent::BatchAssignmentCompleted {
                scale,
                approximate_groups,
                objective_units,
                upper_units,
                ..
            } => {
                let mode = active
                    .take()
                    .ok_or_else(|| invalid("batch_assignment_completed without start"))?;
                if position + 1 != events.len() {
                    return Err(invalid(
                        "batch assignment export requires a single run log segment",
                    ));
                }
                if *scale != crate::matching::assignment::SCALE
                    || *objective_units < 0
                    || upper_units < objective_units
                {
                    return Err(invalid("invalid batch assignment objective bounds"));
                }
                if mode == BatchAssignmentMode::Exact
                    && (*approximate_groups != 0 || upper_units != objective_units)
                {
                    return Err(invalid("exact batch assignment completion is not exact"));
                }
            }
            _ => {}
        }
    }
    if active.is_some() {
        return Err(invalid("batch assignment log lacks completion"));
    }
    Ok(())
}

/// Reconstruct membership independently of a possibly newer CSV snapshot.
///
/// Delta expectations on endpoints with no earlier membership history describe
/// initially preloaded pairs. Later stale expectations are never inferred.
/// Untouched snapshot pairs are retained; restored links carry no provenance.
pub fn reconstruct_assignment_history(
    events: &[MatchLogEvent],
    snapshot: &[(String, String)],
) -> io::Result<Vec<(String, String)>> {
    use crate::crossmap::{CrossMapOps, MemoryCrossMap};

    let mut footprint = HashSet::new();
    for event in events {
        match event {
            MatchLogEvent::AssignmentDelta(delta) => {
                delta.validate()?;
                for pair in &delta.expected {
                    footprint.insert((pair.a_id.as_str(), pair.b_id.as_str()));
                }
                for pair in &delta.replacements {
                    footprint.insert((pair.a_id.as_str(), pair.b_id.as_str()));
                }
            }
            MatchLogEvent::CrossMapReassign {
                old_a_id,
                old_b_id,
                replacements,
            } => {
                footprint.insert((old_a_id.as_str(), old_b_id.as_str()));
                for pair in replacements {
                    footprint.insert((pair.a_id.as_str(), pair.b_id.as_str()));
                }
            }
            MatchLogEvent::CrossMapConfirm { a_id, b_id, .. }
            | MatchLogEvent::CrossMapBreak { a_id, b_id } => {
                footprint.insert((a_id.as_str(), b_id.as_str()));
            }
            _ => {}
        }
    }
    let map = MemoryCrossMap::new();
    for (a, b) in snapshot {
        // Unknown snapshot ownership must not be silently discarded merely
        // because it overlaps a delta. Claims must detect that conflict.
        if !footprint.contains(&(a.as_str(), b.as_str())) && !map.claim(a, b) {
            return Err(io::Error::other("invalid snapshot bijection"));
        }
    }
    let mut seen = HashSet::new();
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "WAL assignment ownership invariant failed",
        )
    };
    for event in events {
        match event {
            MatchLogEvent::AssignmentDelta(delta) => {
                for pair in &delta.expected {
                    if !seen.contains(&(Side::A, pair.a_id.clone()))
                        && !seen.contains(&(Side::B, pair.b_id.clone()))
                        && !map.claim(&pair.a_id, &pair.b_id)
                    {
                        return Err(invalid());
                    }
                }
                let expected: Vec<_> = delta
                    .expected
                    .iter()
                    .map(|p| (p.a_id.as_str(), p.b_id.as_str()))
                    .collect();
                let replacements: Vec<_> = delta
                    .replacements
                    .iter()
                    .map(|p| (p.a_id.as_str(), p.b_id.as_str()))
                    .collect();
                let already_applied = replacements.iter().all(|(a, b)| {
                    map.get_b(a).as_deref() == Some(*b) && map.get_a(b).as_deref() == Some(*a)
                }) && delta.newly_unmatched.iter().all(|e| match e.side {
                    Side::A => !map.has_a(&e.id),
                    Side::B => !map.has_b(&e.id),
                });
                if !already_applied
                    && !map
                        .apply_assignment_delta(&expected, &replacements)
                        .map_err(io::Error::other)?
                {
                    return Err(invalid());
                }
                for (a, b) in expected.into_iter().chain(replacements) {
                    seen.insert((Side::A, a.to_string()));
                    seen.insert((Side::B, b.to_string()));
                }
            }
            MatchLogEvent::CrossMapConfirm { a_id, b_id, .. } => {
                let displaced = map.confirm(a_id, b_id).map_err(io::Error::other)?;
                if let Some(a) = displaced.displaced_a {
                    seen.insert((Side::A, a));
                }
                if let Some(b) = displaced.displaced_b {
                    seen.insert((Side::B, b));
                }
                seen.insert((Side::A, a_id.clone()));
                seen.insert((Side::B, b_id.clone()));
            }
            MatchLogEvent::CrossMapBreak { a_id, b_id } => {
                map.remove(a_id, b_id);
                seen.insert((Side::A, a_id.clone()));
                seen.insert((Side::B, b_id.clone()));
            }
            MatchLogEvent::CrossMapReassign {
                old_a_id,
                old_b_id,
                replacements,
            } => {
                let pairs = replacements
                    .each_ref()
                    .map(|p| (p.a_id.as_str(), p.b_id.as_str()));
                let applied = pairs.iter().all(|(a, b)| {
                    map.get_b(a).as_deref() == Some(*b) && map.get_a(b).as_deref() == Some(*a)
                });
                if !applied
                    && !map
                        .replace_pairs(&[(old_a_id, old_b_id)], &pairs)
                        .map_err(io::Error::other)?
                {
                    return Err(invalid());
                }
                seen.insert((Side::A, old_a_id.clone()));
                seen.insert((Side::B, old_b_id.clone()));
                for (a, b) in pairs {
                    seen.insert((Side::A, a.to_string()));
                    seen.insert((Side::B, b.to_string()));
                }
            }
            MatchLogEvent::RemoveRecord { side, id } => {
                match side {
                    Side::A => {
                        if let Some(b) = map.get_b(id) {
                            map.remove(id, &b);
                            seen.insert((Side::B, b));
                        }
                    }
                    Side::B => {
                        if let Some(a) = map.get_a(id) {
                            map.remove(&a, id);
                            seen.insert((Side::A, a));
                        }
                    }
                }
                seen.insert((*side, id.clone()));
            }
            _ => {}
        }
    }
    Ok(map.pairs())
}

// Accept only incomplete JSON with a recognized canonical header. For a torn
// UTF-8 code point, parse the valid prefix; other invalid UTF-8 is corruption.
fn is_torn_final_frame(line: &[u8]) -> bool {
    if line.ends_with(b"\n") {
        return false;
    }
    let text = match std::str::from_utf8(line) {
        Ok(text) => text,
        Err(e) if e.error_len().is_none() => {
            // valid_up_to is guaranteed to be a UTF-8 boundary.
            std::str::from_utf8(&line[..e.valid_up_to()]).expect("valid UTF-8 prefix")
        }
        Err(_) => return false,
    };
    let header = text
        .strip_prefix(r#"{"ts":""#)
        .and_then(|rest| rest.split_once(r#"","#))
        .map(|(_, event)| event)
        .or_else(|| text.strip_prefix('{'));
    let known_header = header.is_some_and(|header| {
        [
            r#""type":"upsert_record","#,
            r#""type":"crossmap_confirm","#,
            r#""type":"crossmap_reassign","#,
            r#""type":"assignment_delta","version":1,"#,
            r#""type":"review_match","#,
            r#""type":"no_match_below","#,
            r#""type":"crossmap_break","#,
            r#""type":"remove_record","#,
            r#""type":"exclude","#,
            r#""type":"unexclude","#,
        ]
        .iter()
        .any(|prefix| header.starts_with(prefix))
    });
    known_header && serde_json::from_str::<MatchLogEvent>(text).is_err_and(|error| error.is_eof())
}

/// Write-ahead log for crash recovery.
///
/// Events are JSON-serialized, one per line. The log is append-only.
/// A `Mutex<BufWriter>` protects concurrent writes.
///
/// Each server run creates a new timestamped WAL file. On startup,
/// all WAL files matching the configured base path are replayed in
/// chronological order.
pub struct MatchLog {
    /// The timestamped path for the current run's WAL file.
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl MatchLog {
    /// Open a new timestamped WAL file for the current server run.
    ///
    /// The `base_path` is the configured path (e.g. `bench/live_upserts.ndjson`).
    /// The actual file created will be timestamped (e.g.
    /// `bench/live_upserts_20260311T184207Z_000001.ndjson`).
    pub fn open(base_path: &Path) -> io::Result<Self> {
        Self::open_timestamped(base_path, &filename_timestamp())
    }

    fn open_timestamped(base_path: &Path, timestamp: &str) -> io::Result<Self> {
        let path = timestamped_wal_path(base_path, timestamp, 1);
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent)?;
        }

        let prefix = format!(
            "{}_{}_",
            base_path.file_stem().unwrap_or_default().to_string_lossy(),
            timestamp,
        );
        let mut sequence = Self::find_wal_files(base_path)
            .iter()
            .filter_map(|path| {
                path.file_stem()?
                    .to_str()?
                    .strip_prefix(&prefix)?
                    .parse::<u32>()
                    .ok()
            })
            .max()
            .unwrap_or(0);
        loop {
            sequence = sequence
                .checked_add(1)
                .filter(|n| *n <= 999_999)
                .ok_or_else(|| io::Error::other("WAL segment sequence exhausted"))?;
            let path = timestamped_wal_path(base_path, timestamp, sequence);
            match OpenOptions::new().create_new(true).append(true).open(&path) {
                Ok(file) => {
                    info!(path = %path.display(), "WAL opened for writing");
                    return Ok(Self {
                        path,
                        writer: Mutex::new(BufWriter::new(file)),
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// Append an event to the WAL, with an ISO-8601 UTC timestamp.
    ///
    /// Serializes to JSON + newline, writes to buffer. Does not fsync —
    /// background flush handles that periodically.
    pub fn append(&self, event: &MatchLogEvent) -> io::Result<()> {
        let bytes = Self::serialize_event(event)?;
        self.append_bytes(&bytes)
    }

    /// Serialize an event to bytes (JSON + newline) without acquiring the lock.
    ///
    /// Use with `append_bytes()` to move serialization out of the critical
    /// section in hot paths (e.g. Rayon parallel scoring).
    pub fn serialize_event(event: &MatchLogEvent) -> io::Result<Vec<u8>> {
        #[derive(serde::Serialize)]
        struct Timestamped<'a> {
            ts: String,
            #[serde(flatten)]
            event: &'a MatchLogEvent,
        }
        if let MatchLogEvent::AssignmentDelta(delta) = event {
            delta.validate()?;
        }
        let wrapped = Timestamped {
            ts: iso8601_now(),
            event,
        };
        let mut bytes = serde_json::to_vec(&wrapped)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Append pre-serialized bytes to the WAL.
    ///
    /// The mutex is held only for the `write_all` call (~100ns memcpy),
    /// not for serialization.
    pub fn append_bytes(&self, bytes: &[u8]) -> io::Result<()> {
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        w.write_all(bytes)?;
        Ok(())
    }

    /// Append an upsert event without cloning the record.
    ///
    /// Uses a borrowing struct for serialization to avoid the clone that
    /// `MatchLogEvent::UpsertRecord { record: record.clone() }` would require.
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

    /// Flush the buffered writer to stable storage.
    ///
    /// Calls `BufWriter::flush()` to push to the OS page cache, then
    /// `File::sync_all()` to fsync to disk. Without the sync, a kernel
    /// crash within the OS buffer window could lose events the WAL was
    /// supposed to protect.
    pub fn flush(&self) -> io::Result<()> {
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        w.flush()?;
        w.get_ref().sync_all()
    }

    /// Seal a graph log only after its events have been synced. On failure,
    /// remove any partially written completion bytes so export fails closed.
    pub(crate) fn complete_batch_assignment(&self, event: &MatchLogEvent) -> io::Result<()> {
        self.seal_batch_assignment(event, |mut file, bytes| {
            file.write_all(bytes)?;
            file.sync_all()
        })
    }

    pub(crate) fn seal_batch_assignment(
        &self,
        event: &MatchLogEvent,
        seal: impl FnOnce(&File, &[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        if !matches!(event, MatchLogEvent::BatchAssignmentCompleted { .. }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a batch completion event",
            ));
        }
        let bytes = Self::serialize_event(event)?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        writer.flush()?;
        let file = writer.get_ref();
        file.sync_all()?;
        let prefix_len = file.metadata()?.len();
        if let Err(e) = seal(file, &bytes) {
            if file
                .set_len(prefix_len)
                .and_then(|()| file.sync_all())
                .is_err()
            {
                // Prefer making this segment unavailable to leaving a false success.
                std::fs::remove_file(&self.path).map_err(|unlink| {
                    io::Error::other(format!(
                        "batch completion failed ({e}); could not invalidate log ({unlink})"
                    ))
                })?;
            }
            return Err(e);
        }
        Ok(())
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
    fn replay_file(path: &Path) -> io::Result<Vec<MatchLogEvent>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut events = Vec::new();
        let mut line_num = 0;
        let mut truncated = 0;

        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            line_num += 1;

            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }

            match serde_json::from_slice::<MatchLogEvent>(&line) {
                Ok(event) => {
                    if let MatchLogEvent::AssignmentDelta(delta) = &event {
                        delta.validate().map_err(|e| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("{}:{}: {}", path.display(), line_num, e),
                            )
                        })?;
                    }
                    events.push(event);
                }
                Err(e) => {
                    // Recover a torn final frame written by this logger, but
                    // never skip a corrupt complete line or an unknown schema.
                    if !is_torn_final_frame(&line) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("{}:{}: {}", path.display(), line_num, e),
                        ));
                    }
                    truncated += 1;
                    warn!(
                        line = line_num,
                        path = %path.display(),
                        error = %e,
                        "WAL skipping malformed line"
                    );
                }
            }
        }

        if truncated > 0 {
            warn!(
                path = %path.display(),
                recovered = events.len(),
                skipped = truncated,
                "WAL replay had malformed lines"
            );
        }

        Ok(events)
    }

    /// Replay all events from all WAL files matching the base path.
    ///
    /// Files are replayed in lexicographic (chronological) order.
    /// Also replays the exact base path for backwards compatibility
    /// with pre-timestamp WAL files.
    pub fn replay(base_path: &Path) -> io::Result<Vec<MatchLogEvent>> {
        let mut all_events = Vec::new();

        // First replay the exact base path if it exists (old-style WAL)
        if base_path.exists() {
            let events = Self::replay_file(base_path)?;
            if !events.is_empty() {
                info!(events = events.len(), path = %base_path.display(), "WAL replayed legacy file");
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
                info!(events = events.len(), path = %path.display(), "WAL replayed");
                all_events.extend(events);
            }
        }

        Ok(all_events)
    }

    /// Compact the WAL: deduplicate UpsertRecord events (last-write-wins
    /// per side+id), keep all CrossMap events in order.
    ///
    /// Rewrites only the current segment (synced temp + rename). Prior segments
    /// must be retained: this is not a checkpoint of recovered history.
    ///
    /// Holds the writer lock for the entire operation so that concurrent
    /// `append()` calls block until compaction is complete and the writer
    /// has been swapped to the new file. Without this, writes between
    /// `flush()` and the writer swap would go to the old (renamed-away)
    /// file and be silently lost.
    pub fn compact(&self, a_id_field: &str, b_id_field: &str) -> io::Result<()> {
        // Hold the writer lock for the entire compaction to prevent
        // concurrent append() calls from writing to a stale file handle.
        let mut w = self
            .writer
            .lock()
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Flush current buffer first (under the lock).
        w.flush()?;

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
                MatchLogEvent::UpsertRecord { side, record } => {
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
                MatchLogEvent::RemoveRecord { side, id } => {
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

        // A removal can break membership installed in an older segment, even
        // when this segment contains no assignment delta. Likewise, an upsert
        // clears its query's reviews. Without checkpoint offsets, retain the
        // complete ordering whenever either interaction is possible. Do this
        // for legacy segments too: a later run may introduce assignment history.
        let preserve_history = events.iter().any(|event| {
            matches!(
                event,
                MatchLogEvent::AssignmentDelta(_)
                    | MatchLogEvent::RemoveRecord { .. }
                    | MatchLogEvent::ReviewMatch { .. }
            )
        });
        // Build compacted event list
        let mut compacted: Vec<&MatchLogEvent> = Vec::new();
        for (idx, event) in events.iter().enumerate() {
            match event {
                MatchLogEvent::UpsertRecord { side, record } => {
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
                    if preserve_history || record_latest.get(&(side_str, id)) == Some(&idx) {
                        compacted.push(event);
                    }
                }
                MatchLogEvent::RemoveRecord { side, id } => {
                    let side_str = match side {
                        Side::A => "a".to_string(),
                        Side::B => "b".to_string(),
                    };
                    // Keep remove events for IDs that are currently removed
                    if preserve_history || removed_ids.contains(&(side_str, id.clone())) {
                        compacted.push(event);
                    }
                }
                MatchLogEvent::CrossMapConfirm { .. }
                | MatchLogEvent::CrossMapBreak { .. }
                | MatchLogEvent::CrossMapReassign { .. }
                | MatchLogEvent::AssignmentDelta(_)
                | MatchLogEvent::BatchAssignmentStarted { .. }
                | MatchLogEvent::BatchAssignmentCompleted { .. }
                | MatchLogEvent::ReviewMatch { .. }
                | MatchLogEvent::NoMatchBelow { .. }
                | MatchLogEvent::Exclude { .. }
                | MatchLogEvent::Unexclude { .. } => {
                    compacted.push(event);
                }
            }
        }

        // Write to temp file
        let temp_path = self.path.with_extension("wal.tmp");
        {
            let file = File::create(&temp_path)?;
            let mut tmp_writer = BufWriter::new(file);
            for event in &compacted {
                let line = serde_json::to_string(event)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                tmp_writer.write_all(line.as_bytes())?;
                tmp_writer.write_all(b"\n")?;
            }
            tmp_writer.flush()?;
            tmp_writer.get_ref().sync_all()?;
        }

        // Atomic rename (cross-platform: see rename_replacing)
        rename_replacing(&temp_path, &self.path)?;

        // Reopen the file for appending and swap the writer (still under lock).
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        *w = BufWriter::new(file);

        info!(
            before = events.len(),
            after = compacted.len(),
            "WAL compacted"
        );

        Ok(())
    }

    /// Returns the path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for MatchLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchLog")
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

    fn make_batch_completion() -> MatchLogEvent {
        MatchLogEvent::BatchAssignmentCompleted {
            retained_edges: 3,
            fixed_pairs: 0,
            scale: crate::matching::assignment::SCALE,
            exact_groups: 1,
            approximate_groups: 0,
            largest_group: 4,
            objective_units: i128::MAX,
            upper_units: i128::MAX,
        }
    }

    #[test]
    fn batch_assignment_framing_rejects_incomplete_and_misordered_logs() {
        let start = MatchLogEvent::BatchAssignmentStarted {
            mode: BatchAssignmentMode::Exact,
        };
        let complete = make_batch_completion();
        for (name, events, valid) in [
            ("legacy", vec![make_reassignment()], true),
            ("live", vec![make_assignment_delta(None, "B", 0)], true),
            ("complete", vec![start.clone(), complete.clone()], true),
            ("unfinished", vec![start.clone()], false),
            ("orphan", vec![complete.clone()], false),
            ("reversed", vec![complete.clone(), start.clone()], false),
            (
                "nested",
                vec![start.clone(), start.clone(), complete.clone()],
                false,
            ),
            (
                "duplicate completion",
                vec![start.clone(), complete.clone(), complete.clone()],
                false,
            ),
            (
                "unfinished second run",
                vec![start.clone(), complete, start],
                false,
            ),
        ] {
            assert_eq!(
                validate_batch_assignment_framing(&events).is_ok(),
                valid,
                "{name}"
            );
        }
    }

    #[test]
    fn batch_assignment_totals_roundtrip_and_compaction_preserve_framing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("batch.wal");
        let log = MatchLog::open(&path).expect("log");
        log.append(&MatchLogEvent::BatchAssignmentStarted {
            mode: BatchAssignmentMode::Exact,
        })
        .expect("start");
        let completion = make_batch_completion();
        let bytes = MatchLog::serialize_event(&completion).expect("serialize i128");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        assert_eq!(
            json["objective_units"],
            i128::MAX.to_string(),
            "totals are decimal strings"
        );
        log.append_bytes(&bytes).expect("complete");
        log.flush().expect("flush");
        log.compact("id", "id").expect("compact");
        let events = MatchLog::replay(&path).expect("replay");
        validate_batch_assignment_framing(&events).expect("complete framing survives compaction");
        assert!(
            matches!(
                events.last(),
                Some(MatchLogEvent::BatchAssignmentCompleted {
                    objective_units: i128::MAX,
                    upper_units: i128::MAX,
                    ..
                })
            ),
            "large totals survive tagged enum roundtrip"
        );
    }

    #[test]
    fn batch_assignment_totals_reject_lossy_or_overflowing_json() {
        for value in [
            "1.5",
            "170141183460469231731687303715884105727",
            "\"170141183460469231731687303715884105728\"",
            "\"1.5\"",
        ] {
            let json = format!(
                r#"{{"type":"batch_assignment_completed","retained_edges":0,"exact_groups":0,"approximate_groups":0,"largest_group":0,"objective_units":{value},"upper_units":"0"}}"#,
            );
            assert!(
                serde_json::from_str::<MatchLogEvent>(&json).is_err(),
                "reject non-string, fractional or overflowing total: {value}"
            );
        }
    }

    #[test]
    fn batch_assignment_completion_rejects_inconsistent_bounds() {
        for (mode, approximate_groups, objective_units, upper_units) in [
            (BatchAssignmentMode::Approximate, 1, -1, 0),
            (BatchAssignmentMode::Approximate, 1, 2, 1),
            (BatchAssignmentMode::Exact, 0, 1, 2),
            (BatchAssignmentMode::Exact, 1, 1, 1),
        ] {
            let events = [
                MatchLogEvent::BatchAssignmentStarted { mode },
                MatchLogEvent::BatchAssignmentCompleted {
                    retained_edges: 1,
                    fixed_pairs: 0,
                    scale: crate::matching::assignment::SCALE,
                    exact_groups: 0,
                    approximate_groups,
                    largest_group: 2,
                    objective_units,
                    upper_units,
                },
            ];
            assert!(
                validate_batch_assignment_framing(&events).is_err(),
                "reject invalid {mode:?} completion"
            );
        }
    }

    fn make_reassignment() -> MatchLogEvent {
        serde_json::from_str(
            r#"{"type":"crossmap_reassign","old_a_id":"A1","old_b_id":"B1","replacements":[{"a_id":"A1","b_id":"B2","score":0.91,"rank":2},{"a_id":"A2","b_id":"B1","score":0.87,"reason":"exact"}]}"#,
        )
        .expect("valid reassignment fixture")
    }

    fn make_assignment_delta(old: Option<&str>, new: &str, step: usize) -> MatchLogEvent {
        MatchLogEvent::AssignmentDelta(AssignmentDelta {
            version: 1,
            attempt_id: 42,
            step_index: step,
            proposer: AssignmentEndpoint {
                side: Side::B,
                id: new.into(),
            },
            expected: old
                .map(|b| AssignmentPair {
                    a_id: "A".into(),
                    b_id: b.into(),
                })
                .into_iter()
                .collect(),
            replacements: vec![AssignmentMatch {
                a_id: "A".into(),
                b_id: new.into(),
                score_b_to_a: 0.91,
                score_a_to_b: 0.95,
                rank: None,
                reason: None,
            }],
            newly_unmatched: old
                .map(|b| AssignmentEndpoint {
                    side: Side::B,
                    id: b.into(),
                })
                .into_iter()
                .collect(),
            previous_receiver_score: old.map(|_| 0.9),
        })
    }

    #[test]
    fn quiet_compaction_retains_prior_crash_segments_without_duplication() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path().join("crash.ndjson");
        let timestamp = "20260311T184207Z";
        let legacy = dir.path().join(format!("crash_{timestamp}.ndjson"));
        let upsert = MatchLogEvent::UpsertRecord {
            side: Side::B,
            record: make_record(&[("id", "old"), ("name", "recovered record")]),
        };
        fs::write(
            &legacy,
            MatchLog::serialize_event(&upsert).expect("serialize"),
        )
        .expect("legacy segment");
        let first = MatchLog::open_timestamped(&base, timestamp).expect("first run");
        first
            .append(&make_assignment_delta(Some("old"), "middle", 0))
            .expect("first displacement");
        first.flush().expect("persist before crash");
        let first_path = first.path().to_path_buf();
        drop(first); // No compaction: simulate a crashed run's persisted prefix.
        let second = MatchLog::open_timestamped(&base, timestamp).expect("second run");
        assert!(second.path() > first_path.as_path(), "same-second ordering");
        second
            .append(&make_assignment_delta(Some("middle"), "last", 1))
            .expect("second displacement");
        second.flush().expect("persist before second crash");
        let second_path = second.path().to_path_buf();
        drop(second);
        OpenOptions::new()
            .append(true)
            .open(&second_path)
            .expect("crashed segment")
            .write_all(b"{\"type\":\"crossmap_confirm\",\"a_id\":\"torn")
            .expect("torn append");
        let old_bytes = fs::read(&second_path).expect("read crash segment");
        let before = MatchLog::replay(&base).expect("recover crashes");
        assert_eq!(before.len(), 3, "recover record and both audit steps once");
        let expected = serde_json::to_value(&before).expect("expected history");
        // Quiet restart -> shutdown -> restart, repeated, must retain all history.
        for _ in 0..3 {
            let quiet = MatchLog::open_timestamped(&base, timestamp).expect("quiet restart");
            assert!(
                quiet.path() > second_path.as_path(),
                "fresh chronological segment"
            );
            quiet.compact("id", "id").expect("quiet shutdown");
            drop(quiet);
            let events = MatchLog::replay(&base).expect("restart after quiet shutdown");
            assert_eq!(
                serde_json::to_value(&events).expect("actual history"),
                expected,
                "prior records, touched pairs and audit survive without duplication"
            );
            assert_eq!(
                reconstruct_assignment_history(&events, &[("A".into(), "last".into())])
                    .expect("recover with latest snapshot"),
                vec![("A".into(), "last".into())],
                "complete footprint reconstructs membership from a newer snapshot"
            );
            assert_eq!(
                fs::read(&second_path).expect("retained crash segment"),
                old_bytes
            );
        }
    }

    #[test]
    fn segment_sequences_sort_after_legacy_and_never_fill_old_gaps() {
        for name in ["segments.ndjson", "segments"] {
            let dir = tempdir().expect("tempdir");
            let base = dir.path().join(name);
            let timestamp = "20260311T184207Z";
            let legacy = timestamped_wal_path(&base, timestamp, 1);
            let legacy = legacy.with_file_name(
                legacy
                    .file_name()
                    .expect("filename")
                    .to_string_lossy()
                    .replace("_000001", ""),
            );
            fs::write(&legacy, b"").expect("legacy segment");
            let high = timestamped_wal_path(&base, timestamp, 9);
            fs::write(&high, b"").expect("existing segment with gaps");
            let log = MatchLog::open_timestamped(&base, timestamp).expect("new segment");
            assert_eq!(log.path(), timestamped_wal_path(&base, timestamp, 10));
            assert_eq!(
                MatchLog::find_wal_files(&base),
                vec![legacy, high, log.path().to_path_buf()],
                "extensionless and legacy names retain chronological order"
            );
        }
    }

    #[test]
    fn torn_payloads_of_every_known_event_recover_including_unicode_bytes() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("torn.ndjson");
        let ordinary = [
            r#"{"type":"upsert_record","side":"a","record":{"id":"é東京"}}"#,
            r#"{"type":"crossmap_confirm","a_id":"é東京","b_id":"B"}"#,
            r#"{"type":"review_match","id":"é東京","side":"b","candidate_id":"A","score":0.7}"#,
            r#"{"type":"no_match_below","query_id":"é東京","query_side":"b"}"#,
            r#"{"type":"crossmap_break","a_id":"é東京","b_id":"B"}"#,
            r#"{"type":"remove_record","side":"a","id":"é東京"}"#,
            r#"{"type":"exclude","a_id":"é東京","b_id":"B"}"#,
            r#"{"type":"unexclude","a_id":"é東京","b_id":"B"}"#,
        ];
        let mut events: Vec<MatchLogEvent> = ordinary
            .iter()
            .map(|json| serde_json::from_str(json).expect("ordinary fixture"))
            .collect();
        events.push(make_reassignment());
        events.push(make_assignment_delta(Some("old"), "é東京", 0));
        let prior = MatchLog::serialize_event(&events[0]).expect("prior complete event");
        for event in events {
            // Test both timestamped logger output and timestamp-free compacted output.
            for bytes in [MatchLog::serialize_event(&event).expect("serialize"), {
                let mut bytes = serde_json::to_vec(&event).expect("compact serialization");
                bytes.push(b'\n');
                bytes
            }] {
                let text = std::str::from_utf8(&bytes).expect("serialized UTF-8");
                let header_end = if matches!(event, MatchLogEvent::AssignmentDelta(_)) {
                    text.find("\"version\":1,").expect("version header") + "\"version\":1,".len()
                } else {
                    let start = text.find("\"type\":").expect("type header");
                    start + text[start..].find(',').expect("end of type header") + 1
                };
                for cut in header_end..bytes.len() - 1 {
                    let mut crash = prior.clone();
                    crash.extend_from_slice(&bytes[..cut]);
                    fs::write(&path, &crash).expect("crash fixture");
                    assert_eq!(
                        MatchLog::replay(&path)
                            .expect("recover torn known frame")
                            .len(),
                        1,
                        "only prior complete frame survives cut {cut}: {text}"
                    );
                    crash.push(b'\n');
                    fs::write(&path, &crash).expect("malformed complete fixture");
                    assert!(
                        MatchLog::replay(&path).is_err(),
                        "newline frame must fail at cut {cut}"
                    );
                }
            }
        }
    }

    #[test]
    fn corrupt_tails_and_unknown_headers_fail_closed() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad.ndjson");
        for bytes in [
            br#"{"type":"future_event","id":"torn"#.as_slice(),
            br#"{"type":"assignment_delta","version":2,"attempt_id":"#.as_slice(),
            br#"{"type":"assignment_delta","version":10,"attempt_id":"#.as_slice(),
            br#"{"type":"assignment_delta","version":1"#.as_slice(),
            br#"{"type":"crossmap_break","a_id":!"#.as_slice(),
            b"{\"type\":\"remove_record\",\"side\":\"a\",\"id\":\"\xff".as_slice(),
            b"{\"type\":\"remove_record\",\"side\":\"a\",\"id\":\"\xe6\xff".as_slice(),
            b"{\"type\":\"remove_record\",\"side\":\"a\",\"id\":\"\xe6\n".as_slice(),
        ] {
            fs::write(&path, bytes).expect("corrupt tail");
            assert!(
                MatchLog::replay(&path).is_err(),
                "reject corrupt tail: {bytes:?}"
            );
        }
    }

    #[test]
    fn assignment_history_recovers_preloaded_pairs_and_newer_snapshot() {
        let first = make_assignment_delta(Some("old"), "middle", 0);
        let last = make_assignment_delta(Some("middle"), "last", 1);
        for snapshot_b in ["old", "middle", "last"] {
            let snapshot = vec![
                ("A".into(), snapshot_b.into()),
                ("protected".into(), "P".into()),
            ];
            let mut recovered = reconstruct_assignment_history(
                &[first.clone(), last.clone(), last.clone()],
                &snapshot,
            )
            .expect("recover history");
            recovered.sort();
            assert_eq!(
                recovered,
                vec![
                    ("A".into(), "last".into()),
                    ("protected".into(), "P".into())
                ],
                "snapshot age must not affect result"
            );
        }
        let mut invalid = match last.clone() {
            MatchLogEvent::AssignmentDelta(d) => d,
            _ => unreachable!(),
        };
        invalid.expected[0].b_id = "old".into();
        invalid.newly_unmatched[0].id = "old".into();
        assert!(
            reconstruct_assignment_history(
                &[first.clone(), MatchLogEvent::AssignmentDelta(invalid)],
                &[]
            )
            .is_err(),
            "stale expectations are not inferred after first use"
        );
        let mixed = vec![
            ("A".into(), "last".into()),
            ("other".into(), "middle".into()),
        ];
        assert!(
            reconstruct_assignment_history(&[first, last], &mixed).is_err(),
            "mixed snapshot must not silently discard unrelated ownership"
        );
    }

    #[test]
    fn assignment_history_claim_remove_and_side_namespaces() {
        let claim = make_assignment_delta(None, "B", 0);
        let remove = MatchLogEvent::RemoveRecord {
            side: Side::A,
            id: "A".into(),
        };
        let snapshot = vec![("B".into(), "A".into())];
        let pairs =
            reconstruct_assignment_history(&[claim, remove], &snapshot).expect("remove history");
        assert_eq!(
            pairs, snapshot,
            "removal must use actual side-qualified ownership"
        );
        let conflict = vec![("A".into(), "unrelated".into())];
        assert!(
            reconstruct_assignment_history(&[make_assignment_delta(None, "B", 0)], &conflict)
                .is_err(),
            "free claim must not evict unknown snapshot owner"
        );
    }

    #[test]
    fn separate_segment_compaction_retains_remove_before_recreate() {
        for assignment_first in [false, true] {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("segments.wal");
            {
                let log = MatchLog::open(&path).expect("first segment");
                let claim = if assignment_first {
                    make_assignment_delta(None, "B", 0)
                } else {
                    MatchLogEvent::CrossMapConfirm {
                        a_id: "A".into(),
                        b_id: "B".into(),
                        score: Some(1.0),
                        rank: None,
                        reason: None,
                    }
                };
                log.append(&claim).expect("claim");
                log.compact("id", "id").expect("first compact");
            }
            {
                let log = MatchLog::open(&path).expect("second segment");
                log.append(&MatchLogEvent::RemoveRecord {
                    side: Side::B,
                    id: "B".into(),
                })
                .expect("remove");
                log.append(&MatchLogEvent::UpsertRecord {
                    side: Side::B,
                    record: [("id".into(), "B".into())].into_iter().collect(),
                })
                .expect("recreate");
                log.compact("id", "id").expect("second compact");
            }
            {
                let log = MatchLog::open(&path).expect("third segment");
                // Introducing a delta after legacy removals must also be safe.
                log.append(&make_assignment_delta(None, "other", 1))
                    .expect("new claim");
                log.compact("id", "id").expect("third compact");
            }
            let events = MatchLog::replay(&path).expect("all segments");
            assert!(
                events
                    .iter()
                    .any(|event| matches!(event, MatchLogEvent::RemoveRecord { .. })),
                "older ownership invalidation must survive compaction"
            );
            assert_eq!(
                reconstruct_assignment_history(&events, &[]).expect("reconstruct"),
                vec![("A".into(), "other".into())],
                "neither legacy nor assignment ownership may resurrect"
            );
        }
    }

    #[test]
    fn assignment_compaction_retains_remove_recreate_membership_history() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("remove.wal");
        let log = MatchLog::open(&path).expect("open");
        for event in [
            make_assignment_delta(Some("old"), "new", 0),
            MatchLogEvent::RemoveRecord {
                side: Side::B,
                id: "new".into(),
            },
            MatchLogEvent::UpsertRecord {
                side: Side::B,
                record: [("id".into(), "new".into())].into_iter().collect(),
            },
            make_assignment_delta(None, "last", 1),
        ] {
            log.append(&event).expect("append history");
        }
        log.compact("id", "id").expect("compact");
        let events = MatchLog::replay(&path).expect("replay compacted");
        assert_eq!(
            events.len(),
            4,
            "a recreated ID's remove still changed membership"
        );
        assert_eq!(
            reconstruct_assignment_history(&events, &[]).expect("history"),
            vec![("A".into(), "last".into())],
            "compaction preserves final acceptance"
        );
    }

    #[test]
    fn assignment_roundtrip_compaction_and_fail_closed() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("assignment.wal");
        let log = MatchLog::open(&path).expect("open log");
        let event = make_assignment_delta(Some("old"), "new", 0);
        log.append(&event).expect("append");
        log.compact("id", "id").expect("compact");
        let events = MatchLog::replay(&path).expect("replay");
        assert!(
            matches!(&events[..], [MatchLogEvent::AssignmentDelta(d)] if d.version == 1 && d.replacements[0].rank.is_none()),
            "retain complete typed delta"
        );
        for line in [
            r#"{"type":"assignment_delta","version":2}"#,
            r#"{"type":"assignment_delta","version":1"#,
            r#"{"type":"future_assignment"}"#,
        ] {
            let bad = dir.path().join("bad.ndjson");
            std::fs::write(&bad, line).expect("write malformed event");
            assert!(
                MatchLog::replay(&bad).is_err(),
                "unsupported or malformed transition must fail closed: {line}"
            );
        }
        let mut delta = match event {
            MatchLogEvent::AssignmentDelta(d) => d,
            _ => unreachable!(),
        };
        delta.version = 2;
        let unsupported = dir.path().join("unsupported.ndjson");
        std::fs::write(
            &unsupported,
            serde_json::to_vec(&MatchLogEvent::AssignmentDelta(delta.clone()))
                .expect("raw future event"),
        )
        .expect("future event file");
        let error = MatchLog::replay(&unsupported).expect_err("unsupported schema");
        assert!(
            error
                .to_string()
                .contains("unsupported assignment_delta version 2"),
            "report schema version clearly: {error}"
        );
        delta.version = 1;
        delta.newly_unmatched.clear();
        assert!(
            MatchLog::serialize_event(&MatchLogEvent::AssignmentDelta(delta)).is_err(),
            "validate complete event before mutation"
        );
    }

    #[test]
    fn reassignment_roundtrip_and_compaction_order() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("reassign.wal");
        let log = MatchLog::open(&path).expect("open log");
        let confirm = MatchLogEvent::CrossMapConfirm {
            a_id: "A1".into(),
            b_id: "B1".into(),
            score: Some(0.8),
            rank: Some(1),
            reason: None,
        };
        let event = make_reassignment();
        let bytes = MatchLog::serialize_event(&event).expect("serialize");
        assert_eq!(
            bytes.iter().filter(|&&b| b == b'\n').count(),
            1,
            "one WAL frame"
        );
        let roundtrip: MatchLogEvent = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(
            serde_json::to_value(&roundtrip).expect("value"),
            serde_json::to_value(&event).expect("value"),
            "metadata roundtrip"
        );
        log.append(&confirm).expect("append confirm");
        log.append(&event).expect("append reassignment");
        log.append(&MatchLogEvent::CrossMapBreak {
            a_id: "A1".into(),
            b_id: "B2".into(),
        })
        .expect("append break");
        log.compact("id", "id").expect("compact");
        let events = MatchLog::replay(&path).expect("replay");
        assert_eq!(events.len(), 3, "whole crossmap history retained");
        assert!(
            matches!(events[0], MatchLogEvent::CrossMapConfirm { .. }),
            "confirm first"
        );
        assert_eq!(
            serde_json::to_value(&events[1]).expect("value"),
            serde_json::to_value(&event).expect("value"),
            "whole event retained in order"
        );
        assert!(
            matches!(events[2], MatchLogEvent::CrossMapBreak { .. }),
            "break last"
        );
    }

    #[test]
    fn truncated_reassignment_preserves_old_pair_event() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("truncated-reassign.wal");
        let old = MatchLogEvent::CrossMapConfirm {
            a_id: "A1".into(),
            b_id: "B1".into(),
            score: Some(0.8),
            rank: None,
            reason: None,
        };
        let mut bytes = MatchLog::serialize_event(&old).expect("serialize old");
        let new = MatchLog::serialize_event(&make_reassignment()).expect("serialize replacement");
        bytes.extend_from_slice(&new[..new.len() - 10]);
        fs::write(&path, bytes).expect("write crash fixture");
        let events = MatchLog::replay(&path).expect("replay");
        assert_eq!(events.len(), 1, "no half replacement survives");
        assert!(
            matches!(&events[0], MatchLogEvent::CrossMapConfirm { a_id, b_id, .. } if a_id == "A1" && b_id == "B1"),
            "old pair remains"
        );
    }

    #[test]
    fn append_and_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Append 100 UpsertRecord + 5 CrossMapConfirm
        {
            let log = MatchLog::open(&path).unwrap();
            for i in 0..100 {
                let rec = make_record(&[
                    ("entity_id", &format!("A-{}", i)),
                    ("legal_name", &format!("Company {}", i)),
                ]);
                log.append(&MatchLogEvent::UpsertRecord {
                    side: Side::A,
                    record: rec,
                })
                .unwrap();
            }
            for i in 0..5 {
                log.append(&MatchLogEvent::CrossMapConfirm {
                    a_id: format!("A-{}", i),
                    b_id: format!("B-{}", i),
                    score: Some(0.9),
                    rank: Some(1),
                    reason: None,
                })
                .unwrap();
            }
            log.flush().unwrap();
        }

        // Replay
        let events = MatchLog::replay(&path).unwrap();
        assert_eq!(events.len(), 105);

        let upserts = events
            .iter()
            .filter(|e| matches!(e, MatchLogEvent::UpsertRecord { .. }))
            .count();
        let confirms = events
            .iter()
            .filter(|e| matches!(e, MatchLogEvent::CrossMapConfirm { .. }))
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
            let log = MatchLog::open(&base_path).unwrap();
            actual_path = log.path().to_path_buf();
            for i in 0..10 {
                let rec = make_record(&[("entity_id", &format!("A-{}", i))]);
                log.append(&MatchLogEvent::UpsertRecord {
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
        let events = MatchLog::replay(&base_path).unwrap();
        assert_eq!(events.len(), 10);
    }

    #[test]
    fn compact_deduplicates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("compact.wal");

        {
            let log = MatchLog::open(&path).unwrap();

            // Write 50 records, then overwrite them (50 duplicates)
            for i in 0..50 {
                let rec = make_record(&[
                    ("entity_id", &format!("A-{}", i)),
                    ("legal_name", &format!("Company {}", i)),
                ]);
                log.append(&MatchLogEvent::UpsertRecord {
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
                log.append(&MatchLogEvent::UpsertRecord {
                    side: Side::A,
                    record: rec,
                })
                .unwrap();
            }

            // Add crossmap events
            for i in 0..3 {
                log.append(&MatchLogEvent::CrossMapConfirm {
                    a_id: format!("A-{}", i),
                    b_id: format!("B-{}", i),
                    score: Some(0.88),
                    rank: Some(1),
                    reason: None,
                })
                .unwrap();
            }
            log.append(&MatchLogEvent::CrossMapBreak {
                a_id: "A-0".into(),
                b_id: "B-0".into(),
            })
            .unwrap();

            log.compact("entity_id", "counterparty_id").unwrap();
        }

        // Replay compacted file
        let events = MatchLog::replay(&path).unwrap();

        // Should have: 50 unique upserts (latest version) + 3 confirms + 1 break = 54
        let upserts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, MatchLogEvent::UpsertRecord { .. }))
            .collect();
        assert_eq!(upserts.len(), 50, "expected 50 deduplicated upserts");

        // Verify latest version was kept
        if let MatchLogEvent::UpsertRecord { record, .. } = &upserts[0] {
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
                    MatchLogEvent::CrossMapConfirm { .. } | MatchLogEvent::CrossMapBreak { .. }
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
            MatchLog::replay(&Path::new("nonexistent_dir_for_test").join("wal.log")).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn replay_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.wal");
        File::create(&path).unwrap();
        let events = MatchLog::replay(&path).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn backwards_compat_old_events_without_rank_reason() {
        // Old WAL lines lack rank/reason fields — they should deserialize fine
        // via #[serde(default)].
        let old_confirm = r#"{"type":"crossmap_confirm","a_id":"A-1","b_id":"B-1","score":0.9}"#;
        let event: MatchLogEvent = serde_json::from_str(old_confirm).unwrap();
        if let MatchLogEvent::CrossMapConfirm {
            rank,
            reason,
            score,
            ..
        } = event
        {
            assert_eq!(rank, None);
            assert_eq!(reason, None);
            assert_eq!(score, Some(0.9));
        } else {
            panic!("wrong variant");
        }

        let old_review =
            r#"{"type":"review_match","id":"B-1","side":"b","candidate_id":"A-1","score":0.72}"#;
        let event: MatchLogEvent = serde_json::from_str(old_review).unwrap();
        if let MatchLogEvent::ReviewMatch { rank, reason, .. } = event {
            assert_eq!(rank, None);
            assert_eq!(reason, None);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn no_match_below_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nomatch.wal");

        {
            let log = MatchLog::open(&path).unwrap();
            log.append(&MatchLogEvent::NoMatchBelow {
                query_id: "B-42".into(),
                query_side: Side::B,
                best_candidate_id: Some("A-7".into()),
                best_score: Some(0.45),
            })
            .unwrap();
            log.append(&MatchLogEvent::NoMatchBelow {
                query_id: "B-99".into(),
                query_side: Side::B,
                best_candidate_id: None,
                best_score: None,
            })
            .unwrap();
            log.flush().unwrap();
        }

        let events = MatchLog::replay(&path).unwrap();
        assert_eq!(events.len(), 2);
        if let MatchLogEvent::NoMatchBelow {
            query_id,
            best_candidate_id,
            best_score,
            ..
        } = &events[0]
        {
            assert_eq!(query_id, "B-42");
            assert_eq!(best_candidate_id.as_deref(), Some("A-7"));
            assert_eq!(*best_score, Some(0.45));
        } else {
            panic!("wrong variant");
        }
        if let MatchLogEvent::NoMatchBelow {
            best_candidate_id,
            best_score,
            ..
        } = &events[1]
        {
            assert_eq!(*best_candidate_id, None);
            assert_eq!(*best_score, None);
        } else {
            panic!("wrong variant");
        }
    }
}
