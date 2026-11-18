//! Scoring log: opt-in enrichment layer recording every scored record's
//! full top_n candidate set with per-field breakdowns.
//!
//! Format: append-only ndjson, optionally zstd-compressed.
//! First line: self-describing header. Subsequent lines: scored records.
//!
//! Writer pattern: producers serialize to `Vec<u8>` on their own thread,
//! send pre-serialized bytes through a bounded channel, and a single writer
//! thread drains the channel to disk.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;
use tracing::{info, warn};

use crate::models::{FieldScore, MatchResult, Side};

/// Header written as the first line of the scoring log.
#[derive(Serialize)]
struct Header {
    #[serde(rename = "type")]
    ty: &'static str,
    schema: u8,
    mode: String,
    job: String,
    model: String,
    top_n: usize,
    thresholds: ThresholdsSnapshot,
    a_fields: Vec<String>,
    b_fields: Vec<String>,
}

#[derive(Serialize)]
struct ThresholdsSnapshot {
    auto_match: f64,
    review_floor: f64,
    min_score_gap: Option<f64>,
}

/// A single scored record entry in the scoring log.
#[derive(Serialize)]
struct ScoredRecord {
    #[serde(rename = "type")]
    ty: &'static str,
    query_id: String,
    query_side: Side,
    outcome: String,
    reason: Option<String>,
    candidates: Vec<CandidateEntry>,
}

#[derive(Serialize)]
struct CandidateEntry {
    rank: u8,
    matched_id: String,
    score: f64,
    field_scores: Vec<FieldScoreEntry>,
}

#[derive(Serialize)]
struct FieldScoreEntry {
    field_a: String,
    field_b: String,
    method: String,
    score: f64,
    weight: f64,
}

impl From<&FieldScore> for FieldScoreEntry {
    fn from(fs: &FieldScore) -> Self {
        Self {
            field_a: fs.field_a.clone(),
            field_b: fs.field_b.clone(),
            method: fs.method.clone(),
            score: fs.score,
            weight: fs.weight,
        }
    }
}

/// Handle for sending scored records to the scoring log writer.
pub struct ScoringLogSender {
    tx: crossbeam_channel::Sender<Vec<u8>>,
    failed: Arc<AtomicBool>,
}

impl ScoringLogSender {
    /// Send a scored record's results to the scoring log.
    ///
    /// Serializes on the calling thread, sends pre-serialized bytes
    /// through the channel. Blocks if the channel is full.
    pub fn send(&self, query_id: &str, query_side: Side, results: &[MatchResult]) {
        if self.failed.load(Ordering::Relaxed) {
            return; // Writer is dead, don't block
        }
        if results.is_empty() {
            return;
        }

        let outcome = results[0].classification.as_str().to_string();
        let reason = results[0].reason.clone();

        let candidates: Vec<CandidateEntry> = results
            .iter()
            .enumerate()
            .map(|(i, mr)| CandidateEntry {
                rank: (i + 1).min(255) as u8,
                matched_id: mr.matched_id.clone(),
                score: mr.score,
                field_scores: mr.field_scores.iter().map(FieldScoreEntry::from).collect(),
            })
            .collect();

        let entry = ScoredRecord {
            ty: "scored",
            query_id: query_id.to_string(),
            query_side,
            outcome,
            reason,
            candidates,
        };

        match serde_json::to_vec(&entry) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                // Blocking send — scoring log is canonical data.
                if self.tx.send(bytes).is_err() {
                    self.failed.store(true, Ordering::Relaxed);
                }
            }
            Err(e) => {
                self.failed.store(true, Ordering::Relaxed);
                warn!(error = %e, "scoring log serialization failed");
            }
        }
    }

    /// Check if the writer has failed.
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }
}

/// Writer thread handle. Drop to shut down the writer.
pub struct ScoringLogWriter {
    handle: Option<std::thread::JoinHandle<()>>,
    path: PathBuf,
    failed: Arc<AtomicBool>,
    publication: Option<PathBuf>,
}

impl ScoringLogWriter {
    /// Returns the path to the scoring log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Shut down the writer, waiting for it to finish.
    pub fn shutdown(self) {
        if let Err(e) = self.shutdown_checked() {
            warn!(error = %e, "scoring log shutdown failed");
        }
    }

    /// Flush the log and publish graph evidence only after successful completion.
    /// All senders must be dropped before calling this method.
    pub fn shutdown_checked(mut self) -> std::io::Result<()> {
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            self.failed.store(true, Ordering::Relaxed);
        }
        if self.failed.load(Ordering::Relaxed) {
            return Err(std::io::Error::other("scoring log evidence writer failed"));
        }
        if let Some(pending) = self.publication.take() {
            // Hard-link publication is atomic and refuses to overwrite evidence.
            std::fs::hard_link(&pending, &self.path)?;
            std::fs::remove_file(pending)?;
        }
        Ok(())
    }
}

impl Drop for ScoringLogWriter {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Open a scoring log writer.
///
/// Returns a sender (for producers) and a writer (owns the thread).
/// The sender can be cloned and shared across threads.
///
/// `use_zstd`: whether to compress output with zstd.
pub fn open_scoring_log(
    path: &Path,
    use_zstd: bool,
    header: &crate::output::manifest::OutputManifest,
) -> std::io::Result<(ScoringLogSender, ScoringLogWriter)> {
    let actual_path = if use_zstd {
        PathBuf::from(format!("{}.ndjson.zst", path.display()))
    } else {
        PathBuf::from(format!("{}.ndjson", path.display()))
    };

    if let Some(parent) = actual_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    open_writer(actual_path, use_zstd, header, None)
}

fn graph_scoring_base(match_log_path: &Path) -> PathBuf {
    // Never share the WAL prefix: legacy segment discovery matches by prefix.
    let stem = match_log_path.file_stem().unwrap_or_default();
    let mut name = std::ffi::OsString::from("scores.");
    name.push(stem);
    name.push(".scoring_log");
    match_log_path.with_file_name(name)
}

fn scoring_path(base: &Path, use_zstd: bool) -> PathBuf {
    let mut name = base.as_os_str().to_os_string();
    name.push(if use_zstd { ".ndjson.zst" } else { ".ndjson" });
    PathBuf::from(name)
}

/// Discover only the scoring evidence belonging to this exact graph run.
/// Compression is inferred from the companion, never from current config.
/// Missing evidence means membership-only output; two companions are ambiguous.
pub fn graph_scoring_path(match_log_path: &Path) -> std::io::Result<Option<PathBuf>> {
    let base = graph_scoring_base(match_log_path);
    let mut found = None;
    for compressed in [false, true] {
        let path = scoring_path(&base, compressed);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if !metadata.file_type().is_file() {
                    return Err(std::io::Error::other(format!(
                        "graph scoring companion is not a regular file: {}",
                        path.display()
                    )));
                }
                if found.is_some() {
                    return Err(std::io::Error::other("ambiguous graph scoring companions"));
                }
                found = Some(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(found)
}

/// Open exclusive, run-associated graph evidence. Incomplete writes remain
/// unpublished and cannot enrich historical output after a crash or failure.
pub fn open_graph_scoring_log(
    match_log_path: &Path,
    use_zstd: bool,
    header: &crate::output::manifest::OutputManifest,
) -> std::io::Result<(ScoringLogSender, ScoringLogWriter)> {
    if graph_scoring_path(match_log_path)?.is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "graph scoring companion already exists",
        ));
    }
    let base = graph_scoring_base(match_log_path);
    let pending = base.with_extension("scoring_log.pending");
    open_writer(
        scoring_path(&base, use_zstd),
        use_zstd,
        header,
        Some(pending),
    )
}

fn open_writer(
    actual_path: PathBuf,
    use_zstd: bool,
    header: &crate::output::manifest::OutputManifest,
    publication: Option<PathBuf>,
) -> std::io::Result<(ScoringLogSender, ScoringLogWriter)> {
    let file = if let Some(pending) = &publication {
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(pending)?
    } else {
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&actual_path)?
    };

    // Serialize header
    let hdr = Header {
        ty: "header",
        schema: 1,
        mode: format!("{:?}", header.mode),
        job: header.job_name.clone(),
        model: header.model.clone(),
        top_n: header.top_n,
        thresholds: ThresholdsSnapshot {
            auto_match: header.auto_match,
            review_floor: header.review_floor,
            min_score_gap: header.min_score_gap,
        },
        a_fields: header.a_fields.clone(),
        b_fields: header.b_fields.clone(),
    };
    let mut header_bytes = serde_json::to_vec(&hdr)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    header_bytes.push(b'\n');

    let (tx, rx) = crossbeam_channel::bounded::<Vec<u8>>(10_000);
    let failed = Arc::new(AtomicBool::new(false));
    let failed_clone = failed.clone();
    let writer_path = actual_path.clone();

    let handle = std::thread::Builder::new()
        .name("scoring-log-writer".to_string())
        .spawn(move || {
            let result = if use_zstd {
                run_writer_zstd(file, &header_bytes, &rx)
            } else {
                run_writer_plain(file, &header_bytes, &rx)
            };
            if let Err(e) = result {
                failed_clone.store(true, Ordering::Relaxed);
                tracing::error!(
                    error = %e,
                    path = %writer_path.display(),
                    "scoring log writer failed"
                );
            }
        })?;

    info!(path = %actual_path.display(), zstd = use_zstd, "scoring log opened");

    let writer = ScoringLogWriter {
        handle: Some(handle),
        path: actual_path,
        failed: failed.clone(),
        publication,
    };
    let sender = ScoringLogSender { tx, failed };

    Ok((sender, writer))
}

fn run_writer_plain(
    file: std::fs::File,
    header_bytes: &[u8],
    rx: &crossbeam_channel::Receiver<Vec<u8>>,
) -> std::io::Result<()> {
    let mut w = BufWriter::with_capacity(1024 * 1024, file);
    w.write_all(header_bytes)?;
    for bytes in rx {
        w.write_all(&bytes)?;
    }
    w.flush()?;
    Ok(())
}

fn run_writer_zstd(
    file: std::fs::File,
    header_bytes: &[u8],
    rx: &crossbeam_channel::Receiver<Vec<u8>>,
) -> std::io::Result<()> {
    let encoder = zstd::Encoder::new(file, 3)?;
    let mut w = BufWriter::with_capacity(1024 * 1024, encoder);
    w.write_all(header_bytes)?;
    for bytes in rx {
        w.write_all(&bytes)?;
    }
    w.flush()?;
    let encoder = w
        .into_inner()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    encoder.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manifest() -> crate::output::manifest::OutputManifest {
        crate::output::manifest::OutputManifest {
            mode: crate::config::Mode::Match,
            job_name: "job".into(),
            a_id_field: "aid".into(),
            b_id_field: "bid".into(),
            a_fields: vec![],
            b_fields: vec![],
            auto_match: 0.9,
            review_floor: 0.5,
            min_score_gap: None,
            top_n: 5,
            model: "none".into(),
        }
    }

    #[test]
    fn graph_companions_are_run_specific_and_compression_independent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = dir
            .path()
            .join("job.graph.match_log_20260401T120000Z_000001.ndjson");
        let second = dir
            .path()
            .join("job.graph.match_log_20260401T120000Z_000002.ndjson");
        std::fs::write(dir.path().join("job.scoring_log.ndjson"), "stale").expect("stale log");
        assert!(
            graph_scoring_path(&first).expect("discovery").is_none(),
            "ignore job-wide evidence"
        );
        for (run, compressed) in [(&first, false), (&second, true)] {
            let (sender, writer) =
                open_graph_scoring_log(run, compressed, &make_manifest()).expect("open run");
            let expected = writer.path().to_path_buf();
            assert!(
                graph_scoring_path(run)
                    .expect("pending discovery")
                    .is_none(),
                "do not expose incomplete evidence"
            );
            drop(sender);
            writer.shutdown_checked().expect("flush and publish");
            assert_eq!(
                graph_scoring_path(run).expect("discovery"),
                Some(expected.clone()),
                "discover exact run regardless of compression"
            );
            let bytes = std::fs::read(expected).expect("evidence");
            let decoded = if compressed {
                zstd::decode_all(bytes.as_slice()).expect("zstd")
            } else {
                bytes
            };
            assert!(
                decoded.starts_with(b"{\"type\":\"header\""),
                "complete readable header"
            );
            assert!(
                open_graph_scoring_log(run, !compressed, &make_manifest()).is_err(),
                "refuse replacement with other compression"
            );
        }
        assert_ne!(
            graph_scoring_path(&first).expect("first"),
            graph_scoring_path(&second).expect("second"),
            "same-second sequence stays in companion identity"
        );
    }

    #[test]
    fn graph_discovery_rejects_ambiguous_or_nonregular_companions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run = dir
            .path()
            .join("job.graph.match_log_20260401T120000Z.ndjson");
        let base = graph_scoring_base(&run);
        let plain = scoring_path(&base, false);
        let compressed = scoring_path(&base, true);
        std::fs::write(&plain, "plain").expect("plain");
        std::fs::write(&compressed, "compressed").expect("compressed");
        assert!(
            graph_scoring_path(&run).is_err(),
            "ambiguous evidence must not be selected"
        );
        std::fs::remove_file(&compressed).expect("remove compressed");
        std::fs::remove_file(&plain).expect("remove plain");
        std::fs::create_dir(&plain).expect("directory");
        assert!(
            graph_scoring_path(&run).is_err(),
            "directory is not evidence"
        );
    }

    #[test]
    fn graph_failed_or_abandoned_evidence_is_never_published() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run = dir
            .path()
            .join("job.graph.match_log_20260401T120000Z_000001.ndjson");
        let (sender, writer) = open_graph_scoring_log(&run, false, &make_manifest()).expect("open");
        sender.failed.store(true, Ordering::Relaxed);
        drop(sender);
        assert!(
            writer.shutdown_checked().is_err(),
            "writer failure must propagate"
        );
        assert!(
            graph_scoring_path(&run).expect("discovery").is_none(),
            "failed evidence remains unavailable"
        );
        assert!(
            open_graph_scoring_log(&run, true, &make_manifest()).is_err(),
            "pending reservation prevents compression collision"
        );

        let abandoned = dir
            .path()
            .join("job.graph.match_log_20260401T120000Z_000002.ndjson");
        let (sender, writer) =
            open_graph_scoring_log(&abandoned, false, &make_manifest()).expect("open abandoned");
        drop(sender);
        drop(writer);
        assert!(
            graph_scoring_path(&abandoned).expect("discovery").is_none(),
            "drop is not successful graph publication"
        );
    }

    #[test]
    fn greedy_scoring_log_keeps_job_wide_truncation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("job.scoring_log");
        let path = scoring_path(&base, false);
        std::fs::write(&path, "stale evidence").expect("old log");
        let (sender, writer) =
            open_scoring_log(&base, false, &make_manifest()).expect("greedy open");
        drop(sender);
        writer.shutdown();
        assert!(
            std::fs::read_to_string(path)
                .expect("new log")
                .starts_with("{\"type\":\"header\""),
            "greedy still replaces job-wide evidence"
        );
    }
}
