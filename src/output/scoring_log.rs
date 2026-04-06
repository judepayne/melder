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
}

impl ScoringLogWriter {
    /// Returns the path to the scoring log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Shut down the writer, waiting for it to finish.
    pub fn shutdown(mut self) {
        // Dropping the sender (done by caller) closes the channel.
        // The writer thread drains remaining items and exits.
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
        path.with_extension("ndjson.zst")
    } else {
        path.with_extension("ndjson")
    };

    if let Some(parent) = actual_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&actual_path)?;

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

    let sender = ScoringLogSender { tx, failed };
    let writer = ScoringLogWriter {
        handle: Some(handle),
        path: actual_path,
    };

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
