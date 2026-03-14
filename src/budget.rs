//! Memory budget estimation and auto-configuration decisions.
//!
//! Parses `memory_budget: auto | "24GB"` config values, estimates total
//! footprint from record count and embedding dimensions, and decides whether
//! to use SQLite for the record store and/or mmap for the vector index.

use std::io::{BufRead, BufReader};
use std::path::Path;

use sysinfo::System;

use crate::error::ConfigError;
use crate::vectordb::manifest::read_manifest;

// --- Constants ---

/// Estimated bytes per record in the in-memory DashMap store.
const BYTES_PER_RECORD: u64 = 500;

/// Bytes per vector dimension at f16/bf16 quantisation (2 bytes).
const BYTES_PER_DIM_F16: u64 = 2;

/// Bytes per vector dimension at f32 (default, no quantisation; 4 bytes).
const BYTES_PER_DIM_F32: u64 = 4;

// --- Public types ---

/// Decided storage backend configuration given a memory budget.
///
/// Returned by `decide()`. Both `use_sqlite` and `use_mmap` may be true
/// simultaneously when the dataset exceeds the budget on both dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetDecision {
    /// Use SQLite for the record store instead of the in-memory DashMap store.
    pub use_sqlite: bool,
    /// Use mmap for the vector index instead of a full in-memory load.
    pub use_mmap: bool,
    /// SQLite page cache size in bytes. Only meaningful when `use_sqlite`.
    pub sqlite_cache_bytes: u64,
}

// --- Public API ---

/// Parse a budget string into bytes.
///
/// Accepts:
/// - `"auto"` — detects available RAM at the time of the call and uses 80%
///   of it as the budget ceiling.
/// - `"24GB"`, `"8gb"`, `"512MB"`, `"256mb"`, `"1024KB"` — explicit size
///   strings. Supported units: TB, GB, MB, KB, B (case-insensitive).
///   Fractional values like `"0.5GB"` are also accepted.
pub fn parse_budget(s: &str) -> Result<u64, ConfigError> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Ok(available_ram());
    }
    parse_size_string(s)
}

/// Detect available (not total) RAM and return 80% as the usable budget.
///
/// Uses `sysinfo` to read the OS-reported available memory. Returns a
/// conservative 1 GB if detection fails or reports zero.
pub fn available_ram() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    let available = sys.available_memory();
    if available == 0 {
        // Fallback when sysinfo reports nothing useful (e.g. some CI envs).
        return 1024 * 1024 * 1024; // 1 GB
    }
    (available as f64 * 0.8) as u64
}

/// Estimate the number of records in dataset A without loading the full file.
///
/// 1. Reads `record_count` from any `*.manifest` sidecar in `cache_dir`
///    (fast — zero data file I/O; populated by prior runs).
/// 2. Falls back to counting lines in the data file: CSV subtracts one for
///    the header row; JSONL counts all lines; Parquet returns 0 (cannot count
///    without parsing the columnar format).
pub fn estimate_record_count(cache_dir: &str, data_path: &str, format: Option<&str>) -> usize {
    if let Some(count) = count_from_manifest(cache_dir) {
        return count;
    }
    count_from_file(data_path, format)
}

/// Decide which storage backends to use given a budget and workload.
///
/// Uses the 70/30 split from the scaling architecture:
/// - **Record store** (30%): if the estimated record footprint exceeds 30% of
///   the budget, use SQLite instead of the in-memory DashMap store.
/// - **Vector index** (70%): if the estimated vector footprint exceeds 70% of
///   the budget, use mmap (`index.view()`) instead of a full in-memory load.
/// - **SQLite cache**: sized at 30% of the total budget (same allocation as
///   the record store threshold, so the cache holds the working set).
///
/// When `record_count` is 0 (unknown — e.g. Parquet with no cached manifest),
/// returns `BudgetDecision { use_sqlite: false, use_mmap: false }`.
pub fn decide(
    budget_bytes: u64,
    record_count: usize,
    embedding_dim: usize,
    n_embedding_fields: usize,
    use_f16: bool,
) -> BudgetDecision {
    if record_count == 0 || budget_bytes == 0 {
        return BudgetDecision {
            use_sqlite: false,
            use_mmap: false,
            sqlite_cache_bytes: 0,
        };
    }

    let n = record_count as u64;
    let dim = embedding_dim as u64;
    // When no embedding fields are configured, there is no vector index.
    let fields = n_embedding_fields as u64;

    let bytes_per_dim = if use_f16 {
        BYTES_PER_DIM_F16
    } else {
        BYTES_PER_DIM_F32
    };

    let record_footprint = n * BYTES_PER_RECORD;
    // Zero when n_embedding_fields == 0 (no vector index to mmap).
    let vector_footprint = n * dim * bytes_per_dim * fields;

    // 30% threshold: if record store exceeds this, use SQLite.
    let record_threshold = budget_bytes * 30 / 100;
    // 70% threshold: if vector index exceeds this, use mmap.
    let vector_threshold = budget_bytes * 70 / 100;

    let use_sqlite = record_footprint > record_threshold;
    let use_mmap = vector_footprint > vector_threshold;

    // SQLite page cache: 30% of budget.
    let sqlite_cache_bytes = budget_bytes * 30 / 100;

    BudgetDecision {
        use_sqlite,
        use_mmap,
        sqlite_cache_bytes,
    }
}

// --- Internal helpers ---

/// Read record_count from the first valid manifest file found in cache_dir.
fn count_from_manifest(cache_dir: &str) -> Option<usize> {
    let dir = std::fs::read_dir(cache_dir).ok()?;
    for entry in dir.flatten() {
        let path = entry.path();
        let name = path.to_string_lossy();
        if !name.ends_with(".manifest") {
            continue;
        }
        // Strip ".manifest" to reconstruct the cache file path that
        // read_manifest() expects (it appends ".manifest" internally).
        let cache_path_str = &name[..name.len() - ".manifest".len()];
        if let Ok(Some(m)) = read_manifest(Path::new(cache_path_str))
            && m.record_count > 0
        {
            return Some(m.record_count);
        }
    }
    None
}

/// Count records in a data file by line count.
///
/// - CSV/TSV: line count minus 1 (header row).
/// - JSONL/NDJSON: line count.
/// - Parquet and unknown formats: 0 (cannot count without parsing).
fn count_from_file(data_path: &str, format: Option<&str>) -> usize {
    let fmt = format.unwrap_or_else(|| {
        Path::new(data_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
    });

    let file = match std::fs::File::open(data_path) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    let reader = BufReader::new(file);
    let lines = reader.lines().count();

    match fmt.to_lowercase().as_str() {
        "csv" | "tsv" => lines.saturating_sub(1), // subtract header row
        "jsonl" | "ndjson" => lines,
        _ => 0, // parquet and unknown: cannot determine without loading
    }
}

/// Parse a size string like "24GB", "512MB", "1024KB" into bytes.
fn parse_size_string(s: &str) -> Result<u64, ConfigError> {
    // Find where the numeric part ends (digits and decimal point).
    let digit_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let (num_str, suffix) = s.split_at(digit_end);

    if num_str.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "memory_budget".into(),
            message: format!(
                "invalid size {:?}; expected 'auto' or a size string like '24GB', '512MB'",
                s
            ),
        });
    }

    let num: f64 = num_str.parse().map_err(|_| ConfigError::InvalidValue {
        field: "memory_budget".into(),
        message: format!(
            "invalid size {:?}; expected 'auto' or a size string like '24GB', '512MB'",
            s
        ),
    })?;

    if num <= 0.0 {
        return Err(ConfigError::InvalidValue {
            field: "memory_budget".into(),
            message: "memory_budget must be > 0".into(),
        });
    }

    let suffix = suffix.trim().to_uppercase();
    let multiplier: u64 = match suffix.as_str() {
        "TB" | "T" => 1024 * 1024 * 1024 * 1024,
        "GB" | "G" => 1024 * 1024 * 1024,
        "MB" | "M" => 1024 * 1024,
        "KB" | "K" => 1024,
        "B" | "" => 1,
        _ => {
            return Err(ConfigError::InvalidValue {
                field: "memory_budget".into(),
                message: format!("unknown unit {:?}; expected TB, GB, MB, KB, or B", suffix),
            });
        }
    };

    Ok((num * multiplier as f64) as u64)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_budget_auto_returns_nonzero() {
        let bytes = parse_budget("auto").expect("auto should succeed");
        assert!(bytes > 0, "auto budget should be > 0");
    }

    #[test]
    fn parse_budget_gb() {
        assert_eq!(
            parse_budget("24GB").unwrap(),
            24 * 1024 * 1024 * 1024,
            "24GB"
        );
    }

    #[test]
    fn parse_budget_mb() {
        assert_eq!(parse_budget("512MB").unwrap(), 512 * 1024 * 1024, "512MB");
    }

    #[test]
    fn parse_budget_kb() {
        assert_eq!(parse_budget("1024KB").unwrap(), 1024 * 1024, "1024KB");
    }

    #[test]
    fn parse_budget_lowercase() {
        assert_eq!(
            parse_budget("8gb").unwrap(),
            8 * 1024 * 1024 * 1024,
            "8gb lowercase"
        );
    }

    #[test]
    fn parse_budget_fractional() {
        // 0.5 GB = 512 MB
        let bytes = parse_budget("0.5GB").unwrap();
        // Allow 1-byte rounding tolerance from f64 multiplication.
        assert!(
            (bytes as i64 - 512 * 1024 * 1024).abs() <= 1,
            "0.5GB should be ~512MB, got {}",
            bytes
        );
    }

    #[test]
    fn parse_budget_invalid_unit() {
        let err = parse_budget("8PB").unwrap_err();
        assert!(
            err.to_string().contains("unknown unit"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn parse_budget_invalid_number() {
        assert!(parse_budget("xGB").is_err(), "non-number should fail");
    }

    #[test]
    fn parse_budget_zero_rejected() {
        let err = parse_budget("0GB").unwrap_err();
        assert!(
            err.to_string().contains("must be > 0"),
            "unexpected error: {}",
            err
        );
    }

    // --- Four canonical decision scenarios ---

    #[test]
    fn decide_fits_in_memory() {
        // 1 000 records at 384-dim f32 with an 8 GB budget: both fit.
        let d = decide(8 * 1024 * 1024 * 1024, 1_000, 384, 1, false);
        assert!(!d.use_sqlite, "1k records should fit in 8 GB");
        assert!(!d.use_mmap, "small vector index should fit in 8 GB");
    }

    #[test]
    fn decide_needs_mmap_only() {
        // 800 000 records, 384-dim f32, budget = 1.5 GB:
        //   records  = 800 000 × 500 B       =  400 MB
        //   vectors  = 800 000 × 384 × 4 B   = ~1.23 GB
        //   record threshold (30%) = 450 MB → records fit   → no SQLite
        //   vector threshold (70%) = 1.05 GB → vectors don't → mmap
        let budget = 1_500 * 1024 * 1024; // 1.5 GB
        let d = decide(budget, 800_000, 384, 1, false);
        assert!(!d.use_sqlite, "800k records should fit in 30% of 1.5 GB");
        assert!(
            d.use_mmap,
            "800k × 384-dim f32 vectors should exceed 70% of 1.5 GB"
        );
    }

    #[test]
    fn decide_needs_sqlite_only() {
        // 2 000 000 records, no embedding fields, budget = 2 GB:
        //   records  = 2 000 000 × 500 B = 1 GB
        //   vectors  = 0 (n_embedding_fields = 0)
        //   record threshold (30%) = 614 MB → records don't fit → SQLite
        //   vector threshold (70%)           → no vectors       → no mmap
        let budget = 2 * 1024 * 1024 * 1024; // 2 GB
        let d = decide(budget, 2_000_000, 384, 0, false);
        assert!(d.use_sqlite, "2M records need SQLite at a 2 GB budget");
        assert!(!d.use_mmap, "no embedding fields → no vector index to mmap");
    }

    #[test]
    fn decide_needs_both() {
        // 10 000 000 records, 384-dim f32, budget = 8 GB:
        //   records = 10M × 500 B     =  5.0 GB > 2.4 GB (30%) → SQLite
        //   vectors = 10M × 384 × 4 B = 14.6 GB > 5.6 GB (70%) → mmap
        let budget = 8 * 1024 * 1024 * 1024; // 8 GB
        let d = decide(budget, 10_000_000, 384, 1, false);
        assert!(d.use_sqlite, "10M records need SQLite at an 8 GB budget");
        assert!(d.use_mmap, "10M × 384-dim f32 vectors need mmap at 8 GB");
    }

    #[test]
    fn decide_zero_records_no_action() {
        // Unknown record count → no auto-configuration.
        let d = decide(8 * 1024 * 1024 * 1024, 0, 384, 1, false);
        assert!(!d.use_sqlite, "unknown record count: no SQLite");
        assert!(!d.use_mmap, "unknown record count: no mmap");
    }

    #[test]
    fn estimate_record_count_nonexistent_dir_and_file() {
        // Both paths missing → 0, not a panic.
        let count =
            estimate_record_count("/nonexistent/cache", "/nonexistent/data.csv", Some("csv"));
        assert_eq!(count, 0, "missing cache + file should return 0");
    }

    #[test]
    fn estimate_record_count_from_csv_file() {
        // Write a tiny CSV to a tempdir and count its data rows.
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("data.csv");
        std::fs::write(&csv_path, "id,name\n1,Alice\n2,Bob\n").unwrap();
        let count = estimate_record_count(
            dir.path().to_str().unwrap(),
            csv_path.to_str().unwrap(),
            Some("csv"),
        );
        // Cache dir has no manifest → falls back to line count.
        // 3 lines - 1 header = 2 records.
        assert_eq!(count, 2, "CSV line count should give 2 records");
    }
}
