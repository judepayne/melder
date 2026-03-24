pub mod csv;
pub mod jsonl;
#[cfg(feature = "parquet-format")]
pub mod parquet;

pub use self::csv::{load_csv, stream_csv};
pub use self::jsonl::{load_jsonl, stream_jsonl};
#[cfg(feature = "parquet-format")]
pub use self::parquet::{load_parquet, stream_parquet};

use std::collections::HashMap;
use std::path::Path;

use crate::error::DataError;
use crate::models::Record;

/// Load a dataset by format, dispatching to the appropriate loader.
///
/// If `format` is `None`, the format is inferred from the file extension:
/// - `.csv` → CSV
/// - `.jsonl`, `.ndjson` → JSONL
/// - `.parquet` → Parquet (requires `parquet-format` feature)
///
/// Returns `(records, sorted_ids)`.
pub fn load_dataset(
    path: &Path,
    id_field: &str,
    required_fields: &[String],
    format: Option<&str>,
) -> Result<(HashMap<String, Record>, Vec<String>), DataError> {
    let fmt = match format {
        Some(f) => f.to_lowercase(),
        None => infer_format(path),
    };

    match fmt.as_str() {
        "csv" => load_csv(path, id_field, required_fields),
        "jsonl" | "ndjson" => load_jsonl(path, id_field, required_fields),
        #[cfg(feature = "parquet-format")]
        "parquet" => load_parquet(path, id_field, required_fields),
        #[cfg(not(feature = "parquet-format"))]
        "parquet" => Err(DataError::Parse(format!(
            "parquet format requires the 'parquet-format' feature: cargo build --features parquet-format (file: {})",
            path.display()
        ))),
        other => Err(DataError::Parse(format!(
            "unsupported data format {:?} for {}",
            other,
            path.display()
        ))),
    }
}

/// Stream a dataset in chunks, dispatching to the appropriate format loader.
///
/// Calls `callback` with each chunk of `(id, record)` pairs. Returns total
/// record count. Uses the same format inference as `load_dataset()`.
pub fn stream_dataset(
    path: &Path,
    id_field: &str,
    required_fields: &[String],
    format: Option<&str>,
    chunk_size: usize,
    callback: &mut dyn FnMut(Vec<(String, Record)>),
) -> Result<usize, DataError> {
    let fmt = match format {
        Some(f) => f.to_lowercase(),
        None => infer_format(path),
    };

    match fmt.as_str() {
        "csv" => stream_csv(path, id_field, required_fields, chunk_size, callback),
        "jsonl" | "ndjson" => stream_jsonl(path, id_field, required_fields, chunk_size, callback),
        #[cfg(feature = "parquet-format")]
        "parquet" => stream_parquet(path, id_field, required_fields, chunk_size, callback),
        #[cfg(not(feature = "parquet-format"))]
        "parquet" => Err(DataError::Parse(format!(
            "parquet format requires the 'parquet-format' feature: cargo build --features parquet-format (file: {})",
            path.display()
        ))),
        other => Err(DataError::Parse(format!(
            "unsupported data format {:?} for {}",
            other,
            path.display()
        ))),
    }
}

/// Count rows in a dataset without loading records into memory.
///
/// For CSV files, counts data rows (excludes header). For other formats,
/// falls back to `load_dataset` and returns the count.
pub fn count_rows(path: &Path, format: Option<&str>) -> Result<usize, DataError> {
    let fmt = match format {
        Some(f) => f.to_lowercase(),
        None => infer_format(path),
    };

    match fmt.as_str() {
        "csv" => {
            let mut rdr = ::csv::ReaderBuilder::new()
                .has_headers(true)
                .from_path(path)?;
            Ok(rdr.records().count())
        }
        "jsonl" | "ndjson" => {
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            Ok(reader
                .lines()
                .filter(|l| l.as_ref().is_ok_and(|s| !s.trim().is_empty()))
                .count())
        }
        _ => Err(DataError::Parse(format!(
            "count_rows not supported for format {:?}; use load_dataset instead",
            fmt,
        ))),
    }
}

/// Infer data format from file extension.
fn infer_format(path: &Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("csv") => "csv".into(),
        Some("jsonl") | Some("ndjson") => "jsonl".into(),
        Some("parquet") => "parquet".into(),
        _ => "csv".into(), // default
    }
}
