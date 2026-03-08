//! CSV data loading.

use std::collections::HashMap;
use std::path::Path;

use crate::error::DataError;
use crate::models::Record;

/// Load a CSV file into a map of records keyed by ID, plus a sorted ID list.
///
/// # Arguments
/// - `path` — path to the CSV file
/// - `id_field` — column name to use as the record key
/// - `required_fields` — fields expected in headers (logged warning if missing)
///
/// # Returns
/// `(records, sorted_ids)` where `records` maps `id → Record` and
/// `sorted_ids` is a `Vec<String>` of IDs in sorted order.
pub fn load_csv(
    path: &Path,
    id_field: &str,
    required_fields: &[String],
) -> Result<(HashMap<String, Record>, Vec<String>), DataError> {
    if !path.exists() {
        return Err(DataError::NotFound {
            path: path.display().to_string(),
        });
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::Headers)
        .from_path(path)?;

    // Get headers
    let headers: Vec<String> = rdr
        .headers()?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    // Validate id_field is present
    let path_str = path.display().to_string();
    if !headers.contains(&id_field.to_string()) {
        return Err(DataError::MissingIdField {
            field: id_field.to_string(),
            path: path_str,
        });
    }

    // Warn about missing required fields (don't error)
    for rf in required_fields {
        if !headers.contains(rf) {
            eprintln!(
                "warning: required field {:?} not found in headers of {}",
                rf, path_str
            );
        }
    }

    let id_idx = headers.iter().position(|h| h == id_field).unwrap();

    let mut records: HashMap<String, Record> = HashMap::new();
    let mut ids: Vec<String> = Vec::new();

    for result in rdr.records() {
        let row = result?;
        let id = row.get(id_idx).unwrap_or("").trim().to_string();

        if records.contains_key(&id) {
            return Err(DataError::DuplicateId { id, path: path_str });
        }

        let mut record = Record::new();
        for (i, value) in row.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                record.insert(header.clone(), value.to_string());
            }
        }

        ids.push(id.clone());
        records.insert(id, record);
    }

    ids.sort();

    Ok((records, ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_dataset_a_1000() {
        let (records, ids) = load_csv(
            Path::new("testdata/dataset_a_1000.csv"),
            "entity_id",
            &[
                "entity_id".into(),
                "legal_name".into(),
                "short_name".into(),
                "country_code".into(),
                "lei".into(),
            ],
        )
        .unwrap();

        assert_eq!(records.len(), 1000);
        assert_eq!(ids.len(), 1000);

        // IDs should be sorted
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);

        // First record should have expected fields
        let first = records.get(&ids[0]).unwrap();
        assert!(first.contains_key("entity_id"));
        assert!(first.contains_key("legal_name"));
        assert!(first.contains_key("short_name"));
        assert!(first.contains_key("country_code"));
        assert!(first.contains_key("lei"));
    }

    #[test]
    fn load_nonexistent_file() {
        let err = load_csv(Path::new("nonexistent.csv"), "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::NotFound { .. }));
    }

    #[test]
    fn missing_id_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "name,value").unwrap();
        writeln!(f, "foo,1").unwrap();

        let err = load_csv(&path, "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::MissingIdField { .. }));
    }

    #[test]
    fn duplicate_id_detection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "id,name").unwrap();
        writeln!(f, "1,foo").unwrap();
        writeln!(f, "1,bar").unwrap();

        let err = load_csv(&path, "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::DuplicateId { .. }));
    }
}
