//! JSONL (newline-delimited JSON) data loading.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use crate::error::DataError;
use crate::models::Record;

/// Load a JSONL file into a map of records keyed by ID, plus a sorted ID list.
///
/// Each line is a JSON object. All values are flattened to strings.
///
/// # Arguments
/// - `path` — path to the JSONL file
/// - `id_field` — JSON key to use as the record key
/// - `required_fields` — fields expected in each record (logged warning if missing)
///
/// # Returns
/// `(records, sorted_ids)` where `records` maps `id → Record` and
/// `sorted_ids` is a `Vec<String>` of IDs in sorted order.
pub fn load_jsonl(
    path: &Path,
    id_field: &str,
    required_fields: &[String],
) -> Result<(HashMap<String, Record>, Vec<String>), DataError> {
    if !path.exists() {
        return Err(DataError::NotFound {
            path: path.display().to_string(),
        });
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let path_str = path.display().to_string();

    let mut records: HashMap<String, Record> = HashMap::new();
    let mut ids: Vec<String> = Vec::new();
    let mut warned_fields: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            DataError::Parse(format!(
                "{}:{}: invalid JSON: {}",
                path_str,
                line_num + 1,
                e
            ))
        })?;

        let map = match obj {
            serde_json::Value::Object(m) => m,
            _ => {
                return Err(DataError::Parse(format!(
                    "{}:{}: expected JSON object, got {}",
                    path_str,
                    line_num + 1,
                    obj
                )));
            }
        };

        // Extract ID
        let id = match map.get(id_field) {
            Some(v) => json_value_to_string(v),
            None => {
                return Err(DataError::MissingIdField {
                    field: id_field.to_string(),
                    path: format!("{}:{}", path_str, line_num + 1),
                });
            }
        };

        let id = id.trim().to_string();
        if id.is_empty() {
            return Err(DataError::MissingIdField {
                field: id_field.to_string(),
                path: format!("{}:{}", path_str, line_num + 1),
            });
        }

        if records.contains_key(&id) {
            return Err(DataError::DuplicateId { id, path: path_str });
        }

        // Warn about missing required fields (once per field)
        for rf in required_fields {
            if !map.contains_key(rf.as_str()) && !warned_fields.contains(rf) {
                eprintln!(
                    "warning: required field {:?} not found in record at {}:{}",
                    rf,
                    path_str,
                    line_num + 1
                );
                warned_fields.insert(rf.clone());
            }
        }

        // Flatten all values to strings
        let mut record = Record::new();
        for (key, value) in &map {
            record.insert(key.clone(), json_value_to_string(value));
        }

        ids.push(id.clone());
        records.insert(id, record);
    }

    ids.sort();
    Ok((records, ids))
}

/// Convert a JSON value to a string representation.
/// - Strings → the string itself
/// - Numbers/booleans → their string form
/// - Null → empty string
/// - Arrays/objects → JSON serialization
fn json_value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        _ => v.to_string(), // arrays, objects → JSON string
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_basic_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"id": "A-1", "name": "Foo Corp", "country": "GB"}}"#).unwrap();
        writeln!(f, r#"{{"id": "A-2", "name": "Bar Ltd", "country": "DE"}}"#).unwrap();
        writeln!(f, r#"{{"id": "A-3", "name": "Baz Inc", "country": "US"}}"#).unwrap();

        let (records, ids) = load_jsonl(&path, "id", &["name".into(), "country".into()]).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(ids.len(), 3);

        let rec = records.get("A-1").unwrap();
        assert_eq!(rec.get("name").unwrap(), "Foo Corp");
        assert_eq!(rec.get("country").unwrap(), "GB");
    }

    #[test]
    fn load_nonexistent_file() {
        let err = load_jsonl(Path::new("nonexistent.jsonl"), "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::NotFound { .. }));
    }

    #[test]
    fn missing_id_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"name": "Foo"}}"#).unwrap();

        let err = load_jsonl(&path, "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::MissingIdField { .. }));
    }

    #[test]
    fn duplicate_id_detection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"id": "1", "name": "foo"}}"#).unwrap();
        writeln!(f, r#"{{"id": "1", "name": "bar"}}"#).unwrap();

        let err = load_jsonl(&path, "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::DuplicateId { .. }));
    }

    #[test]
    fn numeric_values_to_string() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nums.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"id": "1", "count": 42, "active": true, "note": null}}"#
        )
        .unwrap();

        let (records, _) = load_jsonl(&path, "id", &[]).unwrap();
        let rec = records.get("1").unwrap();
        assert_eq!(rec.get("count").unwrap(), "42");
        assert_eq!(rec.get("active").unwrap(), "true");
        assert_eq!(rec.get("note").unwrap(), "");
    }

    #[test]
    fn skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blanks.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"id": "1", "name": "foo"}}"#).unwrap();
        writeln!(f).unwrap(); // blank line
        writeln!(f, r#"{{"id": "2", "name": "bar"}}"#).unwrap();

        let (records, _) = load_jsonl(&path, "id", &[]).unwrap();
        assert_eq!(records.len(), 2);
    }
}
