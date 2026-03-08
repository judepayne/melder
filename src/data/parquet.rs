//! Parquet data loading (feature-gated behind `parquet-format`).

use std::collections::HashMap;
use std::path::Path;

use arrow::array::{Array, AsArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::DataError;
use crate::models::Record;

/// Load a Parquet file into a map of records keyed by ID, plus a sorted ID list.
///
/// All column values are converted to strings. Null values become empty strings.
///
/// # Arguments
/// - `path` — path to the Parquet file
/// - `id_field` — column name to use as the record key
/// - `required_fields` — fields expected in the schema (logged warning if missing)
///
/// # Returns
/// `(records, sorted_ids)` where `records` maps `id → Record` and
/// `sorted_ids` is a `Vec<String>` of IDs in sorted order.
pub fn load_parquet(
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
    let path_str = path.display().to_string();

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::Parse(format!("{}: failed to read parquet: {}", path_str, e)))?;

    let schema = builder.schema().clone();
    let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    // Validate id_field is present
    if !column_names.contains(&id_field.to_string()) {
        return Err(DataError::MissingIdField {
            field: id_field.to_string(),
            path: path_str,
        });
    }

    // Warn about missing required fields
    for rf in required_fields {
        if !column_names.contains(rf) {
            eprintln!(
                "warning: required field {:?} not found in parquet schema of {}",
                rf, path_str
            );
        }
    }

    let id_col_idx = column_names.iter().position(|n| n == id_field).unwrap();

    let reader = builder.build().map_err(|e| {
        DataError::Parse(format!(
            "{}: failed to build parquet reader: {}",
            path_str, e
        ))
    })?;

    let mut records: HashMap<String, Record> = HashMap::new();
    let mut ids: Vec<String> = Vec::new();

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| DataError::Parse(format!("{}: error reading batch: {}", path_str, e)))?;

        let num_rows = batch.num_rows();
        let num_cols = batch.num_columns();

        for row_idx in 0..num_rows {
            // Extract all column values as strings
            let mut record = Record::new();
            let mut id_value = String::new();

            for col_idx in 0..num_cols {
                let col = batch.column(col_idx);
                let col_name = &column_names[col_idx];
                let value = column_value_to_string(col, row_idx);

                if col_idx == id_col_idx {
                    id_value = value.clone();
                }
                record.insert(col_name.clone(), value);
            }

            let id = id_value.trim().to_string();
            if id.is_empty() {
                return Err(DataError::MissingIdField {
                    field: id_field.to_string(),
                    path: format!("{}:row {}", path_str, row_idx),
                });
            }

            if records.contains_key(&id) {
                return Err(DataError::DuplicateId { id, path: path_str });
            }

            ids.push(id.clone());
            records.insert(id, record);
        }
    }

    ids.sort();
    Ok((records, ids))
}

/// Convert a single cell value from an Arrow array to a String.
fn column_value_to_string(col: &dyn Array, row: usize) -> String {
    if col.is_null(row) {
        return String::new();
    }

    match col.data_type() {
        DataType::Utf8 => col.as_string::<i32>().value(row).to_string(),
        DataType::LargeUtf8 => col.as_string::<i64>().value(row).to_string(),
        DataType::Int8 => col
            .as_primitive::<arrow::datatypes::Int8Type>()
            .value(row)
            .to_string(),
        DataType::Int16 => col
            .as_primitive::<arrow::datatypes::Int16Type>()
            .value(row)
            .to_string(),
        DataType::Int32 => col
            .as_primitive::<arrow::datatypes::Int32Type>()
            .value(row)
            .to_string(),
        DataType::Int64 => col
            .as_primitive::<arrow::datatypes::Int64Type>()
            .value(row)
            .to_string(),
        DataType::UInt8 => col
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(row)
            .to_string(),
        DataType::UInt16 => col
            .as_primitive::<arrow::datatypes::UInt16Type>()
            .value(row)
            .to_string(),
        DataType::UInt32 => col
            .as_primitive::<arrow::datatypes::UInt32Type>()
            .value(row)
            .to_string(),
        DataType::UInt64 => col
            .as_primitive::<arrow::datatypes::UInt64Type>()
            .value(row)
            .to_string(),
        DataType::Float32 => col
            .as_primitive::<arrow::datatypes::Float32Type>()
            .value(row)
            .to_string(),
        DataType::Float64 => col
            .as_primitive::<arrow::datatypes::Float64Type>()
            .value(row)
            .to_string(),
        DataType::Boolean => if col.as_boolean().value(row) {
            "true"
        } else {
            "false"
        }
        .to_string(),
        _ => {
            // Fallback: use the debug/display representation from arrow
            let formatter = arrow::util::display::ArrayFormatter::try_new(col, &Default::default());
            match formatter {
                Ok(fmt) => fmt.value(row).to_string(),
                Err(_) => String::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_dataset_a_parquet() {
        let (records, ids) = load_parquet(
            Path::new("testdata/dataset_a_10000.parquet"),
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

        assert_eq!(records.len(), 10000);
        assert_eq!(ids.len(), 10000);

        // IDs should be sorted
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);

        // Spot check a record has expected fields
        let first = records.get(&ids[0]).unwrap();
        assert!(first.contains_key("entity_id"));
        assert!(first.contains_key("legal_name"));
        assert!(first.contains_key("short_name"));
        assert!(first.contains_key("country_code"));
        assert!(first.contains_key("lei"));
        // Numeric fields should be converted to strings
        assert!(first.contains_key("num_employees"));
        let num_emp = first.get("num_employees").unwrap();
        assert!(!num_emp.is_empty(), "num_employees should be non-empty");
        // Should parse as a number
        num_emp
            .parse::<i64>()
            .expect("num_employees should be a valid integer string");
    }

    #[test]
    fn load_dataset_b_parquet() {
        let (records, ids) = load_parquet(
            Path::new("testdata/dataset_b_10000.parquet"),
            "counterparty_id",
            &[
                "counterparty_id".into(),
                "counterparty_name".into(),
                "domicile".into(),
            ],
        )
        .unwrap();

        assert_eq!(records.len(), 10000);
        assert_eq!(ids.len(), 10000);

        let first = records.get(&ids[0]).unwrap();
        assert!(first.contains_key("counterparty_id"));
        assert!(first.contains_key("counterparty_name"));
        assert!(first.contains_key("domicile"));
    }

    #[test]
    fn parquet_csv_parity() {
        // Load both formats and verify they produce the same records
        let (csv_records, csv_ids) =
            crate::data::load_csv(Path::new("testdata/dataset_a_10000.csv"), "entity_id", &[])
                .unwrap();

        let (pq_records, pq_ids) = load_parquet(
            Path::new("testdata/dataset_a_10000.parquet"),
            "entity_id",
            &[],
        )
        .unwrap();

        assert_eq!(csv_records.len(), pq_records.len());
        assert_eq!(csv_ids.len(), pq_ids.len());
        assert_eq!(csv_ids, pq_ids);

        // Check a sample of records have matching string fields
        for id in csv_ids.iter().take(100) {
            let csv_rec = csv_records.get(id).unwrap();
            let pq_rec = pq_records.get(id).unwrap();

            // String fields should match exactly
            for field in &[
                "entity_id",
                "legal_name",
                "short_name",
                "country_code",
                "lei",
            ] {
                assert_eq!(
                    csv_rec.get(*field),
                    pq_rec.get(*field),
                    "field {} mismatch for record {}",
                    field,
                    id
                );
            }
        }
    }

    #[test]
    fn load_nonexistent_parquet() {
        let err = load_parquet(Path::new("nonexistent.parquet"), "id", &[]).unwrap_err();
        assert!(matches!(err, DataError::NotFound { .. }));
    }

    #[test]
    fn missing_id_field_parquet() {
        let err = load_parquet(
            Path::new("testdata/dataset_a_10000.parquet"),
            "nonexistent_field",
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, DataError::MissingIdField { .. }));
    }
}
