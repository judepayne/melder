//! Batch output writers: results, review, and unmatched CSVs.

use std::path::Path;

use crate::config::Config;
use crate::error::DataError;
use crate::models::{MatchResult, Record, Side};

/// Write matched results to a CSV file.
///
/// Headers: a_id, b_id, score, classification, [field_score columns]
pub fn write_results_csv(
    path: &Path,
    results: &[MatchResult],
    config: &Config,
) -> Result<(), DataError> {
    ensure_parent(path)?;
    let mut wtr = csv::Writer::from_path(path)?;

    // Build headers
    let mut headers = vec![
        "a_id".to_string(),
        "b_id".to_string(),
        "score".to_string(),
        "classification".to_string(),
    ];
    // Add field score columns
    for mf in &config.match_fields {
        headers.push(format!("{}_{}_score", mf.field_a, mf.field_b));
    }
    wtr.write_record(&headers)?;

    for r in results {
        let (a_id, b_id) = match r.query_side {
            Side::A => (&r.query_id, &r.matched_id),
            Side::B => (&r.matched_id, &r.query_id),
        };

        let mut row = vec![
            a_id.to_string(),
            b_id.to_string(),
            format!("{:.4}", r.score),
            r.classification.as_str().to_string(),
        ];

        // Add field scores
        for mf in &config.match_fields {
            let fs = r
                .field_scores
                .iter()
                .find(|fs| fs.field_a == mf.field_a && fs.field_b == mf.field_b);
            row.push(format!("{:.4}", fs.map(|f| f.score).unwrap_or(0.0)));
        }

        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}

/// Write review results to a CSV file.
///
/// Same format as results, but only review-classified entries.
pub fn write_review_csv(
    path: &Path,
    results: &[MatchResult],
    config: &Config,
) -> Result<(), DataError> {
    write_results_csv(path, results, config)
}

/// Write unmatched B records to a CSV file.
///
/// All fields from unmatched B records, prefixed by the ID field.
pub fn write_unmatched_csv(
    path: &Path,
    records: &[(String, Record)],
    id_field: &str,
) -> Result<(), DataError> {
    ensure_parent(path)?;

    if records.is_empty() {
        // Write header-only file
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record([id_field])?;
        wtr.flush()?;
        return Ok(());
    }

    // Collect all unique field names across all records
    let mut all_fields: Vec<String> = Vec::new();
    // ID field first
    all_fields.push(id_field.to_string());
    // Then remaining fields sorted
    let mut other_fields: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, rec) in records {
        for key in rec.keys() {
            if key != id_field {
                other_fields.insert(key.clone());
            }
        }
    }
    all_fields.extend(other_fields);

    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(&all_fields)?;

    for (_, rec) in records {
        let row: Vec<String> = all_fields
            .iter()
            .map(|f| rec.get(f).cloned().unwrap_or_default())
            .collect();
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<(), DataError> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Classification, FieldScore, Side};
    use tempfile::tempdir;

    fn make_config() -> Config {
        crate::config::load_config(Path::new("testdata/configs/bench1kx1k.yaml"))
            .expect("failed to load config")
    }

    #[test]
    fn write_and_read_results() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("results.csv");
        let config = make_config();

        let results = vec![MatchResult {
            query_id: "B-1".into(),
            matched_id: "A-1".into(),
            query_side: Side::B,
            score: 0.92,
            field_scores: vec![FieldScore {
                field_a: "legal_name".into(),
                field_b: "counterparty_name".into(),
                method: "embedding".into(),
                score: 0.95,
                weight: 0.55,
            }],
            classification: Classification::Auto,
            matched_record: None,
            from_crossmap: false,
        }];

        write_results_csv(&path, &results, &config).unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("a_id,b_id,score,classification"));
        assert!(content.contains("A-1,B-1,0.9200,auto"));
    }

    #[test]
    fn write_unmatched() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("unmatched.csv");

        let records = vec![("B-1".into(), {
            let mut r = Record::new();
            r.insert("counterparty_id".into(), "B-1".into());
            r.insert("counterparty_name".into(), "Test Corp".into());
            r
        })];

        write_unmatched_csv(&path, &records, "counterparty_id").unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("counterparty_id"));
        assert!(content.contains("B-1"));
        assert!(content.contains("Test Corp"));
    }
}
