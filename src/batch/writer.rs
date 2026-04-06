//! Batch output writers: results, review, and unmatched CSVs.

use std::path::Path;

use crate::config::{Config, MatchMethod};
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
        // BM25 match fields have empty field_a/field_b; use "bm25" for the
        // column header to match the FieldScore keys produced by score_pair.
        let (col_a, col_b) = if mf.method == MatchMethod::Bm25 {
            ("bm25", "bm25")
        } else {
            (mf.field_a.as_str(), mf.field_b.as_str())
        };
        headers.push(format!("{}_{}_score", col_a, col_b));
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

        // Add field scores. BM25 FieldScores use field_a="bm25", field_b="bm25"
        // (not the empty strings from the MatchField), so match on those.
        for mf in &config.match_fields {
            let (fa, fb) = if mf.method == MatchMethod::Bm25 {
                ("bm25".to_string(), "bm25".to_string())
            } else {
                (mf.field_a.clone(), mf.field_b.clone())
            };
            let fs = r
                .field_scores
                .iter()
                .find(|fs| fs.field_a == fa && fs.field_b == fb);
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
/// Columns: id_field, score (best candidate score or empty), then all other B-record fields.
/// The score column allows threshold sweeping post-hoc without re-running the match.
pub fn write_unmatched_csv(
    path: &Path,
    records: &[(String, Record, Option<f64>)],
    id_field: &str,
) -> Result<(), DataError> {
    ensure_parent(path)?;

    if records.is_empty() {
        // Write header-only file
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record([id_field, "score"])?;
        wtr.flush()?;
        return Ok(());
    }

    // Collect all unique field names across all records
    let mut all_fields: Vec<String> = Vec::new();
    // ID field first, then score, then remaining fields sorted
    all_fields.push(id_field.to_string());
    all_fields.push("score".to_string());
    let mut other_fields: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, rec, _) in records {
        for key in rec.keys() {
            if key != id_field {
                other_fields.insert(key.clone());
            }
        }
    }
    all_fields.extend(other_fields);

    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(&all_fields)?;

    for (id, rec, score) in records {
        let score_str = score.map(|s| format!("{:.4}", s)).unwrap_or_default();
        let mut row: Vec<String> = Vec::with_capacity(all_fields.len());
        for f in &all_fields {
            if f == id_field {
                row.push(id.clone());
            } else if f == "score" {
                row.push(score_str.clone());
            } else {
                row.push(rec.get(f).cloned().unwrap_or_default());
            }
        }
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<(), DataError> {
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Classification, FieldScore, Side};
    use tempfile::tempdir;

    fn make_config() -> Config {
        crate::config::load_config(Path::new("benchmarks/batch/10kx10k_flat/cold/config.yaml"))
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
            rank: Some(1),
            reason: None,
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

        let records = vec![(
            "B-1".into(),
            {
                let mut r = Record::new();
                r.insert("counterparty_id".into(), "B-1".into());
                r.insert("counterparty_name".into(), "Test Corp".into());
                r
            },
            Some(0.75_f64),
        )];

        write_unmatched_csv(&path, &records, "counterparty_id").unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("counterparty_id"));
        assert!(content.contains("B-1"));
        assert!(content.contains("Test Corp"));
    }
}
