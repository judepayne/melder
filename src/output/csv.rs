//! CSV output writers for the build pipeline.

use std::path::Path;

use crate::models::Record;

use super::build::Relationship;

/// Write relationships.csv: all confirmed matches and review-band pairs.
///
/// Columns: b_id, a_id, score, relationship_type, reason, rank, then one
/// column per A-side field (from the output manifest).
pub fn write_relationships_csv(
    path: &Path,
    relationships: &[Relationship],
    a_records: &std::collections::HashMap<String, Record>,
    a_fields: &[String],
    a_id_field: &str,
    b_id_field: &str,
) -> Result<(), csv::Error> {
    let tmp = path.with_extension("csv.tmp");
    {
        let mut wtr = csv::Writer::from_path(&tmp)?;

        // Header
        let mut header = vec![
            b_id_field.to_string(),
            a_id_field.to_string(),
            "score".to_string(),
            "relationship_type".to_string(),
            "reason".to_string(),
            "rank".to_string(),
        ];
        for f in a_fields {
            if f != a_id_field {
                header.push(f.clone());
            }
        }
        wtr.write_record(&header)?;

        // Rows — only match and review types
        for rel in relationships {
            if rel.relationship_type != "match" && rel.relationship_type != "review" {
                continue;
            }
            let mut row = vec![
                rel.b_id.clone(),
                rel.a_id.clone(),
                rel.score.map(|s| format!("{:.4}", s)).unwrap_or_default(),
                rel.relationship_type.clone(),
                rel.reason.clone().unwrap_or_default(),
                rel.rank.map(|r| r.to_string()).unwrap_or_default(),
            ];
            // Append A-side fields
            if let Some(a_rec) = a_records.get(&rel.a_id) {
                for f in a_fields {
                    if f != a_id_field {
                        row.push(a_rec.get(f).cloned().unwrap_or_default());
                    }
                }
            } else {
                for f in a_fields {
                    if f != a_id_field {
                        row.push(String::new());
                    }
                }
            }
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
    }
    crate::util::rename_replacing(&tmp, path)
        .map_err(|e| csv::Error::from(std::io::Error::other(e.to_string())))?;
    Ok(())
}

/// Write unmatched.csv: B-side records with no match or review relationship.
///
/// Columns: b_id, then B-side fields, then best_score, best_a_id.
pub fn write_unmatched_csv(
    path: &Path,
    unmatched: &[(String, Record, Option<f64>, Option<String>)],
    b_fields: &[String],
    b_id_field: &str,
) -> Result<(), csv::Error> {
    let tmp = path.with_extension("csv.tmp");
    {
        let mut wtr = csv::Writer::from_path(&tmp)?;

        let mut header = vec![b_id_field.to_string()];
        for f in b_fields {
            if f != b_id_field {
                header.push(f.clone());
            }
        }
        header.push("best_score".to_string());
        header.push("best_a_id".to_string());
        wtr.write_record(&header)?;

        for (b_id, rec, best_score, best_a_id) in unmatched {
            let mut row = vec![b_id.clone()];
            for f in b_fields {
                if f != b_id_field {
                    row.push(rec.get(f).cloned().unwrap_or_default());
                }
            }
            row.push(best_score.map(|s| format!("{:.4}", s)).unwrap_or_default());
            row.push(best_a_id.clone().unwrap_or_default());
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
    }
    crate::util::rename_replacing(&tmp, path)
        .map_err(|e| csv::Error::from(std::io::Error::other(e.to_string())))?;
    Ok(())
}

/// Write candidates.csv: scored candidates at ranks 2..N (scoring log only).
pub fn write_candidates_csv(
    path: &Path,
    candidates: &[super::build::CandidateRow],
    b_id_field: &str,
    a_id_field: &str,
) -> Result<(), csv::Error> {
    let tmp = path.with_extension("csv.tmp");
    {
        let mut wtr = csv::Writer::from_path(&tmp)?;
        wtr.write_record([b_id_field, "rank", a_id_field, "score"])?;
        for c in candidates {
            wtr.write_record([
                c.b_id.as_str(),
                &c.rank.to_string(),
                c.a_id.as_str(),
                &format!("{:.4}", c.score),
            ])?;
        }
        wtr.flush()?;
    }
    crate::util::rename_replacing(&tmp, path)
        .map_err(|e| csv::Error::from(std::io::Error::other(e.to_string())))?;
    Ok(())
}
