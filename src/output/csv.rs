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
        let header = super::columns::relationships(a_fields, a_id_field, b_id_field);
        wtr.write_record(&header)?;

        // Rows — only match and review types
        for rel in relationships {
            if rel.relationship_type != "match"
                && rel.relationship_type != "review"
                && rel.relationship_type != "enroll_edge"
            {
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

        let header = super::columns::unmatched(b_fields, b_id_field);
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
        wtr.write_record(super::columns::candidates(b_id_field, a_id_field))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_headers_preserve_values_with_shared_or_reserved_ids() {
        let dir = tempfile::tempdir().expect("temporary output directory");
        for (b_id, a_id) in [("id", "id"), ("entity_id", "entity_id"), ("rank", "score")] {
            let path = dir.path().join("candidates.csv");
            write_candidates_csv(
                &path,
                &[super::super::build::CandidateRow {
                    b_id: "B".into(),
                    rank: 2,
                    a_id: "A".into(),
                    score: 0.75,
                }],
                b_id,
                a_id,
            )
            .expect("write candidates");
            let mut reader = csv::Reader::from_path(path).expect("read candidates");
            assert_eq!(
                reader.headers().expect("header").iter().collect::<Vec<_>>(),
                super::super::columns::candidates(b_id, a_id),
                "shared naming policy"
            );
            assert_eq!(
                reader
                    .records()
                    .next()
                    .expect("one row")
                    .expect("valid row")
                    .iter()
                    .collect::<Vec<_>>(),
                ["B", "2", "A", "0.7500"],
                "column values retain their positions"
            );
        }
    }
}
