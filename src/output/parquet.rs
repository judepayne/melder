//! Parquet output writers for the build pipeline (feature-gated behind `parquet-format`).

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, StringBuilder, UInt8Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use crate::models::Record;

use super::build::{CandidateRow, Relationship};

/// Write relationships.parquet: all confirmed matches and review-band pairs.
pub fn write_relationships_parquet(
    path: &Path,
    relationships: &[Relationship],
    a_records: &std::collections::HashMap<String, Record>,
    a_fields: &[String],
    a_id_field: &str,
    b_id_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build schema: b_id, a_id, score, relationship_type, reason, rank, then A-side fields.
    let mut fields = vec![
        Field::new(b_id_field, DataType::Utf8, false),
        Field::new(a_id_field, DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
        Field::new("relationship_type", DataType::Utf8, false),
        Field::new("reason", DataType::Utf8, true),
        Field::new("rank", DataType::UInt8, true),
    ];
    let extra_a_fields: Vec<&String> = a_fields
        .iter()
        .filter(|f| f.as_str() != a_id_field)
        .collect();
    for f in &extra_a_fields {
        fields.push(Field::new(f.as_str(), DataType::Utf8, true));
    }
    let schema = Arc::new(Schema::new(fields));

    // Build column arrays.
    let filtered: Vec<&Relationship> = relationships
        .iter()
        .filter(|r| r.relationship_type == "match" || r.relationship_type == "review")
        .collect();

    let mut b_ids = StringBuilder::new();
    let mut a_ids = StringBuilder::new();
    let mut scores = Float64Builder::new();
    let mut rel_types = StringBuilder::new();
    let mut reasons = StringBuilder::new();
    let mut ranks = UInt8Builder::new();
    let mut a_field_builders: Vec<StringBuilder> = extra_a_fields
        .iter()
        .map(|_| StringBuilder::new())
        .collect();

    for rel in &filtered {
        b_ids.append_value(&rel.b_id);
        a_ids.append_value(&rel.a_id);
        match rel.score {
            Some(s) => scores.append_value(s),
            None => scores.append_null(),
        }
        rel_types.append_value(&rel.relationship_type);
        match &rel.reason {
            Some(r) => reasons.append_value(r),
            None => reasons.append_null(),
        }
        match rel.rank {
            Some(r) => ranks.append_value(r),
            None => ranks.append_null(),
        }
        let a_rec = a_records.get(&rel.a_id);
        for (i, f) in extra_a_fields.iter().enumerate() {
            match a_rec.and_then(|r| r.get(f.as_str())) {
                Some(v) => a_field_builders[i].append_value(v),
                None => a_field_builders[i].append_null(),
            }
        }
    }

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(b_ids.finish()),
        Arc::new(a_ids.finish()),
        Arc::new(scores.finish()),
        Arc::new(rel_types.finish()),
        Arc::new(reasons.finish()),
        Arc::new(ranks.finish()),
    ];
    for builder in &mut a_field_builders {
        columns.push(Arc::new(builder.finish()));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    crate::util::rename_replacing(&tmp, path)?;
    Ok(())
}

/// Write unmatched.parquet: B-side records with no match or review relationship.
pub fn write_unmatched_parquet(
    path: &Path,
    unmatched: &[(String, Record, Option<f64>, Option<String>)],
    b_fields: &[String],
    b_id_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut fields = vec![Field::new(b_id_field, DataType::Utf8, false)];
    let extra_b_fields: Vec<&String> = b_fields
        .iter()
        .filter(|f| f.as_str() != b_id_field)
        .collect();
    for f in &extra_b_fields {
        fields.push(Field::new(f.as_str(), DataType::Utf8, true));
    }
    fields.push(Field::new("best_score", DataType::Float64, true));
    fields.push(Field::new("best_a_id", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(fields));

    let mut b_ids = StringBuilder::new();
    let mut b_field_builders: Vec<StringBuilder> = extra_b_fields
        .iter()
        .map(|_| StringBuilder::new())
        .collect();
    let mut best_scores = Float64Builder::new();
    let mut best_a_ids = StringBuilder::new();

    for (b_id, rec, best_score, best_a_id) in unmatched {
        b_ids.append_value(b_id);
        for (i, f) in extra_b_fields.iter().enumerate() {
            match rec.get(f.as_str()) {
                Some(v) => b_field_builders[i].append_value(v),
                None => b_field_builders[i].append_null(),
            }
        }
        match best_score {
            Some(s) => best_scores.append_value(*s),
            None => best_scores.append_null(),
        }
        match best_a_id {
            Some(id) => best_a_ids.append_value(id),
            None => best_a_ids.append_null(),
        }
    }

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![Arc::new(b_ids.finish())];
    for builder in &mut b_field_builders {
        columns.push(Arc::new(builder.finish()));
    }
    columns.push(Arc::new(best_scores.finish()));
    columns.push(Arc::new(best_a_ids.finish()));

    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    crate::util::rename_replacing(&tmp, path)?;
    Ok(())
}

/// Write candidates.parquet: scored candidates at ranks 2..N (scoring log only).
pub fn write_candidates_parquet(
    path: &Path,
    candidates: &[CandidateRow],
    b_id_field: &str,
    a_id_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(b_id_field, DataType::Utf8, false),
        Field::new("rank", DataType::UInt8, false),
        Field::new(a_id_field, DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let mut b_ids = StringBuilder::new();
    let mut ranks = UInt8Builder::new();
    let mut a_ids = StringBuilder::new();
    let mut scores = Float64Builder::new();

    for c in candidates {
        b_ids.append_value(&c.b_id);
        ranks.append_value(c.rank);
        a_ids.append_value(&c.a_id);
        scores.append_value(c.score);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(b_ids.finish()),
            Arc::new(ranks.finish()),
            Arc::new(a_ids.finish()),
            Arc::new(scores.finish()),
        ],
    )?;

    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    crate::util::rename_replacing(&tmp, path)?;
    Ok(())
}
