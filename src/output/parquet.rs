//! Parquet output writers for the build pipeline (feature-gated behind `parquet-format`).

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, StringBuilder, UInt8Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use crate::models::Record;

use super::build::{CandidateRow, FieldScoreRow, Relationship};

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
    let names = super::columns::relationships(a_fields, a_id_field, b_id_field);
    let mut fields = vec![
        Field::new(&names[0], DataType::Utf8, false),
        Field::new(&names[1], DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
        Field::new("relationship_type", DataType::Utf8, false),
        Field::new("reason", DataType::Utf8, true),
        Field::new("rank", DataType::UInt8, true),
    ];
    let extra_a_fields: Vec<&String> = a_fields
        .iter()
        .filter(|f| f.as_str() != a_id_field)
        .collect();
    for name in &names[6..] {
        fields.push(Field::new(name, DataType::Utf8, true));
    }
    let schema = Arc::new(Schema::new(fields));

    // Build column arrays.
    let filtered: Vec<&Relationship> = relationships
        .iter()
        .filter(|r| {
            matches!(
                r.relationship_type.as_str(),
                "match" | "review" | "enroll_edge"
            )
        })
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
    let names = super::columns::unmatched(b_fields, b_id_field);
    let mut fields = vec![Field::new(&names[0], DataType::Utf8, false)];
    let extra_b_fields: Vec<&String> = b_fields
        .iter()
        .filter(|f| f.as_str() != b_id_field)
        .collect();
    for name in &names[1..names.len() - 2] {
        fields.push(Field::new(name, DataType::Utf8, true));
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
    let names = super::columns::candidates(b_id_field, a_id_field);
    let schema = Arc::new(Schema::new(vec![
        Field::new(&names[0], DataType::Utf8, false),
        Field::new("rank", DataType::UInt8, false),
        Field::new(&names[2], DataType::Utf8, false),
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

/// Write field_scores.parquet with SQLite column order and Float64 numeric columns.
/// Each bounded batch is flushed as a row group to limit additional write memory.
pub fn write_field_scores_parquet(
    path: &Path,
    rows: &[FieldScoreRow],
) -> Result<(), Box<dyn std::error::Error>> {
    const BATCH_SIZE: usize = 4096;
    let schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Utf8, false),
        Field::new("b_id", DataType::Utf8, false),
        Field::new("field_a", DataType::Utf8, false),
        Field::new("field_b", DataType::Utf8, false),
        Field::new("method", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    for chunk in rows.chunks(BATCH_SIZE) {
        let mut strings: [StringBuilder; 5] = std::array::from_fn(|_| StringBuilder::new());
        let mut scores = Float64Builder::new();
        let mut weights = Float64Builder::new();
        for row in chunk {
            for (builder, value) in strings.iter_mut().zip([
                &row.a_id,
                &row.b_id,
                &row.field_a,
                &row.field_b,
                &row.method,
            ]) {
                builder.append_value(value);
            }
            scores.append_value(row.score);
            weights.append_value(row.weight);
        }
        let mut columns: Vec<Arc<dyn arrow::array::Array>> = strings
            .iter_mut()
            .map(|builder| Arc::new(builder.finish()) as Arc<dyn arrow::array::Array>)
            .collect();
        columns.push(Arc::new(scores.finish()));
        columns.push(Arc::new(weights.finish()));
        writer.write(&RecordBatch::try_new(schema.clone(), columns)?)?;
        writer.flush()?;
    }
    writer.close()?;
    crate::util::rename_replacing(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;

    #[test]
    fn field_scores_roundtrip_batched_and_empty() {
        use arrow::array::{Float64Array, StringArray};

        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("field_scores.parquet");
        let rows: Vec<_> = (0..8200)
            .map(|i| FieldScoreRow {
                a_id: format!("A{i}"),
                b_id: "B".into(),
                field_a: "name".into(),
                field_b: "title".into(),
                method: "exact".into(),
                score: 0.12345678901234567,
                weight: 1.2345678901234567,
            })
            .collect();
        for data in [&rows[..], &[][..]] {
            write_field_scores_parquet(&path, data).expect("write field scores");
            let builder = ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(&path).expect("open parquet"),
            )
            .expect("reader builder");
            let schema = builder.schema();
            assert_eq!(
                schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>(),
                [
                    "a_id", "b_id", "field_a", "field_b", "method", "score", "weight"
                ],
                "SQLite column order"
            );
            for (index, field) in schema.fields().iter().enumerate() {
                assert_eq!(
                    field.data_type(),
                    if index < 5 {
                        &DataType::Utf8
                    } else {
                        &DataType::Float64
                    },
                    "typed column"
                );
                assert!(!field.is_nullable(), "required column");
            }
            assert_eq!(
                builder.metadata().num_row_groups(),
                data.len().div_ceil(4096),
                "bounded row groups"
            );
            let mut count = 0;
            for batch in builder.build().expect("reader") {
                let batch = batch.expect("batch");
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("IDs");
                let scores = batch
                    .column(5)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("scores");
                let weights = batch
                    .column(6)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("weights");
                for i in 0..batch.num_rows() {
                    assert_eq!(ids.value(i), data[count].a_id, "row order");
                    assert_eq!(scores.value(i), data[count].score, "full score precision");
                    assert_eq!(
                        weights.value(i),
                        data[count].weight,
                        "full weight precision"
                    );
                    count += 1;
                }
            }
            assert_eq!(count, data.len(), "all rows preserved");
        }
    }

    #[test]
    fn collision_free_outputs_roundtrip() {
        let dir = tempfile::tempdir().expect("temporary output directory");
        let fields = ["id", "score", "target_score", "best_score", "best_a_id"].map(String::from);
        let record: Record = fields
            .iter()
            .map(|f| (f.clone(), format!("payload:{f}")))
            .collect();
        let records = [("A".into(), record.clone())].into_iter().collect();
        let rels = [Relationship {
            a_id: "A".into(),
            b_id: "B".into(),
            score: Some(0.75),
            rank: Some(1),
            relationship_type: "match".into(),
            reason: None,
        }];
        let unmatched = [("B".into(), record, Some(0.5), Some("A".into()))];
        let candidates = [CandidateRow {
            b_id: "B".into(),
            rank: 2,
            a_id: "A".into(),
            score: 0.5,
        }];
        for id in ["id", "entity_id", "score"] {
            write_relationships_parquet(
                &dir.path().join("relationships.parquet"),
                &rels,
                &records,
                &fields,
                id,
                id,
            )
            .expect("write relationships");
            super::super::csv::write_relationships_csv(
                &dir.path().join("relationships.csv"),
                &rels,
                &records,
                &fields,
                id,
                id,
            )
            .expect("write CSV relationships");
            write_unmatched_parquet(
                &dir.path().join("unmatched.parquet"),
                &unmatched,
                &fields,
                id,
            )
            .expect("write unmatched");
            super::super::csv::write_unmatched_csv(
                &dir.path().join("unmatched.csv"),
                &unmatched,
                &fields,
                id,
            )
            .expect("write CSV unmatched");
            write_candidates_parquet(&dir.path().join("candidates.parquet"), &candidates, id, id)
                .expect("write candidates");
            super::super::csv::write_candidates_csv(
                &dir.path().join("candidates.csv"),
                &candidates,
                id,
                id,
            )
            .expect("write CSV candidates");
            for name in ["relationships", "unmatched", "candidates"] {
                let file = std::fs::File::open(dir.path().join(format!("{name}.parquet")))
                    .expect("open parquet");
                let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
                    .expect("parquet reader")
                    .build()
                    .expect("build reader");
                let batch = reader.next().expect("one batch").expect("read batch");
                let schema = batch.schema();
                let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                assert_eq!(
                    names.iter().collect::<std::collections::HashSet<_>>().len(),
                    names.len(),
                    "unique {name} names"
                );
                let mut csv = csv::Reader::from_path(dir.path().join(format!("{name}.csv")))
                    .expect("CSV reader");
                assert_eq!(
                    csv.headers()
                        .expect("CSV headers")
                        .iter()
                        .collect::<Vec<_>>(),
                    names,
                    "format naming parity"
                );
                assert_eq!(batch.num_rows(), 1, "one {name} row");
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .expect("string IDs");
                assert_eq!(ids.value(0), "B", "source ID preserved");
                if name != "candidates" {
                    let start = if name == "relationships" { 6 } else { 1 };
                    for (index, field) in fields.iter().filter(|f| f.as_str() != id).enumerate() {
                        let values = batch
                            .column(start + index)
                            .as_any()
                            .downcast_ref::<arrow::array::StringArray>()
                            .expect("string payload");
                        assert_eq!(
                            values.value(0),
                            format!("payload:{field}"),
                            "original field lookup survives aliasing"
                        );
                    }
                }
            }
            // Opt-in interoperability check, requiring Python with pyarrow installed.
            if std::env::var_os("MELDER_TEST_PYARROW").is_some() {
                let result = std::process::Command::new("python3").arg("-c").arg(
                    "import sys, pyarrow.parquet as pq; from pathlib import Path\nfor p in Path(sys.argv[1]).glob('*.parquet'):\n t = pq.read_table(p)\n assert len(t.column_names) == len(set(t.column_names))\n assert t.column(0).to_pylist() == ['B']"
                ).arg(dir.path()).output().expect("run pyarrow");
                assert!(
                    result.status.success(),
                    "ordinary pyarrow.read_table failed: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }
}
