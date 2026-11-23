//! Bounded, run-associated batch shortlist exports, independent of relationships.

use std::io::{BufRead, BufReader};
use std::path::Path;

/// Stream every listed pair and its field evidence into configured output sinks.
/// Rank is the array ordinal, including for legacy logs with saturated ranks.
pub fn export(
    log: &Path,
    csv_dir: Option<&Path>,
    parquet_dir: Option<&Path>,
    db_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(log)?;
    let input: Box<dyn std::io::Read> = if log.extension().is_some_and(|e| e == "zst") {
        Box::new(zstd::stream::read::Decoder::new(file)?)
    } else {
        Box::new(file)
    };
    let mut csv = if let Some(dir) = csv_dir {
        std::fs::create_dir_all(dir)?;
        let mut edges = csv::Writer::from_path(dir.join("candidate_edges.csv.tmp"))?;
        edges.write_record(["a_id", "b_id", "rank", "score"])?;
        let mut fields = csv::Writer::from_path(dir.join("field_scores.csv.tmp"))?;
        fields.write_record([
            "a_id", "b_id", "field_a", "field_b", "method", "score", "weight",
        ])?;
        Some((edges, fields))
    } else {
        None
    };
    let mut conn = db_path.map(rusqlite::Connection::open).transpose()?;
    let transaction = conn.as_mut().map(|c| c.transaction()).transpose()?;
    if let Some(tx) = &transaction {
        // A losing candidate need not occur in relationships. Its field evidence
        // references the shortlisted edge, not an accepted relationship.
        tx.execute_batch("DROP TABLE field_scores;
            CREATE TABLE IF NOT EXISTS candidate_edges (a_id TEXT NOT NULL,b_id TEXT NOT NULL,rank INTEGER NOT NULL,score REAL NOT NULL,PRIMARY KEY(a_id,b_id));
            DELETE FROM candidate_edges;
            CREATE INDEX IF NOT EXISTS idx_candidate_edges_query ON candidate_edges(b_id,rank);
            CREATE TABLE field_scores (a_id TEXT NOT NULL,b_id TEXT NOT NULL,field_a TEXT NOT NULL,field_b TEXT NOT NULL,method TEXT NOT NULL,score REAL NOT NULL,weight REAL NOT NULL,FOREIGN KEY(a_id,b_id) REFERENCES candidate_edges(a_id,b_id));
            CREATE INDEX idx_fscores_relationship ON field_scores(a_id,b_id);
            UPDATE metadata SET value='true' WHERE key='scoring_log_enabled';")?;
    }
    #[cfg(feature = "parquet-format")]
    let mut parquet = parquet_dir.map(ParquetSinks::new).transpose()?;
    #[cfg(not(feature = "parquet-format"))]
    if parquet_dir.is_some() {
        return Err("Parquet output requires parquet-format".into());
    }
    for line in BufReader::new(input).lines() {
        let entry: serde_json::Value = serde_json::from_str(&line?)?;
        if entry["type"] != "scored" {
            continue;
        }
        let query = entry["query_id"].as_str().ok_or("missing query_id")?;
        let side_a = entry["query_side"]
            .as_str()
            .is_some_and(|s| s.eq_ignore_ascii_case("a"));
        let candidates = entry["candidates"].as_array().ok_or("missing candidates")?;
        for (ordinal, candidate) in candidates.iter().enumerate() {
            let matched = candidate["matched_id"]
                .as_str()
                .ok_or("missing matched_id")?;
            let (a, b) = if side_a {
                (query, matched)
            } else {
                (matched, query)
            };
            let rank = i64::try_from(ordinal + 1)?;
            let score = candidate["score"].as_f64().ok_or("missing score")?;
            if let Some((edges, _)) = &mut csv {
                edges.serialize((a, b, rank, score))?;
            }
            if let Some(tx) = &transaction {
                tx.prepare_cached("INSERT INTO candidate_edges VALUES (?1,?2,?3,?4)")?
                    .execute(rusqlite::params![a, b, rank, score])?;
            }
            #[cfg(feature = "parquet-format")]
            if let Some(sinks) = &mut parquet {
                sinks
                    .edges
                    .push(vec![a.into(), b.into(), rank.into(), score.into()]);
            }
            if let Some(fields) = candidate["field_scores"].as_array() {
                for field in fields {
                    let fa = field["field_a"].as_str().ok_or("missing field_a")?;
                    let fb = field["field_b"].as_str().ok_or("missing field_b")?;
                    let method = field["method"].as_str().ok_or("missing method")?;
                    let value = field["score"].as_f64().ok_or("missing field score")?;
                    let weight = field["weight"].as_f64().ok_or("missing weight")?;
                    if let Some((_, fields)) = &mut csv {
                        fields.serialize((a, b, fa, fb, method, value, weight))?;
                    }
                    if let Some(tx) = &transaction {
                        tx.prepare_cached(
                            "INSERT INTO field_scores VALUES (?1,?2,?3,?4,?5,?6,?7)",
                        )?
                        .execute(rusqlite::params![a, b, fa, fb, method, value, weight])?;
                    }
                    #[cfg(feature = "parquet-format")]
                    if let Some(sinks) = &mut parquet {
                        sinks.fields.push(vec![
                            a.into(),
                            b.into(),
                            fa.into(),
                            fb.into(),
                            method.into(),
                            value.into(),
                            weight.into(),
                        ]);
                        sinks.flush(false)?;
                    }
                }
            }
            #[cfg(feature = "parquet-format")]
            if let Some(sinks) = &mut parquet {
                sinks.flush(false)?;
            }
        }
    }
    if let Some((edges, fields)) = &mut csv {
        edges.flush()?;
        fields.flush()?;
    }
    drop(csv);
    #[cfg(feature = "parquet-format")]
    if let Some(mut sinks) = parquet {
        sinks.flush(true)?;
        sinks.edge_writer.close()?;
        sinks.field_writer.close()?;
    }
    if let Some(tx) = transaction {
        tx.commit()?;
    }
    if let Some(dir) = csv_dir {
        for name in ["candidate_edges.csv", "field_scores.csv"] {
            std::fs::rename(dir.join(format!("{name}.tmp")), dir.join(name))?;
        }
    }
    #[cfg(feature = "parquet-format")]
    if let Some(dir) = parquet_dir {
        for name in ["candidate_edges.parquet", "field_scores.parquet"] {
            std::fs::rename(dir.join(format!("{name}.tmp")), dir.join(name))?;
        }
    }
    Ok(())
}

#[cfg(feature = "parquet-format")]
struct ParquetSinks {
    edge_writer: parquet::arrow::ArrowWriter<std::fs::File>,
    field_writer: parquet::arrow::ArrowWriter<std::fs::File>,
    edge_schema: std::sync::Arc<arrow::datatypes::Schema>,
    field_schema: std::sync::Arc<arrow::datatypes::Schema>,
    edges: Vec<Vec<serde_json::Value>>,
    fields: Vec<Vec<serde_json::Value>>,
}

#[cfg(feature = "parquet-format")]
impl ParquetSinks {
    fn new(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        std::fs::create_dir_all(dir)?;
        let edge_schema = Arc::new(Schema::new(vec![
            Field::new("a_id", DataType::Utf8, false),
            Field::new("b_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int64, false),
            Field::new("score", DataType::Float64, false),
        ]));
        let field_schema = Arc::new(Schema::new(
            [
                "a_id", "b_id", "field_a", "field_b", "method", "score", "weight",
            ]
            .iter()
            .enumerate()
            .map(|(i, n)| {
                Field::new(
                    *n,
                    if i < 5 {
                        DataType::Utf8
                    } else {
                        DataType::Float64
                    },
                    false,
                )
            })
            .collect::<Vec<_>>(),
        ));
        Ok(Self {
            edge_writer: parquet::arrow::ArrowWriter::try_new(
                std::fs::File::create(dir.join("candidate_edges.parquet.tmp"))?,
                edge_schema.clone(),
                Some(
                    parquet::file::properties::WriterProperties::builder()
                        .set_max_row_group_size(8192)
                        .build(),
                ),
            )?,
            field_writer: parquet::arrow::ArrowWriter::try_new(
                std::fs::File::create(dir.join("field_scores.parquet.tmp"))?,
                field_schema.clone(),
                Some(
                    parquet::file::properties::WriterProperties::builder()
                        .set_max_row_group_size(8192)
                        .build(),
                ),
            )?,
            edge_schema,
            field_schema,
            edges: Vec::new(),
            fields: Vec::new(),
        })
    }
    fn flush(&mut self, force: bool) -> Result<(), Box<dyn std::error::Error>> {
        use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::DataType;
        use std::sync::Arc;
        for (rows, writer, schema) in [
            (&mut self.edges, &mut self.edge_writer, &self.edge_schema),
            (&mut self.fields, &mut self.field_writer, &self.field_schema),
        ] {
            if rows.len() < 1024 && !force {
                continue;
            }
            if rows.is_empty() {
                continue;
            }
            let schema = schema.clone();
            let arrays: Vec<ArrayRef> = schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| match f.data_type() {
                    DataType::Int64 => Arc::new(Int64Array::from(
                        rows.iter().map(|r| r[i].as_i64()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    DataType::Float64 => Arc::new(Float64Array::from(
                        rows.iter().map(|r| r[i].as_f64()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    _ => Arc::new(StringArray::from(
                        rows.iter().map(|r| r[i].as_str()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                })
                .collect();
            writer.write(&arrow::record_batch::RecordBatch::try_new(schema, arrays)?)?;
            rows.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::export;

    #[test]
    fn all_ranks_and_fields_stream_without_selected_duplicates()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let log = dir.path().join("scores.ndjson");
        let db = dir.path().join("output.db");
        let conn = rusqlite::Connection::open(&db)?;
        conn.execute_batch("CREATE TABLE field_scores(a_id TEXT,b_id TEXT,field_a TEXT,field_b TEXT,method TEXT,score REAL,weight REAL); CREATE TABLE metadata(key TEXT,value TEXT);")?;
        let candidates: Vec<_> = (0..300).map(|i| serde_json::json!({"matched_id":format!("a{i}"),"rank":255,"score":0.9,"field_scores":[{"field_a":"name","field_b":"name","method":"exact","score":1.0,"weight":1.0}]})).collect();
        std::fs::write(
            &log,
            serde_json::to_vec(
                &serde_json::json!({"type":"scored","query_id":"b","query_side":"b","candidates":candidates}),
            )?,
        )?;
        export(&log, Some(dir.path()), None, Some(&db))?;
        let counts: (i64, i64) =
            conn.query_row("SELECT count(*),max(rank) FROM candidate_edges", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        assert_eq!(
            counts,
            (300, 300),
            "every ordinal must survive saturated legacy ranks"
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM field_scores", [], |r| r
                .get::<_, i64>(0))?,
            300,
            "all listed pairs need field evidence"
        );
        assert_eq!(
            csv::Reader::from_path(dir.path().join("candidate_edges.csv"))?
                .records()
                .count(),
            300,
            "selected rows must not be duplicated"
        );
        std::fs::write(&log, "{\"type\":\"header\"}\n")?;
        export(&log, Some(dir.path()), None, Some(&db))?;
        assert_eq!(
            conn.query_row("SELECT count(*) FROM candidate_edges", [], |r| r
                .get::<_, i64>(0))?,
            0,
            "enabled empty evidence must clear prior rows"
        );
        assert_eq!(
            csv::Reader::from_path(dir.path().join("candidate_edges.csv"))?
                .records()
                .count(),
            0,
            "empty output retains only its header"
        );
        Ok(())
    }
}
