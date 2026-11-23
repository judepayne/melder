//! End-to-end graph assignment through real CLI subprocesses with an offline stub encoder.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult = Result<(), Box<dyn Error>>;
type Pairs = BTreeSet<(String, String)>;

struct Fixture {
    dir: TempDir,
    config: Value,
}

impl Fixture {
    fn new(sqlite: bool, assignment: Value) -> Result<Self, Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("out"))?;
        fs::write(
            root.join("a.csv"),
            "aid,name,text\na1,aaaa,shared\na2,aabb,shared\n",
        )?;
        fs::write(
            root.join("b.csv"),
            "bid,name,text\nb1,aaab,shared\nb2,aaaa,shared\n",
        )?;
        let stub = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stub_encoder.py");
        let encoder = format!("python3 '{}' --vector-dim 16", stub.display());
        let mut config = json!({
            "job": {"name": "graph_cli"},
            "datasets": {
                "a": {"path": root.join("a.csv"), "id_field": "aid"},
                "b": {"path": root.join("b.csv"), "id_field": "bid"}
            },
            "cross_map": {"backend": "local", "path": root.join("crossmap.csv"),
                          "a_id_field": "aid", "b_id_field": "bid"},
            "performance": {"encoder_pool_size": 1},
            "embeddings": {"remote_encoder_cmd": encoder,
                           "a_cache_dir": root.join("cache-a"), "b_cache_dir": root.join("cache-b")},
            "vector_backend": "flat",
            "exact_prefilter": {"enabled": false},
            "top_n": 2,
            "bm25_candidates": 2,
            "match_fields": [
                {"method": "bm25", "weight": 0.1,
                 "fields": [{"field_a": "text", "field_b": "text"}]},
                {"method": "fuzzy", "scorer": "ratio", "weight": 0.9,
                 "field_a": "name", "field_b": "name"}
            ],
            "thresholds": {"auto_match": 0.5, "review_floor": 0.2},
            "output": {"csv_dir_path": root.join("out"), "db_path": root.join("output.db")},
            "scoring_log": {"enabled": true, "compression": "none"},
            "batch": {}
        });
        if cfg!(feature = "parquet-format") {
            config["output"]["parquet_dir_path"] = json!(root.join("parquet"));
        }
        if !assignment.is_null() {
            config["batch"]["assignment"] = assignment;
        }
        if sqlite {
            config["batch"]["db_path"] = json!(root.join("storage.db"));
            config["batch"]["sqlite_read_pool_size"] = json!(2);
        }
        Ok(Self { dir, config })
    }

    fn path(&self, path: &str) -> PathBuf {
        self.dir.path().join(path)
    }

    fn command(&self, subcommand: &str, threads: usize) -> Result<Command, Box<dyn Error>> {
        let config_path = self.path("config.yaml");
        fs::write(&config_path, serde_yaml::to_string(&self.config)?)?;
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_meld"));
        cmd.current_dir(self.dir.path())
            .env("RAYON_NUM_THREADS", threads.to_string())
            .arg(subcommand)
            .arg("--config")
            .arg(config_path);
        Ok(cmd)
    }

    fn run(&self, threads: usize) -> Result<Output, Box<dyn Error>> {
        let output = self.command("run", threads)?.output()?;
        assert_success(&output);
        Ok(output)
    }

    fn export(&mut self, out: &str) -> Result<Output, Box<dyn Error>> {
        // Export's public interface is live-only: point it at the batch WAL,
        // not at batch.db_path (the SQLite ingestion DB is ephemeral).
        self.config["live"] = json!({"match_log_path": self.log_path()?});
        Ok(self
            .command("export", 1)?
            .arg("--out-dir")
            .arg(self.path(out))
            .output()?)
    }

    fn pairs(&self) -> Result<Pairs, Box<dyn Error>> {
        csv_pairs(&self.path("crossmap.csv"), false)
    }

    fn log_path(&self) -> Result<PathBuf, Box<dyn Error>> {
        let base = if self.config["batch"]["assignment"]["strategy"] == "greedy" {
            "out/graph_cli.match_log.ndjson"
        } else {
            "out/graph_cli.graph.match_log.ndjson"
        };
        let files = melder::state::match_log::MatchLog::find_wal_files(&self.path(base));
        assert_eq!(files.len(), 1, "one CLI run must produce one WAL segment");
        Ok(files[0].clone())
    }

    fn events(&self) -> Result<Vec<Value>, Box<dyn Error>> {
        Ok(fs::read_to_string(self.log_path()?)?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?)
    }
}

#[test]
fn sqlite_scratch_guard_rejects_reuse_without_changing_original_files() -> TestResult {
    for target in [
        "a.csv",
        "b.csv",
        "crossmap.csv",
        "exclusions.csv",
        "synonyms.csv",
        "config.yaml",
        "output.db",
        "live.db",
        "unrelated.bin",
        "storage.db-wal",
        "storage.db-shm",
        "storage.db-journal",
    ] {
        for command in ["validate", "run", "dry-run"] {
            let mut fixture = Fixture::new(true, Value::Null)?;
            fixture.config["exclusions"] = json!({"path": fixture.path("exclusions.csv"), "a_id_field":"aid", "b_id_field":"bid"});
            fixture.config["synonym_dictionary"] = json!({"path":fixture.path("synonyms.csv")});
            fixture.config["live"] = json!({"db_path":fixture.path("live.db")});
            if !target.starts_with("storage.db-") {
                fixture.config["batch"]["db_path"] = json!(fixture.path(target));
            }
            if !["config.yaml", "a.csv", "b.csv"].contains(&target) {
                fs::write(fixture.path(target), b"original bytes must survive")?;
            }
            let mut cmd = fixture.command(
                if command == "validate" {
                    "validate"
                } else {
                    "run"
                },
                1,
            )?;
            if command == "dry-run" {
                cmd.arg("--dry-run");
            }
            let before = fs::read(fixture.path(target))?;
            let a = fs::read(fixture.path("a.csv"))?;
            let b = fs::read(fixture.path("b.csv"))?;
            let output = cmd.output()?;
            assert!(!output.status.success(), "reject {target} for {command}");
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("batch.db_path"),
                "scratch diagnostic: {:?}",
                output
            );
            assert_eq!(
                fs::read(fixture.path(target))?,
                before,
                "preserve {target} for {command}"
            );
            assert_eq!(fs::read(fixture.path("a.csv"))?, a, "preserve A");
            assert_eq!(fs::read(fixture.path("b.csv"))?, b, "preserve B");
        }
    }
    Ok(())
}

#[test]
fn sqlite_scratch_guard_rejects_missing_output_aliases_and_sidecar_collisions() -> TestResult {
    for target in [
        "out/relationships.csv",
        "out/UNMATCHED.CSV",
        "OUT/RELATIONSHIPS.csv.TMP",
        "parquet/CANDIDATES.parquet.TMP",
        "OUTPUT.DB-WAL",
        "LIVE.DB-ShM",
        "out/unmatched.csv",
        "out/candidates.csv",
        "out/relationships.csv.tmp",
        "out/unmatched.csv.tmp",
        "out/candidates.csv.tmp",
        "parquet/relationships.parquet",
        "parquet/unmatched.parquet",
        "parquet/candidates.parquet",
        "parquet/relationships.parquet.tmp",
        "parquet/unmatched.parquet.tmp",
        "parquet/candidates.parquet.tmp",
        "output.db",
        "output.db-wal",
        "output.db-shm",
        "output.db-journal",
        "live.db",
        "live.db-wal",
        "live.db-shm",
        "live.db-journal",
    ] {
        for command in ["validate", "run", "dry-run"] {
            let mut fixture = Fixture::new(true, Value::Null)?;
            fixture.config["batch"]["db_path"] = json!(format!("./{target}"));
            fixture.config["output"]["parquet_dir_path"] = json!(fixture.path("parquet"));
            fixture.config["live"] = json!({"db_path":fixture.path("live.db")});
            let mut cmd = fixture.command(
                if command == "validate" {
                    "validate"
                } else {
                    "run"
                },
                1,
            )?;
            if command == "dry-run" {
                cmd.arg("--dry-run");
            }
            let output = cmd.output()?;
            assert!(
                !output.status.success(),
                "reject missing collision {target} for {command}"
            );
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("batch.db_path"),
                "scratch diagnostic: {:?}",
                output
            );
            assert!(!fixture.path(target).exists(), "must not create {target}");
        }
    }
    // A scratch sidecar must not be an output database either.
    let mut fixture = Fixture::new(true, Value::Null)?;
    fixture.config["output"]["db_path"] = json!(fixture.path("storage.db-wal"));
    let output = fixture.command("validate", 1)?.output()?;
    assert!(
        !output.status.success(),
        "protect output against scratch sidecar"
    );
    // Without a CSV sink, cwd CSV names are no longer reserved.
    fixture.config["output"] = json!({"db_path":fixture.path("output.db")});
    fixture.config["batch"]["db_path"] = json!("relationships.csv.tmp");
    let output = fixture.command("validate", 1)?.output()?;
    assert!(
        output.status.success(),
        "unconfigured CSV paths are not output targets: {:?}",
        output
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn sqlite_scratch_guard_rejects_symlinks_and_hardlinks() -> TestResult {
    for alias in ["symlink", "dangling", "hardlink", "parent"] {
        let mut fixture = Fixture::new(true, Value::Null)?;
        let original = fs::read(fixture.path("a.csv"))?;
        match alias {
            "symlink" => std::os::unix::fs::symlink(fixture.path("a.csv"), fixture.path("alias"))?,
            "dangling" => {
                std::os::unix::fs::symlink(fixture.path("absent"), fixture.path("alias"))?
            }
            "hardlink" => fs::hard_link(fixture.path("a.csv"), fixture.path("alias"))?,
            _ => std::os::unix::fs::symlink(fixture.path("out"), fixture.path("alias"))?,
        }
        fixture.config["batch"]["db_path"] = json!(if alias == "parent" {
            "alias/relationships.csv.tmp"
        } else {
            "alias"
        });
        for command in ["validate", "run", "dry-run"] {
            let mut cmd = fixture.command(
                if command == "validate" {
                    "validate"
                } else {
                    "run"
                },
                1,
            )?;
            if command == "dry-run" {
                cmd.arg("--dry-run");
            }
            assert!(
                !cmd.output()?.status.success(),
                "reject {alias} for {command}"
            );
            assert_eq!(
                fs::read(fixture.path("a.csv"))?,
                original,
                "preserve symlink/hardlink target"
            );
            assert!(
                fs::symlink_metadata(fixture.path("alias")).is_ok(),
                "preserve alias"
            );
        }
    }
    Ok(())
}

#[test]
fn sqlite_rejects_embedding_fields_but_accepts_unused_model_and_fresh_scratch() -> TestResult {
    let mut fixture = Fixture::new(true, Value::Null)?;
    fixture.config["match_fields"] =
        json!([{"field_a":"name", "field_b":"name", "method":"embedding", "weight":1.0}]);
    for command in ["validate", "run", "dry-run"] {
        let mut cmd = fixture.command(
            if command == "validate" {
                "validate"
            } else {
                "run"
            },
            1,
        )?;
        if command == "dry-run" {
            cmd.arg("--dry-run");
        }
        let output = cmd.output()?;
        assert!(
            !output.status.success(),
            "reject unsupported embedding for {command}"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("embedding match fields"),
            "embedding diagnostic: {:?}",
            output
        );
        assert!(
            !fixture.path("storage.db").exists(),
            "reject before creating scratch"
        );
    }
    fixture.config["match_fields"] =
        json!([{"field_a":"name", "field_b":"name", "method":"exact", "weight":1.0}]);
    fixture.config["embeddings"]
        .as_object_mut()
        .ok_or("embeddings object")?
        .remove("remote_encoder_cmd");
    fixture.config["embeddings"]["model"] = json!("unused-do-not-download");
    assert_success(&fixture.command("validate", 1)?.output()?);
    let a = fs::read(fixture.path("a.csv"))?;
    let b = fs::read(fixture.path("b.csv"))?;
    assert_success(&fixture.command("run", 1)?.arg("--dry-run").output()?);
    assert!(
        !fixture.path("storage.db").exists(),
        "dry-run removes its own fresh scratch"
    );
    fixture.run(1)?;
    assert!(
        !fixture.path("storage.db").exists(),
        "run removes its own fresh scratch"
    );
    assert_eq!(
        fs::read(fixture.path("a.csv"))?,
        a,
        "preserve A after successful runs"
    );
    assert_eq!(
        fs::read(fixture.path("b.csv"))?,
        b,
        "preserve B after successful runs"
    );
    // Merely selecting a model, or embedding fields without SQLite batch,
    // must remain valid for the live/memory configuration path.
    fixture.config["batch"]
        .as_object_mut()
        .ok_or("batch object")?
        .remove("db_path");
    fixture.config["match_fields"][0]["method"] = json!("embedding");
    assert_success(&fixture.command("validate", 1)?.output()?);
    Ok(())
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "CLI failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn expected(pairs: &[(&str, &str)]) -> Pairs {
    pairs
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn csv_rows(path: &Path) -> Result<Vec<BTreeMap<String, String>>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    Ok(reader
        .records()
        .map(|row| {
            let row = row?;
            let mut values: BTreeMap<String, String> = headers
                .iter()
                .zip(row.iter())
                .map(|(k, v)| (k.into(), v.into()))
                .collect();
            // SQLite's legacy batch writer uses generic IDs/classification.
            // Compare semantics without imposing an unrelated CSV schema change.
            for (old, new) in [("a_id", "aid"), ("b_id", "bid")] {
                if let Some(value) = values.remove(old) {
                    values.insert(new.into(), value);
                }
            }
            if let Some(classification) = values.remove("classification") {
                values.insert(
                    "relationship_type".into(),
                    if classification == "auto" {
                        "match".into()
                    } else {
                        classification
                    },
                );
            }
            Ok(values)
        })
        .collect::<Result<_, csv::Error>>()?)
}

fn csv_pairs(path: &Path, relationships: bool) -> Result<Pairs, Box<dyn Error>> {
    Ok(csv_rows(path)?
        .into_iter()
        .filter(|row| !relationships || row["relationship_type"] == "match")
        .map(|row| (row["aid"].clone(), row["bid"].clone()))
        .collect())
}

fn db_pairs(path: &Path) -> Result<Pairs, Box<dyn Error>> {
    assert!(
        path.exists(),
        "requested SQLite output was not written: {}",
        path.display()
    );
    let conn = rusqlite::Connection::open(path)?;
    let mut stmt =
        conn.prepare("SELECT a_id, b_id FROM relationships WHERE relationship_type = 'match'")?;
    Ok(stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?)
}

#[cfg(feature = "parquet-format")]
fn parquet_pairs(path: &Path) -> Result<Pairs, Box<dyn Error>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;

    let reader = SerializedFileReader::new(fs::File::open(path)?)?;
    let mut pairs = Pairs::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        if row.get_string(3)? == "match" {
            pairs.insert((row.get_string(1)?.clone(), row.get_string(0)?.clone()));
        }
    }
    Ok(pairs)
}

fn assert_outputs(fixture: &Fixture, pairs: &Pairs) -> TestResult {
    assert_eq!(
        &fixture.pairs()?,
        pairs,
        "saved CrossMap must be final membership"
    );
    assert_eq!(
        &csv_pairs(&fixture.path("out/relationships.csv"), true)?,
        pairs,
        "candidate Auto classifications must not masquerade as selected pairs"
    );
    assert_eq!(
        &db_pairs(&fixture.path("output.db"))?,
        pairs,
        "SQLite output must agree with final CrossMap"
    );
    #[cfg(feature = "parquet-format")]
    assert_eq!(
        &parquet_pairs(&fixture.path("parquet/relationships.parquet"))?,
        pairs,
        "Parquet selected pairs must equal final CrossMap, not scoring-log candidates"
    );
    Ok(())
}

#[test]
fn batch_configured_db_sink_without_implicit_csv_for_both_strategies_and_backends() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["graph", "greedy"] {
            let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
            fixture.config["output"] = json!({"db_path": fixture.path("output.db")});
            fs::write(fixture.path("b.csv"), "bid,name,text\nb1,aaaa,shared\n")?;
            fixture.run(1)?;
            assert_eq!(
                db_pairs(&fixture.path("output.db"))?,
                fixture.pairs()?,
                "DB membership for {sqlite}/{strategy}"
            );
            for file in [
                "relationships.csv",
                "unmatched.csv",
                "candidates.csv",
                "out/relationships.csv",
            ] {
                assert!(
                    !fixture.path(file).exists(),
                    "no unconfigured CSV: {file} ({sqlite}/{strategy})"
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parquet-format")]
#[test]
fn batch_parquet_only_for_both_strategies_and_backends() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["graph", "greedy"] {
            let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
            fixture.config["output"] = json!({"parquet_dir_path": fixture.path("parquet")});
            fs::write(fixture.path("b.csv"), "bid,name,text\nb1,aaaa,shared\n")?;
            fixture.run(1)?;
            assert_eq!(
                parquet_pairs(&fixture.path("parquet/relationships.parquet"))?,
                fixture.pairs()?,
                "Parquet membership for {sqlite}/{strategy}"
            );
            assert!(
                fixture.path("parquet/unmatched.parquet").exists(),
                "requested unmatched Parquet"
            );
            assert!(!fixture.path("output.db").exists(), "no unconfigured DB");
            for file in [
                "relationships.csv",
                "unmatched.csv",
                "candidates.csv",
                "out/relationships.csv",
            ] {
                assert!(
                    !fixture.path(file).exists(),
                    "no extra CSV: {file} ({sqlite}/{strategy})"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn batch_source_id_named_id_for_both_strategies_and_backends() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["graph", "greedy"] {
            let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
            fixture.config["datasets"]["a"]["id_field"] = json!("id");
            fixture.config["datasets"]["b"]["id_field"] = json!("id");
            fs::write(fixture.path("a.csv"), "id,name,text\na1,aaaa,shared\n")?;
            fs::write(fixture.path("b.csv"), "id,name,text\nb1,aaaa,shared\n")?;
            fixture.run(1)?;
            let pairs = expected(&[("a1", "b1")]);
            assert_eq!(fixture.pairs()?, pairs, "common source ID name must work");
            assert_eq!(
                db_pairs(&fixture.path("output.db"))?,
                pairs,
                "SQLite output uses final pairs"
            );
            let mut csv = csv::Reader::from_path(fixture.path("out/relationships.csv"))?;
            let rows = csv.records().collect::<Result<Vec<_>, _>>()?;
            assert_eq!(rows.len(), 1, "one accepted pair in CSV");
            assert!(
                rows[0].iter().any(|value| value == "a1")
                    && rows[0].iter().any(|value| value == "b1"),
                "both IDs survive CSV serialization"
            );
            #[cfg(feature = "parquet-format")]
            assert_eq!(
                parquet_pairs(&fixture.path("parquet/relationships.parquet"))?,
                pairs,
                "Parquet output uses final pairs"
            );
        }
    }
    Ok(())
}

#[test]
fn graph_coordinates_optimum_and_is_repeatable_across_input_order_and_workers() -> TestResult {
    // Fuzzy ratio plus a constant BM25 self-score gives:
    //             a1      a2
    // b1          .775    .775
    // b2         1.000    .550
    // Both prefer a1 (ID tie-break for b1). Graph must coordinate b1-a2/b2-a1.
    let optimum = expected(&[("a2", "b1"), ("a1", "b2")]);
    for sqlite in [false, true] {
        for assignment in [
            json!({"strategy": "graph"}),
            json!({"strategy": "graph", "mode": "exact"}),
        ] {
            for (shuffle, threads) in [(false, 1), (true, 4)] {
                let fixture = Fixture::new(sqlite, assignment.clone())?;
                if shuffle {
                    fs::write(
                        fixture.path("a.csv"),
                        "aid,name,text\na2,aabb,shared\na1,aaaa,shared\n",
                    )?;
                    fs::write(
                        fixture.path("b.csv"),
                        "bid,name,text\nb2,aaaa,shared\nb1,aaab,shared\n",
                    )?;
                }
                fixture.run(threads)?;
                assert_outputs(&fixture, &optimum)?;
                let rows = csv_rows(&fixture.path("out/relationships.csv"))?;
                assert_eq!(
                    rows.len(),
                    2,
                    "accepted queries must not retain stale reviews"
                );
                for row in rows {
                    let score: f64 = row["score"].parse()?;
                    let target = if row["bid"] == "b1" { 0.775 } else { 1.0 };
                    assert!(
                        (score - target).abs() < 0.0001,
                        "selected metadata must retain pipeline score: {row:?}"
                    );
                }
                if !sqlite {
                    let events = fixture.events()?;
                    assert!(
                        events
                            .iter()
                            .any(|e| e["type"] == "batch_assignment_started"),
                        "graph must start framed log"
                    );
                    let completion = events
                        .iter()
                        .find(|e| e["type"] == "batch_assignment_completed")
                        .expect("completed graph must log summary");
                    assert_eq!(completion["exact_groups"], 1);
                    assert_eq!(completion["approximate_groups"], 0);
                    assert_eq!(completion["largest_group"], 4);
                }
            }
        }
    }
    Ok(())
}

const EDGE_COLUMNS: &[&str] = &["a_id", "b_id", "rank", "score"];
const FIELD_COLUMNS: &[&str] = &[
    "a_id", "b_id", "field_a", "field_b", "method", "score", "weight",
];

type EvidenceRows = Vec<Vec<String>>;

fn assert_evidence_rows(mut actual: EvidenceRows, mut expected: EvidenceRows) {
    // Compare the complete multiset, not just counts or the selected links.
    actual.sort();
    expected.sort();
    assert_eq!(actual.len(), expected.len(), "all raw evidence survives");
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.len(), expected.len(), "fixed evidence schema");
        for (actual, expected) in actual.iter().zip(expected) {
            match (actual.parse::<f64>(), expected.parse::<f64>()) {
                (Ok(a), Ok(b)) => assert!((a - b).abs() < 0.000001, "{a} != {b}"),
                _ => assert_eq!(actual, expected, "evidence value"),
            }
        }
    }
}

fn raw_evidence(root: &Path) -> Result<(EvidenceRows, EvidenceRows), Box<dyn Error>> {
    let mut paths = vec![root.to_path_buf()];
    let mut logs = Vec::new();
    while let Some(path) = paths.pop() {
        for entry in fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_dir() {
                paths.push(path);
            } else if path.to_string_lossy().ends_with("scoring_log.ndjson") {
                logs.push(path);
            }
        }
    }
    assert_eq!(logs.len(), 1, "one full raw scoring log");
    let mut edges = Vec::new();
    let mut fields = Vec::new();
    for line in fs::read_to_string(&logs[0])?.lines() {
        let record: Value = serde_json::from_str(line)?;
        let Some(candidates) = record["candidates"].as_array() else {
            continue;
        };
        for (index, candidate) in candidates.iter().enumerate() {
            assert_eq!(
                candidate["rank"],
                json!(index + 1),
                "raw ranks are one-based"
            );
            let query = record["query_id"].as_str().expect("query ID");
            let matched = candidate["matched_id"].as_str().expect("matched ID");
            let (a, b) = if record["query_side"] == "A" {
                (query, matched)
            } else {
                (matched, query)
            };
            edges.push(vec![
                a.into(),
                b.into(),
                (index + 1).to_string(),
                candidate["score"].to_string(),
            ]);
            for field in candidate["field_scores"].as_array().expect("field scores") {
                fields.push(vec![
                    a.into(),
                    b.into(),
                    field["field_a"].as_str().expect("field A").into(),
                    field["field_b"].as_str().expect("field B").into(),
                    field["method"].as_str().expect("method").into(),
                    field["score"].to_string(),
                    field["weight"].to_string(),
                ]);
            }
        }
    }
    Ok((edges, fields))
}

fn assert_evidence(fixture: &Fixture, files: bool) -> TestResult {
    let (edges, fields) = raw_evidence(fixture.dir.path())?;
    for (table, columns, expected) in [
        ("candidate_edges", EDGE_COLUMNS, edges),
        ("field_scores", FIELD_COLUMNS, fields),
    ] {
        let db = rusqlite::Connection::open(fixture.path("output.db"))?;
        let mut schema = db.prepare(&format!("PRAGMA table_info({table})"))?;
        let names = schema
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(names, columns, "fixed SQL schema for {table}");
        let mut stmt = db.prepare(&format!("SELECT {} FROM {table}", columns.join(",")))?;
        let rows = stmt
            .query_map([], |row| {
                (0..columns.len())
                    .map(|i| {
                        Ok(match row.get_ref(i)? {
                            rusqlite::types::ValueRef::Text(v) => {
                                String::from_utf8_lossy(v).into_owned()
                            }
                            rusqlite::types::ValueRef::Integer(v) => v.to_string(),
                            rusqlite::types::ValueRef::Real(v) => v.to_string(),
                            other => panic!("unexpected evidence value {other:?}"),
                        })
                    })
                    .collect::<rusqlite::Result<Vec<_>>>()
            })?
            .collect::<Result<Vec<_>, _>>()?;
        assert_evidence_rows(rows, expected.clone());
        if files {
            let mut csv = csv::Reader::from_path(fixture.path(&format!("out/{table}.csv")))?;
            assert_eq!(
                csv.headers()?.iter().collect::<Vec<_>>(),
                columns,
                "fixed CSV schema"
            );
            let rows = csv
                .records()
                .map(|row| row.map(|r| r.iter().map(str::to_owned).collect()))
                .collect::<Result<Vec<_>, _>>()?;
            assert_evidence_rows(rows, expected.clone());
            #[cfg(feature = "parquet-format")]
            {
                use parquet::file::reader::{FileReader, SerializedFileReader};
                let reader = SerializedFileReader::new(fs::File::open(
                    fixture.path(&format!("parquet/{table}.parquet")),
                )?)?;
                let names = reader
                    .metadata()
                    .file_metadata()
                    .schema_descr()
                    .columns()
                    .iter()
                    .map(|c| c.name())
                    .collect::<Vec<_>>();
                assert_eq!(names, columns, "fixed Parquet schema");
                let mut rows = Vec::new();
                for row in reader.get_row_iter(None)? {
                    rows.push(
                        row?.get_column_iter()
                            .map(|(_, value)| match value {
                                parquet::record::Field::Str(s) => s.clone(),
                                value => value.to_string(),
                            })
                            .collect(),
                    );
                }
                assert_evidence_rows(rows, expected.clone());
            }
        }
    }
    Ok(())
}

#[test]
fn candidate_evidence_matches_full_raw_log_for_every_strategy_and_backend() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["graph", "greedy"] {
            for db_only in [false, true] {
                let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
                if db_only {
                    fixture.config["output"] = json!({"db_path": fixture.path("output.db")});
                }
                fixture.run(1)?;
                assert_evidence(&fixture, !db_only)?;
                let (edges, _) = raw_evidence(fixture.dir.path())?;
                assert!(edges.iter().any(|r| r[2] == "1"), "include rank one");
                if strategy == "graph" {
                    assert_eq!(fixture.pairs()?, expected(&[("a2", "b1"), ("a1", "b2")]));
                    assert!(
                        edges
                            .iter()
                            .any(|r| r[0] == "a2" && r[1] == "b1" && r[2] == "2"),
                        "selected rank-two winner survives"
                    );
                    assert!(
                        edges
                            .iter()
                            .any(|r| r[0] == "a1" && r[1] == "b1" && r[2] == "1"),
                        "losing rank-one edge survives"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn candidate_evidence_replaces_stale_rows_on_rerun() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["graph", "greedy"] {
            let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
            fixture.run(1)?;
            fs::remove_file(fixture.path("crossmap.csv"))?;
            fs::write(fixture.path("b.csv"), "bid,name,text\nnew-b,aaaa,shared\n")?;
            fixture.run(1)?;
            let db = rusqlite::Connection::open(fixture.path("output.db"))?;
            for table in ["candidate_edges", "field_scores"] {
                let count: usize = db.query_row(
                    &format!("SELECT count(*) FROM {table} WHERE b_id != 'new-b'"),
                    [],
                    |r| r.get(0),
                )?;
                assert_eq!(count, 0, "no stale SQL evidence in {table}");
                let rows = csv_rows(&fixture.path(&format!("out/{table}.csv")))?;
                assert!(!rows.is_empty(), "new evidence replaces old evidence");
                assert!(
                    rows.iter().all(|r| r["bid"] == "new-b"),
                    "no stale CSV evidence"
                );
            }
            let previous = fs::read(fixture.path("out/candidate_edges.csv"))?;
            fixture.config["scoring_log"]["enabled"] = json!(false);
            fs::remove_file(fixture.path("crossmap.csv"))?;
            fixture.run(1)?;
            assert_eq!(
                previous,
                fs::read(fixture.path("out/candidate_edges.csv"))?,
                "disabled runs must not overwrite existing companion files"
            );
        }
    }
    Ok(())
}

#[test]
fn empty_and_isolated_queries_publish_empty_evidence() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["graph", "greedy"] {
            for empty in [false, true] {
                let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
                if empty {
                    fs::write(fixture.path("b.csv"), "bid,name,text\n")?;
                } else {
                    fixture.config["blocking"] = json!({"enabled": true, "fields": [{"field_a": "text", "field_b": "text"}]});
                    fs::write(
                        fixture.path("b.csv"),
                        "bid,name,text\nb-isolated,zzzz,unrelated\n",
                    )?;
                }
                fixture.run(1)?;
                let (edges, fields) = raw_evidence(fixture.dir.path())?;
                assert!(
                    edges.is_empty() && fields.is_empty(),
                    "no invented evidence"
                );
                assert_evidence(&fixture, true)?;
            }
        }
    }
    Ok(())
}

#[test]
fn field_score_companions_follow_batch_opt_in() -> TestResult {
    for sqlite in [false, true] {
        for strategy in ["greedy", "graph"] {
            for enabled in [false, true] {
                let mut fixture = Fixture::new(sqlite, json!({"strategy": strategy}))?;
                fixture.config["scoring_log"]["enabled"] = json!(enabled);
                fixture.run(1)?;
                let csv = fixture.path("out/field_scores.csv");
                assert_eq!(
                    csv.exists(),
                    enabled,
                    "CSV detail for {strategy}, sqlite={sqlite}"
                );
                #[cfg(feature = "parquet-format")]
                assert_eq!(
                    fixture.path("parquet/field_scores.parquet").exists(),
                    enabled,
                    "Parquet detail for {strategy}, sqlite={sqlite}"
                );
                if !enabled {
                    assert!(
                        !fixture.path("out/candidate_edges.csv").exists(),
                        "disabled evidence stays absent"
                    );
                    #[cfg(feature = "parquet-format")]
                    assert!(
                        !fixture.path("parquet/candidate_edges.parquet").exists(),
                        "disabled Parquet evidence stays absent"
                    );
                    let db = rusqlite::Connection::open(fixture.path("output.db"))?;
                    let count: usize = db.query_row("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='candidate_edges'", [], |r| r.get(0))?;
                    assert_eq!(count, 0, "disabled scoring does not add evidence tables");
                }
                if enabled {
                    let count = csv::Reader::from_path(csv)?
                        .records()
                        .collect::<Result<Vec<_>, _>>()?
                        .len();
                    let db = rusqlite::Connection::open(fixture.path("output.db"))?;
                    let expected: usize =
                        db.query_row("SELECT count(*) FROM field_scores", [], |r| r.get(0))?;
                    assert!(expected > 0, "fixture has scored pairs");
                    assert_eq!(count, expected, "companion agrees with SQLite evidence");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn default_graph_agrees_with_explicit_approximate_on_both_backends() -> TestResult {
    let selected = expected(&[("a2", "b1"), ("a1", "b2")]);
    for sqlite in [false, true] {
        for assignment in [
            Value::Null,
            json!({"strategy": "graph", "mode": "approximate"}),
        ] {
            let mut fixture = Fixture::new(sqlite, assignment)?;
            if !sqlite && fixture.config["batch"]["assignment"].is_null() {
                fixture
                    .config
                    .as_object_mut()
                    .expect("fixture config object")
                    .remove("batch");
            }
            let output = fixture.run(2)?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                stdout.contains("Batch assignment: graph (approximate)"),
                "report the effective default: {stdout}"
            );
            assert!(
                stdout.contains("Groups: 1 exact · 0 approximate · largest: 4 records"),
                "report the solved group: {stdout}"
            );
            assert_outputs(&fixture, &selected)?;
        }
    }
    Ok(())
}

#[test]
fn explicit_greedy_preserves_claims_and_graph_objective_dominates_greedy() -> TestResult {
    for sqlite in [false, true] {
        let greedy = Fixture::new(sqlite, json!({"strategy": "greedy"}))?;
        fs::write(greedy.path("b.csv"), "bid,name,text\nb1,aaab,shared\n")?;
        let output = greedy.run(1)?;
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains("Batch assignment: graph"),
            "explicit greedy must not run the graph solver"
        );
        assert_eq!(
            greedy.pairs()?,
            expected(&[("a1", "b1")]),
            "single-query greedy must preserve score/ID tie-break"
        );
        assert_outputs(&greedy, &expected(&[("a1", "b1")]))?;
        if sqlite {
            let mut csv = csv::Reader::from_path(greedy.path("out/relationships.csv"))?;
            let headers = csv.headers()?;
            assert!(
                headers.iter().any(|h| h == "classification"),
                "preserve legacy greedy CSV classification"
            );
            assert!(
                !headers.iter().any(|h| h == "relationship_type"),
                "do not replace legacy greedy CSV schema"
            );
        }
        if !sqlite {
            assert!(
                !greedy
                    .events()?
                    .iter()
                    .any(|e| e["type"] == "batch_assignment_started"),
                "greedy must not collect/log optimizer graph"
            );
        }
        // Greedy query order is deliberately NOT a determinism promise: even
        // one worker sees HashMap-derived IDs. It can obtain either 1.325 or
        // 1.775 here; never require a random process hash seed to hit the trap.
        let trap = Fixture::new(sqlite, json!({"strategy": "greedy"}))?;
        trap.run(1)?;
        let graph = Fixture::new(sqlite, json!({"strategy": "graph", "mode": "exact"}))?;
        graph.run(1)?;
        let objective = |fixture: &Fixture| -> Result<f64, Box<dyn Error>> {
            csv_rows(&fixture.path("out/relationships.csv"))?
                .iter()
                .filter(|r| r["relationship_type"] == "match")
                .map(|r| r["score"].parse::<f64>().map_err(Into::into))
                .sum()
        };
        let greedy_score = objective(&trap)?;
        let graph_score = objective(&graph)?;
        assert!(
            (graph_score - 1.775).abs() < 0.0001,
            "graph reaches independent optimum"
        );
        assert!(
            graph_score >= greedy_score,
            "graph must not regress greedy objective"
        );
        // Independent b1-first greedy takes the tied b1-a1 edge first.
        assert!(
            graph_score > 0.775 + 0.55,
            "graph must escape coordinated trap"
        );
    }
    Ok(())
}

#[test]
fn protected_links_and_whole_list_gap_survive_graph_assignment() -> TestResult {
    for sqlite in [false, true] {
        let protected = Fixture::new(sqlite, json!({"strategy": "graph", "mode": "exact"}))?;
        fs::write(protected.path("crossmap.csv"), "aid,bid\na1,b1\n")?;
        protected.run(2)?;
        assert_outputs(&protected, &expected(&[("a1", "b1"), ("a2", "b2")]))?;

        let mut gap = Fixture::new(sqlite, json!({"strategy": "graph"}))?;
        gap.config["thresholds"]["min_score_gap"] = json!(0.01);
        gap.run(2)?;
        assert_outputs(&gap, &expected(&[("a1", "b2")]))?;
        let rows = csv_rows(&gap.path("out/relationships.csv"))?;
        assert!(
            rows.iter()
                .any(|row| row["bid"] == "b1" && row["relationship_type"] == "review"),
            "ambiguous query must remain review, not recover lower-ranked Auto edge"
        );
    }
    Ok(())
}

#[test]
fn graph_export_rejects_absent_empty_torn_and_ambiguous_inputs_without_writes() -> TestResult {
    let mut f = Fixture::new(false, json!({"strategy":"graph"}))?;
    f.run(1)?;
    let segment = f.log_path()?;
    let complete = fs::read(&segment)?;
    let base = f.path("out/graph_cli.graph.match_log.ndjson");
    let output_dir = f.path("previous");
    fs::create_dir(&output_dir)?;
    fs::write(
        output_dir.join("relationships.csv"),
        b"existing relationships",
    )?;
    let original_db = fs::read(f.path("output.db"))?;
    for case in [
        "missing",
        "empty",
        "torn start",
        "no markers",
        "two segments",
    ] {
        let selected = match case {
            "missing" => {
                fs::remove_file(&segment)?;
                &segment
            }
            "empty" => {
                fs::write(&segment, [])?;
                &segment
            }
            "torn start" => {
                fs::write(&segment, b"{\"ts\":\"")?;
                &segment
            }
            "no markers" => {
                fs::write(
                    &segment,
                    b"{\"type\":\"upsert_record\",\"side\":\"b\",\"record\":{\"bid\":\"b1\"}}\n",
                )?;
                &segment
            }
            _ => {
                fs::write(&segment, &complete)?;
                fs::write(
                    f.path("out/graph_cli.graph.match_log_20991231T235959Z_000001.ndjson"),
                    [],
                )?;
                &base
            }
        };
        f.config["live"] = json!({"match_log_path": selected});
        let out = f
            .command("export", 1)?
            .arg("--out-dir")
            .arg(&output_dir)
            .output()?;
        assert!(!out.status.success(), "accepted {case}");
        assert_eq!(
            fs::read(output_dir.join("relationships.csv"))?,
            b"existing relationships",
            "{case}"
        );
        assert_eq!(
            fs::read(f.path("output.db"))?,
            original_db,
            "{case} modified DB"
        );
        assert!(
            !output_dir.join("unmatched.csv").exists(),
            "{case} wrote output"
        );
    }
    Ok(())
}

#[test]
fn graph_refuses_malformed_constraints_before_overwriting_any_snapshot() -> TestResult {
    for sqlite in [false, true] {
        for (input, csv) in [
            ("crossmap.csv", "wrong,bid\na1,b1\n"),
            ("exclusions.csv", "aid,typo\na1,b1\n"),
            ("crossmap.csv", "aid,bid\na1,\n"),
            ("exclusions.csv", "aid,bid\n,b1\n"),
            ("crossmap.csv", "aid,bid\na1,b1\na1,b2\n"),
            ("crossmap.csv", "aid,bid\na1,b2\na1,b1\n"),
            ("crossmap.csv", "aid,bid\na1,b1\na2,b1\n"),
            ("crossmap.csv", "aid,bid\na2,b1\na1,b1\n"),
        ] {
            let mut f = Fixture::new(sqlite, json!({"strategy":"graph"}))?;
            fs::write(f.path("crossmap.csv"), "aid,bid\na2,b2\n")?;
            fs::write(f.path(input), csv)?;
            if input == "exclusions.csv" {
                f.config["exclusions"] =
                    json!({"path":f.path(input),"a_id_field":"aid","b_id_field":"bid"});
            }
            let previous = fs::read(f.path("crossmap.csv"))?;
            let output = f.command("run", 2)?.output()?;
            assert!(
                !output.status.success(),
                "accepted sqlite={sqlite}, {input}: {csv}"
            );
            assert_eq!(fs::read(f.path("crossmap.csv"))?, previous);
            assert!(!f.path("out/relationships.csv").exists());
            assert!(!f.path("output.db").exists());
        }
    }
    Ok(())
}

#[test]
fn live_wal_export_is_independent_of_the_batch_graph_default() -> TestResult {
    use melder::models::Side;
    use melder::state::match_log::{MatchLog, MatchLogEvent};

    let mut fixture = Fixture::new(false, Value::Null)?;
    let base = fixture.path("live.match_log.ndjson");
    let log = MatchLog::open(&base)?;
    log.append_upsert(
        Side::A,
        &std::collections::HashMap::from([
            ("aid".into(), "a1".into()),
            ("name".into(), "aaaa".into()),
        ]),
    )?;
    log.append_upsert(
        Side::B,
        &std::collections::HashMap::from([
            ("bid".into(), "b1".into()),
            ("name".into(), "aaab".into()),
        ]),
    )?;
    log.append(&MatchLogEvent::CrossMapConfirm {
        a_id: "a1".into(),
        b_id: "b1".into(),
        score: Some(0.775),
        rank: Some(1),
        reason: None,
    })?;
    log.flush()?;
    fixture.config["live"] = json!({"match_log_path": base});
    let output = fixture
        .command("export", 1)?
        .arg("--out-dir")
        .arg(fixture.path("live-export"))
        .output()?;
    assert_success(&output);
    assert_eq!(
        csv_pairs(&fixture.path("live-export/relationships.csv"), true)?,
        expected(&[("a1", "b1")]),
        "live export must not require batch assignment markers"
    );
    Ok(())
}

#[test]
fn historical_export_uses_its_own_scoring_evidence() -> TestResult {
    let mut f = Fixture::new(false, json!({"strategy":"graph"}))?;
    f.run(1)?;
    let old = f.log_path()?;
    let expected_candidates = fs::read(f.path("out/candidates.csv"))?;
    fs::remove_file(f.path("crossmap.csv"))?;
    fs::write(f.path("b.csv"), "bid,name,text\nnew-b,aaaa,shared\n")?;
    f.config["scoring_log"]["compression"] = json!("zstd");
    f.run(1)?;
    let new_candidates = fs::read(f.path("out/candidates.csv"))?;
    assert_ne!(new_candidates, expected_candidates);
    f.config["live"] = json!({"match_log_path":old});
    let out = f
        .command("export", 1)?
        .arg("--out-dir")
        .arg(f.path("historical"))
        .output()?;
    assert_success(&out);
    assert_eq!(
        fs::read(f.path("historical/candidates.csv"))?,
        expected_candidates
    );
    let db = rusqlite::Connection::open(f.path("output.db"))?;
    let (old_rows, later_rows): (usize, usize) = db.query_row(
        "SELECT count(*), count(CASE WHEN b_id='new-b' THEN 1 END) FROM field_scores",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert!(
        old_rows > 0 && later_rows == 0,
        "field scores belong to the original run"
    );
    Ok(())
}

#[test]
fn completed_batch_exports_agree_and_incomplete_logs_fail_closed() -> TestResult {
    let mut fixture = Fixture::new(false, json!({"strategy": "graph", "mode": "exact"}))?;
    fixture.run(2)?;
    let selected = fixture.pairs()?;
    assert_success(&fixture.export("exported")?);
    assert_eq!(
        csv_pairs(&fixture.path("exported/relationships.csv"), true)?,
        selected,
        "CLI export must reproduce completed graph membership"
    );
    let log_path = fixture.log_path()?;
    let log = fs::read_to_string(&log_path)?;
    let incomplete = log
        .lines()
        .filter(|line| !line.contains("batch_assignment_completed"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(log_path, incomplete)?;
    let output = fixture.export("incomplete-export")?;
    assert!(
        !output.status.success(),
        "incomplete optimizer log must fail export"
    );
    assert!(
        !fixture.path("incomplete-export/relationships.csv").exists(),
        "failed export must not publish seemingly complete relationships"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("completion"),
        "failure must explain incomplete graph run: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn sqlite_graph_scoring_protection_gap_and_exclusions_without_encoder() -> TestResult {
    // Independently runnable while the memory startup/output adapters are being
    // integrated; the other tests require full CSV + output-DB parity.
    for assignment in [
        json!({"strategy": "graph"}),
        json!({"strategy": "graph", "mode": "exact"}),
    ] {
        for case in ["ordinary", "protected", "gap", "excluded"] {
            let mut fixture = Fixture::new(true, assignment.clone())?;
            let selected = match case {
                "protected" => {
                    fs::write(fixture.path("crossmap.csv"), "aid,bid\na1,b1\n")?;
                    expected(&[("a1", "b1"), ("a2", "b2")])
                }
                "gap" => {
                    fixture.config["thresholds"]["min_score_gap"] = json!(0.01);
                    expected(&[("a1", "b2")])
                }
                "excluded" => {
                    fs::write(fixture.path("exclusions.csv"), "aid,bid\na1,b2\n")?;
                    fixture.config["exclusions"] = json!({"path": fixture.path("exclusions.csv"),
                        "a_id_field": "aid", "b_id_field": "bid"});
                    expected(&[("a1", "b1"), ("a2", "b2")])
                }
                _ => expected(&[("a2", "b1"), ("a1", "b2")]),
            };
            fixture.run(4)?;
            assert_eq!(fixture.pairs()?, selected, "SQLite case: {case}");
            assert_eq!(
                csv_pairs(&fixture.path("out/relationships.csv"), true)?,
                selected,
                "SQLite CSV case: {case}"
            );
        }
    }
    Ok(())
}

#[test]
fn excluded_protected_input_fails_without_replacing_the_snapshot() -> TestResult {
    for sqlite in [true, false] {
        let mut fixture = Fixture::new(sqlite, json!({"strategy": "graph"}))?;
        let snapshot = "aid,bid\na1,b1\n";
        fs::write(fixture.path("crossmap.csv"), snapshot)?;
        fs::write(fixture.path("exclusions.csv"), snapshot)?;
        fixture.config["exclusions"] = json!({"path": fixture.path("exclusions.csv"),
            "a_id_field": "aid", "b_id_field": "bid"});
        let output = fixture.command("run", 2)?.output()?;
        assert!(
            !output.status.success(),
            "excluded protected input must be rejected"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("also excluded"),
            "must explain inconsistent protected input: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read_to_string(fixture.path("crossmap.csv"))?,
            snapshot,
            "failed graph run must not overwrite protected snapshot"
        );
        assert!(
            !fixture.path("out/relationships.csv").exists(),
            "failed run must not publish output"
        );
    }
    Ok(())
}

#[test]
fn invalid_assignment_yaml_is_rejected_by_real_cli() -> TestResult {
    for assignment in [
        json!({"strategy": "greedy", "mode": "exact"}),
        json!({"strategy": "greedy", "mode": "approximate"}),
        json!({"strategy": "maximum_weight"}),
        json!({"strategy": "graph", "mode": "unknown"}),
    ] {
        let fixture = Fixture::new(false, assignment)?;
        let output = fixture.command("run", 1)?.output()?;
        assert!(
            !output.status.success(),
            "invalid assignment YAML must fail"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("batch.assignment.mode: requires batch.assignment.strategy: graph")
                || stderr.contains("unknown variant"),
            "must reject assignment config itself, not fail later during startup: {stderr}"
        );
        assert!(
            !fixture.path("out/relationships.csv").exists(),
            "invalid YAML must not publish output"
        );
    }
    Ok(())
}

#[test]
fn export_reports_the_match_config_error_if_enroll_parsing_also_fails() -> TestResult {
    let fixture = Fixture::new(false, json!({"strategy":"greedy", "mode":"exact"}))?;
    let output = fixture
        .command("export", 1)?
        .arg("--out-dir")
        .arg(fixture.path("export"))
        .output()?;
    assert!(!output.status.success(), "invalid config must not export");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("batch.assignment.mode"),
        "show the original match configuration error"
    );
    Ok(())
}

#[test]
fn default_approximate_dispatch_counts_distinct_endpoints_at_512_and_513() -> TestResult {
    boundary_cases(false)
}

#[test]
fn sqlite_cli_dispatch_at_512_and_513_without_encoder() -> TestResult {
    boundary_cases(true)
}

fn boundary_cases(sqlite: bool) -> TestResult {
    // One connected star, not 512 disconnected pairs: 1 A + N B endpoints.
    for (endpoints, approximate) in [(512, 0), (513, 1)] {
        for mode in [None, Some("exact")] {
            let assignment = mode.map_or(Value::Null, |mode| json!({"mode": mode}));
            let mut fixture = Fixture::new(sqlite, assignment)?;
            fs::write(fixture.path("a.csv"), "aid,name,text\na1,aaaa,shared\n")?;
            let mut b = String::from("bid,name,text\n");
            for i in 0..endpoints - 1 {
                b.push_str(&format!("b{i:04},aaaa,shared\n"));
            }
            fs::write(fixture.path("b.csv"), b)?;
            fixture.config["top_n"] = json!(1);
            fixture.config["bm25_candidates"] = json!(1);
            let output = fixture.run(4)?;
            let fast = if mode.is_some() { 0 } else { approximate };
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                stdout.contains(&format!(
                    "Groups: {} exact · {fast} approximate · largest: {endpoints} records",
                    1 - fast
                )),
                "CLI completion must report effective dispatch, not only requested mode: {stdout}"
            );
            if !sqlite {
                let events = fixture.events()?;
                let completion = events
                    .iter()
                    .find(|e| e["type"] == "batch_assignment_completed")
                    .expect("successful graph completion");
                assert_eq!(
                    completion["largest_group"], endpoints,
                    "count distinct participating A+B records"
                );
                assert_eq!(
                    completion["approximate_groups"], fast,
                    "explicit exact must never fall back"
                );
                assert_eq!(completion["exact_groups"], 1 - fast);
            }
            assert_eq!(fixture.pairs()?.len(), 1, "star permits exactly one pair");
            assert_eq!(
                csv_pairs(&fixture.path("out/relationships.csv"), true)?,
                fixture.pairs()?
            );
            if !sqlite {
                assert_outputs(&fixture, &fixture.pairs()?)?;
            }
            let rows = csv_rows(&fixture.path("out/relationships.csv"))?;
            assert_eq!(
                rows.len(),
                endpoints - 1,
                "all losing queries retain review evidence"
            );
            assert_eq!(
                rows.iter()
                    .filter(|row| row["relationship_type"] == "review")
                    .count(),
                endpoints - 2,
                "competition is review, not accepted Auto"
            );
            if !sqlite {
                assert_eq!(
                    rows.iter()
                        .filter(|row| row["reason"] == "assignment_conflict")
                        .count(),
                    endpoints - 2,
                    "competition review must explain assignment conflict"
                );
            }
        }
    }
    Ok(())
}
