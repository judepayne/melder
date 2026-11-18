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
        let base = if self.config["batch"]["assignment"]["strategy"] == "graph" {
            "out/graph_cli.graph.match_log.ndjson"
        } else {
            "out/graph_cli.match_log.ndjson"
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

#[test]
fn default_greedy_is_compatible_and_graph_objective_dominates_greedy() -> TestResult {
    for sqlite in [false, true] {
        let default = Fixture::new(sqlite, Value::Null)?;
        fs::write(default.path("b.csv"), "bid,name,text\nb1,aaab,shared\n")?;
        default.run(1)?;
        let greedy = Fixture::new(sqlite, json!({"strategy": "greedy"}))?;
        fs::write(greedy.path("b.csv"), "bid,name,text\nb1,aaab,shared\n")?;
        greedy.run(1)?;
        assert_eq!(
            default.pairs()?,
            greedy.pairs()?,
            "omission must preserve greedy behavior"
        );
        assert_eq!(
            greedy.pairs()?,
            expected(&[("a1", "b1")]),
            "single-query greedy must preserve score/ID tie-break"
        );
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
        json!({"mode": "exact"}),
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
            let assignment = mode.map_or(
                json!({"strategy": "graph"}),
                |mode| json!({"strategy": "graph", "mode": mode}),
            );
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
