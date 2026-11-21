//! Real-server startup assignment and durable handoff regressions, using an offline encoder.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn Error>>;

struct Server {
    child: Child,
    port: u16,
}

impl Server {
    fn start(root: &Path, config: &Value) -> Result<Self, Box<dyn Error>> {
        fs::write(root.join("config.yaml"), serde_yaml::to_string(config)?)?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let log = fs::File::create(root.join("server.log"))?;
        let child = Command::new(env!("CARGO_BIN_EXE_meld"))
            .args(["serve", "--config"])
            .arg(root.join("config.yaml"))
            .args(["--port", &port.to_string()])
            .env("RAYON_NUM_THREADS", "1")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()?;
        let mut server = Self { child, port };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if server.request("GET", "/api/v1/health", None).is_ok() {
                return Ok(server);
            }
            if server.child.try_wait()?.is_some() || Instant::now() >= deadline {
                return Err(format!(
                    "server failed: {}",
                    fs::read_to_string(root.join("server.log"))?
                )
                .into());
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, Box<dyn Error>> {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port))?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        let body = body.map(|v| v.to_string()).unwrap_or_default();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let (headers, payload) = response
            .split_once("\r\n\r\n")
            .ok_or("invalid HTTP response")?;
        assert!(
            headers.starts_with("HTTP/1.1 2"),
            "request failed: {response}"
        );
        Ok(serde_json::from_str(payload)?)
    }

    fn pairs(&self) -> Result<BTreeSet<(String, String)>, Box<dyn Error>> {
        let response = self.request("GET", "/api/v1/crossmap/pairs", None)?;
        response["pairs"]
            .as_array()
            .ok_or("missing pairs")?
            .iter()
            .map(|pair| {
                Ok((
                    pair["a_id"].as_str().ok_or("missing a_id")?.into(),
                    pair["b_id"].as_str().ok_or("missing b_id")?.into(),
                ))
            })
            .collect()
    }

    fn stop(mut self) -> TestResult {
        self.request("POST", "/admin/shutdown", Some(json!({})))?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait()? {
                assert!(status.success(), "server shutdown must succeed");
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err("server shutdown timed out".into());
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn make_config(root: &Path, sqlite: bool) -> Result<Value, Box<dyn Error>> {
    fs::write(
        root.join("a.csv"),
        "aid,name,text\na1,aaaa,shared\na2,aabb,shared\n",
    )?;
    fs::write(
        root.join("b.csv"),
        "bid,name,text\nb1,aaab,shared\nb2,aaaa,shared\n",
    )?;
    let stub = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stub_encoder.py");
    let mut config = json!({
        "job": {"name":"startup_test"},
        "datasets": {"a":{"path":root.join("a.csv"),"id_field":"aid"},"b":{"path":root.join("b.csv"),"id_field":"bid"}},
        "cross_map":{"backend":"local","path":root.join("crossmap.csv"),"a_id_field":"aid","b_id_field":"bid"},
        "embeddings":{"remote_encoder_cmd":format!("python3 '{}' --vector-dim 16",stub.display()),"a_cache_dir":root.join("cache-a"),"b_cache_dir":root.join("cache-b")},
        "vector_backend":"flat", "performance":{"encoder_pool_size":1},
        "exact_prefilter":{"enabled":false}, "top_n":2,"bm25_candidates":2,
        "match_fields":[{"method":"bm25","weight":0.1,"fields":[{"field_a":"text","field_b":"text"}]},{"method":"fuzzy","scorer":"ratio","weight":0.9,"field_a":"name","field_b":"name"}],
        "thresholds":{"auto_match":0.5,"review_floor":0.2},
        "output":{"csv_dir_path":root.join("out")},
        "live":{"match_log_path":root.join("matches.ndjson"),"crossmap_flush_secs":3600}
    });
    if sqlite {
        config["live"]["db_path"] = json!(root.join("live.db"));
    }
    Ok(config)
}

fn pairs(items: &[(&str, &str)]) -> BTreeSet<(String, String)> {
    items
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

#[test]
fn graph_startup_survives_first_request_and_restart_on_both_backends() -> TestResult {
    for sqlite in [false, true] {
        let dir = tempfile::tempdir()?;
        let mut config = make_config(dir.path(), sqlite)?;
        // The standalone batch policy must not select the live startup policy.
        config["batch"] = json!({"assignment":{"strategy":"greedy"}});
        let expected = pairs(&[("a1", "b2"), ("a2", "b1")]);
        let server = Server::start(dir.path(), &config)?;
        assert_eq!(
            server.pairs()?,
            expected,
            "default startup graph chooses the better global pairing (sqlite={sqlite})"
        );
        assert_eq!(
            server.request("GET", "/api/v1/a/unmatched", None)?["total"],
            0,
            "accepted A flags must be visible before requests"
        );
        assert_eq!(
            server.request("GET", "/api/v1/b/unmatched", None)?["total"],
            0,
            "accepted B flags must be visible before requests"
        );
        server.request(
            "POST",
            "/api/v1/b/add",
            Some(json!({"record":{"bid":"b3","name":"aaaa","text":"shared"}})),
        )?;
        assert_eq!(
            server.pairs()?,
            expected,
            "incoming live request must retain startup ownership"
        );
        server.stop()?;
        config["live"]["skip_initial_match"] = json!(true);
        let server = Server::start(dir.path(), &config)?;
        assert_eq!(
            server.pairs()?,
            expected,
            "restart with startup disabled must restore the graph decisions"
        );
        server.stop()?;
    }
    Ok(())
}

#[test]
fn graph_modes_leave_protected_pairs_for_incoming_deferred_acceptance() -> TestResult {
    for sqlite in [false, true] {
        for mode in ["approximate", "exact"] {
            let dir = tempfile::tempdir()?;
            let mut config = make_config(dir.path(), sqlite)?;
            config["live"]["startup"] = json!({"assignment": {"strategy": "graph", "mode": mode}});
            config["live"]["assignment"] = json!({"strategy": "deferred_acceptance"});
            let expected = pairs(&[("a1", "b2"), ("a2", "b1")]);
            let server = Server::start(dir.path(), &config)?;
            assert_eq!(
                server.pairs()?,
                expected,
                "startup must use its graph mode instead of the live request policy"
            );
            server.request(
                "POST",
                "/api/v1/b/add",
                Some(json!({"record": {"bid": "b3", "name": "aabb", "text": "shared"}})),
            )?;
            assert_eq!(
                server.pairs()?,
                expected,
                "a new perfect match must not displace a protected startup pair"
            );
            server.stop()?;
        }
    }
    Ok(())
}

#[test]
fn explicit_greedy_and_skipped_startup_are_independent_of_live_request_policy() -> TestResult {
    for sqlite in [false, true] {
        for skip in [false, true] {
            let dir = tempfile::tempdir()?;
            let mut config = make_config(dir.path(), sqlite)?;
            config["live"]["startup"] = json!({"assignment":{"strategy":"greedy"}});
            config["live"]["assignment"] = json!({"strategy":"deferred_acceptance"});
            config["live"]["skip_initial_match"] = json!(skip);
            let server = Server::start(dir.path(), &config)?;
            if skip {
                assert!(
                    server.pairs()?.is_empty(),
                    "skip must suppress all startup assignment"
                );
            } else {
                assert_eq!(
                    server.pairs()?,
                    pairs(&[("a1", "b1"), ("a2", "b2")]),
                    "explicit startup greedy must not run deferred acceptance"
                );
            }
            server.stop()?;
        }
    }
    Ok(())
}

#[test]
fn graph_startup_clears_prior_reviews_when_candidates_become_accepted() -> TestResult {
    for sqlite in [false, true] {
        let dir = tempfile::tempdir()?;
        let mut config = make_config(dir.path(), sqlite)?;
        config["thresholds"]["auto_match"] = json!(0.99);
        let server = Server::start(dir.path(), &config)?;
        assert!(
            server.request("GET", "/api/v1/review/list", None)?["total"]
                .as_u64()
                .is_some_and(|n| n > 0),
            "high threshold must create reviews before acceptance (sqlite={sqlite})"
        );
        server.stop()?;

        config["thresholds"]["auto_match"] = json!(0.5);
        let expected = pairs(&[("a1", "b2"), ("a2", "b1")]);
        let server = Server::start(dir.path(), &config)?;
        assert_eq!(
            server.pairs()?,
            expected,
            "lower threshold must accept both graph pairs (sqlite={sqlite})"
        );
        assert_eq!(
            server.request("GET", "/api/v1/review/list", None)?["total"],
            0,
            "accepted pairs must clear prior candidate reviews (sqlite={sqlite})"
        );
        server.stop()?;

        config["live"]["skip_initial_match"] = json!(true);
        let server = Server::start(dir.path(), &config)?;
        assert_eq!(
            server.pairs()?,
            expected,
            "accepted pairs must survive skipped startup (sqlite={sqlite})"
        );
        assert_eq!(
            server.request("GET", "/api/v1/review/list", None)?["total"],
            0,
            "cleared reviews must stay cleared after restart (sqlite={sqlite})"
        );
        server.stop()?;
    }
    Ok(())
}

#[test]
fn startup_preserves_protected_pairs_exclusions_and_review_state() -> TestResult {
    for sqlite in [false, true] {
        let dir = tempfile::tempdir()?;
        let mut config = make_config(dir.path(), sqlite)?;
        fs::write(dir.path().join("crossmap.csv"), "aid,bid\na2,b2\n")?;
        fs::write(dir.path().join("exclusions.csv"), "aid,bid\na1,b1\n")?;
        config["exclusions"] =
            json!({"path":dir.path().join("exclusions.csv"),"a_id_field":"aid","b_id_field":"bid"});
        let server = Server::start(dir.path(), &config)?;
        assert_eq!(
            server.pairs()?,
            pairs(&[("a2", "b2")]),
            "protected pair must survive and excluded pair must not be accepted"
        );
        server.stop()?;

        let review_dir = tempfile::tempdir()?;
        let mut config = make_config(review_dir.path(), sqlite)?;
        config["thresholds"]["auto_match"] = json!(0.99);
        let server = Server::start(review_dir.path(), &config)?;
        let before = server.request("GET", "/api/v1/review/list", None)?;
        assert!(
            before["total"].as_u64().is_some_and(|n| n > 0),
            "review-band startup decisions must populate the live queue: {before}"
        );
        server.stop()?;
        config["live"]["skip_initial_match"] = json!(true);
        let server = Server::start(review_dir.path(), &config)?;
        assert_eq!(
            server.request("GET", "/api/v1/review/list", None)?,
            before,
            "review entries must survive restart without re-scoring"
        );
        server.stop()?;
    }
    Ok(())
}
