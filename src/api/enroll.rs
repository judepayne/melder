//! Production single-pool HTTP dispatch and administrative snapshots.
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::Serialize;

use super::handlers::{
    AddRequest, BatchRecordsRequest, ExcludeRequest, QueryParams, RemoveRequest, error_response,
    json_ok,
};
use crate::error::SessionError;
use crate::session::enroll::EnrollSession;

/// HTTP lifecycle state, independent of paired matching resources.
pub struct EnrollApi {
    pub session: Arc<EnrollSession>,
    pub shutdown_notify: Arc<tokio::sync::Notify>,
}
impl EnrollApi {
    /// Attach HTTP lifecycle to the single-pool session.
    pub fn new(session: Arc<EnrollSession>) -> Arc<Self> {
        Arc::new(Self {
            session,
            shutdown_notify: Arc::new(tokio::sync::Notify::new()),
        })
    }
}
async fn run<T: Serialize + Send + 'static>(
    f: impl FnOnce() -> Result<T, SessionError> + Send + 'static,
) -> Response {
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(value)) => json_ok(value),
        Ok(Err(e)) => error_response(
            StatusCode::from_u16(e.status_code()).unwrap_or(StatusCode::BAD_REQUEST),
            &e.to_string(),
        )
        .into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}
async fn enroll(State(s): State<Arc<EnrollApi>>, Json(b): Json<AddRequest>) -> Response {
    run(move || s.session.enroll(b.record)).await
}
async fn batch(State(s): State<Arc<EnrollApi>>, Json(b): Json<BatchRecordsRequest>) -> Response {
    run(move || s.session.enroll_batch(b.records)).await
}
async fn remove(State(s): State<Arc<EnrollApi>>, Json(b): Json<RemoveRequest>) -> Response {
    run(move || s.session.remove_record(&b.id)).await
}
async fn query(State(s): State<Arc<EnrollApi>>, Query(b): Query<QueryParams>) -> Response {
    run(move || s.session.query_record(&b.id)).await
}
async fn count(State(s): State<Arc<EnrollApi>>) -> Response {
    run(move || Ok(serde_json::json!({"count": s.session.count()?}))).await
}
async fn health(State(s): State<Arc<EnrollApi>>) -> Response {
    run(move || Ok(s.session.health())).await
}
async fn status(State(s): State<Arc<EnrollApi>>) -> Response {
    run(move || Ok(s.session.status())).await
}
async fn exclude(State(s): State<Arc<EnrollApi>>, Json(b): Json<ExcludeRequest>) -> Response {
    run(move || s.session.exclude(&b.a_id, &b.b_id)).await
}
async fn unexclude(State(s): State<Arc<EnrollApi>>, Json(b): Json<ExcludeRequest>) -> Response {
    run(move || s.session.unexclude(&b.a_id, &b.b_id)).await
}
fn outputs(s: &EnrollApi) -> Result<crate::output::BuildReport, Box<dyn std::error::Error>> {
    let state = &s.session.state;
    let cfg = &state.config;
    let (records, events) = state.snapshot()?;
    let scoring = crate::output::manifest::enroll_scoring_log_path(cfg);
    crate::output::build::build_outputs_from_events_with_authority(
        &events,
        scoring.as_deref(),
        cfg.output.csv_dir_path.as_deref().map(std::path::Path::new),
        cfg.output
            .parquet_dir_path
            .as_deref()
            .map(std::path::Path::new),
        cfg.output.db_path.as_deref().map(std::path::Path::new),
        &crate::output::OutputManifest::from_enroll_job(cfg),
        crate::output::build::BuildInput {
            a_records: records.into_iter().collect(),
            b_records: Default::default(),
            snapshot_pairs: Vec::new(),
            current_reviews: None,
        },
        crate::output::build::EnrollRecordAuthority::Snapshot,
    )
}
async fn admin(s: Arc<EnrollApi>, shutdown: bool) -> Response {
    let cfg = &s.session.state.config;
    let configured = cfg.output.csv_dir_path.is_some()
        || cfg.output.parquet_dir_path.is_some()
        || cfg.output.db_path.is_some();
    if !configured && !shutdown {
        return error_response(
            StatusCode::BAD_REQUEST,
            "no output.csv_dir_path, output.parquet_dir_path, or output.db_path configured",
        )
        .into_response();
    }
    let worker = s.clone();
    let result = tokio::task::spawn_blocking(move || {
        if configured { outputs(&worker).map(|r| serde_json::json!({"status":"complete","matches":r.match_count,"enroll_edges":r.enroll_edge_count,"review":r.review_count,"unmatched":r.no_match_count,"elapsed_secs":r.elapsed_secs})).map_err(|e|e.to_string()) }
        else { Ok(serde_json::json!({"status":"shutting_down"})) }
    }).await;
    if shutdown {
        match result {
            Ok(Err(e)) => tracing::warn!(error = %e, "admin/shutdown output build failed"),
            Err(e) => tracing::warn!(error = %e, "admin/shutdown output worker failed"),
            Ok(Ok(_)) => {}
        }
        s.shutdown_notify.notify_one();
        return (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({"status":"shutting_down"})),
        )
            .into_response();
    }
    match result {
        Ok(Ok(r)) => json_ok(r),
        Ok(Err(e)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}
async fn flush(State(s): State<Arc<EnrollApi>>) -> Response {
    admin(s, false).await
}
async fn shutdown(State(s): State<Arc<EnrollApi>>) -> Response {
    admin(s, true).await
}
/// Build the single-pool routes without constructing a paired live state.
pub fn router(state: Arc<EnrollApi>) -> Router {
    Router::new()
        .route("/api/v1/enroll", post(enroll))
        .route("/api/v1/enroll-batch", post(batch))
        .route("/api/v1/enroll/remove", post(remove))
        .route("/api/v1/enroll/query", get(query))
        .route("/api/v1/enroll/count", get(count))
        .route("/api/v1/health", get(health))
        .route("/api/v1/status", get(status))
        .route("/api/v1/exclude", post(exclude).merge(delete(unexclude)))
        .route("/admin/flush", post(flush))
        .route("/admin/shutdown", post(shutdown))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct UnusedEncoder;
    impl crate::encoder::Encoder for UnusedEncoder {
        fn encode(&self, _: &[&str]) -> Result<Vec<Vec<f32>>, crate::error::EncoderError> {
            panic!("exact-only HTTP operations must not encode")
        }
        fn dim(&self) -> usize {
            0
        }
        fn pool_size(&self) -> usize {
            1
        }
        fn encode_batch_size(&self) -> usize {
            1
        }
    }

    #[tokio::test]
    async fn typed_dispatch_preserves_records_errors_and_admin_snapshots() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let config = dir.path().join("config.yaml");
        std::fs::write(
            &config,
            format!(
                r#"
job: {{name: api_enroll}}
dataset: {{path: "", id_field: id}}
embeddings: {{model: unused, cache_dir: "{}"}}
match_fields:
  - {{field: name, method: exact, weight: 1.0}}
thresholds: {{auto_match: 0.85, review_floor: 0.6}}
vector_backend: flat
live: {{match_log_path: "{}"}}
output: {{csv_dir_path: "{}"}}
"#,
                dir.path().join("cache").display(),
                dir.path().join("wal").display(),
                dir.path().join("out").display()
            ),
        )
        .expect("write config");
        let cfg = crate::config::load_enroll_job_config(&config).expect("valid config");
        let state =
            crate::state::enroll::EnrollState::load_job_with_encoder(cfg, Arc::new(UnusedEncoder))
                .expect("load state");
        let api = EnrollApi::new(Arc::new(EnrollSession::new(state, None)));
        let missing = query(
            State(api.clone()),
            Query(QueryParams {
                id: "missing".into(),
            }),
        )
        .await;
        assert_eq!(
            missing.status(),
            StatusCode::NOT_FOUND,
            "missing record preserves 404"
        );
        let added = enroll(
            State(api.clone()),
            Json(AddRequest {
                record: [("id".into(), "one".into()), ("name".into(), "Alice".into())]
                    .into_iter()
                    .collect(),
            }),
        )
        .await;
        assert_eq!(added.status(), StatusCode::OK, "typed enrollment succeeds");
        assert_eq!(
            api.session.count().expect("count"),
            1,
            "one physical pool record"
        );
        let report = outputs(&api).expect("snapshot output");
        assert_eq!(
            report.no_match_count, 1,
            "isolated record survives snapshot output"
        );
        let flushed = flush(State(api.clone())).await;
        assert_eq!(flushed.status(), StatusCode::OK, "admin flush succeeds");
        assert_shutdown_accepted(api).await;
        for destination in [None, Some(config.to_string_lossy().into_owned())] {
            // An existing regular file cannot be used as an output directory.
            let mut cfg = crate::config::load_enroll_job_config(&config).expect("valid config");
            cfg.output.csv_dir_path = destination;
            let state = crate::state::enroll::EnrollState::load_job_with_encoder(
                cfg,
                Arc::new(UnusedEncoder),
            )
            .expect("load state");
            let api = EnrollApi::new(Arc::new(EnrollSession::new(state, None)));
            if api.session.state.config.output.csv_dir_path.is_some() {
                assert!(outputs(&api).is_err(), "fixture must fail its output build");
            }
            assert_shutdown_accepted(api).await;
        }
    }

    async fn assert_shutdown_accepted(api: Arc<EnrollApi>) {
        let stopped = shutdown(State(api.clone())).await;
        assert_eq!(
            stopped.status(),
            StatusCode::ACCEPTED,
            "shutdown acknowledges the request, not output success"
        );
        let body = axum::body::to_bytes(stopped.into_body(), 1024)
            .await
            .expect("read shutdown response");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("shutdown JSON"),
            serde_json::json!({"status":"shutting_down"})
        );
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            api.shutdown_notify.notified(),
        )
        .await
        .expect("shutdown notification retained");
    }
}
