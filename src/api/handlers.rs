//! HTTP request handlers for the live matching API.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use tracing::{info, warn};

use crate::models::{Record, Side};
use crate::session::Session;

/// Shared application state for axum handlers.
pub type AppState = Arc<Session>;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct AddRequest {
    pub record: HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct ConfirmRequest {
    pub a_id: String,
    pub b_id: String,
}

#[derive(Deserialize)]
pub struct BreakRequest {
    pub a_id: String,
    pub b_id: String,
}

#[derive(Deserialize)]
pub struct LookupParams {
    pub id: String,
    pub side: String,
}

#[derive(Deserialize)]
pub struct RemoveRequest {
    pub id: String,
}

#[derive(Deserialize)]
pub struct AddBatchRequest {
    pub records: Vec<HashMap<String, String>>,
}

#[derive(Deserialize)]
pub struct RemoveBatchRequest {
    pub ids: Vec<String>,
}

#[derive(Deserialize)]
pub struct QueryParams {
    pub id: String,
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn error_response(status: StatusCode, msg: &str) -> impl IntoResponse {
    (
        status,
        Json(serde_json::json!({ "error": msg })),
    )
}

fn json_ok(value: impl serde::Serialize) -> axum::response::Response {
    match serde_json::to_value(value) {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("serialization error: {}", e),
        )
        .into_response(),
    }
}

/// Run a blocking closure on `spawn_blocking` and map the result.
///
/// On `Ok(Ok(val))` -> 200 JSON. On `Ok(Err(e))` -> `err_status` JSON error.
/// On `Err(join_error)` -> 500 JSON error.
async fn run_blocking<F, T, E>(
    f: F,
    err_status: StatusCode,
) -> axum::response::Response
where
    F: FnOnce() -> Result<T, E> + Send + 'static,
    T: serde::Serialize + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(val)) => json_ok(val),
        Ok(Err(e)) => error_response(err_status, &e.to_string()).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

// ---------------------------------------------------------------------------
// A/B record handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/a/add
pub async fn add_a(
    State(session): State<AppState>,
    Json(body): Json<AddRequest>,
) -> axum::response::Response {
    let record: Record = body.record;
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.upsert_record(Side::A, record)).await {
        Ok(Ok(resp)) => {
            info!(side = "A", id = %resp.id, status = %resp.status, matches = resp.matches.len(), latency_ms = %t0.elapsed().as_millis(), "add");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "A", error = %e, "add failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/b/add
pub async fn add_b(
    State(session): State<AppState>,
    Json(body): Json<AddRequest>,
) -> axum::response::Response {
    let record: Record = body.record;
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.upsert_record(Side::B, record)).await {
        Ok(Ok(resp)) => {
            info!(side = "B", id = %resp.id, status = %resp.status, matches = resp.matches.len(), latency_ms = %t0.elapsed().as_millis(), "add");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "B", error = %e, "add failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/a/match
pub async fn match_a(
    State(session): State<AppState>,
    Json(body): Json<AddRequest>,
) -> axum::response::Response {
    let record: Record = body.record;
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.try_match(Side::A, record)).await {
        Ok(Ok(resp)) => {
            info!(side = "A", matches = resp.matches.len(), latency_ms = %t0.elapsed().as_millis(), "try_match");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "A", error = %e, "try_match failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/b/match
pub async fn match_b(
    State(session): State<AppState>,
    Json(body): Json<AddRequest>,
) -> axum::response::Response {
    let record: Record = body.record;
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.try_match(Side::B, record)).await {
        Ok(Ok(resp)) => {
            info!(side = "B", matches = resp.matches.len(), latency_ms = %t0.elapsed().as_millis(), "try_match");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "B", error = %e, "try_match failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/a/remove
pub async fn remove_a(
    State(session): State<AppState>,
    Json(body): Json<RemoveRequest>,
) -> axum::response::Response {
    let id = body.id;
    run_blocking(move || {
        let r = session.remove_record(Side::A, &id);
        if r.is_ok() { info!(side = "A", id = %id, "remove"); }
        else { warn!(side = "A", id = %id, "remove failed"); }
        r
    }, StatusCode::NOT_FOUND).await
}

/// POST /api/v1/b/remove
pub async fn remove_b(
    State(session): State<AppState>,
    Json(body): Json<RemoveRequest>,
) -> axum::response::Response {
    let id = body.id;
    run_blocking(move || {
        let r = session.remove_record(Side::B, &id);
        if r.is_ok() { info!(side = "B", id = %id, "remove"); }
        else { warn!(side = "B", id = %id, "remove failed"); }
        r
    }, StatusCode::NOT_FOUND).await
}

/// GET /api/v1/a/query?id=X
pub async fn query_a(
    State(session): State<AppState>,
    Query(params): Query<QueryParams>,
) -> axum::response::Response {
    let id = params.id;
    run_blocking(move || session.query_record(Side::A, &id), StatusCode::NOT_FOUND).await
}

/// GET /api/v1/b/query?id=X
pub async fn query_b(
    State(session): State<AppState>,
    Query(params): Query<QueryParams>,
) -> axum::response::Response {
    let id = params.id;
    run_blocking(move || session.query_record(Side::B, &id), StatusCode::NOT_FOUND).await
}

// ---------------------------------------------------------------------------
// CrossMap handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/crossmap/confirm
pub async fn crossmap_confirm(
    State(session): State<AppState>,
    Json(body): Json<ConfirmRequest>,
) -> axum::response::Response {
    let a_id = body.a_id;
    let b_id = body.b_id;
    let a_log = a_id.clone();
    let b_log = b_id.clone();
    run_blocking(move || {
        let r = session.confirm_match(&a_id, &b_id);
        if r.is_ok() { info!(a_id = %a_log, b_id = %b_log, "crossmap confirm"); }
        else { warn!(a_id = %a_log, b_id = %b_log, "crossmap confirm failed"); }
        r
    }, StatusCode::BAD_REQUEST).await
}

/// GET /api/v1/crossmap/lookup?id=X&side=a|b
pub async fn crossmap_lookup(
    State(session): State<AppState>,
    Query(params): Query<LookupParams>,
) -> axum::response::Response {
    let side = match params.side.to_lowercase().as_str() {
        "a" => Side::A,
        "b" => Side::B,
        _ => {
            return error_response(StatusCode::BAD_REQUEST, "side must be 'a' or 'b'")
                .into_response();
        }
    };
    let id = params.id;
    run_blocking(move || session.lookup_crossmap(&id, side), StatusCode::BAD_REQUEST).await
}

/// POST /api/v1/crossmap/break
pub async fn crossmap_break(
    State(session): State<AppState>,
    Json(body): Json<BreakRequest>,
) -> axum::response::Response {
    let a_id = body.a_id;
    let b_id = body.b_id;
    let a_log = a_id.clone();
    let b_log = b_id.clone();
    run_blocking(move || {
        let r = session.break_crossmap(&a_id, &b_id);
        if r.is_ok() { info!(a_id = %a_log, b_id = %b_log, "crossmap break"); }
        else { warn!(a_id = %a_log, b_id = %b_log, "crossmap break failed"); }
        r
    }, StatusCode::BAD_REQUEST).await
}

// ---------------------------------------------------------------------------
// Synchronous handlers
// ---------------------------------------------------------------------------

/// GET /api/v1/health
pub async fn health(State(session): State<AppState>) -> axum::response::Response {
    json_ok(session.health())
}

/// GET /api/v1/status
pub async fn status(State(session): State<AppState>) -> axum::response::Response {
    json_ok(session.status())
}

// ---------------------------------------------------------------------------
// Batch handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/a/add-batch
pub async fn add_batch_a(
    State(session): State<AppState>,
    Json(body): Json<AddBatchRequest>,
) -> axum::response::Response {
    let count = body.records.len();
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.upsert_batch(Side::A, body.records)).await {
        Ok(Ok(resp)) => {
            info!(side = "A", count = count, latency_ms = %t0.elapsed().as_millis(), "add-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "A", error = %e, "add-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/b/add-batch
pub async fn add_batch_b(
    State(session): State<AppState>,
    Json(body): Json<AddBatchRequest>,
) -> axum::response::Response {
    let count = body.records.len();
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.upsert_batch(Side::B, body.records)).await {
        Ok(Ok(resp)) => {
            info!(side = "B", count = count, latency_ms = %t0.elapsed().as_millis(), "add-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "B", error = %e, "add-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/a/match-batch
pub async fn match_batch_a(
    State(session): State<AppState>,
    Json(body): Json<AddBatchRequest>,
) -> axum::response::Response {
    let count = body.records.len();
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.match_batch(Side::A, body.records)).await {
        Ok(Ok(resp)) => {
            info!(side = "A", count = count, latency_ms = %t0.elapsed().as_millis(), "match-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "A", error = %e, "match-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/b/match-batch
pub async fn match_batch_b(
    State(session): State<AppState>,
    Json(body): Json<AddBatchRequest>,
) -> axum::response::Response {
    let count = body.records.len();
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.match_batch(Side::B, body.records)).await {
        Ok(Ok(resp)) => {
            info!(side = "B", count = count, latency_ms = %t0.elapsed().as_millis(), "match-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "B", error = %e, "match-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/a/remove-batch
pub async fn remove_batch_a(
    State(session): State<AppState>,
    Json(body): Json<RemoveBatchRequest>,
) -> axum::response::Response {
    let count = body.ids.len();
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.remove_batch(Side::A, body.ids)).await {
        Ok(Ok(resp)) => {
            info!(side = "A", count = count, latency_ms = %t0.elapsed().as_millis(), "remove-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "A", error = %e, "remove-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

/// POST /api/v1/b/remove-batch
pub async fn remove_batch_b(
    State(session): State<AppState>,
    Json(body): Json<RemoveBatchRequest>,
) -> axum::response::Response {
    let count = body.ids.len();
    let t0 = std::time::Instant::now();
    let sess = session.clone();
    match tokio::task::spawn_blocking(move || sess.remove_batch(Side::B, body.ids)).await {
        Ok(Ok(resp)) => {
            info!(side = "B", count = count, latency_ms = %t0.elapsed().as_millis(), "remove-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = "B", error = %e, "remove-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}
