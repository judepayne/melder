//! HTTP request handlers for the live matching API.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Deserialize;
use tracing::{info, warn};

use crate::error::SessionError;
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

#[derive(Deserialize)]
pub struct PaginationParams {
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct UnmatchedParams {
    pub include_records: Option<bool>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn side_str(side: Side) -> &'static str {
    match side {
        Side::A => "A",
        Side::B => "B",
    }
}

fn error_response(status: StatusCode, msg: &str) -> impl IntoResponse {
    (status, Json(serde_json::json!({ "error": msg })))
}

fn json_ok(value: impl serde::Serialize) -> axum::response::Response {
    (StatusCode::OK, Json(value)).into_response()
}

/// Run a blocking closure on `spawn_blocking` and map the result.
///
/// On `Ok(Ok(val))` -> 200 JSON. On `Ok(Err(e))` -> `err_status` JSON error.
/// On `Err(join_error)` -> 500 JSON error.
async fn run_blocking<F, T, E>(f: F, err_status: StatusCode) -> axum::response::Response
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
// Handler helpers — one per operation family, parameterised by Side
// ---------------------------------------------------------------------------

async fn upsert_handler(side: Side, session: AppState, record: Record) -> axum::response::Response {
    let t0 = std::time::Instant::now();
    let s = side_str(side);
    match tokio::task::spawn_blocking(move || session.upsert_record(side, record)).await {
        Ok(Ok(resp)) => {
            info!(side = s, id = %resp.id, status = %resp.status, matches = resp.matches.len(), latency_ms = %t0.elapsed().as_millis(), "add");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = s, error = %e, "add failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

async fn match_handler(side: Side, session: AppState, record: Record) -> axum::response::Response {
    let t0 = std::time::Instant::now();
    let s = side_str(side);
    match tokio::task::spawn_blocking(move || session.try_match(side, record)).await {
        Ok(Ok(resp)) => {
            info!(side = s, matches = resp.matches.len(), latency_ms = %t0.elapsed().as_millis(), "try_match");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = s, error = %e, "try_match failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

async fn remove_handler(side: Side, session: AppState, id: String) -> axum::response::Response {
    let s = side_str(side);
    run_blocking(
        move || {
            let r = session.remove_record(side, &id);
            if r.is_ok() {
                info!(side = s, id = %id, "remove");
            } else {
                warn!(side = s, id = %id, "remove failed");
            }
            r
        },
        StatusCode::NOT_FOUND,
    )
    .await
}

async fn query_handler(side: Side, session: AppState, id: String) -> axum::response::Response {
    run_blocking(
        move || session.query_record(side, &id),
        StatusCode::NOT_FOUND,
    )
    .await
}

async fn add_batch_handler(
    side: Side,
    session: AppState,
    records: Vec<Record>,
) -> axum::response::Response {
    let count = records.len();
    let t0 = std::time::Instant::now();
    let s = side_str(side);
    match tokio::task::spawn_blocking(move || session.upsert_batch(side, records)).await {
        Ok(Ok(resp)) => {
            info!(side = s, count, latency_ms = %t0.elapsed().as_millis(), "add-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = s, error = %e, "add-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

async fn match_batch_handler(
    side: Side,
    session: AppState,
    records: Vec<Record>,
) -> axum::response::Response {
    let count = records.len();
    let t0 = std::time::Instant::now();
    let s = side_str(side);
    match tokio::task::spawn_blocking(move || session.match_batch(side, records)).await {
        Ok(Ok(resp)) => {
            info!(side = s, count, latency_ms = %t0.elapsed().as_millis(), "match-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = s, error = %e, "match-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

async fn remove_batch_handler(
    side: Side,
    session: AppState,
    ids: Vec<String>,
) -> axum::response::Response {
    let count = ids.len();
    let t0 = std::time::Instant::now();
    let s = side_str(side);
    match tokio::task::spawn_blocking(move || session.remove_batch(side, ids)).await {
        Ok(Ok(resp)) => {
            info!(side = s, count, latency_ms = %t0.elapsed().as_millis(), "remove-batch");
            json_ok(resp)
        }
        Ok(Err(e)) => {
            warn!(side = s, error = %e, "remove-batch failed");
            error_response(StatusCode::BAD_REQUEST, &e.to_string()).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Public A/B record handlers — thin wrappers over the helpers above
// ---------------------------------------------------------------------------

pub async fn add_a(
    State(s): State<AppState>,
    Json(b): Json<AddRequest>,
) -> axum::response::Response {
    upsert_handler(Side::A, s, b.record).await
}
pub async fn add_b(
    State(s): State<AppState>,
    Json(b): Json<AddRequest>,
) -> axum::response::Response {
    upsert_handler(Side::B, s, b.record).await
}

pub async fn match_a(
    State(s): State<AppState>,
    Json(b): Json<AddRequest>,
) -> axum::response::Response {
    match_handler(Side::A, s, b.record).await
}
pub async fn match_b(
    State(s): State<AppState>,
    Json(b): Json<AddRequest>,
) -> axum::response::Response {
    match_handler(Side::B, s, b.record).await
}

pub async fn remove_a(
    State(s): State<AppState>,
    Json(b): Json<RemoveRequest>,
) -> axum::response::Response {
    remove_handler(Side::A, s, b.id).await
}
pub async fn remove_b(
    State(s): State<AppState>,
    Json(b): Json<RemoveRequest>,
) -> axum::response::Response {
    remove_handler(Side::B, s, b.id).await
}

pub async fn query_a(
    State(s): State<AppState>,
    Query(p): Query<QueryParams>,
) -> axum::response::Response {
    query_handler(Side::A, s, p.id).await
}
pub async fn query_b(
    State(s): State<AppState>,
    Query(p): Query<QueryParams>,
) -> axum::response::Response {
    query_handler(Side::B, s, p.id).await
}

// ---------------------------------------------------------------------------
// Public batch handlers
// ---------------------------------------------------------------------------

pub async fn add_batch_a(
    State(s): State<AppState>,
    Json(b): Json<AddBatchRequest>,
) -> axum::response::Response {
    add_batch_handler(Side::A, s, b.records).await
}
pub async fn add_batch_b(
    State(s): State<AppState>,
    Json(b): Json<AddBatchRequest>,
) -> axum::response::Response {
    add_batch_handler(Side::B, s, b.records).await
}

pub async fn match_batch_a(
    State(s): State<AppState>,
    Json(b): Json<AddBatchRequest>,
) -> axum::response::Response {
    match_batch_handler(Side::A, s, b.records).await
}
pub async fn match_batch_b(
    State(s): State<AppState>,
    Json(b): Json<AddBatchRequest>,
) -> axum::response::Response {
    match_batch_handler(Side::B, s, b.records).await
}

pub async fn remove_batch_a(
    State(s): State<AppState>,
    Json(b): Json<RemoveBatchRequest>,
) -> axum::response::Response {
    remove_batch_handler(Side::A, s, b.ids).await
}
pub async fn remove_batch_b(
    State(s): State<AppState>,
    Json(b): Json<RemoveBatchRequest>,
) -> axum::response::Response {
    remove_batch_handler(Side::B, s, b.ids).await
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
    run_blocking(
        move || {
            let r = session.confirm_match(&a_id, &b_id);
            if r.is_ok() {
                info!(a_id = %a_log, b_id = %b_log, "crossmap confirm");
            } else {
                warn!(a_id = %a_log, b_id = %b_log, "crossmap confirm failed");
            }
            r
        },
        StatusCode::BAD_REQUEST,
    )
    .await
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
    run_blocking(
        move || session.lookup_crossmap(&id, side),
        StatusCode::BAD_REQUEST,
    )
    .await
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
    run_blocking(
        move || {
            let r = session.break_crossmap(&a_id, &b_id);
            if r.is_ok() {
                info!(a_id = %a_log, b_id = %b_log, "crossmap break");
            } else {
                warn!(a_id = %a_log, b_id = %b_log, "crossmap break failed");
            }
            r
        },
        StatusCode::BAD_REQUEST,
    )
    .await
}

// ---------------------------------------------------------------------------
// Utility handlers
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
// Crossmap, unmatched, stats, and review handlers
// ---------------------------------------------------------------------------

/// GET /api/v1/crossmap/pairs?offset=0&limit=100
pub async fn crossmap_pairs(
    State(session): State<AppState>,
    Query(params): Query<PaginationParams>,
) -> axum::response::Response {
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit;
    run_blocking(
        move || Ok::<_, SessionError>(session.crossmap_pairs(offset, limit)),
        StatusCode::INTERNAL_SERVER_ERROR,
    )
    .await
}

/// GET /api/v1/a/unmatched or /api/v1/b/unmatched
async fn unmatched_handler(
    side: Side,
    session: AppState,
    params: UnmatchedParams,
) -> axum::response::Response {
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit;
    let include_records = params.include_records.unwrap_or(false);
    run_blocking(
        move || {
            Ok::<_, SessionError>(session.unmatched_records(side, offset, limit, include_records))
        },
        StatusCode::INTERNAL_SERVER_ERROR,
    )
    .await
}

pub async fn unmatched_a(
    State(s): State<AppState>,
    Query(p): Query<UnmatchedParams>,
) -> axum::response::Response {
    unmatched_handler(Side::A, s, p).await
}

pub async fn unmatched_b(
    State(s): State<AppState>,
    Query(p): Query<UnmatchedParams>,
) -> axum::response::Response {
    unmatched_handler(Side::B, s, p).await
}

/// GET /api/v1/crossmap/stats
pub async fn crossmap_stats(State(session): State<AppState>) -> axum::response::Response {
    run_blocking(
        move || Ok::<_, SessionError>(session.crossmap_stats()),
        StatusCode::INTERNAL_SERVER_ERROR,
    )
    .await
}

/// GET /api/v1/review/list?offset=0&limit=100
pub async fn review_list(
    State(session): State<AppState>,
    Query(params): Query<PaginationParams>,
) -> axum::response::Response {
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit;
    run_blocking(
        move || Ok::<_, SessionError>(session.review_list(offset, limit)),
        StatusCode::INTERNAL_SERVER_ERROR,
    )
    .await
}
