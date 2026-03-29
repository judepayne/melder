//! HTTP server setup: routing, middleware, binding.

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::trace::TraceLayer;

use tracing::info;

use crate::session::Session;

use super::handlers;

/// Maximum request body size (10 MB).
///
/// Batch endpoints accept up to 1,000 records, each of which is a
/// `HashMap<String, String>`. With typical record sizes of a few KB,
/// 10 MB is generous while still protecting against accidental or
/// malicious multi-GB payloads.
const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Build the axum Router with all API routes.
///
/// In enroll mode, only the `/enroll` endpoints and health/status are mounted.
/// In match mode, the full A/B/crossmap/review API is mounted.
pub fn build_router(session: Arc<Session>) -> Router {
    let router = if session.state.config.is_enroll_mode() {
        build_enroll_router()
    } else {
        build_match_router()
    };

    router
        // Health / status (always available)
        .route("/api/v1/health", get(handlers::health))
        .route("/api/v1/status", get(handlers::status))
        // Middleware
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
        .layer(CatchPanicLayer::new())
        .layer(TraceLayer::new_for_http())
        // State
        .with_state(session)
}

/// Routes for standard two-sided matching mode.
fn build_match_router() -> Router<Arc<Session>> {
    Router::new()
        // A-side endpoints
        .route("/api/v1/a/add", post(handlers::add_a))
        .route("/api/v1/a/remove", post(handlers::remove_a))
        .route("/api/v1/a/match", post(handlers::match_a))
        .route("/api/v1/a/query", get(handlers::query_a))
        // A-side batch endpoints
        .route("/api/v1/a/add-batch", post(handlers::add_batch_a))
        .route("/api/v1/a/match-batch", post(handlers::match_batch_a))
        .route("/api/v1/a/remove-batch", post(handlers::remove_batch_a))
        // B-side endpoints
        .route("/api/v1/b/add", post(handlers::add_b))
        .route("/api/v1/b/remove", post(handlers::remove_b))
        .route("/api/v1/b/match", post(handlers::match_b))
        .route("/api/v1/b/query", get(handlers::query_b))
        // B-side batch endpoints
        .route("/api/v1/b/add-batch", post(handlers::add_batch_b))
        .route("/api/v1/b/match-batch", post(handlers::match_batch_b))
        .route("/api/v1/b/remove-batch", post(handlers::remove_batch_b))
        // Backward-compat alias
        .route("/api/v1/match/b", post(handlers::match_b))
        // Unmatched endpoints
        .route("/api/v1/a/unmatched", get(handlers::unmatched_a))
        .route("/api/v1/b/unmatched", get(handlers::unmatched_b))
        // CrossMap endpoints
        .route("/api/v1/crossmap/confirm", post(handlers::crossmap_confirm))
        .route("/api/v1/crossmap/lookup", get(handlers::crossmap_lookup))
        .route("/api/v1/crossmap/break", post(handlers::crossmap_break))
        .route("/api/v1/crossmap/pairs", get(handlers::crossmap_pairs))
        .route("/api/v1/crossmap/stats", get(handlers::crossmap_stats))
        // Review endpoints
        .route("/api/v1/review/list", get(handlers::review_list))
}

/// Routes for single-pool enrollment mode.
fn build_enroll_router() -> Router<Arc<Session>> {
    Router::new()
        .route("/api/v1/enroll", post(handlers::enroll))
        .route("/api/v1/enroll-batch", post(handlers::enroll_batch))
        .route("/api/v1/enroll/remove", post(handlers::enroll_remove))
        .route("/api/v1/enroll/query", get(handlers::enroll_query))
        .route("/api/v1/enroll/count", get(handlers::enroll_count))
}

/// Start the HTTP server on a TCP port.
pub async fn start_server(
    session: Arc<Session>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = build_router(session);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    info!(port, "server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM for graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("shutdown signal received");
}
