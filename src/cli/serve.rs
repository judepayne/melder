//! `meld serve` command.

use std::path::Path;
use std::process;

use tracing::{info, warn};

/// Start the live-mode HTTP server.
pub fn cmd_serve(config_path: &Path, port: u16) {
    // 1. Load config
    let cfg = super::load_config_or_exit(config_path);

    // Warn if mmap mode is configured — it is not safe for live mode because
    // upserts write to the index and will fail on a read-only mmap'd index.
    if cfg.performance.vector_index_mode.as_deref() == Some("mmap") {
        warn!("mmap vector index is read-only; upserts will fail — use mmap only with meld run");
    }

    // 2. Load live match state
    let mut state = match crate::state::LiveMatchState::load(cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load live state: {}", e);
            process::exit(1);
        }
    };

    // 3. Start tokio runtime and run server
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        // Initialise the encoding coordinator inside the runtime so that
        // tokio::spawn is available.
        if let Some(s) = std::sync::Arc::get_mut(&mut state) {
            s.init_coordinator();
        }

        // Spawn hook writer task if configured.
        let hook_tx = if let Some(ref cmd) = state.config.hooks.command {
            let (tx, rx) = tokio::sync::mpsc::channel(1000);
            let cmd = cmd.clone();
            tokio::spawn(async move {
                crate::hooks::writer::run(cmd, rx).await;
            });
            Some(tx)
        } else {
            None
        };

        // Create session (must happen after init_coordinator so Arc is shared
        // only after the coordinator is set up).
        let session = std::sync::Arc::new(crate::session::Session::new(state.clone(), hook_tx));

        // Start background crossmap flusher
        let flush_state = state.clone();
        let flush_secs = state.config.live.crossmap_flush_secs.unwrap_or(5);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(flush_secs));
            loop {
                interval.tick().await;
                if let Err(e) = flush_state.flush_crossmap() {
                    warn!(error = %e, "crossmap flush failed");
                }
            }
        });

        // Start background WAL flusher
        let wal_state = state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                if let Err(e) = wal_state.wal.flush() {
                    warn!(error = %e, "wal flush failed");
                }
            }
        });

        // Start HTTP server
        if let Err(e) = crate::api::server::start_server(session.clone(), port).await {
            eprintln!("Server error: {}", e);
            process::exit(1);
        }

        // Shutdown sequence
        info!("shutdown sequence started");

        // Flush WAL
        if let Err(e) = state.wal.flush() {
            warn!(error = %e, "wal flush failed");
        }

        // Compact WAL
        let a_id_field = &state.config.datasets.a.id_field;
        let b_id_field = &state.config.datasets.b.id_field;
        if let Err(e) = state.wal.compact(a_id_field, b_id_field) {
            warn!(error = %e, "wal compact failed");
        }
        state.wal.cleanup_old_files();

        // Final crossmap flush
        state.mark_crossmap_dirty(); // force flush
        if let Err(e) = state.flush_crossmap() {
            warn!(error = %e, "final crossmap flush failed");
        }

        // Save combined embedding index caches
        if let Err(e) = state.save_combined_index_caches() {
            warn!(error = %e, "combined index cache save failed");
        }

        let sess = session.as_ref();
        info!(
            uptime_s = format!("{:.0}", sess.start_time.elapsed().as_secs_f64()),
            additions = sess.upsert_count.load(std::sync::atomic::Ordering::Relaxed),
            matches = sess.match_count.load(std::sync::atomic::Ordering::Relaxed),
            "shutdown complete",
        );
    });
}
