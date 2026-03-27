//! `meld enroll` command — single-pool entity resolution server.

use std::path::Path;
use std::process;

use tracing::{info, warn};

/// Start the enroll-mode HTTP server.
pub fn cmd_enroll(config_path: &Path, port: u16) {
    // 1. Load enroll config
    let cfg = match crate::config::load_enroll_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // 2. Load live match state (enroll path)
    let mut state = match crate::state::LiveMatchState::load(cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load enroll state: {}", e);
            process::exit(1);
        }
    };

    // 3. Start tokio runtime and run server
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        // Initialise the encoding coordinator inside the runtime.
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

        let session = std::sync::Arc::new(crate::session::Session::new(state.clone(), hook_tx));

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

        // No crossmap flusher — enroll mode has no crossmap.

        // Start HTTP server
        info!(port, "starting enroll server");
        if let Err(e) = crate::api::server::start_server(session.clone(), port).await {
            eprintln!("Server error: {}", e);
            process::exit(1);
        }

        // Shutdown sequence
        info!("running shutdown sequence");

        // Flush WAL
        if let Err(e) = state.wal.flush() {
            warn!(error = %e, "wal flush failed");
        }

        // Compact WAL
        let id_field = &state.config.datasets.a.id_field;
        if let Err(e) = state.wal.compact(id_field, id_field) {
            warn!(error = %e, "wal compact failed");
        }

        // Save combined embedding index caches
        if let Err(e) = state.save_combined_index_caches() {
            warn!(error = %e, "combined index cache save failed");
        }

        let sess = session.as_ref();
        let uptime_s = format!("{:.0}", sess.start_time.elapsed().as_secs_f64());
        let enrollments = sess.upsert_count.load(std::sync::atomic::Ordering::Relaxed);
        info!(uptime_s, enrollments, "shutdown complete");
    });
}
