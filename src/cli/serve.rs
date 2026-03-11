//! `meld serve` command.

use std::path::Path;
use std::process;

/// Start the live-mode HTTP server.
pub fn cmd_serve(config_path: &Path, port: u16) {
    // 1. Load config
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

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

        // Create session (must happen after init_coordinator so Arc is shared
        // only after the coordinator is set up).
        let session = std::sync::Arc::new(crate::session::Session::new(state.clone()));

        // Start background crossmap flusher
        let flush_state = state.clone();
        let flush_secs = state.config.live.crossmap_flush_secs.unwrap_or(5);
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(flush_secs));
            loop {
                interval.tick().await;
                if let Err(e) = flush_state.flush_crossmap() {
                    eprintln!("Crossmap flush error: {}", e);
                }
            }
        });

        // Start background WAL flusher
        let wal_state = state.clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                if let Err(e) = wal_state.wal.flush() {
                    eprintln!("WAL flush error: {}", e);
                }
            }
        });

        // Start HTTP server
        if let Err(e) = crate::api::server::start_server(session.clone(), port).await {
            eprintln!("Server error: {}", e);
            process::exit(1);
        }

        // Shutdown sequence
        eprintln!("Running shutdown sequence...");

        // Flush WAL
        if let Err(e) = state.wal.flush() {
            eprintln!("Warning: WAL flush failed: {}", e);
        }

        // Compact WAL
        let a_id_field = &state.config.datasets.a.id_field;
        let b_id_field = &state.config.datasets.b.id_field;
        if let Err(e) = state.wal.compact(a_id_field, b_id_field) {
            eprintln!("Warning: WAL compact failed: {}", e);
        }

        // Final crossmap flush
        state.mark_crossmap_dirty(); // force flush
        if let Err(e) = state.flush_crossmap() {
            eprintln!("Warning: final crossmap flush failed: {}", e);
        }

        // Save combined embedding index caches
        if let Err(e) = state.save_combined_index_caches() {
            eprintln!("Warning: combined index cache save failed: {}", e);
        }

        let sess = session.as_ref();
        eprintln!(
            "Shutdown complete. Uptime: {:.0}s, additions: {}, matches: {}",
            sess.start_time.elapsed().as_secs_f64(),
            sess.upsert_count.load(std::sync::atomic::Ordering::Relaxed),
            sess.match_count.load(std::sync::atomic::Ordering::Relaxed),
        );
    });
}
