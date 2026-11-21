//! `meld enroll` command — single-pool entity resolution server.

use std::path::Path;
use std::process;

use tracing::{info, warn};

/// Start the enroll-mode HTTP server.
pub fn cmd_enroll(config_path: &Path, port: u16, bind: &str) {
    // 0. Early port check — fail fast before expensive state loading.
    if let Err(e) = std::net::TcpListener::bind(format!("{bind}:{port}")) {
        eprintln!("Port {port} on {bind} is already in use: {e}");
        process::exit(1);
    }

    // 1. Load enroll config
    let cfg = match crate::config::load_enroll_job_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // 2. Load the enrollment pool.
    let mut state = match crate::state::enroll::EnrollState::load_job(cfg) {
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

        // Open scoring log if enabled.
        let (scoring_log_sender, _scoring_log_writer) = if state.config.scoring_log.enabled {
            let sl_dir = state.config.output.csv_dir_path.as_deref().unwrap_or(".");
            let sl_base =
                std::path::Path::new(sl_dir).join(format!("{}.scoring_log", state.config.job.name));
            let manifest = crate::output::OutputManifest::from_enroll_job(&state.config);
            let use_zstd = state.config.scoring_log.compression != "none";
            match crate::output::scoring_log::open_scoring_log(&sl_base, use_zstd, &manifest) {
                Ok((sender, writer)) => (Some(sender), Some(writer)),
                Err(e) => {
                    warn!(error = %e, "failed to open scoring log — continuing without");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        let session = std::sync::Arc::new(
            crate::session::enroll::EnrollSession::new(state.clone(), scoring_log_sender)
                .with_hooks(hook_tx),
        );

        // Start background WAL flusher
        let wal_state = state.clone();
        let wal_flusher = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                let state = wal_state.clone();
                match tokio::task::spawn_blocking(move || state.flush()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => warn!(error = %e, "wal flush failed"),
                    Err(e) => warn!(error = %e, "wal flush worker failed"),
                }
            }
        });

        // No crossmap flusher — enroll mode has no crossmap.

        // Start HTTP server
        info!(port, "starting enroll server");
        if let Err(e) = crate::api::server::start_enroll_server(session.clone(), port, bind).await {
            eprintln!("Server error: {}", e);
            process::exit(1);
        }

        // Shutdown sequence
        info!("running shutdown sequence");

        wal_flusher.abort();
        if let Err(e) = state.shutdown() {
            warn!(error = %e, "enroll shutdown flush failed");
        }
        if let Err(e) = state.compact() {
            warn!(error = %e, "wal compact failed");
        }

        info!("shutdown complete");

        // Drop session (and its ScoringLogSender) before the writer, so the
        // channel closes and the writer thread can drain and finalize.
        drop(session);
        // _scoring_log_writer drops here, joining the writer thread.
    });
}

#[cfg(all(test, feature = "usearch"))]
mod tests {
    #[test]
    fn enroll_benchmark_configs_load_as_single_pool_jobs() {
        for file in [
            "benchmarks/enroll/10kx10k_enroll3k_scoring_log/config.yaml",
            "benchmarks/enroll/10kx10k_enroll3k_usearch/warm/config.yaml",
            "benchmarks/accuracy/enroll_5k_inject1k/config.yaml",
        ] {
            let job = crate::config::load_enroll_job_config(std::path::Path::new(file))
                .unwrap_or_else(|error| panic!("invalid enrollment benchmark {file}: {error}"));
            assert!(
                !job.dataset.path.is_empty(),
                "benchmark {file} needs its seed pool"
            );
        }
    }
}
