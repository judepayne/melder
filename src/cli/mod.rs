//! CLI command implementations extracted from `main.rs`.

pub mod cache;
pub mod crossmap;
pub mod enroll;
pub mod export;
pub mod review;
pub mod run;
pub mod serve;
pub mod tune;
pub mod validate;

/// Load config or print an error and exit.
///
/// Eliminates the repeated `match load_config { Ok => ..., Err => eprintln + exit(1) }`
/// boilerplate across CLI commands.
pub fn load_config_or_exit(config_path: &std::path::Path) -> crate::config::Config {
    match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Initialise the `tracing` subscriber based on log format.
pub fn init_tracing(log_format: &str) {
    use tracing_subscriber::EnvFilter;

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("melder=info"));

    match log_format {
        "json" => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .json()
                .with_target(true)
                .init();
        }
        _ => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .init();
        }
    }
}
