use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// melder — record matching engine
#[derive(Parser)]
#[command(name = "meld", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log format: "pretty" (default) or "json"
    #[arg(long, global = true, default_value = "pretty")]
    log_format: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a config file
    Validate {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Run a batch matching job
    Run {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Dry run (validate + load data, don't match)
        #[arg(long)]
        dry_run: bool,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
        /// Limit number of B records to process
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Start the live-mode HTTP server
    Serve {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// HTTP port
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Unix socket path (alternative to port)
        #[cfg(unix)]
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Tune thresholds on labelled data
    Tune {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Cache management
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    /// Review queue management
    Review {
        #[command(subcommand)]
        action: ReviewAction,
    },
    /// Cross-map management
    Crossmap {
        #[command(subcommand)]
        action: CrossmapAction,
    },
    /// Export live-mode state to CSV files
    Export {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory for exported CSV files
        #[arg(short, long)]
        out_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Build embedding caches
    Build {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Show cache status
    Status {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Clear caches
    Clear {
        #[arg(short, long)]
        config: PathBuf,
        /// Delete ALL cache files (including current). Default: only delete stale files.
        #[arg(long)]
        all: bool,
    },
}

#[derive(Subcommand)]
enum ReviewAction {
    /// List pending reviews
    List {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Import review decisions
    Import {
        #[arg(short, long)]
        config: PathBuf,
        /// Path to decisions file
        #[arg(short, long)]
        file: PathBuf,
    },
}

#[derive(Subcommand)]
enum CrossmapAction {
    /// Show cross-map statistics
    Stats {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Export cross-map to CSV
    Export {
        #[arg(short, long)]
        config: PathBuf,
        /// Output path
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Import cross-map from CSV
    Import {
        #[arg(short, long)]
        config: PathBuf,
        /// Input file
        #[arg(short, long)]
        file: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    melder::cli::init_tracing(&cli.log_format);

    match cli.command {
        Commands::Validate { config } => {
            melder::cli::validate::cmd_validate(&config);
        }
        Commands::Run {
            config,
            dry_run,
            verbose,
            limit,
        } => melder::cli::run::cmd_run(&config, dry_run, verbose, limit),
        Commands::Serve {
            config,
            port,
            #[cfg(unix)]
                socket: _socket,
        } => melder::cli::serve::cmd_serve(&config, port),
        Commands::Tune { config, verbose } => melder::cli::tune::cmd_tune(&config, verbose),
        Commands::Cache { action } => match action {
            CacheAction::Build { config } => melder::cli::cache::cmd_cache_build(&config),
            CacheAction::Status { config } => melder::cli::cache::cmd_cache_status(&config),
            CacheAction::Clear { config, all } => melder::cli::cache::cmd_cache_clear(&config, all),
        },
        Commands::Review { action } => match action {
            ReviewAction::List { config } => melder::cli::review::cmd_review_list(&config),
            ReviewAction::Import { config, file } => {
                melder::cli::review::cmd_review_import(&config, &file)
            }
        },
        Commands::Crossmap { action } => match action {
            CrossmapAction::Stats { config } => melder::cli::crossmap::cmd_crossmap_stats(&config),
            CrossmapAction::Export { config, out } => {
                melder::cli::crossmap::cmd_crossmap_export(&config, &out)
            }
            CrossmapAction::Import { config, file } => {
                melder::cli::crossmap::cmd_crossmap_import(&config, &file)
            }
        },
        Commands::Export { config, out_dir } => melder::cli::export::cmd_export(&config, &out_dir),
    }
}
