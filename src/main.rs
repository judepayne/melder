//! CLI entry point for the `meld` binary.

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
        /// Bind address (default: 127.0.0.1, use 0.0.0.0 for all interfaces)
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
    },
    /// Start the enroll-mode HTTP server (single-pool entity resolution)
    Enroll {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// HTTP port
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Bind address (default: 127.0.0.1, use 0.0.0.0 for all interfaces)
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
    },
    /// Tune thresholds — score distribution, overlap analysis, and accuracy metrics
    Tune {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
        /// Skip the scoring pipeline, re-analyse cached output files
        #[arg(long)]
        no_run: bool,
        /// Histogram bucket width (default: 0.04)
        #[arg(long, default_value_t = 0.04)]
        bucket_width: f64,
        /// Lower bound of display range (default: auto)
        #[arg(long)]
        min_score: Option<f64>,
        /// Upper bound of display range (default: auto)
        #[arg(long)]
        max_score: Option<f64>,
        /// Maximum bar width in characters (default: 50)
        #[arg(long, default_value_t = 50)]
        bar_width: usize,
        /// Max records to show per population in overlap zone (default: 5)
        #[arg(long, default_value_t = 5)]
        overlap_limit: usize,
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
    /// Export live/enroll match log to output files (CSVs and/or SQLite DB)
    Export {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory for exported files
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
        Commands::Serve { config, port, bind } => {
            melder::cli::serve::cmd_serve(&config, port, &bind)
        }
        Commands::Enroll { config, port, bind } => {
            melder::cli::enroll::cmd_enroll(&config, port, &bind)
        }
        Commands::Tune {
            config,
            verbose,
            no_run,
            bucket_width,
            min_score,
            max_score,
            bar_width,
            overlap_limit,
        } => melder::cli::tune::cmd_tune(
            &config,
            verbose,
            no_run,
            bucket_width,
            min_score,
            max_score,
            bar_width,
            overlap_limit,
        ),
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
