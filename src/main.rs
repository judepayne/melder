use std::path::{Path, PathBuf};
use std::process;

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
        /// Only clear index files, keep embeddings
        #[arg(long)]
        index_only: bool,
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

fn init_tracing(log_format: &str) {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("melder=info"));

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

fn main() {
    let cli = Cli::parse();
    init_tracing(&cli.log_format);

    match cli.command {
        Commands::Validate { config } => match melder::config::load_config(&config) {
            Ok(cfg) => {
                println!("Config valid: job={:?}", cfg.job.name);
                println!(
                    "  datasets: A={}, B={}",
                    cfg.datasets.a.path, cfg.datasets.b.path
                );
                println!("  match_fields: {} fields", cfg.match_fields.len());
                println!(
                    "  thresholds: auto_match={}, review_floor={}",
                    cfg.thresholds.auto_match, cfg.thresholds.review_floor
                );
                println!("  blocking: enabled={}", cfg.blocking.enabled);
                let cand_has_config = cfg.candidates.field_a.is_some()
                    || cfg.candidates.field_b.is_some()
                    || cfg.candidates.method.is_some();
                let cand_enabled = cfg.candidates.enabled.unwrap_or(cand_has_config);
                if cand_enabled {
                    println!(
                        "  candidates: {}/{}  method={}  n={}",
                        cfg.candidates.field_a.as_deref().unwrap_or("?"),
                        cfg.candidates.field_b.as_deref().unwrap_or("?"),
                        cfg.candidates.method.as_deref().unwrap_or("?"),
                        cfg.candidates.n.unwrap_or(10),
                    );
                } else {
                    println!("  candidates: disabled");
                }
                if let Some(pool) = cfg.performance.encoder_pool_size {
                    println!("  encoder_pool_size: {}", pool);
                }
                if let Some(w) = cfg.performance.workers {
                    println!("  workers: {}", w);
                }
                println!("  required_fields_a: {:?}", cfg.required_fields_a);
                println!("  required_fields_b: {:?}", cfg.required_fields_b);
            }
            Err(e) => {
                eprintln!("Config error: {}", e);
                process::exit(1);
            }
        },
        Commands::Run {
            config,
            dry_run,
            verbose,
            limit,
        } => cmd_run(&config, dry_run, verbose, limit),
        Commands::Serve {
            config,
            port,
            socket: _socket,
        } => cmd_serve(&config, port),
        Commands::Tune { config, verbose } => cmd_tune(&config, verbose),
        Commands::Cache { action } => match action {
            CacheAction::Build { config } => cmd_cache_build(&config),
            CacheAction::Status { config } => cmd_cache_status(&config),
            CacheAction::Clear { config, index_only } => cmd_cache_clear(&config, index_only),
        },
        Commands::Review { action } => match action {
            ReviewAction::List { config } => cmd_review_list(&config),
            ReviewAction::Import { config, file } => cmd_review_import(&config, &file),
        },
        Commands::Crossmap { action } => match action {
            CrossmapAction::Stats { config } => cmd_crossmap_stats(&config),
            CrossmapAction::Export { config, out } => cmd_crossmap_export(&config, &out),
            CrossmapAction::Import { config, file } => cmd_crossmap_import(&config, &file),
        },
    }
}

// ---------------------------------------------------------------------------
// Serve command
// ---------------------------------------------------------------------------

fn cmd_serve(config_path: &Path, port: u16) {
    // 1. Load config
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // 2. Load live match state
    let state = match melder::state::LiveMatchState::load(cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load live state: {}", e);
            process::exit(1);
        }
    };

    // 3. Create session
    let session = std::sync::Arc::new(melder::session::Session::new(state.clone()));

    // 4. Start tokio runtime and run server
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
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
        if let Err(e) = melder::api::server::start_server(session.clone(), port).await {
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

        // Save field vector caches
        if let Err(e) = state.save_field_vecs_caches() {
            eprintln!("Warning: field vector cache save failed: {}", e);
        }

        let sess = session.as_ref();
        eprintln!(
            "Shutdown complete. Uptime: {:.0}s, upserts: {}, matches: {}",
            sess.start_time.elapsed().as_secs_f64(),
            sess.upsert_count.load(std::sync::atomic::Ordering::Relaxed),
            sess.match_count.load(std::sync::atomic::Ordering::Relaxed),
        );
    });
}

// ---------------------------------------------------------------------------
// Run command
// ---------------------------------------------------------------------------

fn cmd_run(config_path: &Path, dry_run: bool, verbose: bool, limit: Option<usize>) {
    // 1. Load and validate config
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    if verbose {
        eprintln!("Job: {} ({})", cfg.job.name, cfg.job.description);
        eprintln!(
            "Datasets: A={} B={}",
            cfg.datasets.a.path, cfg.datasets.b.path
        );
        eprintln!(
            "Thresholds: auto_match={}, review_floor={}",
            cfg.thresholds.auto_match, cfg.thresholds.review_floor
        );
    }

    // 2. Load state (batch mode: don't pre-load B into state — batch engine loads it)
    let opts = melder::state::LoadOptions {
        load_b: false,
        batch_mode: true,
    };

    let state = match melder::state::state::load_state(cfg, &opts) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load state: {}", e);
            process::exit(1);
        }
    };

    // 3. Load crossmap
    let crossmap_path = state
        .config
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");
    let mut crossmap = match melder::crossmap::CrossMap::load(
        Path::new(crossmap_path),
        &state.config.cross_map.a_id_field,
        &state.config.cross_map.b_id_field,
    ) {
        Ok(cm) => {
            if cm.len() > 0 {
                eprintln!("Loaded crossmap: {} existing pairs", cm.len());
            }
            cm
        }
        Err(e) => {
            eprintln!("Warning: failed to load crossmap ({}), starting fresh", e);
            melder::crossmap::CrossMap::new()
        }
    };

    // 4. Dry-run: report what would be processed, exit
    if dry_run {
        // Count B records to show what would be processed
        let b_count = match melder::data::load_dataset(
            Path::new(&state.config.datasets.b.path),
            &state.config.datasets.b.id_field,
            &state.config.required_fields_b,
            state.config.datasets.b.format.as_deref(),
        ) {
            Ok((_, ids)) => ids.len(),
            Err(e) => {
                eprintln!("Failed to load B dataset: {}", e);
                process::exit(1);
            }
        };
        let effective = limit.map(|l| l.min(b_count)).unwrap_or(b_count);
        let skippable = crossmap.len().min(effective);

        println!("Dry run:");
        println!("  A records:       {}", state.records_a.len());
        println!("  A field vectors: {}", state.field_vecs_a.len());
        println!("  B records:       {}", b_count);
        println!(
            "  Limit:           {}",
            limit.map(|l| l.to_string()).unwrap_or("none".into())
        );
        println!("  Already mapped:  {}", crossmap.len());
        println!("  Would process:   ~{}", effective - skippable);
        println!("  Output paths:");
        println!("    results:   {}", state.config.output.results_path);
        println!("    review:    {}", state.config.output.review_path);
        println!("    unmatched: {}", state.config.output.unmatched_path);
        return;
    }

    // 5. Run batch engine
    let result = match melder::batch::run_batch(
        &state.config,
        &state.records_a,
        &state.field_vecs_a,
        &state.encoder_pool,
        &mut crossmap,
        limit,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Batch matching failed: {}", e);
            process::exit(1);
        }
    };

    // 6. Write output files
    let results_path = Path::new(&state.config.output.results_path);
    let review_path = Path::new(&state.config.output.review_path);
    let unmatched_path = Path::new(&state.config.output.unmatched_path);

    if let Err(e) = melder::batch::write_results_csv(results_path, &result.matched, &state.config) {
        eprintln!("Failed to write results: {}", e);
        process::exit(1);
    }
    if let Err(e) = melder::batch::write_review_csv(review_path, &result.review, &state.config) {
        eprintln!("Failed to write review: {}", e);
        process::exit(1);
    }
    if let Err(e) = melder::batch::write_unmatched_csv(
        unmatched_path,
        &result.unmatched,
        &state.config.datasets.b.id_field,
    ) {
        eprintln!("Failed to write unmatched: {}", e);
        process::exit(1);
    }

    // 7. Save updated crossmap
    if let Err(e) = crossmap.save(
        Path::new(crossmap_path),
        &state.config.cross_map.a_id_field,
        &state.config.cross_map.b_id_field,
    ) {
        eprintln!("Warning: failed to save crossmap: {}", e);
    }

    // 8. Print summary
    let stats = &result.stats;
    println!();
    println!("Batch matching complete:");
    println!("  Total B records: {}", stats.total_b);
    println!("  Skipped (crossmap): {}", stats.skipped);
    println!("  Auto-matched: {}", stats.auto_matched);
    println!("  Review:       {}", stats.review_count);
    println!("  No match:     {}", stats.no_match);
    println!("  Elapsed:      {:.1}s", stats.elapsed_secs);
    if stats.elapsed_secs > 0.0 {
        let processed = stats.total_b - stats.skipped;
        println!(
            "  Throughput:   {:.0} records/sec",
            processed as f64 / stats.elapsed_secs
        );
    }
    println!();
    println!("Output files:");
    println!(
        "  results:   {} ({} rows)",
        state.config.output.results_path,
        result.matched.len()
    );
    println!(
        "  review:    {} ({} rows)",
        state.config.output.review_path,
        result.review.len()
    );
    println!(
        "  unmatched: {} ({} rows)",
        state.config.output.unmatched_path,
        result.unmatched.len()
    );

    if verbose {
        // Print crossmap status
        println!();
        println!(
            "Crossmap: {} total pairs (saved to {})",
            crossmap.len(),
            crossmap_path
        );
    }
}

// ---------------------------------------------------------------------------
// Tune command
// ---------------------------------------------------------------------------

fn cmd_tune(config_path: &Path, verbose: bool) {
    // 1. Load config
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let current_auto = cfg.thresholds.auto_match;
    let current_floor = cfg.thresholds.review_floor;

    if verbose {
        eprintln!(
            "Tune: {} — current thresholds: auto_match={}, review_floor={}",
            cfg.job.name, current_auto, current_floor
        );
    }

    // 2. Load state (batch mode)
    let opts = melder::state::LoadOptions {
        load_b: false,
        batch_mode: true,
    };

    let state = match melder::state::state::load_state(cfg, &opts) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load state: {}", e);
            process::exit(1);
        }
    };

    // 3. Run batch engine with a throwaway crossmap (no persistence)
    let mut crossmap = melder::crossmap::CrossMap::new();
    eprintln!("Running batch matching for score analysis...");
    let result = match melder::batch::run_batch(
        &state.config,
        &state.records_a,
        &state.field_vecs_a,
        &state.encoder_pool,
        &mut crossmap,
        None,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Batch matching failed: {}", e);
            process::exit(1);
        }
    };

    // 4. Collect all top-match scores
    let mut all_scores: Vec<f64> = Vec::new();
    // Collect per-field scores: field_key -> Vec<score>
    let mut field_scores: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();

    for mr in result.matched.iter().chain(result.review.iter()) {
        all_scores.push(mr.score);
        for fs in &mr.field_scores {
            let key = format!("{}_{}", fs.field_a, fs.field_b);
            field_scores.entry(key).or_default().push(fs.score);
        }
    }
    // Unmatched records have no match result score, but we should count them
    // They contribute 0.0 scores conceptually but no actual MatchResult exists

    all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("=== Score Distribution ===");
    println!();

    // Histogram: 10 buckets
    let mut buckets = [0usize; 10];
    for &s in &all_scores {
        let idx = if s >= 1.0 {
            9
        } else {
            (s * 10.0).floor() as usize
        };
        buckets[idx] += 1;
    }

    let max_count = *buckets.iter().max().unwrap_or(&1);
    let bar_width = 40;

    for i in 0..10 {
        let lo = i as f64 * 0.1;
        let hi = lo + 0.1;
        let label = if i == 9 {
            format!("[{:.1}-{:.1}]", lo, hi)
        } else {
            format!("[{:.1}-{:.1})", lo, hi)
        };

        let count = buckets[i];
        let bar_len = if max_count > 0 {
            (count as f64 / max_count as f64 * bar_width as f64).round() as usize
        } else {
            0
        };
        let bar: String = "#".repeat(bar_len);
        let pct = if !all_scores.is_empty() {
            count as f64 / all_scores.len() as f64 * 100.0
        } else {
            0.0
        };

        println!("  {:12} {:>5} ({:5.1}%) {}", label, count, pct, bar);
    }

    // 5. Per-field score statistics
    println!();
    println!("=== Per-Field Score Statistics ===");
    println!();
    println!(
        "  {:<40} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Field", "Min", "Max", "Mean", "Median", "StdDev"
    );
    println!("  {}", "-".repeat(76));

    let mut field_keys: Vec<&String> = field_scores.keys().collect();
    field_keys.sort();

    for key in &field_keys {
        let scores = field_scores.get(*key).unwrap();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
        let std_dev = variance.sqrt();

        println!(
            "  {:<40} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
            key, min, max, mean, median, std_dev
        );
    }

    // 6. Threshold analysis at current settings
    println!();
    println!("=== Threshold Analysis (current settings) ===");
    println!();

    let total_scored = all_scores.len();
    let total_with_unmatched = total_scored + result.unmatched.len();

    let auto_count = all_scores.iter().filter(|&&s| s >= current_auto).count();
    let review_count = all_scores
        .iter()
        .filter(|&&s| s >= current_floor && s < current_auto)
        .count();
    let below_floor = all_scores.iter().filter(|&&s| s < current_floor).count();
    let no_match = below_floor + result.unmatched.len();

    let pct = |n: usize| -> f64 {
        if total_with_unmatched > 0 {
            n as f64 / total_with_unmatched as f64 * 100.0
        } else {
            0.0
        }
    };

    println!(
        "  auto_match  >= {:.2}: {:>5} ({:.1}%)",
        current_auto,
        auto_count,
        pct(auto_count)
    );
    println!(
        "  review      >= {:.2}: {:>5} ({:.1}%)",
        current_floor,
        review_count,
        pct(review_count)
    );
    println!(
        "  no_match     < {:.2}: {:>5} ({:.1}%) (incl. {} with no candidates)",
        current_floor,
        no_match,
        pct(no_match),
        result.unmatched.len()
    );
    println!("  total:            {:>5}", total_with_unmatched);

    // 7. Suggested thresholds
    println!();
    println!("=== Suggested Thresholds ===");
    println!();

    if all_scores.is_empty() {
        println!("  (no scores available for threshold suggestions)");
    } else {
        let p50_idx = (all_scores.len() as f64 * 0.50).floor() as usize;
        let p90_idx = (all_scores.len() as f64 * 0.90).floor() as usize;
        let p50 = all_scores[p50_idx.min(all_scores.len() - 1)];
        let p90 = all_scores[p90_idx.min(all_scores.len() - 1)];

        println!("  Suggested auto_match  (90th percentile): {:.4}", p90);
        println!("  Suggested review_floor (50th percentile): {:.4}", p50);
        println!();
        if (p90 - current_auto).abs() > 0.05 || (p50 - current_floor).abs() > 0.05 {
            println!("  Consider adjusting thresholds based on score distribution.");
        } else {
            println!("  Current thresholds appear reasonable for this dataset.");
        }
    }
}

// ---------------------------------------------------------------------------
// Cache commands
// ---------------------------------------------------------------------------

fn cmd_cache_build(config_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let load_b = cfg.embeddings.b_index_cache.is_some();
    let opts = melder::state::LoadOptions {
        load_b,
        batch_mode: false,
    };

    let start = std::time::Instant::now();
    match melder::state::state::load_state(cfg, &opts) {
        Ok(state) => {
            let elapsed = start.elapsed();
            println!("Cache build complete:");
            println!(
                "  A-side: {} records, {} field vectors",
                state.records_a.len(),
                state.field_vecs_a.len()
            );
            if let Some(ref rb) = state.records_b {
                println!(
                    "  B-side: {} records, {} field vectors",
                    rb.len(),
                    state.field_vecs_b.as_ref().map(|fv| fv.len()).unwrap_or(0)
                );
            }
            println!("  A cache: {}", state.config.embeddings.a_index_cache);
            if let Some(ref b_cache) = state.config.embeddings.b_index_cache {
                println!("  B index cache: {}", b_cache);
            }
            println!("  Total time: {:.1}s", elapsed.as_secs_f64());
        }
        Err(e) => {
            eprintln!("Cache build failed: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_cache_status(config_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // Check A-side cache
    print_cache_status("A index", &cfg.embeddings.a_index_cache);

    // Check B-side cache
    if let Some(ref b_cache) = cfg.embeddings.b_index_cache {
        print_cache_status("B index", b_cache);
    }
}

fn print_cache_status(label: &str, path: &str) {
    let p = Path::new(path);
    if p.exists() {
        let meta = std::fs::metadata(p);
        let size = meta.map(|m| m.len()).unwrap_or(0);
        let size_str = if size > 1_048_576 {
            format!("{:.1} MB", size as f64 / 1_048_576.0)
        } else if size > 1024 {
            format!("{:.1} KB", size as f64 / 1024.0)
        } else {
            format!("{} bytes", size)
        };

        // Try to read header for record count
        let count = if path.ends_with(".index") {
            match std::fs::File::open(p) {
                Ok(mut f) => {
                    let mut buf = [0u8; 4];
                    if std::io::Read::read_exact(&mut f, &mut buf).is_ok() {
                        Some(u32::from_le_bytes(buf) as usize)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        if let Some(n) = count {
            println!("  {:<16} {} ({}, {} records)", label, "fresh", size_str, n);
        } else {
            println!("  {:<16} {} ({})", label, "exists", size_str);
        }
    } else {
        println!("  {:<16} {}", label, "missing");
    }
}

fn cmd_cache_clear(config_path: &Path, index_only: bool) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let _ = index_only; // no longer meaningful — kept for CLI compat
    let mut deleted = Vec::new();

    // Delete index cache files
    delete_if_exists(&cfg.embeddings.a_index_cache, &mut deleted);
    if let Some(ref b_cache) = cfg.embeddings.b_index_cache {
        delete_if_exists(b_cache, &mut deleted);
    }

    if deleted.is_empty() {
        println!("No cache files found to delete.");
    } else {
        for path in &deleted {
            println!("  deleted: {}", path);
        }
        println!("Cleared {} cache file(s).", deleted.len());
    }
}

// ---------------------------------------------------------------------------
// Review commands
// ---------------------------------------------------------------------------

fn cmd_review_list(config_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let review_path = Path::new(&cfg.output.review_path);
    if !review_path.exists() {
        println!("No review records (file not found: {})", cfg.output.review_path);
        return;
    }

    let mut rdr = match csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(review_path)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to read review CSV: {}", e);
            process::exit(1);
        }
    };

    let headers: Vec<String> = match rdr.headers() {
        Ok(h) => h.iter().map(|s| s.to_string()).collect(),
        Err(e) => {
            eprintln!("Failed to read headers: {}", e);
            process::exit(1);
        }
    };

    // Compute column widths
    let mut rows: Vec<Vec<String>> = Vec::new();
    for result in rdr.records() {
        match result {
            Ok(record) => {
                let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                rows.push(row);
            }
            Err(e) => {
                eprintln!("Warning: skipping malformed row: {}", e);
            }
        }
    }

    if rows.is_empty() {
        println!("No review records (file is empty: {})", cfg.output.review_path);
        return;
    }

    // Calculate column widths
    let num_cols = headers.len();
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in &rows {
        for (i, val) in row.iter().enumerate() {
            if i < num_cols {
                widths[i] = widths[i].max(val.len());
            }
        }
    }

    // Print header
    let header_line: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| format!("{:<width$}", h, width = widths[i]))
        .collect();
    println!("{}", header_line.join("  "));
    let sep: Vec<String> = widths.iter().map(|&w| "-".repeat(w)).collect();
    println!("{}", sep.join("  "));

    // Print rows
    for row in &rows {
        let line: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = if i < widths.len() { widths[i] } else { v.len() };
                format!("{:<width$}", v, width = w)
            })
            .collect();
        println!("{}", line.join("  "));
    }

    println!("\n{} review record(s)", rows.len());
}

fn cmd_review_import(config_path: &Path, decisions_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // Load crossmap
    let crossmap_path = cfg
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");
    let mut crossmap = match melder::crossmap::CrossMap::load(
        Path::new(crossmap_path),
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("Failed to load crossmap: {}", e);
            process::exit(1);
        }
    };
    let initial_count = crossmap.len();

    // Read decisions CSV: a_id, b_id, decision
    let mut rdr = match csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(decisions_path)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to read decisions file: {}", e);
            process::exit(1);
        }
    };

    let headers: Vec<String> = match rdr.headers() {
        Ok(h) => h.iter().map(|s| s.trim().to_string()).collect(),
        Err(e) => {
            eprintln!("Failed to read headers: {}", e);
            process::exit(1);
        }
    };

    let a_idx = headers.iter().position(|h| h == "a_id");
    let b_idx = headers.iter().position(|h| h == "b_id");
    let dec_idx = headers.iter().position(|h| h == "decision");

    let (a_idx, b_idx, dec_idx) = match (a_idx, b_idx, dec_idx) {
        (Some(a), Some(b), Some(d)) => (a, b, d),
        _ => {
            eprintln!(
                "Decisions CSV must have columns: a_id, b_id, decision. Found: {:?}",
                headers
            );
            process::exit(1);
        }
    };

    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut skipped = 0usize;

    for result in rdr.records() {
        let row = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: skipping malformed row: {}", e);
                skipped += 1;
                continue;
            }
        };

        let a_id = row.get(a_idx).unwrap_or("").trim();
        let b_id = row.get(b_idx).unwrap_or("").trim();
        let decision = row.get(dec_idx).unwrap_or("").trim().to_lowercase();

        if a_id.is_empty() || b_id.is_empty() {
            skipped += 1;
            continue;
        }

        match decision.as_str() {
            "accept" => {
                crossmap.add(a_id, b_id);
                accepted += 1;
            }
            "reject" => {
                rejected += 1;
            }
            other => {
                eprintln!(
                    "Warning: unknown decision {:?} for ({}, {}), skipping",
                    other, a_id, b_id
                );
                skipped += 1;
            }
        }
    }

    // Save updated crossmap
    if accepted > 0 {
        if let Err(e) = crossmap.save(
            Path::new(crossmap_path),
            &cfg.cross_map.a_id_field,
            &cfg.cross_map.b_id_field,
        ) {
            eprintln!("Failed to save crossmap: {}", e);
            process::exit(1);
        }
    }

    // Update review CSV: remove accepted/rejected pairs
    let review_path = Path::new(&cfg.output.review_path);
    if review_path.exists() && (accepted > 0 || rejected > 0) {
        // Build set of decided pairs
        // Re-read decisions to get the pairs
        let mut decided_pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        if let Ok(mut rdr2) = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(decisions_path)
        {
            for result in rdr2.records() {
                if let Ok(row) = result {
                    let a_id = row.get(a_idx).unwrap_or("").trim().to_string();
                    let b_id = row.get(b_idx).unwrap_or("").trim().to_string();
                    let dec = row.get(dec_idx).unwrap_or("").trim().to_lowercase();
                    if dec == "accept" || dec == "reject" {
                        decided_pairs.insert((a_id, b_id));
                    }
                }
            }
        }

        // Re-read review CSV and filter out decided pairs
        if let Ok(mut review_rdr) = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(review_path)
        {
            if let Ok(review_headers) = review_rdr.headers().cloned() {
                let review_a_idx = review_headers.iter().position(|h| h == "a_id");
                let review_b_idx = review_headers.iter().position(|h| h == "b_id");

                if let (Some(ra), Some(rb)) = (review_a_idx, review_b_idx) {
                    let mut remaining_rows: Vec<csv::StringRecord> = Vec::new();
                    for result in review_rdr.records() {
                        if let Ok(row) = result {
                            let a_id = row.get(ra).unwrap_or("").trim().to_string();
                            let b_id = row.get(rb).unwrap_or("").trim().to_string();
                            if !decided_pairs.contains(&(a_id, b_id)) {
                                remaining_rows.push(row);
                            }
                        }
                    }

                    // Rewrite review CSV
                    if let Ok(mut wtr) = csv::Writer::from_path(review_path) {
                        let _ = wtr.write_record(&review_headers);
                        for row in &remaining_rows {
                            let _ = wtr.write_record(row);
                        }
                        let _ = wtr.flush();
                    }
                }
            }
        }
    }

    println!("Review import complete:");
    println!("  Accepted: {} (added to crossmap)", accepted);
    println!("  Rejected: {} (removed from review)", rejected);
    if skipped > 0 {
        println!("  Skipped:  {}", skipped);
    }
    println!(
        "  Crossmap: {} -> {} pairs",
        initial_count,
        crossmap.len()
    );
}

// ---------------------------------------------------------------------------
// Crossmap commands
// ---------------------------------------------------------------------------

fn cmd_crossmap_stats(config_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // Load crossmap
    let crossmap_path = cfg
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");
    let crossmap = match melder::crossmap::CrossMap::load(
        Path::new(crossmap_path),
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("Failed to load crossmap: {}", e);
            process::exit(1);
        }
    };

    // Load datasets to get totals
    let a_count = match melder::data::load_dataset(
        Path::new(&cfg.datasets.a.path),
        &cfg.datasets.a.id_field,
        &cfg.required_fields_a,
        cfg.datasets.a.format.as_deref(),
    ) {
        Ok((records, _)) => records.len(),
        Err(e) => {
            eprintln!("Warning: failed to load A dataset: {}", e);
            0
        }
    };

    let b_count = match melder::data::load_dataset(
        Path::new(&cfg.datasets.b.path),
        &cfg.datasets.b.id_field,
        &cfg.required_fields_b,
        cfg.datasets.b.format.as_deref(),
    ) {
        Ok((records, _)) => records.len(),
        Err(e) => {
            eprintln!("Warning: failed to load B dataset: {}", e);
            0
        }
    };

    let total_pairs = crossmap.len();
    let a_pct = if a_count > 0 {
        total_pairs as f64 / a_count as f64 * 100.0
    } else {
        0.0
    };
    let b_pct = if b_count > 0 {
        total_pairs as f64 / b_count as f64 * 100.0
    } else {
        0.0
    };

    println!("Crossmap statistics:");
    println!("  File:     {}", crossmap_path);
    println!("  Pairs:    {}", total_pairs);
    println!(
        "  A coverage: {}/{} ({:.1}%)",
        total_pairs.min(a_count),
        a_count,
        a_pct
    );
    println!(
        "  B coverage: {}/{} ({:.1}%)",
        total_pairs.min(b_count),
        b_count,
        b_pct
    );
}

fn cmd_crossmap_export(config_path: &Path, out_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let crossmap_path = cfg
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");
    let crossmap = match melder::crossmap::CrossMap::load(
        Path::new(crossmap_path),
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("Failed to load crossmap: {}", e);
            process::exit(1);
        }
    };

    if let Err(e) = crossmap.save(
        out_path,
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        eprintln!("Failed to export crossmap: {}", e);
        process::exit(1);
    }

    println!(
        "Exported {} pairs to {}",
        crossmap.len(),
        out_path.display()
    );
}

fn cmd_crossmap_import(config_path: &Path, import_path: &Path) {
    let cfg = match melder::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let crossmap_path = cfg
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");

    // Load existing crossmap
    let mut crossmap = match melder::crossmap::CrossMap::load(
        Path::new(crossmap_path),
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("Failed to load crossmap: {}", e);
            process::exit(1);
        }
    };
    let initial_count = crossmap.len();

    // Load import file
    let import_cm = match melder::crossmap::CrossMap::load(
        import_path,
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("Failed to read import file: {}", e);
            process::exit(1);
        }
    };

    let import_count = import_cm.len();
    for (a_id, b_id) in import_cm.iter() {
        crossmap.add(a_id, b_id);
    }

    // Save
    if let Err(e) = crossmap.save(
        Path::new(crossmap_path),
        &cfg.cross_map.a_id_field,
        &cfg.cross_map.b_id_field,
    ) {
        eprintln!("Failed to save crossmap: {}", e);
        process::exit(1);
    }

    println!(
        "Imported {} pairs from {}",
        import_count,
        import_path.display()
    );
    println!(
        "Crossmap: {} -> {} pairs",
        initial_count,
        crossmap.len()
    );
}

fn delete_if_exists(path: &str, deleted: &mut Vec<String>) {
    let p = Path::new(path);
    if p.exists() {
        if let Err(e) = std::fs::remove_file(p) {
            eprintln!("  warning: failed to delete {}: {}", path, e);
        } else {
            deleted.push(path.to_string());
        }
    }
}
