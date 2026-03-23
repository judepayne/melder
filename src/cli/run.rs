//! `meld run` command.

use std::path::Path;
use std::process;
use std::sync::Arc;

use crate::crossmap::CrossMapOps;
use crate::models::Side;

/// Run a batch matching job.
pub fn cmd_run(config_path: &Path, dry_run: bool, verbose: bool, limit: Option<usize>) {
    // 1. Load and validate config
    let cfg = match crate::config::load_config(config_path) {
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

    // Dispatch to SQLite or in-memory path
    if cfg.batch.db_path.is_some() {
        cmd_run_sqlite(cfg, dry_run, verbose, limit);
    } else {
        cmd_run_memory(cfg, dry_run, verbose, limit);
    }
}

// ---

/// In-memory batch path (existing behavior).
fn cmd_run_memory(cfg: crate::config::Config, dry_run: bool, verbose: bool, limit: Option<usize>) {
    // Load state: A records + A combined index + encoder pool
    let opts = crate::state::LoadOptions {
        load_b: false,
        batch_mode: true,
    };

    let state = match crate::state::state::load_state(cfg, &opts) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load state: {}", e);
            process::exit(1);
        }
    };

    // Load crossmap
    let crossmap_path = state
        .config
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");
    let crossmap = load_crossmap(crossmap_path, &state.config);

    // Dry-run
    if dry_run {
        let b_count = match crate::data::load_dataset(
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
        print_dry_run(
            &state.config,
            state.store.len(Side::A),
            b_count,
            &crossmap,
            limit,
        );
        return;
    }

    // Load B records and insert into store + build B embedding index
    let b_start = std::time::Instant::now();
    let (b_records_map, b_ids) = match crate::data::load_dataset(
        Path::new(&state.config.datasets.b.path),
        &state.config.datasets.b.id_field,
        &state.config.required_fields_b,
        state.config.datasets.b.format.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load B dataset: {}", e);
            process::exit(1);
        }
    };
    eprintln!(
        "Loaded {} B records in {:.1}s",
        b_records_map.len(),
        b_start.elapsed().as_secs_f64()
    );

    // Build B combined embedding index (if configured)
    let combined_index_b = match crate::vectordb::build_or_load_combined_index(
        &state.config.vector_backend,
        state.config.embeddings.b_cache_dir.as_deref(),
        &b_records_map,
        &b_ids,
        &state.config,
        false,
        &state.encoder_pool,
        false,
    ) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln!("Failed to build B embedding index: {}", e);
            process::exit(1);
        }
    };

    // Insert B records into the store
    for (id, rec) in &b_records_map {
        state.store.insert(Side::B, id, rec);
    }
    drop(b_records_map); // free memory

    // Run batch engine
    let result = match crate::batch::run_batch(
        &state.config,
        state.store.as_ref(),
        state.combined_index_a.as_deref(),
        combined_index_b.as_deref(),
        &crossmap,
        limit,
        false,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Batch matching failed: {}", e);
            process::exit(1);
        }
    };

    // Write output + save crossmap + print summary
    write_outputs(&state.config, &result);
    save_crossmap(&crossmap, crossmap_path, &state.config);
    print_summary(&state.config, &result, &crossmap, crossmap_path, verbose);
}

/// SQLite-backed batch path for large datasets.
fn cmd_run_sqlite(cfg: crate::config::Config, dry_run: bool, verbose: bool, limit: Option<usize>) {
    let db_path_str = cfg.batch.db_path.as_ref().unwrap().clone();
    let db_path = Path::new(&db_path_str);

    // Always fresh: delete existing DB
    for suffix in &["", "-wal", "-shm"] {
        let p = format!("{}{}", db_path_str, suffix);
        let _ = std::fs::remove_file(&p);
    }

    // Pool config: default read pool to num_cpus for Rayon parallelism
    let default_pool_size = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4);
    let pool_cfg = crate::store::sqlite::SqlitePoolConfig {
        writer_cache_kb: cfg.batch.sqlite_cache_mb.unwrap_or(64) * 1024,
        read_pool_size: cfg.batch.sqlite_read_pool_size.unwrap_or(default_pool_size),
        reader_cache_kb: cfg.batch.sqlite_pool_worker_cache_mb.unwrap_or(128) * 1024,
    };

    eprintln!(
        "SQLite batch mode: {} (readers: {}, cache: {}MB/reader)",
        db_path_str,
        pool_cfg.read_pool_size,
        pool_cfg.reader_cache_kb / 1024
    );

    let (sqlite_store, sqlite_crossmap, _writer) = match crate::store::sqlite::open_sqlite(
        db_path,
        &cfg.blocking,
        Some(pool_cfg),
        &cfg.required_fields_a,
        &cfg.required_fields_b,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open SQLite: {}", e);
            process::exit(1);
        }
    };

    // Stream + bulk-load A records (only one chunk in memory at a time)
    let a_path = cfg.datasets.a.path.clone();
    let a_id_field = cfg.datasets.a.id_field.clone();
    let a_required = cfg.required_fields_a.clone();
    let a_format = cfg.datasets.a.format.clone();
    let a_count = match sqlite_store.bulk_load(Side::A, |cb| {
        crate::data::stream_dataset(
            Path::new(&a_path),
            &a_id_field,
            &a_required,
            a_format.as_deref(),
            10_000,
            cb,
        )
    }) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load A dataset: {}", e);
            process::exit(1);
        }
    };

    // Stream + bulk-load B records
    let b_path = cfg.datasets.b.path.clone();
    let b_id_field = cfg.datasets.b.id_field.clone();
    let b_required = cfg.required_fields_b.clone();
    let b_format = cfg.datasets.b.format.clone();
    let b_count = match sqlite_store.bulk_load(Side::B, |cb| {
        crate::data::stream_dataset(
            Path::new(&b_path),
            &b_id_field,
            &b_required,
            b_format.as_deref(),
            10_000,
            cb,
        )
    }) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load B dataset: {}", e);
            process::exit(1);
        }
    };

    // Load crossmap
    let crossmap_path = cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv");
    let mem_crossmap = load_crossmap(crossmap_path, &cfg);
    // Import into SqliteCrossMap
    for (a_id, b_id) in mem_crossmap.pairs() {
        sqlite_crossmap.add(&a_id, &b_id);
    }
    if !mem_crossmap.is_empty() {
        eprintln!("Imported {} crossmap pairs into SQLite", mem_crossmap.len());
    }
    drop(mem_crossmap);

    // Dry-run
    if dry_run {
        print_dry_run(&cfg, a_count, b_count, &sqlite_crossmap, limit);
        cleanup_sqlite(&db_path_str);
        return;
    }

    let store: Arc<dyn crate::store::RecordStore> = Arc::new(sqlite_store);

    // No embedding indices for SQLite batch (BM25-only first cut)
    let combined_index_a: Option<&dyn crate::vectordb::VectorDB> = None;
    let combined_index_b: Option<&dyn crate::vectordb::VectorDB> = None;

    // Run batch engine
    let result = match crate::batch::run_batch(
        &cfg,
        store.as_ref(),
        combined_index_a,
        combined_index_b,
        &sqlite_crossmap,
        limit,
        false,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Batch matching failed: {}", e);
            process::exit(1);
        }
    };

    // Write output + save crossmap + print summary
    write_outputs(&cfg, &result);
    // Export SQLite crossmap back to CSV
    let pairs = sqlite_crossmap.pairs();
    let save_cm = crate::crossmap::MemoryCrossMap::new();
    for (a_id, b_id) in &pairs {
        save_cm.add(a_id, b_id);
    }
    save_crossmap(&save_cm, crossmap_path, &cfg);
    print_summary(&cfg, &result, &sqlite_crossmap, crossmap_path, verbose);

    // Drop all SQLite connections before deleting the file
    drop(sqlite_crossmap);
    drop(store);
    cleanup_sqlite(&db_path_str);
}

// ── Shared helpers ──────────────────────────────────────────────────────────

fn load_crossmap(path: &str, config: &crate::config::Config) -> crate::crossmap::MemoryCrossMap {
    match crate::crossmap::MemoryCrossMap::load(
        Path::new(path),
        &config.cross_map.a_id_field,
        &config.cross_map.b_id_field,
    ) {
        Ok(cm) => {
            if !cm.is_empty() {
                eprintln!("Loaded crossmap: {} existing pairs", cm.len());
            }
            cm
        }
        Err(e) => {
            eprintln!("Warning: failed to load crossmap ({}), starting fresh", e);
            crate::crossmap::MemoryCrossMap::new()
        }
    }
}

fn print_dry_run(
    config: &crate::config::Config,
    a_count: usize,
    b_count: usize,
    crossmap: &dyn CrossMapOps,
    limit: Option<usize>,
) {
    let effective = limit.map(|l| l.min(b_count)).unwrap_or(b_count);
    let skippable = crossmap.len().min(effective);

    println!("Dry run:");
    println!("  A records:       {}", a_count);
    println!("  B records:       {}", b_count);
    println!(
        "  Limit:           {}",
        limit.map(|l| l.to_string()).unwrap_or("none".into())
    );
    println!("  Already mapped:  {}", crossmap.len());
    println!("  Would process:   ~{}", effective - skippable);
    println!("  Output paths:");
    println!("    results:   {}", config.output.results_path);
    println!("    review:    {}", config.output.review_path);
    println!("    unmatched: {}", config.output.unmatched_path);
}

fn write_outputs(config: &crate::config::Config, result: &crate::batch::BatchResult) {
    let results_path = Path::new(&config.output.results_path);
    let review_path = Path::new(&config.output.review_path);
    let unmatched_path = Path::new(&config.output.unmatched_path);

    if let Err(e) = crate::batch::write_results_csv(results_path, &result.matched, config) {
        eprintln!("Failed to write results: {}", e);
        process::exit(1);
    }
    if let Err(e) = crate::batch::write_review_csv(review_path, &result.review, config) {
        eprintln!("Failed to write review: {}", e);
        process::exit(1);
    }
    if let Err(e) = crate::batch::write_unmatched_csv(
        unmatched_path,
        &result.unmatched,
        &config.datasets.b.id_field,
    ) {
        eprintln!("Failed to write unmatched: {}", e);
        process::exit(1);
    }
}

fn save_crossmap(
    crossmap: &crate::crossmap::MemoryCrossMap,
    path: &str,
    config: &crate::config::Config,
) {
    if let Err(e) = crossmap.save(
        Path::new(path),
        &config.cross_map.a_id_field,
        &config.cross_map.b_id_field,
    ) {
        eprintln!("Warning: failed to save crossmap: {}", e);
    }
}

fn print_summary(
    config: &crate::config::Config,
    result: &crate::batch::BatchResult,
    crossmap: &dyn CrossMapOps,
    crossmap_path: &str,
    verbose: bool,
) {
    let stats = &result.stats;
    println!();
    println!("Batch matching complete:");
    println!("  Total B records: {}", stats.total_b);
    println!("  Skipped (crossmap): {}", stats.skipped);
    println!("  Auto-matched: {}", stats.auto_matched);
    println!("  Review:       {}", stats.review_count);
    println!("  No match:     {}", stats.no_match);
    let index_build_secs = stats.elapsed_secs - stats.scoring_elapsed_secs;
    if index_build_secs > 0.5 {
        println!("  Index build:  {:.1}s", index_build_secs);
    }
    println!("  Scoring time: {:.1}s", stats.scoring_elapsed_secs);
    println!("  Total elapsed:{:.1}s", stats.elapsed_secs);
    if stats.scoring_elapsed_secs > 0.0 {
        let processed = stats.total_b - stats.skipped;
        println!(
            "  Throughput:   {:.0} records/sec",
            processed as f64 / stats.scoring_elapsed_secs
        );
    }
    println!();
    println!("Output files:");
    println!(
        "  results:   {} ({} rows)",
        config.output.results_path,
        result.matched.len()
    );
    println!(
        "  review:    {} ({} rows)",
        config.output.review_path,
        result.review.len()
    );
    println!(
        "  unmatched: {} ({} rows)",
        config.output.unmatched_path,
        result.unmatched.len()
    );

    if verbose {
        println!();
        println!(
            "Crossmap: {} total pairs (saved to {})",
            crossmap.len(),
            crossmap_path
        );
    }
}

fn cleanup_sqlite(db_path: &str) {
    for suffix in &["", "-wal", "-shm"] {
        let p = format!("{}{}", db_path, suffix);
        let _ = std::fs::remove_file(&p);
    }
}
