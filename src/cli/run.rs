//! `meld run` command.

use std::path::Path;
use std::process;

use crate::crossmap::CrossMapOps;
use crate::store::RecordStore;

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

    // 2. Load state (batch mode: don't pre-load B into state -- batch engine loads it)
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

    // 3. Load crossmap
    let crossmap_path = state
        .config
        .cross_map
        .path
        .as_deref()
        .unwrap_or("crossmap.csv");
    let crossmap = match crate::crossmap::MemoryCrossMap::load(
        Path::new(crossmap_path),
        &state.config.cross_map.a_id_field,
        &state.config.cross_map.b_id_field,
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
    };

    // 4. Dry-run: report what would be processed, exit
    if dry_run {
        // Count B records to show what would be processed
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
        let effective = limit.map(|l| l.min(b_count)).unwrap_or(b_count);
        let skippable = crossmap.len().min(effective);

        println!("Dry run:");
        println!(
            "  A records:       {}",
            state.store.len(crate::models::Side::A)
        );
        println!(
            "  A combined index: {} vecs",
            state
                .combined_index_a
                .as_ref()
                .map(|i| i.len())
                .unwrap_or(0)
        );
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
    let result = match crate::batch::run_batch(
        &state.config,
        state.store.as_ref(),
        state.combined_index_a.as_deref(),
        &state.encoder_pool,
        &crossmap,
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

    if let Err(e) = crate::batch::write_results_csv(results_path, &result.matched, &state.config) {
        eprintln!("Failed to write results: {}", e);
        process::exit(1);
    }
    if let Err(e) = crate::batch::write_review_csv(review_path, &result.review, &state.config) {
        eprintln!("Failed to write review: {}", e);
        process::exit(1);
    }
    if let Err(e) = crate::batch::write_unmatched_csv(
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
