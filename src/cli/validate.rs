//! `meld validate` command.

use std::path::Path;

/// Validate a config file and print a summary.
pub fn cmd_validate(config_path: &Path) {
    let cfg = super::load_config_or_exit(config_path);
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
    println!(
        "  top_n: {}",
        cfg.top_n
            .map(|n| n.to_string())
            .unwrap_or_else(|| "5 (default)".into())
    );
    println!("  vector_backend: {}", cfg.vector_backend);
    if let Some(es) = cfg.performance.expansion_search {
        println!("  expansion_search: {}", es);
    }
    if let Some(bs) = cfg.bm25_commit_batch_size {
        println!("  bm25_commit_batch_size: {}", bs);
    }
    if let Some(pool) = cfg.performance.encoder_pool_size {
        println!("  encoder_pool_size: {}", pool);
    }
    if cfg.performance.quantized {
        println!("  quantized: true");
    }

    println!("  required_fields_a: {:?}", cfg.required_fields_a);
    println!("  required_fields_b: {:?}", cfg.required_fields_b);
}
