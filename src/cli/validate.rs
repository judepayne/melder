//! `meld validate` command.

use std::path::Path;
use std::process;

/// Validate a config file and print a summary.
pub fn cmd_validate(config_path: &Path) {
    match crate::config::load_config(config_path) {
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
            println!(
                "  top_n: {}",
                cfg.top_n
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "5 (default)".into())
            );
            println!("  vector_backend: {}", cfg.vector_backend);
            if let Some(pool) = cfg.performance.encoder_pool_size {
                println!("  encoder_pool_size: {}", pool);
            }
            if cfg.performance.quantized {
                println!("  quantized: true");
            }

            println!("  required_fields_a: {:?}", cfg.required_fields_a);
            println!("  required_fields_b: {:?}", cfg.required_fields_b);
        }
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    }
}
