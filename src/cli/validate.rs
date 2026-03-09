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
    }
}
