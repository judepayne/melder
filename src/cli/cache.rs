//! `meld cache` subcommands: build, status, clear.

use std::path::Path;
use std::process;

/// Build embedding caches.
pub fn cmd_cache_build(config_path: &Path) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let load_b = cfg.embeddings.b_cache_dir.is_some();
    let opts = crate::state::LoadOptions {
        load_b,
        batch_mode: false,
    };

    let start = std::time::Instant::now();
    match crate::state::state::load_state(cfg, &opts) {
        Ok(state) => {
            let elapsed = start.elapsed();
            println!("Cache build complete:");
            println!(
                "  A-side: {} records, {} field index vecs",
                state.records_a.len(),
                state.field_indexes_a.len()
            );
            if let Some(ref rb) = state.records_b {
                println!(
                    "  B-side: {} records, {} field index vecs",
                    rb.len(),
                    state
                        .field_indexes_b
                        .as_ref()
                        .map(|fi| fi.len())
                        .unwrap_or(0)
                );
            }
            println!("  A cache dir: {}", state.config.embeddings.a_cache_dir);
            if let Some(ref b_dir) = state.config.embeddings.b_cache_dir {
                println!("  B cache dir: {}", b_dir);
            }
            println!("  Total time: {:.1}s", elapsed.as_secs_f64());
        }
        Err(e) => {
            eprintln!("Cache build failed: {}", e);
            process::exit(1);
        }
    }
}

/// Show cache status.
pub fn cmd_cache_status(config_path: &Path) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // Check A-side cache dir
    print_cache_dir_status("A cache", &cfg.embeddings.a_cache_dir);

    // Check B-side cache dir
    if let Some(ref b_dir) = cfg.embeddings.b_cache_dir {
        print_cache_dir_status("B cache", b_dir);
    } else {
        println!("  {:<16} {}", "B cache", "not configured");
    }
}

fn print_cache_dir_status(label: &str, dir: &str) {
    let p = Path::new(dir);
    if p.exists() && p.is_dir() {
        // Count .index files and .usearchdb directories
        let mut file_count = 0usize;
        let mut total_size = 0u64;
        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "index").unwrap_or(false)
                    || path.extension().map(|e| e == "usearchdb").unwrap_or(false)
                {
                    file_count += 1;
                    if path.is_file() {
                        total_size += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    } else if path.is_dir() {
                        // Sum up directory contents
                        if let Ok(sub) = std::fs::read_dir(&path) {
                            for se in sub.flatten() {
                                total_size +=
                                    std::fs::metadata(se.path()).map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
            }
        }
        let size_str = if total_size > 1_048_576 {
            format!("{:.1} MB", total_size as f64 / 1_048_576.0)
        } else if total_size > 1024 {
            format!("{:.1} KB", total_size as f64 / 1024.0)
        } else {
            format!("{} bytes", total_size)
        };
        println!(
            "  {:<16} {} ({} index files, {})",
            label, dir, file_count, size_str
        );
    } else {
        println!("  {:<16} {} (empty)", label, dir);
    }
}

/// Clear caches.
pub fn cmd_cache_clear(config_path: &Path, index_only: bool) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let _ = index_only; // no longer meaningful -- kept for CLI compat
    let mut deleted = Vec::new();

    // Clear cache directories
    clear_cache_dir(&cfg.embeddings.a_cache_dir, &mut deleted);
    if let Some(ref b_dir) = cfg.embeddings.b_cache_dir {
        clear_cache_dir(b_dir, &mut deleted);
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

fn clear_cache_dir(dir: &str, deleted: &mut Vec<String>) {
    let p = Path::new(dir);
    if !p.exists() || !p.is_dir() {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(p) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_cache = path.extension().map(|e| e == "index").unwrap_or(false)
                || path.extension().map(|e| e == "usearchdb").unwrap_or(false);
            if !is_cache {
                continue;
            }
            let path_str = path.to_string_lossy().to_string();
            if path.is_dir() {
                if let Err(e) = std::fs::remove_dir_all(&path) {
                    eprintln!("  warning: failed to delete {}: {}", path_str, e);
                } else {
                    deleted.push(path_str);
                }
            } else {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!("  warning: failed to delete {}: {}", path_str, e);
                } else {
                    deleted.push(path_str);
                }
            }
        }
    }
}
