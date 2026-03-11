//! `meld cache` subcommands: build, status, clear.

use std::collections::HashSet;
use std::path::Path;
use std::process;

use crate::vectordb::manifest::manifest_path;
use crate::vectordb::texthash::texthash_sidecar_path;

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
                "  A-side: {} records, {} combined index vecs",
                state.records_a.len(),
                state
                    .combined_index_a
                    .as_ref()
                    .map(|i| i.len())
                    .unwrap_or(0)
            );
            if let Some(ref rb) = state.records_b {
                println!(
                    "  B-side: {} records, {} combined index vecs",
                    rb.len(),
                    state
                        .combined_index_b
                        .as_ref()
                        .map(|i| i.len())
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
        // Collect index files (and their sidecar sizes).
        let mut file_count = 0usize;
        let mut total_size = 0u64;
        let mut manifests: Vec<crate::vectordb::manifest::CacheManifest> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                let path = entry.path();
                let is_index = path.extension().map(|e| e == "index").unwrap_or(false);
                let is_usearchdb = path.extension().map(|e| e == "usearchdb").unwrap_or(false);

                if is_index || is_usearchdb {
                    file_count += 1;
                    if path.is_file() {
                        total_size += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    } else if path.is_dir() {
                        if let Ok(sub) = std::fs::read_dir(&path) {
                            for se in sub.flatten() {
                                total_size +=
                                    std::fs::metadata(se.path()).map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                    // Count sidecar sizes too.
                    let mp = manifest_path(&path);
                    total_size += std::fs::metadata(&mp).map(|m| m.len()).unwrap_or(0);
                    let tp = texthash_sidecar_path(&path);
                    total_size += std::fs::metadata(&tp).map(|m| m.len()).unwrap_or(0);

                    // Collect manifest info for display.
                    if is_index {
                        if let Ok(Some(m)) = crate::vectordb::manifest::read_manifest(&path) {
                            manifests.push(m);
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
        for m in &manifests {
            println!(
                "    model={} spec={} blocking={} records={} built={}",
                m.model, m.spec_hash, m.blocking_hash, m.record_count, m.built_at
            );
        }
    } else {
        println!("  {:<16} {} (empty)", label, dir);
    }
}

/// Clear caches.
///
/// Smart clear: only deletes `.index` and `.usearchdb` files that are NOT
/// reachable by the current config's embedding field spec hash.  This means
/// stale files from old field specs are removed, but the current cache is
/// left untouched (pass `--force` to wipe everything, mapped to `all=true`).
pub fn cmd_cache_clear(config_path: &Path, all: bool) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let mut deleted = Vec::new();

    if all {
        // Full wipe: delete everything in both cache dirs.
        clear_cache_dir_all(&cfg.embeddings.a_cache_dir, &mut deleted);
        if let Some(ref b_dir) = cfg.embeddings.b_cache_dir {
            clear_cache_dir_all(b_dir, &mut deleted);
        }
    } else {
        // Smart clear: compute the reachable file names and only delete others.
        let emb_specs = crate::vectordb::embedding_field_specs(&cfg);
        let vq = cfg
            .performance
            .vector_quantization
            .as_deref()
            .unwrap_or("f32");
        let hash = crate::vectordb::spec_hash(&emb_specs, vq);

        // Build the set of reachable base names (stem + extension, no dir).
        let mut reachable: HashSet<String> = HashSet::new();
        let a_path = crate::vectordb::combined_cache_path(&cfg.embeddings.a_cache_dir, "a", &hash);
        if let Some(name) = Path::new(&a_path).file_name() {
            reachable.insert(name.to_string_lossy().to_string());
            // usearch backend stores a directory with the same stem
            let stem = Path::new(&a_path)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            reachable.insert(format!("{}.usearchdb", stem));
        }
        if let Some(ref b_dir) = cfg.embeddings.b_cache_dir {
            let b_path = crate::vectordb::combined_cache_path(b_dir, "b", &hash);
            if let Some(name) = Path::new(&b_path).file_name() {
                reachable.insert(name.to_string_lossy().to_string());
                let stem = Path::new(&b_path)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                reachable.insert(format!("{}.usearchdb", stem));
            }
        }

        clear_cache_dir_smart(&cfg.embeddings.a_cache_dir, &reachable, &mut deleted);
        if let Some(ref b_dir) = cfg.embeddings.b_cache_dir {
            clear_cache_dir_smart(b_dir, &reachable, &mut deleted);
        }
    }

    if deleted.is_empty() {
        println!("No stale cache files found to delete.");
    } else {
        for path in &deleted {
            println!("  deleted: {}", path);
        }
        println!("Cleared {} cache file(s).", deleted.len());
    }
}

/// Delete all `.index` and `.usearchdb` entries in a directory,
/// plus their `.manifest` and `.texthash` sidecars.
fn clear_cache_dir_all(dir: &str, deleted: &mut Vec<String>) {
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
            } else if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("  warning: failed to delete {}: {}", path_str, e);
            } else {
                deleted.push(path_str);
            }
            // Always attempt sidecar deletion (silent if absent).
            delete_sidecar(&manifest_path(&path), deleted);
            delete_sidecar(&texthash_sidecar_path(&path), deleted);
        }
    }
}

/// Delete `.index` and `.usearchdb` entries that are NOT in `reachable`,
/// plus their `.manifest` and `.texthash` sidecars.
fn clear_cache_dir_smart(dir: &str, reachable: &HashSet<String>, deleted: &mut Vec<String>) {
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
            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if reachable.contains(&file_name) {
                continue; // current cache — leave it (and its sidecars)
            }
            let path_str = path.to_string_lossy().to_string();
            if path.is_dir() {
                if let Err(e) = std::fs::remove_dir_all(&path) {
                    eprintln!("  warning: failed to delete {}: {}", path_str, e);
                } else {
                    deleted.push(path_str);
                }
            } else if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("  warning: failed to delete {}: {}", path_str, e);
            } else {
                deleted.push(path_str);
            }
            // Delete orphaned sidecars alongside the stale index file.
            delete_sidecar(&manifest_path(&path), deleted);
            delete_sidecar(&texthash_sidecar_path(&path), deleted);
        }
    }
}

/// Delete a sidecar file, recording it in `deleted` if present.
fn delete_sidecar(path: &Path, deleted: &mut Vec<String>) {
    if path.exists() {
        let path_str = path.to_string_lossy().to_string();
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!("  warning: failed to delete sidecar {}: {}", path_str, e);
        } else {
            deleted.push(path_str);
        }
    }
}
