//! `meld crossmap` subcommands: stats, export, import.

use std::path::Path;
use std::process;

use crate::crossmap::CrossMapOps;

/// Show cross-map statistics.
pub fn cmd_crossmap_stats(config_path: &Path) {
    let cfg = super::load_config_or_exit(config_path);

    // Load crossmap
    let crossmap_path = cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv");
    let crossmap = match crate::crossmap::MemoryCrossMap::load(
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

    // Count dataset rows without loading full records into memory
    let a_count = match crate::data::count_rows(
        Path::new(&cfg.datasets.a.path),
        cfg.datasets.a.format.as_deref(),
    ) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Warning: failed to count A dataset: {}", e);
            0
        }
    };

    let b_count = match crate::data::count_rows(
        Path::new(&cfg.datasets.b.path),
        cfg.datasets.b.format.as_deref(),
    ) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Warning: failed to count B dataset: {}", e);
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

/// Export cross-map to csv.
pub fn cmd_crossmap_export(config_path: &Path, out_path: &Path) {
    let cfg = super::load_config_or_exit(config_path);

    let crossmap_path = cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv");
    let crossmap = match crate::crossmap::MemoryCrossMap::load(
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

/// Import cross-map from csv.
pub fn cmd_crossmap_import(config_path: &Path, import_path: &Path) {
    let cfg = super::load_config_or_exit(config_path);

    let crossmap_path = cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv");

    // Load existing crossmap
    let crossmap = match crate::crossmap::MemoryCrossMap::load(
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
    let import_cm = match crate::crossmap::MemoryCrossMap::load(
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
    for (a_id, b_id) in import_cm.pairs() {
        crossmap.add(&a_id, &b_id);
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
    println!("Crossmap: {} -> {} pairs", initial_count, crossmap.len());
}
