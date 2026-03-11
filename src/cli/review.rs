//! `meld review` subcommands: list, import.

use std::path::Path;
use std::process;

/// List pending reviews.
pub fn cmd_review_list(config_path: &Path) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    let review_path = Path::new(&cfg.output.review_path);
    if !review_path.exists() {
        println!(
            "No review records (file not found: {})",
            cfg.output.review_path
        );
        return;
    }

    let mut rdr = match csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(review_path)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to read review csv: {}", e);
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
        println!(
            "No review records (file is empty: {})",
            cfg.output.review_path
        );
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

/// Import review decisions.
pub fn cmd_review_import(config_path: &Path, decisions_path: &Path) {
    let cfg = match crate::config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config error: {}", e);
            process::exit(1);
        }
    };

    // Load crossmap
    let crossmap_path = cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv");
    let crossmap = match crate::crossmap::CrossMap::load(
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

    // Read decisions csv: a_id, b_id, decision
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
                "Decisions csv must have columns: a_id, b_id, decision. Found: {:?}",
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

    // Update review csv: remove accepted/rejected pairs
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

        // Re-read review csv and filter out decided pairs
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

                    // Rewrite review csv
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
    println!("  Crossmap: {} -> {} pairs", initial_count, crossmap.len());
}
