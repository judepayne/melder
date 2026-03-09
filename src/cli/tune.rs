//! `meld tune` command.

use std::path::Path;
use std::process;

/// Tune thresholds on labelled data.
pub fn cmd_tune(config_path: &Path, verbose: bool) {
    // 1. Load config
    let cfg = match crate::config::load_config(config_path) {
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
            "Tune: {} -- current thresholds: auto_match={}, review_floor={}",
            cfg.job.name, current_auto, current_floor
        );
    }

    // 2. Load state (batch mode)
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

    // 3. Run batch engine with a throwaway crossmap (no persistence)
    let mut crossmap = crate::crossmap::CrossMap::new();
    eprintln!("Running batch matching for score analysis...");
    let result = match crate::batch::run_batch(
        &state.config,
        &state.records_a,
        &state.field_indexes_a,
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
