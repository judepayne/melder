//! `meld tune` command.
//!
//! Runs the batch pipeline and produces diagnostic output: score distribution
//! histogram, overlap analysis, and accuracy metrics. When `common_id_field`
//! is configured on both datasets, scores are classified against ground truth
//! into two populations (known match vs known non-match) for a richer analysis.

use std::collections::HashMap;
use std::path::Path;
use std::process;

use crate::models::Side;

// --- Display characters ---

const MATCHED_CHAR: char = '█';
const UNMATCHED_CHAR: char = '░';
const ALL_CHAR: char = '█';
const THRESHOLD_CHAR: char = '▼';

// --- Tune options ---

pub struct TuneOpts {
    pub no_run: bool,
    pub bucket_width: f64,
    pub min_score: Option<f64>,
    pub max_score: Option<f64>,
    pub bar_width: usize,
    pub overlap_limit: usize,
}

/// Entry point for `meld tune`.
#[allow(clippy::too_many_arguments)]
pub fn cmd_tune(
    config_path: &Path,
    verbose: bool,
    no_run: bool,
    bucket_width: f64,
    min_score: Option<f64>,
    max_score: Option<f64>,
    bar_width: usize,
    overlap_limit: usize,
) {
    let opts = TuneOpts {
        no_run,
        bucket_width,
        min_score,
        max_score,
        bar_width,
        overlap_limit,
    };

    // 1. Load config
    let cfg = super::load_config_or_exit(config_path);

    let current_auto = cfg.thresholds.auto_match;
    let current_floor = cfg.thresholds.review_floor;

    if verbose {
        eprintln!(
            "Tune: {} -- current thresholds: auto_match={}, review_floor={}",
            cfg.job.name, current_auto, current_floor
        );
    }

    // 2. Build ground-truth map if common_id_field is configured on both sides.
    //    This is done BEFORE loading state so we can use the raw dataset files.
    let ground_truth = build_ground_truth(&cfg);
    let has_ground_truth = ground_truth.is_some();

    if let Some(gt) = ground_truth.as_ref() {
        eprintln!(
            "Ground truth: {} known matches from common_id_field",
            gt.len()
        );
    }

    // 3. Run pipeline or load cached results.
    let (matched, review, unmatched, _field_scores_map, store) = if opts.no_run {
        let (m, r, u, f) = load_cached_results(&cfg);
        (m, r, u, f, None)
    } else {
        let (m, r, u, f, s) = run_pipeline(cfg, verbose);
        (m, r, u, f, Some(s))
    };

    // 4. Classify scores into populations.
    let (match_scores, nonmatch_scores, all_scores, accuracy) = classify_scores(
        &matched,
        &review,
        &unmatched,
        ground_truth.as_ref(),
        current_auto,
        current_floor,
    );

    // 5. Collect per-field scores split by population.
    let (field_scores_all, field_scores_match, field_scores_nonmatch) =
        classify_field_scores(&matched, &review, ground_truth.as_ref());

    // 6. Output: histogram
    if !has_ground_truth {
        println!();
        println!("  NOTE: no common_id_field in config. Add common_id_field to both datasets");
        println!("  for two-population analysis, overlap coefficient, and accuracy metrics.");
    }

    println!();
    println!("=== Score Distribution ===");
    println!();

    if has_ground_truth {
        println!(
            "  {} known match  {} known non-match",
            MATCHED_CHAR, UNMATCHED_CHAR
        );
    } else {
        println!("  {} scored pairs", ALL_CHAR);
    }
    println!();

    if has_ground_truth {
        render_histogram_two_pop(
            &match_scores,
            &nonmatch_scores,
            &opts,
            current_auto,
            current_floor,
        );
    } else {
        render_histogram_one_pop(&all_scores, &opts, current_auto, current_floor);
    }

    // Threshold counts — follows naturally from the histogram.
    let total_with_unmatched = all_scores.len();
    let auto_count = all_scores.iter().filter(|&&s| s >= current_auto).count();
    let review_count = all_scores
        .iter()
        .filter(|&&s| s >= current_floor && s < current_auto)
        .count();
    let below_floor = all_scores.iter().filter(|&&s| s < current_floor).count();

    let pct = |n: usize| -> f64 {
        if total_with_unmatched > 0 {
            n as f64 / total_with_unmatched as f64 * 100.0
        } else {
            0.0
        }
    };

    println!();
    println!(
        "  auto_match    >= {:.2}:  {:>6} ({:.1}%)",
        current_auto,
        fmt_count(auto_count),
        pct(auto_count)
    );
    println!(
        "  review_floor  >= {:.2}:  {:>6} ({:.1}%)",
        current_floor,
        fmt_count(review_count),
        pct(review_count)
    );
    println!(
        "  no_match       < {:.2}:  {:>6} ({:.1}%)",
        current_floor,
        fmt_count(below_floor),
        pct(below_floor)
    );
    println!(
        "  total:                {:>6}",
        fmt_count(total_with_unmatched)
    );

    // 7. Overlap coefficient (ground truth only).
    if has_ground_truth {
        let overlap = overlap_coefficient(&match_scores, &nonmatch_scores, 0.02);
        println!();
        println!(
            "  Overlap: {:.4}  (0 = perfect separation, 1 = identical)",
            overlap
        );
    }

    // 8. Ground-truth accuracy section.
    if let Some(ref acc) = accuracy {
        println!();
        println!("=== Ground-Truth Accuracy ===");
        println!();
        println!(
            "  Auto-matched:    {:>7}",
            fmt_count(acc.auto_tp + acc.auto_fp)
        );
        println!("    Correct (TP):  {:>7}", fmt_count(acc.auto_tp));
        println!("    Incorrect (FP):{:>7}", fmt_count(acc.auto_fp));
        println!(
            "  Review:          {:>7}",
            fmt_count(acc.review_tp + acc.review_fp)
        );
        println!("    Correct (TP):  {:>7}", fmt_count(acc.review_tp));
        println!("    Incorrect (FP):{:>7}", fmt_count(acc.review_fp));
        println!("  Missed (FN):     {:>7}", fmt_count(acc.fn_count));
        println!();
        if acc.auto_tp + acc.auto_fp > 0 {
            let precision = acc.auto_tp as f64 / (acc.auto_tp + acc.auto_fp) as f64;
            println!("  Precision:       {:>7.1}%", precision * 100.0);
        }
        let total_matches = acc.auto_tp + acc.review_tp + acc.fn_count;
        if total_matches > 0 {
            let recall_auto = acc.auto_tp as f64 / total_matches as f64;
            let recall_combined = (acc.auto_tp + acc.review_tp) as f64 / total_matches as f64;
            println!("  Recall (auto):   {:>7.1}%", recall_auto * 100.0);
            println!("  Combined recall: {:>7.1}%", recall_combined * 100.0);
        }
    }

    // 9. Per-field score statistics.
    if !field_scores_all.is_empty() {
        println!();
        println!("=== Per-Field Analysis ===");
        println!();

        let mut field_keys: Vec<&String> = field_scores_all.keys().collect();
        field_keys.sort();

        for key in &field_keys {
            println!("  {}", key);
            println!(
                "    {:<42} {:>6} {:>6} {:>6} {:>6} {:>6}",
                "", "Min", "Max", "Mean", "Median", "StdDev"
            );

            if let Some(scores) = field_scores_all.get(*key) {
                let stats = compute_stats(scores);
                println!(
                    "    {:<42} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
                    "All:", stats.min, stats.max, stats.mean, stats.median, stats.std_dev
                );
            }

            if has_ground_truth {
                if let Some(scores) = field_scores_match.get(*key)
                    && !scores.is_empty()
                {
                    let stats = compute_stats(scores);
                    println!(
                        "    {:<42} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
                        "With common_id (expect to match):",
                        stats.min,
                        stats.max,
                        stats.mean,
                        stats.median,
                        stats.std_dev
                    );
                }

                if let Some(scores) = field_scores_nonmatch.get(*key)
                    && !scores.is_empty()
                {
                    let stats = compute_stats(scores);
                    println!(
                        "    {:<42} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
                        "No common_id (don't expect to match):",
                        stats.min,
                        stats.max,
                        stats.mean,
                        stats.median,
                        stats.std_dev
                    );
                }

                // Separation indicator: difference between match mean and non-match mean.
                let m_mean = field_scores_match
                    .get(*key)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.iter().sum::<f64>() / s.len() as f64);
                let nm_mean = field_scores_nonmatch
                    .get(*key)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.iter().sum::<f64>() / s.len() as f64);
                if let (Some(m), Some(nm)) = (m_mean, nm_mean) {
                    let gap = m - nm;
                    let label = if gap < 0.05 {
                        "weak separation"
                    } else if gap < 0.15 {
                        "moderate separation"
                    } else {
                        "strong separation"
                    };
                    println!("    Mean gap: {:.3} ({})", gap, label);
                }
            }
            println!();
        }
    }

    // 10. Overlap zone analysis (ground truth only, requires field scores).
    if let Some(gt) = ground_truth.as_ref()
        && !field_scores_all.is_empty()
    {
        render_overlap_zone(
            &matched,
            &review,
            &unmatched,
            gt,
            store.as_deref(),
            current_auto,
            current_floor,
            opts.overlap_limit,
        );
    }

    println!();
}

// --- Overlap zone analysis ---

/// Show per-record details for records in the overlap zone between thresholds.
///
/// For each record shows:
/// 1. Score breakdown by field
/// 2. Actual field values from the record
/// 3. What it matched against (if it matched)
#[allow(clippy::too_many_arguments)]
fn render_overlap_zone(
    matched: &[crate::models::MatchResult],
    review: &[crate::models::MatchResult],
    unmatched: &[(String, crate::models::Record, Option<f64>)],
    ground_truth: &HashMap<String, String>,
    store: Option<&dyn crate::store::RecordStore>,
    auto_threshold: f64,
    review_threshold: f64,
    limit: usize,
) {
    let mut overlap_expect: Vec<&crate::models::MatchResult> = Vec::new();
    let mut overlap_no_expect: Vec<&crate::models::MatchResult> = Vec::new();

    for mr in matched.iter().chain(review.iter()) {
        if mr.score < review_threshold || mr.score >= auto_threshold {
            continue;
        }
        if ground_truth.contains_key(&mr.query_id) {
            overlap_expect.push(mr);
        } else {
            overlap_no_expect.push(mr);
        }
    }

    // Missed matches near the floor.
    let mut missed_near_floor: Vec<(&str, f64)> = Vec::new();
    for (b_id, _rec, best_score) in unmatched {
        if ground_truth.contains_key(b_id) {
            let score = best_score.unwrap_or(0.0);
            if score >= review_threshold * 0.8 {
                missed_near_floor.push((b_id, score));
            }
        }
    }

    if overlap_expect.is_empty() && overlap_no_expect.is_empty() && missed_near_floor.is_empty() {
        return;
    }

    println!();
    println!(
        "=== Overlap Zone ({:.2} - {:.2}) ===",
        review_threshold, auto_threshold
    );

    // With common_id: expect to match
    if !overlap_expect.is_empty() {
        println!();
        let total = overlap_expect.len();
        println!(
            "  With common_id: expect to match ({} record{} in overlap):",
            total,
            if total == 1 { "" } else { "s" }
        );
        let mut sorted = overlap_expect.clone();
        sorted.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for mr in sorted.iter().take(limit) {
            print_overlap_record(mr, store, Some(ground_truth));
        }
        if total > limit {
            println!("    ... and {} more", total - limit);
        }
    }

    // No common_id: don't expect to match
    if !overlap_no_expect.is_empty() {
        println!();
        let total = overlap_no_expect.len();
        println!(
            "  No common_id: don't expect to match ({} record{} in overlap):",
            total,
            if total == 1 { "" } else { "s" }
        );
        let mut sorted = overlap_no_expect.clone();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for mr in sorted.iter().take(limit) {
            print_overlap_record(mr, store, None);
        }
        if total > limit {
            println!("    ... and {} more", total - limit);
        }
    }

    // Missed matches near the floor
    if !missed_near_floor.is_empty() {
        missed_near_floor
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!();
        let total = missed_near_floor.len();
        println!(
            "  With common_id: missed ({} record{} — scoring below review_floor {:.2}):",
            total,
            if total == 1 { "" } else { "s" },
            review_threshold,
        );
        for (b_id, score) in missed_near_floor.iter().take(limit) {
            println!("    {}  best score: {:.3}", b_id, score);
        }
        if total > limit {
            println!("    ... and {} more", total - limit);
        }
    }
}

/// Print a single overlap zone record with field scores, values, and match info.
fn print_overlap_record(
    mr: &crate::models::MatchResult,
    store: Option<&dyn crate::store::RecordStore>,
    ground_truth: Option<&HashMap<String, String>>,
) {
    // Line 1: IDs and composite score
    println!();
    println!(
        "    {} -> {}  score: {:.3}",
        mr.query_id, mr.matched_id, mr.score
    );

    // Line 2: field score breakdown
    if !mr.field_scores.is_empty() {
        print!("      scores: ");
        for (i, fs) in mr.field_scores.iter().enumerate() {
            if i > 0 {
                print!("  ");
            }
            let label = if fs.method == "bm25" {
                "bm25".to_string()
            } else if fs.method == "synonym" {
                "synonym".to_string()
            } else {
                format!("{}_{}", fs.field_a, fs.field_b)
            };
            print!("{}: {:.2}", label, fs.score);
        }
        println!();
    }

    // Line 3: actual field values from B record (query)
    if let Some(st) = store {
        let query_side = mr.query_side;
        if let Ok(Some(b_rec)) = st.get(query_side, &mr.query_id) {
            print!("      query:  ");
            let mut first = true;
            for fs in &mr.field_scores {
                if fs.method == "bm25" || fs.method == "synonym" {
                    continue;
                }
                let field = match query_side {
                    Side::A => &fs.field_a,
                    Side::B => &fs.field_b,
                };
                let val = b_rec.get(field).map(|s| s.as_str()).unwrap_or("");
                if !first {
                    print!("  |  ");
                }
                // Truncate long values (char-boundary safe)
                let display = if val.chars().count() > 40 {
                    let truncated: String = val.chars().take(37).collect();
                    format!("{}...", truncated)
                } else {
                    val.to_string()
                };
                print!("{}", display);
                first = false;
            }
            println!();
        }

        // Line 4: matched record values (A side for B queries, B side for A queries)
        let match_side = match query_side {
            Side::A => Side::B,
            Side::B => Side::A,
        };
        if let Ok(Some(a_rec)) = st.get(match_side, &mr.matched_id) {
            print!("      match:  ");
            let mut first = true;
            for fs in &mr.field_scores {
                if fs.method == "bm25" || fs.method == "synonym" {
                    continue;
                }
                let field = match match_side {
                    Side::A => &fs.field_a,
                    Side::B => &fs.field_b,
                };
                let val = a_rec.get(field).map(|s| s.as_str()).unwrap_or("");
                if !first {
                    print!("  |  ");
                }
                let display = if val.chars().count() > 40 {
                    let truncated: String = val.chars().take(37).collect();
                    format!("{}...", truncated)
                } else {
                    val.to_string()
                };
                print!("{}", display);
                first = false;
            }
            println!();
        }

        // Line 5: if this is a "with common_id" record, show expected A
        if let Some(gt) = ground_truth
            && let Some(expected_a) = gt.get(&mr.query_id)
            && *expected_a != mr.matched_id
        {
            println!("      expected: {} (matched wrong record)", expected_a);
        }
    }
}

// --- Ground truth ---

/// Build ground-truth mapping from common_id_field.
///
/// Returns `Some(b_id → a_id)` when both datasets have common_id_field,
/// `None` otherwise.
fn build_ground_truth(cfg: &crate::config::Config) -> Option<HashMap<String, String>> {
    let a_cid = cfg.datasets.a.common_id_field.as_ref()?;
    let b_cid = cfg.datasets.b.common_id_field.as_ref()?;

    // Load A dataset to build common_id → a_id index.
    let (a_records, _a_ids) = match crate::data::load_dataset(
        Path::new(&cfg.datasets.a.path),
        &cfg.datasets.a.id_field,
        std::slice::from_ref(a_cid),
        cfg.datasets.a.format.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Warning: could not load A dataset for ground truth: {}", e);
            return None;
        }
    };

    let mut a_common_index: HashMap<String, String> = HashMap::new();
    for (a_id, a_rec) in &a_records {
        if let Some(val) = a_rec.get(a_cid) {
            let val = val.trim();
            if !val.is_empty() {
                a_common_index.insert(val.to_string(), a_id.clone());
            }
        }
    }

    // Load B dataset to look up each B's common_id.
    let (b_records, _b_ids) = match crate::data::load_dataset(
        Path::new(&cfg.datasets.b.path),
        &cfg.datasets.b.id_field,
        std::slice::from_ref(b_cid),
        cfg.datasets.b.format.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Warning: could not load B dataset for ground truth: {}", e);
            return None;
        }
    };

    let mut ground_truth: HashMap<String, String> = HashMap::new();
    for (b_id, b_rec) in &b_records {
        if let Some(b_val) = b_rec.get(b_cid) {
            let b_val = b_val.trim();
            if !b_val.is_empty()
                && let Some(a_id) = a_common_index.get(b_val)
            {
                ground_truth.insert(b_id.clone(), a_id.clone());
            }
        }
    }

    if ground_truth.is_empty() {
        eprintln!(
            "Warning: common_id_field configured but zero matching records — falling back to single-population analysis"
        );
        return None;
    }

    Some(ground_truth)
}

// --- Pipeline execution ---

/// Run the full scoring pipeline (normal tune path).
#[allow(clippy::type_complexity)]
fn run_pipeline(
    cfg: crate::config::Config,
    verbose: bool,
) -> (
    Vec<crate::models::MatchResult>,
    Vec<crate::models::MatchResult>,
    Vec<(String, crate::models::Record, Option<f64>)>,
    HashMap<String, Vec<f64>>,
    std::sync::Arc<dyn crate::store::RecordStore>,
) {
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

    // Load B records, build B embedding index, insert into store.
    let (b_records_map, _b_ids) = match crate::data::load_dataset(
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

    let combined_index_b = match crate::vectordb::build_or_load_combined_index(
        &state.config.vector_backend,
        state.config.embeddings.b_cache_dir.as_deref(),
        &b_records_map,
        &_b_ids,
        &state.config,
        false,
        &state.encoder_pool,
        false,
        Some(std::path::Path::new(&state.config.datasets.b.path)),
    ) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln!("Failed to build B embedding index: {}", e);
            process::exit(1);
        }
    };

    for (id, rec) in &b_records_map {
        let _ = state.store.insert(Side::B, id, rec);
    }
    drop(b_records_map);

    let crossmap = crate::crossmap::MemoryCrossMap::new();
    let no_exclusions = crate::matching::exclusions::Exclusions::new();
    if verbose {
        eprintln!("Running batch matching for score analysis...");
    }
    let result = match crate::batch::run_batch(
        &state.config,
        state.store.as_ref(),
        state.combined_index_a.as_deref(),
        combined_index_b.as_deref(),
        &crossmap,
        &no_exclusions,
        None,
        true, // skip pre-match
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Batch matching failed: {}", e);
            process::exit(1);
        }
    };

    // Write output CSVs so --no-run can re-use them.
    let results_path = Path::new(&state.config.output.results_path);
    let review_path = Path::new(&state.config.output.review_path);
    let unmatched_path = Path::new(&state.config.output.unmatched_path);

    if let Err(e) = crate::batch::write_results_csv(results_path, &result.matched, &state.config) {
        eprintln!("Warning: failed to write results CSV: {}", e);
    }
    if let Err(e) = crate::batch::write_review_csv(review_path, &result.review, &state.config) {
        eprintln!("Warning: failed to write review CSV: {}", e);
    }
    if let Err(e) = crate::batch::write_unmatched_csv(
        unmatched_path,
        &result.unmatched,
        &state.config.datasets.b.id_field,
    ) {
        eprintln!("Warning: failed to write unmatched CSV: {}", e);
    }

    // Collect per-field scores.
    let mut field_scores_map: HashMap<String, Vec<f64>> = HashMap::new();
    for mr in result.matched.iter().chain(result.review.iter()) {
        for fs in &mr.field_scores {
            let key = format!("{}_{}", fs.field_a, fs.field_b);
            field_scores_map.entry(key).or_default().push(fs.score);
        }
    }

    (
        result.matched,
        result.review,
        result.unmatched,
        field_scores_map,
        state.store.clone(),
    )
}

/// Load cached results from output CSVs (--no-run path).
#[allow(clippy::type_complexity)]
fn load_cached_results(
    cfg: &crate::config::Config,
) -> (
    Vec<crate::models::MatchResult>,
    Vec<crate::models::MatchResult>,
    Vec<(String, crate::models::Record, Option<f64>)>,
    HashMap<String, Vec<f64>>,
) {
    // Read results.csv
    let matched = load_match_results(&cfg.output.results_path, "results");
    let review = load_match_results(&cfg.output.review_path, "review");
    let unmatched = load_unmatched(&cfg.output.unmatched_path, &cfg.datasets.b.id_field);

    let mut field_scores_map: HashMap<String, Vec<f64>> = HashMap::new();
    for mr in matched.iter().chain(review.iter()) {
        for fs in &mr.field_scores {
            let key = format!("{}_{}", fs.field_a, fs.field_b);
            field_scores_map.entry(key).or_default().push(fs.score);
        }
    }

    (matched, review, unmatched, field_scores_map)
}

/// Load MatchResults from a CSV file (results or review).
fn load_match_results(path: &str, label: &str) -> Vec<crate::models::MatchResult> {
    let p = Path::new(path);
    if !p.exists() {
        eprintln!(
            "Error: --no-run requires output file '{}' ({}), but it does not exist.",
            path, label
        );
        eprintln!("Run `meld tune` without --no-run first to generate output files.");
        process::exit(1);
    }

    let mut results = Vec::new();
    let mut rdr = match csv::Reader::from_path(p) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error reading {}: {}", path, e);
            process::exit(1);
        }
    };

    // Read headers before iterating records.
    let a_id_idx = rdr
        .headers()
        .ok()
        .and_then(|h| h.iter().position(|c| c == "a_id"))
        .unwrap_or(0);
    let b_id_idx = rdr
        .headers()
        .ok()
        .and_then(|h| h.iter().position(|c| c == "b_id"))
        .unwrap_or(1);
    let score_idx = rdr
        .headers()
        .ok()
        .and_then(|h| h.iter().position(|c| c == "score"))
        .unwrap_or(2);

    for record in rdr.records() {
        let row = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error parsing {}: {}", path, e);
                continue;
            }
        };

        let a_id = row.get(a_id_idx).unwrap_or("").to_string();
        let b_id = row.get(b_id_idx).unwrap_or("").to_string();
        let score: f64 = row
            .get(score_idx)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        results.push(crate::models::MatchResult {
            query_id: b_id,
            matched_id: a_id,
            query_side: Side::B,
            score,
            field_scores: vec![],
            classification: crate::models::Classification::Auto,
            matched_record: None,
            from_crossmap: false,
        });
    }

    results
}

/// Load unmatched records from CSV.
fn load_unmatched(path: &str, id_field: &str) -> Vec<(String, crate::models::Record, Option<f64>)> {
    let p = Path::new(path);
    if !p.exists() {
        return Vec::new();
    }

    let mut results = Vec::new();
    let mut rdr = match csv::Reader::from_path(p) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    // Read headers before iterating.
    let id_idx = rdr
        .headers()
        .ok()
        .and_then(|h| h.iter().position(|c| c == id_field))
        .unwrap_or(0);
    let score_col = rdr
        .headers()
        .ok()
        .and_then(|h| h.iter().position(|c| c == "score"));

    for record in rdr.records() {
        let row = match record {
            Ok(r) => r,
            Err(_) => continue,
        };

        let id = row.get(id_idx).unwrap_or("").to_string();
        let score = score_col
            .and_then(|i| row.get(i))
            .and_then(|s| s.parse::<f64>().ok());

        results.push((id, HashMap::new(), score));
    }

    results
}

// --- Score classification ---

struct Accuracy {
    auto_tp: usize,
    auto_fp: usize,
    review_tp: usize,
    review_fp: usize,
    fn_count: usize,
}

/// Classify scores into match/nonmatch populations and compute accuracy.
fn classify_scores(
    matched: &[crate::models::MatchResult],
    review: &[crate::models::MatchResult],
    unmatched: &[(String, crate::models::Record, Option<f64>)],
    ground_truth: Option<&HashMap<String, String>>,
    auto_threshold: f64,
    review_threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Option<Accuracy>) {
    let mut match_scores: Vec<f64> = Vec::new();
    let mut nonmatch_scores: Vec<f64> = Vec::new();
    let mut all_scores: Vec<f64> = Vec::new();

    if let Some(gt) = ground_truth {
        let mut acc = Accuracy {
            auto_tp: 0,
            auto_fp: 0,
            review_tp: 0,
            review_fp: 0,
            fn_count: 0,
        };

        // Matched (auto) results.
        for mr in matched {
            all_scores.push(mr.score);
            if let Some(true_a) = gt.get(&mr.query_id) {
                if mr.matched_id == *true_a {
                    match_scores.push(mr.score);
                    acc.auto_tp += 1;
                } else {
                    // Matched to wrong A — this is a false positive AND
                    // the B record's true match is still out there somewhere.
                    nonmatch_scores.push(mr.score);
                    acc.auto_fp += 1;
                }
            } else {
                // B has no ground truth entry → non-match
                nonmatch_scores.push(mr.score);
                acc.auto_fp += 1;
            }
        }

        // Review results.
        for mr in review {
            all_scores.push(mr.score);
            if let Some(true_a) = gt.get(&mr.query_id) {
                if mr.matched_id == *true_a {
                    match_scores.push(mr.score);
                    acc.review_tp += 1;
                } else {
                    nonmatch_scores.push(mr.score);
                    acc.review_fp += 1;
                }
            } else {
                nonmatch_scores.push(mr.score);
                acc.review_fp += 1;
            }
        }

        // Unmatched.
        for (b_id, _rec, best_score) in unmatched {
            let score = best_score.unwrap_or(0.0);
            all_scores.push(score);
            if gt.contains_key(b_id) {
                // This B should have matched but didn't.
                match_scores.push(score);
                acc.fn_count += 1;
            } else {
                // True negative.
                nonmatch_scores.push(score);
            }
        }

        (match_scores, nonmatch_scores, all_scores, Some(acc))
    } else {
        // No ground truth — single population.
        for mr in matched.iter().chain(review.iter()) {
            all_scores.push(mr.score);
        }
        for (_b_id, _rec, best_score) in unmatched {
            all_scores.push(best_score.unwrap_or(0.0));
        }

        let _ = (auto_threshold, review_threshold); // unused without GT

        (Vec::new(), Vec::new(), all_scores, None)
    }
}

// --- Per-field classification ---

/// Collect per-field scores split by population (all / match / non-match).
#[allow(clippy::type_complexity)]
fn classify_field_scores(
    matched: &[crate::models::MatchResult],
    review: &[crate::models::MatchResult],
    ground_truth: Option<&HashMap<String, String>>,
) -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, Vec<f64>>,
    HashMap<String, Vec<f64>>,
) {
    let mut all: HashMap<String, Vec<f64>> = HashMap::new();
    let mut matches: HashMap<String, Vec<f64>> = HashMap::new();
    let mut nonmatches: HashMap<String, Vec<f64>> = HashMap::new();

    for mr in matched.iter().chain(review.iter()) {
        let is_tp = ground_truth
            .and_then(|gt| gt.get(&mr.query_id))
            .map(|true_a| mr.matched_id == *true_a)
            .unwrap_or(false);
        let has_gt_entry = ground_truth
            .map(|gt| gt.contains_key(&mr.query_id))
            .unwrap_or(false);

        for fs in &mr.field_scores {
            let key = format!("{}_{}", fs.field_a, fs.field_b);
            all.entry(key.clone()).or_default().push(fs.score);

            if ground_truth.is_some() {
                if is_tp {
                    matches.entry(key).or_default().push(fs.score);
                } else if !has_gt_entry {
                    // B has no ground truth — non-match
                    nonmatches.entry(key).or_default().push(fs.score);
                } else {
                    // Matched to wrong A — non-match
                    nonmatches.entry(key).or_default().push(fs.score);
                }
            }
        }
    }

    (all, matches, nonmatches)
}

struct FieldStats {
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    std_dev: f64,
}

fn compute_stats(scores: &[f64]) -> FieldStats {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let mean = sorted.iter().sum::<f64>() / sorted.len().max(1) as f64;
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len().is_multiple_of(2) && sorted.len() >= 2 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let variance = if sorted.is_empty() {
        0.0
    } else {
        sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64
    };

    FieldStats {
        min,
        max,
        mean,
        median,
        std_dev: variance.sqrt(),
    }
}

// --- Histogram rendering ---

fn render_histogram_one_pop(
    scores: &[f64],
    opts: &TuneOpts,
    auto_threshold: f64,
    review_threshold: f64,
) {
    let w = opts.bucket_width;
    let n = (1.0 / w).ceil() as usize + 1;
    let mut hist = vec![0usize; n];

    for &s in scores {
        let bkt = (s / w).floor() as usize;
        let bkt = bkt.min(n - 1);
        hist[bkt] += 1;
    }

    let (first, last) = display_range(&hist, n, w, opts, auto_threshold, review_threshold);
    let max_count = hist[first..=last].iter().copied().max().unwrap_or(1).max(1);
    let scale = opts.bar_width as f64 / max_count as f64;

    let thresholds = threshold_buckets(w, auto_threshold, review_threshold);

    for (i, &count) in hist.iter().enumerate().take(last + 1).skip(first) {
        let bar_len = if count > 0 {
            (count as f64 * scale + 0.5) as usize
        } else {
            0
        };
        let bar: String = std::iter::repeat_n(ALL_CHAR, bar_len).collect();
        let label = format!("  {:.2} ", i as f64 * w);
        if let Some(tlabel) = thresholds.get(&i) {
            println!("{}{}  {} {}", label, bar, THRESHOLD_CHAR, tlabel);
        } else {
            println!("{}{}", label, bar);
        }
    }
}

fn render_histogram_two_pop(
    match_scores: &[f64],
    nonmatch_scores: &[f64],
    opts: &TuneOpts,
    auto_threshold: f64,
    review_threshold: f64,
) {
    let w = opts.bucket_width;
    let n = (1.0 / w).ceil() as usize + 1;
    let mut m_hist = vec![0usize; n];
    let mut u_hist = vec![0usize; n];

    for &s in match_scores {
        let bkt = (s / w).floor() as usize;
        m_hist[bkt.min(n - 1)] += 1;
    }
    for &s in nonmatch_scores {
        let bkt = (s / w).floor() as usize;
        u_hist[bkt.min(n - 1)] += 1;
    }

    // Merge for range detection.
    let merged: Vec<usize> = m_hist
        .iter()
        .zip(u_hist.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (first, last) = display_range(&merged, n, w, opts, auto_threshold, review_threshold);

    let max_count = merged[first..=last]
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let scale = opts.bar_width as f64 / max_count as f64;

    let thresholds = threshold_buckets(w, auto_threshold, review_threshold);

    for i in first..=last {
        let m = m_hist[i];
        let u = u_hist[i];
        let m_len = if m > 0 {
            (m as f64 * scale + 0.5) as usize
        } else {
            0
        };
        let u_len = if u > 0 {
            (u as f64 * scale + 0.5) as usize
        } else {
            0
        };
        let bar: String = std::iter::repeat_n(MATCHED_CHAR, m_len)
            .chain(std::iter::repeat_n(UNMATCHED_CHAR, u_len))
            .collect();
        let label = format!("  {:.2} ", i as f64 * w);
        if let Some(tlabel) = thresholds.get(&i) {
            println!("{}{}  {} {}", label, bar, THRESHOLD_CHAR, tlabel);
        } else {
            println!("{}{}", label, bar);
        }
    }
}

/// Determine display range (first..=last bucket indices).
fn display_range(
    hist: &[usize],
    n: usize,
    w: f64,
    opts: &TuneOpts,
    auto_threshold: f64,
    review_threshold: f64,
) -> (usize, usize) {
    let first = if let Some(min_s) = opts.min_score {
        (min_s / w).floor() as usize
    } else {
        // Auto: first non-empty bucket.
        hist.iter().position(|&c| c > 0).unwrap_or(0)
    };

    let last = if let Some(max_s) = opts.max_score {
        ((max_s / w).ceil() as usize).min(n - 1)
    } else {
        hist.iter().rposition(|&c| c > 0).unwrap_or(n - 1)
    };

    // Expand to include threshold buckets.
    let auto_bkt = (auto_threshold / w).floor() as usize;
    let review_bkt = (review_threshold / w).floor() as usize;
    let first = first.min(review_bkt).min(auto_bkt);
    let last = last.max(review_bkt).max(auto_bkt).min(n - 1);

    (first, last)
}

/// Build threshold bucket annotations.
fn threshold_buckets(w: f64, auto_threshold: f64, review_threshold: f64) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    let auto_bkt = (auto_threshold / w).floor() as usize;
    let review_bkt = (review_threshold / w).floor() as usize;

    if auto_bkt == review_bkt {
        map.insert(
            auto_bkt,
            format!(
                "auto_match ({:.2}) + review_floor ({:.2})",
                auto_threshold, review_threshold
            ),
        );
    } else {
        map.insert(auto_bkt, format!("auto_match ({:.2})", auto_threshold));
        map.insert(
            review_bkt,
            format!("review_floor ({:.2})", review_threshold),
        );
    }
    map
}

// --- Overlap coefficient ---

fn overlap_coefficient(match_scores: &[f64], nonmatch_scores: &[f64], bucket_width: f64) -> f64 {
    let n = (1.0 / bucket_width).ceil() as usize + 1;
    let mut m_hist = vec![0usize; n];
    let mut u_hist = vec![0usize; n];

    for &s in match_scores {
        let bkt = (s / bucket_width).floor() as usize;
        m_hist[bkt.min(n - 1)] += 1;
    }
    for &s in nonmatch_scores {
        let bkt = (s / bucket_width).floor() as usize;
        u_hist[bkt.min(n - 1)] += 1;
    }

    let m_total = match_scores.len().max(1) as f64;
    let u_total = nonmatch_scores.len().max(1) as f64;

    m_hist
        .iter()
        .zip(u_hist.iter())
        .map(|(&m, &u)| {
            let p_m = m as f64 / m_total;
            let p_u = u as f64 / u_total;
            p_m.min(p_u)
        })
        .sum()
}

// --- Helpers ---

fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!(
            "{},{:03},{:03}",
            n / 1_000_000,
            (n / 1_000) % 1_000,
            n % 1_000
        )
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        format!("{}", n)
    }
}
