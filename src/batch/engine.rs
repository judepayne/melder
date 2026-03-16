//! Batch matching engine.
//!
//! Pipeline per B record (parallelized with Rayon):
//! 1. Common ID pre-match (optional, runs first for all records)
//! 2. Skip if already in CrossMap
//! 3. Blocking filter → candidate selection → full scoring (via shared pipeline)
//! 4. Classify top result; if auto_match → try to claim in CrossMap

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::config::Config;
use crate::crossmap::CrossMapOps;
use crate::error::MelderError;
use crate::matching::pipeline;
#[cfg(feature = "bm25")]
use crate::matching::pipeline::Bm25Ctx;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// Result of a batch matching run.
pub struct BatchResult {
    pub matched: Vec<MatchResult>,
    pub review: Vec<MatchResult>,
    pub unmatched: Vec<(String, Record)>,
    pub stats: BatchStats,
}

/// Statistics from a batch run.
pub struct BatchStats {
    pub total_b: usize,
    pub auto_matched: usize,
    pub review_count: usize,
    pub no_match: usize,
    pub skipped: usize,
    /// Total wall time for the entire run_batch() call (load + encode + score).
    pub elapsed_secs: f64,
    /// Wall time for the scoring phase only (excludes B-side load and encoding).
    pub scoring_elapsed_secs: f64,
}

/// Outcome for a single B record after scoring.
enum RecordOutcome {
    Auto(MatchResult),
    Review(MatchResult),
    NoMatch(String, Record),
}

/// Run the batch matching engine.
///
/// Both A and B records must already be loaded into `store` before calling.
/// The B-side combined embedding index, if needed, should be passed in via
/// `combined_index_b`. This function handles scoring, classification, and
/// crossmap claiming.
pub fn run_batch(
    config: &Config,
    store: &dyn RecordStore,
    combined_index_a: Option<&dyn VectorDB>,
    combined_index_b: Option<&dyn VectorDB>,
    crossmap: &dyn CrossMapOps,
    limit: Option<usize>,
) -> Result<BatchResult, MelderError> {
    let start = Instant::now();

    // Get B IDs from the store (already loaded by caller).
    let b_ids = store.ids(Side::B);

    let total_b = if let Some(lim) = limit {
        lim.min(b_ids.len())
    } else {
        b_ids.len()
    };

    let mut matched = Vec::new();
    let mut review = Vec::new();
    let mut unmatched = Vec::new();
    let mut skipped = 0;

    // Common ID pre-match phase: if common_id_field is configured, match
    // records with identical common IDs before any scoring.
    let common_id_matched = if let (Some(a_cid_field), Some(b_cid_field)) = (
        &config.datasets.a.common_id_field,
        &config.datasets.b.common_id_field,
    ) {
        let cid_start = Instant::now();
        let mut a_common_index: HashMap<String, String> = HashMap::new();
        for a_id in store.ids(Side::A) {
            if let Some(a_rec) = store.get(Side::A, &a_id)
                && let Some(val) = a_rec.get(a_cid_field)
            {
                let val = val.trim();
                if !val.is_empty() {
                    a_common_index.insert(val.to_string(), a_id);
                }
            }
        }

        let mut common_count = 0usize;
        for b_id in b_ids.iter().take(total_b) {
            if crossmap.has_b(b_id) {
                continue;
            }
            if let Some(b_rec) = store.get(Side::B, b_id)
                && let Some(b_val) = b_rec.get(b_cid_field)
            {
                let b_val = b_val.trim();
                if !b_val.is_empty()
                    && let Some(a_id) = a_common_index.get(b_val)
                {
                    crossmap.add(a_id, b_id);
                    let a_rec = store.get(Side::A, a_id);
                    let mr = MatchResult {
                        query_id: b_id.clone(),
                        matched_id: a_id.clone(),
                        query_side: Side::B,
                        score: 1.0,
                        field_scores: vec![],
                        classification: Classification::Auto,
                        matched_record: a_rec,
                        from_crossmap: false,
                    };
                    matched.push(mr);
                    common_count += 1;
                }
            }
        }
        if common_count > 0 {
            eprintln!(
                "Common ID pre-match: {} pairs matched in {:.1}ms",
                common_count,
                cid_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        common_count
    } else {
        0
    };
    let _ = common_id_matched;

    // Partition: separate crossmapped (skip) from work items.
    let b_ids_slice = &b_ids[..total_b];
    let mut work_ids: Vec<&str> = Vec::with_capacity(total_b);
    for b_id in b_ids_slice {
        if crossmap.has_b(b_id) {
            skipped += 1;
        } else if store.contains(Side::B, b_id) {
            work_ids.push(b_id.as_str());
        }
    }

    // Pre-compute top_n from config (0 = no limit in batch → best result only).
    let top_n = config.top_n.unwrap_or(5);

    // Build A-side BM25 index if method: bm25 is configured.
    #[cfg(feature = "bm25")]
    let bm25_index_a: Option<std::sync::RwLock<crate::bm25::index::BM25Index>> = {
        let has_bm25 = config.match_fields.iter().any(|mf| mf.method == "bm25");
        if has_bm25 && !config.bm25_fields.is_empty() {
            let bm25_start = Instant::now();
            let idx = crate::bm25::index::BM25Index::build(
                store,
                Side::A,
                &config.bm25_fields,
                &config.blocking.fields,
            )?;
            eprintln!(
                "Built BM25 index for {} A records in {:.1}ms",
                store.len(Side::A),
                bm25_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(std::sync::RwLock::new(idx))
        } else {
            None
        }
    };

    // Score all B records in a single parallel pass.
    let scoring_start = Instant::now();
    let progress = AtomicUsize::new(0);
    let work_total = work_ids.len();

    let outcomes: Vec<RecordOutcome> = work_ids
        .par_iter()
        .filter_map(|b_id| {
            let b_record = store.get(Side::B, b_id)?;

            // Progress reporting (atomic, lock-free)
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(1000) || done == work_total {
                let elapsed = scoring_start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let eta = if done < work_total {
                    (work_total - done) as f64 / rate
                } else {
                    0.0
                };
                eprintln!(
                    "  scored {}/{} B records ({:.0} rec/s, ETA {:.0}s)",
                    done, work_total, rate, eta
                );
            }

            // Check if this B record was already claimed by another thread.
            if crossmap.has_b(b_id) {
                return None;
            }

            // Pre-query the blocking index via the store.
            let blocked_ids: Vec<String> = if config.blocking.enabled {
                store.blocking_query(&b_record, Side::B)
            } else {
                store.ids(Side::A)
            };

            // Fetch the query combined vec from the B-side index.
            let query_combined_vec: Vec<f32> = combined_index_b
                .and_then(|idx| idx.get(b_id).ok().flatten())
                .unwrap_or_default();

            // Shared pipeline: candidate selection → full scoring
            let ann_candidates = config.ann_candidates.unwrap_or(50);
            let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);

            // Build BM25 context: lock the index, build query text, pass to pipeline.
            #[cfg(feature = "bm25")]
            let results = if let Some(ref mtx) = bm25_index_a {
                let guard = mtx.read().unwrap_or_else(|e| e.into_inner());
                let query_text = guard.query_text_for(&b_record, Side::B);
                let ctx = Bm25Ctx::new(&*guard, query_text);
                pipeline::score_pool(
                    b_id,
                    &b_record,
                    Side::B,
                    &query_combined_vec,
                    store,
                    Side::A,
                    combined_index_a,
                    &blocked_ids,
                    config,
                    ann_candidates,
                    bm25_candidates_n,
                    top_n,
                    Some(ctx),
                )
            } else {
                pipeline::score_pool(
                    b_id,
                    &b_record,
                    Side::B,
                    &query_combined_vec,
                    store,
                    Side::A,
                    combined_index_a,
                    &blocked_ids,
                    config,
                    ann_candidates,
                    bm25_candidates_n,
                    top_n,
                    None,
                )
            };

            #[cfg(not(feature = "bm25"))]
            let results = pipeline::score_pool(
                b_id,
                &b_record,
                Side::B,
                &query_combined_vec,
                store,
                Side::A,
                combined_index_a,
                &blocked_ids,
                config,
                ann_candidates,
                bm25_candidates_n,
                top_n,
                None,
            );

            // Claim loop: try each auto-match candidate in ranked order.
            // Uses crossmap.claim() so concurrent threads can't double-match.
            let mut outcome = None;
            for mut result in results {
                if result.score < config.thresholds.review_floor {
                    outcome = Some(RecordOutcome::NoMatch(b_id.to_string(), b_record.clone()));
                    break;
                }
                if result.score >= config.thresholds.auto_match {
                    let a_id = &result.matched_id;
                    if crossmap.claim(a_id, b_id) {
                        // Attach matched record
                        if result.matched_record.is_none() {
                            result.matched_record = store.get(Side::A, a_id);
                        }
                        outcome = Some(RecordOutcome::Auto(result));
                        break;
                    }
                    // B already claimed — try next candidate
                    continue;
                }
                // Review band
                outcome = Some(RecordOutcome::Review(result));
                break;
            }

            outcome.or_else(|| Some(RecordOutcome::NoMatch(b_id.to_string(), b_record)))
        })
        .collect();

    for outcome in outcomes {
        match outcome {
            RecordOutcome::Auto(mr) => matched.push(mr),
            RecordOutcome::Review(mr) => review.push(mr),
            RecordOutcome::NoMatch(id, rec) => unmatched.push((id, rec)),
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let scoring_elapsed = scoring_start.elapsed().as_secs_f64();
    let stats = BatchStats {
        total_b,
        auto_matched: matched.len(),
        review_count: review.len(),
        no_match: unmatched.len(),
        skipped,
        elapsed_secs: elapsed,
        scoring_elapsed_secs: scoring_elapsed,
    };

    Ok(BatchResult {
        matched,
        review,
        unmatched,
        stats,
    })
}
