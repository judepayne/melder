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

use crate::bm25::scorer::normalise_bm25;
use crate::config::{Config, MatchMethod};
use crate::crossmap::CrossMapOps;
use crate::error::MelderError;
use crate::matching::pipeline;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// Result of a batch matching run.
pub struct BatchResult {
    pub matched: Vec<MatchResult>,
    pub review: Vec<MatchResult>,
    /// Unmatched B records with their best candidate score (None if no candidates found).
    pub unmatched: Vec<(String, Record, Option<f64>)>,
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
    /// B record that scored below review_floor. Best candidate score is preserved
    /// (None if no candidates were found at all).
    NoMatch(String, Record, Option<f64>),
}

/// Run the batch matching engine.
///
/// When `skip_prematch` is true, the common-ID pre-match and exact prefilter
/// phases are skipped — all records go through full scoring. Used by `meld tune`
/// so that ground-truth pairs receive real scores instead of being short-circuited
/// at 1.0.
#[allow(clippy::too_many_arguments)]
pub fn run_batch(
    config: &Config,
    store: &dyn RecordStore,
    combined_index_a: Option<&dyn VectorDB>,
    combined_index_b: Option<&dyn VectorDB>,
    crossmap: &dyn CrossMapOps,
    exclusions: &crate::matching::exclusions::Exclusions,
    limit: Option<usize>,
    skip_prematch: bool,
) -> Result<BatchResult, MelderError> {
    let start = Instant::now();

    // Get B IDs from the store (already loaded by caller).
    let b_ids = store.ids(Side::B)?;

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
    // Skipped in tune mode so ground-truth pairs get real scores.
    let common_id_matched = if skip_prematch {
        0
    } else if let (Some(a_cid_field), Some(b_cid_field)) = (
        &config.datasets.a.common_id_field,
        &config.datasets.b.common_id_field,
    ) {
        let cid_start = Instant::now();
        let mut a_common_index: HashMap<String, String> = HashMap::new();
        for a_id in store.ids(Side::A)? {
            if let Ok(Some(a_rec)) = store.get(Side::A, &a_id)
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
            if let Ok(Some(b_rec)) = store.get(Side::B, b_id)
                && let Some(b_val) = b_rec.get(b_cid_field)
            {
                let b_val = b_val.trim();
                if !b_val.is_empty()
                    && let Some(a_id) = a_common_index.get(b_val)
                {
                    // Use claim() to atomically enforce the bijection
                    // invariant (Constitution §3). If either side is
                    // already mapped, skip — the record will go through
                    // normal scoring instead.
                    if !crossmap.claim(a_id, b_id) {
                        continue;
                    }
                    let a_rec = store.get(Side::A, a_id).ok().flatten();
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

    // Exact prefilter phase: for each B record, check if all configured
    // exact field pairs match an A record. If so, auto-confirm immediately
    // at score 1.0 — no scoring required. Runs before blocking so cross-block
    // exact matches (e.g. wrong country but matching LEI) are still found.
    // Skipped in tune mode so all records go through full scoring.
    if !skip_prematch && config.exact_prefilter.enabled && !config.exact_prefilter.fields.is_empty()
    {
        let ep_start = Instant::now();

        // Extract A-side field names from the config pairs.
        let a_field_names: Vec<String> = config
            .exact_prefilter
            .fields
            .iter()
            .map(|fp| fp.field_a.clone())
            .collect();

        // Build the A-side index (HashMap for MemoryStore, SQL index for SQLite).
        store.build_exact_index(Side::A, &a_field_names)?;

        let mut exact_count = 0usize;
        for b_id in b_ids.iter().take(total_b) {
            if crossmap.has_b(b_id) {
                continue;
            }
            let b_rec = match store.get(Side::B, b_id) {
                Ok(Some(r)) => r,
                _ => continue,
            };

            // Build (a_field, b_value) pairs for lookup — we query A using B's values.
            let kvs: Vec<(String, String)> = config
                .exact_prefilter
                .fields
                .iter()
                .map(|fp| {
                    let val = b_rec
                        .get(&fp.field_b)
                        .map(|v| v.trim().to_string())
                        .unwrap_or_default();
                    (fp.field_a.clone(), val)
                })
                .collect();

            if let Ok(Some(a_id)) = store.exact_lookup(Side::A, &kvs) {
                // Use claim() to atomically enforce the bijection
                // invariant (Constitution §3). The old has_a() check was
                // a TOCTOU race under Rayon parallelism — claim() checks
                // both directions atomically.
                if !crossmap.claim(&a_id, b_id) {
                    continue;
                }
                let a_rec = store.get(Side::A, &a_id).ok().flatten();
                matched.push(MatchResult {
                    query_id: b_id.clone(),
                    matched_id: a_id,
                    query_side: Side::B,
                    score: 1.0,
                    field_scores: vec![],
                    classification: Classification::Auto,
                    matched_record: a_rec,
                    from_crossmap: false,
                });
                exact_count += 1;
            }
        }

        if exact_count > 0 {
            eprintln!(
                "Exact prefilter: {} pairs matched in {:.1}ms",
                exact_count,
                ep_start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    // Partition: separate crossmapped (skip) from work items.
    let b_ids_slice = &b_ids[..total_b];
    let mut work_ids: Vec<&str> = Vec::with_capacity(total_b);
    for b_id in b_ids_slice {
        if crossmap.has_b(b_id) {
            skipped += 1;
        } else if store.contains(Side::B, b_id).unwrap_or(false) {
            work_ids.push(b_id.as_str());
        }
    }

    // Pre-compute top_n from config (0 = no limit in batch → best result only).
    let top_n = config.top_n.unwrap_or(5);

    // Build A-side BM25 index if method: bm25 is configured.
    // Uses SimpleBm25 — no RwLock, no commit, no self-score pre-computation
    // (analytical self-score is O(K) at query time).
    let bm25_index_a: Option<crate::bm25::simple::SimpleBm25> = {
        let has_bm25 = config
            .match_fields
            .iter()
            .any(|mf| mf.method == MatchMethod::Bm25);
        if has_bm25 && !config.bm25_fields.is_empty() {
            let bm25_start = Instant::now();
            let idx = crate::bm25::simple::SimpleBm25::build(store, Side::A, &config.bm25_fields);
            eprintln!(
                "Built BM25 index for {} A records in {:.1}ms",
                store.len(Side::A).unwrap_or(0),
                bm25_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(idx)
        } else {
            None
        }
    };

    // Load synonym dictionary if configured.
    let synonym_dict: Option<std::sync::Arc<crate::synonym::dictionary::SynonymDictionary>> =
        if let Some(ref sd_cfg) = config.synonym_dictionary {
            let dict_start = Instant::now();
            let dict = crate::synonym::dictionary::SynonymDictionary::load(std::path::Path::new(
                &sd_cfg.path,
            ))?;
            eprintln!(
                "Loaded synonym dictionary ({} groups, {} terms) in {:.1}ms",
                dict.len(),
                dict.len() * 2, // approximate
                dict_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(std::sync::Arc::new(dict))
        } else {
            None
        };

    // Build A-side synonym index if method: synonym is configured.
    let synonym_index_a: Option<crate::synonym::index::SynonymIndex> =
        if !config.synonym_fields.is_empty() {
            let syn_start = Instant::now();
            let idx = crate::synonym::index::SynonymIndex::build(
                store,
                Side::A,
                &config.synonym_fields,
                synonym_dict.clone(),
            );
            eprintln!(
                "Built synonym index for A records ({} keys) in {:.1}ms",
                idx.len(),
                syn_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(idx)
        } else {
            None
        };

    // Score all B records in a single parallel pass.
    let scoring_start = Instant::now();
    let progress = AtomicUsize::new(0);
    let work_total = work_ids.len();

    let outcomes: Vec<RecordOutcome> = work_ids
        .par_iter()
        .filter_map(|b_id| {
            let b_record = store.get(Side::B, b_id).ok()??;

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
                store
                    .blocking_query(&b_record, Side::B, Side::A)
                    .unwrap_or_default()
            } else {
                store.ids(Side::A).unwrap_or_default()
            };

            // Fetch the query combined vec from the B-side index.
            let query_combined_vec: Vec<f32> = combined_index_b
                .and_then(|idx| idx.get(b_id).ok().flatten())
                .unwrap_or_default();

            // Shared pipeline: candidate selection → full scoring
            let ann_candidates = config.ann_candidates.unwrap_or(50);
            let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);

            // BM25 candidate generation (pre-computed, passed to pipeline).
            let (bm25_cand_ids, bm25_scores_map) = if let Some(ref bm25) = bm25_index_a {
                let query_text = bm25.query_text_for(&b_record, Side::B);
                let self_score = bm25.analytical_self_score(&query_text);
                let raw_results = bm25.score_blocked(&query_text, &blocked_ids, bm25_candidates_n);
                let scored: Vec<(String, f64)> = raw_results
                    .into_iter()
                    .map(|(cid, raw)| {
                        let norm = normalise_bm25(raw, self_score);
                        (cid, norm)
                    })
                    .collect();
                let ids: Vec<String> = scored.iter().map(|(cid, _)| cid.clone()).collect();
                let map: HashMap<String, f64> = scored.into_iter().collect();
                (ids, map)
            } else {
                (Vec::new(), HashMap::new())
            };

            // Synonym candidate generation.
            let synonym_cand_ids = if !config.synonym_fields.is_empty() {
                if let Some(ref idx) = synonym_index_a {
                    pipeline::synonym_candidate_stage(
                        idx,
                        &b_record,
                        Side::B,
                        &config.synonym_fields,
                    )
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let scoring_query = pipeline::ScoringQuery {
                id: b_id,
                record: &b_record,
                side: Side::B,
                combined_vec: &query_combined_vec,
            };
            let scoring_pool = pipeline::ScoringPool {
                store,
                side: Side::A,
                combined_index: combined_index_a,
                blocked_ids: &blocked_ids,
                bm25_candidate_ids: &bm25_cand_ids,
                bm25_scores_map: &bm25_scores_map,
                synonym_candidate_ids: &synonym_cand_ids,
                synonym_dictionary: synonym_dict.as_deref(),
                exclusions,
            };
            let results =
                pipeline::score_pool(&scoring_query, &scoring_pool, config, ann_candidates, top_n);

            // Claim loop: try each auto-match candidate in ranked order.
            // Uses crossmap.claim() so concurrent threads can't double-match.
            let mut outcome = None;
            let mut best_score: Option<f64> = None;
            for mut result in results {
                // Track the best score seen regardless of outcome.
                if best_score.is_none() {
                    best_score = Some(result.score);
                }
                if result.score < config.thresholds.review_floor {
                    outcome = Some(RecordOutcome::NoMatch(
                        b_id.to_string(),
                        b_record.clone(),
                        best_score,
                    ));
                    break;
                }
                if result.score >= config.thresholds.auto_match {
                    let a_id = &result.matched_id;
                    if crossmap.claim(a_id, b_id) {
                        // Attach matched record
                        if result.matched_record.is_none() {
                            result.matched_record = store.get(Side::A, a_id).ok().flatten();
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

            outcome.or_else(|| {
                Some(RecordOutcome::NoMatch(
                    b_id.to_string(),
                    b_record,
                    best_score,
                ))
            })
        })
        .collect();

    for outcome in outcomes {
        match outcome {
            RecordOutcome::Auto(mr) => matched.push(mr),
            RecordOutcome::Review(mr) => review.push(mr),
            RecordOutcome::NoMatch(id, rec, score) => unmatched.push((id, rec, score)),
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
