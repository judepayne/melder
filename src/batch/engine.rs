//! Batch matching engine.
//!
//! Pipeline per B record (parallelized with Rayon):
//! 1. Common ID pre-match (optional, runs first for all records)
//! 2. Skip if already in CrossMap
//! 3. Blocking filter → candidate selection → full scoring (via shared pipeline)
//! 4. Classify top result; if auto_match → try to claim in CrossMap

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use super::assignment::{self, AssignmentSummary, BatchAssignmentError, ScoredQuery};
use crate::bm25::scorer::normalise_bm25;
use crate::config::{Config, MatchMethod};
use crate::crossmap::CrossMapOps;
use crate::error::MelderError;
use crate::matching::pipeline;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::state::match_log::{MatchLog, MatchLogEvent};
use crate::store::RecordStore;
use crate::vectordb::VectorDB;

/// Result of a batch matching run.
pub struct BatchResult {
    /// Processed B IDs for graph-mode direct output (includes protected/skipped B).
    pub processed_graph_b: Option<Vec<String>>,
    pub matched: Vec<MatchResult>,
    pub review: Vec<MatchResult>,
    /// Unmatched B records with their best candidate score (None if no candidates found).
    pub unmatched: Vec<(String, Record, Option<f64>)>,
    pub stats: BatchStats,
}

/// Statistics from a batch run.
pub struct BatchStats {
    pub assignment: Option<AssignmentSummary>,
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
pub(super) enum RecordOutcome {
    Scored(ScoredQuery),
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
    match_log: Option<Arc<MatchLog>>,
    scoring_log: Option<&crate::output::scoring_log::ScoringLogSender>,
) -> Result<BatchResult, MelderError> {
    let start = Instant::now();

    // Get B IDs from the store (already loaded by caller).
    let mut b_ids = store.ids(Side::B)?;
    // Tune measures scoring behavior, not the configured assignment policy.
    let graph_mode = (!skip_prematch)
        .then(|| config.batch.assignment.graph_mode())
        .flatten();
    let graph = graph_mode.is_some();
    if let Some(mode) = graph_mode {
        b_ids.sort_unstable();
        assignment::emit(
            match_log.as_deref(),
            &MatchLogEvent::BatchAssignmentStarted { mode },
        )?;
        if let Some(log) = &match_log {
            log.flush().map_err(BatchAssignmentError::from)?;
        }
    }

    let total_b = if let Some(lim) = limit {
        lim.min(b_ids.len())
    } else {
        b_ids.len()
    };

    let mut matched = Vec::new();
    let mut review = Vec::new();
    let mut unmatched = Vec::new();
    let mut skipped = 0;

    // Match log error tracking — log errors but don't abort the scoring run.
    let ml_errors = Arc::new(AtomicBool::new(false));

    // Serialize-then-write: serialization happens outside the mutex,
    // the lock is held only for the memcpy into BufWriter (~100ns).
    let ml_append = |log: &MatchLog, event: &MatchLogEvent, errors: &AtomicBool| {
        let result = MatchLog::serialize_event(event).and_then(|bytes| log.append_bytes(&bytes));
        if let Err(e) = result
            && !errors.swap(true, Ordering::Relaxed)
        {
            tracing::error!(error = %e, "match log write failed (further errors suppressed)");
        }
    };

    // Graph logs record the protected constraints separately from scored links.
    if graph {
        let mut fixed = crossmap.pairs();
        fixed.sort_unstable();
        for (a, b) in &fixed {
            if exclusions.contains(a, b) {
                return Err(BatchAssignmentError::ExcludedProtected {
                    a: a.clone(),
                    b: b.clone(),
                }
                .into());
            }
        }
        if let Some(log) = &match_log {
            let mut a_ids = store.ids(Side::A)?;
            a_ids.sort_unstable();
            for (side, ids) in [(Side::A, a_ids.as_slice()), (Side::B, &b_ids[..total_b])] {
                for id in ids {
                    assignment::emit(
                        Some(log),
                        &MatchLogEvent::UpsertRecord {
                            side,
                            record: assignment::record(store, side, id)?,
                        },
                    )?;
                }
            }
            for (a_id, b_id) in fixed {
                assignment::emit(
                    Some(log),
                    &MatchLogEvent::CrossMapConfirm {
                        a_id,
                        b_id,
                        score: None,
                        rank: None,
                        reason: Some("crossmap".into()),
                    },
                )?;
            }
        }
        matched = assignment::prematch(
            config,
            store,
            crossmap,
            exclusions,
            &b_ids[..total_b],
            match_log.as_deref(),
        )?;
    }

    // Emit UpsertRecord events for all A and B records.
    if !graph && let Some(ref ml) = match_log {
        let a_id_field = &config.datasets.a.id_field;
        for a_id in store.ids(Side::A)? {
            if let Ok(Some(rec)) = store.get(Side::A, &a_id) {
                ml_append(
                    ml,
                    &MatchLogEvent::UpsertRecord {
                        side: Side::A,
                        record: rec,
                    },
                    &ml_errors,
                );
            }
        }
        let b_id_field = &config.datasets.b.id_field;
        for b_id in b_ids.iter().take(total_b) {
            if let Ok(Some(rec)) = store.get(Side::B, b_id) {
                ml_append(
                    ml,
                    &MatchLogEvent::UpsertRecord {
                        side: Side::B,
                        record: rec,
                    },
                    &ml_errors,
                );
            }
        }
        let _ = (a_id_field, b_id_field); // used for documentation only
    }

    // Common ID pre-match phase: if common_id_field is configured, match
    // records with identical common IDs before any scoring.
    // Skipped in tune mode so ground-truth pairs get real scores.
    let common_id_matched = if graph || skip_prematch {
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
                    if let Some(ref ml) = match_log {
                        ml_append(
                            ml,
                            &MatchLogEvent::CrossMapConfirm {
                                a_id: a_id.clone(),
                                b_id: b_id.clone(),
                                score: Some(1.0),
                                rank: None,
                                reason: Some("canonical".into()),
                            },
                            &ml_errors,
                        );
                    }
                    let mr = MatchResult {
                        query_id: b_id.clone(),
                        matched_id: a_id.clone(),
                        query_side: Side::B,
                        score: 1.0,
                        field_scores: vec![],
                        classification: Classification::Auto,
                        matched_record: a_rec,
                        from_crossmap: false,
                        rank: None,
                        reason: Some("canonical".into()),
                    };
                    matched.push(mr);
                    common_count += 1;
                }
            }
        }
        if common_count > 0 {
            tracing::info!(
                pairs = common_count,
                elapsed_ms = format!("{:.1}", cid_start.elapsed().as_secs_f64() * 1000.0),
                "common ID pre-match complete"
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
    if !graph
        && !skip_prematch
        && config.exact_prefilter.enabled
        && !config.exact_prefilter.fields.is_empty()
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
                if let Some(ref ml) = match_log {
                    ml_append(
                        ml,
                        &MatchLogEvent::CrossMapConfirm {
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                            score: Some(1.0),
                            rank: None,
                            reason: Some("exact".into()),
                        },
                        &ml_errors,
                    );
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
                    rank: None,
                    reason: Some("exact".into()),
                });
                exact_count += 1;
            }
        }

        if exact_count > 0 {
            tracing::info!(
                pairs = exact_count,
                elapsed_ms = format!("{:.1}", ep_start.elapsed().as_secs_f64() * 1000.0),
                "exact prefilter complete"
            );
        }
    }

    // Partition: separate crossmapped (skip) from work items.
    let b_ids_slice = &b_ids[..total_b];
    let mut work_ids: Vec<&str> = Vec::with_capacity(total_b);
    for b_id in b_ids_slice {
        if crossmap.has_b(b_id) {
            if graph {
                skipped += 1;
                continue;
            }
            // Emit pre-loaded crossmap pairs to the match log.
            if let Some(ref ml) = match_log
                && let Some(a_id) = crossmap.get_a(b_id)
            {
                ml_append(
                    ml,
                    &MatchLogEvent::CrossMapConfirm {
                        a_id,
                        b_id: b_id.clone(),
                        score: None,
                        rank: None,
                        reason: Some("crossmap".into()),
                    },
                    &ml_errors,
                );
            }
            skipped += 1;
        } else if graph || store.contains(Side::B, b_id).unwrap_or(false) {
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
            use crate::bm25::simple::SimpleBm25;
            let idx = if graph {
                SimpleBm25::try_build(store, Side::A, &config.bm25_fields)?
            } else {
                SimpleBm25::build(store, Side::A, &config.bm25_fields)
            };
            tracing::info!(
                a_records = store.len(Side::A).unwrap_or(0),
                elapsed_ms = format!("{:.1}", bm25_start.elapsed().as_secs_f64() * 1000.0),
                "BM25 index build complete"
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
            tracing::info!(
                groups = dict.len(),
                terms = dict.len() * 2,
                elapsed_ms = format!("{:.1}", dict_start.elapsed().as_secs_f64() * 1000.0),
                "synonym dictionary loaded"
            );
            Some(std::sync::Arc::new(dict))
        } else {
            None
        };

    // Build A-side synonym index if method: synonym is configured.
    let synonym_index_a: Option<crate::synonym::index::SynonymIndex> = if !config
        .synonym_fields
        .is_empty()
    {
        let syn_start = Instant::now();
        use crate::synonym::index::SynonymIndex;
        let idx = if graph {
            SynonymIndex::try_build(store, Side::A, &config.synonym_fields, synonym_dict.clone())?
        } else {
            SynonymIndex::build(store, Side::A, &config.synonym_fields, synonym_dict.clone())
        };
        tracing::info!(
            keys = idx.len(),
            elapsed_ms = format!("{:.1}", syn_start.elapsed().as_secs_f64() * 1000.0),
            "synonym index build complete"
        );
        Some(idx)
    } else {
        None
    };

    // Score all B records in a single parallel pass.
    let scoring_start = Instant::now();

    // Without blocking, share one A-side ID snapshot across all B queries.
    let all_a_ids = if config.blocking.enabled {
        Vec::new()
    } else {
        store.ids(Side::A)?
    };
    let progress = AtomicUsize::new(0);
    let work_total = work_ids.len();

    let outcomes: Vec<Option<RecordOutcome>> = work_ids
        .par_iter()
        .map(|b_id| -> Result<_, MelderError> {
            let b_record = if graph {
                assignment::record(store, Side::B, b_id)?
            } else if let Ok(Some(record)) = store.get(Side::B, b_id) {
                record
            } else {
                return Ok(None);
            };

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
                tracing::info!(
                    done,
                    total = work_total,
                    rate = format!("{:.0}", rate),
                    eta_s = format!("{:.0}", eta),
                    "scoring progress"
                );
            }

            // Check if this B record was already claimed by another thread.
            if !graph && crossmap.has_b(b_id) {
                return Ok(None);
            }

            // Pre-query the blocking index via the store.
            let query_blocked_ids;
            let blocked_ids: &[String] = if config.blocking.enabled {
                let blocked = store.blocking_query(&b_record, Side::B, Side::A);
                query_blocked_ids = if graph {
                    blocked?
                } else {
                    blocked.unwrap_or_default()
                };
                &query_blocked_ids
            } else {
                &all_a_ids
            };

            // Fetch the query combined vec from the B-side index.
            let query_combined_vec: Vec<f32> = if let Some(idx) = combined_index_b {
                let vector = idx.get(b_id);
                if graph {
                    vector
                        .map_err(BatchAssignmentError::from)?
                        .unwrap_or_default()
                } else {
                    vector.ok().flatten().unwrap_or_default()
                }
            } else {
                Vec::new()
            };

            // Shared pipeline: candidate selection → full scoring
            let ann_candidates = config.ann_candidates.unwrap_or(50);
            let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);

            // BM25 candidate generation (pre-computed, passed to pipeline).
            let (bm25_cand_ids, bm25_scores_map) = if let Some(ref bm25) = bm25_index_a {
                let query_text = bm25.query_text_for(&b_record, Side::B);
                let self_score = bm25.analytical_self_score(&query_text);
                let raw_results = if config.blocking.enabled {
                    bm25.score_blocked(&query_text, blocked_ids, bm25_candidates_n)
                } else {
                    bm25.score_all(&query_text, bm25_candidates_n)
                };
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
                blocked_ids,
                bm25_candidate_ids: &bm25_cand_ids,
                bm25_scores_map: &bm25_scores_map,
                synonym_candidate_ids: &synonym_cand_ids,
                synonym_dictionary: synonym_dict.as_deref(),
                exclusions,
            };
            let results = if graph {
                pipeline::score_pool_strict(
                    &scoring_query,
                    &scoring_pool,
                    config,
                    ann_candidates,
                    top_n,
                )?
            } else {
                pipeline::score_pool(&scoring_query, &scoring_pool, config, ann_candidates, top_n)
            };

            // Send to scoring log if enabled.
            if let Some(sl) = scoring_log {
                sl.send(b_id, Side::B, &results);
            }

            if graph {
                return Ok(Some(RecordOutcome::Scored(assignment::retain(
                    config, crossmap, results,
                )?)));
            }

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
                        // Emit match log event
                        if let Some(ref ml) = match_log {
                            ml_append(
                                ml,
                                &MatchLogEvent::CrossMapConfirm {
                                    a_id: a_id.clone(),
                                    b_id: b_id.to_string(),
                                    score: Some(result.score),
                                    rank: result.rank,
                                    reason: result.reason.clone(),
                                },
                                &ml_errors,
                            );
                        }
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
                if let Some(ref ml) = match_log {
                    ml_append(
                        ml,
                        &MatchLogEvent::ReviewMatch {
                            id: b_id.to_string(),
                            side: Side::B,
                            candidate_id: result.matched_id.clone(),
                            score: result.score,
                            rank: result.rank,
                            reason: result.reason.clone(),
                        },
                        &ml_errors,
                    );
                }
                outcome = Some(RecordOutcome::Review(result));
                break;
            }

            // Emit NoMatchBelow for records with no match/review outcome.
            let final_outcome = outcome.or_else(|| {
                Some(RecordOutcome::NoMatch(
                    b_id.to_string(),
                    b_record,
                    best_score,
                ))
            });
            if let Some(RecordOutcome::NoMatch(ref id, _, ref bs)) = final_outcome
                && let Some(ref ml) = match_log
            {
                ml_append(
                    ml,
                    &MatchLogEvent::NoMatchBelow {
                        query_id: id.clone(),
                        query_side: Side::B,
                        best_candidate_id: None,
                        best_score: *bs,
                    },
                    &ml_errors,
                );
            }
            Ok(final_outcome)
        })
        .collect::<Result<_, _>>()?;
    let scoring_elapsed = scoring_start.elapsed().as_secs_f64();
    let (outcomes, assignment) = if let Some(mode) = graph_mode {
        let scored = outcomes
            .into_iter()
            .map(|outcome| match outcome {
                Some(RecordOutcome::Scored(query)) => Ok(query),
                _ => Err(BatchAssignmentError::Invariant(
                    "missing graph query result",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (outcomes, summary) = assignment::finish(
            mode,
            store,
            crossmap,
            &work_ids,
            scored,
            match_log.as_deref(),
        )?;
        // The partition includes prematches among occupied B IDs. They are counted
        // under Auto, not Skip. Retain existing greedy counting independently.
        skipped -= matched.len();
        (outcomes, Some(summary))
    } else {
        (outcomes.into_iter().flatten().collect(), None)
    };

    // Flush match log and report errors on the legacy greedy path.
    if !graph && let Some(ref ml) = match_log {
        if let Err(e) = ml.flush() {
            tracing::warn!(error = %e, "match log flush failed");
        } else {
            tracing::info!(path = %ml.path().display(), "batch match log written");
        }
        if ml_errors.load(Ordering::Relaxed) {
            tracing::warn!("match log had write errors — output may be incomplete");
        }
    }

    for outcome in outcomes {
        match outcome {
            RecordOutcome::Scored(_) => {
                return Err(BatchAssignmentError::Invariant("unresolved scoring outcome").into());
            }
            RecordOutcome::Auto(mr) => matched.push(mr),
            RecordOutcome::Review(mr) => review.push(mr),
            RecordOutcome::NoMatch(id, rec, score) => unmatched.push((id, rec, score)),
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let stats = BatchStats {
        assignment,
        total_b,
        auto_matched: matched.len(),
        review_count: review.len(),
        no_match: unmatched.len(),
        skipped,
        elapsed_secs: elapsed,
        scoring_elapsed_secs: scoring_elapsed,
    };

    b_ids.truncate(total_b);
    Ok(BatchResult {
        processed_graph_b: graph.then_some(b_ids),
        matched,
        review,
        unmatched,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::crossmap::MemoryCrossMap;
    use crate::matching::exclusions::Exclusions;
    use crate::store::memory::MemoryStore;

    use super::*;

    fn make_config(blocking: bool) -> Result<Config, Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.yaml");
        std::fs::write(
            &path,
            format!(
                r#"
job:
  name: batch_bm25_test
vector_backend: flat
datasets:
  a: {{ path: a.csv, id_field: id, format: csv }}
  b: {{ path: b.csv, id_field: id, format: csv }}
cross_map:
  path: crossmap.csv
  a_id_field: id
  b_id_field: id
embeddings:
  model: test
  a_cache_dir: cache
blocking:
  enabled: {blocking}
  fields:
    - field_a: region
      field_b: region
match_fields:
  - method: bm25
    weight: 1.0
    fields:
      - field_a: name
        field_b: name
thresholds:
  auto_match: 0.85
  review_floor: 0.60
output:
  csv_dir_path: output
"#
            ),
        )?;
        Ok(crate::config::load_config(&path)?)
    }

    fn make_record(id: &str, name: &str, region: &str) -> Record {
        [("id", id), ("name", name), ("region", region)]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    fn make_pool(size: usize) -> HashMap<String, Record> {
        let mut records = HashMap::new();
        if size > 0 {
            records.insert("a1".into(), make_record("a1", "quartz cobalt", "west"));
        }
        if size > 1 {
            records.insert("a2".into(), make_record("a2", "saffron orchid", "east"));
        }
        for i in 2..size {
            let id = format!("filler{i}");
            records.insert(id.clone(), make_record(&id, "unrelated filler", "other"));
        }
        records
    }

    #[test]
    fn batch_bm25_unblocked_blocked_and_empty_pools() -> Result<(), Box<dyn Error>> {
        // Exercise both small pools and the large-pool BM25 search path.
        for size in [2, 5002] {
            for (blocking, empty_block, empty_pool) in [
                (false, false, false),
                (true, false, false),
                (true, true, false),
                (false, false, true),
                (true, false, true),
            ] {
                let config = make_config(blocking)?;
                let records_b = HashMap::from([
                    (
                        "b1".into(),
                        make_record(
                            "b1",
                            "quartz cobalt",
                            if empty_block { "missing" } else { "west" },
                        ),
                    ),
                    (
                        "b2".into(),
                        make_record(
                            "b2",
                            "saffron orchid",
                            if empty_block { "missing" } else { "east" },
                        ),
                    ),
                ]);
                let store = MemoryStore::from_records(
                    make_pool(if empty_pool { 0 } else { size }),
                    records_b,
                    &config.blocking,
                );
                let crossmap = MemoryCrossMap::new();
                let result = run_batch(
                    &config,
                    &store,
                    None,
                    None,
                    &crossmap,
                    &Exclusions::new(),
                    None,
                    true,
                    None,
                    None,
                )?;
                assert_eq!(result.stats.total_b, 2, "all B queries must be processed");
                assert!(result.review.is_empty(), "no review scores expected");
                if empty_block || empty_pool {
                    assert!(
                        result.matched.is_empty(),
                        "empty candidate pools cannot match"
                    );
                    assert_eq!(result.unmatched.len(), 2, "both queries must be unmatched");
                    assert!(
                        result.unmatched.iter().all(|(_, _, score)| score.is_none()),
                        "empty pools must not produce candidate scores"
                    );
                    assert_eq!(crossmap.get_a("b1"), None, "no claim for b1");
                    assert_eq!(crossmap.get_a("b2"), None, "no claim for b2");
                } else {
                    let mut links: Vec<_> = result
                        .matched
                        .iter()
                        .map(|m| (m.query_id.as_str(), m.matched_id.as_str()))
                        .collect();
                    links.sort_unstable();
                    assert_eq!(
                        links,
                        vec![("b1", "a1"), ("b2", "a2")],
                        "BM25 links must agree for size={size}, blocking={blocking}"
                    );
                    assert!(
                        result.matched.iter().all(|m| (m.score - 1.0).abs() < 1e-6),
                        "identical text must retain normalized self-score"
                    );
                    assert!(result.unmatched.is_empty(), "both queries should match");
                    assert_eq!(crossmap.get_a("b1").as_deref(), Some("a1"), "claim b1");
                    assert_eq!(crossmap.get_a("b2").as_deref(), Some("a2"), "claim b2");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn batch_bm25_preserves_flat_vector_candidates() -> Result<(), Box<dyn Error>> {
        let mut config = make_config(false)?;
        config.vector_backend = "flat".into();
        config.ann_candidates = Some(1);
        config.top_n = Some(1);
        config.thresholds.auto_match = 0.7;
        config.match_fields[0].weight = 0.25;
        config.match_fields.push(serde_yaml::from_str(
            "{field_a: name, field_b: name, method: embedding, weight: 0.75}",
        )?);
        let query = make_record("b1", "quartz cobalt", "west");
        let store = MemoryStore::from_records(
            make_pool(2),
            HashMap::from([("b1".into(), query.clone())]),
            &config.blocking,
        );
        let index_a = crate::vectordb::flat::FlatVectorDB::new(2);
        let index_b = crate::vectordb::flat::FlatVectorDB::new(2);
        let scale = 0.75_f32.sqrt();
        for (id, vector) in [("a1", [0.0, scale]), ("a2", [scale, 0.0])] {
            let record = store.get(Side::A, id)?.ok_or("missing test A record")?;
            index_a.upsert(id, &vector, &record, Side::A)?;
        }
        index_b.upsert("b1", &[scale, 0.0], &query, Side::B)?;
        let crossmap = MemoryCrossMap::new();
        let result = run_batch(
            &config,
            &store,
            Some(&index_a),
            Some(&index_b),
            &crossmap,
            &Exclusions::new(),
            None,
            true,
            None,
            None,
        )?;
        assert_eq!(result.matched.len(), 1, "vector candidate must be retained");
        assert_eq!(
            result.matched[0].matched_id, "a2",
            "flat search must find a candidate absent from BM25 results"
        );
        assert!(
            (result.matched[0].score - 0.75).abs() < 1e-6,
            "weighted embedding score must be preserved"
        );
        Ok(())
    }

    #[test]
    fn batch_bm25_respects_nonempty_blocks() -> Result<(), Box<dyn Error>> {
        let config = make_config(true)?;
        // The globally perfect match is outside the block; the nonempty block
        // contains only unrelated text, so unrestricted BM25 must not be used.
        let store = MemoryStore::from_records(
            make_pool(2),
            HashMap::from([("b1".into(), make_record("b1", "quartz cobalt", "east"))]),
            &config.blocking,
        );
        let crossmap = MemoryCrossMap::new();
        let result = run_batch(
            &config,
            &store,
            None,
            None,
            &crossmap,
            &Exclusions::new(),
            None,
            true,
            None,
            None,
        )?;
        assert!(result.matched.is_empty(), "cannot match outside the block");
        assert!(
            result.review.is_empty(),
            "unrelated block has no review candidate"
        );
        assert_eq!(result.unmatched.len(), 1, "query must remain unmatched");
        assert_eq!(result.unmatched[0].2, None, "no BM25 overlap inside block");
        assert_eq!(crossmap.get_a("b1"), None, "no out-of-block claim");
        Ok(())
    }

    #[test]
    fn batch_bm25_preserves_exclusions_and_atomic_claims() -> Result<(), Box<dyn Error>> {
        for blocking in [false, true] {
            let config = make_config(blocking)?;
            let store = MemoryStore::from_records(
                make_pool(5002),
                HashMap::from([
                    ("b1".into(), make_record("b1", "quartz cobalt", "west")),
                    ("b2".into(), make_record("b2", "quartz cobalt", "west")),
                    ("b3".into(), make_record("b3", "saffron orchid", "east")),
                ]),
                &config.blocking,
            );
            let crossmap = MemoryCrossMap::new();
            let exclusions = Exclusions::new();
            exclusions.add("a2", "b3");
            let result = run_batch(
                &config,
                &store,
                None,
                None,
                &crossmap,
                &exclusions,
                None,
                true,
                None,
                None,
            )?;
            assert_eq!(result.matched.len(), 1, "only one query can claim a1");
            assert_eq!(
                result.matched[0].matched_id, "a1",
                "excluded a2 cannot match"
            );
            assert!(result.review.is_empty(), "no review candidates expected");
            assert_eq!(
                result.unmatched.len(),
                2,
                "excluded query and claim loser are unmatched"
            );
            assert_eq!(
                crossmap.has_b("b1") as usize + crossmap.has_b("b2") as usize,
                1,
                "parallel claims must preserve the bijection"
            );
            assert!(!crossmap.has_a("a2"), "excluded A must remain unclaimed");
            assert!(!crossmap.has_b("b3"), "excluded B must remain unclaimed");
            assert!(
                result
                    .unmatched
                    .iter()
                    .any(|(id, _, score)| id == "b3" && score.is_none()),
                "excluded candidate must be filtered before scoring"
            );
        }
        Ok(())
    }
}
