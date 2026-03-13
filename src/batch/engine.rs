//! Batch matching engine.
//!
//! Pipeline per B record (parallelized with Rayon):
//! 1. Common ID pre-match (optional, runs first for all records)
//! 2. Skip if already in CrossMap
//! 3. Blocking filter → candidate selection → full scoring (via shared pipeline)
//! 4. Classify top result; if auto_match → try to claim in CrossMap

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use dashmap::DashMap;
use rayon::prelude::*;

use crate::config::Config;
use crate::crossmap::CrossMap;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::matching::blocking::BlockingIndex;
use crate::matching::pipeline;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::vectordb::{self, VectorDB};

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
    pub elapsed_secs: f64,
}

/// Outcome for a single B record after scoring.
enum RecordOutcome {
    Auto(MatchResult),
    Review(MatchResult),
    NoMatch(String, Record),
}

/// Run the batch matching engine.
///
/// Loads B records from csv, processes each against the pre-built A-side data.
/// If `b_cache_dir` is configured, B combined index is cached to disk so
/// subsequent runs (e.g. with different thresholds) skip ONNX encoding.
pub fn run_batch(
    config: &Config,
    records_a: &DashMap<String, Record>,
    combined_index_a: Option<&dyn VectorDB>,
    encoder_pool: &EncoderPool,
    crossmap: &CrossMap,
    limit: Option<usize>,
) -> Result<BatchResult, MelderError> {
    let start = Instant::now();

    // Build blocking index from A records if blocking is enabled
    let bi: Option<BlockingIndex> = if config.blocking.enabled {
        let bi_start = Instant::now();
        let mut bi = BlockingIndex::from_config(&config.blocking);
        for entry in records_a.iter() {
            bi.insert(entry.key(), entry.value(), Side::A);
        }
        eprintln!(
            "Built blocking index for {} A records in {:.1}ms",
            records_a.len(),
            bi_start.elapsed().as_secs_f64() * 1000.0
        );
        Some(bi)
    } else {
        None
    };
    let bi_ref = bi.as_ref();

    // Load B records (into HashMap from data loaders, then convert to DashMap)
    let (b_records_map, b_ids) = data::load_dataset(
        Path::new(&config.datasets.b.path),
        &config.datasets.b.id_field,
        &config.required_fields_b,
        config.datasets.b.format.as_deref(),
    )
    .map_err(MelderError::Data)?;

    eprintln!(
        "Loaded {} B records for batch matching",
        b_records_map.len()
    );

    let total_b = if let Some(lim) = limit {
        lim.min(b_ids.len())
    } else {
        b_ids.len()
    };

    // Build or load B-side combined embedding index.
    let combined_index_b_owned = vectordb::build_or_load_combined_index(
        &config.vector_backend,
        config.embeddings.b_cache_dir.as_deref(),
        &b_records_map,
        &b_ids,
        config,
        false,
        encoder_pool,
        false,
    )?;
    let combined_index_b: Option<&dyn VectorDB> = combined_index_b_owned.as_deref();

    // Convert B records to DashMap for shared pipeline use
    let b_records: DashMap<String, Record> = DashMap::with_capacity(b_records_map.len());
    for (id, rec) in b_records_map {
        b_records.insert(id, rec);
    }

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
        for entry in records_a.iter() {
            if let Some(val) = entry.value().get(a_cid_field) {
                let val = val.trim();
                if !val.is_empty() {
                    a_common_index.insert(val.to_string(), entry.key().clone());
                }
            }
        }

        let mut common_count = 0usize;
        for b_id in b_ids.iter().take(total_b) {
            if crossmap.has_b(b_id) {
                continue;
            }
            if let Some(b_entry) = b_records.get(b_id) {
                if let Some(b_val) = b_entry.value().get(b_cid_field) {
                    let b_val = b_val.trim();
                    if !b_val.is_empty() {
                        if let Some(a_id) = a_common_index.get(b_val) {
                            crossmap.add(a_id, b_id);
                            let a_rec = records_a.get(a_id).map(|e| e.value().clone());
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
        } else if b_records.contains_key(b_id) {
            work_ids.push(b_id.as_str());
        }
    }

    // Pre-compute top_n from config (0 = no limit in batch → best result only).
    let top_n = config.top_n.unwrap_or(5);

    // Precompute embedding specs once for combined-vec lookup during scoring.
    let emb_specs = vectordb::embedding_field_specs(config);

    // Score all B records in a single parallel pass.
    let progress = AtomicUsize::new(0);
    let work_total = work_ids.len();

    let outcomes: Vec<RecordOutcome> = work_ids
        .par_iter()
        .filter_map(|b_id| {
            let b_record = b_records.get(*b_id)?;

            // Progress reporting (atomic, lock-free)
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 1000 == 0 || done == work_total {
                let elapsed = start.elapsed().as_secs_f64();
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

            // Pre-query the blocking index (lock-free for the rest of scoring).
            let blocked_ids: Vec<String> = if config.blocking.enabled {
                if let Some(bi) = bi_ref {
                    bi.query(b_record.value(), Side::B).into_iter().collect()
                } else {
                    records_a.iter().map(|e| e.key().clone()).collect()
                }
            } else {
                records_a.iter().map(|e| e.key().clone()).collect()
            };

            // Fetch the query combined vec from the B-side index.
            let query_combined_vec: Vec<f32> = combined_index_b
                .and_then(|idx| idx.get(b_id).ok().flatten())
                .unwrap_or_default();

            // Shared pipeline: candidate selection → full scoring
            let results = pipeline::score_pool(
                b_id,
                b_record.value(),
                Side::B,
                &query_combined_vec,
                records_a,
                combined_index_a,
                &blocked_ids,
                config,
                top_n,
            );

            // Claim loop: try each auto-match candidate in ranked order.
            // Uses crossmap.claim() so concurrent threads can't double-match.
            let mut outcome = None;
            for mut result in results {
                if result.score < config.thresholds.review_floor {
                    outcome = Some(RecordOutcome::NoMatch(
                        b_id.to_string(),
                        b_record.value().clone(),
                    ));
                    break;
                }
                if result.score >= config.thresholds.auto_match {
                    let a_id = &result.matched_id;
                    if crossmap.claim(a_id, b_id) {
                        // Attach matched record
                        if result.matched_record.is_none() {
                            if let Some(a_entry) = records_a.get(a_id) {
                                result.matched_record = Some(a_entry.value().clone());
                            }
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
                    b_record.value().clone(),
                ))
            })
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
    let stats = BatchStats {
        total_b,
        auto_matched: matched.len(),
        review_count: review.len(),
        no_match: unmatched.len(),
        skipped,
        elapsed_secs: elapsed,
    };

    let _ = emb_specs;

    Ok(BatchResult {
        matched,
        review,
        unmatched,
        stats,
    })
}
