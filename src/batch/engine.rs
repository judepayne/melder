//! Batch matching engine.
//!
//! For each B record:
//! 1. Skip if already in CrossMap
//! 2. Encode B record's embedding text (batched for throughput)
//! 3. Generate candidates from A pool (parallelized with Rayon)
//! 4. Score candidates
//! 5. Classify top result; if auto_match → add to CrossMap

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::config::Config;
use crate::crossmap::CrossMap;
use crate::data;
use crate::encoder::EncoderPool;
use crate::error::MelderError;
use crate::index::VecIndex;
use crate::matching::blocking::BlockingIndex;
use crate::matching::{candidates, engine as match_engine};
use crate::models::{Classification, MatchResult, Record, Side};
use crate::state::state::primary_embedding_text;

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

/// Batch size for ONNX encoding — amortizes session overhead.
const ENCODE_BATCH_SIZE: usize = 64;

/// Outcome for a single B record after scoring.
enum RecordOutcome {
    Auto(MatchResult),
    Review(MatchResult),
    NoMatch(String, Record),
}

/// Run the batch matching engine.
///
/// Loads B records from CSV, processes each against the pre-built A index.
/// If `b_index_cache` is configured, B embeddings are cached to disk so
/// subsequent runs (e.g. with different thresholds) skip ONNX encoding.
pub fn run_batch(
    config: &Config,
    records_a: &HashMap<String, Record>,
    index_a: &VecIndex,
    encoder_pool: &EncoderPool,
    crossmap: &mut CrossMap,
    limit: Option<usize>,
) -> Result<BatchResult, MelderError> {
    let start = Instant::now();

    // Build blocking index from A records if blocking is enabled
    let bi: Option<BlockingIndex> = if config.blocking.enabled {
        let bi_start = Instant::now();
        let mut bi = BlockingIndex::from_config(&config.blocking);
        for (id, record) in records_a {
            bi.insert(id, record, Side::A);
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

    // Load B records
    let (b_records, b_ids) = data::load_dataset(
        Path::new(&config.datasets.b.path),
        &config.datasets.b.id_field,
        &config.required_fields_b,
        config.datasets.b.format.as_deref(),
    )
    .map_err(MelderError::Data)?;

    eprintln!("Loaded {} B records for batch matching", b_records.len());

    let total_b = if let Some(lim) = limit {
        lim.min(b_ids.len())
    } else {
        b_ids.len()
    };

    // Build or load B-side VecIndex (for embedding cache).
    // If b_index_cache is configured and fresh, load from cache (fast).
    // Otherwise encode all B records and build the index from scratch.
    let b_index_cache_path = config.embeddings.b_index_cache.as_deref();
    let index_b =
        build_or_load_b_index(b_index_cache_path, &b_records, &b_ids, config, encoder_pool)?;

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
        // Build reverse index: common_id_value -> a_id
        let mut a_common_index: HashMap<String, String> = HashMap::new();
        for (a_id, a_rec) in records_a {
            if let Some(val) = a_rec.get(a_cid_field) {
                let val = val.trim();
                if !val.is_empty() {
                    a_common_index.insert(val.to_string(), a_id.clone());
                }
            }
        }

        let mut common_count = 0usize;
        for b_id in b_ids.iter().take(total_b) {
            if crossmap.has_b(b_id) {
                continue;
            }
            if let Some(b_rec) = b_records.get(b_id) {
                if let Some(b_val) = b_rec.get(b_cid_field) {
                    let b_val = b_val.trim();
                    if !b_val.is_empty() {
                        if let Some(a_id) = a_common_index.get(b_val) {
                            // Immediate match
                            crossmap.add(a_id, b_id);
                            let mr = MatchResult {
                                query_id: b_id.clone(),
                                matched_id: a_id.clone(),
                                query_side: Side::B,
                                score: 1.0,
                                field_scores: vec![],
                                classification: Classification::Auto,
                                matched_record: records_a.get(a_id).cloned(),
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
    let _ = common_id_matched; // used in summary stats

    // Process B records — vectors come from the pre-built index_b
    let b_ids_slice = &b_ids[..total_b];
    let mut processed = 0;

    for chunk_ids in b_ids_slice.chunks(ENCODE_BATCH_SIZE) {
        // 1. Partition: skip already-crossmapped, collect those needing work
        let mut work_items: Vec<(&str, Vec<f32>)> = Vec::with_capacity(chunk_ids.len());
        for b_id in chunk_ids {
            if crossmap.has_b(b_id) {
                skipped += 1;
            } else if b_records.contains_key(b_id) {
                if let Some(vec) = index_b.get(b_id) {
                    work_items.push((b_id.as_str(), vec.to_vec()));
                }
            }
        }

        if work_items.is_empty() {
            processed += chunk_ids.len();
            print_progress(processed, total_b, &start);
            continue;
        }

        // 2. Score in parallel: candidate generation + scoring is read-only
        //    on shared state (records_a, index_a, config, blocking_index)
        let outcomes: Vec<RecordOutcome> = work_items
            .par_iter()
            .filter_map(|(b_id, b_vec)| {
                let b_record = b_records.get(*b_id)?;

                let cands = candidates::generate_candidates_batch_indexed(
                    b_id, b_record, b_vec, index_a, records_a, config, crossmap, bi_ref,
                );

                if cands.is_empty() {
                    return Some(RecordOutcome::NoMatch(b_id.to_string(), b_record.clone()));
                }

                let results = match_engine::score_candidates(
                    b_id,
                    b_record,
                    Some(b_vec),
                    Side::B,
                    &cands,
                    config,
                );

                if let Some(top) = results.into_iter().next() {
                    match top.classification {
                        Classification::Auto => Some(RecordOutcome::Auto(top)),
                        Classification::Review => Some(RecordOutcome::Review(top)),
                        Classification::NoMatch => {
                            Some(RecordOutcome::NoMatch(b_id.to_string(), b_record.clone()))
                        }
                    }
                } else {
                    Some(RecordOutcome::NoMatch(b_id.to_string(), b_record.clone()))
                }
            })
            .collect();

        // 3. Sequential: collect results and update crossmap
        for outcome in outcomes {
            match outcome {
                RecordOutcome::Auto(mr) => {
                    crossmap.add(&mr.matched_id, &mr.query_id);
                    matched.push(mr);
                }
                RecordOutcome::Review(mr) => {
                    review.push(mr);
                }
                RecordOutcome::NoMatch(id, rec) => {
                    unmatched.push((id, rec));
                }
            }
        }

        processed += chunk_ids.len();
        print_progress(processed, total_b, &start);
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

    Ok(BatchResult {
        matched,
        review,
        unmatched,
        stats,
    })
}

/// Build the B-side VecIndex: load from cache if fresh, otherwise encode
/// all B records and optionally save the cache.
fn build_or_load_b_index(
    cache_path: Option<&str>,
    b_records: &HashMap<String, Record>,
    b_ids: &[String],
    config: &Config,
    encoder_pool: &EncoderPool,
) -> Result<VecIndex, MelderError> {
    use crate::index::cache;

    let dim = encoder_pool.dim();

    // Try loading from cache
    if let Some(path_str) = cache_path {
        let path = Path::new(path_str);
        if !cache::is_cache_stale(path, b_records.len()) {
            let load_start = Instant::now();
            match cache::load_index(path) {
                Ok(index) => {
                    eprintln!(
                        "Loaded B index from cache: {} vecs in {:.1}ms",
                        index.len(),
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                    return Ok(index);
                }
                Err(e) => {
                    eprintln!("Warning: B cache load failed ({}), rebuilding...", e);
                }
            }
        }
    }

    // Build from scratch
    eprintln!("Building B index ({} records)...", b_records.len());
    let build_start = Instant::now();

    let mut index = VecIndex::new(dim);
    let batch_size = 256;

    for (batch_idx, chunk) in b_ids.chunks(batch_size).enumerate() {
        let texts: Vec<String> = chunk
            .iter()
            .map(|id| {
                let record = b_records
                    .get(id)
                    .expect("id from b_ids must exist in b_records");
                primary_embedding_text(record, config, false)
            })
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let vecs = encoder_pool
            .encode(&text_refs)
            .map_err(MelderError::Encoder)?;

        for (id, vec) in chunk.iter().zip(vecs.into_iter()) {
            index.upsert(id, &vec);
        }

        let done = (batch_idx + 1) * batch_size;
        if done % 1000 == 0 || done >= b_ids.len() {
            eprintln!(
                "  encoded {}/{} B records...",
                done.min(b_ids.len()),
                b_ids.len()
            );
        }
    }

    let build_elapsed = build_start.elapsed();
    eprintln!(
        "B index built: {} vecs in {:.1}s ({:.0} records/sec)",
        index.len(),
        build_elapsed.as_secs_f64(),
        index.len() as f64 / build_elapsed.as_secs_f64()
    );

    // Save cache if path configured
    if let Some(path_str) = cache_path {
        let path = Path::new(path_str);
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        if let Err(e) = cache::save_index(path, &index) {
            eprintln!("Warning: failed to save B index cache: {}", e);
        } else {
            eprintln!("Saved B index cache to {}", path_str);
        }
    }

    Ok(index)
}

fn print_progress(processed: usize, total: usize, start: &Instant) {
    if processed % 100 == 0 || processed == total {
        let elapsed = start.elapsed().as_secs_f64();
        let rate = processed as f64 / elapsed;
        let eta = if processed < total {
            (total - processed) as f64 / rate
        } else {
            0.0
        };
        eprintln!(
            "  processed {}/{} B records ({:.0} rec/s, ETA {:.0}s)",
            processed, total, rate, eta
        );
    }
}
