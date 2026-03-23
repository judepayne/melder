//! Session: the core live matching logic.
//!
//! Provides upsert, try-match, and crossmap management operations.
//! All operations are synchronous internally; the HTTP layer wraps them
//! in `spawn_blocking` as needed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::error::SessionError;
use crate::matching::pipeline;
use crate::matching::pipeline::Bm25Ctx;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::state::live::{LiveMatchState, ReviewEntry, review_queue_key};
use crate::state::upsert_log::WalEvent;

/// Response types for API serialization.
pub mod response {
    use serde::Serialize;

    use crate::models::Side;

    #[derive(Debug, Serialize)]
    pub struct MatchEntry {
        pub id: String,
        pub score: f64,
        pub classification: String,
        pub field_scores: Vec<FieldScoreEntry>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub matched_record: Option<std::collections::HashMap<String, String>>,
    }

    #[derive(Debug, Serialize)]
    pub struct FieldScoreEntry {
        pub field_a: String,
        pub field_b: String,
        pub method: String,
        pub score: f64,
        pub weight: f64,
    }

    #[derive(Debug, Serialize)]
    pub struct OldMapping {
        pub a_id: String,
        pub b_id: String,
    }

    #[derive(Debug, Serialize)]
    pub struct UpsertResponse {
        pub status: String,
        pub id: String,
        pub side: Side,
        pub classification: String,
        pub from_crossmap: bool,
        pub matches: Vec<MatchEntry>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub old_mapping: Option<OldMapping>,
    }

    #[derive(Debug, Serialize)]
    pub struct MatchResponse {
        pub status: String,
        pub id: String,
        pub side: Side,
        pub classification: String,
        pub from_crossmap: bool,
        pub matches: Vec<MatchEntry>,
    }

    #[derive(Debug, Serialize)]
    pub struct ConfirmResponse {
        pub status: String,
    }

    #[derive(Debug, Serialize)]
    pub struct LookupResponse {
        pub id: String,
        pub side: Side,
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub paired_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub matched_record: Option<std::collections::HashMap<String, String>>,
    }

    #[derive(Debug, Serialize)]
    pub struct BreakResponse {
        pub status: String,
        pub a_id: String,
        pub b_id: String,
    }

    #[derive(Debug, Serialize)]
    pub struct RemoveResponse {
        pub status: String,
        pub id: String,
        pub side: Side,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub crossmap_broken: Vec<String>,
    }

    #[derive(Debug, Serialize)]
    pub struct QueryResponse {
        pub id: String,
        pub side: Side,
        pub record: std::collections::HashMap<String, String>,
        pub crossmap: QueryCrossmap,
    }

    #[derive(Debug, Serialize)]
    pub struct QueryCrossmap {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub paired_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub paired_record: Option<std::collections::HashMap<String, String>>,
    }

    #[derive(Debug, Serialize)]
    pub struct HealthResponse {
        pub status: String,
        pub model: String,
        pub records_a: usize,
        pub records_b: usize,
        pub crossmap_entries: usize,
    }

    #[derive(Debug, Serialize)]
    pub struct StatusResponse {
        pub job: String,
        pub uptime_seconds: f64,
        pub upserts: u64,
        pub matches: u64,
    }

    // Batch response wrappers

    #[derive(Debug, Serialize)]
    pub struct BatchUpsertResponse {
        pub results: Vec<UpsertResponse>,
    }

    #[derive(Debug, Serialize)]
    pub struct BatchMatchResponse {
        pub results: Vec<MatchResponse>,
    }

    #[derive(Debug, Serialize)]
    pub struct BatchRemoveResponse {
        pub results: Vec<RemoveResponse>,
    }

    // --- Crossmap, unmatched, stats, and review responses ---

    #[derive(Debug, Serialize)]
    pub struct CrossmapPairEntry {
        pub a_id: String,
        pub b_id: String,
    }

    #[derive(Debug, Serialize)]
    pub struct CrossmapPairsResponse {
        pub total: usize,
        pub offset: usize,
        pub pairs: Vec<CrossmapPairEntry>,
    }

    #[derive(Debug, Serialize)]
    pub struct UnmatchedEntry {
        pub id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub record: Option<std::collections::HashMap<String, String>>,
    }

    #[derive(Debug, Serialize)]
    pub struct UnmatchedResponse {
        pub side: Side,
        pub total: usize,
        pub offset: usize,
        pub records: Vec<UnmatchedEntry>,
    }

    #[derive(Debug, Serialize)]
    pub struct CrossmapStatsResponse {
        pub records_a: usize,
        pub records_b: usize,
        pub crossmap_pairs: usize,
        pub matched_a: usize,
        pub matched_b: usize,
        pub unmatched_a: usize,
        pub unmatched_b: usize,
        pub coverage_a: f64,
        pub coverage_b: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    pub struct ReviewListEntry {
        pub id: String,
        pub side: Side,
        pub candidate_id: String,
        pub score: f64,
    }

    #[derive(Debug, Serialize)]
    pub struct ReviewListResponse {
        pub total: usize,
        pub offset: usize,
        pub reviews: Vec<ReviewListEntry>,
    }
}

use response::*;

/// Live matching session.
pub struct Session {
    pub state: Arc<LiveMatchState>,
    pub start_time: Instant,
    pub upsert_count: AtomicU64,
    pub match_count: AtomicU64,
    /// Pre-computed embedding field specs (avoids per-request allocation).
    emb_specs: Vec<(String, String, f64)>,
}

impl Session {
    pub fn new(state: Arc<LiveMatchState>) -> Self {
        let emb_specs = crate::vectordb::embedding_field_specs(&state.config);
        Self {
            state,
            start_time: Instant::now(),
            upsert_count: AtomicU64::new(0),
            match_count: AtomicU64::new(0),
            emb_specs,
        }
    }

    /// Upsert a record into the live state and attempt matching.
    ///
    /// Encodes the combined embedding vector, stores it in the combined index,
    /// then performs insertion and scoring.
    ///
    /// When an `EncoderCoordinator` is configured, encoding is batched with
    /// other concurrent requests via the coordinator (requires a tokio runtime
    /// context — this is always the case since handlers call via
    /// `spawn_blocking`). Otherwise falls back to direct `EncoderPool` encoding.
    pub fn upsert_record(
        &self,
        side: Side,
        record: Record,
    ) -> Result<UpsertResponse, SessionError> {
        let config = &self.state.config;
        let is_a_side = side == Side::A;
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };
        let id = record
            .get(id_field)
            .cloned()
            .ok_or_else(|| SessionError::MissingField {
                field: id_field.clone(),
            })?;

        // Encode and store the combined embedding vector, unless the
        // embedding fields haven't changed (text-hash skip).
        let emb_specs = &self.emb_specs;
        if !emb_specs.is_empty() {
            let this_side = self.state.side(side);
            let needs_encoding = match this_side.combined_index {
                Some(ref idx) => {
                    let current_hash =
                        crate::vectordb::texthash::compute_text_hash(&record, emb_specs, side);
                    idx.text_hash_for(&id) != Some(current_hash)
                }
                None => true,
            };

            if needs_encoding {
                let combined_vec = self.encode_combined(&record, emb_specs, is_a_side)?;
                if !combined_vec.is_empty()
                    && let Some(ref idx) = this_side.combined_index
                {
                    let _ = idx.upsert(&id, &combined_vec, &record, side);
                }
            }
        }

        self.upsert_record_inner(side, record)
    }

    /// Encode a record's embedding fields into a combined vector.
    ///
    /// If a coordinator is available, routes encoding through it (batching
    /// with concurrent requests). Otherwise, calls `encode_combined_vector`
    /// directly on the encoder pool.
    fn encode_combined(
        &self,
        record: &Record,
        emb_specs: &[(String, String, f64)],
        is_a_side: bool,
    ) -> Result<Vec<f32>, SessionError> {
        if let Some(ref coordinator) = self.state.coordinator {
            // Coordinator path: encode each embedding field via the
            // coordinator and assemble the combined vector.
            let handle = tokio::runtime::Handle::current();
            let field_dim = self.state.encoder_pool.dim();
            let mut combined = Vec::with_capacity(field_dim * emb_specs.len());

            // Gather all field texts and submit them in one encode_many call.
            let texts: Vec<String> = emb_specs
                .iter()
                .map(|(field_a, field_b, _weight)| {
                    let field_name = if is_a_side { field_a } else { field_b };
                    record
                        .get(field_name)
                        .map(|v| v.trim().to_string())
                        .unwrap_or_default()
                })
                .collect();

            let vecs = handle
                .block_on(coordinator.encode_many(texts))
                .map_err(|e| melder_to_session_err(crate::error::MelderError::Encoder(e)))?;

            for (vec, (_fa, _fb, weight)) in vecs.into_iter().zip(emb_specs.iter()) {
                let sqrt_w = weight.sqrt() as f32;
                let scaled: Vec<f32> = vec.into_iter().map(|v| v * sqrt_w).collect();
                combined.extend_from_slice(&scaled);
            }
            Ok(combined)
        } else {
            // Direct path: no coordinator, encode synchronously.
            crate::vectordb::encode_combined_vector(
                record,
                emb_specs,
                &self.state.encoder_pool,
                is_a_side,
            )
            .map_err(melder_to_session_err)
        }
    }

    /// Upsert a record whose per-field vectors have already been stored in
    /// the combined index. Handles insertion, crossmap management, and scoring.
    fn upsert_record_inner(
        &self,
        side: Side,
        record: Record,
    ) -> Result<UpsertResponse, SessionError> {
        let config = &self.state.config;
        let store = &self.state.store;
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };

        // 1. Extract ID
        let id = record
            .get(id_field)
            .cloned()
            .ok_or_else(|| SessionError::MissingField {
                field: id_field.clone(),
            })?;
        if id.trim().is_empty() {
            return Err(SessionError::EmptyId);
        }

        let opp = side.opposite();

        // 2. Check if existing record
        let mut old_mapping: Option<OldMapping> = None;
        let mut status = "added";

        if store.contains(side, &id) {
            status = "updated";

            // Clear stale review entries for this record (re-upsert
            // invalidates any prior review-band match).
            self.state.drain_reviews_for_id(&id);

            // Check crossmap — if matched, atomically read-and-remove the pair.
            let paired_id = match side {
                Side::A => self.state.crossmap.take_a(&id),
                Side::B => self.state.crossmap.take_b(&id),
            };

            if let Some(paired_id) = paired_id {
                let (a_id, b_id) = match side {
                    Side::A => (id.clone(), paired_id.clone()),
                    Side::B => (paired_id.clone(), id.clone()),
                };
                // Add both back to unmatched
                store.mark_unmatched(Side::A, &a_id);
                store.mark_unmatched(Side::B, &b_id);

                // WAL
                let _ = self.state.wal.append(&WalEvent::CrossMapBreak {
                    a_id: a_id.clone(),
                    b_id: b_id.clone(),
                });

                old_mapping = Some(OldMapping { a_id, b_id });
                self.state.mark_crossmap_dirty();
            }

            // Remove old record from blocking index
            if let Some(old_rec) = store.get(side, &id) {
                store.blocking_remove(side, &id, &old_rec);
            }
        }

        // 3. WAL append (zero-clone borrowing serialization).
        let _ = self.state.wal.append_upsert(side, &record);

        // 4. Insert/replace record
        store.insert(side, &id, &record);

        // 5. Add to unmatched
        store.mark_unmatched(side, &id);

        // 6. Update blocking index
        store.blocking_insert(side, &id, &record);

        // 6b. Update BM25 index
        if let Some(ref bm25_mtx) = self.state.side(side).bm25_index {
            let mut idx = bm25_mtx.write().unwrap_or_else(|e| e.into_inner());
            idx.upsert(&id, &record);
        }

        // 6c. Update synonym index
        if let Some(ref syn_mtx) = self.state.side(side).synonym_index {
            let mut idx = syn_mtx.write().unwrap_or_else(|e| e.into_inner());
            idx.upsert(&id, &record, side, &config.synonym_fields);
        }

        // 8. Update common_id_index and check for common ID match
        let this_cid_field = match side {
            Side::A => config.datasets.a.common_id_field.as_deref(),
            Side::B => config.datasets.b.common_id_field.as_deref(),
        };
        if let Some(cid_field) = this_cid_field {
            // Update this side's common_id_index
            if let Some(cid_val) = record.get(cid_field) {
                let cid_val = cid_val.trim();
                if !cid_val.is_empty() {
                    store.common_id_insert(side, cid_val, &id);

                    // Check opposite side for a matching common ID
                    if let Some(opp_id) = store.common_id_lookup(opp, cid_val) {
                        // Break any existing crossmap for either record.
                        if let Some(old_opp) = match side {
                            Side::A => self.state.crossmap.take_a(&id),
                            Side::B => self.state.crossmap.take_b(&id),
                        } {
                            store.mark_unmatched(opp, &old_opp);
                        }
                        if let Some(old_this) = match side {
                            Side::A => self.state.crossmap.take_b(&opp_id),
                            Side::B => self.state.crossmap.take_a(&opp_id),
                        } {
                            store.mark_unmatched(side, &old_this);
                        }

                        // Create the common ID match
                        let (a_id, b_id) = match side {
                            Side::A => (id.clone(), opp_id.clone()),
                            Side::B => (opp_id.clone(), id.clone()),
                        };
                        self.state.crossmap.add(&a_id, &b_id);
                        store.mark_matched(Side::A, &a_id);
                        store.mark_matched(Side::B, &b_id);

                        let _ = self.state.wal.append(&WalEvent::CrossMapConfirm {
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                            score: Some(1.0),
                        });
                        self.state.mark_crossmap_dirty();

                        let opp_record = store.get(opp, &opp_id);
                        let match_entry = MatchEntry {
                            id: opp_id,
                            score: 1.0,
                            classification: "auto".to_string(),
                            field_scores: vec![],
                            matched_record: opp_record,
                        };

                        self.upsert_count.fetch_add(1, Ordering::Relaxed);

                        return Ok(UpsertResponse {
                            status: status.to_string(),
                            id,
                            side,
                            classification: "auto".to_string(),
                            from_crossmap: false,
                            matches: vec![match_entry],
                            old_mapping,
                        });
                    }
                }
            }
        }

        // 9. Pipeline: blocking → candidate selection → full scoring
        let top_n = config.top_n.unwrap_or(5);
        let blocked_ids: Vec<String> = if config.blocking.enabled {
            store.blocking_query(&record, side)
        } else {
            store.ids(opp)
        };

        // Fetch the query combined vec from this side's index.
        let this_side = self.state.side(side);
        let opp_side = self.state.opposite_side(side);
        let query_combined_vec: Vec<f32> = this_side
            .combined_index
            .as_ref()
            .and_then(|idx| idx.get(&id).ok().flatten())
            .unwrap_or_default();
        let ann_candidates = config.ann_candidates.unwrap_or(50);
        let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);

        // Acquire opposite-side synonym index read lock (if configured).
        let opp_syn_guard = opp_side.synonym_index.as_ref().map(|mtx| {
            mtx.read().unwrap_or_else(|e| e.into_inner())
        });

        // Build BM25 context from opposite side's index.
        // Commit any buffered writes first (write lock, brief), then
        // query under a read lock so concurrent requests can score in parallel.
        let results = if let Some(ref opp_bm25_mtx) = opp_side.bm25_index {
            // Brief write lock for commit only
            {
                let mut w = opp_bm25_mtx.write().unwrap_or_else(|e| e.into_inner());
                w.commit_if_dirty();
            }
            // Read lock for query — concurrent readers allowed
            let guard = opp_bm25_mtx.read().unwrap_or_else(|e| e.into_inner());
            let query_text = guard.query_text_for(&record, side);
            let ctx = Bm25Ctx::new(&guard, query_text);
            pipeline::score_pool(
                &id,
                &record,
                side,
                &query_combined_vec,
                store.as_ref(),
                opp,
                opp_side.combined_index.as_deref(),
                &blocked_ids,
                config,
                ann_candidates,
                bm25_candidates_n,
                top_n,
                Some(ctx),
                opp_syn_guard.as_deref(),
            )
        } else {
            pipeline::score_pool(
                &id,
                &record,
                side,
                &query_combined_vec,
                store.as_ref(),
                opp,
                opp_side.combined_index.as_deref(),
                &blocked_ids,
                config,
                ann_candidates,
                bm25_candidates_n,
                top_n,
                None,
                opp_syn_guard.as_deref(),
            )
        };

        // 10. Claim loop: try candidates in ranked order.
        let mut classification = "no_match".to_string();
        let mut _claimed_idx: Option<usize> = None;

        for (i, result) in results.iter().enumerate() {
            if result.score < config.thresholds.review_floor {
                break;
            }
            if result.score >= config.thresholds.auto_match {
                let (a_id, b_id) = match side {
                    Side::A => (id.clone(), result.matched_id.clone()),
                    Side::B => (result.matched_id.clone(), id.clone()),
                };
                if self.state.crossmap.claim(&a_id, &b_id) {
                    store.mark_matched(Side::A, &a_id);
                    store.mark_matched(Side::B, &b_id);
                    let _ = self.state.wal.append(&WalEvent::CrossMapConfirm {
                        a_id,
                        b_id,
                        score: Some(result.score),
                    });
                    self.state.mark_crossmap_dirty();
                    classification = "auto".to_string();
                    _claimed_idx = Some(i);
                    break;
                }
                continue;
            }
            // Score is in review band
            let _ = self.state.wal.append(&WalEvent::ReviewMatch {
                id: id.clone(),
                side,
                candidate_id: result.matched_id.clone(),
                score: result.score,
            });
            let key = review_queue_key(side, &id, &result.matched_id);
            self.state.insert_review(
                key,
                ReviewEntry {
                    id: id.clone(),
                    side,
                    candidate_id: result.matched_id.clone(),
                    score: result.score,
                },
            );
            classification = "review".to_string();
            break;
        }

        let matches = build_match_entries(&results);

        self.upsert_count.fetch_add(1, Ordering::Relaxed);

        Ok(UpsertResponse {
            status: status.to_string(),
            id,
            side,
            classification,
            from_crossmap: false,
            matches,
            old_mapping,
        })
    }

    /// Remove a record by ID. Breaks any crossmap pair, removes from all indices.
    pub fn remove_record(&self, side: Side, id: &str) -> Result<RemoveResponse, SessionError> {
        let store = &self.state.store;
        let opp = side.opposite();

        // Check the record exists
        let record = store
            .get(side, id)
            .ok_or_else(|| SessionError::MissingField {
                field: format!("record {} not found on side {:?}", id, side),
            })?;

        // Break any crossmap pair
        let paired = match side {
            Side::A => self.state.crossmap.get_b(id),
            Side::B => self.state.crossmap.get_a(id),
        };
        let broken: Vec<String> = if let Some(paired_id) = paired {
            match side {
                Side::A => self.state.crossmap.remove(id, &paired_id),
                Side::B => self.state.crossmap.remove(&paired_id, id),
            }
            store.mark_unmatched(opp, &paired_id);
            self.state.mark_crossmap_dirty();
            vec![paired_id]
        } else {
            vec![]
        };

        // Drain any review entries involving this record.
        self.state.drain_reviews_for_id(id);

        // Remove from blocking index
        store.blocking_remove(side, id, &record);

        // Remove from combined index
        if let Some(ref idx) = self.state.side(side).combined_index {
            let _ = idx.remove(id);
        }

        // Remove from BM25 index
        if let Some(ref bm25_mtx) = self.state.side(side).bm25_index {
            let mut idx = bm25_mtx.write().unwrap_or_else(|e| e.into_inner());
            idx.remove(id);
        }

        // Remove from synonym index
        if let Some(ref syn_mtx) = self.state.side(side).synonym_index {
            let mut idx = syn_mtx.write().unwrap_or_else(|e| e.into_inner());
            idx.remove(id, side);
        }

        // Remove from unmatched set
        store.mark_matched(side, id);

        // Remove from common_id_index if configured
        let cid_field = match side {
            Side::A => self.state.config.datasets.a.common_id_field.as_deref(),
            Side::B => self.state.config.datasets.b.common_id_field.as_deref(),
        };
        if let Some(cid_field) = cid_field
            && let Some(cid_val) = record.get(cid_field)
        {
            let cid_val = cid_val.trim();
            if !cid_val.is_empty() {
                store.common_id_remove(side, cid_val);
            }
        }

        // Remove from records
        store.remove(side, id);

        // Append to WAL
        let _ = self.state.wal.append(&WalEvent::RemoveRecord {
            side,
            id: id.to_string(),
        });

        Ok(RemoveResponse {
            status: "removed".to_string(),
            id: id.to_string(),
            side,
            crossmap_broken: broken,
        })
    }

    /// Try matching a record without persisting it (read-only).
    pub fn try_match(&self, side: Side, record: Record) -> Result<MatchResponse, SessionError> {
        let config = &self.state.config;
        let is_a_side = side == Side::A;
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };
        let id = record.get(id_field).cloned().unwrap_or_default();

        // Encode and store combined embedding vector, with text-hash skip.
        let emb_specs = &self.emb_specs;
        if !emb_specs.is_empty() {
            let this_side = self.state.side(side);
            let needs_encoding = match this_side.combined_index {
                Some(ref idx) => {
                    let current_hash =
                        crate::vectordb::texthash::compute_text_hash(&record, emb_specs, side);
                    idx.text_hash_for(&id) != Some(current_hash)
                }
                None => true,
            };

            if needs_encoding {
                let combined_vec = self.encode_combined(&record, emb_specs, is_a_side)?;
                if !combined_vec.is_empty() {
                    let this_side = self.state.side(side);
                    if let Some(ref idx) = this_side.combined_index {
                        let _ = idx.upsert(&id, &combined_vec, &record, side);
                    }
                }
            }
        }

        self.try_match_inner(side, record, &id)
    }

    /// Try matching a record whose per-field vectors are already in the
    /// combined index (read-only scoring).
    fn try_match_inner(
        &self,
        side: Side,
        record: Record,
        id: &str,
    ) -> Result<MatchResponse, SessionError> {
        let config = &self.state.config;
        let store = &self.state.store;
        let opp = side.opposite();
        let opp_side = self.state.opposite_side(side);

        // Check crossmap for existing match
        let existing = match side {
            Side::A => self.state.crossmap.get_b(id),
            Side::B => self.state.crossmap.get_a(id),
        };

        if let Some(paired_id) = existing {
            // Return existing match from crossmap
            let matched_record = store.get(opp, &paired_id);

            let entry = MatchEntry {
                id: paired_id.clone(),
                score: 1.0,
                classification: "auto".to_string(),
                field_scores: vec![],
                matched_record,
            };

            self.match_count.fetch_add(1, Ordering::Relaxed);

            return Ok(MatchResponse {
                status: "already_matched".to_string(),
                id: id.to_string(),
                side,
                classification: "auto".to_string(),
                from_crossmap: true,
                matches: vec![entry],
            });
        }

        // Pipeline: blocking → candidate selection → full scoring
        let top_n = config.top_n.unwrap_or(5);
        let blocked_ids: Vec<String> = if config.blocking.enabled {
            store.blocking_query(&record, side)
        } else {
            store.ids(opp)
        };

        let this_side = self.state.side(side);
        // Fetch the query combined vec from this side's index.
        let query_combined_vec: Vec<f32> = this_side
            .combined_index
            .as_ref()
            .and_then(|idx| idx.get(id).ok().flatten())
            .unwrap_or_default();
        let ann_candidates = config.ann_candidates.unwrap_or(50);
        let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);

        // Acquire opposite-side synonym index read lock (if configured).
        let opp_syn_guard = opp_side.synonym_index.as_ref().map(|mtx| {
            mtx.read().unwrap_or_else(|e| e.into_inner())
        });

        // Build BM25 context from opposite side's index.
        // Commit any buffered writes first (write lock, brief), then
        // query under a read lock so concurrent requests can score in parallel.
        let results = if let Some(ref opp_bm25_mtx) = opp_side.bm25_index {
            {
                let mut w = opp_bm25_mtx.write().unwrap_or_else(|e| e.into_inner());
                w.commit_if_dirty();
            }
            let guard = opp_bm25_mtx.read().unwrap_or_else(|e| e.into_inner());
            let query_text = guard.query_text_for(&record, side);
            let ctx = Bm25Ctx::new(&guard, query_text);
            pipeline::score_pool(
                id,
                &record,
                side,
                &query_combined_vec,
                store.as_ref(),
                opp,
                opp_side.combined_index.as_deref(),
                &blocked_ids,
                config,
                ann_candidates,
                bm25_candidates_n,
                top_n,
                Some(ctx),
                opp_syn_guard.as_deref(),
            )
        } else {
            pipeline::score_pool(
                id,
                &record,
                side,
                &query_combined_vec,
                store.as_ref(),
                opp,
                opp_side.combined_index.as_deref(),
                &blocked_ids,
                config,
                ann_candidates,
                bm25_candidates_n,
                top_n,
                None,
                opp_syn_guard.as_deref(),
            )
        };

        let classification = results
            .first()
            .map(|r| r.classification.as_str().to_string())
            .unwrap_or("no_match".to_string());

        let status = if results
            .first()
            .map(|r| r.classification == Classification::Auto)
            .unwrap_or(false)
        {
            "match_found"
        } else {
            "no_match"
        };

        let matches = build_match_entries(&results);

        self.match_count.fetch_add(1, Ordering::Relaxed);

        Ok(MatchResponse {
            status: status.to_string(),
            id: id.to_string(),
            side,
            classification,
            from_crossmap: false,
            matches,
        })
    }

    /// Confirm a match: add to crossmap.
    pub fn confirm_match(&self, a_id: &str, b_id: &str) -> Result<ConfirmResponse, SessionError> {
        let store = &self.state.store;

        // Validate both IDs exist
        if !store.contains(Side::A, a_id) {
            return Err(SessionError::MissingField {
                field: format!("a_id '{}' not found", a_id),
            });
        }
        if !store.contains(Side::B, b_id) {
            return Err(SessionError::MissingField {
                field: format!("b_id '{}' not found", b_id),
            });
        }

        self.state.crossmap.add(a_id, b_id);
        store.mark_matched(Side::A, a_id);
        store.mark_matched(Side::B, b_id);

        // Drain any review entries involving either ID.
        self.state.drain_reviews_for_pair(a_id, b_id);

        let _ = self.state.wal.append(&WalEvent::CrossMapConfirm {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
            score: None,
        });
        self.state.mark_crossmap_dirty();

        Ok(ConfirmResponse {
            status: "confirmed".to_string(),
        })
    }

    /// Lookup a crossmap entry.
    pub fn lookup_crossmap(&self, id: &str, side: Side) -> Result<LookupResponse, SessionError> {
        let store = &self.state.store;
        let opp = side.opposite();

        let paired_id = match side {
            Side::A => self.state.crossmap.get_b(id),
            Side::B => self.state.crossmap.get_a(id),
        };

        if let Some(ref pid) = paired_id {
            let matched_record = store.get(opp, pid);

            Ok(LookupResponse {
                id: id.to_string(),
                side,
                status: "matched".to_string(),
                paired_id: Some(pid.clone()),
                matched_record,
            })
        } else {
            Ok(LookupResponse {
                id: id.to_string(),
                side,
                status: "unmatched".to_string(),
                paired_id: None,
                matched_record: None,
            })
        }
    }

    /// Break a crossmap pair.
    pub fn break_crossmap(&self, a_id: &str, b_id: &str) -> Result<BreakResponse, SessionError> {
        let store = &self.state.store;

        // Validate the pair exists
        let existing = self.state.crossmap.get_b(a_id);
        if existing.as_deref() != Some(b_id) {
            return Err(SessionError::MissingField {
                field: format!("crossmap pair ({}, {}) not found", a_id, b_id),
            });
        }

        self.state.crossmap.remove(a_id, b_id);

        store.mark_unmatched(Side::A, a_id);
        store.mark_unmatched(Side::B, b_id);

        // Drain any review entries involving either ID.
        self.state.drain_reviews_for_pair(a_id, b_id);

        let _ = self.state.wal.append(&WalEvent::CrossMapBreak {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        });
        self.state.mark_crossmap_dirty();

        Ok(BreakResponse {
            status: "broken".to_string(),
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        })
    }

    /// Query a record by ID, returning the record and its crossmap status.
    pub fn query_record(&self, side: Side, id: &str) -> Result<QueryResponse, SessionError> {
        let store = &self.state.store;
        let opp = side.opposite();

        // Look up the record
        let record = store
            .get(side, id)
            .ok_or_else(|| SessionError::MissingField {
                field: format!("record {} not found on side {:?}", id, side),
            })?;

        // Check crossmap
        let crossmap = match side {
            Side::A => {
                if let Some(b_id) = self.state.crossmap.get_b(id) {
                    let paired_record = store.get(opp, &b_id);
                    QueryCrossmap {
                        status: "matched".to_string(),
                        paired_id: Some(b_id),
                        paired_record,
                    }
                } else {
                    QueryCrossmap {
                        status: "unmatched".to_string(),
                        paired_id: None,
                        paired_record: None,
                    }
                }
            }
            Side::B => {
                if let Some(a_id) = self.state.crossmap.get_a(id) {
                    let paired_record = store.get(opp, &a_id);
                    QueryCrossmap {
                        status: "matched".to_string(),
                        paired_id: Some(a_id),
                        paired_record,
                    }
                } else {
                    QueryCrossmap {
                        status: "unmatched".to_string(),
                        paired_id: None,
                        paired_record: None,
                    }
                }
            }
        };

        Ok(QueryResponse {
            id: id.to_string(),
            side,
            record,
            crossmap,
        })
    }

    // -----------------------------------------------------------------------
    // Batch operations
    // -----------------------------------------------------------------------

    /// Maximum batch size accepted by batch endpoints.
    const MAX_BATCH_SIZE: usize = 1000;

    /// Add/update multiple records in a single call.
    pub fn upsert_batch(
        &self,
        side: Side,
        records: Vec<Record>,
    ) -> Result<BatchUpsertResponse, SessionError> {
        if records.is_empty() {
            return Err(SessionError::MissingField {
                field: "records (empty array)".into(),
            });
        }
        if records.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::MissingField {
                field: format!(
                    "batch size {} exceeds maximum of {}",
                    records.len(),
                    Self::MAX_BATCH_SIZE
                ),
            });
        }

        let is_a_side = side == Side::A;
        let config = &self.state.config;
        let emb_specs = &self.emb_specs;
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };
        let this_side = self.state.side(side);

        // 1. Extract record IDs up front
        let ids: Vec<String> = records
            .iter()
            .map(|r| r.get(id_field).cloned().unwrap_or_default())
            .collect();

        // 2. Batch encode combined vectors
        if !emb_specs.is_empty() {
            let field_dim = self.state.encoder_pool.dim();
            let combined_dim = field_dim * emb_specs.len();
            let mut combined_vecs: Vec<Vec<f32>> =
                vec![Vec::with_capacity(combined_dim); records.len()];

            for (field_a, field_b, weight) in emb_specs.iter() {
                let field_name = if is_a_side { field_a } else { field_b };
                let texts: Vec<String> = records
                    .iter()
                    .map(|r| {
                        r.get(field_name)
                            .map(|v| v.trim().to_string())
                            .unwrap_or_default()
                    })
                    .collect();
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let mut vecs = self.state.encoder_pool.encode(&text_refs)?;
                let sqrt_w = weight.sqrt() as f32;
                for (i, vec) in vecs.iter_mut().enumerate() {
                    for v in vec.iter_mut() {
                        *v *= sqrt_w;
                    }
                    combined_vecs[i].extend_from_slice(vec);
                }
            }

            // Store combined vecs in the combined index.
            if let Some(ref idx) = this_side.combined_index {
                for (i, rec_id) in ids.iter().enumerate() {
                    let _ = idx.upsert(rec_id, &combined_vecs[i], &records[i], side);
                }
            }
        }

        // 3. Insert + score each record sequentially
        let mut results = Vec::with_capacity(records.len());
        for record in records {
            let resp = self.upsert_record_inner(side, record);
            match resp {
                Ok(r) => results.push(r),
                Err(e) => {
                    results.push(UpsertResponse {
                        status: format!("error: {}", e),
                        id: String::new(),
                        side,
                        classification: "error".to_string(),
                        from_crossmap: false,
                        matches: vec![],
                        old_mapping: None,
                    });
                }
            }
        }

        Ok(BatchUpsertResponse { results })
    }

    /// Score multiple records against the opposite side without storing
    /// them (read-only batch).
    pub fn match_batch(
        &self,
        side: Side,
        records: Vec<Record>,
    ) -> Result<BatchMatchResponse, SessionError> {
        if records.is_empty() {
            return Err(SessionError::MissingField {
                field: "records (empty array)".into(),
            });
        }
        if records.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::MissingField {
                field: format!(
                    "batch size {} exceeds maximum of {}",
                    records.len(),
                    Self::MAX_BATCH_SIZE
                ),
            });
        }

        let is_a_side = side == Side::A;
        let config = &self.state.config;
        let emb_specs = &self.emb_specs;
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };
        let this_side = self.state.side(side);

        // 1. Extract record IDs
        let ids: Vec<String> = records
            .iter()
            .map(|r| r.get(id_field).cloned().unwrap_or_default())
            .collect();

        // 2. Batch encode combined vectors
        if !emb_specs.is_empty() {
            let field_dim = self.state.encoder_pool.dim();
            let combined_dim = field_dim * emb_specs.len();
            let mut combined_vecs: Vec<Vec<f32>> =
                vec![Vec::with_capacity(combined_dim); records.len()];

            for (field_a, field_b, weight) in emb_specs.iter() {
                let field_name = if is_a_side { field_a } else { field_b };
                let texts: Vec<String> = records
                    .iter()
                    .map(|r| {
                        r.get(field_name)
                            .map(|v| v.trim().to_string())
                            .unwrap_or_default()
                    })
                    .collect();
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let mut vecs = self.state.encoder_pool.encode(&text_refs)?;
                let sqrt_w = weight.sqrt() as f32;
                for (i, vec) in vecs.iter_mut().enumerate() {
                    for v in vec.iter_mut() {
                        *v *= sqrt_w;
                    }
                    combined_vecs[i].extend_from_slice(vec);
                }
            }

            if let Some(ref idx) = this_side.combined_index {
                for (i, rec_id) in ids.iter().enumerate() {
                    let _ = idx.upsert(rec_id, &combined_vecs[i], &records[i], side);
                }
            }
        }

        // 3. Score each record sequentially
        let mut results = Vec::with_capacity(records.len());
        for (record, rec_id) in records.into_iter().zip(ids.iter()) {
            let resp = self.try_match_inner(side, record, rec_id);
            match resp {
                Ok(r) => results.push(r),
                Err(e) => {
                    results.push(MatchResponse {
                        status: format!("error: {}", e),
                        id: String::new(),
                        side,
                        classification: "error".to_string(),
                        from_crossmap: false,
                        matches: vec![],
                    });
                }
            }
        }

        Ok(BatchMatchResponse { results })
    }

    /// Remove multiple records by ID.
    pub fn remove_batch(
        &self,
        side: Side,
        ids: Vec<String>,
    ) -> Result<BatchRemoveResponse, SessionError> {
        if ids.is_empty() {
            return Err(SessionError::MissingField {
                field: "ids (empty array)".into(),
            });
        }
        if ids.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::MissingField {
                field: format!(
                    "batch size {} exceeds maximum of {}",
                    ids.len(),
                    Self::MAX_BATCH_SIZE
                ),
            });
        }

        let mut results = Vec::with_capacity(ids.len());
        for id in &ids {
            match self.remove_record(side, id) {
                Ok(r) => results.push(r),
                Err(_) => {
                    results.push(RemoveResponse {
                        status: "not_found".to_string(),
                        id: id.clone(),
                        side,
                        crossmap_broken: vec![],
                    });
                }
            }
        }

        Ok(BatchRemoveResponse { results })
    }

    /// Health check response.
    pub fn health(&self) -> HealthResponse {
        let store = &self.state.store;
        let cm_len = self.state.crossmap.len();
        HealthResponse {
            status: "ready".to_string(),
            model: self.state.config.embeddings.model.clone(),
            records_a: store.len(Side::A),
            records_b: store.len(Side::B),
            crossmap_entries: cm_len,
        }
    }

    /// Status response.
    pub fn status(&self) -> StatusResponse {
        StatusResponse {
            job: self.state.config.job.name.clone(),
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            upserts: self.upsert_count.load(Ordering::Relaxed),
            matches: self.match_count.load(Ordering::Relaxed),
        }
    }

    // -----------------------------------------------------------------------
    // Crossmap, unmatched, stats, and review queries
    // -----------------------------------------------------------------------

    /// Return all confirmed crossmap pairs with pagination.
    pub fn crossmap_pairs(&self, offset: usize, limit: Option<usize>) -> CrossmapPairsResponse {
        let mut all = self.state.crossmap.pairs();
        all.sort();
        let total = all.len();
        let start = offset.min(total);
        let end = match limit {
            Some(l) => (start + l).min(total),
            None => total,
        };
        let pairs = all[start..end]
            .iter()
            .map(|(a, b)| CrossmapPairEntry {
                a_id: a.clone(),
                b_id: b.clone(),
            })
            .collect();
        CrossmapPairsResponse {
            total,
            offset: start,
            pairs,
        }
    }

    /// Return unmatched record IDs for a given side with pagination.
    pub fn unmatched_records(
        &self,
        side: Side,
        offset: usize,
        limit: Option<usize>,
        include_records: bool,
    ) -> UnmatchedResponse {
        let store = &self.state.store;
        let ids = store.unmatched_ids(side); // already sorted
        let total = ids.len();
        let start = offset.min(total);
        let end = match limit {
            Some(l) => (start + l).min(total),
            None => total,
        };
        let records = ids[start..end]
            .iter()
            .map(|id| {
                let record = if include_records {
                    store.get(side, id)
                } else {
                    None
                };
                UnmatchedEntry {
                    id: id.clone(),
                    record,
                }
            })
            .collect();
        UnmatchedResponse {
            side,
            total,
            offset: start,
            records,
        }
    }

    /// Return crossmap coverage statistics.
    pub fn crossmap_stats(&self) -> CrossmapStatsResponse {
        let store = &self.state.store;
        let records_a = store.len(Side::A);
        let records_b = store.len(Side::B);
        let crossmap_pairs = self.state.crossmap.len();
        let unmatched_a = store.unmatched_count(Side::A);
        let unmatched_b = store.unmatched_count(Side::B);
        let matched_a = records_a.saturating_sub(unmatched_a);
        let matched_b = records_b.saturating_sub(unmatched_b);
        let coverage_a = if records_a > 0 {
            matched_a as f64 / records_a as f64
        } else {
            0.0
        };
        let coverage_b = if records_b > 0 {
            matched_b as f64 / records_b as f64
        } else {
            0.0
        };
        CrossmapStatsResponse {
            records_a,
            records_b,
            crossmap_pairs,
            matched_a,
            matched_b,
            unmatched_a,
            unmatched_b,
            coverage_a,
            coverage_b,
        }
    }

    /// Return pending review-band matches with pagination.
    pub fn review_list(&self, offset: usize, limit: Option<usize>) -> ReviewListResponse {
        let mut reviews: Vec<ReviewListEntry> = self
            .state
            .review_queue
            .iter()
            .map(|entry| {
                let v = entry.value();
                ReviewListEntry {
                    id: v.id.clone(),
                    side: v.side,
                    candidate_id: v.candidate_id.clone(),
                    score: v.score,
                }
            })
            .collect();
        // Sort by score descending for deterministic output with highest
        // confidence reviews first.
        reviews.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total = reviews.len();
        let start = offset.min(total);
        let end = match limit {
            Some(l) => (start + l).min(total),
            None => total,
        };
        ReviewListResponse {
            total,
            offset: start,
            reviews: reviews[start..end].to_vec(),
        }
    }
}

/// Convert a `MelderError` to a `SessionError`.
fn melder_to_session_err(e: crate::error::MelderError) -> SessionError {
    match e {
        crate::error::MelderError::Encoder(enc) => SessionError::Encoder(enc),
        other => SessionError::MissingField {
            field: format!("encoding failed: {}", other),
        },
    }
}

/// Convert MatchResult list to API response entries.
fn build_match_entries(results: &[MatchResult]) -> Vec<MatchEntry> {
    results
        .iter()
        .map(|r| MatchEntry {
            id: r.matched_id.clone(),
            score: r.score,
            classification: r.classification.as_str().to_string(),
            field_scores: r
                .field_scores
                .iter()
                .map(|fs| FieldScoreEntry {
                    field_a: fs.field_a.clone(),
                    field_b: fs.field_b.clone(),
                    method: fs.method.clone(),
                    score: fs.score,
                    weight: fs.weight,
                })
                .collect(),
            matched_record: r.matched_record.clone(),
        })
        .collect()
}
