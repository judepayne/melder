//! Session: the core live matching logic.
//!
//! Provides upsert, try-match, and crossmap management operations.
//! All operations are synchronous internally; the HTTP layer wraps them
//! in `spawn_blocking` as needed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tracing::{info, info_span, warn};

use crate::bm25::scorer::normalise_bm25;
use crate::error::SessionError;
use crate::matching::pipeline;
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
    pub struct ExcludeResponse {
        pub excluded: bool,
        pub match_was_broken: bool,
        pub a_id: String,
        pub b_id: String,
    }

    #[derive(Debug, Serialize)]
    pub struct UnexcludeResponse {
        pub removed: bool,
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
        pub pairs: Vec<CrossmapPairEntry>,
        /// Opaque cursor for the next page. `None` when this is the last page.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
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
        pub records: Vec<UnmatchedEntry>,
        /// Opaque cursor for the next page. `None` when this is the last page.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
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
        pub reviews: Vec<ReviewListEntry>,
        /// Opaque cursor for the next page. `None` when this is the last page.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
    }

    // --- Enroll-mode response types ---

    #[derive(Debug, Serialize)]
    pub struct EnrollEdge {
        pub id: String,
        pub score: f64,
        pub field_scores: Vec<EnrollFieldScore>,
    }

    #[derive(Debug, Serialize)]
    pub struct EnrollFieldScore {
        pub field: String,
        pub method: String,
        pub score: f64,
        pub weight: f64,
    }

    #[derive(Debug, Serialize)]
    pub struct EnrollResponse {
        pub id: String,
        pub enrolled: bool,
        pub edges: Vec<EnrollEdge>,
    }

    #[derive(Debug, Serialize)]
    pub struct BatchEnrollResponse {
        pub results: Vec<EnrollResponse>,
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
    /// Hook event sender. `None` when hooks are not configured.
    hook_tx: Option<tokio::sync::mpsc::Sender<crate::hooks::HookEvent>>,
}

impl Session {
    pub fn new(
        state: Arc<LiveMatchState>,
        hook_tx: Option<tokio::sync::mpsc::Sender<crate::hooks::HookEvent>>,
    ) -> Self {
        let emb_specs = crate::vectordb::embedding_field_specs(&state.config);
        Self {
            state,
            start_time: Instant::now(),
            upsert_count: AtomicU64::new(0),
            match_count: AtomicU64::new(0),
            emb_specs,
            hook_tx,
        }
    }

    /// Send a hook event (best-effort, never blocks).
    fn send_hook(&self, event: crate::hooks::HookEvent) {
        if let Some(ref tx) = self.hook_tx
            && let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) = tx.try_send(event)
        {
            warn!("hook event dropped: channel full");
        }
    }

    /// Run an initial matching pass for all unmatched B records against A.
    ///
    /// Called at startup after datasets are loaded and indices are built,
    /// before the HTTP server starts listening. Uses the same scoring
    /// pipeline and claim logic as live upserts, including WAL writes,
    /// hooks, and review queue entries.
    ///
    /// Skips encoding — records are already in the store and vector index.
    pub fn initial_match_pass(&self) {
        let config = &self.state.config;
        let store = &self.state.store;

        // Only run if both sides have data
        let a_count = store.len(Side::A).unwrap_or(0);
        let b_count = store.len(Side::B).unwrap_or(0);
        if a_count == 0 || b_count == 0 {
            return;
        }

        // Collect unmatched B IDs
        let unmatched_b: Vec<String> = store.unmatched_ids(Side::B).unwrap_or_default();
        if unmatched_b.is_empty() {
            info!(
                a_count,
                b_count, "initial match: all B records already matched"
            );
            return;
        }

        info!(
            unmatched = unmatched_b.len(),
            a_count, b_count, "initial match pass starting"
        );

        let start = Instant::now();
        let opp_side = self.state.opposite_side(Side::B); // = A side
        let opp_bm25 = opp_side.bm25_index.as_ref();
        let top_n = config.top_n.unwrap_or(5);
        let ann_candidates = config.ann_candidates.unwrap_or(50);
        let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);
        let syn_fields = &config.synonym_fields;
        let b_side = self.state.side(Side::B);

        let mut auto_matched = 0usize;
        let mut review_count = 0usize;
        let mut no_match = 0usize;

        for (i, b_id) in unmatched_b.iter().enumerate() {
            // Progress logging every 1000 records
            if (i + 1) % 1000 == 0 || i + 1 == unmatched_b.len() {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                info!(
                    done = i + 1,
                    total = unmatched_b.len(),
                    rate = format!("{:.0}", rate),
                    "initial match progress"
                );
            }

            // Skip if already matched (by a prior iteration's claim)
            if self.state.crossmap.has_b(b_id) {
                continue;
            }

            // Get the B record from the store
            let b_record = match store.get(Side::B, b_id) {
                Ok(Some(rec)) => rec,
                _ => continue,
            };

            // Blocking query
            let blocked_ids: Vec<String> = if config.blocking.enabled {
                store
                    .blocking_query(&b_record, Side::B, Side::A)
                    .unwrap_or_default()
            } else {
                store.ids(Side::A).unwrap_or_default()
            };

            // Get combined vec from index (already encoded during startup)
            let query_combined_vec: Vec<f32> = b_side
                .combined_index
                .as_ref()
                .and_then(|idx| idx.get(b_id).ok().flatten())
                .unwrap_or_default();

            // BM25 candidate generation
            let (bm25_cand_ids, bm25_scores_map) = if let Some(bm25) = opp_bm25 {
                let query_text = bm25.query_text_for(&b_record, Side::B);
                let self_score = bm25.analytical_self_score(&query_text);
                let raw_results = bm25.score_blocked(&query_text, &blocked_ids, bm25_candidates_n);
                let scored: Vec<(String, f64)> = raw_results
                    .into_iter()
                    .map(|(cid, raw)| {
                        let norm = crate::bm25::scorer::normalise_bm25(raw, self_score);
                        (cid, norm)
                    })
                    .collect();
                let ids: Vec<String> = scored.iter().map(|(cid, _)| cid.clone()).collect();
                let map: std::collections::HashMap<String, f64> = scored.into_iter().collect();
                (ids, map)
            } else {
                (Vec::new(), std::collections::HashMap::new())
            };

            // Synonym candidate generation
            let opp_syn_guard = opp_side
                .synonym_index
                .as_ref()
                .map(|mtx| mtx.read().unwrap_or_else(|e| e.into_inner()));
            let synonym_cand_ids = if !syn_fields.is_empty() {
                if let Some(ref guard) = opp_syn_guard {
                    pipeline::synonym_candidate_stage(guard, &b_record, Side::B, syn_fields)
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            // Score
            let scoring_query = pipeline::ScoringQuery {
                id: b_id,
                record: &b_record,
                side: Side::B,
                combined_vec: &query_combined_vec,
            };
            let scoring_pool = pipeline::ScoringPool {
                store: store.as_ref(),
                side: Side::A,
                combined_index: opp_side.combined_index.as_deref(),
                blocked_ids: &blocked_ids,
                bm25_candidate_ids: &bm25_cand_ids,
                bm25_scores_map: &bm25_scores_map,
                synonym_candidate_ids: &synonym_cand_ids,
                synonym_dictionary: self.state.synonym_dictionary.as_deref(),
                exclusions: &self.state.exclusions,
            };
            let results =
                pipeline::score_pool(&scoring_query, &scoring_pool, config, ann_candidates, top_n);

            // Claim loop — same logic as upsert_record_inner
            for result in &results {
                if result.score < config.thresholds.review_floor {
                    break;
                }
                if result.score >= config.thresholds.auto_match {
                    let a_id = &result.matched_id;
                    if self.state.crossmap.claim(a_id, b_id) {
                        let _ = store.mark_matched(Side::A, a_id);
                        let _ = store.mark_matched(Side::B, b_id);

                        self.send_hook(crate::hooks::HookEvent::Confirm {
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                            score: result.score,
                            source: "auto".into(),
                            field_scores: result.field_scores.clone(),
                        });

                        if let Err(e) = self.state.wal.append(&WalEvent::CrossMapConfirm {
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                            score: Some(result.score),
                        }) {
                            warn!(error = %e, "WAL append failed for initial match confirm");
                        }
                        self.state.mark_crossmap_dirty();
                        auto_matched += 1;
                        break;
                    }
                    continue;
                }
                // Review band
                if let Err(e) = self.state.wal.append(&WalEvent::ReviewMatch {
                    id: b_id.clone(),
                    side: Side::B,
                    candidate_id: result.matched_id.clone(),
                    score: result.score,
                }) {
                    warn!(error = %e, "WAL append failed for initial match review");
                }
                let key = review_queue_key(Side::B, b_id, &result.matched_id);
                self.state.insert_review(
                    key,
                    ReviewEntry {
                        id: b_id.clone(),
                        side: Side::B,
                        candidate_id: result.matched_id.clone(),
                        score: result.score,
                    },
                );
                review_count += 1;
                break;
            }
            if results.is_empty()
                || results
                    .first()
                    .is_some_and(|r| r.score < config.thresholds.review_floor)
            {
                no_match += 1;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        info!(
            auto_matched,
            review_count,
            no_match,
            elapsed_s = format!("{:.1}", elapsed),
            "initial match pass complete"
        );
    }

    /// Upsert a record into the live state and attempt matching.
    ///
    /// Encodes the combined embedding vector in parallel with BM25/synonym
    /// upserts and blocking query via `rayon::scope`. After the join, inserts
    /// the vector, runs ANN search, and scores candidates.
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
        let _span = info_span!("upsert_record").entered();
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

        // Check if encoding is needed (text-hash skip).
        let emb_specs = &self.emb_specs;
        let needs_encoding = if emb_specs.is_empty() {
            false
        } else {
            match self.state.side(side).combined_index {
                Some(ref idx) => {
                    let current_hash =
                        crate::vectordb::texthash::compute_text_hash(&record, emb_specs, side);
                    idx.text_hash_for(&id) != Some(current_hash)
                }
                None => true,
            }
        };

        self.upsert_record_inner(side, record, needs_encoding, is_a_side)
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
        let _span = info_span!("encode_combined").entered();
        if let Some(ref coordinator) = self.state.coordinator {
            // Coordinator path: submit texts for batched encoding.
            // Fully synchronous — safe from rayon/spawn_blocking threads.
            let field_dim = self.state.encoder_pool.dim();
            let mut combined = Vec::with_capacity(field_dim * emb_specs.len());

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

            let vecs = coordinator
                .encode_many(texts)
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

    /// Upsert a record, running encoding in parallel with BM25/synonym/blocking
    /// work via `rayon::scope`. Handles insertion, crossmap management, and scoring.
    fn upsert_record_inner(
        &self,
        side: Side,
        record: Record,
        needs_encoding: bool,
        is_a_side: bool,
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

        if store.contains(side, &id)? {
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
                store.mark_unmatched(Side::A, &a_id)?;
                store.mark_unmatched(Side::B, &b_id)?;

                // WAL
                if let Err(e) = self.state.wal.append(&WalEvent::CrossMapBreak {
                    a_id: a_id.clone(),
                    b_id: b_id.clone(),
                }) {
                    warn!(error = %e, "WAL append failed for crossmap break");
                }

                old_mapping = Some(OldMapping { a_id, b_id });
                self.state.mark_crossmap_dirty();
            }

            // Remove old record from blocking index
            if let Some(old_rec) = store.get(side, &id)? {
                store.blocking_remove(side, &id, &old_rec)?;
            }
        }

        // 3. WAL append (zero-clone borrowing serialization).
        if let Err(e) = self.state.wal.append_upsert(side, &record) {
            warn!(error = %e, "WAL append failed for upsert");
        }

        // 4. Insert/replace record
        store.insert(side, &id, &record)?;

        // 5. Add to unmatched
        store.mark_unmatched(side, &id)?;

        // 6. Update blocking index
        store.blocking_insert(side, &id, &record)?;

        // 6b-6c + 8 + 9: Parallel pipeline.
        //
        // Encoding (4-6ms) runs in parallel with BM25/synonym upserts,
        // common_id check, and blocking query (<0.5ms combined).
        // After the join: vector index upsert, BM25 scoring, score_pool.

        let top_n = config.top_n.unwrap_or(5);
        let ann_candidates = config.ann_candidates.unwrap_or(50);
        let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);
        let opp_side = self.state.opposite_side(side);
        let this_side = self.state.side(side);
        let emb_specs = &self.emb_specs;

        // Extract references for rayon closures (avoid borrowing &self).
        let this_bm25 = this_side.bm25_index.as_ref();
        let opp_bm25 = opp_side.bm25_index.as_ref();
        let this_syn = this_side.synonym_index.as_ref();
        let syn_fields = &config.synonym_fields;
        let blocking_enabled = config.blocking.enabled;
        let this_cid_field = match side {
            Side::A => config.datasets.a.common_id_field.as_deref(),
            Side::B => config.datasets.b.common_id_field.as_deref(),
        };

        // --- Encoding + index updates + blocking query ---
        //
        // When encoding is needed (4-6ms), run it in parallel with the
        // fast work (BM25/synonym upserts + blocking query, <0.5ms) via
        // rayon::scope. When encoding is skipped (text-hash hit), run
        // everything sequentially to avoid rayon overhead.
        let mut combined_vec: Vec<f32> = Vec::new();
        let mut blocked_ids: Vec<String> = Vec::new();

        if needs_encoding && self.state.coordinator.is_none() {
            // Parallel path: encoding dominates, hide fast work underneath.
            // Only used when encoding goes through the pool directly (no
            // coordinator). When the coordinator is active, rayon workers
            // would block on the coordinator channel while fastembed (which
            // also uses rayon internally) tries to use the same thread pool
            // — causing a deadlock. See: fastembed → rayon dependency.
            let record_ref = &record;
            let id_ref = id.as_str();
            let mut encode_error: Option<SessionError> = None;

            rayon::scope(|s| {
                let combined_out = &mut combined_vec;
                let encode_err = &mut encode_error;
                s.spawn(move |_| {
                    let _span = info_span!("encode_combined").entered();
                    match self.encode_combined(record_ref, emb_specs, is_a_side) {
                        Ok(vec) => *combined_out = vec,
                        Err(e) => *encode_err = Some(e),
                    }
                });

                let blocked_out = &mut blocked_ids;
                s.spawn(move |_| {
                    if let Some(bm25) = this_bm25 {
                        let _span = info_span!("bm25_upsert").entered();
                        bm25.upsert(id_ref, record_ref);
                    }
                    if let Some(syn_mtx) = this_syn {
                        let mut idx = syn_mtx.write().unwrap_or_else(|e| e.into_inner());
                        idx.upsert(id_ref, record_ref, side, syn_fields);
                    }
                    let _span = info_span!("blocking_query").entered();
                    *blocked_out = if blocking_enabled {
                        store
                            .blocking_query(record_ref, side, opp)
                            .unwrap_or_default()
                    } else {
                        store.ids(opp).unwrap_or_default()
                    };
                });
            });

            if let Some(e) = encode_error {
                return Err(e);
            }
        } else if needs_encoding {
            // Coordinator path: run encoding sequentially to avoid
            // rayon deadlock. BM25/synonym/blocking work runs after.
            combined_vec = self.encode_combined(&record, emb_specs, is_a_side)?;
            if let Some(bm25) = this_bm25 {
                let _span = info_span!("bm25_upsert").entered();
                bm25.upsert(&id, &record);
            }
            if let Some(syn_mtx) = this_syn {
                let mut idx = syn_mtx.write().unwrap_or_else(|e| e.into_inner());
                idx.upsert(&id, &record, side, syn_fields);
            }
            let _span = info_span!("blocking_query").entered();
            blocked_ids = if blocking_enabled {
                store.blocking_query(&record, side, opp)?
            } else {
                store.ids(opp)?
            };
        } else {
            // Sequential path: no encoding, avoid rayon overhead.
            if let Some(bm25) = this_bm25 {
                let _span = info_span!("bm25_upsert").entered();
                bm25.upsert(&id, &record);
            }
            if let Some(syn_mtx) = this_syn {
                let mut idx = syn_mtx.write().unwrap_or_else(|e| e.into_inner());
                idx.upsert(&id, &record, side, syn_fields);
            }
            let _span = info_span!("blocking_query").entered();
            blocked_ids = if blocking_enabled {
                store.blocking_query(&record, side, opp)?
            } else {
                store.ids(opp)?
            };
        }

        // --- Common ID check (must be sequential — has side effects) ---
        if let Some(cid_field) = this_cid_field
            && let Some(cid_val) = record.get(cid_field)
        {
            let cid_val: &str = cid_val.trim();
            if !cid_val.is_empty() {
                store.common_id_insert(side, cid_val, &id)?;

                if let Some(opp_id) = store.common_id_lookup(opp, cid_val)? {
                    if let Some(old_opp) = match side {
                        Side::A => self.state.crossmap.take_a(&id),
                        Side::B => self.state.crossmap.take_b(&id),
                    } {
                        store.mark_unmatched(opp, &old_opp)?;
                    }
                    if let Some(old_this) = match side {
                        Side::A => self.state.crossmap.take_b(&opp_id),
                        Side::B => self.state.crossmap.take_a(&opp_id),
                    } {
                        store.mark_unmatched(side, &old_this)?;
                    }

                    let (a_id, b_id) = match side {
                        Side::A => (id.clone(), opp_id.clone()),
                        Side::B => (opp_id.clone(), id.clone()),
                    };
                    self.state.crossmap.add(&a_id, &b_id);
                    store.mark_matched(Side::A, &a_id)?;
                    store.mark_matched(Side::B, &b_id)?;

                    if let Err(e) = self.state.wal.append(&WalEvent::CrossMapConfirm {
                        a_id: a_id.clone(),
                        b_id: b_id.clone(),
                        score: Some(1.0),
                    }) {
                        warn!(error = %e, "WAL append failed for crossmap confirm");
                    }
                    self.state.mark_crossmap_dirty();

                    self.send_hook(crate::hooks::HookEvent::Confirm {
                        a_id: a_id.clone(),
                        b_id: b_id.clone(),
                        score: 1.0,
                        source: "auto".into(),
                        field_scores: vec![],
                    });

                    let opp_record = store.get(opp, &opp_id)?;
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

        // --- Sequential tail (after parallel join) ---

        // Vector index upsert (needs combined_vec from branch 1).
        if !combined_vec.is_empty()
            && let Some(ref idx) = this_side.combined_index
        {
            let _ = idx.upsert(&id, &combined_vec, &record, side);
        }

        // Fetch the query combined vec: use freshly encoded vec if available,
        // otherwise read from the index (text-hash skip case).
        let query_combined_vec = if !combined_vec.is_empty() {
            combined_vec
        } else {
            this_side
                .combined_index
                .as_ref()
                .and_then(|idx| idx.get(&id).ok().flatten())
                .unwrap_or_default()
        };

        // BM25 candidate generation (needs blocked_ids from branch 2).
        let (bm25_cand_ids, bm25_scores_map) = if let Some(opp_bm25) = opp_bm25 {
            let _span = info_span!("bm25_score").entered();
            let query_text = opp_bm25.query_text_for(&record, side);
            let self_score = opp_bm25.analytical_self_score(&query_text);
            let raw_results = opp_bm25.score_blocked(&query_text, &blocked_ids, bm25_candidates_n);
            let scored: Vec<(String, f64)> = raw_results
                .into_iter()
                .map(|(cid, raw)| {
                    let norm = normalise_bm25(raw, self_score);
                    (cid, norm)
                })
                .collect();
            let ids: Vec<String> = scored.iter().map(|(cid, _)| cid.clone()).collect();
            let map: std::collections::HashMap<String, f64> = scored.into_iter().collect();
            (ids, map)
        } else {
            (Vec::new(), std::collections::HashMap::new())
        };

        // Synonym candidate generation.
        let opp_syn_guard = opp_side
            .synonym_index
            .as_ref()
            .map(|mtx| mtx.read().unwrap_or_else(|e| e.into_inner()));
        let synonym_cand_ids = if !syn_fields.is_empty() {
            if let Some(ref guard) = opp_syn_guard {
                pipeline::synonym_candidate_stage(guard, &record, side, syn_fields)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let _span = info_span!("score_pool").entered();
        let scoring_query = pipeline::ScoringQuery {
            id: &id,
            record: &record,
            side,
            combined_vec: &query_combined_vec,
        };
        let scoring_pool = pipeline::ScoringPool {
            store: store.as_ref(),
            side: opp,
            combined_index: opp_side.combined_index.as_deref(),
            blocked_ids: &blocked_ids,
            bm25_candidate_ids: &bm25_cand_ids,
            bm25_scores_map: &bm25_scores_map,
            synonym_candidate_ids: &synonym_cand_ids,
            synonym_dictionary: self.state.synonym_dictionary.as_deref(),
            exclusions: &self.state.exclusions,
        };
        let results =
            pipeline::score_pool(&scoring_query, &scoring_pool, config, ann_candidates, top_n);

        // 10. Claim loop: try candidates in ranked order.
        let _span = info_span!("claim_loop").entered();
        let mut classification = "no_match".to_string();

        for result in &results {
            if result.score < config.thresholds.review_floor {
                break;
            }
            if result.score >= config.thresholds.auto_match {
                let (a_id, b_id) = match side {
                    Side::A => (id.clone(), result.matched_id.clone()),
                    Side::B => (result.matched_id.clone(), id.clone()),
                };
                if self.state.crossmap.claim(&a_id, &b_id) {
                    store.mark_matched(Side::A, &a_id)?;
                    store.mark_matched(Side::B, &b_id)?;

                    // Hook: on_confirm (auto-match via claim)
                    self.send_hook(crate::hooks::HookEvent::Confirm {
                        a_id: a_id.clone(),
                        b_id: b_id.clone(),
                        score: result.score,
                        source: "auto".into(),
                        field_scores: result.field_scores.clone(),
                    });

                    if let Err(e) = self.state.wal.append(&WalEvent::CrossMapConfirm {
                        a_id,
                        b_id,
                        score: Some(result.score),
                    }) {
                        warn!(error = %e, "WAL append failed for crossmap confirm");
                    }
                    self.state.mark_crossmap_dirty();
                    classification = "auto".to_string();
                    break;
                }
                continue;
            }
            // Score is in review band
            if let Err(e) = self.state.wal.append(&WalEvent::ReviewMatch {
                id: id.clone(),
                side,
                candidate_id: result.matched_id.clone(),
                score: result.score,
            }) {
                warn!(error = %e, "WAL append failed for review match");
            }
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
            // Hook: on_review
            {
                let (hook_a, hook_b) = match side {
                    Side::A => (id.clone(), result.matched_id.clone()),
                    Side::B => (result.matched_id.clone(), id.clone()),
                };
                self.send_hook(crate::hooks::HookEvent::Review {
                    a_id: hook_a,
                    b_id: hook_b,
                    score: result.score,
                    field_scores: result.field_scores.clone(),
                });
            }

            classification = "review".to_string();
            break;
        }

        // Hook: on_nomatch
        if classification == "no_match" {
            self.send_hook(crate::hooks::HookEvent::NoMatch {
                side,
                id: id.clone(),
                best_score: results.first().map(|r| r.score),
                best_candidate_id: results.first().map(|r| r.matched_id.clone()),
            });
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
        let record = store.get(side, id)?.ok_or_else(|| SessionError::NotFound {
            message: format!("record {} not found on side {:?}", id, side),
        })?;

        // Break any crossmap pair (atomic read+remove to avoid TOCTOU)
        let paired_id = match side {
            Side::A => self.state.crossmap.take_a(id),
            Side::B => self.state.crossmap.take_b(id),
        };
        let broken: Vec<String> = if let Some(ref pid) = paired_id {
            store.mark_unmatched(opp, pid)?;
            self.state.mark_crossmap_dirty();
            vec![pid.clone()]
        } else {
            vec![]
        };

        // Drain any review entries involving this record.
        self.state.drain_reviews_for_id(id);

        // Remove from blocking index
        store.blocking_remove(side, id, &record)?;

        // Remove from combined index
        if let Some(ref idx) = self.state.side(side).combined_index {
            let _ = idx.remove(id);
        }

        // Remove from BM25 index (SimpleBm25: lock-free)
        if let Some(ref bm25) = self.state.side(side).bm25_index {
            bm25.remove(id);
        }

        // Remove from synonym index
        if let Some(ref syn_mtx) = self.state.side(side).synonym_index {
            let mut idx = syn_mtx.write().unwrap_or_else(|e| e.into_inner());
            idx.remove(id, side);
        }

        // Remove from unmatched set
        store.mark_matched(side, id)?;

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
                store.common_id_remove(side, cid_val)?;
            }
        }

        // Remove from records
        store.remove(side, id)?;

        // Append to WAL
        if let Err(e) = self.state.wal.append(&WalEvent::RemoveRecord {
            side,
            id: id.to_string(),
        }) {
            warn!(error = %e, "WAL append failed for remove record");
        }

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

        // Check if encoding is needed (text-hash skip).
        let emb_specs = &self.emb_specs;
        let needs_encoding = if emb_specs.is_empty() {
            false
        } else {
            match self.state.side(side).combined_index {
                Some(ref idx) => {
                    let current_hash =
                        crate::vectordb::texthash::compute_text_hash(&record, emb_specs, side);
                    idx.text_hash_for(&id) != Some(current_hash)
                }
                None => true,
            }
        };

        self.try_match_inner(side, record, &id, needs_encoding, is_a_side)
    }

    /// Try matching a record, running encoding in parallel with the blocking
    /// query via `rayon::scope` (read-only — no store/BM25/synonym upserts).
    fn try_match_inner(
        &self,
        side: Side,
        record: Record,
        id: &str,
        needs_encoding: bool,
        is_a_side: bool,
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
            let matched_record = store.get(opp, &paired_id)?;

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

        // --- Parallel branches: encode vs blocking query ---
        let top_n = config.top_n.unwrap_or(5);
        let ann_candidates = config.ann_candidates.unwrap_or(50);
        let bm25_candidates_n = config.bm25_candidates.unwrap_or(10);
        let this_side = self.state.side(side);
        let opp_bm25 = opp_side.bm25_index.as_ref();
        let emb_specs = &self.emb_specs;
        let syn_fields = &config.synonym_fields;
        let blocking_enabled = config.blocking.enabled;

        let mut combined_vec: Vec<f32> = Vec::new();
        let mut blocked_ids: Vec<String> = Vec::new();

        if needs_encoding && self.state.coordinator.is_none() {
            // Parallel path: encoding dominates, hide blocking query
            // underneath. Only used when encoding goes through the pool
            // directly (no coordinator). When the coordinator is active,
            // rayon workers would block on the coordinator channel while
            // fastembed (which also uses rayon internally) tries to use
            // the same thread pool — causing a deadlock.
            let record_ref = &record;
            let mut encode_error: Option<SessionError> = None;

            rayon::scope(|s| {
                let combined_out = &mut combined_vec;
                let encode_err = &mut encode_error;
                s.spawn(move |_| {
                    let _span = info_span!("encode_combined").entered();
                    match self.encode_combined(record_ref, emb_specs, is_a_side) {
                        Ok(vec) => *combined_out = vec,
                        Err(e) => *encode_err = Some(e),
                    }
                });

                let blocked_out = &mut blocked_ids;
                s.spawn(move |_| {
                    let _span = info_span!("blocking_query").entered();
                    *blocked_out = if blocking_enabled {
                        store
                            .blocking_query(record_ref, side, opp)
                            .unwrap_or_default()
                    } else {
                        store.ids(opp).unwrap_or_default()
                    };
                });
            });

            if let Some(e) = encode_error {
                return Err(e);
            }
        } else if needs_encoding {
            // Coordinator path: run encoding sequentially to avoid
            // rayon deadlock. Blocking query runs after.
            combined_vec = self.encode_combined(&record, emb_specs, is_a_side)?;
            let _span = info_span!("blocking_query").entered();
            blocked_ids = if blocking_enabled {
                store.blocking_query(&record, side, opp)?
            } else {
                store.ids(opp)?
            };
        } else {
            let _span = info_span!("blocking_query").entered();
            blocked_ids = if blocking_enabled {
                store.blocking_query(&record, side, opp)?
            } else {
                store.ids(opp)?
            };
        }

        // --- Sequential tail ---
        // Note: try_match is read-only — do NOT upsert the encoded vector
        // into the index. Use the freshly-encoded vector for the query, or
        // fall back to an existing cached vector if encoding was skipped.
        let query_combined_vec = if !combined_vec.is_empty() {
            combined_vec
        } else {
            this_side
                .combined_index
                .as_ref()
                .and_then(|idx| idx.get(id).ok().flatten())
                .unwrap_or_default()
        };

        // BM25 candidate generation (needs blocked_ids).
        let (bm25_cand_ids, bm25_scores_map) = if let Some(opp_bm25) = opp_bm25 {
            let query_text = opp_bm25.query_text_for(&record, side);
            let self_score = opp_bm25.analytical_self_score(&query_text);
            let raw_results = opp_bm25.score_blocked(&query_text, &blocked_ids, bm25_candidates_n);
            let scored: Vec<(String, f64)> = raw_results
                .into_iter()
                .map(|(cid, raw)| {
                    let norm = normalise_bm25(raw, self_score);
                    (cid, norm)
                })
                .collect();
            let ids: Vec<String> = scored.iter().map(|(cid, _)| cid.clone()).collect();
            let map: std::collections::HashMap<String, f64> = scored.into_iter().collect();
            (ids, map)
        } else {
            (Vec::new(), std::collections::HashMap::new())
        };

        // Synonym candidate generation.
        let opp_syn_guard = opp_side
            .synonym_index
            .as_ref()
            .map(|mtx| mtx.read().unwrap_or_else(|e| e.into_inner()));
        let synonym_cand_ids = if !syn_fields.is_empty() {
            if let Some(ref guard) = opp_syn_guard {
                pipeline::synonym_candidate_stage(guard, &record, side, syn_fields)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let scoring_query = pipeline::ScoringQuery {
            id,
            record: &record,
            side,
            combined_vec: &query_combined_vec,
        };
        let scoring_pool = pipeline::ScoringPool {
            store: store.as_ref(),
            side: opp,
            combined_index: opp_side.combined_index.as_deref(),
            blocked_ids: &blocked_ids,
            bm25_candidate_ids: &bm25_cand_ids,
            bm25_scores_map: &bm25_scores_map,
            synonym_candidate_ids: &synonym_cand_ids,
            synonym_dictionary: self.state.synonym_dictionary.as_deref(),
            exclusions: &self.state.exclusions,
        };
        let results =
            pipeline::score_pool(&scoring_query, &scoring_pool, config, ann_candidates, top_n);

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

    /// Confirm a match: insert into crossmap, breaking any existing pairs
    /// for either side first.
    ///
    /// Manual confirmation is an explicit operator action, so it always
    /// succeeds (unlike `claim()` which fails if either side is taken).
    /// If `a_id` or `b_id` already has a crossmap partner, that old pair
    /// is broken and the old partner is marked unmatched before inserting
    /// the new pair. This preserves the bijection invariant (Constitution §3).
    pub fn confirm_match(&self, a_id: &str, b_id: &str) -> Result<ConfirmResponse, SessionError> {
        let store = &self.state.store;

        // Validate both IDs exist
        if !store.contains(Side::A, a_id)? {
            return Err(SessionError::NotFound {
                message: format!("a_id '{}' not found", a_id),
            });
        }
        if !store.contains(Side::B, b_id)? {
            return Err(SessionError::NotFound {
                message: format!("b_id '{}' not found", b_id),
            });
        }

        // Break any existing pair for a_id (e.g. a_id was matched to B-other).
        if let Some(old_b) = self.state.crossmap.take_a(a_id) {
            store.mark_unmatched(Side::B, &old_b)?;
        }
        // Break any existing pair for b_id (e.g. b_id was matched to A-other).
        if let Some(old_a) = self.state.crossmap.take_b(b_id) {
            store.mark_unmatched(Side::A, &old_a)?;
        }

        // Now insert the new pair — both sides are guaranteed free.
        self.state.crossmap.add(a_id, b_id);
        store.mark_matched(Side::A, a_id)?;
        store.mark_matched(Side::B, b_id)?;

        // Drain any review entries involving either ID.
        self.state.drain_reviews_for_pair(a_id, b_id);

        if let Err(e) = self.state.wal.append(&WalEvent::CrossMapConfirm {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
            score: None,
        }) {
            warn!(error = %e, "WAL append failed for crossmap confirm");
        }
        self.state.mark_crossmap_dirty();

        // Hook: on_confirm (manual)
        self.send_hook(crate::hooks::HookEvent::Confirm {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
            score: 0.0,
            source: "manual".into(),
            field_scores: vec![],
        });

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
            let matched_record = store.get(opp, pid)?;

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
            return Err(SessionError::NotFound {
                message: format!("crossmap pair ({}, {}) not found", a_id, b_id),
            });
        }

        self.state.crossmap.remove(a_id, b_id);

        store.mark_unmatched(Side::A, a_id)?;
        store.mark_unmatched(Side::B, b_id)?;

        // Drain any review entries involving either ID.
        self.state.drain_reviews_for_pair(a_id, b_id);

        if let Err(e) = self.state.wal.append(&WalEvent::CrossMapBreak {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        }) {
            warn!(error = %e, "WAL append failed for crossmap break");
        }
        self.state.mark_crossmap_dirty();

        // Hook: on_break
        self.send_hook(crate::hooks::HookEvent::Break {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        });

        Ok(BreakResponse {
            status: "broken".to_string(),
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        })
    }

    /// Exclude a pair of records (known non-match).
    ///
    /// If the pair is currently matched to each other in the CrossMap, the match
    /// is broken first. The pair is then added to the exclusions set and will
    /// never be scored or matched again (until unexcluded).
    pub fn exclude(&self, a_id: &str, b_id: &str) -> ExcludeResponse {
        let store = &self.state.store;
        let mut match_was_broken = false;

        // Check if the pair is currently matched to each other
        if self.state.crossmap.get_b(a_id).as_deref() == Some(b_id) {
            // Break the match
            self.state.crossmap.remove(a_id, b_id);
            let _ = store.mark_unmatched(Side::A, a_id);
            let _ = store.mark_unmatched(Side::B, b_id);
            self.state.drain_reviews_for_pair(a_id, b_id);

            if let Err(e) = self.state.wal.append(&WalEvent::CrossMapBreak {
                a_id: a_id.to_string(),
                b_id: b_id.to_string(),
            }) {
                warn!(error = %e, "WAL append failed for crossmap break (exclude)");
            }
            self.state.mark_crossmap_dirty();

            // Hook: on_break (the match was broken as part of the exclude)
            self.send_hook(crate::hooks::HookEvent::Break {
                a_id: a_id.to_string(),
                b_id: b_id.to_string(),
            });

            match_was_broken = true;
        }

        // Add to exclusions
        self.state.exclusions.add(a_id, b_id);

        // WAL: exclude event
        if let Err(e) = self.state.wal.append(&WalEvent::Exclude {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        }) {
            warn!(error = %e, "WAL append failed for exclude");
        }

        // Hook: on_exclude
        self.send_hook(crate::hooks::HookEvent::Exclude {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
            match_was_broken,
        });

        info!(a_id, b_id, match_was_broken, "exclude");

        ExcludeResponse {
            excluded: true,
            match_was_broken,
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        }
    }

    /// Remove an exclusion for a pair of records.
    ///
    /// After unexcluding, the pair can be matched again on the next upsert
    /// or try-match.
    pub fn unexclude(&self, a_id: &str, b_id: &str) -> UnexcludeResponse {
        let was_excluded = self.state.exclusions.contains(a_id, b_id);

        if was_excluded {
            self.state.exclusions.remove(a_id, b_id);

            // WAL: unexclude event
            if let Err(e) = self.state.wal.append(&WalEvent::Unexclude {
                a_id: a_id.to_string(),
                b_id: b_id.to_string(),
            }) {
                warn!(error = %e, "WAL append failed for unexclude");
            }
        }

        info!(a_id, b_id, was_excluded, "unexclude");

        UnexcludeResponse {
            removed: was_excluded,
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        }
    }

    /// Query a record by ID, returning the record and its crossmap status.
    pub fn query_record(&self, side: Side, id: &str) -> Result<QueryResponse, SessionError> {
        let store = &self.state.store;
        let opp = side.opposite();

        // Look up the record
        let record = store.get(side, id)?.ok_or_else(|| SessionError::NotFound {
            message: format!("record {} not found on side {:?}", id, side),
        })?;

        // Check crossmap
        let crossmap = match side {
            Side::A => {
                if let Some(b_id) = self.state.crossmap.get_b(id) {
                    let paired_record = store.get(opp, &b_id)?;
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
                    let paired_record = store.get(opp, &a_id)?;
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
            return Err(SessionError::BatchValidation {
                message: "records array is empty".into(),
            });
        }
        if records.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::BatchValidation {
                message: format!(
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
            let resp = self.upsert_record_inner(side, record, false, side == Side::A);
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

    /// Score multiple records against the opposite side.
    ///
    /// Encoded vectors are cached in the combined index for future queries,
    /// but no crossmap pairs or unmatched state is modified.
    pub fn match_batch(
        &self,
        side: Side,
        records: Vec<Record>,
    ) -> Result<BatchMatchResponse, SessionError> {
        if records.is_empty() {
            return Err(SessionError::BatchValidation {
                message: "records array is empty".into(),
            });
        }
        if records.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::BatchValidation {
                message: format!(
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
            let resp = self.try_match_inner(side, record, rec_id, false, side == Side::A);
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
            return Err(SessionError::BatchValidation {
                message: "ids array is empty".into(),
            });
        }
        if ids.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::BatchValidation {
                message: format!(
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
                Err(e) => {
                    // Surface the actual error rather than masking store/IO
                    // errors as "not_found". NotFound is a legitimate status;
                    // other errors indicate infrastructure problems.
                    let status = match &e {
                        SessionError::NotFound { .. } => "not_found".to_string(),
                        other => format!("error: {}", other),
                    };
                    results.push(RemoveResponse {
                        status,
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
            records_a: store.len(Side::A).unwrap_or(0),
            records_b: store.len(Side::B).unwrap_or(0),
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

    /// Return confirmed crossmap pairs with cursor-based pagination.
    ///
    /// Pairs are sorted by `(a_id, b_id)`. The cursor is the `a_id` of the
    /// last entry on the previous page — the scan starts after that position.
    pub fn crossmap_pairs(
        &self,
        cursor: Option<&str>,
        limit: Option<usize>,
    ) -> CrossmapPairsResponse {
        let mut all = self.state.crossmap.pairs();
        let total = all.len();

        let start = match cursor {
            Some(c) => all
                .iter()
                .position(|(a, _)| a.as_str() > c)
                .unwrap_or(total),
            None => 0,
        };

        let end = match limit {
            Some(l) => (start + l).min(total),
            None => total,
        };

        // Partial sort: only sort the window we need, not all pairs.
        if start < end && end < total {
            let slice = &mut all[start..];
            let nth = end - start;
            slice.select_nth_unstable_by(nth, |a, b| a.0.cmp(&b.0));
            slice[..nth].sort_by(|a, b| a.0.cmp(&b.0));
        } else if start < end {
            all[start..end].sort_by(|a, b| a.0.cmp(&b.0));
        }

        let pairs: Vec<CrossmapPairEntry> = all[start..end]
            .iter()
            .map(|(a, b)| CrossmapPairEntry {
                a_id: a.clone(),
                b_id: b.clone(),
            })
            .collect();

        let next_cursor = if end < total {
            pairs.last().map(|p| p.a_id.clone())
        } else {
            None
        };

        CrossmapPairsResponse {
            total,
            pairs,
            next_cursor,
        }
    }

    /// Return unmatched record IDs with cursor-based pagination.
    ///
    /// IDs are already sorted. The cursor is the last ID on the previous
    /// page — the scan starts after that position.
    pub fn unmatched_records(
        &self,
        side: Side,
        cursor: Option<&str>,
        limit: Option<usize>,
        include_records: bool,
    ) -> UnmatchedResponse {
        let store = &self.state.store;
        let ids = store.unmatched_ids(side).unwrap_or_default(); // already sorted
        let total = ids.len();

        let start = match cursor {
            Some(c) => ids.iter().position(|id| id.as_str() > c).unwrap_or(total),
            None => 0,
        };

        let end = match limit {
            Some(l) => (start + l).min(total),
            None => total,
        };

        let records: Vec<UnmatchedEntry> = ids[start..end]
            .iter()
            .map(|id| {
                let record = if include_records {
                    store.get(side, id).unwrap_or(None)
                } else {
                    None
                };
                UnmatchedEntry {
                    id: id.clone(),
                    record,
                }
            })
            .collect();

        let next_cursor = if end < total {
            records.last().map(|r| r.id.clone())
        } else {
            None
        };

        UnmatchedResponse {
            side,
            total,
            records,
            next_cursor,
        }
    }

    /// Return crossmap coverage statistics.
    pub fn crossmap_stats(&self) -> CrossmapStatsResponse {
        let store = &self.state.store;
        let records_a = store.len(Side::A).unwrap_or(0);
        let records_b = store.len(Side::B).unwrap_or(0);
        let crossmap_pairs = self.state.crossmap.len();
        let unmatched_a = store.unmatched_count(Side::A).unwrap_or(0);
        let unmatched_b = store.unmatched_count(Side::B).unwrap_or(0);
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

    // -------------------------------------------------------------------
    // Enroll mode
    // -------------------------------------------------------------------

    /// Enroll a record into the single pool, returning scored edges.
    ///
    /// 1. Encode combined vector
    /// 2. Score against the pool (same side, excluding self)
    /// 3. Add record to pool (store, blocking, BM25, synonym, vector index)
    /// 4. Return edges above `review_floor`, capped at `top_n`
    pub fn enroll(&self, record: Record) -> Result<EnrollResponse, SessionError> {
        let config = &self.state.config;
        let store = &self.state.store;
        let side = Side::A; // Enroll always uses A-side as the pool.

        // Extract ID
        let id_field = &config.datasets.a.id_field;
        let id = record
            .get(id_field)
            .ok_or_else(|| SessionError::MissingField {
                field: id_field.clone(),
            })?
            .to_string();

        if id.is_empty() {
            return Err(SessionError::MissingField {
                field: id_field.clone(),
            });
        }

        // 1. Encode combined vector and upsert into index
        let combined_vec = self.encode_combined(&record, &self.emb_specs, true)?;
        if let Some(idx) = self.state.a.combined_index.as_ref()
            && !combined_vec.is_empty()
        {
            let _ = idx.upsert(&id, &combined_vec, &record, side);
        }

        // 2. Score against the pool (same side)
        let top_n = config.top_n.unwrap_or(5);
        let ann_candidates = config.ann_candidates.unwrap_or(50);
        let bm25_candidates = config.bm25_candidates.unwrap_or(10);

        let blocked_ids: Vec<String> = if config.blocking.enabled {
            store.blocking_query(&record, side, side)?
        } else {
            store.ids(side)?
        };

        let pool_state = &self.state.a;

        // Acquire synonym index read lock (if configured).
        let syn_guard = pool_state
            .synonym_index
            .as_ref()
            .map(|lock| lock.read().unwrap_or_else(|e| e.into_inner()));

        let synonym_dict = self.state.synonym_dictionary.as_deref();

        // BM25 candidate generation (SimpleBm25: lock-free, no commit).
        let (bm25_cand_ids, bm25_scores_map) = if let Some(ref pool_bm25) = pool_state.bm25_index {
            let query_text = pool_bm25.query_text_for(&record, side);
            let self_score = pool_bm25.analytical_self_score(&query_text);
            let raw_results = pool_bm25.score_blocked(&query_text, &blocked_ids, bm25_candidates);
            let scored: Vec<(String, f64)> = raw_results
                .into_iter()
                .map(|(cid, raw)| {
                    let norm = normalise_bm25(raw, self_score);
                    (cid, norm)
                })
                .collect();
            let ids: Vec<String> = scored.iter().map(|(cid, _)| cid.clone()).collect();
            let map: std::collections::HashMap<String, f64> = scored.into_iter().collect();
            (ids, map)
        } else {
            (Vec::new(), std::collections::HashMap::new())
        };

        // Synonym candidate generation.
        let synonym_cand_ids = if !config.synonym_fields.is_empty() {
            if let Some(ref guard) = syn_guard {
                pipeline::synonym_candidate_stage(guard, &record, side, &config.synonym_fields)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let scoring_query = pipeline::ScoringQuery {
            id: &id,
            record: &record,
            side,
            combined_vec: &combined_vec,
        };
        let scoring_pool = pipeline::ScoringPool {
            store: store.as_ref(),
            side, // pool_side = same side
            combined_index: pool_state.combined_index.as_deref(),
            blocked_ids: &blocked_ids,
            bm25_candidate_ids: &bm25_cand_ids,
            bm25_scores_map: &bm25_scores_map,
            synonym_candidate_ids: &synonym_cand_ids,
            synonym_dictionary: synonym_dict,
            exclusions: &self.state.exclusions,
        };
        let results =
            pipeline::score_pool(&scoring_query, &scoring_pool, config, ann_candidates, top_n);

        // 3. Add record to pool
        // Handle upsert: remove old blocking entry if exists
        if let Some(old_rec) = store.get(side, &id)? {
            store.blocking_remove(side, &id, &old_rec)?;
        }

        // WAL append
        if let Err(e) = self.state.wal.append_upsert(side, &record) {
            tracing::warn!(error = %e, "WAL append failed for enroll upsert");
        }

        // Insert into store
        store.insert(side, &id, &record)?;
        store.mark_unmatched(side, &id)?;

        // Update blocking index
        store.blocking_insert(side, &id, &record)?;

        // Update BM25 index (SimpleBm25: lock-free, instant visibility)
        if let Some(ref bm25) = pool_state.bm25_index {
            bm25.upsert(&id, &record);
        }

        // Update synonym index
        if let Some(ref syn_lock) = pool_state.synonym_index {
            let mut syn = syn_lock.write().unwrap_or_else(|e| e.into_inner());
            syn.upsert(&id, &record, side, &config.synonym_fields);
        }

        self.upsert_count.fetch_add(1, Ordering::Relaxed);

        // 4. Build response — filter by review_floor, cap at top_n
        let review_floor = config.thresholds.review_floor;
        let edges: Vec<EnrollEdge> = results
            .iter()
            .filter(|r| r.score >= review_floor)
            .take(top_n)
            .map(|r| EnrollEdge {
                id: r.matched_id.clone(),
                score: r.score,
                field_scores: r
                    .field_scores
                    .iter()
                    .map(|fs| EnrollFieldScore {
                        // In enroll mode field_a == field_b; use field_a
                        field: fs.field_a.clone(),
                        method: fs.method.clone(),
                        score: fs.score,
                        weight: fs.weight,
                    })
                    .collect(),
            })
            .collect();

        Ok(EnrollResponse {
            id,
            enrolled: true,
            edges,
        })
    }

    /// Enroll a batch of records sequentially.
    ///
    /// Each record is scored against the pool (including previously enrolled
    /// records from this batch), then added to the pool.
    ///
    /// Per-item errors are collected into the response (with `enrolled: false`)
    /// rather than aborting the entire batch. This matches the error handling
    /// policy of `upsert_batch` and `match_batch`.
    pub fn enroll_batch(&self, records: Vec<Record>) -> Result<BatchEnrollResponse, SessionError> {
        if records.is_empty() {
            return Err(SessionError::BatchValidation {
                message: "records array is empty".into(),
            });
        }
        if records.len() > Self::MAX_BATCH_SIZE {
            return Err(SessionError::BatchValidation {
                message: format!(
                    "batch size {} exceeds maximum of {}",
                    records.len(),
                    Self::MAX_BATCH_SIZE
                ),
            });
        }

        let mut results = Vec::with_capacity(records.len());
        for record in records {
            match self.enroll(record) {
                Ok(resp) => results.push(resp),
                Err(e) => {
                    tracing::warn!(error = %e, "enroll-batch item failed");
                    results.push(EnrollResponse {
                        id: format!("error: {}", e),
                        enrolled: false,
                        edges: vec![],
                    });
                }
            }
        }
        Ok(BatchEnrollResponse { results })
    }

    /// Return pending review-band matches with cursor-based pagination.
    ///
    /// Reviews are sorted by score descending, then by `id` for stability.
    /// The cursor encodes `score|id` of the last entry on the previous page.
    pub fn review_list(&self, cursor: Option<&str>, limit: Option<usize>) -> ReviewListResponse {
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
        // Sort by score descending, then by id for stability.
        reviews.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        let total = reviews.len();

        // Parse cursor: "score|id" — find first entry strictly after it.
        let start = match cursor {
            Some(c) => {
                if let Some((score_str, cursor_id)) = c.split_once('|') {
                    let cursor_score: f64 = score_str.parse().unwrap_or(0.0);
                    reviews
                        .iter()
                        .position(|r| {
                            r.score < cursor_score
                                || (r.score == cursor_score && r.id.as_str() > cursor_id)
                        })
                        .unwrap_or(total)
                } else {
                    0
                }
            }
            None => 0,
        };

        let end = match limit {
            Some(l) => (start + l).min(total),
            None => total,
        };

        let page = reviews[start..end].to_vec();

        let next_cursor = if end < total {
            page.last().map(|r| format!("{}|{}", r.score, r.id))
        } else {
            None
        };

        ReviewListResponse {
            total,
            reviews: page,
            next_cursor,
        }
    }
}

/// Convert a `MelderError` to a `SessionError`.
fn melder_to_session_err(e: crate::error::MelderError) -> SessionError {
    match e {
        crate::error::MelderError::Encoder(enc) => SessionError::Encoder(enc),
        other => SessionError::Encoder(crate::error::EncoderError::Inference(format!(
            "encoding failed: {}",
            other
        ))),
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
