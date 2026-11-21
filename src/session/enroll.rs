//! Typed single-collection enrollment session using the shared search pipeline.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use tracing::info_span;

use super::response::*;
use super::{SessionError, melder_to_session_err};
use crate::matching::pipeline;
use crate::models::{MatchResult, Record, Side};
use crate::store::RecordCollection;
/// Enrollment lifecycle, with no matching-only resources.
pub struct EnrollSession {
    pub state: Arc<crate::state::enroll::EnrollState>,
    search_enroll: pipeline::SearchPlan,
    scoring_log: Option<crate::output::scoring_log::ScoringLogSender>,
    hook_tx: Option<tokio::sync::mpsc::Sender<crate::hooks::HookEvent>>,
    pub upsert_count: AtomicU64,
    pub match_count: AtomicU64,
    pub start_time: std::time::Instant,
}
impl EnrollSession {
    /// Acquire the enrollment mutation/snapshot boundary.
    pub fn assignment_guard(&self) -> Result<Option<std::sync::MutexGuard<'_, ()>>, SessionError> {
        Ok(Some(
            self.state.gate.lock().unwrap_or_else(|e| e.into_inner()),
        ))
    }

    /// Flush buffered enrollment evidence.
    pub fn flush_assignment_log(&self) -> Result<(), SessionError> {
        self.state.wal.flush()?;
        Ok(())
    }

    /// Number of authoritative enrollment records.
    pub fn count(&self) -> Result<usize, SessionError> {
        Ok(self.state.collection.len()?)
    }

    /// Request counters and uptime in the legacy wire shape.
    pub fn status(&self) -> StatusResponse {
        StatusResponse {
            job: self.state.config.job.name.clone(),
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            upserts: self.upsert_count.load(Ordering::Relaxed),
            matches: self.match_count.load(Ordering::Relaxed),
        }
    }

    /// Enrollment has no paired membership.
    pub fn crossmap_stats(&self) -> CrossmapStatsResponse {
        let count = self.count().unwrap_or(0);
        CrossmapStatsResponse {
            records_a: count,
            records_b: 0,
            crossmap_pairs: 0,
            matched_a: 0,
            matched_b: 0,
            unmatched_a: count,
            unmatched_b: 0,
            coverage_a: 0.0,
            coverage_b: 0.0,
        }
    }

    /// Health in the historical wire shape.
    pub fn health(&self) -> HealthResponse {
        HealthResponse {
            status: "ready".into(),
            model: self.state.config.embeddings.model.clone(),
            records_a: self.count().unwrap_or(0),
            records_b: 0,
            crossmap_entries: 0,
        }
    }

    /// Remove an exclusion without introducing any new matching policy.
    pub fn unexclude(&self, id: &str, other_id: &str) -> Result<UnexcludeResponse, SessionError> {
        let _guard = self.assignment_guard()?;
        let removed = self.state.exclusions.contains(id, other_id);
        if removed {
            self.state.exclusions.remove(id, other_id);
            self.state
                .wal
                .append(&crate::state::match_log::MatchLogEvent::Unexclude {
                    a_id: id.to_owned(),
                    b_id: other_id.to_owned(),
                })?;
        }
        Ok(UnexcludeResponse {
            removed,
            a_id: id.to_owned(),
            b_id: other_id.to_owned(),
        })
    }

    /// Remove an enrollment record, invalidating its evidence through the WAL.
    pub fn remove_record(&self, id: &str) -> Result<RemoveResponse, SessionError> {
        let _guard = self.state.gate.lock().unwrap_or_else(|e| e.into_inner());
        let collection = &self.state.collection;
        let record = collection.get(id)?.ok_or_else(|| SessionError::NotFound {
            message: format!("record {} not found on side A", id),
        })?;
        collection.blocking_remove(id, &record)?;
        if let Some(index) = &self.state.pool.combined_index {
            let _ = index.remove(id);
        }
        if let Some(index) = &self.state.pool.bm25_index {
            index.remove(id);
        }
        if let Some(index) = &self.state.pool.synonym_index {
            index
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .remove(id, Side::A);
        }
        self.state
            .wal
            .append(&crate::state::match_log::MatchLogEvent::RemoveRecord {
                side: Side::A,
                id: id.to_owned(),
            })?;
        collection.remove(id)?;
        Ok(RemoveResponse {
            status: "removed".into(),
            id: id.to_owned(),
            side: Side::A,
            crossmap_broken: Vec::new(),
            assignment: None,
        })
    }

    /// Query one enrollment record using the historical HTTP response shape.
    pub fn query_record(&self, id: &str) -> Result<QueryResponse, SessionError> {
        let record = self
            .state
            .collection
            .get(id)?
            .ok_or_else(|| SessionError::NotFound {
                message: format!("record {} not found on side A", id),
            })?;
        Ok(QueryResponse {
            id: id.to_owned(),
            side: Side::A,
            record,
            crossmap: QueryCrossmap {
                status: "unmatched".into(),
                paired_id: None,
                paired_record: None,
            },
        })
    }

    /// Exclude two enrollment IDs; historical A/B event fields are wire-only.
    pub fn exclude(&self, id: &str, other_id: &str) -> Result<ExcludeResponse, SessionError> {
        let _guard = self.state.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.state
            .wal
            .append(&crate::state::match_log::MatchLogEvent::Exclude {
                a_id: id.to_owned(),
                b_id: other_id.to_owned(),
            })?;
        self.state.exclusions.add(id, other_id);
        if let Some(tx) = &self.hook_tx
            && tx
                .try_send(crate::hooks::HookEvent::Exclude {
                    a_id: id.to_owned(),
                    b_id: other_id.to_owned(),
                    match_was_broken: false,
                })
                .is_err()
        {
            tracing::warn!("hook event dropped: channel unavailable");
        }
        Ok(ExcludeResponse {
            excluded: true,
            match_was_broken: false,
            a_id: id.to_owned(),
            b_id: other_id.to_owned(),
        })
    }
    /// Bind an enrollment state and optional scoring log.
    pub fn new(
        state: Arc<crate::state::enroll::EnrollState>,
        scoring_log: Option<crate::output::scoring_log::ScoringLogSender>,
    ) -> Self {
        let search_enroll = pipeline::SearchPlan::enroll(&state.config);
        Self {
            search_enroll,
            state,
            scoring_log,
            hook_tx: None,
            upsert_count: AtomicU64::new(0),
            match_count: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
        }
    }
    /// Attach the configured best-effort enrollment hook channel.
    pub(crate) fn with_hooks(
        mut self,
        tx: Option<tokio::sync::mpsc::Sender<crate::hooks::HookEvent>>,
    ) -> Self {
        self.hook_tx = tx;
        self
    }

    const MAX_BATCH_SIZE: usize = 1000;
    fn encode_combined(&self, record: &Record) -> Result<Vec<f32>, SessionError> {
        encode_combined(
            record,
            self.search_enroll.embedding_fields(),
            self.state.encoder_pool.as_ref(),
            &self.state.coordinator,
        )
    }

    fn score_against_pool(
        &self,
        id: &str,
        record: &Record,
        combined_vec: &[f32],
        blocked_ids: &[String],
    ) -> Vec<MatchResult> {
        let config = &self.state.config;
        let pool = &self.state.pool;
        // BM25 candidate generation.
        let (bm25_cand_ids, bm25_scores_map) = if let Some(ref pool_bm25) = pool.bm25_index {
            let _span = info_span!("bm25_score").entered();
            pipeline::bm25_candidate_stage(
                pool_bm25,
                record,
                Side::A,
                Some(blocked_ids),
                config.bm25_candidates.unwrap_or(10),
            )
        } else {
            (Vec::new(), std::collections::HashMap::new())
        };

        // Synonym candidate generation.
        let syn_guard = pool
            .synonym_index
            .as_ref()
            .map(|mtx| mtx.read().unwrap_or_else(|e| e.into_inner()));
        let synonym_cand_ids = syn_guard
            .as_ref()
            .map(|guard| {
                pipeline::synonym_candidates(
                    guard,
                    record,
                    config.synonym_fields.iter().map(|field| {
                        let name = field.field.as_str();
                        (
                            name,
                            name,
                            name,
                            field.generators.first().map_or(3, |g| g.min_length),
                        )
                    }),
                )
            })
            .unwrap_or_default();

        let _span = info_span!("score_pool").entered();
        let store = &self.state.collection;
        let scoring_query = pipeline::PreparedQuery {
            id,
            record,
            combined_vec,
        };
        let scoring_pool = pipeline::SearchPool {
            collection: store,
            combined_index: pool.combined_index.as_deref(),
            synonym_dictionary: self.state.synonym_dictionary.as_deref(),
        };
        let inputs = pipeline::SearchCandidates {
            eligible_ids: blocked_ids,
            bm25_candidate_ids: &bm25_cand_ids,
            bm25_scores_map: &bm25_scores_map,
            synonym_candidate_ids: &synonym_cand_ids,
            exclusions: &self.state.exclusions,
            same_collection: true,
        };
        let plan = &self.search_enroll;
        let results = pipeline::search(
            &scoring_query,
            &scoring_pool,
            plan,
            config,
            &inputs,
            config.ann_candidates.unwrap_or(50),
            config.top_n.unwrap_or(5),
            pipeline::SearchPolicy::Ordinary,
        )
        .expect("lenient scoring does not propagate store errors");

        if let Some(sl) = &self.scoring_log {
            sl.send(id, Side::A, &results);
        }

        results
    }

    /// Enroll a record through the shared single-collection pipeline.
    pub fn enroll(&self, record: Record) -> Result<EnrollResponse, SessionError> {
        let id = self.enroll_id(&record)?;
        // Encoding is pure preparation: do not serialize it with pool mutations.
        let combined_vec = self.encode_combined(&record)?;
        self.enroll_prepared(record, id, combined_vec)
    }

    fn enroll_id(&self, record: &Record) -> Result<String, SessionError> {
        let field = &self.state.config.dataset.id_field;
        record
            .get(field)
            .filter(|id| !id.is_empty())
            .cloned()
            .ok_or_else(|| SessionError::MissingField {
                field: field.to_owned(),
            })
    }

    fn enroll_prepared(
        &self,
        record: Record,
        id: String,
        combined_vec: Vec<f32>,
    ) -> Result<EnrollResponse, SessionError> {
        let _guard = self.state.gate.lock().unwrap_or_else(|e| e.into_inner());
        let config = &self.state.config;
        let store = &self.state.collection;
        let side = Side::A; // The single enrollment pool.

        // Install the prepared vector only while holding the mutation gate.
        if let Some(idx) = self.state.pool.combined_index.as_ref()
            && !combined_vec.is_empty()
        {
            let _ = idx.upsert(&id, &combined_vec, &record, side);
        }

        // 2. Score against the pool (same side)
        let top_n = config.top_n.unwrap_or(5);

        let blocked_ids: Vec<String> = if config.blocking.enabled {
            store.blocking_query(&record)?
        } else {
            store.ids()?
        };

        let pool_state = &self.state.pool;

        let results = self.score_against_pool(&id, &record, &combined_vec, &blocked_ids);

        // 3. Add record to pool
        // Handle upsert: remove old blocking entry if exists
        if let Some(old_rec) = store.get(&id)? {
            store.blocking_remove(&id, &old_rec)?;
        }

        // WAL append
        self.state.wal.append_upsert(side, &record)?;

        // Insert into store
        store.insert(&id, &record)?;

        // Update blocking index
        store.blocking_insert(&id, &record)?;

        // Update BM25 index (SimpleBm25: lock-free, instant visibility)
        if let Some(ref bm25) = pool_state.bm25_index {
            bm25.upsert(&id, &record);
        }

        // Update synonym index
        if let Some(ref syn_lock) = pool_state.synonym_index {
            let mut syn = syn_lock.write().unwrap_or_else(|e| e.into_inner());
            syn.upsert_enroll(&id, &record, &config.synonym_fields);
        }

        self.upsert_count.fetch_add(1, Ordering::Relaxed);

        // 4. Build response — filter by review_floor, cap at top_n
        let review_floor = config.thresholds.review_floor;
        let edges: Vec<EnrollEdge> = results
            .iter()
            .filter(|r| r.matched_id != id && r.score >= review_floor)
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

        // Essential single-pool evidence belongs in the WAL, not the optional,
        // asynchronous scoring log. Never express these edges as bijective claims.
        self.state
            .wal
            .append(&crate::state::match_log::MatchLogEvent::EnrollOutcome {
                query_id: id.clone(),
                edges: edges
                    .iter()
                    .map(|edge| crate::state::match_log::EnrollEvidence {
                        id: edge.id.clone(),
                        score: edge.score,
                        field_scores: if self.scoring_log.is_some() {
                            edge.field_scores
                                .iter()
                                .map(|field| crate::state::match_log::EnrollEvidenceField {
                                    field: field.field.clone(),
                                    method: field.method.clone(),
                                    score: field.score,
                                    weight: field.weight,
                                })
                                .collect()
                        } else {
                            Vec::new()
                        },
                    })
                    .collect(),
            })?;

        Ok(EnrollResponse {
            id,
            enrolled: true,
            edges,
        })
    }

    fn encode_enroll_batch(&self, records: &[&Record]) -> Result<Vec<Vec<f32>>, SessionError> {
        let mut combined = vec![Vec::new(); records.len()];
        if records.is_empty() {
            return Ok(combined);
        }
        for (field, weight) in self.search_enroll.embedding_fields() {
            let texts: Vec<&str> = records
                .iter()
                .map(|r| r.get(field).map(|v| v.trim()).unwrap_or_default())
                .collect();
            let vectors = self.state.encoder_pool.encode(&texts)?;
            if vectors.len() != records.len()
                || vectors
                    .iter()
                    .any(|v| v.len() != self.state.encoder_pool.dim())
            {
                return Err(crate::error::EncoderError::Inference(
                    "enrollment encoder returned invalid vector shape".into(),
                )
                .into());
            }
            let scale = weight.sqrt() as f32;
            for (out, vector) in combined.iter_mut().zip(vectors) {
                out.extend(vector.into_iter().map(|v| v * scale));
            }
        }
        Ok(combined)
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

        // Validate before encoding, but keep per-item errors and input order.
        let ids: Vec<_> = records
            .iter()
            .map(|record| self.enroll_id(record))
            .collect();
        let valid: Vec<_> = records
            .iter()
            .zip(&ids)
            .filter_map(|(r, id)| id.as_ref().ok().map(|_| r))
            .collect();
        let prepared = self.encode_enroll_batch(&valid);
        let mut vectors = prepared.as_ref().ok().map(|v| v.iter());
        let mut results = Vec::with_capacity(records.len());
        for (record, id) in records.into_iter().zip(ids) {
            let result = id.and_then(|id| {
                // A batch encoder failure must not turn one bad item into a
                // request-wide failure. Retry individually to preserve semantics.
                let vector = match vectors.as_mut() {
                    Some(v) => v
                        .next()
                        .expect("one prepared vector per validated record")
                        .clone(),
                    None => self.encode_combined(&record)?,
                };
                self.enroll_prepared(record, id, vector)
            });
            match result {
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
}

pub(super) fn encode_combined<'a>(
    record: &Record,
    emb_specs: impl Iterator<Item = (&'a str, f64)> + Clone,
    encoder_pool: &dyn crate::encoder::Encoder,
    coordinator: &Option<crate::encoder::coordinator::EncoderCoordinator>,
) -> Result<Vec<f32>, SessionError> {
    let _span = info_span!("encode_combined").entered();
    if let Some(coordinator) = coordinator {
        // Coordinator path: submit texts for batched encoding.
        // Fully synchronous — safe from rayon/spawn_blocking threads.
        let field_dim = encoder_pool.dim();
        let mut combined = Vec::with_capacity(field_dim * emb_specs.clone().count());

        let texts: Vec<String> = emb_specs
            .clone()
            .map(|(field_name, _weight)| {
                record
                    .get(field_name)
                    .map(|v| v.trim().to_string())
                    .unwrap_or_default()
            })
            .collect();

        let vecs = coordinator
            .encode_many(texts)
            .map_err(|e| melder_to_session_err(crate::error::MelderError::Encoder(e)))?;

        for (vec, (_, weight)) in vecs.into_iter().zip(emb_specs) {
            let sqrt_w = weight.sqrt() as f32;
            let scaled: Vec<f32> = vec.into_iter().map(|v| v * sqrt_w).collect();
            combined.extend_from_slice(&scaled);
        }
        Ok(combined)
    } else {
        // Direct path: no coordinator, encode synchronously.
        let mut combined = Vec::with_capacity(encoder_pool.dim() * emb_specs.clone().count());
        for (field, weight) in emb_specs {
            let text = record.get(field).map(|v| v.trim()).unwrap_or_default();
            let mut vector = encoder_pool.encode_one(text)?;
            let scale = weight.sqrt() as f32;
            for value in &mut vector {
                *value *= scale;
            }
            combined.extend_from_slice(&vector);
        }
        Ok(combined)
    }
}
