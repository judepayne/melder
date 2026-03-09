//! Session: the core live matching logic.
//!
//! Provides upsert, try-match, and crossmap management operations.
//! All operations are synchronous internally; the HTTP layer wraps them
//! in `spawn_blocking` as needed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::error::SessionError;
use crate::matching::pipeline;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::state::live::LiveMatchState;

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
}

use response::*;

/// Live matching session.
pub struct Session {
    pub state: Arc<LiveMatchState>,
    pub start_time: Instant,
    pub upsert_count: AtomicU64,
    pub match_count: AtomicU64,
}

impl Session {
    pub fn new(state: Arc<LiveMatchState>) -> Self {
        Self {
            state,
            start_time: Instant::now(),
            upsert_count: AtomicU64::new(0),
            match_count: AtomicU64::new(0),
        }
    }

    /// Upsert a record into the live state and attempt matching.
    ///
    /// Encodes per-field embedding vectors, stores them in FieldVectors,
    /// then performs insertion and scoring.
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

        // Encode per-field embedding vectors
        let emb_fields = crate::vectordb::embedding_field_keys(config);
        let this_side = self.state.side(side);
        for (field_key, field_a_name, field_b_name) in &emb_fields {
            let field_name = if is_a_side {
                field_a_name
            } else {
                field_b_name
            };
            let text = record
                .get(field_name)
                .map(|v| v.trim().to_string())
                .unwrap_or_default();
            let vec = self.state.encoder_pool.encode_one(&text)?;
            this_side.field_vecs.insert(&id, field_key, vec);
        }

        self.upsert_record_inner(side, record)
    }

    /// Upsert a record whose per-field vectors have already been stored in
    /// FieldVectors. Handles insertion, crossmap management, and scoring.
    fn upsert_record_inner(
        &self,
        side: Side,
        record: Record,
    ) -> Result<UpsertResponse, SessionError> {
        let config = &self.state.config;
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

        let this_side = self.state.side(side);
        let opp_side = self.state.opposite_side(side);

        // 2. Check if existing record
        let mut old_mapping: Option<OldMapping> = None;
        let mut status = "added";

        if this_side.records.contains_key(&id) {
            status = "updated";

            // Check crossmap — if matched, break the pair
            let paired = {
                let cm = self
                    .state
                    .crossmap
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                match side {
                    Side::A => cm.get_b(&id).map(|s| s.to_string()),
                    Side::B => cm.get_a(&id).map(|s| s.to_string()),
                }
            };

            if let Some(paired_id) = paired {
                // Break the pair
                let (a_id, b_id) = match side {
                    Side::A => (id.clone(), paired_id.clone()),
                    Side::B => (paired_id.clone(), id.clone()),
                };
                {
                    let mut cm = self
                        .state
                        .crossmap
                        .write()
                        .unwrap_or_else(|e| e.into_inner());
                    cm.remove(&a_id, &b_id);
                }
                // Add both back to unmatched
                self.state.a.unmatched.insert(a_id.clone());
                self.state.b.unmatched.insert(b_id.clone());

                // WAL
                let _ = self.state.wal.append(&WalEvent::CrossMapBreak {
                    a_id: a_id.clone(),
                    b_id: b_id.clone(),
                });

                old_mapping = Some(OldMapping { a_id, b_id });
                self.state.mark_crossmap_dirty();
            }

            // Remove old record from blocking index
            if let Some(old_rec) = this_side.records.get(&id) {
                let mut bi = this_side
                    .blocking_index
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                bi.remove(&id, &old_rec, side);
            }
        }

        // 3. Insert/replace record
        this_side.records.insert(id.clone(), record.clone());

        // 4. Add to unmatched
        this_side.unmatched.insert(id.clone());

        // 5. Update blocking index
        {
            let mut bi = this_side
                .blocking_index
                .write()
                .unwrap_or_else(|e| e.into_inner());
            bi.insert(&id, &record, side);
        }

        // 6. (Per-field vectors already stored by caller)

        // 7. WAL append
        let _ = self.state.wal.append(&WalEvent::UpsertRecord {
            side,
            record: record.clone(),
        });

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
                    this_side
                        .common_id_index
                        .insert(cid_val.to_string(), id.clone());

                    // Check opposite side for a matching common ID
                    if let Some(opp_entry) = opp_side.common_id_index.get(cid_val) {
                        let opp_id = opp_entry.value().clone();
                        drop(opp_entry);

                        // Break any existing crossmap for either record
                        {
                            let cm = self
                                .state
                                .crossmap
                                .read()
                                .unwrap_or_else(|e| e.into_inner());
                            let existing_this = match side {
                                Side::A => cm.get_b(&id).map(|s| s.to_string()),
                                Side::B => cm.get_a(&id).map(|s| s.to_string()),
                            };
                            let existing_opp = match side {
                                Side::A => cm.get_a(&opp_id).map(|s| s.to_string()),
                                Side::B => cm.get_b(&opp_id).map(|s| s.to_string()),
                            };
                            drop(cm);
                            let mut cm = self
                                .state
                                .crossmap
                                .write()
                                .unwrap_or_else(|e| e.into_inner());
                            if let Some(old_paired) = existing_this {
                                match side {
                                    Side::A => cm.remove(&id, &old_paired),
                                    Side::B => cm.remove(&old_paired, &id),
                                }
                                opp_side.unmatched.insert(old_paired);
                            }
                            if let Some(old_paired) = existing_opp {
                                match side {
                                    Side::A => cm.remove(&old_paired, &opp_id),
                                    Side::B => cm.remove(&opp_id, &old_paired),
                                }
                                this_side.unmatched.insert(old_paired);
                            }
                        }

                        // Create the common ID match
                        let (a_id, b_id) = match side {
                            Side::A => (id.clone(), opp_id.clone()),
                            Side::B => (opp_id.clone(), id.clone()),
                        };
                        {
                            let mut cm = self
                                .state
                                .crossmap
                                .write()
                                .unwrap_or_else(|e| e.into_inner());
                            cm.add(&a_id, &b_id);
                        }
                        self.state.a.unmatched.remove(&a_id);
                        self.state.b.unmatched.remove(&b_id);

                        let _ = self.state.wal.append(&WalEvent::CrossMapConfirm {
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                        });
                        self.state.mark_crossmap_dirty();

                        let opp_record = opp_side.records.get(&opp_id).map(|r| r.value().clone());
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
        let top_n = config.live.top_n.unwrap_or(5);
        let opp_bi = opp_side
            .blocking_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let results = pipeline::score_pool(
            &id,
            &record,
            side,
            &opp_side.records,
            &this_side.field_vecs,
            &opp_side.field_vecs,
            Some(&opp_bi),
            config,
            top_n,
        );
        drop(opp_bi);

        // 10. Classify top result
        let classification;
        let matches = build_match_entries(&results);

        if let Some(top) = results.first() {
            classification = top.classification.as_str().to_string();
            if top.classification == Classification::Auto {
                let (a_id, b_id) = match side {
                    Side::A => (id.clone(), top.matched_id.clone()),
                    Side::B => (top.matched_id.clone(), id.clone()),
                };
                {
                    let mut cm = self
                        .state
                        .crossmap
                        .write()
                        .unwrap_or_else(|e| e.into_inner());
                    cm.add(&a_id, &b_id);
                }
                self.state.a.unmatched.remove(&a_id);
                self.state.b.unmatched.remove(&b_id);

                let _ = self
                    .state
                    .wal
                    .append(&WalEvent::CrossMapConfirm { a_id, b_id });
                self.state.mark_crossmap_dirty();
            }
        } else {
            classification = "no_match".to_string();
        }

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
        let (this_side, _other_side) = match side {
            Side::A => (&self.state.a, &self.state.b),
            Side::B => (&self.state.b, &self.state.a),
        };

        // Check the record exists
        let record = this_side
            .records
            .get(id)
            .map(|r| r.value().clone())
            .ok_or_else(|| SessionError::MissingField {
                field: format!("record {} not found on side {:?}", id, side),
            })?;

        // Break any crossmap pair
        let mut broken = Vec::new();
        {
            let cm = self
                .state
                .crossmap
                .read()
                .unwrap_or_else(|e| e.into_inner());
            match side {
                Side::A => {
                    if let Some(b_id) = cm.get_b(id) {
                        broken.push(b_id.to_string());
                    }
                }
                Side::B => {
                    if let Some(a_id) = cm.get_a(id) {
                        broken.push(a_id.to_string());
                    }
                }
            }
        }
        if !broken.is_empty() {
            let mut cm = self
                .state
                .crossmap
                .write()
                .unwrap_or_else(|e| e.into_inner());
            for paired_id in &broken {
                match side {
                    Side::A => cm.remove(id, paired_id),
                    Side::B => cm.remove(paired_id, id),
                }
                // Re-add the opposite-side ID to its unmatched set
                _other_side.unmatched.insert(paired_id.clone());
            }
            self.state.mark_crossmap_dirty();
        }

        // Remove from blocking index
        {
            let mut bi = this_side
                .blocking_index
                .write()
                .unwrap_or_else(|e| e.into_inner());
            bi.remove(id, &record, side);
        }

        // Remove field vectors
        this_side.field_vecs.remove_record(id);

        // Remove from unmatched set
        this_side.unmatched.remove(id);

        // Remove from common_id_index if configured
        let cid_field = match side {
            Side::A => self.state.config.datasets.a.common_id_field.as_deref(),
            Side::B => self.state.config.datasets.b.common_id_field.as_deref(),
        };
        if let Some(cid_field) = cid_field {
            if let Some(cid_val) = record.get(cid_field) {
                let cid_val = cid_val.trim();
                if !cid_val.is_empty() {
                    this_side.common_id_index.remove(cid_val);
                }
            }
        }

        // Remove from records
        this_side.records.remove(id);

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
    ///
    /// Encodes per-field embedding vectors into a temporary FieldVectors
    /// store, then scores against the opposite side.
    pub fn try_match(&self, side: Side, record: Record) -> Result<MatchResponse, SessionError> {
        let config = &self.state.config;
        let is_a_side = side == Side::A;
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };
        let id = record.get(id_field).cloned().unwrap_or_default();

        // Build temporary per-field vectors for this query record
        let emb_fields = crate::vectordb::embedding_field_keys(config);
        let dim = self.state.encoder_pool.dim();
        let query_fv = crate::vectordb::field_vectors::FieldVectors::new(dim);
        for (field_key, field_a_name, field_b_name) in &emb_fields {
            let field_name = if is_a_side {
                field_a_name
            } else {
                field_b_name
            };
            let text = record
                .get(field_name)
                .map(|v| v.trim().to_string())
                .unwrap_or_default();
            let vec = self.state.encoder_pool.encode_one(&text)?;
            query_fv.insert(&id, field_key, vec);
        }

        self.try_match_inner(side, record, &id, &query_fv)
    }

    /// Try matching a record with pre-computed per-field vectors (read-only).
    ///
    /// Used by both the single-record path and the batch path.
    fn try_match_inner(
        &self,
        side: Side,
        record: Record,
        id: &str,
        query_field_vecs: &crate::vectordb::field_vectors::FieldVectors,
    ) -> Result<MatchResponse, SessionError> {
        let config = &self.state.config;

        let opp_side = self.state.opposite_side(side);

        // Check crossmap for existing match
        let existing = {
            let cm = self
                .state
                .crossmap
                .read()
                .unwrap_or_else(|e| e.into_inner());
            match side {
                Side::A => cm.get_b(&id).map(|s| s.to_string()),
                Side::B => cm.get_a(&id).map(|s| s.to_string()),
            }
        };

        if let Some(paired_id) = existing {
            // Return existing match from crossmap
            let matched_record = opp_side.records.get(&paired_id).map(|r| r.value().clone());

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
        let top_n = config.live.top_n.unwrap_or(5);
        let opp_bi = opp_side
            .blocking_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let results = pipeline::score_pool(
            id,
            &record,
            side,
            &opp_side.records,
            query_field_vecs,
            &opp_side.field_vecs,
            Some(&opp_bi),
            config,
            top_n,
        );
        drop(opp_bi);

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
        // Validate both IDs exist
        if !self.state.a.records.contains_key(a_id) {
            return Err(SessionError::MissingField {
                field: format!("a_id '{}' not found", a_id),
            });
        }
        if !self.state.b.records.contains_key(b_id) {
            return Err(SessionError::MissingField {
                field: format!("b_id '{}' not found", b_id),
            });
        }

        {
            let mut cm = self
                .state
                .crossmap
                .write()
                .unwrap_or_else(|e| e.into_inner());
            cm.add(a_id, b_id);
        }

        self.state.a.unmatched.remove(a_id);
        self.state.b.unmatched.remove(b_id);

        let _ = self.state.wal.append(&WalEvent::CrossMapConfirm {
            a_id: a_id.to_string(),
            b_id: b_id.to_string(),
        });
        self.state.mark_crossmap_dirty();

        Ok(ConfirmResponse {
            status: "confirmed".to_string(),
        })
    }

    /// Lookup a crossmap entry.
    pub fn lookup_crossmap(&self, id: &str, side: Side) -> Result<LookupResponse, SessionError> {
        let cm = self
            .state
            .crossmap
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let paired_id = match side {
            Side::A => cm.get_b(id).map(|s| s.to_string()),
            Side::B => cm.get_a(id).map(|s| s.to_string()),
        };

        if let Some(ref pid) = paired_id {
            let opp = self.state.opposite_side(side);
            let matched_record = opp.records.get(pid).map(|r| r.value().clone());

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
        // Validate the pair exists
        {
            let cm = self
                .state
                .crossmap
                .read()
                .unwrap_or_else(|e| e.into_inner());
            let existing = cm.get_b(a_id);
            if existing.map(|b| b != b_id).unwrap_or(true) {
                return Err(SessionError::MissingField {
                    field: format!("crossmap pair ({}, {}) not found", a_id, b_id),
                });
            }
        }

        {
            let mut cm = self
                .state
                .crossmap
                .write()
                .unwrap_or_else(|e| e.into_inner());
            cm.remove(a_id, b_id);
        }

        self.state.a.unmatched.insert(a_id.to_string());
        self.state.b.unmatched.insert(b_id.to_string());

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
        let (this_side, _other_side) = match side {
            Side::A => (&self.state.a, &self.state.b),
            Side::B => (&self.state.b, &self.state.a),
        };

        // Look up the record
        let record = this_side
            .records
            .get(id)
            .map(|r| r.value().clone())
            .ok_or_else(|| SessionError::MissingField {
                field: format!("record {} not found on side {:?}", id, side),
            })?;

        // Check crossmap
        let cm = self
            .state
            .crossmap
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let crossmap = match side {
            Side::A => {
                if let Some(b_id) = cm.get_b(id) {
                    let paired_record = _other_side.records.get(b_id).map(|r| r.value().clone());
                    QueryCrossmap {
                        status: "matched".to_string(),
                        paired_id: Some(b_id.to_string()),
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
                if let Some(a_id) = cm.get_a(id) {
                    let paired_record = _other_side.records.get(a_id).map(|r| r.value().clone());
                    QueryCrossmap {
                        status: "matched".to_string(),
                        paired_id: Some(a_id.to_string()),
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
    ///
    /// For each embedding field, all record texts are gathered and encoded
    /// in a single ONNX batch call. The resulting per-field vectors are
    /// stored in FieldVectors, then records are inserted and scored
    /// sequentially so each record sees the crossmap state left by the
    /// previous one.
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
        let emb_fields = crate::vectordb::embedding_field_keys(config);
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

        // 2. Batch encode per embedding field — one ONNX call per field
        for (field_key, field_a_name, field_b_name) in &emb_fields {
            let field_name = if is_a_side {
                field_a_name
            } else {
                field_b_name
            };
            let texts: Vec<String> = records
                .iter()
                .map(|r| {
                    r.get(field_name)
                        .map(|v| v.trim().to_string())
                        .unwrap_or_default()
                })
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let vecs = self.state.encoder_pool.encode(&text_refs)?;
            for (rec_id, vec) in ids.iter().zip(vecs.into_iter()) {
                this_side.field_vecs.insert(rec_id, field_key, vec);
            }
        }

        // 3. Insert + score each record sequentially (vectors already stored)
        let mut results = Vec::with_capacity(records.len());
        for record in records {
            let resp = self.upsert_record_inner(side, record);
            match resp {
                Ok(r) => results.push(r),
                Err(e) => {
                    // Per-record error: produce an error entry rather
                    // than failing the whole batch.
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
    ///
    /// For each embedding field, all record texts are gathered and encoded
    /// in a single ONNX batch call. The resulting per-field vectors are
    /// stored in a temporary FieldVectors, then each record is scored
    /// sequentially.
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
        let emb_fields = crate::vectordb::embedding_field_keys(config);
        let id_field = match side {
            Side::A => &config.datasets.a.id_field,
            Side::B => &config.datasets.b.id_field,
        };

        // 1. Extract record IDs
        let ids: Vec<String> = records
            .iter()
            .map(|r| r.get(id_field).cloned().unwrap_or_default())
            .collect();

        // 2. Build temporary FieldVectors with batch-encoded per-field vectors
        let dim = self.state.encoder_pool.dim();
        let query_fv = crate::vectordb::field_vectors::FieldVectors::new(dim);
        for (field_key, field_a_name, field_b_name) in &emb_fields {
            let field_name = if is_a_side {
                field_a_name
            } else {
                field_b_name
            };
            let texts: Vec<String> = records
                .iter()
                .map(|r| {
                    r.get(field_name)
                        .map(|v| v.trim().to_string())
                        .unwrap_or_default()
                })
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let vecs = self.state.encoder_pool.encode(&text_refs)?;
            for (rec_id, vec) in ids.iter().zip(vecs.into_iter()) {
                query_fv.insert(rec_id, field_key, vec);
            }
        }

        // 3. Score each record sequentially
        let mut results = Vec::with_capacity(records.len());
        for (record, rec_id) in records.into_iter().zip(ids.iter()) {
            let resp = self.try_match_inner(side, record, rec_id, &query_fv);
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
    ///
    /// Each removal is processed sequentially. Records not found produce
    /// a "not_found" entry rather than failing the whole batch.
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
        let cm_len = self.state.crossmap.read().map(|c| c.len()).unwrap_or(0);
        HealthResponse {
            status: "ready".to_string(),
            model: self.state.config.embeddings.model.clone(),
            records_a: self.state.a.records.len(),
            records_b: self.state.b.records.len(),
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
