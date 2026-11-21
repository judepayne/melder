//! Unified scoring pipeline shared by batch and live modes.
//!
//! ## Pipeline stages
//!
//!   1. Blocking filter — done by the *caller* before invoking score_pool;
//!      `blocked_ids` is passed in as a pre-queried slice.
//!   2. Independent candidate generation — each method runs in parallel:
//!      - ANN search (O(log N) HNSW, `ann_candidates`)
//!      - BM25 search (Block-Max WAND via SimpleBm25, `bm25_candidates`)
//!      - Synonym lookup (inverted index)
//!   3. Union — deduplicate candidates by ID across all generators.
//!   4. Full scoring — score candidates on all match_fields, classify, sort,
//!      truncate to `top_n`.
//!
//! Each candidate generator runs independently against the blocked pool. No
//! method filters another method's candidates — the union captures candidates
//! found by ANY method. Full scoring is the single place where all signals
//! combine.
//!
//! Batch, live and enroll call `search()` with a directional plan and borrowed
//! pool and query inputs. The public `score_pool` wrappers use the same path.
//!
//! ## Why blocked_ids is passed in rather than BlockingIndex
//!
//! In live mode the blocking index is a `RwLock<BlockingIndex>`. Previously
//! `score_pool` held the read lock for its entire duration (~5-15ms at 100k
//! scale), starving opposite-side writes. The caller now takes the lock only
//! for the `.query()` call (~1µs) and passes the resulting owned `Vec<String>`
//! here. The pipeline is therefore lock-free for its entire execution.

use std::collections::HashMap;

use tracing::info_span;

use crate::config::{Config, MatchMethod};
use crate::error::StoreError;
use crate::matching::candidates;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::scoring;
use crate::scoring::embedding::dot_product_f32;
use crate::store::{RecordCollection, RecordStore, ScopedCollection};
use crate::vectordb::VectorDB;

/// The record being scored against the pool.
pub struct ScoringQuery<'a> {
    pub id: &'a str,
    pub record: &'a Record,
    pub side: Side,
    pub combined_vec: &'a [f32],
}

/// The pool (opposite side) and pre-computed candidate sets to score against.
pub struct ScoringPool<'a> {
    pub store: &'a dyn RecordStore,
    pub side: Side,
    pub combined_index: Option<&'a dyn VectorDB>,
    pub blocked_ids: &'a [String],
    /// Pre-computed BM25 candidate IDs (from caller).
    pub bm25_candidate_ids: &'a [String],
    /// Pre-computed BM25 scores keyed by pool record ID.
    pub bm25_scores_map: &'a HashMap<String, f64>,
    /// Pre-computed synonym candidate IDs (from caller).
    pub synonym_candidate_ids: &'a [String],
    /// Optional synonym dictionary for per-field synonym scoring.
    pub synonym_dictionary: Option<&'a crate::synonym::dictionary::SynonymDictionary>,
    /// Known non-matching pairs to skip during scoring.
    pub exclusions: &'a crate::matching::exclusions::Exclusions,
}

/// A prepared, borrowed query, independent of retrieval scratch.
pub(crate) struct PreparedQuery<'a> {
    pub id: &'a str,
    pub record: &'a Record,
    pub combined_vec: &'a [f32],
}

/// Stable side-bound resources; constructing a view does not allocate or lock.
pub(crate) struct SearchPool<'a> {
    pub collection: &'a dyn RecordCollection,
    pub combined_index: Option<&'a dyn VectorDB>,
    pub synonym_dictionary: Option<&'a crate::synonym::dictionary::SynonymDictionary>,
}

/// Per-invocation retrieval results and matching constraints.
pub(crate) struct SearchCandidates<'a> {
    pub eligible_ids: &'a [String],
    pub bm25_candidate_ids: &'a [String],
    pub bm25_scores_map: &'a HashMap<String, f64>,
    pub synonym_candidate_ids: &'a [String],
    pub exclusions: &'a crate::matching::exclusions::Exclusions,
    /// Explicit collection identity, not equality of literal record IDs.
    pub same_collection: bool,
}

/// Immutable directional settings reusable across queries.
pub(crate) struct SearchPlan {
    query_side: Side,
    scoring_fields: Vec<String>,
    bindings: Vec<BoundField>,
}

enum BindingNames {
    Single(String),
    Paired { a: String, b: String },
}

struct BoundField {
    names: BindingNames,
    method: MatchMethod,
    scorer: Option<crate::config::FuzzyScorer>,
    weight: f64,
}
pub(crate) trait SearchSettings {
    fn vector_backend(&self) -> &str;
    fn thresholds(&self) -> &crate::config::schema::ThresholdsConfig;
    fn output_record(&self, record: &Record) -> Record;
}
impl SearchSettings for Config {
    fn vector_backend(&self) -> &str {
        &self.vector_backend
    }
    fn thresholds(&self) -> &crate::config::schema::ThresholdsConfig {
        &self.thresholds
    }
    fn output_record(&self, record: &Record) -> Record {
        apply_output_mapping(record, self)
    }
}
impl SearchSettings for crate::config::enroll_schema::EnrollJobConfig {
    fn vector_backend(&self) -> &str {
        &self.vector_backend
    }
    fn thresholds(&self) -> &crate::config::schema::ThresholdsConfig {
        &self.thresholds
    }
    fn output_record(&self, record: &Record) -> Record {
        record.clone()
    }
}
impl SearchPlan {
    /// Borrow the query-side embedding layout without mirroring native names.
    pub(crate) fn embedding_fields(&self) -> impl Iterator<Item = (&str, f64)> + Clone {
        self.embedding_bindings().map(|(a, b, weight)| {
            (
                match self.query_side {
                    Side::A => a,
                    Side::B => b,
                },
                weight,
            )
        })
    }

    fn embedding_bindings(&self) -> impl Iterator<Item = (&str, &str, f64)> + Clone {
        self.bindings
            .iter()
            .filter(|f| f.method == MatchMethod::Embedding)
            .map(|f| {
                let (a, b) = match &f.names {
                    BindingNames::Single(name) => (name.as_str(), name.as_str()),
                    BindingNames::Paired { a, b } => (a.as_str(), b.as_str()),
                };
                (a, b, f.weight)
            })
    }
    /// Compile the projected fields and embedding layout for a direction.
    pub fn new(config: &Config, query_side: Side, pool_side: Side) -> Self {
        let scoring_fields = config
            .match_fields
            .iter()
            .filter(|mf| mf.method != MatchMethod::Bm25 && mf.method != MatchMethod::Embedding)
            .map(|mf| match pool_side {
                Side::A => mf.field_a.clone(),
                Side::B => mf.field_b.clone(),
            })
            .collect();
        Self {
            query_side,
            scoring_fields,
            bindings: config
                .match_fields
                .iter()
                .map(|f| BoundField {
                    names: BindingNames::Paired {
                        a: f.field_a.clone(),
                        b: f.field_b.clone(),
                    },
                    method: f.method,
                    scorer: f.scorer,
                    weight: f.weight,
                })
                .collect(),
        }
    }
    /// Compile native single-name enrollment bindings once for the session.
    pub fn enroll(config: &crate::config::enroll_schema::EnrollJobConfig) -> Self {
        Self {
            query_side: Side::A,
            scoring_fields: config
                .match_fields
                .iter()
                .filter(|f| !matches!(f.method, MatchMethod::Bm25 | MatchMethod::Embedding))
                .map(|f| f.field.clone())
                .collect(),
            bindings: config
                .match_fields
                .iter()
                .map(|f| BoundField {
                    names: BindingNames::Single(f.field.clone()),
                    method: f.method,
                    scorer: f.scorer,
                    weight: f.weight,
                })
                .collect(),
        }
    }
}

/// Error and verification policy for a search invocation.
pub(crate) enum SearchPolicy<'a> {
    Ordinary,
    Graph,
    Verified(&'a [String]),
}

/// Score a single query record against the opposite-side pool.
///
/// Runs independent candidate generators (ANN, BM25, synonym), unions their
/// results, then scores all candidates on every configured match_field.
/// Returns a sorted `Vec<MatchResult>` (descending by score, then ascending
/// by matched record ID for equal scores), truncated to `top_n` entries.
///
/// BM25 and synonym candidate generation is handled by the *caller* — this
/// function receives pre-computed candidate IDs and BM25 scores via
/// `ScoringPool`. ANN candidate generation still runs internally.
pub fn score_pool(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
) -> Vec<MatchResult> {
    score_pool_internal(
        query,
        pool,
        config,
        ann_candidates,
        top_n,
        SearchPolicy::Ordinary,
    )
    .expect("lenient scoring does not propagate store errors")
}

/// Score the ordinary candidate union, propagating store and vector input errors.
///
/// Intended for batch graph construction: preserves ordinary retrieval budgets,
/// ranking and missing-vector semantics without adding verification candidates.
pub fn score_pool_strict(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
) -> Result<Vec<MatchResult>, StoreError> {
    score_pool_internal(
        query,
        pool,
        config,
        ann_candidates,
        top_n,
        SearchPolicy::Graph,
    )
}

/// Score the ordinary candidate union plus caller-validated verification IDs.
///
/// The caller supplies at most two verification IDs, validates current vectors,
/// and provides their normalized BM25 scores in `pool.bm25_scores_map` (absent
/// hits mean zero). Verification IDs outside `blocked_ids` are ignored. All
/// candidates share exclusion, scoring, ranking, gap and truncation handling;
/// use `top_n = 0` to retain the entire union for verification.
///
/// Candidate record fetch errors are propagated instead of suppressed.
pub fn score_pool_verified(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
    verification_ids: &[String],
) -> Result<Vec<MatchResult>, StoreError> {
    score_pool_internal(
        query,
        pool,
        config,
        ann_candidates,
        top_n,
        SearchPolicy::Verified(verification_ids),
    )
}

#[allow(clippy::too_many_arguments)]
fn score_pool_internal(
    query: &ScoringQuery<'_>,
    pool: &ScoringPool<'_>,
    config: &Config,
    ann_candidates: usize,
    top_n: usize,
    policy: SearchPolicy<'_>,
) -> Result<Vec<MatchResult>, StoreError> {
    let plan = SearchPlan::new(config, query.side, pool.side);
    search(
        &PreparedQuery {
            id: query.id,
            record: query.record,
            combined_vec: query.combined_vec,
        },
        &SearchPool {
            collection: &ScopedCollection::new(pool.store, pool.side),
            combined_index: pool.combined_index,
            synonym_dictionary: pool.synonym_dictionary,
        },
        &plan,
        config,
        &SearchCandidates {
            eligible_ids: pool.blocked_ids,
            bm25_candidate_ids: pool.bm25_candidate_ids,
            bm25_scores_map: pool.bm25_scores_map,
            synonym_candidate_ids: pool.synonym_candidate_ids,
            exclusions: pool.exclusions,
            same_collection: query.side == pool.side,
        },
        ann_candidates,
        top_n,
        policy,
    )
}

/// Search borrowed views through the shared scoring pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    query: &PreparedQuery<'_>,
    pool: &SearchPool<'_>,
    plan: &SearchPlan,
    config: &dyn SearchSettings,
    inputs: &SearchCandidates<'_>,
    ann_candidates: usize,
    top_n: usize,
    policy: SearchPolicy<'_>,
) -> Result<Vec<MatchResult>, StoreError> {
    let (strict, strict_vectors, verification_ids) = match policy {
        SearchPolicy::Ordinary => (false, false, &[][..]),
        SearchPolicy::Graph => (true, true, &[][..]),
        SearchPolicy::Verified(ids) => (true, false, ids),
    };
    if inputs.eligible_ids.is_empty() {
        return Ok(Vec::new());
    }

    let has_embeddings = !query.combined_vec.is_empty();

    // Pool-side fields needed for scoring (used by get_many_fields to
    // avoid full JSON deserialization in SQLite).
    let scoring_fields = &plan.scoring_fields;

    // --- Stage 2: Independent candidate generation ---
    //
    // Each generator runs independently against the blocked pool. The union
    // of all candidate sets flows to full scoring.

    // ANN candidates (empty if no embedding fields configured).
    let ann_cands = {
        let _span = info_span!("ann_candidates").entered();
        candidates::select_candidates_checked(
            query.combined_vec,
            ann_candidates,
            pool.combined_index,
            inputs.eligible_ids,
            pool.collection,
            query.record,
            plan.query_side,
            config.vector_backend(),
            strict,
            strict_vectors,
        )?
    };

    // --- Stage 3: Union candidates ---
    //
    // Merge ANN, BM25, and synonym candidate sets, deduplicating by ID.
    // Candidates not in the ANN set need their records fetched from the store.
    let ann_id_set: std::collections::HashSet<&str> =
        ann_cands.iter().map(|c| c.id.as_str()).collect();

    // Collect non-ANN candidate IDs (BM25 + synonym + verification), deduplicated.
    // Only caller-validated verification IDs within the blocking pool may be added.
    let verified_ids = verification_ids
        .iter()
        .filter(|id| inputs.eligible_ids.contains(id));
    let mut extra_ids: Vec<String> = Vec::new();
    let mut extra_id_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for id in inputs
        .bm25_candidate_ids
        .iter()
        .chain(inputs.synonym_candidate_ids.iter())
        .chain(verified_ids)
    {
        if !ann_id_set.contains(id.as_str()) && extra_id_set.insert(id.clone()) {
            extra_ids.push(id.clone());
        }
    }

    // Fetch records for non-ANN candidates.
    let extra_cands: Vec<candidates::Candidate> = if extra_ids.is_empty() {
        Vec::new()
    } else {
        let fetched = if scoring_fields.is_empty() {
            pool.collection.get_many(&extra_ids)
        } else {
            pool.collection.get_many_fields(&extra_ids, scoring_fields)
        };
        let records = match fetched {
            Ok(records) => records,
            Err(e) if strict => return Err(e),
            Err(e) => {
                tracing::warn!(error = %e, "failed to fetch extra candidate records");
                Vec::new()
            }
        };
        records
            .into_iter()
            .map(|(id, record)| candidates::Candidate {
                id,
                record,
                combined_dot: 0.0,
            })
            .collect()
    };

    // Build the full union: ANN candidates + extra candidates (BM25/synonym).
    if ann_cands.is_empty() && extra_cands.is_empty() {
        return Ok(Vec::new());
    }

    // Self-match exclusion: when query_side == pool_side (enroll mode),
    // the query record may appear in the pool index. Filter it out before
    // scoring so a record never matches itself.
    // Collection identity is supplied by the search boundary.

    // --- Stage 4: Full scoring with per-field decomposition ---
    let _span = info_span!("full_scoring").entered();
    let has_emb_specs = has_embeddings && plan.embedding_bindings().next().is_some();

    let mut results: Vec<MatchResult> = Vec::with_capacity(ann_cands.len() + extra_cands.len());

    // Score all candidates through the same path.
    for cand in ann_cands.iter().chain(extra_cands.iter()) {
        // Skip self-match in enroll mode
        if inputs.same_collection && cand.id == query.id {
            continue;
        }
        // Skip known non-matching pairs
        if !inputs.exclusions.is_empty() && inputs.exclusions.contains(query.id, &cand.id) {
            continue;
        }
        results.push(score_candidate(
            cand,
            query.id,
            query.record,
            plan.query_side,
            query.combined_vec,
            pool.combined_index,
            config,
            plan,
            inputs.bm25_scores_map,
            has_emb_specs,
            pool.synonym_dictionary,
            strict_vectors,
        )?);
    }

    // Sort by score descending, breaking ties by record ID ascending.
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.matched_id.cmp(&b.matched_id))
    });

    // Assign rank (1-based, capped at u8::MAX) after sorting.
    if results.len() > 254 {
        tracing::warn!(
            candidates = results.len(),
            "more than 254 candidates — ranks above 255 will be capped"
        );
    }
    for (i, r) in results.iter_mut().enumerate() {
        r.rank = Some((i + 1).min(255) as u8);
    }

    // Apply score gap check before truncation so rank-2 is still present.
    if let Some(min_gap) = config.thresholds().min_score_gap {
        apply_score_gap_check(&mut results, min_gap);
    }

    // Truncate to top_n if specified.
    if top_n > 0 {
        results.truncate(top_n);
    }

    // Attach matched record (with output mapping) to the top result only.
    if let Some(top) = results.first_mut() {
        let matched = ann_cands
            .iter()
            .chain(extra_cands.iter())
            .find(|c| c.id == top.matched_id);
        if let Some(cand) = matched {
            top.matched_record = Some(config.output_record(&cand.record));
        }
    }

    Ok(results)
}

/// Score a single candidate through full per-field scoring.
///
/// Decomposes combined embedding vectors into per-field cosines, combines
/// with precomputed BM25 scores (if any), and runs all non-embedding,
/// non-BM25 scoring methods (exact, fuzzy, numeric).
#[allow(clippy::too_many_arguments)]
fn score_candidate(
    cand: &candidates::Candidate,
    query_id: &str,
    query_record: &Record,
    query_side: Side,
    query_combined_vec: &[f32],
    pool_combined_index: Option<&dyn VectorDB>,
    config: &dyn SearchSettings,
    plan: &SearchPlan,
    bm25_scores: &HashMap<String, f64>,
    has_emb_specs: bool,
    synonym_dictionary: Option<&crate::synonym::dictionary::SynonymDictionary>,
    strict_vectors: bool,
) -> Result<MatchResult, StoreError> {
    // Decompose combined vecs → per-field cosine similarities.
    let emb_scores: Option<HashMap<(&str, &str), f64>> = if has_emb_specs {
        let pool_vec = match pool_combined_index.map(|idx| idx.get(&cand.id)) {
            Some(Err(error)) if strict_vectors => return Err(error.into()),
            Some(Ok(vector)) => vector,
            _ => None,
        };
        pool_vec.map(|pool_vec| {
            decompose_emb_bindings(query_combined_vec, &pool_vec, plan.embedding_bindings())
        })
    } else {
        None
    };

    let precomputed_bm25 = bm25_scores.get(&cand.id).copied();

    let score_result = scoring::score_bound_pair(
        query_record,
        &cand.record,
        plan.bindings.iter().map(|f| {
            let (a, b) = match &f.names {
                BindingNames::Single(name) => (name.as_str(), name.as_str()),
                BindingNames::Paired { a, b } => (a.as_str(), b.as_str()),
            };
            let (query, candidate) = match plan.query_side {
                Side::A => (a, b),
                Side::B => (b, a),
            };
            scoring::FieldBinding {
                query,
                candidate,
                field_a: a,
                field_b: b,
                method: f.method,
                scorer: f.scorer,
                weight: f.weight,
            }
        }),
        emb_scores.as_ref(),
        precomputed_bm25,
        synonym_dictionary,
    );

    Ok(scoring::build_match_result(
        query_id,
        &cand.id,
        query_side,
        score_result,
        config.thresholds().auto_match,
        config.thresholds().review_floor,
        None,
        false,
    ))
}

/// Retrieve and normalize ordinary BM25 candidates once.
///
/// `None` selects native unblocked retrieval (including WAND); `Some(&[])`
/// is an empty eligible set, never an unrestricted search. Verification may
/// supply additional per-ID scores independently of this ordinary retrieval.
pub(crate) fn bm25_candidate_stage(
    index: &crate::bm25::simple::SimpleBm25,
    record: &Record,
    side: Side,
    eligible_ids: Option<&[String]>,
    limit: usize,
) -> (Vec<String>, HashMap<String, f64>) {
    let query_text = index.query_text_for(record, side);
    let self_score = index.analytical_self_score(&query_text);
    let raw = match eligible_ids {
        Some(ids) => index.score_blocked(&query_text, ids, limit),
        None => index.score_all(&query_text, limit),
    };
    let scored: Vec<_> = raw
        .into_iter()
        .map(|(id, score)| (id, crate::bm25::scorer::normalise_bm25(score, self_score)))
        .collect();
    let ids = scored.iter().map(|(id, _)| id.clone()).collect();
    (ids, scored.into_iter().collect())
}

/// Independent synonym candidate generation stage.
///
/// Looks up each synonym match field value in the synonym index and returns
/// the union of all matching record IDs that are not already in the ANN or
/// BM25 candidate sets.
pub fn synonym_candidate_stage(
    index: &crate::synonym::index::SynonymIndex,
    query_record: &Record,
    query_side: Side,
    synonym_fields: &[crate::config::SynonymFieldConfig],
) -> Vec<String> {
    synonym_candidates(
        index,
        query_record,
        synonym_fields.iter().map(|f| {
            (
                match query_side {
                    Side::A => f.field_a.as_str(),
                    Side::B => f.field_b.as_str(),
                },
                f.field_a.as_str(),
                f.field_b.as_str(),
                f.generators.first().map_or(3, |g| g.min_length),
            )
        }),
    )
}

pub(crate) fn synonym_candidates<'a>(
    index: &crate::synonym::index::SynonymIndex,
    query_record: &Record,
    fields: impl Iterator<Item = (&'a str, &'a str, &'a str, usize)>,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for (query_field, field_a, field_b, min_len) in fields {
        let query_value = query_record
            .get(query_field)
            .map(|s| s.as_str())
            .unwrap_or("");
        if query_value.trim().is_empty() {
            continue;
        }

        let candidates = index.lookup(query_value, field_a, field_b, min_len);
        for id in candidates {
            if seen.insert(id.clone()) {
                result.push(id);
            }
        }
    }

    result
}

/// Decompose two combined embedding vectors into per-field cosine similarities.
///
/// Given combined vectors built as `[√w₁·a₁, √w₂·a₂, …]` (dim = 384×N):
///   - Sub-slice `i` is `√wᵢ·aᵢ` (query) and `√wᵢ·bᵢ` (pool)
///   - `dot(√wᵢ·aᵢ, √wᵢ·bᵢ) = wᵢ · cosᵢ`
///   - Therefore `cosᵢ = dot(sub_query_i, sub_pool_i) / wᵢ`
///
/// Returns a `HashMap<"field_a/field_b", cosᵢ>` compatible with
/// `scoring::score_pair`'s `precomputed_emb_scores` argument.
#[cfg(test)]
fn decompose_emb_scores<'a>(
    query_combined: &[f32],
    pool_combined: &[f32],
    emb_specs: &'a [(String, String, f64)],
) -> HashMap<(&'a str, &'a str), f64> {
    decompose_emb_bindings(
        query_combined,
        pool_combined,
        emb_specs
            .iter()
            .map(|(a, b, w)| (a.as_str(), b.as_str(), *w)),
    )
}

fn decompose_emb_bindings<'a>(
    query_combined: &[f32],
    pool_combined: &[f32],
    emb_specs: impl Iterator<Item = (&'a str, &'a str, f64)> + Clone,
) -> HashMap<(&'a str, &'a str), f64> {
    let field_count = emb_specs.clone().count();
    if field_count == 0 || query_combined.is_empty() || pool_combined.is_empty() {
        return HashMap::new();
    }
    let field_dim = query_combined.len() / field_count;
    if field_dim == 0 {
        return HashMap::new();
    }
    debug_assert_eq!(
        query_combined.len() % field_count,
        0,
        "combined vector length {} not evenly divisible by {} fields",
        query_combined.len(),
        field_count
    );
    if !query_combined.len().is_multiple_of(field_count) {
        tracing::warn!(
            vec_len = query_combined.len(),
            fields = field_count,
            "combined vector not evenly divisible by field count — last field will be truncated"
        );
    }

    let mut scores = HashMap::with_capacity(field_count);

    for (i, (field_a, field_b, weight)) in emb_specs.enumerate() {
        let start = i * field_dim;
        let end = start + field_dim;

        if end > query_combined.len() || end > pool_combined.len() {
            tracing::warn!(
                field_a,
                field_b,
                end,
                query_len = query_combined.len(),
                pool_len = pool_combined.len(),
                "embedding decomposition truncated — field scored as 0.0"
            );
            break;
        }

        let q_sub = &query_combined[start..end];
        let p_sub = &pool_combined[start..end];

        let dot: f32 = dot_product_f32(q_sub, p_sub);

        let cos = if weight > 0.0 {
            (dot as f64 / weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        scores.insert((field_a, field_b), cos);
    }

    scores
}

/// Apply output_mapping to a matched record: rename fields per config.
/// Rename fields in a matched record per output_mapping config.
///
/// Each mapping renames `from` → `to`: the original key is removed
/// and the new key is inserted with the same value.
fn apply_output_mapping(record: &Record, config: &Config) -> Record {
    if config.output_mapping.is_empty() {
        return record.clone();
    }

    let mut result = record.clone();
    for mapping in &config.output_mapping {
        if let Some(val) = result.remove(&mapping.from) {
            result.insert(mapping.to.clone(), val);
        }
    }
    result
}

/// Downgrade the top result from Auto to Review when the score gap to rank-2
/// is below `min_gap`.
///
/// A high composite score with a close second candidate indicates model
/// uncertainty — the top match is more likely to be a false positive than
/// when the winning margin is large. Downgrading to Review surfaces these
/// borderline cases for human inspection rather than auto-confirming them.
///
/// Only applies when there are ≥ 2 results. Single-candidate results are
/// left unchanged: no rank-2 exists, so there is no ambiguity signal.
fn apply_score_gap_check(results: &mut [MatchResult], min_gap: f64) {
    if results.len() < 2 {
        return;
    }
    if results[0].classification != Classification::Auto {
        return;
    }
    let gap = results[0].score - results[1].score;
    if gap < min_gap {
        results[0].classification = Classification::Review;
        results[0].reason = Some("downgraded".into());
    }
}

/// Classify the top match result.
pub fn top_classification(results: &[MatchResult]) -> Classification {
    results
        .first()
        .map(|r| r.classification)
        .unwrap_or(Classification::NoMatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingConfig;
    use crate::store::memory::MemoryStore;
    use crate::vectordb::flat::FlatVectorDB;

    fn make_match_result(score: f64, classification: Classification) -> MatchResult {
        MatchResult {
            query_id: "q".to_string(),
            matched_id: "m".to_string(),
            query_side: Side::B,
            score,
            field_scores: vec![],
            classification,
            matched_record: None,
            from_crossmap: false,
            rank: None,
            reason: None,
        }
    }

    // --- apply_score_gap_check ---

    #[test]
    fn score_gap_check_downgrades_auto_when_gap_too_small() {
        let mut results = vec![
            make_match_result(0.90, Classification::Auto),
            make_match_result(0.87, Classification::Auto),
        ];
        apply_score_gap_check(&mut results, 0.10);
        // gap = 0.03 < 0.10 → top result downgraded
        assert_eq!(
            results[0].classification,
            Classification::Review,
            "expected Review when gap < min_gap"
        );
        assert_eq!(
            results[1].classification,
            Classification::Auto,
            "rank-2 must be untouched"
        );
    }

    #[test]
    fn score_gap_check_preserves_auto_when_gap_sufficient() {
        let mut results = vec![
            make_match_result(0.95, Classification::Auto),
            make_match_result(0.80, Classification::Auto),
        ];
        apply_score_gap_check(&mut results, 0.10);
        // gap = 0.15 >= 0.10 → no change
        assert_eq!(
            results[0].classification,
            Classification::Auto,
            "expected Auto when gap >= min_gap"
        );
    }

    #[test]
    fn score_gap_check_noop_with_single_result() {
        let mut results = vec![make_match_result(0.90, Classification::Auto)];
        apply_score_gap_check(&mut results, 0.10);
        // only one result — no rank-2 to compare against
        assert_eq!(results[0].classification, Classification::Auto);
    }

    #[test]
    fn score_gap_check_noop_when_top_is_not_auto() {
        let mut results = vec![
            make_match_result(0.75, Classification::Review),
            make_match_result(0.74, Classification::Review),
        ];
        apply_score_gap_check(&mut results, 0.10);
        // top is Review, not Auto — nothing to downgrade
        assert_eq!(results[0].classification, Classification::Review);
    }

    #[test]
    fn score_gap_check_empty_results_is_noop() {
        let mut results: Vec<MatchResult> = vec![];
        apply_score_gap_check(&mut results, 0.10); // must not panic
    }

    #[test]
    fn decompose_emb_scores_single_field() {
        let w = 1.0_f64;
        let dim = 4usize;

        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];

        let q = a.clone();
        let p = b.clone();

        let specs = vec![("f_a".to_string(), "f_b".to_string(), w)];
        let scores = decompose_emb_scores(&q, &p, &specs);

        let cos = *scores.get(&("f_a", "f_b")).unwrap();
        assert!(cos.abs() < 0.001, "expected ~0.0, got {}", cos);
        let _ = dim;
    }

    #[test]
    fn decompose_emb_scores_self_match() {
        let w = 0.55_f64;
        let sqrt_w = w.sqrt() as f32;

        let scaled = vec![sqrt_w, 0.0, 0.0, 0.0];
        let specs = vec![("f_a".to_string(), "f_b".to_string(), w)];
        let scores = decompose_emb_scores(&scaled, &scaled, &specs);

        let cos = *scores.get(&("f_a", "f_b")).unwrap();
        assert!((cos - 1.0).abs() < 0.001, "expected ~1.0, got {}", cos);
    }

    #[test]
    fn decompose_emb_scores_two_fields_recovers_both() {
        let w1 = 0.55_f64;
        let w2 = 0.20_f64;
        let sqrt_w1 = w1.sqrt() as f32;
        let sqrt_w2 = w2.sqrt() as f32;

        let f1_a = vec![sqrt_w1, 0.0];
        let f1_b = vec![sqrt_w1, 0.0];

        let f2_a = vec![0.0, sqrt_w2];
        let f2_b = vec![sqrt_w2, 0.0];

        let mut q = f1_a.clone();
        q.extend_from_slice(&f2_a);
        let mut p = f1_b.clone();
        p.extend_from_slice(&f2_b);

        let specs = vec![
            ("f1_a".to_string(), "f1_b".to_string(), w1),
            ("f2_a".to_string(), "f2_b".to_string(), w2),
        ];
        let scores = decompose_emb_scores(&q, &p, &specs);

        let cos1 = *scores.get(&("f1_a", "f1_b")).unwrap();
        let cos2 = *scores.get(&("f2_a", "f2_b")).unwrap();

        assert!(
            (cos1 - 1.0).abs() < 0.001,
            "cos1 expected ~1.0, got {}",
            cos1
        );
        assert!(cos2.abs() < 0.001, "cos2 expected ~0.0, got {}", cos2);
    }

    // --- score_pool union tests ---

    fn make_record(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_store() -> MemoryStore {
        MemoryStore::new(&BlockingConfig::default())
    }

    fn make_config_exact(field_a: &str, field_b: &str) -> Config {
        let yaml = format!(
            r#"
job:
  name: test
datasets:
  a: {{ path: x.csv, id_field: id, format: csv }}
  b: {{ path: x.csv, id_field: id, format: csv }}
cross_map:
  path: /tmp/test/crossmap.csv
  a_id_field: id
  b_id_field: id
embeddings:
  model: test
  a_cache_dir: /tmp/test
match_fields:
  - field_a: {field_a}
    field_b: {field_b}
    method: exact
    weight: 1.0
thresholds:
  auto_match: 0.85
  review_floor: 0.60
output:
  csv_dir_path: /tmp/test/output
vector_backend: flat
"#
        );
        // Use load pipeline: parse → defaults → validate → derive.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        crate::config::load_config(tmp.path()).unwrap()
    }

    fn make_enroll_config_exact(field: &str) -> crate::config::enroll_schema::EnrollJobConfig {
        let yaml = format!(
            "job: {{name: test}}\nembeddings: {{model: test, cache_dir: cache}}\nvector_backend: flat\nmatch_fields: [{{field: {field}, method: exact, weight: 1}}]\nthresholds: {{auto_match: 0.85, review_floor: 0.6}}\n"
        );
        let file = tempfile::NamedTempFile::new().expect("temporary config");
        std::fs::write(file.path(), yaml).expect("write enrollment config");
        crate::config::load_enroll_job_config(file.path()).expect("native enrollment config")
    }

    #[test]
    fn native_embedding_bindings_preserve_unequal_weights() {
        let mut config = make_enroll_config_exact("first");
        config.match_fields = [("first", 0.25), ("second", 0.75)]
            .into_iter()
            .map(
                |(field, weight)| crate::config::enroll_schema::EnrollMatchField {
                    field: field.into(),
                    method: MatchMethod::Embedding,
                    scorer: None,
                    weight,
                    fields: None,
                },
            )
            .collect();
        let plan = SearchPlan::enroll(&config);
        assert_eq!(
            plan.embedding_fields().collect::<Vec<_>>(),
            vec![("first", 0.25), ("second", 0.75)],
            "native layout preserves order and weights"
        );
        let q = [0.5, 0.0, 0.75_f32.sqrt(), 0.0];
        let p = [0.5, 0.0, 0.0, 0.75_f32.sqrt()];
        let scores = decompose_emb_bindings(&q, &p, plan.embedding_bindings());
        assert_eq!(scores[&("first", "first")], 1.0, "first weight is removed");
        assert_eq!(
            scores[&("second", "second")],
            0.0,
            "orthogonal second field stays zero"
        );
    }

    #[test]
    fn native_enroll_bindings_score_single_named_fields() {
        let config = make_enroll_config_exact("name");
        let plan = SearchPlan::enroll(&config);
        let record = make_record(&[("name", "Alice")]);
        let candidate = candidates::Candidate {
            id: "other".into(),
            record: record.clone(),
            combined_dot: 0.0,
        };
        let result = score_candidate(
            &candidate,
            "query",
            &record,
            Side::A,
            &[],
            None,
            &config,
            &plan,
            &HashMap::new(),
            false,
            None,
            false,
        )
        .expect("native exact scoring succeeds");
        assert_eq!(
            result.score, 1.0,
            "single-name bindings read the same field"
        );
        assert_eq!(result.field_scores[0].field_a, "name");
        assert_eq!(result.field_scores[0].field_b, "name");
    }

    fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as i32 as f32) / (i32::MAX as f32)
            })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[derive(Debug)]
    struct FaultVectorDB {
        inner: FlatVectorDB,
        gets: std::sync::atomic::AtomicUsize,
        fail_get_at: usize,
        fail_search: bool,
    }

    impl VectorDB for FaultVectorDB {
        fn upsert(
            &self,
            id: &str,
            vec: &[f32],
            record: &Record,
            side: Side,
        ) -> Result<(), crate::vectordb::VectorDBError> {
            self.inner.upsert(id, vec, record, side)
        }
        fn remove(&self, id: &str) -> Result<bool, crate::vectordb::VectorDBError> {
            self.inner.remove(id)
        }
        fn search(
            &self,
            query: &[f32],
            k: usize,
            record: &Record,
            side: Side,
        ) -> Result<Vec<crate::vectordb::SearchResult>, crate::vectordb::VectorDBError> {
            if self.fail_search {
                return Err(crate::vectordb::VectorDBError::Backend(
                    "injected ANN search failure".into(),
                ));
            }
            self.inner.search(query, k, record, side)
        }
        fn search_filtered(
            &self,
            query: &[f32],
            k: usize,
            allowed: &std::collections::HashSet<String>,
            record: &Record,
            side: Side,
        ) -> Result<Vec<crate::vectordb::SearchResult>, crate::vectordb::VectorDBError> {
            self.inner.search_filtered(query, k, allowed, record, side)
        }
        fn get(&self, id: &str) -> Result<Option<Vec<f32>>, crate::vectordb::VectorDBError> {
            let call = self.gets.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if call >= self.fail_get_at {
                return Err(crate::vectordb::VectorDBError::Backend(
                    "injected A vector fetch failure".into(),
                ));
            }
            self.inner.get(id)
        }
        fn contains(&self, id: &str) -> bool {
            self.inner.contains(id)
        }
        fn len(&self) -> usize {
            self.inner.len()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn save(&self, path: &std::path::Path) -> Result<(), crate::vectordb::VectorDBError> {
            self.inner.save(path)
        }
        fn is_stale(
            path: &std::path::Path,
            expected_count: usize,
        ) -> Result<bool, crate::vectordb::VectorDBError> {
            FlatVectorDB::is_stale(path, expected_count)
        }
        fn stored_text_hashes(&self) -> HashMap<String, u64> {
            self.inner.stored_text_hashes()
        }
        fn text_hash_for(&self, id: &str) -> Option<u64> {
            self.inner.text_hash_for(id)
        }
    }

    #[test]
    fn strict_propagates_a_vector_errors_without_changing_lenient_scoring() {
        let store = make_store();
        let record = make_record(&[("name", "Foo")]);
        store
            .insert(Side::A, "a1", &record)
            .expect("insert A record");
        let ids = vec!["a1".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let mut config = make_config_exact("name", "name");
        config.match_fields[0].method = MatchMethod::Embedding;
        let query = ScoringQuery {
            id: "b1",
            record: &record,
            side: Side::B,
            combined_vec: &[1.0, 0.0],
        };
        // First fetch: flat retrieval. Second fetch: full-scoring decomposition.
        // A synonym candidate also exercises decomposition outside ANN retrieval.
        for (backend, fail_get_at, fail_search, via_synonym) in [
            ("flat", 1, false, false),
            ("flat", 2, false, false),
            ("flat", 1, false, true),
            #[cfg(feature = "usearch")]
            ("usearch", 1, false, false),
            #[cfg(feature = "usearch")]
            ("usearch", usize::MAX, true, false),
        ] {
            config.vector_backend = backend.into();
            let index = FaultVectorDB {
                inner: FlatVectorDB::new(2),
                gets: std::sync::atomic::AtomicUsize::new(0),
                fail_get_at,
                fail_search,
            };
            index
                .upsert("a1", query.combined_vec, &record, Side::A)
                .expect("insert A vector");
            // A nonexistent blocked ID avoids ANN fetches, isolating the
            // synonym candidate's full-scoring vector fetch.
            let blocked = if via_synonym {
                vec!["absent".into()]
            } else {
                ids.clone()
            };
            let pool = ScoringPool {
                store: &store,
                side: Side::A,
                combined_index: Some(&index),
                blocked_ids: &blocked,
                bm25_candidate_ids: &[],
                bm25_scores_map: &scores,
                synonym_candidate_ids: if via_synonym { &ids } else { &[] },
                synonym_dictionary: None,
                exclusions: &exclusions,
            };
            assert!(
                matches!(
                    score_pool_strict(&query, &pool, &config, 1, 0),
                    Err(StoreError::Vector(_))
                ),
                "strict must reject {backend}, fetch={fail_get_at}, search={fail_search}, synonym={via_synonym}"
            );
            index.gets.store(0, std::sync::atomic::Ordering::Relaxed);
            assert!(
                !score_pool(&query, &pool, &config, 1, 0).is_empty(),
                "ordinary scoring remains lenient"
            );
            index.gets.store(0, std::sync::atomic::Ordering::Relaxed);
            assert!(
                score_pool_verified(&query, &pool, &config, 1, 0, &[]).is_ok(),
                "live verification keeps vector error semantics"
            );
        }
    }

    #[test]
    fn strict_preserves_missing_vector_semantics_and_retrieval_budget() {
        let store = make_store();
        let record = make_record(&[("name", "Foo")]);
        for id in ["a1", "a2", "missing"] {
            store.insert(Side::A, id, &record).expect("insert A record");
        }
        let ids = vec!["a1".into(), "a2".into(), "missing".into()];
        let index = FlatVectorDB::new(2);
        index
            .upsert("a1", &[1.0, 0.0], &record, Side::A)
            .expect("insert A vector");
        index
            .upsert("a2", &[0.8, 0.6], &record, Side::A)
            .expect("insert A vector");
        let mut config = make_config_exact("name", "name");
        config.match_fields[0].method = MatchMethod::Embedding;
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "b1",
            record: &record,
            side: Side::B,
            combined_vec: &[1.0, 0.0],
        };
        let pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&index),
            blocked_ids: &ids,
            bm25_candidate_ids: &[],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        for (ann_budget, top_n, expected) in [(1, 0, 1), (0, 0, 3), (0, 1, 1)] {
            let strict = score_pool_strict(&query, &pool, &config, ann_budget, top_n)
                .expect("missing vector is not an input error");
            let ordinary = score_pool(&query, &pool, &config, ann_budget, top_n);
            assert_eq!(
                strict.len(),
                expected,
                "retrieval and output budgets are preserved"
            );
            assert_eq!(
                format!("{strict:?}"),
                format!("{ordinary:?}"),
                "strict must preserve normal scoring including absent vectors"
            );
        }
    }

    #[test]
    fn search_collection_identity_and_asymmetric_directions() {
        let store = make_store();
        let config = make_config_exact("left_name", "right_name");
        let left = make_record(&[("left_name", "Acme")]);
        let right = make_record(&[("right_name", "Acme")]);
        store.insert(Side::A, "same", &left).expect("insert A");
        store.insert(Side::B, "same", &right).expect("insert B");
        let ids = vec!["same".to_string()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        for (side, record) in [(Side::A, &left), (Side::B, &right)] {
            let pool = SearchPool {
                collection: &ScopedCollection::new(&store, side.opposite()),
                combined_index: None,
                synonym_dictionary: None,
            };
            let plan = SearchPlan::new(&config, side, side.opposite());
            let query = PreparedQuery {
                id: "same",
                record,
                combined_vec: &[],
            };
            let mut inputs = SearchCandidates {
                eligible_ids: &ids,
                bm25_candidate_ids: &ids,
                bm25_scores_map: &scores,
                synonym_candidate_ids: &[],
                exclusions: &exclusions,
                same_collection: false,
            };
            let results = search(
                &query,
                &pool,
                &plan,
                &config,
                &inputs,
                50,
                0,
                SearchPolicy::Graph,
            )
            .expect("directional search");
            assert_eq!(
                results.len(),
                1,
                "equal IDs across collections remain eligible"
            );
            assert_eq!(
                results[0].score, 1.0,
                "asymmetric fields bind in both directions"
            );
            for policy in [SearchPolicy::Ordinary, SearchPolicy::Verified(&ids)] {
                let alternate = search(&query, &pool, &plan, &config, &inputs, 50, 0, policy)
                    .expect("valid search policy");
                assert_eq!(
                    format!("{alternate:?}"),
                    format!("{results:?}"),
                    "all policies preserve directional scoring on valid inputs"
                );
            }
            inputs.same_collection = true;
            assert!(
                search(
                    &query,
                    &pool,
                    &plan,
                    &config,
                    &inputs,
                    50,
                    0,
                    SearchPolicy::Graph,
                )
                .expect("self filter")
                .is_empty(),
                "collection identity explicitly excludes self"
            );
        }
    }

    #[test]
    fn score_pool_empty_blocked_ids_returns_empty() {
        let store = make_store();
        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Foo")]);
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &[],
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 5);
        assert!(results.is_empty(), "empty blocked_ids → no results");
    }

    #[test]
    fn score_pool_no_embeddings_no_bm25_returns_empty() {
        let store = make_store();
        store
            .insert(
                Side::A,
                "a1",
                &make_record(&[("id", "a1"), ("name", "Foo Corp")]),
            )
            .unwrap();

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Foo Corp")]);
        let blocked = vec!["a1".to_string()];
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        // No embeddings (empty vec), no BM25 candidates → both generators
        // return empty → union is empty → no results.
        let results = score_pool(&q, &p, &config, 50, 5);
        assert!(
            results.is_empty(),
            "no candidate generators configured → no results"
        );
    }

    #[test]
    fn score_pool_ann_only_scores_candidates() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Insert two A records with identical names to the query.
        let a1 = make_record(&[("id", "a1"), ("name", "Foo Corp")]);
        let a2 = make_record(&[("id", "a2"), ("name", "Bar Inc")]);
        store.insert(Side::A, "a1", &a1).unwrap();
        store.insert(Side::A, "a2", &a2).unwrap();

        let v1 = unit_vec(dim, 1);
        let v2 = unit_vec(dim, 2);
        idx.upsert("a1", &v1, &a1, Side::A).unwrap();
        idx.upsert("a2", &v2, &a2, Side::A).unwrap();

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Foo Corp")]);
        let blocked = vec!["a1".to_string(), "a2".to_string()];
        let query_vec = unit_vec(dim, 1); // identical to a1
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &query_vec,
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx as &dyn VectorDB),
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 5);
        assert!(
            !results.is_empty(),
            "ANN should surface candidates for scoring"
        );

        // a1 has exact name match → score 1.0. a2 doesn't match → score 0.0.
        let top = &results[0];
        assert_eq!(top.matched_id, "a1", "exact name match should rank first");
        assert!(
            (top.score - 1.0).abs() < 0.01,
            "exact match score should be ~1.0, got {}",
            top.score
        );
    }

    #[test]
    fn score_pool_results_sorted_descending() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Three A records: one exact match, one partial, one no match.
        let a1 = make_record(&[("id", "a1"), ("name", "Acme Corp")]);
        let a2 = make_record(&[("id", "a2"), ("name", "Other Ltd")]);
        let a3 = make_record(&[("id", "a3"), ("name", "Acme Corp")]);
        store.insert(Side::A, "a1", &a1).unwrap();
        store.insert(Side::A, "a2", &a2).unwrap();
        store.insert(Side::A, "a3", &a3).unwrap();

        for (id, seed, rec) in [("a1", 1u64, &a1), ("a2", 2, &a2), ("a3", 3, &a3)] {
            idx.upsert(id, &unit_vec(dim, seed), rec, Side::A).unwrap();
        }

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Acme Corp")]);
        let blocked = vec!["a1".to_string(), "a2".to_string(), "a3".to_string()];
        let query_vec = unit_vec(dim, 99);
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &query_vec,
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx as &dyn VectorDB),
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 5);

        // Results must be sorted descending by score.
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results not sorted: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn score_pool_equal_scores_use_id_order_before_truncation() {
        let store = make_store();
        for (id, name) in [
            ("ENT-000025", "Gibson Group"),
            ("ENR-0000094", "Gibson Group"),
            ("a-no-match", "Other Company"),
        ] {
            store
                .insert(Side::A, id, &make_record(&[("name", name)]))
                .expect("test records should insert");
        }
        let config = make_config_exact("name", "name");
        let record = make_record(&[("name", "Gibson Group")]);
        let bm25_scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let expected = ["ENR-0000094", "ENT-000025", "a-no-match"];

        // Cover both enroll (same side) and batch/live (opposite side),
        // with every input permutation and cutoffs inside/outside the tie.
        for side in [Side::A, Side::B] {
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let ids: Vec<String> = order.iter().map(|&i| expected[i].to_string()).collect();
                let query = ScoringQuery {
                    id: "query",
                    record: &record,
                    side,
                    combined_vec: &[],
                };
                let pool = ScoringPool {
                    store: &store,
                    side: Side::A,
                    combined_index: None,
                    blocked_ids: &ids,
                    bm25_candidate_ids: &ids,
                    bm25_scores_map: &bm25_scores,
                    synonym_candidate_ids: &[],
                    synonym_dictionary: None,
                    exclusions: &exclusions,
                };
                for top_n in [0, 1, 2, 3] {
                    let results = score_pool(&query, &pool, &config, 50, top_n);
                    let plan = SearchPlan::new(&config, side, Side::A);
                    let prepared = PreparedQuery {
                        id: query.id,
                        record: query.record,
                        combined_vec: query.combined_vec,
                    };
                    let resources = SearchPool {
                        collection: &ScopedCollection::new(&store, Side::A),
                        combined_index: None,
                        synonym_dictionary: None,
                    };
                    let inputs = SearchCandidates {
                        eligible_ids: &ids,
                        bm25_candidate_ids: &ids,
                        bm25_scores_map: &bm25_scores,
                        synonym_candidate_ids: &[],
                        exclusions: &exclusions,
                        same_collection: side == Side::A,
                    };
                    let searched = search(
                        &prepared,
                        &resources,
                        &plan,
                        &config,
                        &inputs,
                        50,
                        top_n,
                        SearchPolicy::Ordinary,
                    )
                    .expect("borrowed search should succeed");
                    assert_eq!(
                        format!("{searched:?}"),
                        format!("{results:?}"),
                        "borrowed search and compatibility entry point must agree",
                    );
                    let count = if top_n == 0 { 3 } else { top_n };
                    let actual: Vec<&str> = results
                        .iter()
                        .map(|result| result.matched_id.as_str())
                        .collect();
                    assert_eq!(
                        actual,
                        expected[..count],
                        "score then ID ordering must ignore input order: side={side:?}, order={order:?}, top_n={top_n}",
                    );
                    for (i, result) in results.iter().enumerate() {
                        assert_eq!(result.rank, Some((i + 1) as u8), "rank follows tie order");
                        assert_eq!(
                            result.score,
                            if i < 2 { 1.0 } else { 0.0 },
                            "ID tie-breaking must not change scores or outrank higher scores",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn verified_union_is_deduplicated_before_gap_and_truncation() {
        let store = make_store();
        let idx = FlatVectorDB::new(2);
        let record = make_record(&[("name", "Acme")]);
        for id in ["a1", "a2"] {
            store
                .insert(Side::A, id, &record)
                .expect("test record should insert");
        }
        idx.upsert("a1", &[1.0, 0.0], &record, Side::A)
            .expect("test vector should insert");
        idx.upsert("a2", &[0.0, 1.0], &record, Side::A)
            .expect("test vector should insert");
        let mut config = make_config_exact("name", "name");
        config.vector_backend = "flat".into();
        config.thresholds.min_score_gap = Some(0.1);
        let ids = vec!["a1".into(), "a2".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::B,
            combined_vec: &[1.0, 0.0],
        };
        let pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx),
            blocked_ids: &ids,
            bm25_candidate_ids: &ids[..1],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &ids[..1],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        let ordinary = score_pool(&query, &pool, &config, 1, 0);
        assert_eq!(ordinary.len(), 1, "retrieval limit must stay unchanged");
        assert_eq!(ordinary[0].classification, Classification::Auto);
        let empty_verified = score_pool_verified(&query, &pool, &config, 1, 0, &[])
            .expect("empty verification should score");
        assert_eq!(empty_verified[0].score, ordinary[0].score);
        assert_eq!(empty_verified[0].classification, ordinary[0].classification);

        for verification in [ids.clone(), vec!["a2".into(), "a2".into()]] {
            for top_n in [0, 1] {
                let results = score_pool_verified(&query, &pool, &config, 1, top_n, &verification)
                    .expect("verification should score");
                assert_eq!(results.len(), if top_n == 0 { 2 } else { 1 });
                assert_eq!(results[0].matched_id, "a1", "ties use ID order");
                assert_eq!(results[0].rank, Some(1));
                assert_eq!(
                    results[0].classification,
                    Classification::Review,
                    "verification runner-up must affect gap even with top_n=1",
                );
                assert_eq!(results[0].reason.as_deref(), Some("downgraded"));
                if top_n == 0 {
                    assert_eq!(results[1].matched_id, "a2");
                    assert_eq!(results[1].rank, Some(2));
                }
            }
        }
    }

    #[test]
    fn verified_ids_respect_blocking_exclusions_and_self_filter() {
        let store = make_store();
        let record = make_record(&[("name", "Acme")]);
        for id in ["query", "excluded", "outside", "allowed"] {
            store
                .insert(Side::A, id, &record)
                .expect("test record should insert");
        }
        let config = make_config_exact("name", "name");
        let ids = vec!["query".into(), "excluded".into(), "allowed".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        exclusions.add("query", "excluded");
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::A,
            combined_vec: &[],
        };
        let mut pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &ids,
            bm25_candidate_ids: &[],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        for verification in [
            vec!["query".into(), "excluded".into()],
            vec!["outside".into(), "outside".into()],
        ] {
            let results = score_pool_verified(&query, &pool, &config, 1, 0, &verification)
                .expect("filtered verification should succeed");
            assert!(results.is_empty(), "ineligible IDs must never be scored");
        }
        let verification = vec!["allowed".into()];
        assert_eq!(
            score_pool_verified(&query, &pool, &config, 1, 0, &verification)
                .expect("eligible verification should score")
                .len(),
            1,
            "verification can add an otherwise unretrieved candidate",
        );
        pool.blocked_ids = &[];
        assert!(
            score_pool_verified(&query, &pool, &config, 1, 0, &verification)
                .expect("empty block should succeed")
                .is_empty(),
            "verification cannot bypass an empty block",
        );
    }

    #[test]
    fn verified_bm25_uses_supplied_scores_and_missing_hits_are_zero() {
        let store = make_store();
        let record = Record::new();
        let ids = vec!["scored".to_owned(), "zero".to_owned()];
        for id in &ids {
            store
                .insert(Side::A, id, &record)
                .expect("test record should insert");
        }
        let mut config = make_config_exact("name", "name");
        config.match_fields[0].method = MatchMethod::Bm25;
        let scores = HashMap::from([("scored".into(), 0.93)]);
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        let pool = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: None,
            blocked_ids: &ids,
            bm25_candidate_ids: &[],
            bm25_scores_map: &scores,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &exclusions,
        };
        let results = score_pool_verified(&query, &pool, &config, 1, 0, &ids)
            .expect("BM25 verification should score");
        assert_eq!(results.len(), 2, "zero BM25 hit must not omit the pair");
        assert_eq!(results[0].matched_id, "scored");
        assert!(
            (results[0].score - 0.93).abs() < 1e-9,
            "use normalized BM25"
        );
        assert_eq!(results[1].score, 0.0, "absent hit means zero");
    }

    #[test]
    fn verified_propagates_extra_fetch_errors_but_greedy_stays_lenient() {
        let dir = tempfile::tempdir().expect("test directory should create");
        let columns = vec!["name".into()];
        let (store, _, writer) = crate::store::sqlite::open_sqlite(
            &dir.path().join("records.db"),
            &BlockingConfig::default(),
            None,
            &columns,
            &columns,
        )
        .expect("test store should open");
        writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .execute_batch("DROP TABLE a_records")
            .expect("test table should drop to inject read failure");
        let record = Record::new();
        let ids = vec!["unreadable".into()];
        let scores = HashMap::new();
        let exclusions = crate::matching::exclusions::Exclusions::new();
        let query = ScoringQuery {
            id: "query",
            record: &record,
            side: Side::B,
            combined_vec: &[],
        };
        // Cover both full-record and projected-field extra fetches, and
        // failures from every extra-candidate source, even with no verification.
        for method in [MatchMethod::Exact, MatchMethod::Bm25] {
            let mut config = make_config_exact("name", "name");
            config.match_fields[0].method = method;
            for source in 0..3 {
                let pool = ScoringPool {
                    store: &store,
                    side: Side::A,
                    combined_index: None,
                    blocked_ids: &ids,
                    bm25_candidate_ids: if source == 0 { &ids } else { &[] },
                    bm25_scores_map: &scores,
                    synonym_candidate_ids: if source == 1 { &ids } else { &[] },
                    synonym_dictionary: None,
                    exclusions: &exclusions,
                };
                let verification = if source == 2 { ids.as_slice() } else { &[] };
                if source != 2 {
                    assert!(
                        score_pool_strict(&query, &pool, &config, 1, 0).is_err(),
                        "graph strict propagates store errors: source={source}"
                    );
                }
                assert!(
                    score_pool_verified(&query, &pool, &config, 1, 0, verification).is_err(),
                    "strict extra fetch failure must propagate: source={source}",
                );
                assert!(
                    score_pool(&query, &pool, &config, 1, 0).is_empty(),
                    "greedy fetch failure must retain its lenient behavior",
                );
            }
        }
    }

    #[test]
    fn score_pool_top_n_truncates() {
        let dim = 4usize;
        let store = make_store();
        let idx = FlatVectorDB::new(dim);

        // Insert 10 A records.
        let mut blocked = Vec::new();
        for i in 0..10 {
            let id = format!("a{}", i);
            let rec = make_record(&[("id", &id), ("name", &format!("Corp {}", i))]);
            store.insert(Side::A, &id, &rec).unwrap();
            idx.upsert(&id, &unit_vec(dim, i as u64), &rec, Side::A)
                .unwrap();
            blocked.push(id);
        }

        let config = make_config_exact("name", "name");
        let record = make_record(&[("id", "q1"), ("name", "Corp 0")]);
        let query_vec = unit_vec(dim, 99);
        let empty_bm25: HashMap<String, f64> = HashMap::new();
        let no_exclusions = crate::matching::exclusions::Exclusions::new();

        let q = ScoringQuery {
            id: "q1",
            record: &record,
            side: Side::B,
            combined_vec: &query_vec,
        };
        let p = ScoringPool {
            store: &store,
            side: Side::A,
            combined_index: Some(&idx as &dyn VectorDB),
            blocked_ids: &blocked,
            bm25_candidate_ids: &[],
            bm25_scores_map: &empty_bm25,
            synonym_candidate_ids: &[],
            synonym_dictionary: None,
            exclusions: &no_exclusions,
        };

        let results = score_pool(&q, &p, &config, 50, 3);
        assert!(
            results.len() <= 3,
            "top_n=3 should truncate, got {}",
            results.len()
        );
    }
}
