//! Tantivy-backed BM25 index for candidate retrieval and pair scoring.
//!
//! Each side (A / B) gets its own `BM25Index`. The index concatenates all
//! text fields designated by `bm25_fields` into a single `content` field
//! per document, enabling cross-field BM25 scoring with zero configuration.

use std::collections::HashMap;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, doc};
use tracing::warn;

use crate::config::BlockingFieldPair;
use crate::config::schema::Bm25FieldPair;
use crate::models::{Record, Side};
use crate::store::RecordStore;

/// In-memory BM25 index wrapping Tantivy.
pub struct BM25Index {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    id_field: Field,
    content_field: Field,
    /// Tantivy fields for blocking key columns (one per blocking field pair).
    /// Empty if blocking is not configured.
    block_fields: Vec<Field>,
    /// Which text fields to concatenate per record. The side determines
    /// whether `field_a` or `field_b` is used for lookup.
    fields: Vec<Bm25FieldPair>,
    /// Blocking field pairs — stored for extracting blocking values from
    /// records at upsert time.
    blocking_fields: Vec<BlockingFieldPair>,
    side: Side,
    /// Cached self-scores keyed by hash of query text. Self-score depends
    /// on corpus IDF which changes slowly, so caching is safe — the cache
    /// is cleared on every commit to stay fresh.
    self_score_cache: HashMap<u64, f32>,
    /// True when documents have been added/deleted but not yet committed.
    /// Cleared by `commit_if_dirty()`.
    dirty: bool,
}

impl BM25Index {
    /// Build a BM25 index from a record store.
    ///
    /// `fields` specifies which text fields to index. For each record, the
    /// values of these fields are concatenated (space-separated) into a
    /// single indexed document.
    pub fn build(
        store: &dyn RecordStore,
        side: Side,
        fields: &[Bm25FieldPair],
        blocking_fields: &[BlockingFieldPair],
    ) -> Result<Self, anyhow::Error> {
        let (schema, id_field, content_field, block_fields) = build_schema(blocking_fields.len());
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(50_000_000)?;

        // Use for_each_record for efficient iteration (single table scan
        // for SqliteStore, DashMap iteration for MemoryStore).
        let blk_fields = blocking_fields.to_vec();
        let blk_tantivy = block_fields.clone();
        let mut add_err: Option<tantivy::TantivyError> = None;
        store.for_each_record(side, &mut |id, record| {
            if add_err.is_some() {
                return;
            }
            let text = concat_fields(record, fields, side);
            if text.is_empty() {
                return;
            }
            let mut doc = doc!(
                id_field => id,
                content_field => text.as_str(),
            );
            // Add blocking key values so Tantivy can filter internally.
            for (i, bfp) in blk_fields.iter().enumerate() {
                let key = match side {
                    Side::A => &bfp.field_a,
                    Side::B => &bfp.field_b,
                };
                if let Some(val) = record.get(key) {
                    let normalised = val.trim().to_lowercase();
                    if !normalised.is_empty() {
                        doc.add_text(blk_tantivy[i], &normalised);
                    }
                }
            }
            if let Err(e) = writer.add_document(doc) {
                add_err = Some(e);
            }
        });
        if let Some(e) = add_err {
            return Err(e.into());
        }
        writer.commit()?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            writer,
            reader,
            id_field,
            content_field,
            block_fields,
            fields: fields.to_vec(),
            blocking_fields: blocking_fields.to_vec(),
            side,
            self_score_cache: HashMap::new(),
            dirty: false,
        })
    }

    /// Build an empty BM25 index (for live mode startup with no initial data).
    pub fn build_empty(
        fields: &[Bm25FieldPair],
        blocking_fields: &[BlockingFieldPair],
        side: Side,
    ) -> Result<Self, anyhow::Error> {
        let (schema, id_field, content_field, block_fields) = build_schema(blocking_fields.len());
        let index = Index::create_in_ram(schema);
        let writer = index.writer(50_000_000)?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            writer,
            reader,
            id_field,
            content_field,
            block_fields,
            fields: fields.to_vec(),
            blocking_fields: blocking_fields.to_vec(),
            side,
            self_score_cache: HashMap::new(),
            dirty: false,
        })
    }

    /// Query the index, returning the top-K results as `(id, raw_bm25_score)`.
    pub fn query(&self, text: &str, top_k: usize) -> Vec<(String, f32)> {
        if text.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let query = match query_parser.parse_query(&escape_query(text)) {
            Ok(q) => q,
            Err(e) => {
                warn!(error = %e, "bm25 query parse failed");
                return Vec::new();
            }
        };

        let top_docs = match searcher.search(&query, &TopDocs::with_limit(top_k)) {
            Ok(docs) => docs,
            Err(e) => {
                warn!(error = %e, "bm25 search failed");
                return Vec::new();
            }
        };

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_address)
                && let Some(id_val) = doc.get_first(self.id_field)
                && let Some(id) = Value::as_str(&id_val)
            {
                results.push((id.to_string(), score));
            }
        }
        results
    }

    /// Query the index with blocking filter, returning top-K results as
    /// `(id, raw_bm25_score)`.
    ///
    /// Combines the BM25 text query with term filters on blocking key fields.
    /// Only documents matching ALL blocking keys are scored — equivalent to
    /// post-filtering by `blocked_ids`, but Tantivy skips non-matching
    /// documents entirely (no BM25 scoring wasted on wrong blocks).
    ///
    /// `query_record` is the query-side record; `query_side` determines
    /// which field names to use for extracting blocking values.
    pub fn query_blocked(
        &self,
        text: &str,
        top_k: usize,
        query_record: &Record,
        query_side: Side,
    ) -> Vec<(String, f32)> {
        use tantivy::query::{BooleanQuery, Occur, TermQuery};
        use tantivy::schema::IndexRecordOption;

        if text.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let bm25_query = match query_parser.parse_query(&escape_query(text)) {
            Ok(q) => q,
            Err(e) => {
                warn!(error = %e, "bm25 query parse failed");
                return Vec::new();
            }
        };

        // If no blocking fields, fall back to plain BM25 query.
        if self.block_fields.is_empty() || self.blocking_fields.is_empty() {
            let top_docs = match searcher.search(&bm25_query, &TopDocs::with_limit(top_k)) {
                Ok(docs) => docs,
                Err(e) => {
                    warn!(error = %e, "bm25 search failed");
                    return Vec::new();
                }
            };
            return self.extract_results(&searcher, top_docs);
        }

        // Build blocking term filters from the query record.
        let mut block_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
        for (i, bfp) in self.blocking_fields.iter().enumerate() {
            let key = match query_side {
                Side::A => &bfp.field_a,
                Side::B => &bfp.field_b,
            };
            if let Some(val) = query_record.get(key) {
                let normalised = val.trim().to_lowercase();
                if !normalised.is_empty() {
                    let term = tantivy::Term::from_field_text(self.block_fields[i], &normalised);
                    block_clauses.push((
                        Occur::Must,
                        Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
                    ));
                }
            }
        }

        // Combine: BM25 query MUST match, all blocking terms MUST match.
        let mut clauses = vec![(Occur::Must, bm25_query)];
        clauses.extend(block_clauses);
        let combined = BooleanQuery::new(clauses);

        let top_docs = match searcher.search(&combined, &TopDocs::with_limit(top_k)) {
            Ok(docs) => docs,
            Err(e) => {
                warn!(error = %e, "bm25 blocked search failed");
                return Vec::new();
            }
        };
        self.extract_results(&searcher, top_docs)
    }

    /// Extract (id, score) pairs from Tantivy search results.
    fn extract_results(
        &self,
        searcher: &tantivy::Searcher,
        top_docs: Vec<(f32, tantivy::DocAddress)>,
    ) -> Vec<(String, f32)> {
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_address)
                && let Some(id_val) = doc.get_first(self.id_field)
                && let Some(id) = Value::as_str(&id_val)
            {
                results.push((id.to_string(), score));
            }
        }
        results
    }

    /// Score a set of candidate IDs against a query text in a single BM25 query.
    ///
    /// Returns `(scores_map, self_score)`:
    /// - `scores_map`: candidate ID → raw BM25 score. Candidates not in the
    ///   top results get no entry (treat as 0.0).
    /// - `self_score`: the cached self-score for normalisation. Falls back to
    ///   the top result's raw score if no self-score was pre-computed.
    pub fn score_candidates(
        &self,
        query_text: &str,
        candidate_ids: &[String],
        top_k: usize,
    ) -> (HashMap<String, f32>, f32) {
        // Run one query for enough results to cover the candidates.
        let limit = top_k.max(candidate_ids.len());
        let results = self.query(query_text, limit);

        // Use pre-computed self-score for normalisation. Fall back to
        // max-from-results if not available (e.g. live mode without
        // pre-computation).
        let fallback_max = results.first().map(|(_, s)| *s).unwrap_or(0.0);
        let norm_score = self.cached_self_score(query_text).unwrap_or(fallback_max);

        // Build lookup set for fast membership test.
        let wanted: std::collections::HashSet<&str> =
            candidate_ids.iter().map(|s| s.as_str()).collect();

        let scores: HashMap<String, f32> = results
            .into_iter()
            .filter(|(id, _)| wanted.contains(id.as_str()))
            .collect();

        (scores, norm_score)
    }

    /// Compute the BM25 score for a specific query text against all indexed
    /// documents, returning only the score for `candidate_id` (if found).
    pub fn score_one(&self, query_text: &str, candidate_id: &str) -> Option<f32> {
        let results = self.query(query_text, 1000);
        results
            .into_iter()
            .find(|(id, _)| id == candidate_id)
            .map(|(_, score)| score)
    }

    /// Compute the self-score: query text scored against itself. This is the
    /// theoretical maximum BM25 score for the given text, used for normalisation.
    ///
    /// The result is cached by query text hash. The cache is cleared on every
    /// commit (upsert/remove) to stay fresh as corpus IDF values change.
    pub fn self_score(&mut self, text: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        // Check cache first.
        let hash = text_hash(text);
        if let Some(&cached) = self.self_score_cache.get(&hash) {
            return cached;
        }

        // Cache miss — do the expensive insert/commit/query/delete/commit.
        let sentinel_id = "__bm25_self_score_sentinel__";

        if self
            .writer
            .add_document(doc!(
                self.id_field => sentinel_id,
                self.content_field => text,
            ))
            .is_err()
        {
            return 0.0;
        }
        if self.writer.commit().is_err() {
            return 0.0;
        }
        self.reader.reload().ok();

        let score = self.score_one(text, sentinel_id).unwrap_or(0.0);

        // Remove temp document
        let id_term = tantivy::Term::from_field_text(self.id_field, sentinel_id);
        self.writer.delete_term(id_term);
        let _ = self.writer.commit();
        self.reader.reload().ok();

        self.self_score_cache.insert(hash, score);
        score
    }

    /// Look up a cached self-score for the given text (read-only).
    ///
    /// Returns `None` if the self-score hasn't been pre-computed.
    /// Use `self_score()` or `precompute_self_scores()` to populate the cache
    /// first, then call this from a read-only context during scoring.
    pub fn cached_self_score(&self, text: &str) -> Option<f32> {
        if text.is_empty() {
            return Some(0.0);
        }
        let hash = text_hash(text);
        self.self_score_cache.get(&hash).copied()
    }

    /// Pre-compute self-scores for a batch of query texts.
    ///
    /// Uses batch insertion: all unique sentinel documents are inserted in a
    /// single commit, queried with a shared searcher, then deleted in a single
    /// commit. Total cost: 2 commits + N queries (with shared searcher).
    ///
    /// Call this with a write lock before the scoring loop begins.
    /// Subsequent calls to `cached_self_score()` will hit the cache without
    /// needing mutable access.
    pub fn precompute_self_scores(&mut self, texts: &[String]) {
        // Deduplicate and filter: only compute for texts not already cached.
        let mut unique: Vec<(u64, &str)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for text in texts {
            if text.is_empty() {
                continue;
            }
            let hash = text_hash(text);
            if self.self_score_cache.contains_key(&hash) {
                continue;
            }
            if seen.insert(hash) {
                unique.push((hash, text.as_str()));
            }
        }

        if unique.is_empty() {
            return;
        }

        // Phase 1: Insert all sentinel documents with unique IDs, single commit.
        let sentinel_prefix = "__bm25_self_";
        for (i, (_, text)) in unique.iter().enumerate() {
            let sentinel_id = format!("{sentinel_prefix}{i}__");
            let _ = self.writer.add_document(doc!(
                self.id_field => sentinel_id.as_str(),
                self.content_field => *text,
            ));
        }
        if self.writer.commit().is_err() {
            return;
        }
        self.reader.reload().ok();

        // Phase 2: Query each text with a shared searcher + query parser.
        // Reusing the searcher avoids per-query overhead of opening segment
        // readers. Each query finds its sentinel in the result set.
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);

        for (i, (hash, text)) in unique.iter().enumerate() {
            let sentinel_id = format!("{sentinel_prefix}{i}__");
            let escaped = escape_query(text);
            let query = match query_parser.parse_query(&escaped) {
                Ok(q) => q,
                Err(_) => {
                    self.self_score_cache.insert(*hash, 0.0);
                    continue;
                }
            };

            // Search with limit large enough to find the sentinel among
            // real documents. The sentinel should score highest (exact
            // text match) but use a generous limit just in case.
            let top_docs = match searcher.search(&query, &TopDocs::with_limit(50)) {
                Ok(docs) => docs,
                Err(_) => {
                    self.self_score_cache.insert(*hash, 0.0);
                    continue;
                }
            };

            // Find the sentinel's score in the results.
            let mut found = false;
            for (score, doc_address) in &top_docs {
                if let Ok(doc) = searcher.doc::<TantivyDocument>(*doc_address)
                    && let Some(id_val) = doc.get_first(self.id_field)
                    && let Some(id) = Value::as_str(&id_val)
                    && id == sentinel_id
                {
                    self.self_score_cache.insert(*hash, *score);
                    found = true;
                    break;
                }
            }
            if !found {
                // Sentinel not in top-50 — use top result as fallback.
                let fallback = top_docs.first().map(|(s, _)| *s).unwrap_or(0.0);
                self.self_score_cache.insert(*hash, fallback);
            }
        }

        // Phase 3: Delete all sentinels, single commit.
        for (i, _) in unique.iter().enumerate() {
            let sentinel_id = format!("{sentinel_prefix}{i}__");
            let id_term = tantivy::Term::from_field_text(self.id_field, &sentinel_id);
            self.writer.delete_term(id_term);
        }
        let _ = self.writer.commit();
        self.reader.reload().ok();
    }

    /// Commit pending writes and reload the reader if dirty.
    ///
    /// Call this before querying to make buffered upserts/removes visible.
    /// Clears the self-score cache (corpus IDF may have changed).
    pub fn commit_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        let _ = self.writer.commit();
        self.reader.reload().ok();
        self.self_score_cache.clear();
        self.dirty = false;
    }

    /// Insert or update a record in the index.
    ///
    /// The document is added to the writer but not committed. Call
    /// `commit_if_dirty()` before querying to make pending changes visible.
    pub fn upsert(&mut self, id: &str, record: &Record) {
        // Delete any existing document for this id
        let id_term = tantivy::Term::from_field_text(self.id_field, id);
        self.writer.delete_term(id_term);

        let text = concat_fields(record, &self.fields, self.side);
        if !text.is_empty() {
            let mut doc = doc!(
                self.id_field => id,
                self.content_field => text.as_str(),
            );
            // Add blocking key values
            for (i, bfp) in self.blocking_fields.iter().enumerate() {
                let key = match self.side {
                    Side::A => &bfp.field_a,
                    Side::B => &bfp.field_b,
                };
                if let Some(val) = record.get(key) {
                    let normalised = val.trim().to_lowercase();
                    if !normalised.is_empty() {
                        doc.add_text(self.block_fields[i], &normalised);
                    }
                }
            }
            let _ = self.writer.add_document(doc);
        }
        self.dirty = true;
    }

    /// Remove a record from the index.
    ///
    /// The delete is buffered. Call `commit_if_dirty()` before querying
    /// to make the removal visible.
    pub fn remove(&mut self, id: &str) {
        let id_term = tantivy::Term::from_field_text(self.id_field, id);
        self.writer.delete_term(id_term);
        self.dirty = true;
    }

    /// Build the concatenated query text for a record on the given side.
    ///
    /// Uses the opposite side's field names, since the query record is from
    /// the query side and the index is built from the pool side. For example,
    /// when a B record queries the A-side index, we use `field_b` names to
    /// extract text from the B record.
    pub fn query_text_for(&self, record: &Record, query_side: Side) -> String {
        concat_fields(record, &self.fields, query_side)
    }

    /// Return the number of indexed documents.
    pub fn num_docs(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }
}

// ---
// Internal helpers
// ---

/// Simple hash for self-score cache keys.
fn text_hash(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

fn build_schema(num_block_fields: usize) -> (Schema, Field, Field, Vec<Field>) {
    let mut builder = Schema::builder();
    let id_field = builder.add_text_field("id", STRING | STORED);
    let content_field = builder.add_text_field("content", TEXT);
    let mut block_fields = Vec::with_capacity(num_block_fields);
    for i in 0..num_block_fields {
        // STRING (not TEXT): indexed as a single token for exact term filtering.
        // Not STORED: we never retrieve the value, only filter on it.
        let field = builder.add_text_field(&format!("block_{}", i), STRING);
        block_fields.push(field);
    }
    (builder.build(), id_field, content_field, block_fields)
}

/// Concatenate the relevant text fields from a record into a single string.
fn concat_fields(record: &Record, fields: &[Bm25FieldPair], side: Side) -> String {
    let mut parts = Vec::new();
    for pair in fields {
        let key = match side {
            Side::A => pair.field_a.as_str(),
            Side::B => pair.field_b.as_str(),
        };
        if let Some(val) = record.get(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
    }
    parts.join(" ")
}

/// Escape special Tantivy query syntax characters.
///
/// Tantivy's query parser treats characters like `:`, `(`, `)`, `+`, `-`,
/// `"`, `*`, `~` as syntax. For BM25 scoring we want plain tokenised text,
/// so we escape them.
fn escape_query(text: &str) -> String {
    let special = [
        ':', '(', ')', '[', ']', '{', '}', '+', '-', '!', '"', '~', '*', '?', '\\', '^',
    ];
    let mut escaped = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        if special.contains(&ch) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockingConfig;
    use crate::store::memory::MemoryStore;

    fn make_record(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_store(data: Vec<(&str, Record)>, side: Side) -> MemoryStore {
        let store = MemoryStore::new(&BlockingConfig::default());
        for (id, rec) in data {
            store.insert(side, id, &rec);
        }
        store
    }

    fn test_fields() -> Vec<Bm25FieldPair> {
        vec![
            Bm25FieldPair {
                field_a: "name_a".to_string(),
                field_b: "name_b".to_string(),
            },
            Bm25FieldPair {
                field_a: "desc_a".to_string(),
                field_b: "desc_b".to_string(),
            },
        ]
    }

    #[test]
    fn build_and_query_basic() {
        let store = make_store(
            vec![
                (
                    "1",
                    make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology company")]),
                ),
                (
                    "2",
                    make_record(&[("name_a", "Microsoft"), ("desc_a", "software company")]),
                ),
                (
                    "3",
                    make_record(&[("name_a", "Apple Farms"), ("desc_a", "agriculture farm")]),
                ),
                (
                    "4",
                    make_record(&[("name_a", "Google LLC"), ("desc_a", "search engine")]),
                ),
                (
                    "5",
                    make_record(&[("name_a", "Amazon"), ("desc_a", "e-commerce company")]),
                ),
            ],
            Side::A,
        );

        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        assert_eq!(idx.num_docs(), 5);

        let results = idx.query("apple", 3);
        assert!(!results.is_empty(), "should find apple records");
        // Both Apple Inc and Apple Farms should be in results
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"1"), "Apple Inc should be found");
        assert!(ids.contains(&"3"), "Apple Farms should be found");
    }

    #[test]
    fn query_empty_text() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "Apple Inc")]))],
            Side::A,
        );
        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        let results = idx.query("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn query_zero_top_k() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "Apple Inc")]))],
            Side::A,
        );
        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        let results = idx.query("apple", 0);
        assert!(results.is_empty());
    }

    #[test]
    fn score_one_found() {
        let store = make_store(
            vec![
                (
                    "1",
                    make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology")]),
                ),
                (
                    "2",
                    make_record(&[("name_a", "Microsoft"), ("desc_a", "software")]),
                ),
            ],
            Side::A,
        );
        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        let score = idx.score_one("apple technology", "1");
        assert!(score.is_some(), "should find candidate 1");
        assert!(score.unwrap() > 0.0, "score should be positive");
    }

    #[test]
    fn score_one_not_found() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "Apple Inc")]))],
            Side::A,
        );
        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        let score = idx.score_one("xyz zzz qqq", "1");
        // The query has no overlap with the document, so score should be None
        assert!(score.is_none(), "no-overlap query should return None");
    }

    #[test]
    fn self_score_positive() {
        let store = make_store(
            vec![
                ("1", make_record(&[("name_a", "Apple Inc technology")])),
                ("2", make_record(&[("name_a", "Microsoft software")])),
            ],
            Side::A,
        );
        let mut idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        let ss = idx.self_score("apple inc technology");
        assert!(ss > 0.0, "self-score should be positive, got {}", ss);
        // After self-score, sentinel should be removed
        assert_eq!(idx.num_docs(), 2, "sentinel should be cleaned up");
    }

    #[test]
    fn self_score_empty() {
        let store = MemoryStore::new(&BlockingConfig::default());
        let mut idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        let ss = idx.self_score("");
        assert!(
            (ss - 0.0).abs() < f32::EPSILON,
            "empty self-score should be 0.0"
        );
    }

    #[test]
    fn upsert_and_query() {
        let store = MemoryStore::new(&BlockingConfig::default());
        let mut idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        assert_eq!(idx.num_docs(), 0);

        // Upsert a record (buffered — must commit before querying)
        let rec = make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology")]);
        idx.upsert("1", &rec);
        idx.commit_if_dirty();
        assert_eq!(idx.num_docs(), 1);

        let results = idx.query("apple", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Update the record
        let rec2 = make_record(&[("name_a", "Microsoft"), ("desc_a", "software")]);
        idx.upsert("1", &rec2);
        idx.commit_if_dirty();
        assert_eq!(idx.num_docs(), 1);

        let results = idx.query("microsoft", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Old content should no longer match
        let results = idx.query("apple", 5);
        assert!(
            results.is_empty(),
            "old content should not match after upsert"
        );
    }

    #[test]
    fn remove_from_index() {
        let store = make_store(
            vec![
                ("1", make_record(&[("name_a", "Apple Inc")])),
                ("2", make_record(&[("name_a", "Microsoft")])),
            ],
            Side::A,
        );
        let mut idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        assert_eq!(idx.num_docs(), 2);

        idx.remove("1");
        idx.commit_if_dirty();

        let results = idx.query("apple", 5);
        assert!(results.is_empty(), "removed record should not match");
    }

    #[test]
    fn side_b_uses_field_b_names() {
        let fields = vec![Bm25FieldPair {
            field_a: "name_a".to_string(),
            field_b: "name_b".to_string(),
        }];
        let store = make_store(
            vec![
                ("1", make_record(&[("name_b", "Apple Inc")])),
                ("2", make_record(&[("name_b", "Microsoft")])),
            ],
            Side::B,
        );
        let idx = BM25Index::build(&store, Side::B, &fields, &[]).unwrap();
        assert_eq!(idx.num_docs(), 2);

        let results = idx.query("apple", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn empty_fields_skipped() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", ""), ("desc_a", "")]))],
            Side::A,
        );
        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        // Record with all empty fields should not be indexed
        assert_eq!(idx.num_docs(), 0);
    }

    #[test]
    fn special_characters_handled() {
        let store = make_store(
            vec![
                (
                    "1",
                    make_record(&[("name_a", "O'Brien & Associates (UK) Ltd.")]),
                ),
                ("2", make_record(&[("name_a", "Smith+Jones: Partners")])),
            ],
            Side::A,
        );
        let idx = BM25Index::build(&store, Side::A, &test_fields(), &[]).unwrap();
        assert_eq!(idx.num_docs(), 2);

        // Should be able to query without crashing
        let results = idx.query("O'Brien & Associates", 5);
        assert!(!results.is_empty(), "should find O'Brien record");
    }

    #[test]
    fn build_empty_index() {
        let idx = BM25Index::build_empty(&test_fields(), &[], Side::A).unwrap();
        assert_eq!(idx.num_docs(), 0);
        let results = idx.query("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn concat_fields_helper() {
        let rec = make_record(&[
            ("name_a", "Apple"),
            ("desc_a", "tech"),
            ("other", "ignored"),
        ]);
        let fields = vec![
            Bm25FieldPair {
                field_a: "name_a".to_string(),
                field_b: "name_b".to_string(),
            },
            Bm25FieldPair {
                field_a: "desc_a".to_string(),
                field_b: "desc_b".to_string(),
            },
        ];
        let text = concat_fields(&rec, &fields, Side::A);
        assert_eq!(text, "Apple tech");
    }

    #[test]
    fn escape_query_fn() {
        assert_eq!(escape_query("hello world"), "hello world");
        assert_eq!(escape_query("a:b"), "a\\:b");
        assert_eq!(escape_query("(test)"), "\\(test\\)");
        assert_eq!(escape_query("a+b-c"), "a\\+b\\-c");
    }
}
