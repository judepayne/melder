//! Tantivy-backed BM25 index for candidate retrieval and pair scoring.
//!
//! Each side (A / B) gets its own `BM25Index`. The index concatenates all
//! text fields designated by `bm25_fields` into a single `content` field
//! per document, enabling cross-field BM25 scoring with zero configuration.

use dashmap::DashMap;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};
use tracing::warn;

use crate::models::{Record, Side};

/// In-memory BM25 index wrapping Tantivy.
pub struct BM25Index {
    index: Index,
    writer: IndexWriter,
    id_field: Field,
    content_field: Field,
    /// Which text fields to concatenate per record. Each entry is
    /// `(field_a_name, field_b_name)` — the side determines which name
    /// is used for lookup.
    fields: Vec<(String, String)>,
    side: Side,
}

impl BM25Index {
    /// Build a BM25 index from an existing record store.
    ///
    /// `fields` are `(field_a, field_b)` pairs from fuzzy/embedding match
    /// field entries. For each record, the text values of these fields are
    /// concatenated (space-separated) into a single indexed document.
    pub fn build(
        records: &DashMap<String, Record>,
        fields: &[(String, String)],
        side: Side,
    ) -> Result<Self, anyhow::Error> {
        let (schema, id_field, content_field) = build_schema();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(50_000_000)?;

        for entry in records.iter() {
            let id = entry.key();
            let record = entry.value();
            let text = concat_fields(record, fields, side);
            if text.is_empty() {
                continue;
            }
            writer.add_document(doc!(
                id_field => id.as_str(),
                content_field => text.as_str(),
            ))?;
        }
        writer.commit()?;

        Ok(Self {
            index,
            writer,
            id_field,
            content_field,
            fields: fields.to_vec(),
            side,
        })
    }

    /// Build an empty BM25 index (for live mode startup with no initial data).
    pub fn build_empty(fields: &[(String, String)], side: Side) -> Result<Self, anyhow::Error> {
        let (schema, id_field, content_field) = build_schema();
        let index = Index::create_in_ram(schema);
        let writer = index.writer(50_000_000)?;

        Ok(Self {
            index,
            writer,
            id_field,
            content_field,
            fields: fields.to_vec(),
            side,
        })
    }

    /// Query the index, returning the top-K results as `(id, raw_bm25_score)`.
    pub fn query(&self, text: &str, top_k: usize) -> Vec<(String, f32)> {
        if text.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let reader = match self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
        {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "bm25 reader creation failed");
                return Vec::new();
            }
        };
        let searcher = reader.searcher();

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

    /// Compute the BM25 score for a specific query text against all indexed
    /// documents, returning only the score for `candidate_id` (if found).
    pub fn score_one(&self, query_text: &str, candidate_id: &str) -> Option<f32> {
        // We query for more results than needed to increase the chance of
        // finding the candidate. If the candidate is not in the top results,
        // we return None (score is negligible).
        let results = self.query(query_text, 1000);
        results
            .into_iter()
            .find(|(id, _)| id == candidate_id)
            .map(|(_, score)| score)
    }

    /// Compute the self-score: query text scored against itself. This is the
    /// theoretical maximum BM25 score for the given text, used for normalisation.
    ///
    /// Inserts a temporary document, queries, then removes it. The temporary
    /// document uses a sentinel ID that won't collide with real records.
    pub fn self_score(&mut self, text: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        let sentinel_id = "__bm25_self_score_sentinel__";

        // Insert temp document
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

        // Query for sentinel
        let score = self.score_one(text, sentinel_id).unwrap_or(0.0);

        // Remove temp document
        let id_term = tantivy::Term::from_field_text(self.id_field, sentinel_id);
        self.writer.delete_term(id_term);
        let _ = self.writer.commit();

        score
    }

    /// Insert or update a record in the index.
    pub fn upsert(&mut self, id: &str, record: &Record) {
        // Delete any existing document for this id
        let id_term = tantivy::Term::from_field_text(self.id_field, id);
        self.writer.delete_term(id_term);

        let text = concat_fields(record, &self.fields, self.side);
        if !text.is_empty() {
            let _ = self.writer.add_document(doc!(
                self.id_field => id,
                self.content_field => text.as_str(),
            ));
        }
        let _ = self.writer.commit();
    }

    /// Remove a record from the index.
    pub fn remove(&mut self, id: &str) {
        let id_term = tantivy::Term::from_field_text(self.id_field, id);
        self.writer.delete_term(id_term);
        let _ = self.writer.commit();
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
        let reader = match self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
        {
            Ok(r) => r,
            Err(_) => return 0,
        };
        let searcher = reader.searcher();
        searcher.num_docs()
    }
}

// ---
// Internal helpers
// ---

fn build_schema() -> (Schema, Field, Field) {
    let mut builder = Schema::builder();
    let id_field = builder.add_text_field("id", STRING | STORED);
    let content_field = builder.add_text_field("content", TEXT);
    (builder.build(), id_field, content_field)
}

/// Concatenate the relevant text fields from a record into a single string.
fn concat_fields(record: &Record, fields: &[(String, String)], side: Side) -> String {
    let mut parts = Vec::new();
    for (fa, fb) in fields {
        let key = match side {
            Side::A => fa.as_str(),
            Side::B => fb.as_str(),
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

    fn make_record(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_records(data: Vec<(&str, Record)>) -> DashMap<String, Record> {
        let map = DashMap::new();
        for (id, rec) in data {
            map.insert(id.to_string(), rec);
        }
        map
    }

    fn test_fields() -> Vec<(String, String)> {
        vec![
            ("name_a".to_string(), "name_b".to_string()),
            ("desc_a".to_string(), "desc_b".to_string()),
        ]
    }

    #[test]
    fn build_and_query_basic() {
        let records = make_records(vec![
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
        ]);

        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
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
        let records = make_records(vec![("1", make_record(&[("name_a", "Apple Inc")]))]);
        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        let results = idx.query("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn query_zero_top_k() {
        let records = make_records(vec![("1", make_record(&[("name_a", "Apple Inc")]))]);
        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        let results = idx.query("apple", 0);
        assert!(results.is_empty());
    }

    #[test]
    fn score_one_found() {
        let records = make_records(vec![
            (
                "1",
                make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology")]),
            ),
            (
                "2",
                make_record(&[("name_a", "Microsoft"), ("desc_a", "software")]),
            ),
        ]);
        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        let score = idx.score_one("apple technology", "1");
        assert!(score.is_some(), "should find candidate 1");
        assert!(score.unwrap() > 0.0, "score should be positive");
    }

    #[test]
    fn score_one_not_found() {
        let records = make_records(vec![("1", make_record(&[("name_a", "Apple Inc")]))]);
        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        let score = idx.score_one("xyz zzz qqq", "1");
        // The query has no overlap with the document, so score should be None
        assert!(score.is_none(), "no-overlap query should return None");
    }

    #[test]
    fn self_score_positive() {
        let records = make_records(vec![
            ("1", make_record(&[("name_a", "Apple Inc technology")])),
            ("2", make_record(&[("name_a", "Microsoft software")])),
        ]);
        let mut idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        let ss = idx.self_score("apple inc technology");
        assert!(ss > 0.0, "self-score should be positive, got {}", ss);
        // After self-score, sentinel should be removed
        assert_eq!(idx.num_docs(), 2, "sentinel should be cleaned up");
    }

    #[test]
    fn self_score_empty() {
        let records: DashMap<String, Record> = DashMap::new();
        let mut idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        let ss = idx.self_score("");
        assert!(
            (ss - 0.0).abs() < f32::EPSILON,
            "empty self-score should be 0.0"
        );
    }

    #[test]
    fn upsert_and_query() {
        let records: DashMap<String, Record> = DashMap::new();
        let mut idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        assert_eq!(idx.num_docs(), 0);

        // Upsert a record
        let rec = make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology")]);
        idx.upsert("1", &rec);
        assert_eq!(idx.num_docs(), 1);

        let results = idx.query("apple", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Update the record
        let rec2 = make_record(&[("name_a", "Microsoft"), ("desc_a", "software")]);
        idx.upsert("1", &rec2);
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
        let records = make_records(vec![
            ("1", make_record(&[("name_a", "Apple Inc")])),
            ("2", make_record(&[("name_a", "Microsoft")])),
        ]);
        let mut idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        assert_eq!(idx.num_docs(), 2);

        idx.remove("1");
        assert_eq!(idx.num_docs(), 1);

        let results = idx.query("apple", 5);
        assert!(results.is_empty(), "removed record should not match");
    }

    #[test]
    fn side_b_uses_field_b_names() {
        let fields = vec![("name_a".to_string(), "name_b".to_string())];
        let records = make_records(vec![
            ("1", make_record(&[("name_b", "Apple Inc")])),
            ("2", make_record(&[("name_b", "Microsoft")])),
        ]);
        let idx = BM25Index::build(&records, &fields, Side::B).unwrap();
        assert_eq!(idx.num_docs(), 2);

        let results = idx.query("apple", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn empty_fields_skipped() {
        let records = make_records(vec![("1", make_record(&[("name_a", ""), ("desc_a", "")]))]);
        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        // Record with all empty fields should not be indexed
        assert_eq!(idx.num_docs(), 0);
    }

    #[test]
    fn special_characters_handled() {
        let records = make_records(vec![
            (
                "1",
                make_record(&[("name_a", "O'Brien & Associates (UK) Ltd.")]),
            ),
            ("2", make_record(&[("name_a", "Smith+Jones: Partners")])),
        ]);
        let idx = BM25Index::build(&records, &test_fields(), Side::A).unwrap();
        assert_eq!(idx.num_docs(), 2);

        // Should be able to query without crashing
        let results = idx.query("O'Brien & Associates", 5);
        assert!(!results.is_empty(), "should find O'Brien record");
    }

    #[test]
    fn build_empty_index() {
        let idx = BM25Index::build_empty(&test_fields(), Side::A).unwrap();
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
            ("name_a".to_string(), "name_b".to_string()),
            ("desc_a".to_string(), "desc_b".to_string()),
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
