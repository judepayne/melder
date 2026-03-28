//! DashMap-based BM25 scorer — a lock-free replacement for the Tantivy index.
//!
//! `SimpleBm25` stores per-document term frequencies, global IDF statistics,
//! and block-segmented posting lists in concurrent `DashMap` structures.
//! Writes are instantly visible (no commit step). Queries use either
//! exhaustive blocked-set scoring (small blocks) or inverted-index lookup
//! (large blocks) — the caller sees a single `score_blocked()` interface.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;

use crate::config::BlockingFieldPair;
use crate::config::schema::Bm25FieldPair;
use crate::models::{Record, Side};
use crate::store::RecordStore;

// ---
// BM25 constants (Tantivy / Lucene defaults)
// ---
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Block size threshold: blocks smaller than this use exhaustive scoring;
/// larger blocks use the inverted index path.
const DEFAULT_EXHAUSTIVE_THRESHOLD: usize = 5_000;

// ---
// Data types
// ---

/// A single entry in a posting list.
#[derive(Clone, Debug)]
struct PostingEntry {
    /// Hash of the document's blocking key(s).
    block_key: u64,
    /// Document identifier.
    doc_id: String,
    /// Term frequency in this document.
    tf: u32,
}

/// Per-document metadata stored alongside doc_terms for efficient upsert.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DocMeta {
    /// Term → frequency map for this document.
    terms: HashMap<String, u32>,
    /// Total number of tokens (document length for BM25).
    length: u32,
    /// Block key for posting list placement.
    block_key: u64,
}

// ---
// SimpleBm25
// ---

/// Lock-free, concurrent BM25 scorer backed by `DashMap`.
///
/// Thread-safe for concurrent reads and writes without external
/// synchronisation. Intended as a drop-in replacement for the
/// Tantivy-backed `BM25Index`.
pub struct SimpleBm25 {
    /// doc_id → per-document metadata (term freqs, length, block key).
    docs: DashMap<String, DocMeta>,

    /// term → number of documents containing this term (for IDF).
    doc_freq: DashMap<String, usize>,

    /// term → posting list sorted by (block_key, doc_id).
    postings: DashMap<String, Vec<PostingEntry>>,

    /// Total number of indexed documents.
    total_docs: AtomicUsize,

    /// Sum of all document lengths (for avgdl).
    total_tokens: AtomicU64,

    /// Which record fields to concatenate and tokenise.
    fields: Vec<Bm25FieldPair>,

    /// Blocking field pairs for computing block keys.
    blocking_fields: Vec<BlockingFieldPair>,

    /// Which side (A or B) this index serves.
    side: Side,

    /// Blocks larger than this use the inverted index path.
    exhaustive_threshold: usize,
}

impl std::fmt::Debug for SimpleBm25 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleBm25")
            .field("total_docs", &self.total_docs.load(Ordering::Relaxed))
            .field("side", &self.side)
            .finish()
    }
}

impl SimpleBm25 {
    /// Create a new empty index.
    pub fn new(
        fields: &[Bm25FieldPair],
        blocking_fields: &[BlockingFieldPair],
        side: Side,
    ) -> Self {
        Self {
            docs: DashMap::new(),
            doc_freq: DashMap::new(),
            postings: DashMap::new(),
            total_docs: AtomicUsize::new(0),
            total_tokens: AtomicU64::new(0),
            fields: fields.to_vec(),
            blocking_fields: blocking_fields.to_vec(),
            side,
            exhaustive_threshold: DEFAULT_EXHAUSTIVE_THRESHOLD,
        }
    }

    /// Bulk-build from a `RecordStore`.
    pub fn build(
        store: &dyn RecordStore,
        side: Side,
        fields: &[Bm25FieldPair],
        blocking_fields: &[BlockingFieldPair],
    ) -> Self {
        let idx = Self::new(fields, blocking_fields, side);
        store.for_each_record(side, &mut |id, record| {
            idx.upsert(id, record);
        });
        idx
    }

    // ---------------------------------------------------------------
    // Write path
    // ---------------------------------------------------------------

    /// Insert or update a document in the index. Instantly visible.
    pub fn upsert(&self, id: &str, record: &Record) {
        let text = concat_fields(record, &self.fields, self.side);
        let tokens = tokenise(&text);

        // Remove old entry if exists.
        if let Some((_, old_meta)) = self.docs.remove(id) {
            self.decrement_stats(&old_meta, id);
        }

        if tokens.is_empty() {
            return;
        }

        let tf_map = term_frequencies(&tokens);
        let length = tokens.len() as u32;
        let block_key = compute_block_key(record, &self.blocking_fields, self.side);

        let meta = DocMeta {
            terms: tf_map.clone(),
            length,
            block_key,
        };

        // Insert new doc metadata.
        self.docs.insert(id.to_string(), meta);

        // Update corpus stats.
        self.total_docs.fetch_add(1, Ordering::Relaxed);
        self.total_tokens
            .fetch_add(length as u64, Ordering::Relaxed);

        // Update doc_freq and postings for each term.
        for (term, tf) in &tf_map {
            self.doc_freq
                .entry(term.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            let entry = PostingEntry {
                block_key,
                doc_id: id.to_string(),
                tf: *tf,
            };
            self.postings
                .entry(term.clone())
                .and_modify(|list| {
                    let pos = list
                        .partition_point(|e| (e.block_key, &e.doc_id) < (block_key, &entry.doc_id));
                    list.insert(pos, entry.clone());
                })
                .or_insert_with(|| vec![entry]);
        }
    }

    /// Remove a document from the index.
    pub fn remove(&self, id: &str) {
        if let Some((_, old_meta)) = self.docs.remove(id) {
            self.decrement_stats(&old_meta, id);
        }
    }

    /// Decrement corpus stats and remove posting entries for a document.
    fn decrement_stats(&self, meta: &DocMeta, id: &str) {
        self.total_docs.fetch_sub(1, Ordering::Relaxed);
        self.total_tokens
            .fetch_sub(meta.length as u64, Ordering::Relaxed);

        for term in meta.terms.keys() {
            // Decrement doc_freq; remove entry if it reaches 0.
            let mut remove_key = false;
            if let Some(mut df) = self.doc_freq.get_mut(term) {
                if *df <= 1 {
                    remove_key = true;
                } else {
                    *df -= 1;
                }
            }
            if remove_key {
                self.doc_freq.remove(term);
            }

            // Remove from posting list.
            if let Some(mut list) = self.postings.get_mut(term) {
                list.retain(|e| e.doc_id != id);
                if list.is_empty() {
                    drop(list);
                    self.postings.remove(term);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Read path
    // ---------------------------------------------------------------

    /// Build the concatenated query text for a record.
    pub fn query_text_for(&self, record: &Record, query_side: Side) -> String {
        concat_fields(record, &self.fields, query_side)
    }

    /// Score blocked candidates, returning top-K as `(id, raw_bm25_score)`.
    ///
    /// Automatically selects exhaustive or inverted index strategy based on
    /// the blocked set size.
    ///
    /// `query_record` and `query_side` are used to compute the block key
    /// for the inverted index path.
    pub fn score_blocked(
        &self,
        query_text: &str,
        blocked_ids: &[String],
        top_k: usize,
        query_record: &Record,
        query_side: Side,
    ) -> Vec<(String, f32)> {
        if query_text.is_empty() || blocked_ids.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let sanitised = sanitize_query(query_text);
        let tokens = tokenise(&sanitised);
        if tokens.is_empty() {
            return Vec::new();
        }
        let query_tfs = term_frequencies(&tokens);

        let n = self.total_docs.load(Ordering::Relaxed) as f64;
        if n == 0.0 {
            return Vec::new();
        }
        let avg_dl = self.total_tokens.load(Ordering::Relaxed) as f64 / n;

        // Pre-compute IDF for each query term.
        let term_idfs: Vec<(&str, u32, f64)> = query_tfs
            .iter()
            .filter_map(|(term, &tf)| {
                let df = self.doc_freq.get(term).map(|v| *v as f64).unwrap_or(0.0);
                if df == 0.0 {
                    return None;
                }
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                Some((term.as_str(), tf, idf))
            })
            .collect();

        if term_idfs.is_empty() {
            return Vec::new();
        }

        if blocked_ids.len() <= self.exhaustive_threshold {
            self.score_exhaustive(&term_idfs, blocked_ids, avg_dl, top_k)
        } else {
            let block_key = compute_block_key(query_record, &self.blocking_fields, query_side);
            self.score_inverted(&term_idfs, block_key, blocked_ids, avg_dl, top_k)
        }
    }

    /// Exhaustive scoring: iterate all blocked docs. O(B × K).
    fn score_exhaustive(
        &self,
        term_idfs: &[(&str, u32, f64)],
        blocked_ids: &[String],
        avg_dl: f64,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = blocked_ids
            .iter()
            .filter_map(|doc_id| {
                let meta = self.docs.get(doc_id)?;
                let dl = meta.length as f64;
                let mut score = 0.0f64;
                for &(term, _qtf, idf) in term_idfs {
                    let tf = meta.terms.get(term).copied().unwrap_or(0) as f64;
                    if tf > 0.0 {
                        score += bm25_term_score(tf, idf, dl, avg_dl);
                    }
                }
                if score > 0.0 {
                    Some((doc_id.clone(), score as f32))
                } else {
                    None
                }
            })
            .collect();

        partial_top_k(&mut scores, top_k);
        scores
    }

    /// Inverted index scoring: binary-search posting lists by block_key.
    /// O(K × log P + R × K) where R = result set size.
    fn score_inverted(
        &self,
        term_idfs: &[(&str, u32, f64)],
        block_key: u64,
        blocked_ids: &[String],
        avg_dl: f64,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        // Build a set for fast membership filtering.
        let allowed: std::collections::HashSet<&str> =
            blocked_ids.iter().map(|s| s.as_str()).collect();

        let mut doc_scores: HashMap<String, f64> = HashMap::new();

        for &(term, _qtf, idf) in term_idfs {
            if let Some(list) = self.postings.get(term) {
                // Binary search to first entry with matching block_key.
                let start = list.partition_point(|e| e.block_key < block_key);
                for entry in &list[start..] {
                    if entry.block_key != block_key {
                        break;
                    }
                    if !allowed.contains(entry.doc_id.as_str()) {
                        continue;
                    }
                    let dl = self
                        .docs
                        .get(&entry.doc_id)
                        .map(|m| m.length as f64)
                        .unwrap_or(0.0);
                    let tf = entry.tf as f64;
                    let term_score = bm25_term_score(tf, idf, dl, avg_dl);
                    *doc_scores.entry(entry.doc_id.clone()).or_insert(0.0) += term_score;
                }
            }
        }

        let mut scores: Vec<(String, f32)> = doc_scores
            .into_iter()
            .map(|(id, s)| (id, s as f32))
            .collect();

        partial_top_k(&mut scores, top_k);
        scores
    }

    /// Analytical self-score: BM25 score of a query against itself.
    ///
    /// Uses current corpus IDF statistics. O(K) where K = unique query tokens.
    /// Terms not present in the corpus (df=0) are skipped.
    pub fn analytical_self_score(&self, query_text: &str) -> f32 {
        if query_text.is_empty() {
            return 0.0;
        }

        let sanitised = sanitize_query(query_text);
        let tokens = tokenise(&sanitised);
        if tokens.is_empty() {
            return 0.0;
        }

        let tf_map = term_frequencies(&tokens);
        let query_len = tokens.len() as f64;

        let n = self.total_docs.load(Ordering::Relaxed) as f64;
        if n == 0.0 {
            return 0.0;
        }
        let avg_dl = self.total_tokens.load(Ordering::Relaxed) as f64 / n;

        let mut score = 0.0f64;
        for (term, &tf) in &tf_map {
            let df = self.doc_freq.get(term).map(|v| *v as f64).unwrap_or(0.0);
            if df == 0.0 {
                continue;
            }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            score += bm25_term_score(tf as f64, idf, query_len, avg_dl);
        }

        score as f32
    }

    /// Number of indexed documents.
    pub fn num_docs(&self) -> usize {
        self.total_docs.load(Ordering::Relaxed)
    }
}

// ---
// Internal helpers
// ---

/// BM25 term score: IDF × (tf × (k1 + 1)) / (tf + k1 × (1 − b + b × dl/avgdl))
#[inline]
fn bm25_term_score(tf: f64, idf: f64, dl: f64, avg_dl: f64) -> f64 {
    idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl))
}

/// Partial sort for top-K, then truncate and sort descending.
fn partial_top_k(scores: &mut Vec<(String, f32)>, top_k: usize) {
    if scores.len() > top_k {
        scores.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

/// Tokenise text: replace non-alphanumeric with spaces, split, lowercase.
///
/// This matches the effective behaviour of `sanitize_query()` + Tantivy's
/// default `SimpleTokenizer` (lowercase + split on non-alphanumeric).
fn tokenise(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Count term frequencies.
fn term_frequencies(tokens: &[String]) -> HashMap<String, u32> {
    let mut freqs = HashMap::new();
    for token in tokens {
        *freqs.entry(token.clone()).or_insert(0) += 1;
    }
    freqs
}

/// Concatenate the relevant text fields from a record into a single string.
pub fn concat_fields(record: &Record, fields: &[Bm25FieldPair], side: Side) -> String {
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

/// Sanitize text for BM25 querying. Strips punctuation and special characters,
/// collapses whitespace, lowercases. Matches `index.rs::sanitize_query()`.
pub fn sanitize_query(text: &str) -> String {
    let replaced: String = text
        .chars()
        .map(|ch| match ch {
            ':' | '(' | ')' | '[' | ']' | '{' | '}' | '+' | '!' | '"' | '~' | '*' | '?' | '\\'
            | '^' | '&' | '|' => ' ',
            ',' | '.' | ';' | '\'' | '`' | '#' | '@' | '$' | '%' | '/' => ' ',
            '-' => ' ',
            _ => ch,
        })
        .collect();
    replaced
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Compute block key from a record's blocking field values.
fn compute_block_key(record: &Record, blocking_fields: &[BlockingFieldPair], side: Side) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for bfp in blocking_fields {
        let key = match side {
            Side::A => &bfp.field_a,
            Side::B => &bfp.field_b,
        };
        if let Some(val) = record.get(key) {
            val.trim().to_lowercase().hash(&mut hasher);
        }
    }
    hasher.finish()
}

// ---
// Tests
// ---

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

        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        assert_eq!(idx.num_docs(), 5);

        // All docs share the same block (no blocking fields), so pass empty
        // blocked_ids = all IDs. Use a dummy record for block key.
        let blocked_ids: Vec<String> = (1..=5).map(|i| i.to_string()).collect();
        let dummy_rec = make_record(&[]);
        let results = idx.score_blocked("apple", &blocked_ids, 3, &dummy_rec, Side::B);
        assert!(!results.is_empty(), "should find apple records");
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        let dummy = make_record(&[]);
        let results = idx.score_blocked("", &["1".to_string()], 5, &dummy, Side::B);
        assert!(results.is_empty());
    }

    #[test]
    fn query_zero_top_k() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "Apple Inc")]))],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        let dummy = make_record(&[]);
        let results = idx.score_blocked("apple", &["1".to_string()], 0, &dummy, Side::B);
        assert!(results.is_empty());
    }

    #[test]
    fn upsert_and_query() {
        let idx = SimpleBm25::new(&test_fields(), &[], Side::A);
        assert_eq!(idx.num_docs(), 0);

        let rec = make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology")]);
        idx.upsert("1", &rec);
        assert_eq!(idx.num_docs(), 1);

        let dummy = make_record(&[]);
        let results = idx.score_blocked("apple", &["1".to_string()], 5, &dummy, Side::B);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Update the record.
        let rec2 = make_record(&[("name_a", "Microsoft"), ("desc_a", "software")]);
        idx.upsert("1", &rec2);
        assert_eq!(idx.num_docs(), 1);

        let results = idx.score_blocked("microsoft", &["1".to_string()], 5, &dummy, Side::B);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Old content should no longer match.
        let results = idx.score_blocked("apple", &["1".to_string()], 5, &dummy, Side::B);
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        assert_eq!(idx.num_docs(), 2);

        idx.remove("1");
        assert_eq!(idx.num_docs(), 1);

        let dummy = make_record(&[]);
        let results = idx.score_blocked(
            "apple",
            &["1".to_string(), "2".to_string()],
            5,
            &dummy,
            Side::B,
        );
        assert!(
            results.iter().all(|(id, _)| id != "1"),
            "removed record should not match"
        );
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
        let idx = SimpleBm25::build(&store, Side::B, &fields, &[]);
        assert_eq!(idx.num_docs(), 2);

        let dummy = make_record(&[]);
        let results = idx.score_blocked(
            "apple",
            &["1".to_string(), "2".to_string()],
            5,
            &dummy,
            Side::A,
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn empty_fields_skipped() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", ""), ("desc_a", "")]))],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        assert_eq!(idx.num_docs(), 2);

        let dummy = make_record(&[]);
        let blocked = vec!["1".to_string(), "2".to_string()];
        let results = idx.score_blocked("O'Brien & Associates", &blocked, 5, &dummy, Side::B);
        assert!(!results.is_empty(), "should find O'Brien record");
    }

    #[test]
    fn analytical_self_score_basic() {
        let store = make_store(
            vec![
                (
                    "1",
                    make_record(&[("name_a", "Alpha Corp"), ("desc_a", "technology")]),
                ),
                (
                    "2",
                    make_record(&[("name_a", "Beta Holdings"), ("desc_a", "finance")]),
                ),
                (
                    "3",
                    make_record(&[("name_a", "Alpha Finance"), ("desc_a", "banking")]),
                ),
            ],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        let score = idx.analytical_self_score("Alpha Corp");
        assert!(score > 0.0, "expected positive self-score, got {score}");
    }

    #[test]
    fn analytical_self_score_empty() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "test"), ("desc_a", "data")]))],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        assert_eq!(idx.analytical_self_score(""), 0.0);
    }

    #[test]
    fn analytical_self_score_unknown_tokens() {
        let store = make_store(
            vec![(
                "1",
                make_record(&[("name_a", "apple"), ("desc_a", "fruit")]),
            )],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields(), &[]);
        let score = idx.analytical_self_score("zzzzxyzzy");
        assert_eq!(score, 0.0, "unknown tokens should give 0.0");
    }

    #[test]
    fn sanitize_query_fn() {
        assert_eq!(sanitize_query("hello world"), "hello world");
        assert_eq!(sanitize_query("a:b"), "a b");
        assert_eq!(sanitize_query("(test)"), "test");
        assert_eq!(sanitize_query("a+b-c"), "a b c");
        assert_eq!(
            sanitize_query("Guerrero, Lewis and Cole"),
            "guerrero lewis and cole"
        );
        assert_eq!(
            sanitize_query("Rodriguez-Bailey & Co."),
            "rodriguez bailey co"
        );
        assert_eq!(sanitize_query("S.A.S"), "s a s");
        assert_eq!(
            sanitize_query("123 Main St IN 46720"),
            "123 main st in 46720"
        );
        assert_eq!(sanitize_query("Portland OR 97201"), "portland or 97201");
    }

    #[test]
    fn tokenise_consistency() {
        // Verify tokenise matches sanitize_query + split behaviour
        let text = "O'Brien & Associates (UK) Ltd.";
        let sanitised = sanitize_query(text);
        let tokens = tokenise(&sanitised);
        assert_eq!(tokens, vec!["o", "brien", "associates", "uk", "ltd"]);
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
    fn score_blocked_empty_returns_empty() {
        let idx = SimpleBm25::new(&test_fields(), &[], Side::A);
        let dummy = make_record(&[]);
        let results = idx.score_blocked("apple", &[], 5, &dummy, Side::B);
        assert!(results.is_empty());
    }

    #[test]
    fn exhaustive_vs_inverted_parity() {
        // Build an index with one blocking field, force both paths, compare.
        let blocking = vec![BlockingFieldPair {
            field_a: "country_a".to_string(),
            field_b: "country_b".to_string(),
        }];
        let fields = vec![Bm25FieldPair {
            field_a: "name_a".to_string(),
            field_b: "name_b".to_string(),
        }];

        let mut idx = SimpleBm25::new(&fields, &blocking, Side::A);
        // Set a very low threshold so inverted kicks in even for small sets.
        idx.exhaustive_threshold = 2;

        // Insert some records in the same block.
        let records: Vec<(&str, Record)> = vec![
            (
                "1",
                make_record(&[("name_a", "Alpha Corp"), ("country_a", "US")]),
            ),
            (
                "2",
                make_record(&[("name_a", "Alpha Holdings"), ("country_a", "US")]),
            ),
            (
                "3",
                make_record(&[("name_a", "Beta Group"), ("country_a", "US")]),
            ),
            (
                "4",
                make_record(&[("name_a", "Gamma Partners"), ("country_a", "UK")]),
            ),
        ];
        for (id, rec) in &records {
            idx.upsert(id, rec);
        }

        let query_rec = make_record(&[("country_b", "US")]);
        let us_blocked: Vec<String> = vec!["1", "2", "3"].into_iter().map(String::from).collect();

        // Exhaustive path (blocked_ids.len() = 3 > threshold=2, but let's test
        // exhaustive explicitly by temporarily raising threshold).
        let exhaustive_results = {
            let high_threshold_idx = SimpleBm25 {
                exhaustive_threshold: 100,
                ..SimpleBm25::new(&fields, &blocking, Side::A)
            };
            for (id, rec) in &records {
                high_threshold_idx.upsert(id, rec);
            }
            high_threshold_idx.score_blocked("alpha", &us_blocked, 5, &query_rec, Side::B)
        };

        // Inverted path (threshold=2, blocked_ids.len()=3 > threshold).
        let inverted_results = idx.score_blocked("alpha", &us_blocked, 5, &query_rec, Side::B);

        // Both should return the same IDs with approximately equal scores.
        // Compare as sets — tied scores may sort differently across
        // independent index instances due to DashMap iteration order.
        assert_eq!(
            exhaustive_results.len(),
            inverted_results.len(),
            "both paths should return same number of results"
        );
        let exhaustive_map: std::collections::HashMap<&str, f32> = exhaustive_results
            .iter()
            .map(|(id, s)| (id.as_str(), *s))
            .collect();
        for (id, inv_score) in &inverted_results {
            let exh_score = exhaustive_map
                .get(id.as_str())
                .unwrap_or_else(|| panic!("ID '{id}' in inverted results but not exhaustive"));
            let diff = (exh_score - inv_score).abs();
            assert!(
                diff < 0.001,
                "scores for '{id}' should be very close: {} vs {}",
                exh_score,
                inv_score
            );
        }
    }

    #[test]
    fn blocking_key_partitioning() {
        let blocking = vec![BlockingFieldPair {
            field_a: "country_a".to_string(),
            field_b: "country_b".to_string(),
        }];
        let fields = vec![Bm25FieldPair {
            field_a: "name_a".to_string(),
            field_b: "name_b".to_string(),
        }];

        let mut idx = SimpleBm25::new(&fields, &blocking, Side::A);
        idx.exhaustive_threshold = 0; // Force inverted path always.

        idx.upsert(
            "1",
            &make_record(&[("name_a", "Alpha Corp"), ("country_a", "US")]),
        );
        idx.upsert(
            "2",
            &make_record(&[("name_a", "Alpha Holdings"), ("country_a", "UK")]),
        );

        // Query for US block — should only find doc 1.
        let query_rec = make_record(&[("country_b", "US")]);
        let blocked = vec!["1".to_string(), "2".to_string()];
        let results = idx.score_blocked("alpha", &blocked, 5, &query_rec, Side::B);

        assert_eq!(results.len(), 1, "inverted path should filter by block_key");
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn concurrent_upsert_and_query() {
        use std::sync::Arc;
        use std::thread;

        let idx = Arc::new(SimpleBm25::new(&test_fields(), &[], Side::A));
        let mut handles = Vec::new();

        // Spawn 4 writer threads.
        for t in 0..4 {
            let idx_c = Arc::clone(&idx);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let id = format!("t{t}_{i}");
                    let rec = make_record(&[("name_a", &format!("company {i} thread {t}"))]);
                    idx_c.upsert(&id, &rec);
                }
            }));
        }

        // Spawn 2 reader threads.
        for _ in 0..2 {
            let idx_c = Arc::clone(&idx);
            handles.push(thread::spawn(move || {
                let dummy = make_record(&[]);
                for _ in 0..50 {
                    let blocked: Vec<String> = (0..10).map(|i| format!("t0_{i}")).collect();
                    let _ = idx_c.score_blocked("company", &blocked, 5, &dummy, Side::B);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Verify final state is consistent.
        assert_eq!(idx.num_docs(), 400);
    }
}
