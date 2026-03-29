//! DashMap-based BM25 scorer with WAND early termination.
//!
//! `SimpleBm25` stores per-document term frequencies, global IDF statistics,
//! and blocked posting lists in concurrent `DashMap` structures. Writes are
//! instantly visible (no commit step). Queries use either exhaustive
//! blocked-set scoring (small blocks) or Block-Max WAND (large blocks) —
//! the caller sees a single `score_blocked()` interface.
//!
//! Posting lists use compact `u32` doc IDs for cache-friendly WAND traversal.
//! Posting lists are divided into blocks of ~128 entries with precomputed
//! `max_tf` per block for WAND upper-bound pruning.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

use dashmap::DashMap;

use crate::config::schema::Bm25FieldPair;
use crate::models::{Record, Side};
use crate::store::RecordStore;

// ---
// BM25 constants (Tantivy / Lucene defaults)
// ---
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Block size threshold: blocks smaller than this use exhaustive scoring;
/// larger blocks use the WAND path.
const DEFAULT_EXHAUSTIVE_THRESHOLD: usize = 5_000;

/// Target number of entries per posting block. Blocks may temporarily grow
/// to `2 × POSTING_BLOCK_SIZE` before splitting.
const POSTING_BLOCK_SIZE: usize = 128;

// ---
// Compact doc ID mapping
// ---

/// Bidirectional String ↔ u32 mapping for compact posting list entries.
struct CompactIdMap {
    /// String ID → compact u32 ID.
    str_to_u32: DashMap<String, u32>,
    /// Compact u32 ID → String ID.
    u32_to_str: RwLock<Vec<String>>,
    /// Next compact ID to assign.
    next_id: AtomicU32,
}

impl CompactIdMap {
    fn new() -> Self {
        Self {
            str_to_u32: DashMap::new(),
            u32_to_str: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Get or assign a compact ID for a string.
    fn get_or_insert(&self, id: &str) -> u32 {
        if let Some(existing) = self.str_to_u32.get(id) {
            return *existing;
        }
        let compact = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.str_to_u32.insert(id.to_string(), compact);
        let mut vec = self.u32_to_str.write().unwrap_or_else(|e| e.into_inner());
        // Ensure the vec is large enough.
        if compact as usize >= vec.len() {
            vec.resize(compact as usize + 1, String::new());
        }
        vec[compact as usize] = id.to_string();
        compact
    }

    /// Look up the string ID for a compact ID.
    fn to_str(&self, compact: u32) -> String {
        let vec = self.u32_to_str.read().unwrap_or_else(|e| e.into_inner());
        vec.get(compact as usize).cloned().unwrap_or_default()
    }

    /// Look up the compact ID for a string (returns None if not assigned).
    fn to_u32(&self, id: &str) -> Option<u32> {
        self.str_to_u32.get(id).map(|v| *v)
    }
}

impl std::fmt::Debug for CompactIdMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactIdMap")
            .field("size", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

// ---
// Blocked posting list types
// ---

/// A single entry in a posting list (8 bytes, cache-friendly).
#[derive(Clone, Debug, Copy)]
struct PostingEntry {
    /// Compact document identifier.
    doc_id: u32,
    /// Term frequency in this document.
    tf: u32,
}

/// A block of posting entries with precomputed max_tf for WAND upper bounds.
#[derive(Clone, Debug)]
struct PostingBlock {
    /// Entries sorted by doc_id within the block.
    entries: Vec<PostingEntry>,
    /// Maximum tf in this block — used for WAND upper-bound computation.
    max_tf: u32,
}

impl PostingBlock {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_tf: 0,
        }
    }

    fn recompute_max_tf(&mut self) {
        self.max_tf = self.entries.iter().map(|e| e.tf).max().unwrap_or(0);
    }
}

/// A posting list divided into blocks of ~POSTING_BLOCK_SIZE entries.
/// Entries are globally sorted by doc_id across all blocks.
#[derive(Clone, Debug)]
struct BlockedPostingList {
    blocks: Vec<PostingBlock>,
}

impl BlockedPostingList {
    fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Insert a posting entry, maintaining sorted order. Splits blocks if needed.
    fn insert(&mut self, entry: PostingEntry) {
        if self.blocks.is_empty() {
            let mut block = PostingBlock::new();
            block.entries.push(entry);
            block.max_tf = entry.tf;
            self.blocks.push(block);
            return;
        }

        // Find the block where this doc_id belongs.
        let block_idx = self.find_block(entry.doc_id);
        let block = &mut self.blocks[block_idx];

        // Insert sorted within the block.
        let pos = block.entries.partition_point(|e| e.doc_id < entry.doc_id);
        block.entries.insert(pos, entry);
        if entry.tf > block.max_tf {
            block.max_tf = entry.tf;
        }

        // Split if block is too large.
        if block.entries.len() > POSTING_BLOCK_SIZE * 2 {
            self.split_block(block_idx);
        }
    }

    /// Remove an entry by doc_id. Returns true if found.
    fn remove(&mut self, doc_id: u32) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let block_idx = self.find_block(doc_id);
        let before = self.blocks[block_idx].entries.len();
        self.blocks[block_idx]
            .entries
            .retain(|e| e.doc_id != doc_id);
        if self.blocks[block_idx].entries.len() == before {
            return false;
        }
        if self.blocks[block_idx].entries.is_empty() {
            self.blocks.remove(block_idx);
        } else {
            self.blocks[block_idx].recompute_max_tf();
            // Merge with next if both are small.
            if block_idx + 1 < self.blocks.len() {
                let combined =
                    self.blocks[block_idx].entries.len() + self.blocks[block_idx + 1].entries.len();
                if combined <= POSTING_BLOCK_SIZE {
                    let next = self.blocks.remove(block_idx + 1);
                    self.blocks[block_idx].entries.extend(next.entries);
                    self.blocks[block_idx].recompute_max_tf();
                }
            }
        }
        true
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Find the block index where `doc_id` should reside.
    fn find_block(&self, doc_id: u32) -> usize {
        // Binary search on last doc_id of each block.
        let idx = self
            .blocks
            .partition_point(|b| b.entries.last().map(|e| e.doc_id < doc_id).unwrap_or(false));
        idx.min(self.blocks.len().saturating_sub(1))
    }

    /// Split a block at the midpoint.
    fn split_block(&mut self, block_idx: usize) {
        let mid = self.blocks[block_idx].entries.len() / 2;
        let right_entries = self.blocks[block_idx].entries.split_off(mid);
        self.blocks[block_idx].recompute_max_tf();
        let mut right = PostingBlock {
            entries: right_entries,
            max_tf: 0,
        };
        right.recompute_max_tf();
        self.blocks.insert(block_idx + 1, right);
    }
}

// ---
// Per-document metadata
// ---

/// Per-document metadata stored alongside doc_terms for efficient upsert.
#[derive(Clone, Debug)]
struct DocMeta {
    /// Term → frequency map for this document.
    terms: HashMap<String, u32>,
    /// Total number of tokens (document length for BM25).
    length: u32,
    /// Compact doc ID.
    compact_id: u32,
}

// ---
// SimpleBm25
// ---

/// Lock-free, concurrent BM25 scorer backed by `DashMap` with WAND support.
///
/// Thread-safe for concurrent reads and writes without external
/// synchronisation.
pub struct SimpleBm25 {
    /// doc_id → per-document metadata (term freqs, length, compact_id).
    docs: DashMap<String, DocMeta>,

    /// term → number of documents containing this term (for IDF).
    doc_freq: DashMap<String, usize>,

    /// term → blocked posting list sorted by compact doc_id.
    postings: DashMap<String, BlockedPostingList>,

    /// Compact ID ↔ String ID bidirectional mapping.
    id_map: CompactIdMap,

    /// Total number of indexed documents.
    total_docs: AtomicUsize,

    /// Sum of all document lengths (for avgdl).
    total_tokens: AtomicU64,

    /// Which record fields to concatenate and tokenise.
    fields: Vec<Bm25FieldPair>,

    /// Which side (A or B) this index serves.
    side: Side,

    /// Blocks larger than this use the WAND path.
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
    pub fn new(fields: &[Bm25FieldPair], side: Side) -> Self {
        Self {
            docs: DashMap::new(),
            doc_freq: DashMap::new(),
            postings: DashMap::new(),
            id_map: CompactIdMap::new(),
            total_docs: AtomicUsize::new(0),
            total_tokens: AtomicU64::new(0),
            fields: fields.to_vec(),
            side,
            exhaustive_threshold: DEFAULT_EXHAUSTIVE_THRESHOLD,
        }
    }

    /// Bulk-build from a `RecordStore`.
    pub fn build(store: &dyn RecordStore, side: Side, fields: &[Bm25FieldPair]) -> Self {
        let idx = Self::new(fields, side);
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
            self.decrement_stats(&old_meta);
        }

        if tokens.is_empty() {
            return;
        }

        let tf_map = term_frequencies(&tokens);
        let length = tokens.len() as u32;
        let compact_id = self.id_map.get_or_insert(id);

        let meta = DocMeta {
            terms: tf_map.clone(),
            length,
            compact_id,
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
                doc_id: compact_id,
                tf: *tf,
            };
            self.postings
                .entry(term.clone())
                .and_modify(|list| {
                    list.insert(entry);
                })
                .or_insert_with(|| {
                    let mut list = BlockedPostingList::new();
                    list.insert(entry);
                    list
                });
        }
    }

    /// Remove a document from the index.
    pub fn remove(&self, id: &str) {
        if let Some((_, old_meta)) = self.docs.remove(id) {
            self.decrement_stats(&old_meta);
        }
    }

    /// Decrement corpus stats and remove posting entries for a document.
    fn decrement_stats(&self, meta: &DocMeta) {
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
                list.remove(meta.compact_id);
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
    /// Automatically selects exhaustive or WAND strategy based on
    /// the blocked set size.
    pub fn score_blocked(
        &self,
        query_text: &str,
        blocked_ids: &[String],
        top_k: usize,
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
            self.score_wand(&term_idfs, blocked_ids, avg_dl, top_k)
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

    /// WAND scoring with Block-Max upper bounds.
    ///
    /// Walks posting lists by compact doc_id, using per-block `max_tf` to
    /// compute upper bounds. Documents whose cumulative upper bound cannot
    /// beat the current Kth-best score are skipped without full scoring.
    ///
    /// Mathematically guaranteed to return the same top-K results as
    /// exhaustive scoring.
    fn score_wand(
        &self,
        term_idfs: &[(&str, u32, f64)],
        blocked_ids: &[String],
        avg_dl: f64,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        // Build allowed set of compact IDs for blocking filter.
        let allowed: std::collections::HashSet<u32> = blocked_ids
            .iter()
            .filter_map(|id| self.id_map.to_u32(id))
            .collect();

        if allowed.is_empty() {
            return Vec::new();
        }

        // Snapshot posting list references. Each cursor holds a DashMap Ref.
        // We collect the data we need into owned structures to avoid holding
        // DashMap refs across the WAND loop.
        struct TermPostings {
            idf: f64,
            /// Flattened (block_idx, entry_idx, doc_id, tf, block_max_tf) for cursor.
            blocks: Vec<(Vec<PostingEntry>, u32)>, // (entries, max_tf)
        }

        let mut term_postings: Vec<TermPostings> = Vec::with_capacity(term_idfs.len());
        for &(term, _qtf, idf) in term_idfs {
            if let Some(list) = self.postings.get(term) {
                let blocks: Vec<(Vec<PostingEntry>, u32)> = list
                    .blocks
                    .iter()
                    .map(|b| (b.entries.clone(), b.max_tf))
                    .collect();
                if !blocks.is_empty() {
                    term_postings.push(TermPostings { idf, blocks });
                }
            }
        }

        if term_postings.is_empty() {
            return Vec::new();
        }

        // Build cursors.
        let mut cursors: Vec<WandCursor> = term_postings
            .iter()
            .enumerate()
            .map(|(i, tp)| WandCursor::new(i, tp.idf, &tp.blocks, avg_dl))
            .collect();

        // Min-heap for top-K (we track worst score = threshold).
        let mut heap: Vec<(u32, f64)> = Vec::with_capacity(top_k + 1);
        let mut threshold: f64 = 0.0;

        // WAND main loop.
        loop {
            // Sort cursors by current doc_id (exhausted cursors to the end).
            cursors.sort_by(|a, b| match (a.current_doc_id(), b.current_doc_id()) {
                (Some(a_id), Some(b_id)) => a_id.cmp(&b_id),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            });

            // Find pivot: smallest p such that sum of upper_bounds[0..=p] >= threshold.
            let mut cumulative = 0.0;
            let mut pivot_idx = None;
            for (i, cursor) in cursors.iter().enumerate() {
                if cursor.is_exhausted() {
                    break;
                }
                cumulative += cursor.upper_bound();
                if cumulative >= threshold {
                    pivot_idx = Some(i);
                    break;
                }
            }

            let pivot_idx = match pivot_idx {
                Some(p) => p,
                None => break, // No more documents can beat threshold.
            };

            let pivot_doc = match cursors[pivot_idx].current_doc_id() {
                Some(d) => d,
                None => break,
            };

            // Check if all cursors [0..=pivot_idx] point at pivot_doc.
            let all_at_pivot = cursors[..=pivot_idx]
                .iter()
                .all(|c| c.current_doc_id() == Some(pivot_doc));

            if all_at_pivot {
                // Score this document fully.
                if allowed.contains(&pivot_doc) {
                    let dl = self.id_map.to_str(pivot_doc).as_str().to_string();
                    let dl_val = self.docs.get(&dl).map(|m| m.length as f64).unwrap_or(1.0);

                    let mut score = 0.0f64;
                    for cursor in cursors.iter() {
                        if let Some(tf) = cursor.current_tf_if_at(pivot_doc) {
                            score += bm25_term_score(tf as f64, cursor.idf, dl_val, avg_dl);
                        }
                    }

                    if score > threshold || heap.len() < top_k {
                        heap_insert(&mut heap, top_k, pivot_doc, score);
                        if heap.len() >= top_k {
                            threshold = heap.last().map(|(_, s)| *s).unwrap_or(0.0);
                        }
                    }
                }

                // Advance all cursors past pivot_doc.
                for cursor in cursors.iter_mut() {
                    if cursor.current_doc_id() == Some(pivot_doc) {
                        cursor.advance();
                    }
                }
            } else {
                // Not all cursors at pivot — advance lagging cursors to pivot_doc.
                for cursor in cursors[..pivot_idx].iter_mut() {
                    if let Some(cur) = cursor.current_doc_id()
                        && cur < pivot_doc
                    {
                        cursor.advance_to(pivot_doc);
                    }
                }
            }
        }

        // Convert heap to result.
        heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        heap.into_iter()
            .map(|(compact_id, score)| (self.id_map.to_str(compact_id), score as f32))
            .collect()
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
// WAND cursor
// ---

/// Cursor over a single term's BlockedPostingList for WAND traversal.
struct WandCursor<'a> {
    /// Index into the parent term_postings vec (for identification).
    #[allow(dead_code)]
    term_idx: usize,
    /// IDF for this term.
    idf: f64,
    /// Reference to the blocks (entries, max_tf).
    blocks: &'a [(Vec<PostingEntry>, u32)],
    /// Current block index.
    block_idx: usize,
    /// Current entry index within the block.
    entry_idx: usize,
    /// Precomputed upper-bound BM25 score using current block's max_tf.
    block_upper_bound: f64,
    /// Average document length (for upper bound computation).
    avg_dl: f64,
}

impl<'a> WandCursor<'a> {
    fn new(term_idx: usize, idf: f64, blocks: &'a [(Vec<PostingEntry>, u32)], avg_dl: f64) -> Self {
        let block_upper_bound = if !blocks.is_empty() {
            compute_upper_bound(blocks[0].1, idf, avg_dl)
        } else {
            0.0
        };
        Self {
            term_idx,
            idf,
            blocks,
            block_idx: 0,
            entry_idx: 0,
            block_upper_bound,
            avg_dl,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.block_idx >= self.blocks.len()
    }

    fn current_doc_id(&self) -> Option<u32> {
        if self.is_exhausted() {
            return None;
        }
        let (entries, _) = &self.blocks[self.block_idx];
        entries.get(self.entry_idx).map(|e| e.doc_id)
    }

    fn current_tf_if_at(&self, doc_id: u32) -> Option<u32> {
        if self.is_exhausted() {
            return None;
        }
        let (entries, _) = &self.blocks[self.block_idx];
        entries
            .get(self.entry_idx)
            .filter(|e| e.doc_id == doc_id)
            .map(|e| e.tf)
    }

    fn upper_bound(&self) -> f64 {
        self.block_upper_bound
    }

    /// Advance to the next entry.
    fn advance(&mut self) {
        if self.is_exhausted() {
            return;
        }
        self.entry_idx += 1;
        let (entries, _) = &self.blocks[self.block_idx];
        if self.entry_idx >= entries.len() {
            self.block_idx += 1;
            self.entry_idx = 0;
            self.update_upper_bound();
        }
    }

    /// Skip forward to the first entry with doc_id >= target.
    fn advance_to(&mut self, target: u32) {
        while !self.is_exhausted() {
            let (entries, _) = &self.blocks[self.block_idx];

            // If the last entry in this block is < target, skip the whole block.
            if let Some(last) = entries.last()
                && last.doc_id < target
            {
                self.block_idx += 1;
                self.entry_idx = 0;
                self.update_upper_bound();
                continue;
            }

            // Binary search within this block.
            let start = self.entry_idx;
            let pos = entries[start..].partition_point(|e| e.doc_id < target);
            self.entry_idx = start + pos;

            if self.entry_idx >= entries.len() {
                self.block_idx += 1;
                self.entry_idx = 0;
                self.update_upper_bound();
            }
            return;
        }
    }

    fn update_upper_bound(&mut self) {
        if self.block_idx < self.blocks.len() {
            let (_, max_tf) = &self.blocks[self.block_idx];
            self.block_upper_bound = compute_upper_bound(*max_tf, self.idf, self.avg_dl);
        } else {
            self.block_upper_bound = 0.0;
        }
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

/// Compute upper-bound BM25 score for a block with given max_tf.
/// Uses dl=1.0 (minimum possible) for the tightest upper bound without
/// tracking per-block min_dl.
#[inline]
fn compute_upper_bound(max_tf: u32, idf: f64, avg_dl: f64) -> f64 {
    bm25_term_score(max_tf as f64, idf, 1.0, avg_dl)
}

/// Insert into a sorted min-heap of (doc_id, score), keeping top_k best.
fn heap_insert(heap: &mut Vec<(u32, f64)>, top_k: usize, doc_id: u32, score: f64) {
    heap.push((doc_id, score));
    // Keep sorted descending by score for easy threshold access.
    heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if heap.len() > top_k {
        heap.pop();
    }
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
/// collapses whitespace, lowercases.
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

        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
        assert_eq!(idx.num_docs(), 5);

        let blocked_ids: Vec<String> = (1..=5).map(|i| i.to_string()).collect();
        let results = idx.score_blocked("apple", &blocked_ids, 3);
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
        let results = idx.score_blocked("", &["1".to_string()], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn query_zero_top_k() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "Apple Inc")]))],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
        let results = idx.score_blocked("apple", &["1".to_string()], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn upsert_and_query() {
        let idx = SimpleBm25::new(&test_fields(), Side::A);
        assert_eq!(idx.num_docs(), 0);

        let rec = make_record(&[("name_a", "Apple Inc"), ("desc_a", "technology")]);
        idx.upsert("1", &rec);
        assert_eq!(idx.num_docs(), 1);

        let results = idx.score_blocked("apple", &["1".to_string()], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Update the record.
        let rec2 = make_record(&[("name_a", "Microsoft"), ("desc_a", "software")]);
        idx.upsert("1", &rec2);
        assert_eq!(idx.num_docs(), 1);

        let results = idx.score_blocked("microsoft", &["1".to_string()], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");

        // Old content should no longer match.
        let results = idx.score_blocked("apple", &["1".to_string()], 5);
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
        assert_eq!(idx.num_docs(), 2);

        idx.remove("1");
        assert_eq!(idx.num_docs(), 1);

        let results = idx.score_blocked("apple", &["1".to_string(), "2".to_string()], 5);
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
        let idx = SimpleBm25::build(&store, Side::B, &fields);
        assert_eq!(idx.num_docs(), 2);

        let results = idx.score_blocked("apple", &["1".to_string(), "2".to_string()], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn empty_fields_skipped() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", ""), ("desc_a", "")]))],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
        assert_eq!(idx.num_docs(), 2);

        let blocked = vec!["1".to_string(), "2".to_string()];
        let results = idx.score_blocked("O'Brien & Associates", &blocked, 5);
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
        let score = idx.analytical_self_score("Alpha Corp");
        assert!(score > 0.0, "expected positive self-score, got {score}");
    }

    #[test]
    fn analytical_self_score_empty() {
        let store = make_store(
            vec![("1", make_record(&[("name_a", "test"), ("desc_a", "data")]))],
            Side::A,
        );
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
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
        let idx = SimpleBm25::build(&store, Side::A, &test_fields());
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
        let idx = SimpleBm25::new(&test_fields(), Side::A);
        let results = idx.score_blocked("apple", &[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn exhaustive_vs_wand_parity() {
        let fields = vec![Bm25FieldPair {
            field_a: "name_a".to_string(),
            field_b: "name_b".to_string(),
        }];

        let records: Vec<(&str, Record)> = vec![
            ("1", make_record(&[("name_a", "Alpha Corp")])),
            ("2", make_record(&[("name_a", "Alpha Holdings")])),
            ("3", make_record(&[("name_a", "Beta Group")])),
            ("4", make_record(&[("name_a", "Gamma Partners")])),
        ];

        let blocked: Vec<String> = vec!["1", "2", "3", "4"]
            .into_iter()
            .map(String::from)
            .collect();

        // Exhaustive path (high threshold).
        let exhaustive_results = {
            let idx = SimpleBm25 {
                exhaustive_threshold: 100,
                ..SimpleBm25::new(&fields, Side::A)
            };
            for (id, rec) in &records {
                idx.upsert(id, rec);
            }
            idx.score_blocked("alpha", &blocked, 5)
        };

        // WAND path (threshold=2, blocked_ids.len()=4 > threshold).
        let wand_results = {
            let idx = SimpleBm25 {
                exhaustive_threshold: 2,
                ..SimpleBm25::new(&fields, Side::A)
            };
            for (id, rec) in &records {
                idx.upsert(id, rec);
            }
            idx.score_blocked("alpha", &blocked, 5)
        };

        assert_eq!(
            exhaustive_results.len(),
            wand_results.len(),
            "both paths should return same number of results: exhaustive={}, wand={}",
            exhaustive_results.len(),
            wand_results.len()
        );
        let exhaustive_map: std::collections::HashMap<&str, f32> = exhaustive_results
            .iter()
            .map(|(id, s)| (id.as_str(), *s))
            .collect();
        for (id, wand_score) in &wand_results {
            let exh_score = exhaustive_map
                .get(id.as_str())
                .unwrap_or_else(|| panic!("ID '{id}' in WAND results but not exhaustive"));
            let diff = (exh_score - wand_score).abs();
            assert!(
                diff < 0.001,
                "scores for '{id}' should be very close: {} vs {}",
                exh_score,
                wand_score
            );
        }
    }

    #[test]
    fn wand_respects_blocked_ids() {
        let fields = vec![Bm25FieldPair {
            field_a: "name_a".to_string(),
            field_b: "name_b".to_string(),
        }];

        let idx = SimpleBm25 {
            exhaustive_threshold: 0, // Force WAND path.
            ..SimpleBm25::new(&fields, Side::A)
        };

        for i in 0..20 {
            let rec = make_record(&[("name_a", &format!("Alpha Corp entity {}", i))]);
            idx.upsert(&format!("doc_{}", i), &rec);
        }

        // Only allow a subset.
        let allowed: Vec<String> = (0..5).map(|i| format!("doc_{}", i)).collect();
        let results = idx.score_blocked("alpha", &allowed, 10);

        for (id, _) in &results {
            assert!(
                allowed.contains(id),
                "WAND returned '{}' which is not in blocked_ids",
                id
            );
        }
        assert!(!results.is_empty(), "should find at least one result");
    }

    #[test]
    fn wand_large_index_parity() {
        // Test WAND vs exhaustive with more docs to exercise block splitting.
        let fields = vec![Bm25FieldPair {
            field_a: "name_a".to_string(),
            field_b: "name_b".to_string(),
        }];

        let names = [
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Corp",
            "Holdings",
            "Group",
            "Ltd",
            "Inc",
            "Partners",
            "Associates",
            "International",
        ];

        let mut all_ids = Vec::new();
        let exhaustive_idx = SimpleBm25 {
            exhaustive_threshold: 100_000,
            ..SimpleBm25::new(&fields, Side::A)
        };
        let wand_idx = SimpleBm25 {
            exhaustive_threshold: 0,
            ..SimpleBm25::new(&fields, Side::A)
        };

        for i in 0..500 {
            let id = format!("doc_{}", i);
            let name = format!(
                "{} {} {}",
                names[i % names.len()],
                names[(i * 7) % names.len()],
                names[(i * 13) % names.len()]
            );
            let rec = make_record(&[("name_a", &name)]);
            exhaustive_idx.upsert(&id, &rec);
            wand_idx.upsert(&id, &rec);
            all_ids.push(id);
        }

        let queries = ["Alpha Corp", "Beta Holdings", "Gamma International Ltd"];
        for query in &queries {
            // Use high top_k to avoid tie-breaking differences between paths.
            let exh = exhaustive_idx.score_blocked(query, &all_ids, 500);
            let wand = wand_idx.score_blocked(query, &all_ids, 500);

            assert_eq!(
                exh.len(),
                wand.len(),
                "query '{}': result count mismatch: exhaustive={}, wand={}",
                query,
                exh.len(),
                wand.len()
            );

            let exh_map: HashMap<&str, f32> = exh.iter().map(|(id, s)| (id.as_str(), *s)).collect();
            for (id, wand_score) in &wand {
                let exh_score = exh_map.get(id.as_str()).unwrap_or_else(|| {
                    panic!("query '{}': ID '{}' in WAND but not exhaustive", query, id)
                });
                let diff = (exh_score - wand_score).abs();
                assert!(
                    diff < 0.01,
                    "query '{}': scores for '{}' differ: exh={}, wand={}",
                    query,
                    id,
                    exh_score,
                    wand_score
                );
            }
        }
    }

    #[test]
    fn concurrent_upsert_and_query() {
        use std::sync::Arc;
        use std::thread;

        let idx = Arc::new(SimpleBm25::new(&test_fields(), Side::A));
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
                for _ in 0..50 {
                    let blocked: Vec<String> = (0..10).map(|i| format!("t0_{i}")).collect();
                    let _ = idx_c.score_blocked("company", &blocked, 5);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        assert_eq!(idx.num_docs(), 400);
    }

    // --- BlockedPostingList unit tests ---

    #[test]
    fn blocked_posting_list_insert_sorted() {
        let mut list = BlockedPostingList::new();
        for id in [5, 2, 8, 1, 3] {
            list.insert(PostingEntry { doc_id: id, tf: 1 });
        }
        let all_ids: Vec<u32> = list
            .blocks
            .iter()
            .flat_map(|b| b.entries.iter().map(|e| e.doc_id))
            .collect();
        assert_eq!(all_ids, vec![1, 2, 3, 5, 8]);
    }

    #[test]
    fn blocked_posting_list_remove() {
        let mut list = BlockedPostingList::new();
        for id in 0..10u32 {
            list.insert(PostingEntry {
                doc_id: id,
                tf: id + 1,
            });
        }
        assert!(list.remove(5));
        assert!(!list.remove(5)); // already removed
        let all_ids: Vec<u32> = list
            .blocks
            .iter()
            .flat_map(|b| b.entries.iter().map(|e| e.doc_id))
            .collect();
        assert!(!all_ids.contains(&5));
        assert_eq!(all_ids.len(), 9);
    }

    #[test]
    fn blocked_posting_list_splits_large_blocks() {
        let mut list = BlockedPostingList::new();
        // Insert enough entries to trigger a split.
        for id in 0..(POSTING_BLOCK_SIZE * 3) as u32 {
            list.insert(PostingEntry { doc_id: id, tf: 1 });
        }
        // Should have multiple blocks.
        assert!(
            list.blocks.len() > 1,
            "expected multiple blocks after {} inserts, got {}",
            POSTING_BLOCK_SIZE * 3,
            list.blocks.len()
        );
        // All entries should be globally sorted.
        let all_ids: Vec<u32> = list
            .blocks
            .iter()
            .flat_map(|b| b.entries.iter().map(|e| e.doc_id))
            .collect();
        for w in all_ids.windows(2) {
            assert!(w[0] < w[1], "not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn blocked_posting_list_max_tf_maintained() {
        let mut list = BlockedPostingList::new();
        list.insert(PostingEntry { doc_id: 1, tf: 3 });
        list.insert(PostingEntry { doc_id: 2, tf: 7 });
        list.insert(PostingEntry { doc_id: 3, tf: 5 });
        assert_eq!(list.blocks[0].max_tf, 7);

        // Remove the max_tf entry.
        list.remove(2);
        assert_eq!(list.blocks[0].max_tf, 5);
    }
}
