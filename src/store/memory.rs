//! In-memory record store backed by DashMap, DashSet, and BlockingIndex.
//!
//! Used by both batch mode (`meld run`) and as the default for live mode
//! before SQLite migration.

use std::collections::HashMap;
use std::sync::RwLock;

use dashmap::{DashMap, DashSet};

use crate::config::BlockingConfig;
use crate::matching::blocking::BlockingIndex;
use crate::models::{Record, Side};

use super::RecordStore;

/// In-memory record store wrapping the same data structures previously
/// spread across `LiveSideState`, `LiveMatchState`, and `MatchState`.
pub struct MemoryStore {
    a_records: DashMap<String, Record>,
    b_records: DashMap<String, Record>,
    a_blocking: RwLock<BlockingIndex>,
    b_blocking: RwLock<BlockingIndex>,
    a_unmatched: DashSet<String>,
    b_unmatched: DashSet<String>,
    a_common_ids: DashMap<String, String>,
    b_common_ids: DashMap<String, String>,
    /// Exact prefilter index: composite key → record ID.
    /// Built on demand by `build_exact_index()`. Key is field values joined
    /// by `\0` (null byte separator, safe since field values are plain text).
    a_exact: DashMap<String, String>,
    b_exact: DashMap<String, String>,
}

impl MemoryStore {
    /// Create a new empty store with blocking indices configured from the
    /// blocking config.
    pub fn new(blocking_config: &BlockingConfig) -> Self {
        Self {
            a_records: DashMap::new(),
            b_records: DashMap::new(),
            a_blocking: RwLock::new(BlockingIndex::from_config(blocking_config)),
            b_blocking: RwLock::new(BlockingIndex::from_config(blocking_config)),
            a_unmatched: DashSet::new(),
            b_unmatched: DashSet::new(),
            a_common_ids: DashMap::new(),
            b_common_ids: DashMap::new(),
            a_exact: DashMap::new(),
            b_exact: DashMap::new(),
        }
    }

    /// Load records from HashMaps and build blocking indices.
    ///
    /// Replaces the scattered init code previously in `state.rs` and `live.rs`.
    pub fn from_records(
        records_a: HashMap<String, Record>,
        records_b: HashMap<String, Record>,
        blocking_config: &BlockingConfig,
    ) -> Self {
        let store = Self::new(blocking_config);

        // Load A records and build blocking index
        {
            let mut bi = store.a_blocking.write().unwrap_or_else(|e| e.into_inner());
            for (id, record) in records_a {
                bi.insert(&id, &record, Side::A);
                store.a_records.insert(id, record);
            }
        }

        // Load B records and build blocking index
        {
            let mut bi = store.b_blocking.write().unwrap_or_else(|e| e.into_inner());
            for (id, record) in records_b {
                bi.insert(&id, &record, Side::B);
                store.b_records.insert(id, record);
            }
        }

        store
    }

    // --- Side accessors (for internal use and migration) ---

    fn records(&self, side: Side) -> &DashMap<String, Record> {
        match side {
            Side::A => &self.a_records,
            Side::B => &self.b_records,
        }
    }

    fn blocking(&self, side: Side) -> &RwLock<BlockingIndex> {
        match side {
            Side::A => &self.a_blocking,
            Side::B => &self.b_blocking,
        }
    }

    fn unmatched(&self, side: Side) -> &DashSet<String> {
        match side {
            Side::A => &self.a_unmatched,
            Side::B => &self.b_unmatched,
        }
    }

    fn common_ids(&self, side: Side) -> &DashMap<String, String> {
        match side {
            Side::A => &self.a_common_ids,
            Side::B => &self.b_common_ids,
        }
    }
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStore")
            .field("a_records", &self.a_records.len())
            .field("b_records", &self.b_records.len())
            .field("a_unmatched", &self.a_unmatched.len())
            .field("b_unmatched", &self.b_unmatched.len())
            .finish()
    }
}

impl RecordStore for MemoryStore {
    // --- Records ---

    fn get(&self, side: Side, id: &str) -> Option<Record> {
        self.records(side).get(id).map(|r| r.value().clone())
    }

    fn insert(&self, side: Side, id: &str, record: &Record) -> Option<Record> {
        self.records(side).insert(id.to_string(), record.clone())
    }

    fn remove(&self, side: Side, id: &str) -> Option<Record> {
        self.records(side).remove(id).map(|(_, v)| v)
    }

    fn contains(&self, side: Side, id: &str) -> bool {
        self.records(side).contains_key(id)
    }

    fn len(&self, side: Side) -> usize {
        self.records(side).len()
    }

    fn ids(&self, side: Side) -> Vec<String> {
        self.records(side).iter().map(|e| e.key().clone()).collect()
    }

    fn get_many_fields(
        &self,
        side: Side,
        ids: &[String],
        _fields: &[String],
    ) -> Vec<(String, Record)> {
        // In-memory: no deserialization cost, return full records.
        self.get_many(side, ids)
    }

    fn for_each_record(&self, side: Side, f: &mut dyn FnMut(&str, &Record)) {
        for entry in self.records(side).iter() {
            f(entry.key(), entry.value());
        }
    }

    // --- Blocking ---

    fn blocking_insert(&self, side: Side, id: &str, record: &Record) {
        let mut bi = self
            .blocking(side)
            .write()
            .unwrap_or_else(|e| e.into_inner());
        bi.insert(id, record, side);
    }

    fn blocking_remove(&self, side: Side, id: &str, record: &Record) {
        let mut bi = self
            .blocking(side)
            .write()
            .unwrap_or_else(|e| e.into_inner());
        bi.remove(id, record, side);
    }

    fn blocking_query(&self, record: &Record, query_side: Side) -> Vec<String> {
        let bi = self
            .blocking(query_side.opposite())
            .read()
            .unwrap_or_else(|e| e.into_inner());
        bi.query(record, query_side).into_iter().collect()
    }

    // --- Unmatched ---

    fn mark_unmatched(&self, side: Side, id: &str) {
        self.unmatched(side).insert(id.to_string());
    }

    fn mark_matched(&self, side: Side, id: &str) {
        self.unmatched(side).remove(id);
    }

    fn is_unmatched(&self, side: Side, id: &str) -> bool {
        self.unmatched(side).contains(id)
    }

    fn unmatched_count(&self, side: Side) -> usize {
        self.unmatched(side).len()
    }

    fn unmatched_ids(&self, side: Side) -> Vec<String> {
        let mut ids: Vec<String> = self
            .unmatched(side)
            .iter()
            .map(|r| r.key().clone())
            .collect();
        ids.sort();
        ids
    }

    // --- Common ID index ---

    fn common_id_insert(&self, side: Side, common_id: &str, record_id: &str) {
        self.common_ids(side)
            .insert(common_id.to_string(), record_id.to_string());
    }

    fn common_id_lookup(&self, side: Side, common_id: &str) -> Option<String> {
        self.common_ids(side)
            .get(common_id)
            .map(|e| e.value().clone())
    }

    fn common_id_remove(&self, side: Side, common_id: &str) {
        self.common_ids(side).remove(common_id);
    }

    // --- Exact prefilter ---

    fn build_exact_index(&self, side: Side, field_names: &[String]) {
        let exact = match side {
            Side::A => &self.a_exact,
            Side::B => &self.b_exact,
        };
        exact.clear();
        self.records(side).iter().for_each(|entry| {
            let id = entry.key();
            let rec = entry.value();
            let key = make_exact_key(rec, field_names);
            if !key.is_empty() {
                exact.insert(key, id.clone());
            }
        });
    }

    fn exact_lookup(&self, side: Side, kvs: &[(String, String)]) -> Option<String> {
        if kvs.is_empty() {
            return None;
        }
        // Build composite key from query values — same separator and lowercasing
        // as make_exact_key used when building the index.
        let mut parts = Vec::with_capacity(kvs.len());
        for (_, v) in kvs {
            let v = v.trim();
            if v.is_empty() {
                return None; // AND: any empty value → no match
            }
            parts.push(v.to_lowercase());
        }
        let key = parts.join("\0");
        let exact = match side {
            Side::A => &self.a_exact,
            Side::B => &self.b_exact,
        };
        exact.get(&key).map(|e| e.value().clone())
    }

    // --- Review persistence (no-op for memory — DashMap is source of truth) ---

    fn persist_review(&self, _key: &str, _id: &str, _side: Side, _candidate_id: &str, _score: f64) {
    }

    fn remove_reviews_for_id(&self, _id: &str) {}

    fn remove_reviews_for_pair(&self, _a_id: &str, _b_id: &str) {}

    fn load_reviews(&self) -> Vec<(String, String, Side, String, f64)> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Exact prefilter helpers
// ---------------------------------------------------------------------------

/// Build a composite key from a record's field values for exact prefilter indexing.
///
/// Fields are joined with `\0` (null byte). Returns an empty string if any
/// field value is missing or empty — those records are not indexed.
pub(crate) fn make_exact_key(record: &Record, field_names: &[String]) -> String {
    let mut parts = Vec::with_capacity(field_names.len());
    for name in field_names {
        let val = record.get(name).map(|v| v.trim()).unwrap_or("");
        if val.is_empty() {
            return String::new(); // AND semantics: all fields must be present
        }
        parts.push(val.to_lowercase());
    }
    parts.join("\0")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BlockingConfig, BlockingFieldPair};

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_blocking_config() -> BlockingConfig {
        BlockingConfig {
            enabled: true,
            operator: "and".to_string(),
            fields: vec![BlockingFieldPair {
                field_a: "country".to_string(),
                field_b: "domicile".to_string(),
            }],
            field_a: None,
            field_b: None,
        }
    }

    fn make_store() -> MemoryStore {
        MemoryStore::new(&make_blocking_config())
    }

    // --- Record CRUD ---

    #[test]
    fn insert_and_get() {
        let store = make_store();
        let rec = make_record(&[("id", "1"), ("name", "Foo")]);
        assert!(store.insert(Side::A, "1", &rec).is_none());
        let got = store.get(Side::A, "1").unwrap();
        assert_eq!(got.get("name").unwrap(), "Foo");
    }

    #[test]
    fn insert_replaces_and_returns_old() {
        let store = make_store();
        let rec1 = make_record(&[("id", "1"), ("name", "Foo")]);
        let rec2 = make_record(&[("id", "1"), ("name", "Bar")]);
        store.insert(Side::A, "1", &rec1);
        let old = store.insert(Side::A, "1", &rec2).unwrap();
        assert_eq!(old.get("name").unwrap(), "Foo");
        assert_eq!(store.get(Side::A, "1").unwrap().get("name").unwrap(), "Bar");
    }

    #[test]
    fn remove_returns_old() {
        let store = make_store();
        let rec = make_record(&[("id", "1"), ("name", "Foo")]);
        store.insert(Side::B, "1", &rec);
        let removed = store.remove(Side::B, "1").unwrap();
        assert_eq!(removed.get("name").unwrap(), "Foo");
        assert!(!store.contains(Side::B, "1"));
    }

    #[test]
    fn contains_and_len() {
        let store = make_store();
        assert!(!store.contains(Side::A, "x"));
        assert_eq!(store.len(Side::A), 0);
        store.insert(Side::A, "x", &make_record(&[("id", "x")]));
        assert!(store.contains(Side::A, "x"));
        assert_eq!(store.len(Side::A), 1);
    }

    #[test]
    fn ids_returns_all() {
        let store = make_store();
        store.insert(Side::A, "a", &make_record(&[("id", "a")]));
        store.insert(Side::A, "b", &make_record(&[("id", "b")]));
        let mut ids = store.ids(Side::A);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn sides_are_independent() {
        let store = make_store();
        store.insert(Side::A, "1", &make_record(&[("id", "1")]));
        assert!(store.contains(Side::A, "1"));
        assert!(!store.contains(Side::B, "1"));
    }

    // --- Blocking ---

    #[test]
    fn blocking_insert_query_remove() {
        let store = make_store();
        let rec_gb = make_record(&[("country", "GB")]);
        let rec_us = make_record(&[("country", "US")]);
        store.blocking_insert(Side::A, "a1", &rec_gb);
        store.blocking_insert(Side::A, "a2", &rec_gb);
        store.blocking_insert(Side::A, "a3", &rec_us);

        // Query from B side looking for GB records in A
        let query = make_record(&[("domicile", "GB")]);
        let hits = store.blocking_query(&query, Side::B);
        assert_eq!(hits.len(), 2, "expected 2 GB hits, got {:?}", hits);

        // Remove one and re-query
        store.blocking_remove(Side::A, "a1", &rec_gb);
        let hits = store.blocking_query(&query, Side::B);
        assert_eq!(hits.len(), 1, "expected 1 GB hit after remove");
    }

    // --- Unmatched ---

    #[test]
    fn unmatched_lifecycle() {
        let store = make_store();
        store.mark_unmatched(Side::A, "x");
        assert!(store.is_unmatched(Side::A, "x"));
        assert_eq!(store.unmatched_count(Side::A), 1);
        store.mark_matched(Side::A, "x");
        assert!(!store.is_unmatched(Side::A, "x"));
        assert_eq!(store.unmatched_count(Side::A), 0);
    }

    #[test]
    fn unmatched_ids_sorted() {
        let store = make_store();
        store.mark_unmatched(Side::B, "c");
        store.mark_unmatched(Side::B, "a");
        store.mark_unmatched(Side::B, "b");
        assert_eq!(store.unmatched_ids(Side::B), vec!["a", "b", "c"]);
    }

    // --- Common ID ---

    #[test]
    fn common_id_lifecycle() {
        let store = make_store();
        store.common_id_insert(Side::A, "LEI-123", "a1");
        assert_eq!(
            store.common_id_lookup(Side::A, "LEI-123"),
            Some("a1".to_string())
        );
        store.common_id_remove(Side::A, "LEI-123");
        assert!(store.common_id_lookup(Side::A, "LEI-123").is_none());
    }

    // --- from_records ---

    #[test]
    fn from_records_populates_all() {
        let mut a = HashMap::new();
        a.insert(
            "a1".to_string(),
            make_record(&[("id", "a1"), ("country", "GB")]),
        );
        a.insert(
            "a2".to_string(),
            make_record(&[("id", "a2"), ("country", "US")]),
        );
        let mut b = HashMap::new();
        b.insert(
            "b1".to_string(),
            make_record(&[("id", "b1"), ("domicile", "GB")]),
        );

        let store = MemoryStore::from_records(a, b, &make_blocking_config());
        assert_eq!(store.len(Side::A), 2);
        assert_eq!(store.len(Side::B), 1);

        // Blocking should work: query from B for GB records in A
        let query = make_record(&[("domicile", "GB")]);
        let hits = store.blocking_query(&query, Side::B);
        assert_eq!(hits.len(), 1, "expected 1 GB hit from A-side blocking");
    }
}
