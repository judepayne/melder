//! In-memory record store backed by DashMap, DashSet, and BlockingIndex.
//!
//! Used by both batch mode (`meld run`) and as the default for live mode
//! before SQLite migration.

use std::collections::HashMap;
use std::sync::RwLock;

use dashmap::{DashMap, DashSet};

use crate::config::BlockingConfig;
use crate::error::StoreError;
use crate::matching::blocking::BlockingIndex;
use crate::models::{Record, Side};

use super::RecordStore;

/// Independent in-memory collection without paired matching state.
/// Enrollment uses one set of blocking fields; paired jobs retain their directional mappings.
pub struct MemoryCollection {
    records: DashMap<String, Record>,
    blocking: RwLock<BlockingIndex>,
    common_ids: DashMap<String, String>,
    exact: DashMap<String, String>,
}
impl MemoryCollection {
    /// Construct an empty single collection.
    pub fn new(config: &BlockingConfig) -> Self {
        Self::with_index(BlockingIndex::from_config(config))
    }

    fn with_index(index: BlockingIndex) -> Self {
        Self {
            records: DashMap::new(),
            blocking: RwLock::new(index),
            common_ids: DashMap::new(),
            exact: DashMap::new(),
        }
    }
    /// Load records and build their blocking index.
    pub fn from_records(records: HashMap<String, Record>, config: &BlockingConfig) -> Self {
        Self::new(config).populate(records)
    }

    pub(crate) fn from_enroll_records(
        records: HashMap<String, Record>,
        config: &crate::config::enroll_schema::EnrollBlockingConfig,
    ) -> Self {
        Self::with_index(BlockingIndex::single(
            config.fields.iter().map(|f| f.field.clone()).collect(),
        ))
        .populate(records)
    }

    fn populate(self, records: HashMap<String, Record>) -> Self {
        {
            let mut index = self.blocking.write().unwrap_or_else(|e| e.into_inner());
            for (id, record) in records {
                index.insert(&id, &record, Side::A);
                self.records.insert(id, record);
            }
        }
        self
    }
}
impl super::RecordCollection for MemoryCollection {
    fn get(&self, id: &str) -> Result<Option<Record>, StoreError> {
        Ok(self.records.get(id).map(|r| r.value().clone()))
    }
    fn insert(&self, id: &str, record: &Record) -> Result<Option<Record>, StoreError> {
        Ok(self.records.insert(id.to_string(), record.clone()))
    }
    fn remove(&self, id: &str) -> Result<Option<Record>, StoreError> {
        Ok(self.records.remove(id).map(|(_, v)| v))
    }
    fn contains(&self, id: &str) -> Result<bool, StoreError> {
        Ok(self.records.contains_key(id))
    }
    fn len(&self) -> Result<usize, StoreError> {
        Ok(self.records.len())
    }
    fn ids(&self) -> Result<Vec<String>, StoreError> {
        Ok(self.records.iter().map(|e| e.key().clone()).collect())
    }
    fn for_each_record(&self, f: &mut dyn FnMut(&str, &Record)) -> Result<(), StoreError> {
        for entry in self.records.iter() {
            f(entry.key(), entry.value());
        }
        Ok(())
    }
    fn get_many(&self, ids: &[String]) -> Result<Vec<(String, Record)>, StoreError> {
        Ok(ids
            .iter()
            .filter_map(|id| {
                self.records
                    .get(id)
                    .map(|r| (id.clone(), r.value().clone()))
            })
            .collect())
    }
    fn get_many_fields(
        &self,
        ids: &[String],
        _fields: &[String],
    ) -> Result<Vec<(String, Record)>, StoreError> {
        // In-memory: no deserialization cost, return full records.
        self.get_many(ids)
    }
    fn blocking_insert(&self, id: &str, record: &Record) -> Result<(), StoreError> {
        let mut bi = self.blocking.write().unwrap_or_else(|e| e.into_inner());
        bi.insert(id, record, Side::A);
        Ok(())
    }
    fn blocking_remove(&self, id: &str, record: &Record) -> Result<(), StoreError> {
        let mut bi = self.blocking.write().unwrap_or_else(|e| e.into_inner());
        bi.remove(id, record, Side::A);
        Ok(())
    }
    fn blocking_query(&self, record: &Record) -> Result<Vec<String>, StoreError> {
        let bi = self.blocking.read().unwrap_or_else(|e| e.into_inner());
        Ok(bi.query(record, Side::A).into_iter().collect())
    }
    fn build_exact_index(&self, field_names: &[String]) -> Result<(), StoreError> {
        let exact = &self.exact;
        exact.clear();
        self.records.iter().for_each(|entry| {
            let id = entry.key();
            let rec = entry.value();
            let key = make_exact_key(rec, field_names);
            if !key.is_empty() {
                exact.insert(key, id.clone());
            }
        });
        Ok(())
    }
    fn exact_lookup(&self, kvs: &[(String, String)]) -> Result<Option<String>, StoreError> {
        if kvs.is_empty() {
            return Ok(None);
        }
        // Build composite key from query values — same separator and lowercasing
        // as make_exact_key used when building the index.
        let mut parts = Vec::with_capacity(kvs.len());
        for (_, v) in kvs {
            let v = v.trim();
            if v.is_empty() {
                return Ok(None); // AND: any empty value → no match
            }
            parts.push(v.to_lowercase());
        }
        let key = parts.join("\0");
        let exact = &self.exact;
        Ok(exact.get(&key).map(|e| e.value().clone()))
    }
    fn common_id_insert(&self, common_id: &str, record_id: &str) -> Result<(), StoreError> {
        self.common_ids
            .insert(common_id.to_string(), record_id.to_string());
        Ok(())
    }
    fn common_id_lookup(&self, common_id: &str) -> Result<Option<String>, StoreError> {
        Ok(self.common_ids.get(common_id).map(|e| e.value().clone()))
    }
    fn common_id_remove(&self, common_id: &str) -> Result<(), StoreError> {
        self.common_ids.remove(common_id);
        Ok(())
    }
    fn common_id_remove_owned(&self, common_id: &str, record_id: &str) -> Result<(), StoreError> {
        self.common_ids
            .remove_if(common_id, |_, owner| owner == record_id);
        Ok(())
    }
}

/// In-memory record store wrapping the same data structures previously
/// spread across `LiveSideState`, `LiveMatchState`, and `MatchState`.
pub struct MemoryStore {
    a: MemoryCollection,
    b: MemoryCollection,
    a_unmatched: DashSet<String>,
    b_unmatched: DashSet<String>,
}

impl MemoryStore {
    /// Create a new empty store with blocking indices configured from the
    /// blocking config.
    pub fn new(blocking_config: &BlockingConfig) -> Self {
        Self {
            a: MemoryCollection::new(blocking_config),
            b: MemoryCollection::new(blocking_config),
            a_unmatched: DashSet::new(),
            b_unmatched: DashSet::new(),
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
            let mut bi = store.a.blocking.write().unwrap_or_else(|e| e.into_inner());
            for (id, record) in records_a {
                bi.insert(&id, &record, Side::A);
                store.a.records.insert(id, record);
            }
        }

        // Load B records and build blocking index
        {
            let mut bi = store.b.blocking.write().unwrap_or_else(|e| e.into_inner());
            for (id, record) in records_b {
                bi.insert(&id, &record, Side::B);
                store.b.records.insert(id, record);
            }
        }

        store
    }

    fn collection(&self, side: Side) -> &MemoryCollection {
        match side {
            Side::A => &self.a,
            Side::B => &self.b,
        }
    }

    // --- Side accessors (for internal use and migration) ---

    fn blocking(&self, side: Side) -> &RwLock<BlockingIndex> {
        match side {
            Side::A => &self.a.blocking,
            Side::B => &self.b.blocking,
        }
    }

    fn unmatched(&self, side: Side) -> &DashSet<String> {
        match side {
            Side::A => &self.a_unmatched,
            Side::B => &self.b_unmatched,
        }
    }
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStore")
            .field("a_records", &self.a.records.len())
            .field("b_records", &self.b.records.len())
            .field("a_unmatched", &self.a_unmatched.len())
            .field("b_unmatched", &self.b_unmatched.len())
            .finish()
    }
}

impl RecordStore for MemoryStore {
    // --- Records ---

    fn get(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError> {
        super::RecordCollection::get(self.collection(side), id)
    }

    fn insert(&self, side: Side, id: &str, record: &Record) -> Result<Option<Record>, StoreError> {
        super::RecordCollection::insert(self.collection(side), id, record)
    }

    fn remove(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError> {
        super::RecordCollection::remove(self.collection(side), id)
    }

    fn contains(&self, side: Side, id: &str) -> Result<bool, StoreError> {
        super::RecordCollection::contains(self.collection(side), id)
    }

    fn len(&self, side: Side) -> Result<usize, StoreError> {
        super::RecordCollection::len(self.collection(side))
    }

    fn ids(&self, side: Side) -> Result<Vec<String>, StoreError> {
        super::RecordCollection::ids(self.collection(side))
    }

    fn get_many(&self, side: Side, ids: &[String]) -> Result<Vec<(String, Record)>, StoreError> {
        super::RecordCollection::get_many(self.collection(side), ids)
    }

    fn get_many_fields(
        &self,
        side: Side,
        ids: &[String],
        fields: &[String],
    ) -> Result<Vec<(String, Record)>, StoreError> {
        super::RecordCollection::get_many_fields(self.collection(side), ids, fields)
    }

    fn for_each_record(
        &self,
        side: Side,
        f: &mut dyn FnMut(&str, &Record),
    ) -> Result<(), StoreError> {
        super::RecordCollection::for_each_record(self.collection(side), f)
    }

    // --- Blocking ---

    fn blocking_insert(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError> {
        let mut bi = self
            .blocking(side)
            .write()
            .unwrap_or_else(|e| e.into_inner());
        bi.insert(id, record, side);
        Ok(())
    }

    fn blocking_remove(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError> {
        let mut bi = self
            .blocking(side)
            .write()
            .unwrap_or_else(|e| e.into_inner());
        bi.remove(id, record, side);
        Ok(())
    }

    fn blocking_query(
        &self,
        record: &Record,
        query_side: Side,
        pool_side: Side,
    ) -> Result<Vec<String>, StoreError> {
        let bi = self
            .blocking(pool_side)
            .read()
            .unwrap_or_else(|e| e.into_inner());
        Ok(bi.query(record, query_side).into_iter().collect())
    }

    // --- Unmatched ---

    fn mark_unmatched(&self, side: Side, id: &str) -> Result<(), StoreError> {
        self.unmatched(side).insert(id.to_string());
        Ok(())
    }

    fn mark_matched(&self, side: Side, id: &str) -> Result<(), StoreError> {
        self.unmatched(side).remove(id);
        Ok(())
    }

    fn is_unmatched(&self, side: Side, id: &str) -> Result<bool, StoreError> {
        Ok(self.unmatched(side).contains(id))
    }

    fn unmatched_count(&self, side: Side) -> Result<usize, StoreError> {
        Ok(self.unmatched(side).len())
    }

    fn unmatched_ids(&self, side: Side) -> Result<Vec<String>, StoreError> {
        let mut ids: Vec<String> = self
            .unmatched(side)
            .iter()
            .map(|r| r.key().clone())
            .collect();
        ids.sort();
        Ok(ids)
    }

    // --- Common ID index ---

    fn common_id_insert(
        &self,
        side: Side,
        common_id: &str,
        record_id: &str,
    ) -> Result<(), StoreError> {
        super::RecordCollection::common_id_insert(self.collection(side), common_id, record_id)
    }

    fn common_id_lookup(&self, side: Side, common_id: &str) -> Result<Option<String>, StoreError> {
        super::RecordCollection::common_id_lookup(self.collection(side), common_id)
    }

    fn common_id_remove(&self, side: Side, common_id: &str) -> Result<(), StoreError> {
        super::RecordCollection::common_id_remove(self.collection(side), common_id)
    }

    fn common_id_remove_owned(
        &self,
        side: Side,
        common_id: &str,
        record_id: &str,
    ) -> Result<(), StoreError> {
        super::RecordCollection::common_id_remove_owned(self.collection(side), common_id, record_id)
    }

    // --- Exact prefilter ---

    fn build_exact_index(&self, side: Side, field_names: &[String]) -> Result<(), StoreError> {
        super::RecordCollection::build_exact_index(self.collection(side), field_names)
    }

    fn exact_lookup(
        &self,
        side: Side,
        kvs: &[(String, String)],
    ) -> Result<Option<String>, StoreError> {
        super::RecordCollection::exact_lookup(self.collection(side), kvs)
    }

    // --- Review persistence (no-op for memory — DashMap is source of truth) ---

    fn persist_review(
        &self,
        _key: &str,
        _id: &str,
        _side: Side,
        _candidate_id: &str,
        _score: f64,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    fn remove_query_reviews(&self, _side: Side, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    fn remove_reviews_for_id(&self, _side: Side, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    fn remove_reviews_for_pair(&self, _a_id: &str, _b_id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    fn load_reviews(&self) -> Result<Vec<super::ReviewEntry>, StoreError> {
        Ok(vec![])
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
        assert!(store.insert(Side::A, "1", &rec).unwrap().is_none());
        let got = store.get(Side::A, "1").unwrap().unwrap();
        assert_eq!(got.get("name").unwrap(), "Foo");
    }

    #[test]
    fn insert_replaces_and_returns_old() {
        let store = make_store();
        let rec1 = make_record(&[("id", "1"), ("name", "Foo")]);
        let rec2 = make_record(&[("id", "1"), ("name", "Bar")]);
        store.insert(Side::A, "1", &rec1).unwrap();
        let old = store.insert(Side::A, "1", &rec2).unwrap().unwrap();
        assert_eq!(old.get("name").unwrap(), "Foo");
        assert_eq!(
            store
                .get(Side::A, "1")
                .unwrap()
                .unwrap()
                .get("name")
                .unwrap(),
            "Bar"
        );
    }

    #[test]
    fn remove_returns_old() {
        let store = make_store();
        let rec = make_record(&[("id", "1"), ("name", "Foo")]);
        store.insert(Side::B, "1", &rec).unwrap();
        let removed = store.remove(Side::B, "1").unwrap().unwrap();
        assert_eq!(removed.get("name").unwrap(), "Foo");
        assert!(!store.contains(Side::B, "1").unwrap());
    }

    #[test]
    fn contains_and_len() {
        let store = make_store();
        assert!(!store.contains(Side::A, "x").unwrap());
        assert_eq!(store.len(Side::A).unwrap(), 0);
        store
            .insert(Side::A, "x", &make_record(&[("id", "x")]))
            .unwrap();
        assert!(store.contains(Side::A, "x").unwrap());
        assert_eq!(store.len(Side::A).unwrap(), 1);
    }

    #[test]
    fn ids_returns_all() {
        let store = make_store();
        store
            .insert(Side::A, "a", &make_record(&[("id", "a")]))
            .unwrap();
        store
            .insert(Side::A, "b", &make_record(&[("id", "b")]))
            .unwrap();
        let mut ids = store.ids(Side::A).unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn sides_are_independent() {
        let store = make_store();
        store
            .insert(Side::A, "1", &make_record(&[("id", "1")]))
            .unwrap();
        assert!(store.contains(Side::A, "1").unwrap());
        assert!(!store.contains(Side::B, "1").unwrap());
    }

    // --- Blocking ---

    #[test]
    fn blocking_insert_query_remove() {
        let store = make_store();
        let rec_gb = make_record(&[("country", "GB")]);
        let rec_us = make_record(&[("country", "US")]);
        store.blocking_insert(Side::A, "a1", &rec_gb).unwrap();
        store.blocking_insert(Side::A, "a2", &rec_gb).unwrap();
        store.blocking_insert(Side::A, "a3", &rec_us).unwrap();

        // Query from B side looking for GB records in A
        let query = make_record(&[("domicile", "GB")]);
        let hits = store.blocking_query(&query, Side::B, Side::A).unwrap();
        assert_eq!(hits.len(), 2, "expected 2 GB hits, got {:?}", hits);

        // Remove one and re-query
        store.blocking_remove(Side::A, "a1", &rec_gb).unwrap();
        let hits = store.blocking_query(&query, Side::B, Side::A).unwrap();
        assert_eq!(hits.len(), 1, "expected 1 GB hit after remove");
    }

    // --- Unmatched ---

    #[test]
    fn unmatched_lifecycle() {
        let store = make_store();
        store.mark_unmatched(Side::A, "x").unwrap();
        assert!(store.is_unmatched(Side::A, "x").unwrap());
        assert_eq!(store.unmatched_count(Side::A).unwrap(), 1);
        store.mark_matched(Side::A, "x").unwrap();
        assert!(!store.is_unmatched(Side::A, "x").unwrap());
        assert_eq!(store.unmatched_count(Side::A).unwrap(), 0);
    }

    #[test]
    fn unmatched_ids_sorted() {
        let store = make_store();
        store.mark_unmatched(Side::B, "c").unwrap();
        store.mark_unmatched(Side::B, "a").unwrap();
        store.mark_unmatched(Side::B, "b").unwrap();
        assert_eq!(store.unmatched_ids(Side::B).unwrap(), vec!["a", "b", "c"]);
    }

    // --- Common ID ---

    #[test]
    fn common_id_lifecycle() {
        let store = make_store();
        store.common_id_insert(Side::A, "LEI-123", "a1").unwrap();
        assert_eq!(
            store.common_id_lookup(Side::A, "LEI-123").unwrap(),
            Some("a1".to_string())
        );
        store.common_id_remove(Side::A, "LEI-123").unwrap();
        assert!(
            store
                .common_id_lookup(Side::A, "LEI-123")
                .unwrap()
                .is_none()
        );
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
        assert_eq!(store.len(Side::A).unwrap(), 2);
        assert_eq!(store.len(Side::B).unwrap(), 1);

        // Blocking should work: query from B for GB records in A
        let query = make_record(&[("domicile", "GB")]);
        let hits = store.blocking_query(&query, Side::B, Side::A).unwrap();
        assert_eq!(hits.len(), 1, "expected 1 GB hit from A-side blocking");
    }
}
