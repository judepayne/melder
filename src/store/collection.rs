//! Collection-scoped record access, independent of matching membership.

use crate::error::StoreError;
use crate::models::{Record, Side};

use super::RecordStore;

/// One collection and its local lookup indexes; no matching bookkeeping.
/// Blocking queries use the collection's field names (map query fields first).
/// Record writes and index maintenance remain separate, as in `RecordStore`.
pub trait RecordCollection: Send + Sync {
    /// Collection-local `get` operation.
    fn get(&self, id: &str) -> Result<Option<Record>, StoreError>;
    /// Collection-local `insert` operation.
    fn insert(&self, id: &str, record: &Record) -> Result<Option<Record>, StoreError>;
    /// Collection-local `remove` operation.
    fn remove(&self, id: &str) -> Result<Option<Record>, StoreError>;
    /// Collection-local `contains` operation.
    fn contains(&self, id: &str) -> Result<bool, StoreError>;
    /// Collection-local `len` operation.
    fn len(&self) -> Result<usize, StoreError>;
    /// Collection-local `ids` operation.
    fn ids(&self) -> Result<Vec<String>, StoreError>;
    /// Collection-local `for_each_record` operation.
    fn for_each_record(&self, f: &mut dyn FnMut(&str, &Record)) -> Result<(), StoreError>;
    /// Collection-local `get_many` operation.
    fn get_many(&self, ids: &[String]) -> Result<Vec<(String, Record)>, StoreError>;
    /// Fetch requested fields using the backend's native bulk path.
    /// Memory backends may return additional fields, matching `RecordStore`.
    fn get_many_fields(
        &self,
        ids: &[String],
        fields: &[String],
    ) -> Result<Vec<(String, Record)>, StoreError>;
    /// Collection-local `blocking_insert` operation.
    fn blocking_insert(&self, id: &str, record: &Record) -> Result<(), StoreError>;
    /// Collection-local `blocking_remove` operation.
    fn blocking_remove(&self, id: &str, record: &Record) -> Result<(), StoreError>;
    /// Find candidates using a query expressed in this collection's field names.
    /// Directional query-to-pool field mapping belongs to the caller's plan.
    fn blocking_query(&self, record: &Record) -> Result<Vec<String>, StoreError>;
    /// Collection-local `build_exact_index` operation.
    fn build_exact_index(&self, field_names: &[String]) -> Result<(), StoreError>;
    /// Collection-local `exact_lookup` operation.
    fn exact_lookup(&self, kvs: &[(String, String)]) -> Result<Option<String>, StoreError>;
    /// Collection-local `common_id_insert` operation.
    fn common_id_insert(&self, common_id: &str, record_id: &str) -> Result<(), StoreError>;
    /// Collection-local `common_id_lookup` operation.
    fn common_id_lookup(&self, common_id: &str) -> Result<Option<String>, StoreError>;
    /// Collection-local `common_id_remove` operation.
    fn common_id_remove(&self, common_id: &str) -> Result<(), StoreError>;
    /// Collection-local `common_id_remove_owned` operation.
    fn common_id_remove_owned(&self, common_id: &str, record_id: &str) -> Result<(), StoreError>;
    /// Whether this collection has no records.
    fn is_empty(&self) -> Result<bool, StoreError> {
        Ok(self.len()? == 0)
    }
}

/// Borrowed side-bound view; shares its store's connection and transactions.
pub struct ScopedCollection<'a> {
    store: &'a dyn RecordStore,
    side: Side,
}
impl<'a> ScopedCollection<'a> {
    /// Borrow one collection without allocating or creating a connection.
    pub fn new(store: &'a dyn RecordStore, side: Side) -> Self {
        Self { store, side }
    }

    /// Query this collection's existing block index using the input schema's
    /// field mapping. Avoid cloning a query record to rename its fields.
    pub(crate) fn blocking_query_from(
        &self,
        record: &Record,
        input: Side,
    ) -> Result<Vec<String>, StoreError> {
        self.store.blocking_query(record, input, self.side)
    }
}
impl RecordCollection for ScopedCollection<'_> {
    fn get(&self, id: &str) -> Result<Option<Record>, StoreError> {
        self.store.get(self.side, id)
    }
    fn insert(&self, id: &str, record: &Record) -> Result<Option<Record>, StoreError> {
        self.store.insert(self.side, id, record)
    }
    fn remove(&self, id: &str) -> Result<Option<Record>, StoreError> {
        self.store.remove(self.side, id)
    }
    fn contains(&self, id: &str) -> Result<bool, StoreError> {
        self.store.contains(self.side, id)
    }
    fn len(&self) -> Result<usize, StoreError> {
        self.store.len(self.side)
    }
    fn ids(&self) -> Result<Vec<String>, StoreError> {
        self.store.ids(self.side)
    }
    fn for_each_record(&self, f: &mut dyn FnMut(&str, &Record)) -> Result<(), StoreError> {
        self.store.for_each_record(self.side, f)
    }
    fn get_many(&self, ids: &[String]) -> Result<Vec<(String, Record)>, StoreError> {
        self.store.get_many(self.side, ids)
    }
    fn get_many_fields(
        &self,
        ids: &[String],
        fields: &[String],
    ) -> Result<Vec<(String, Record)>, StoreError> {
        self.store.get_many_fields(self.side, ids, fields)
    }
    fn blocking_insert(&self, id: &str, record: &Record) -> Result<(), StoreError> {
        self.store.blocking_insert(self.side, id, record)
    }
    fn blocking_remove(&self, id: &str, record: &Record) -> Result<(), StoreError> {
        self.store.blocking_remove(self.side, id, record)
    }
    fn blocking_query(&self, record: &Record) -> Result<Vec<String>, StoreError> {
        self.blocking_query_from(record, self.side)
    }
    fn build_exact_index(&self, field_names: &[String]) -> Result<(), StoreError> {
        self.store.build_exact_index(self.side, field_names)
    }
    fn exact_lookup(&self, kvs: &[(String, String)]) -> Result<Option<String>, StoreError> {
        self.store.exact_lookup(self.side, kvs)
    }
    fn common_id_insert(&self, common_id: &str, record_id: &str) -> Result<(), StoreError> {
        self.store.common_id_insert(self.side, common_id, record_id)
    }
    fn common_id_lookup(&self, common_id: &str) -> Result<Option<String>, StoreError> {
        self.store.common_id_lookup(self.side, common_id)
    }
    fn common_id_remove(&self, common_id: &str) -> Result<(), StoreError> {
        self.store.common_id_remove(self.side, common_id)
    }
    fn common_id_remove_owned(&self, common_id: &str, record_id: &str) -> Result<(), StoreError> {
        self.store
            .common_id_remove_owned(self.side, common_id, record_id)
    }
}

#[cfg(test)]
mod tests {

    use crate::config::{BlockingConfig, BlockingFieldPair};
    use crate::models::{Record, Side};

    use crate::store::memory::{MemoryCollection, MemoryStore};
    use crate::store::{RecordCollection, ScopedCollection};

    fn make_config() -> BlockingConfig {
        BlockingConfig {
            enabled: true,
            fields: vec![BlockingFieldPair {
                field_a: "region_left".into(),
                field_b: "region_right".into(),
            }],
            ..Default::default()
        }
    }

    fn check_collection(
        collection: &dyn RecordCollection,
        field: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let record: Record = [
            ("external_key".into(), "payload-id".into()),
            (field.into(), "US".into()),
        ]
        .into();
        assert!(collection.is_empty()?, "collection starts empty");
        assert!(
            collection.insert("storage-id", &record)?.is_none(),
            "first insertion has no predecessor"
        );
        assert_eq!(collection.len()?, 1, "one collection record");
        assert!(
            collection.contains("storage-id")?,
            "storage ID is independent of payload ID"
        );
        let ids = vec!["storage-id".into(), "missing".into()];
        assert_eq!(collection.get_many(&ids)?.len(), 1, "missing IDs omitted");
        let projected = collection.get_many_fields(&ids, &[field.into()])?;
        assert_eq!(
            projected[0].1.get(field),
            Some(&"US".into()),
            "projection retains requested field"
        );
        assert_eq!(
            collection.ids()?,
            vec!["storage-id"],
            "IDs are collection-local"
        );
        let mut count = 0;
        collection.for_each_record(&mut |id, value| {
            assert_eq!(id, "storage-id");
            assert_eq!(value, &record);
            count += 1;
        })?;
        assert_eq!(count, 1, "native scan visits record once");
        collection.blocking_insert("storage-id", &record)?;
        assert_eq!(
            collection.blocking_query(&record)?,
            vec!["storage-id"],
            "query already uses pool field names"
        );
        collection.build_exact_index(&[field.into()])?;
        assert_eq!(
            collection.exact_lookup(&[(field.into(), "US".into())])?,
            Some("storage-id".into()),
            "local exact lookup"
        );
        collection.common_id_insert("shared", "storage-id")?;
        collection.common_id_remove_owned("shared", "someone-else")?;
        assert_eq!(
            collection.common_id_lookup("shared")?,
            Some("storage-id".into()),
            "nonowner cannot remove mapping"
        );
        collection.common_id_remove_owned("shared", "storage-id")?;
        assert_eq!(
            collection.common_id_lookup("shared")?,
            None,
            "owner removes mapping"
        );
        collection.common_id_insert("shared", "storage-id")?;
        collection.common_id_remove("shared")?;
        collection.blocking_remove("storage-id", &record)?;
        assert!(
            collection.blocking_query(&record)?.is_empty(),
            "blocking removal is explicit"
        );
        assert_eq!(
            collection.remove("storage-id")?,
            Some(record),
            "remove returns payload"
        );
        assert_eq!(collection.get("storage-id")?, None, "removed record absent");
        Ok(())
    }

    #[test]
    fn independent_memory_and_scoped_memory() -> Result<(), Box<dyn std::error::Error>> {
        check_collection(&MemoryCollection::new(&make_config()), "region_left")?;
        let store = MemoryStore::new(&make_config());
        check_collection(&ScopedCollection::new(&store, Side::A), "region_left")?;
        check_collection(&ScopedCollection::new(&store, Side::B), "region_right")?;
        check_colliding_ids(&store)
    }

    fn check_colliding_ids(
        store: &dyn crate::store::RecordStore,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let a = ScopedCollection::new(store, Side::A);
        let b = ScopedCollection::new(store, Side::B);
        let left: Record = [
            ("external_key".into(), "left".into()),
            ("region_left".into(), "US".into()),
        ]
        .into();
        let right: Record = [
            ("external_key".into(), "right".into()),
            ("region_right".into(), "UK".into()),
        ]
        .into();
        a.insert("same", &left)?;
        b.insert("same", &right)?;
        a.blocking_insert("same", &left)?;
        let query = Record::from([("region_right".into(), "US".into())]);
        assert_eq!(
            a.blocking_query_from(&query, Side::B)?,
            vec!["same"],
            "cross-pool blocking uses query field names"
        );
        let miss = Record::from([("region_right".into(), "UK".into())]);
        assert!(
            a.blocking_query_from(&miss, Side::B)?.is_empty(),
            "a mismatched block must not widen the search"
        );
        assert_eq!(a.get("same")?, Some(left), "left handle isolated");
        assert_eq!(b.get("same")?, Some(right.clone()), "right handle isolated");
        a.remove("same")?;
        assert_eq!(
            b.get("same")?,
            Some(right),
            "left removal does not mutate right"
        );
        Ok(())
    }

    #[test]
    fn scoped_sqlite_native_reads_and_errors() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let fields_a = vec!["external_key".into(), "region_left".into()];
        let fields_b = vec!["external_key".into(), "region_right".into()];
        let (store, _, writer) = crate::store::sqlite::open_sqlite(
            &dir.path().join("records.db"),
            &make_config(),
            None,
            &fields_a,
            &fields_b,
        )?;
        check_collection(&ScopedCollection::new(&store, Side::A), "region_left")?;
        check_collection(&ScopedCollection::new(&store, Side::B), "region_right")?;
        check_colliding_ids(&store)?;
        let a = ScopedCollection::new(&store, Side::A);
        let record: Record = [
            ("external_key".into(), "left".into()),
            ("region_left".into(), "US".into()),
        ]
        .into();
        a.insert("project", &record)?;
        let rows = a.get_many_fields(&["project".into()], &["region_left".into()])?;
        assert_eq!(
            rows[0].1.len(),
            1,
            "SQLite projection is native, not full-record fallback"
        );
        writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .execute_batch("DROP TABLE a_records")?;
        assert!(
            a.get("project").is_err(),
            "read errors propagate through adapter"
        );
        assert!(
            a.get_many(&["project".into()]).is_err(),
            "bulk errors propagate"
        );
        assert!(
            a.get_many_fields(&["project".into()], &["region_left".into()])
                .is_err(),
            "projection errors propagate"
        );
        assert!(
            a.for_each_record(&mut |_, _| {}).is_err(),
            "scan errors propagate"
        );
        Ok(())
    }
}
