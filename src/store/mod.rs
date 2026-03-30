//! Abstract record storage with blocking, unmatched tracking, and common ID index.
//!
//! Two implementations: `MemoryStore` (DashMap-backed, used by batch mode)
//! and `SqliteStore` (rusqlite-backed, used by live mode).

pub mod memory;
pub mod sqlite;

use crate::error::StoreError;
use crate::models::{Record, Side};

/// A single persisted review entry: `(key, id, side, candidate_id, score)`.
pub type ReviewEntry = (String, String, Side, String, f64);

/// Abstract record storage with blocking, unmatched tracking, and common ID index.
///
/// The trait is per-store (one store holds both sides), not per-side. Methods
/// take a `Side` parameter. This matches SQLite's model (one DB file with
/// `a_records` and `b_records` tables) and simplifies the interface.
///
/// Two implementations:
/// - `MemoryStore`: DashMap-backed, used by batch mode and live mode before
///   SQLite migration.
/// - `SqliteStore`: rusqlite-backed, used by live mode after migration (Step 3).
pub trait RecordStore: Send + Sync {
    // --- Records ---

    /// Get a record by ID on the given side. Returns None if not found.
    fn get(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError>;

    /// Insert or replace a record. Returns the previous record if it existed.
    fn insert(&self, side: Side, id: &str, record: &Record) -> Result<Option<Record>, StoreError>;

    /// Remove a record by ID. Returns the removed record if it existed.
    fn remove(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError>;

    /// Check if a record exists.
    fn contains(&self, side: Side, id: &str) -> Result<bool, StoreError>;

    /// Count of records on the given side.
    fn len(&self, side: Side) -> Result<usize, StoreError>;

    /// Collect all record IDs on the given side.
    fn ids(&self, side: Side) -> Result<Vec<String>, StoreError>;

    /// Iterate all records on the given side, calling `f` for each.
    ///
    /// More efficient than `ids()` + `get()` for large datasets, especially
    /// with SQLite (single table scan vs N individual queries).
    fn for_each_record(
        &self,
        side: Side,
        f: &mut dyn FnMut(&str, &Record),
    ) -> Result<(), StoreError>;

    /// Fetch multiple records by ID in one call.
    ///
    /// More efficient than N individual `get()` calls for SQLite (single
    /// `WHERE id IN (...)` query vs N individual queries). Default
    /// implementation falls back to individual `get()` calls.
    fn get_many(&self, side: Side, ids: &[String]) -> Result<Vec<(String, Record)>, StoreError> {
        let mut results = Vec::new();
        for id in ids {
            if let Some(r) = self.get(side, id)? {
                results.push((id.clone(), r));
            }
        }
        Ok(results)
    }

    /// Fetch multiple records by ID, extracting only specific fields.
    ///
    /// For SQLite, uses `json_extract()` to pull individual fields from
    /// the stored JSON without full deserialization — significantly faster
    /// when only a few fields are needed (e.g. scoring). Default
    /// implementation falls back to `get_many()` with post-filtering.
    fn get_many_fields(
        &self,
        side: Side,
        ids: &[String],
        fields: &[String],
    ) -> Result<Vec<(String, Record)>, StoreError> {
        // Default: get full records, keep only requested fields
        Ok(self
            .get_many(side, ids)?
            .into_iter()
            .map(|(id, mut rec)| {
                rec.retain(|k, _| fields.contains(k));
                (id, rec)
            })
            .collect())
    }

    // --- Blocking ---

    /// Insert a record's blocking keys into the index.
    fn blocking_insert(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError>;

    /// Remove a record's blocking keys from the index.
    fn blocking_remove(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError>;

    /// Query the blocking index: return candidate IDs that share blocking
    /// key values with the query record.
    ///
    /// `query_side` is the side of the query record (determines which
    /// blocking field names to read from the record). `pool_side` is the
    /// side whose index to search. In match mode, `pool_side` is the
    /// opposite of `query_side`. In enroll mode, both are `Side::A`.
    fn blocking_query(
        &self,
        record: &Record,
        query_side: Side,
        pool_side: Side,
    ) -> Result<Vec<String>, StoreError>;

    // --- Unmatched ---

    /// Mark a record as unmatched.
    fn mark_unmatched(&self, side: Side, id: &str) -> Result<(), StoreError>;

    /// Mark a record as matched (remove from unmatched set).
    fn mark_matched(&self, side: Side, id: &str) -> Result<(), StoreError>;

    /// Check if a record is in the unmatched set.
    fn is_unmatched(&self, side: Side, id: &str) -> Result<bool, StoreError>;

    /// Count of unmatched records on the given side.
    fn unmatched_count(&self, side: Side) -> Result<usize, StoreError>;

    /// Collect all unmatched IDs on the given side.
    fn unmatched_ids(&self, side: Side) -> Result<Vec<String>, StoreError>;

    // --- Exact prefilter ---

    /// Build an exact-prefilter index for the given side and field names.
    ///
    /// For `MemoryStore`: constructs an in-memory `HashMap<composite_key, id>`
    /// from all records on the side. For `SqliteStore`: creates a composite
    /// SQL index on the specified field columns.
    ///
    /// `fields_a` are the column/field names on side A that form the key.
    /// Only called when `exact_prefilter.enabled` is true.
    fn build_exact_index(&self, side: Side, field_names: &[String]) -> Result<(), StoreError>;

    /// Look up the ID of a record on `side` whose field values exactly match
    /// all provided `(field_name, value)` pairs (AND semantics).
    ///
    /// Returns `None` if no match, any field is empty, or the index has not
    /// been built.
    fn exact_lookup(
        &self,
        side: Side,
        kvs: &[(String, String)],
    ) -> Result<Option<String>, StoreError>;

    // --- Common ID index ---

    /// Insert or replace a common_id → record_id mapping.
    fn common_id_insert(
        &self,
        side: Side,
        common_id: &str,
        record_id: &str,
    ) -> Result<(), StoreError>;

    /// Look up a record_id by common_id value.
    fn common_id_lookup(&self, side: Side, common_id: &str) -> Result<Option<String>, StoreError>;

    /// Remove a common_id mapping.
    fn common_id_remove(&self, side: Side, common_id: &str) -> Result<(), StoreError>;

    // --- Review persistence ---

    /// Persist a review entry (write-through for SQLite, no-op for memory).
    fn persist_review(
        &self,
        key: &str,
        id: &str,
        side: Side,
        candidate_id: &str,
        score: f64,
    ) -> Result<(), StoreError>;

    /// Remove review entries where `id` or `candidate_id` equals the given ID.
    fn remove_reviews_for_id(&self, id: &str) -> Result<(), StoreError>;

    /// Remove review entries involving either of the two IDs.
    fn remove_reviews_for_pair(&self, a_id: &str, b_id: &str) -> Result<(), StoreError>;

    /// Load all persisted review entries.
    fn load_reviews(&self) -> Result<Vec<ReviewEntry>, StoreError>;
}
