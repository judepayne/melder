//! Bidirectional synonym index.
//!
//! Maps normalised lookup keys to record IDs, tagged by field pair and side.
//! Supports both batch build (from a RecordStore) and incremental upsert/remove
//! for live mode.
//!
//! Bidirectional indexing is essential: the acronym can appear on either side.
//! For each record, both the generated acronyms AND the raw field value are
//! indexed as lookup keys. This enables:
//! - Query "HWAG" → finds record whose full name generates "HWAG" as an acronym
//! - Query "Harris, Watkins and Goodwin BV" → finds record whose raw name is
//!   one of this name's acronyms (if such a record exists)

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::SynonymFieldConfig;
use crate::models::{Record, Side};
use crate::store::RecordStore;
use crate::synonym::dictionary::SynonymDictionary;
use crate::synonym::generator::generate_acronyms;

// --- Types ---

/// A single entry in the synonym index.
#[derive(Debug, Clone)]
pub struct SynonymEntry {
    pub record_id: String,
    pub side: Side,
    pub field_a: String,
    pub field_b: String,
}

/// Bidirectional synonym index.
///
/// One shared index for all configured synonym fields. Entries are tagged
/// with their field pair so lookups can filter to the relevant scoring field.
pub struct SynonymIndex {
    /// Maps normalised lookup key → list of matching entries.
    index: HashMap<String, Vec<SynonymEntry>>,
    /// Optional user-provided synonym dictionary for term expansion.
    dictionary: Option<Arc<SynonymDictionary>>,
}

impl std::fmt::Debug for SynonymIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entry_count: usize = self.index.values().map(|v| v.len()).sum();
        f.debug_struct("SynonymIndex")
            .field("keys", &self.index.len())
            .field("entries", &entry_count)
            .finish()
    }
}

// --- Helpers ---

/// Get the field value from a record for the given side and field config.
fn field_value(record: &Record, side: Side, config: &SynonymFieldConfig) -> String {
    let field = match side {
        Side::A => &config.field_a,
        Side::B => &config.field_b,
    };
    record.get(field).cloned().unwrap_or_default()
}

/// Resolve the min_length for a synonym field config.
///
/// Uses the first generator's min_length, or defaults to 3.
fn min_length(config: &SynonymFieldConfig) -> usize {
    config.generators.first().map(|g| g.min_length).unwrap_or(3)
}

// --- Implementation ---

impl Default for SynonymIndex {
    fn default() -> Self {
        Self::new(None)
    }
}

impl SynonymIndex {
    /// Create an empty index, optionally with a dictionary.
    pub fn new(dictionary: Option<Arc<SynonymDictionary>>) -> Self {
        Self {
            index: HashMap::new(),
            dictionary,
        }
    }

    /// Build the index from a record store for one side.
    ///
    /// For each record, generates acronyms from the configured fields and
    /// indexes both the acronyms and the raw field value as lookup keys.
    /// If a dictionary is provided, also indexes under dictionary-expanded terms.
    pub fn build(
        store: &dyn RecordStore,
        side: Side,
        synonym_fields: &[SynonymFieldConfig],
        dictionary: Option<Arc<SynonymDictionary>>,
    ) -> Self {
        let mut idx = Self::new(dictionary);
        store.for_each_record(side, &mut |id, record| {
            idx.index_record(id, record, side, synonym_fields);
        });
        idx
    }

    /// Index a single record's synonym entries.
    fn index_record(
        &mut self,
        id: &str,
        record: &Record,
        side: Side,
        synonym_fields: &[SynonymFieldConfig],
    ) {
        for config in synonym_fields {
            let value = field_value(record, side, config);
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }

            let min_len = min_length(config);
            let entry = SynonymEntry {
                record_id: id.to_string(),
                side,
                field_a: config.field_a.clone(),
                field_b: config.field_b.clone(),
            };

            // Index generated acronyms as lookup keys.
            // e.g. "Harris, Watkins and Goodwin BV" → index keys "HWG", "HWAG", etc.
            let acronyms = generate_acronyms(trimmed, min_len);
            for acr in &acronyms {
                self.index
                    .entry(acr.clone())
                    .or_default()
                    .push(entry.clone());
            }

            // Index the raw value (uppercased) as a lookup key.
            // This enables reverse lookup: query "HWAG" finds a record whose
            // raw name field contains "HWAG", and that record can then be
            // matched against a full name on the other side.
            let raw_key = trimmed.to_uppercase();
            self.index.entry(raw_key).or_default().push(entry.clone());

            // Dictionary expansion: if the record's name matches a dictionary
            // term, also index under all equivalent terms.
            if let Some(ref dict) = self.dictionary {
                let expanded = dict.expand(trimmed);
                for equiv_term in expanded {
                    self.index
                        .entry(equiv_term.clone())
                        .or_default()
                        .push(entry.clone());
                }
            }
        }
    }

    /// Add or replace a record's synonym entries (live mode).
    ///
    /// Removes any existing entries for this record+side, then re-indexes.
    pub fn upsert(
        &mut self,
        id: &str,
        record: &Record,
        side: Side,
        synonym_fields: &[SynonymFieldConfig],
    ) {
        self.remove(id, side);
        self.index_record(id, record, side, synonym_fields);
    }

    /// Remove all entries for a record (live mode).
    pub fn remove(&mut self, id: &str, side: Side) {
        // Retain only entries that don't match the record+side.
        // Remove empty keys to avoid unbounded growth.
        self.index.retain(|_key, entries| {
            entries.retain(|e| !(e.record_id == id && e.side == side));
            !entries.is_empty()
        });
    }

    /// Look up synonym candidates for a query value on a given field pair.
    ///
    /// Checks both directions:
    /// 1. Direct lookup: is the query value (uppercased) a key in the index?
    ///    → finds records whose acronyms or raw name match the query.
    /// 2. Generated lookup: generate acronyms from the query value, look each up.
    ///    → finds records whose raw name is one of the query's acronyms.
    ///
    /// Returns deduplicated record IDs filtered to the given field pair.
    pub fn lookup(
        &self,
        query_value: &str,
        field_a: &str,
        field_b: &str,
        min_len: usize,
    ) -> Vec<String> {
        let trimmed = query_value.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        let mut add_matches = |key: &str| {
            if let Some(entries) = self.index.get(key) {
                for entry in entries {
                    if entry.field_a == field_a
                        && entry.field_b == field_b
                        && seen.insert(entry.record_id.clone())
                    {
                        result.push(entry.record_id.clone());
                    }
                }
            }
        };

        // Direct lookup: query value as key.
        let upper = trimmed.to_uppercase();
        add_matches(&upper);

        // Generated lookup: query's acronyms as keys.
        let query_acronyms = generate_acronyms(trimmed, min_len);
        for acr in &query_acronyms {
            add_matches(acr);
        }

        // Dictionary expansion: look up equivalent terms and their acronyms.
        if let Some(ref dict) = self.dictionary {
            for equiv_term in dict.expand(trimmed) {
                add_matches(equiv_term);
                // Also generate acronyms of each expanded term.
                for acr in &generate_acronyms(equiv_term, min_len) {
                    add_matches(acr);
                }
            }
        }

        result
    }

    /// Number of unique keys in the index.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SynonymFieldConfig, SynonymGenerator};
    use crate::store::memory::MemoryStore;

    fn make_config() -> Vec<SynonymFieldConfig> {
        vec![SynonymFieldConfig {
            field_a: "legal_name".to_string(),
            field_b: "counterparty_name".to_string(),
            generators: vec![SynonymGenerator {
                gen_type: "acronym".to_string(),
                min_length: 3,
            }],
        }]
    }

    fn make_record(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_store() -> MemoryStore {
        MemoryStore::new(&crate::config::BlockingConfig::default())
    }

    #[test]
    fn build_and_lookup_acronym() {
        let store = make_store();
        store.insert(
            Side::A,
            "a1",
            &make_record(&[("legal_name", "Harris, Watkins and Goodwin BV")]),
        );

        let idx = SynonymIndex::build(&store, Side::A, &make_config(), None);
        assert!(!idx.is_empty(), "index should have entries");

        // Query with the acronym → finds the full name record
        let result = idx.lookup("HWAG", "legal_name", "counterparty_name", 3);
        assert!(
            result.contains(&"a1".to_string()),
            "HWAG should find a1: {:?}",
            result
        );
    }

    #[test]
    fn reverse_lookup() {
        let store = make_store();
        // A record has the short name; query will be the full name
        store.insert(Side::A, "a1", &make_record(&[("legal_name", "HWAG")]));

        let idx = SynonymIndex::build(&store, Side::A, &make_config(), None);

        // Query with the full name → generates acronyms → finds "HWAG" in index
        let result = idx.lookup(
            "Harris, Watkins and Goodwin BV",
            "legal_name",
            "counterparty_name",
            3,
        );
        assert!(
            result.contains(&"a1".to_string()),
            "full name query should find a1 via generated acronym: {:?}",
            result
        );
    }

    #[test]
    fn field_filtering() {
        let store = make_store();
        store.insert(
            Side::A,
            "a1",
            &make_record(&[("legal_name", "Harris, Watkins and Goodwin BV")]),
        );

        let idx = SynonymIndex::build(&store, Side::A, &make_config(), None);

        // Wrong field pair → no results
        let result = idx.lookup("HWAG", "other_field", "other_field", 3);
        assert!(
            result.is_empty(),
            "wrong field pair should return empty: {:?}",
            result
        );
    }

    #[test]
    fn upsert_replaces_entries() {
        let _store = make_store();
        let config = make_config();

        let mut idx = SynonymIndex::new(None);

        // Insert initial record
        let rec1 = make_record(&[("legal_name", "Harris, Watkins and Goodwin BV")]);
        idx.upsert("a1", &rec1, Side::A, &config);

        let result = idx.lookup("HWAG", "legal_name", "counterparty_name", 3);
        assert!(result.contains(&"a1".to_string()));

        // Replace with different name
        let rec2 = make_record(&[("legal_name", "Goldman Sachs Asset Management")]);
        idx.upsert("a1", &rec2, Side::A, &config);

        // Old acronym should no longer match
        let result = idx.lookup("HWAG", "legal_name", "counterparty_name", 3);
        assert!(
            !result.contains(&"a1".to_string()),
            "old acronym should be gone: {:?}",
            result
        );

        // New acronym should match
        let result = idx.lookup("GSAM", "legal_name", "counterparty_name", 3);
        assert!(
            result.contains(&"a1".to_string()),
            "new acronym should find a1: {:?}",
            result
        );
    }

    #[test]
    fn remove_clears_entries() {
        let config = make_config();

        let mut idx = SynonymIndex::new(None);

        // Add a record
        let rec = make_record(&[("legal_name", "Harris, Watkins and Goodwin BV")]);
        idx.upsert("a1", &rec, Side::A, &config);

        // Remove it
        idx.remove("a1", Side::A);

        let result = idx.lookup("HWAG", "legal_name", "counterparty_name", 3);
        assert!(
            result.is_empty(),
            "removed record should not appear: {:?}",
            result
        );
    }

    #[test]
    fn deduplicates_results() {
        let store = make_store();
        store.insert(
            Side::A,
            "a1",
            &make_record(&[("legal_name", "Harris, Watkins and Goodwin BV")]),
        );

        let idx = SynonymIndex::build(&store, Side::A, &make_config(), None);

        // Even if multiple index keys map to the same record, result is deduped
        let result = idx.lookup("HWG", "legal_name", "counterparty_name", 3);
        let count = result.iter().filter(|id| *id == "a1").count();
        assert_eq!(count, 1, "should be deduplicated, got: {:?}", result);
    }
}
