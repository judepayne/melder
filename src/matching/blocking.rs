//! Blocking filter: pre-filter candidates by exact field matches.
//!
//! Two implementations:
//! - `apply_blocking`: linear scan for batch mode
//! - `BlockingIndex`: indexed lookup for live mode

use std::collections::{HashMap, HashSet};

use crate::config::{BlockingConfig, BlockingFieldPair};
use crate::models::{Record, Side};

// ---------------------------------------------------------------------------
// Linear scan blocking (batch mode)
// ---------------------------------------------------------------------------

/// Apply blocking filter via linear scan over pool records.
///
/// Returns IDs of pool records that pass the blocking filter against the
/// query record. Used in batch mode where B queries against A pool.
///
/// - AND: all field pairs must match (case-insensitive, trimmed)
/// - OR: any field pair match passes
/// - Missing query value for a field: skip that constraint
pub fn apply_blocking(
    query_record: &Record,
    pool_records: &HashMap<String, Record>,
    blocking_config: &BlockingConfig,
    query_side: Side,
) -> Vec<String> {
    if !blocking_config.enabled || blocking_config.fields.is_empty() {
        // No blocking — return all pool IDs
        return pool_records.keys().cloned().collect();
    }

    let is_and = blocking_config.operator.eq_ignore_ascii_case("and");

    pool_records
        .iter()
        .filter(|(_, pool_record)| {
            passes_blocking(
                query_record,
                pool_record,
                &blocking_config.fields,
                query_side,
                is_and,
            )
        })
        .map(|(id, _)| id.clone())
        .collect()
}

/// Check whether a single pool record passes the blocking filter for a query.
fn passes_blocking(
    query_record: &Record,
    pool_record: &Record,
    fields: &[BlockingFieldPair],
    query_side: Side,
    is_and: bool,
) -> bool {
    if is_and {
        // AND: all field pairs must match (or query value is missing → skip)
        for fp in fields {
            let (query_field, pool_field) = field_pair_for_side(fp, query_side);
            let query_val = query_record
                .get(query_field)
                .map(|s| s.trim().to_lowercase());
            let pool_val = pool_record.get(pool_field).map(|s| s.trim().to_lowercase());

            // Missing query value → skip this constraint
            if query_val.as_deref().unwrap_or("").is_empty() {
                continue;
            }

            // Values must match
            if query_val != pool_val {
                return false;
            }
        }
        true
    } else {
        // OR: any field pair match passes
        let mut any_checked = false;
        for fp in fields {
            let (query_field, pool_field) = field_pair_for_side(fp, query_side);
            let query_val = query_record
                .get(query_field)
                .map(|s| s.trim().to_lowercase());

            // Missing query value → skip this field
            if query_val.as_deref().unwrap_or("").is_empty() {
                continue;
            }

            any_checked = true;
            let pool_val = pool_record.get(pool_field).map(|s| s.trim().to_lowercase());
            if query_val == pool_val {
                return true;
            }
        }

        // If no fields were checked (all query values missing), pass all
        !any_checked
    }
}

/// Get the (query_field, pool_field) names for a given query side.
///
/// If query is B-side, query uses field_b and pool (A-side) uses field_a.
/// If query is A-side, query uses field_a and pool (B-side) uses field_b.
fn field_pair_for_side<'a>(fp: &'a BlockingFieldPair, query_side: Side) -> (&'a str, &'a str) {
    match query_side {
        Side::B => (&fp.field_b, &fp.field_a),
        Side::A => (&fp.field_a, &fp.field_b),
    }
}

// ---------------------------------------------------------------------------
// Blocking index (live mode)
// ---------------------------------------------------------------------------

/// Indexed blocking for O(1) lookups in live mode.
///
/// Maintains hash maps for fast candidate set retrieval:
/// - AND mode: composite key (all field values) → set of IDs
/// - OR mode: per-field value → set of IDs, results unioned at query time
pub struct BlockingIndex {
    operator: BlockingOperator,
    fields: Vec<BlockingFieldPair>,
    /// AND: composite key → set of record IDs
    and_index: HashMap<Vec<String>, HashSet<String>>,
    /// OR: per-field maps, value → set of record IDs
    or_indices: Vec<HashMap<String, HashSet<String>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockingOperator {
    And,
    Or,
}

impl BlockingIndex {
    /// Create a new blocking index.
    pub fn new(operator: BlockingOperator, fields: Vec<BlockingFieldPair>) -> Self {
        let num_fields = fields.len();
        Self {
            operator,
            fields,
            and_index: HashMap::new(),
            or_indices: (0..num_fields).map(|_| HashMap::new()).collect(),
        }
    }

    /// Create from a BlockingConfig.
    pub fn from_config(config: &BlockingConfig) -> Self {
        let operator = if config.operator.eq_ignore_ascii_case("or") {
            BlockingOperator::Or
        } else {
            BlockingOperator::And
        };
        Self::new(operator, config.fields.clone())
    }

    /// Insert a record into the blocking index.
    pub fn insert(&mut self, id: &str, record: &Record, side: Side) {
        match self.operator {
            BlockingOperator::And => {
                let key = self.composite_key(record, side);
                self.and_index
                    .entry(key)
                    .or_default()
                    .insert(id.to_string());
            }
            BlockingOperator::Or => {
                for (i, fp) in self.fields.iter().enumerate() {
                    let field = match side {
                        Side::A => &fp.field_a,
                        Side::B => &fp.field_b,
                    };
                    let val = record
                        .get(field)
                        .map(|s| s.trim().to_lowercase())
                        .unwrap_or_default();
                    if !val.is_empty() {
                        self.or_indices[i]
                            .entry(val)
                            .or_default()
                            .insert(id.to_string());
                    }
                }
            }
        }
    }

    /// Remove a record from the blocking index.
    pub fn remove(&mut self, id: &str, record: &Record, side: Side) {
        match self.operator {
            BlockingOperator::And => {
                let key = self.composite_key(record, side);
                if let Some(set) = self.and_index.get_mut(&key) {
                    set.remove(id);
                    if set.is_empty() {
                        self.and_index.remove(&key);
                    }
                }
            }
            BlockingOperator::Or => {
                for (i, fp) in self.fields.iter().enumerate() {
                    let field = match side {
                        Side::A => &fp.field_a,
                        Side::B => &fp.field_b,
                    };
                    let val = record
                        .get(field)
                        .map(|s| s.trim().to_lowercase())
                        .unwrap_or_default();
                    if !val.is_empty() {
                        if let Some(set) = self.or_indices[i].get_mut(&val) {
                            set.remove(id);
                            if set.is_empty() {
                                self.or_indices[i].remove(&val);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Query the blocking index: return IDs of records that pass the filter.
    ///
    /// `query_side` is the side of the query record. The index contains
    /// records from the *opposite* side.
    pub fn query(&self, query_record: &Record, query_side: Side) -> HashSet<String> {
        match self.operator {
            BlockingOperator::And => {
                let key = self.composite_key_for_query(query_record, query_side);
                // If any key component is empty (missing query value), return
                // all IDs (can't filter on missing data)
                if key.iter().any(|v| v.is_empty()) {
                    return self.all_ids_and();
                }
                self.and_index.get(&key).cloned().unwrap_or_default()
            }
            BlockingOperator::Or => {
                let mut result = HashSet::new();
                let mut any_checked = false;
                for (i, fp) in self.fields.iter().enumerate() {
                    let field = match query_side {
                        Side::A => &fp.field_a,
                        Side::B => &fp.field_b,
                    };
                    let val = query_record
                        .get(field)
                        .map(|s| s.trim().to_lowercase())
                        .unwrap_or_default();
                    if val.is_empty() {
                        continue;
                    }
                    any_checked = true;
                    if let Some(set) = self.or_indices[i].get(&val) {
                        result.extend(set.iter().cloned());
                    }
                }
                if !any_checked {
                    // All query values missing → return all
                    return self.all_ids_or();
                }
                result
            }
        }
    }

    fn composite_key(&self, record: &Record, side: Side) -> Vec<String> {
        self.fields
            .iter()
            .map(|fp| {
                let field = match side {
                    Side::A => &fp.field_a,
                    Side::B => &fp.field_b,
                };
                record
                    .get(field)
                    .map(|s| s.trim().to_lowercase())
                    .unwrap_or_default()
            })
            .collect()
    }

    fn composite_key_for_query(&self, record: &Record, query_side: Side) -> Vec<String> {
        self.fields
            .iter()
            .map(|fp| {
                let field = match query_side {
                    Side::A => &fp.field_a,
                    Side::B => &fp.field_b,
                };
                record
                    .get(field)
                    .map(|s| s.trim().to_lowercase())
                    .unwrap_or_default()
            })
            .collect()
    }

    fn all_ids_and(&self) -> HashSet<String> {
        self.and_index
            .values()
            .flat_map(|s| s.iter().cloned())
            .collect()
    }

    fn all_ids_or(&self) -> HashSet<String> {
        self.or_indices
            .iter()
            .flat_map(|m| m.values().flat_map(|s| s.iter().cloned()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_pool() -> HashMap<String, Record> {
        let mut pool = HashMap::new();
        for i in 0..10 {
            let cc = if i < 3 {
                "GB"
            } else if i < 7 {
                "US"
            } else {
                "DE"
            };
            let rec = make_record(&[
                ("entity_id", &format!("A-{}", i)),
                ("country_code", cc),
                ("legal_name", &format!("Company {}", i)),
            ]);
            pool.insert(format!("A-{}", i), rec);
        }
        pool
    }

    fn make_blocking_config(enabled: bool, operator: &str) -> BlockingConfig {
        BlockingConfig {
            enabled,
            operator: operator.to_string(),
            fields: vec![BlockingFieldPair {
                field_a: "country_code".to_string(),
                field_b: "domicile".to_string(),
            }],
            field_a: None,
            field_b: None,
        }
    }

    // --- Linear scan tests (batch mode) ---

    #[test]
    fn blocking_disabled_returns_all() {
        let pool = make_pool();
        let query = make_record(&[("domicile", "GB")]);
        let config = make_blocking_config(false, "and");
        let result = apply_blocking(&query, &pool, &config, Side::B);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn blocking_and_filters_correctly() {
        let pool = make_pool();
        let query = make_record(&[("domicile", "GB")]);
        let config = make_blocking_config(true, "and");
        let result = apply_blocking(&query, &pool, &config, Side::B);
        assert_eq!(
            result.len(),
            3,
            "expected 3 GB records, got {}",
            result.len()
        );
        for id in &result {
            let rec = pool.get(id).unwrap();
            assert_eq!(rec.get("country_code").unwrap(), "GB");
        }
    }

    #[test]
    fn blocking_missing_query_value_skips_constraint() {
        let pool = make_pool();
        let query = make_record(&[("domicile", "")]); // empty → skip
        let config = make_blocking_config(true, "and");
        let result = apply_blocking(&query, &pool, &config, Side::B);
        assert_eq!(result.len(), 10, "missing query value should pass all");
    }

    #[test]
    fn blocking_or_mode() {
        let pool = make_pool();
        let query = make_record(&[("domicile", "GB")]);
        let mut config = make_blocking_config(true, "or");
        // Add a second field pair
        config.fields.push(BlockingFieldPair {
            field_a: "legal_name".to_string(),
            field_b: "name".to_string(),
        });
        let result = apply_blocking(&query, &pool, &config, Side::B);
        // Only country_code field checked (name not in query), should return GB records
        assert_eq!(result.len(), 3);
    }

    // --- BlockingIndex tests (live mode) ---

    #[test]
    fn index_and_insert_query() {
        let fields = vec![BlockingFieldPair {
            field_a: "country_code".to_string(),
            field_b: "domicile".to_string(),
        }];
        let mut idx = BlockingIndex::new(BlockingOperator::And, fields);

        // Insert 100 A records with various country codes
        for i in 0..100 {
            let cc = if i % 3 == 0 {
                "GB"
            } else if i % 3 == 1 {
                "US"
            } else {
                "DE"
            };
            let rec = make_record(&[("country_code", cc)]);
            idx.insert(&format!("A-{}", i), &rec, Side::A);
        }

        // Query from B side
        let query = make_record(&[("domicile", "GB")]);
        let results = idx.query(&query, Side::B);
        assert_eq!(
            results.len(),
            34,
            "expected 34 GB records, got {}",
            results.len()
        );
        for id in &results {
            assert!(id.starts_with("A-"));
        }
    }

    #[test]
    fn index_and_remove() {
        let fields = vec![BlockingFieldPair {
            field_a: "country_code".to_string(),
            field_b: "domicile".to_string(),
        }];
        let mut idx = BlockingIndex::new(BlockingOperator::And, fields);

        let rec_gb = make_record(&[("country_code", "GB")]);
        idx.insert("A-1", &rec_gb, Side::A);
        idx.insert("A-2", &rec_gb, Side::A);

        let query = make_record(&[("domicile", "GB")]);
        assert_eq!(idx.query(&query, Side::B).len(), 2);

        idx.remove("A-1", &rec_gb, Side::A);
        assert_eq!(idx.query(&query, Side::B).len(), 1);
    }

    #[test]
    fn index_or_mode() {
        let fields = vec![
            BlockingFieldPair {
                field_a: "country_code".to_string(),
                field_b: "domicile".to_string(),
            },
            BlockingFieldPair {
                field_a: "sector".to_string(),
                field_b: "industry".to_string(),
            },
        ];
        let mut idx = BlockingIndex::new(BlockingOperator::Or, fields);

        let rec1 = make_record(&[("country_code", "GB"), ("sector", "finance")]);
        let rec2 = make_record(&[("country_code", "US"), ("sector", "finance")]);
        let rec3 = make_record(&[("country_code", "DE"), ("sector", "tech")]);
        idx.insert("A-1", &rec1, Side::A);
        idx.insert("A-2", &rec2, Side::A);
        idx.insert("A-3", &rec3, Side::A);

        // Query matching GB → gets A-1; OR matching finance → gets A-1, A-2
        let query = make_record(&[("domicile", "GB"), ("industry", "finance")]);
        let results = idx.query(&query, Side::B);
        assert!(results.contains("A-1"));
        assert!(results.contains("A-2"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn index_missing_query_value_returns_all() {
        let fields = vec![BlockingFieldPair {
            field_a: "country_code".to_string(),
            field_b: "domicile".to_string(),
        }];
        let mut idx = BlockingIndex::new(BlockingOperator::And, fields);

        let rec = make_record(&[("country_code", "GB")]);
        idx.insert("A-1", &rec, Side::A);
        idx.insert("A-2", &rec, Side::A);

        // Missing domicile → returns all
        let query = make_record(&[("other_field", "x")]);
        let results = idx.query(&query, Side::B);
        assert_eq!(results.len(), 2);
    }
}
