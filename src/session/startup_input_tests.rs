//! Startup graph must not publish decisions from degraded vector inputs.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use crate::error::StoreError;
use crate::models::{Record, Side};
use crate::store::ReviewEntry;
use crate::vectordb::{SearchResult, VectorDB, VectorDBError};

use super::{TestResult, make_fixture, make_record};

#[derive(Debug)]
struct FaultIndex {
    inner: Box<dyn VectorDB>,
    fail_get: bool,
    fail_search: bool,
}

impl VectorDB for FaultIndex {
    fn upsert(
        &self,
        id: &str,
        vector: &[f32],
        record: &Record,
        side: Side,
    ) -> Result<(), VectorDBError> {
        self.inner.upsert(id, vector, record, side)
    }
    fn remove(&self, id: &str) -> Result<bool, VectorDBError> {
        self.inner.remove(id)
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        record: &Record,
        side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError> {
        if self.fail_search {
            return Err(VectorDBError::Backend(
                "injected startup search error".into(),
            ));
        }
        self.inner.search(query, k, record, side)
    }
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &HashSet<String>,
        record: &Record,
        side: Side,
    ) -> Result<Vec<SearchResult>, VectorDBError> {
        if self.fail_search {
            return Err(VectorDBError::Backend(
                "injected startup search error".into(),
            ));
        }
        self.inner.search_filtered(query, k, allowed, record, side)
    }
    fn get(&self, id: &str) -> Result<Option<Vec<f32>>, VectorDBError> {
        if self.fail_get {
            return Err(VectorDBError::Backend("injected startup get error".into()));
        }
        self.inner.get(id)
    }
    fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn save(&self, path: &Path) -> Result<(), VectorDBError> {
        self.inner.save(path)
    }
    fn is_stale(_: &Path, _: usize) -> Result<bool, VectorDBError> {
        Ok(false)
    }
    fn stored_text_hashes(&self) -> HashMap<String, u64> {
        self.inner.stored_text_hashes()
    }
    fn text_hash_for(&self, id: &str) -> Option<u64> {
        self.inner.text_hash_for(id)
    }
}

#[test]
fn startup_graph_vector_failures_prevent_all_publication() -> TestResult {
    for fault in [
        "query_get",
        "query_missing",
        "query_index_missing",
        "candidate_get",
        "candidate_search",
    ] {
        if fault == "candidate_search" && !cfg!(feature = "usearch") {
            continue;
        }
        let mut fixture = make_fixture(false, None)?;
        let session = Arc::get_mut(&mut fixture.session).expect("fixture session is unique");
        let state = Arc::get_mut(&mut session.state).expect("fixture state is unique");
        if fault == "candidate_search" {
            state.config.vector_backend = "usearch".into();
        }
        for side in [Side::A, Side::B] {
            let id = if side == Side::A { "a" } else { "b" };
            let record = make_record(side, id, "0");
            state.store.insert(side, id, &record)?;
            state.store.mark_unmatched(side, id)?;
            state
                .side(side)
                .combined_index
                .as_ref()
                .expect("test index")
                .upsert(id, &[1.0, 0.0], &record, side)?;
        }
        if fault == "query_index_missing" {
            state.b.combined_index = None;
        } else if fault == "query_missing" {
            state
                .b
                .combined_index
                .as_ref()
                .expect("test index")
                .remove("b")?;
        } else {
            let target = if fault == "query_get" {
                &mut state.b
            } else {
                &mut state.a
            };
            let inner = target.combined_index.take().expect("test index");
            target.combined_index = Some(Box::new(FaultIndex {
                inner,
                fail_get: fault != "candidate_search",
                fail_search: fault == "candidate_search",
            }));
        }
        let failure = session.initial_match_pass();
        assert!(failure.is_err(), "startup must reject {fault}");
        assert!(
            session.state.crossmap.pairs().is_empty(),
            "no partial assignment for {fault}"
        );
        assert!(
            session.state.review_queue.is_empty(),
            "no reviews from degraded graph for {fault}"
        );
    }
    Ok(())
}

struct FaultStore {
    inner: Arc<dyn crate::store::RecordStore>,
    fail_side: Side,
    fail_stage: &'static str,
}
impl crate::store::RecordStore for FaultStore {
    fn get(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError> {
        if side == self.fail_side && self.fail_stage == "get" {
            return Err(crate::error::StoreError::Sqlite(
                "injected startup store error".into(),
            ));
        }
        self.inner.get(side, id)
    }
    fn insert(&self, side: Side, id: &str, record: &Record) -> Result<Option<Record>, StoreError> {
        self.inner.insert(side, id, record)
    }
    fn remove(&self, side: Side, id: &str) -> Result<Option<Record>, StoreError> {
        self.inner.remove(side, id)
    }
    fn contains(&self, side: Side, id: &str) -> Result<bool, StoreError> {
        self.inner.contains(side, id)
    }
    fn len(&self, side: Side) -> Result<usize, StoreError> {
        if self.fail_stage == "len" {
            return Err(crate::error::StoreError::Sqlite(
                "injected startup store error".into(),
            ));
        }
        self.inner.len(side)
    }
    fn ids(&self, side: Side) -> Result<Vec<String>, StoreError> {
        if self.fail_stage == "ids" {
            return Err(crate::error::StoreError::Sqlite(
                "injected startup store error".into(),
            ));
        }
        self.inner.ids(side)
    }
    fn for_each_record(
        &self,
        side: Side,
        f: &mut dyn FnMut(&str, &Record),
    ) -> Result<(), StoreError> {
        self.inner.for_each_record(side, f)
    }
    fn blocking_insert(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError> {
        self.inner.blocking_insert(side, id, record)
    }
    fn blocking_remove(&self, side: Side, id: &str, record: &Record) -> Result<(), StoreError> {
        self.inner.blocking_remove(side, id, record)
    }
    fn blocking_query(
        &self,
        record: &Record,
        query_side: Side,
        pool_side: Side,
    ) -> Result<Vec<String>, StoreError> {
        if self.fail_stage == "blocking" {
            return Err(crate::error::StoreError::Sqlite(
                "injected startup store error".into(),
            ));
        }
        self.inner.blocking_query(record, query_side, pool_side)
    }
    fn mark_unmatched(&self, side: Side, id: &str) -> Result<(), StoreError> {
        self.inner.mark_unmatched(side, id)
    }
    fn mark_matched(&self, side: Side, id: &str) -> Result<(), StoreError> {
        self.inner.mark_matched(side, id)
    }
    fn is_unmatched(&self, side: Side, id: &str) -> Result<bool, StoreError> {
        self.inner.is_unmatched(side, id)
    }
    fn unmatched_count(&self, side: Side) -> Result<usize, StoreError> {
        self.inner.unmatched_count(side)
    }
    fn unmatched_ids(&self, side: Side) -> Result<Vec<String>, StoreError> {
        if self.fail_stage == "unmatched" {
            return Err(crate::error::StoreError::Sqlite(
                "injected startup store error".into(),
            ));
        }
        self.inner.unmatched_ids(side)
    }
    fn build_exact_index(&self, side: Side, field_names: &[String]) -> Result<(), StoreError> {
        self.inner.build_exact_index(side, field_names)
    }
    fn exact_lookup(
        &self,
        side: Side,
        kvs: &[(String, String)],
    ) -> Result<Option<String>, StoreError> {
        self.inner.exact_lookup(side, kvs)
    }
    fn common_id_insert(
        &self,
        side: Side,
        common_id: &str,
        record_id: &str,
    ) -> Result<(), StoreError> {
        self.inner.common_id_insert(side, common_id, record_id)
    }
    fn common_id_lookup(&self, side: Side, common_id: &str) -> Result<Option<String>, StoreError> {
        self.inner.common_id_lookup(side, common_id)
    }
    fn common_id_remove(&self, side: Side, common_id: &str) -> Result<(), StoreError> {
        self.inner.common_id_remove(side, common_id)
    }
    fn common_id_remove_owned(
        &self,
        side: Side,
        common_id: &str,
        record_id: &str,
    ) -> Result<(), StoreError> {
        self.inner
            .common_id_remove_owned(side, common_id, record_id)
    }
    fn persist_review(
        &self,
        key: &str,
        id: &str,
        side: Side,
        candidate_id: &str,
        score: f64,
    ) -> Result<(), StoreError> {
        self.inner
            .persist_review(key, id, side, candidate_id, score)
    }
    fn remove_query_reviews(&self, side: Side, id: &str) -> Result<(), StoreError> {
        self.inner.remove_query_reviews(side, id)
    }
    fn remove_reviews_for_id(&self, side: Side, id: &str) -> Result<(), StoreError> {
        self.inner.remove_reviews_for_id(side, id)
    }
    fn remove_reviews_for_pair(&self, a_id: &str, b_id: &str) -> Result<(), StoreError> {
        self.inner.remove_reviews_for_pair(a_id, b_id)
    }
    fn load_reviews(&self) -> Result<Vec<ReviewEntry>, StoreError> {
        self.inner.load_reviews()
    }
}

#[test]
fn startup_graph_store_failures_prevent_publication() -> TestResult {
    for (stage, side) in [
        ("get", Side::A),
        ("get", Side::B),
        ("len", Side::A),
        ("unmatched", Side::B),
        ("blocking", Side::A),
        ("ids", Side::A),
    ] {
        let mut fixture = make_fixture(false, None)?;
        let session = Arc::get_mut(&mut fixture.session).expect("unique test session");
        let state = Arc::get_mut(&mut session.state).expect("unique test state");
        state.config.blocking.enabled = stage == "blocking";
        for side in [Side::A, Side::B] {
            let id = if side == Side::A { "a" } else { "b" };
            let record = make_record(side, id, "0");
            state.store.insert(side, id, &record)?;
            state.store.mark_unmatched(side, id)?;
            state
                .side(side)
                .combined_index
                .as_ref()
                .expect("test index")
                .upsert(id, &[1.0, 0.0], &record, side)?;
        }
        state.store = Arc::new(FaultStore {
            inner: state.store.clone(),
            fail_side: side,
            fail_stage: stage,
        });
        assert!(
            session.initial_match_pass().is_err(),
            "store {stage} on {side:?} must abort startup"
        );
        assert!(
            session.state.crossmap.pairs().is_empty(),
            "no pair published after {stage} error"
        );
        assert!(
            session.state.review_queue.is_empty(),
            "no review published after {stage} error"
        );
    }
    Ok(())
}
