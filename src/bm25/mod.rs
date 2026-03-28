//! BM25 full-text search index for candidate filtering and scoring.
//!
//! Uses a DashMap-based lock-free scorer (`SimpleBm25`) that provides
//! instant write visibility and concurrent reads without external locks.

pub mod scorer;
pub mod simple;
