//! BM25 full-text search index for candidate filtering and scoring.
//!
//! Wraps Tantivy to provide BM25-based retrieval and scoring over
//! record text fields. Gated behind `--features bm25`.

pub mod index;
pub mod scorer;
