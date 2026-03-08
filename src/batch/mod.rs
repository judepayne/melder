//! Batch matching engine: process all B records against A pool.

pub mod engine;
pub mod writer;

pub use engine::{run_batch, BatchResult, BatchStats};
pub use writer::{write_results_csv, write_review_csv, write_unmatched_csv};
