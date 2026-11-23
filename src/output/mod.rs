//! Output pipeline: build CSVs, Parquet, and/or SQLite DB from match log.

pub mod build;
mod columns;
pub mod csv;
pub mod db;
pub mod manifest;
#[cfg(feature = "parquet-format")]
pub mod parquet;
pub mod scoring_log;
pub mod shortlist;

pub use build::build_outputs;
pub use manifest::{BuildReport, OutputManifest};
