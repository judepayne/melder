//! Output pipeline: build CSVs and/or SQLite DB from match log.

pub mod build;
pub mod csv;
pub mod db;
pub mod manifest;
pub mod scoring_log;

pub use build::build_outputs;
pub use manifest::{BuildReport, OutputManifest};
