//! Flat brute-force vector index.
//!
//! Stores N vectors of dimension D in a flat `Vec<f32>` matrix (row-major).
//! Supports incremental insert, update, delete, and brute-force top-K search
//! via dot product.

pub mod cache;
pub mod flat;

pub use flat::VecIndex;
