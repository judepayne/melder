//! CrossMap: bidirectional record-pair mapping.
//!
//! The `CrossMapOps` trait defines the runtime interface. `MemoryCrossMap`
//! is the in-memory (RwLock + HashMap) implementation with CSV persistence.

pub mod memory;
pub mod traits;

pub use memory::MemoryCrossMap;
pub use traits::CrossMapOps;
