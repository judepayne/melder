//! Synonym matching: acronym generation, bidirectional index, and binary scorer.
//!
//! Bridges a blind spot that no other scoring method can address: when one side
//! uses an acronym (e.g. "HWAG") and the other the full name ("Harris, Watkins
//! and Goodwin BV"), embeddings, BM25, and fuzzy matching all score near zero
//! because there is no shared semantic, lexical, or character-level content.
//!
//! Three components:
//! - **Generator** — produces candidate acronyms from a full entity name.
//! - **Index** — bidirectional HashMap mapping acronyms ↔ record IDs.
//! - **Scorer** — binary 1.0/0.0 scorer for use in the composite scoring equation.

pub mod generator;
pub mod index;
pub mod scorer;
