//! Encoder pool for embedding inference via `fastembed`.
//!
//! Wraps one or more `TextEmbedding` instances behind `Mutex` locks.
//! Each instance is an independent ONNX session with its own tokenizer
//! state. Model weights are memory-mapped by the OS, so multiple instances
//! share the same physical pages.

pub mod pool;

pub use pool::EncoderPool;
