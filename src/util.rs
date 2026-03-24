//! Shared utility functions used across multiple modules.

use std::path::Path;

/// Cross-platform rename that replaces the destination if it exists.
///
/// On Unix `fs::rename` atomically replaces the target. On Windows it fails
/// if the destination already exists, so we remove-then-rename (tiny window
/// of non-atomicity, acceptable for crossmap flush and WAL compaction).
pub fn rename_replacing(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        std::fs::rename(from, to)
    }
    #[cfg(not(unix))]
    {
        let _ = std::fs::remove_file(to);
        std::fs::rename(from, to)
    }
}

/// Compute a FNV-1a hash over raw bytes, returning the full u64 value.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &byte in data {
        h ^= byte as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h
}

/// Compute a FNV-1a hash of a string, returning an 8-character hex digest
/// (truncated to 32-bit display).
pub fn fnv1a_8(s: &str) -> String {
    format!("{:08x}", fnv1a(s.as_bytes()) as u32)
}
