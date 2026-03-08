//! Index cache serialization: save/load VecIndex to/from binary files.
//!
//! Binary format:
//! ```text
//! [4 bytes] N (u32 little-endian) — number of vectors
//! [4 bytes] D (u32 little-endian) — dimension
//! [N*D*4 bytes] vectors (f32 little-endian, row-major)
//! [variable] N newline-separated ID strings (UTF-8)
//! ```

use std::io::{Read, Write};
use std::path::Path;

use crate::error::IndexError;
use crate::index::VecIndex;

/// Save a VecIndex to a binary cache file.
pub fn save_index(path: &Path, index: &VecIndex) -> Result<(), IndexError> {
    let n = index.len() as u32;
    let d = index.dim() as u32;

    let mut file = std::fs::File::create(path)?;

    // Write header
    file.write_all(&n.to_le_bytes())?;
    file.write_all(&d.to_le_bytes())?;

    // Write vectors as raw f32 bytes (little-endian)
    let vectors = index.vectors();
    let byte_slice =
        unsafe { std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4) };
    file.write_all(byte_slice)?;

    // Write IDs as newline-separated strings
    for id in index.ids() {
        file.write_all(id.as_bytes())?;
        file.write_all(b"\n")?;
    }

    file.flush()?;
    Ok(())
}

/// Load a VecIndex from a binary cache file.
pub fn load_index(path: &Path) -> Result<VecIndex, IndexError> {
    let mut file = std::fs::File::open(path)?;

    // Read header
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    file.read_exact(&mut buf4)?;
    let d = u32::from_le_bytes(buf4) as usize;

    // Read vectors
    let vec_bytes = n * d * 4;
    let mut vec_buf = vec![0u8; vec_bytes];
    file.read_exact(&mut vec_buf)?;

    // Convert bytes to f32 (assumes little-endian platform — true for x86/ARM)
    let vectors: Vec<f32> = vec_buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Read IDs
    let mut id_buf = String::new();
    file.read_to_string(&mut id_buf)?;
    let ids: Vec<String> = id_buf
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();

    if ids.len() != n {
        return Err(IndexError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cache file corrupt: header says {} vectors but found {} IDs",
                n,
                ids.len()
            ),
        )));
    }

    Ok(VecIndex::from_parts(vectors, d, ids))
}

/// Check whether a cache file is stale relative to current data.
///
/// Returns `true` if the cache should be rebuilt:
/// - File doesn't exist
/// - Record count doesn't match
/// - ID hash doesn't match (not implemented yet — just checks count)
pub fn is_cache_stale(path: &Path, record_count: usize) -> bool {
    if !path.exists() {
        return true;
    }

    // Quick check: read just the header to get N
    let Ok(mut file) = std::fs::File::open(path) else {
        return true;
    };

    let mut buf4 = [0u8; 4];
    if file.read_exact(&mut buf4).is_err() {
        return true;
    }
    let cached_n = u32::from_le_bytes(buf4) as usize;

    cached_n != record_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_test_index() -> VecIndex {
        let dim = 4;
        let mut index = VecIndex::new(dim);
        for i in 0..100 {
            let id = format!("id_{}", i);
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
            index.upsert(&id, &vec);
        }
        index
    }

    #[test]
    fn save_and_load_roundtrip() {
        let index = make_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.index");

        save_index(&path, &index).unwrap();
        let loaded = load_index(&path).unwrap();

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.dim(), index.dim());

        // Verify all vectors are bitwise identical
        for id in index.ids() {
            let orig = index.get(id).unwrap();
            let loaded_v = loaded.get(id).unwrap();
            assert_eq!(orig.len(), loaded_v.len());
            for (a, b) in orig.iter().zip(loaded_v.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "vector mismatch for id {}", id);
            }
        }
    }

    #[test]
    fn save_empty_index() {
        let index = VecIndex::new(384);
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.index");

        save_index(&path, &index).unwrap();
        let loaded = load_index(&path).unwrap();

        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.dim(), 384);
    }

    #[test]
    fn staleness_missing_file() {
        assert!(is_cache_stale(Path::new("/nonexistent/file.index"), 100));
    }

    #[test]
    fn staleness_count_mismatch() {
        let index = make_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.index");

        save_index(&path, &index).unwrap();

        // Same count — not stale
        assert!(!is_cache_stale(&path, 100));

        // Different count — stale
        assert!(is_cache_stale(&path, 101));
        assert!(is_cache_stale(&path, 99));
    }

    #[test]
    fn search_after_load() {
        let dim = 4;
        let mut index = VecIndex::new(dim);
        index.upsert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.upsert("b", &[0.0, 1.0, 0.0, 0.0]);
        index.upsert("c", &[0.707, 0.707, 0.0, 0.0]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.index");

        save_index(&path, &index).unwrap();
        let loaded = load_index(&path).unwrap();

        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > 0.99);
    }
}
