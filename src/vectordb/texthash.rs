//! Text-hash deduplication store for incremental cache updates.
//!
//! `TextHashStore` tracks a FNV-1a hash of each record's source text so that
//! on a re-run we can identify which records changed and only re-encode those,
//! skipping the unchanged majority.
//!
//! Also fixes the silent correctness bug where a dataset swap (delete + insert,
//! same record count) passes the old count-based staleness check and silently
//! reuses wrong cached vectors.
//!
//! ## Sidecar format (`{cache_path}.texthash`)
//!
//! ```text
//! [4 bytes]   N         (u32 le) — number of hash entries
//! For each of N entries:
//!   [4 bytes] id_len    (u32 le)
//!   [id_len]  id        (UTF-8)
//!   [8 bytes] hash      (u64 le)
//! [4 bytes]   json_len  (u32 le) — byte length of emb_specs JSON
//! [json_len]  emb_specs (UTF-8 JSON array of [field_a, field_b, weight] triples)
//! ```
//!
//! The `emb_specs` section allows `load` to reconstruct the store without
//! requiring an emb_specs parameter — keeping `load_index` signatures unchanged.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use crate::models::{Record, Side};

// ---------------------------------------------------------------------------
// compute_text_hash — pure function
// ---------------------------------------------------------------------------

/// Compute a FNV-1a hash of a record's embedding field texts.
///
/// For each spec (in order), the relevant field text (trimmed) is hashed with
/// a NUL byte separator between fields. The weight is intentionally excluded —
/// weight changes are caught by the manifest's spec_hash, which forces a full
/// cold rebuild before any text-hash diff is attempted.
pub fn compute_text_hash(record: &Record, emb_specs: &[(String, String, f64)], side: Side) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (field_a, field_b, _weight) in emb_specs {
        let field = match side {
            Side::A => field_a,
            Side::B => field_b,
        };
        let text = record.get(field).map(|s| s.trim()).unwrap_or("");
        for byte in text.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x00000100000001b3);
        }
        // Field separator prevents "ab"+"c" colliding with "a"+"bc"
        // 0xFF is used as a field separator — relies on UTF-8 text never
        // containing raw 0xFF bytes.
        h ^= 0xFFu64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// TextHashStore
// ---------------------------------------------------------------------------

/// Per-side text-hash store embedded inside each VectorDB implementation.
///
/// When `emb_specs` is empty (e.g. test DBs constructed via `new(dim)`),
/// `update()` becomes a no-op and `all()` always returns an empty map.
/// This degrades gracefully: `build_or_load_combined_index` will see all
/// records as "new" and fall through to a cold rebuild.
pub struct TextHashStore {
    hashes: HashMap<String, u64>,
    emb_specs: Vec<(String, String, f64)>,
}

impl TextHashStore {
    /// Create a new store with the given embedding specs.
    pub fn new(emb_specs: Vec<(String, String, f64)>) -> Self {
        Self {
            hashes: HashMap::new(),
            emb_specs,
        }
    }

    /// Create an empty store with no specs (used by backends constructed
    /// without emb_specs, e.g. in tests).
    pub fn empty() -> Self {
        Self {
            hashes: HashMap::new(),
            emb_specs: Vec::new(),
        }
    }

    /// Record the text hash for `id` after an upsert.
    ///
    /// No-op if emb_specs is empty (test / no-embedding context).
    pub fn update(&mut self, id: &str, record: &Record, side: Side) {
        if self.emb_specs.is_empty() {
            return;
        }
        let h = compute_text_hash(record, &self.emb_specs, side);
        self.hashes.insert(id.to_string(), h);
    }

    /// Remove the hash entry for `id`.
    pub fn remove(&mut self, id: &str) {
        self.hashes.remove(id);
    }

    /// Retrieve the stored hash for `id`.
    pub fn get(&self, id: &str) -> Option<u64> {
        self.hashes.get(id).copied()
    }

    /// All stored hashes, as a reference to the internal map.
    pub fn all(&self) -> &HashMap<String, u64> {
        &self.hashes
    }

    /// Number of stored hash entries.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Returns `true` if no hash entries are stored.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    // ---------------------------------------------------------------------------
    // Persistence
    // ---------------------------------------------------------------------------

    /// Write the sidecar to `{cache_path}.texthash`.
    pub fn save(&self, cache_path: &Path) -> Result<(), std::io::Error> {
        let path = texthash_sidecar_path(cache_path);
        let mut file = std::fs::File::create(&path)?;

        // Header: N entries
        let n = self.hashes.len() as u32;
        file.write_all(&n.to_le_bytes())?;

        // Entries
        for (id, &hash) in &self.hashes {
            let id_bytes = id.as_bytes();
            file.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
            file.write_all(id_bytes)?;
            file.write_all(&hash.to_le_bytes())?;
        }

        // emb_specs as length-prefixed JSON
        let json = serde_json::to_string(&self.emb_specs)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let json_bytes = json.as_bytes();
        file.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
        file.write_all(json_bytes)?;

        file.flush()?;
        Ok(())
    }

    /// Load a sidecar from `{cache_path}.texthash`.
    ///
    /// Returns an empty store (with no emb_specs) if the sidecar is absent —
    /// this is treated as "all records are new" by the diff logic, which falls
    /// through to a cold rebuild via the 90% threshold.
    pub fn load(cache_path: &Path) -> Result<Self, std::io::Error> {
        let path = texthash_sidecar_path(cache_path);
        if !path.exists() {
            return Ok(Self::empty());
        }

        let mut file = std::fs::File::open(&path)?;
        let mut buf4 = [0u8; 4];

        // N entries
        file.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;

        let mut hashes = HashMap::with_capacity(n);
        for _ in 0..n {
            // id
            file.read_exact(&mut buf4)?;
            let id_len = u32::from_le_bytes(buf4) as usize;
            if id_len > 10_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("implausible id_len {} in texthash sidecar", id_len),
                ));
            }
            let mut id_bytes = vec![0u8; id_len];
            file.read_exact(&mut id_bytes)?;
            let id = String::from_utf8(id_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

            // hash
            let mut hash_buf = [0u8; 8];
            file.read_exact(&mut hash_buf)?;
            let hash = u64::from_le_bytes(hash_buf);

            hashes.insert(id, hash);
        }

        // emb_specs
        file.read_exact(&mut buf4)?;
        let json_len = u32::from_le_bytes(buf4) as usize;
        if json_len > 1_000_000 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("implausible json_len {} in texthash sidecar", json_len),
            ));
        }
        let mut json_bytes = vec![0u8; json_len];
        file.read_exact(&mut json_bytes)?;
        let emb_specs: Vec<(String, String, f64)> = serde_json::from_slice(&json_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(Self { hashes, emb_specs })
    }
}

// ---------------------------------------------------------------------------
// Sidecar path helpers
// ---------------------------------------------------------------------------

/// Path of the texthash sidecar for a given cache base path.
///
/// Pattern: `{cache_path}.texthash`
/// Example: `bench/cache/a.combined_embedding_a3f7c2b1.index.texthash`
pub fn texthash_sidecar_path(cache_path: &Path) -> std::path::PathBuf {
    let name = cache_path
        .file_name()
        .map(|n| format!("{}.texthash", n.to_string_lossy()))
        .unwrap_or_else(|| "cache.texthash".to_string());
    cache_path.with_file_name(name)
}

/// Delete the texthash sidecar (if it exists).
///
/// Used by `cmd_cache_clear` to keep sidecars in sync with index files.
pub fn delete_texthash(cache_path: &Path) {
    let p = texthash_sidecar_path(cache_path);
    if p.exists() {
        let _ = std::fs::remove_file(&p);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn rec(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn specs() -> Vec<(String, String, f64)> {
        vec![
            ("legal_name".to_string(), "cp_name".to_string(), 0.55),
            ("short_name".to_string(), "cp_name".to_string(), 0.20),
        ]
    }

    // -----------------------------------------------------------------------
    // compute_text_hash
    // -----------------------------------------------------------------------

    #[test]
    fn hash_deterministic() {
        let r = rec(&[("legal_name", "Acme Corp"), ("short_name", "Acme")]);
        let s = specs();
        let h1 = compute_text_hash(&r, &s, Side::A);
        let h2 = compute_text_hash(&r, &s, Side::A);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_changes_on_field_value_change() {
        let r1 = rec(&[("legal_name", "Acme Corp")]);
        let r2 = rec(&[("legal_name", "Acme Corporation")]);
        let s = vec![("legal_name".to_string(), "cp_name".to_string(), 0.55)];
        assert_ne!(
            compute_text_hash(&r1, &s, Side::A),
            compute_text_hash(&r2, &s, Side::A)
        );
    }

    #[test]
    fn hash_uses_correct_side_field() {
        let r = rec(&[("legal_name", "A"), ("cp_name", "B")]);
        let s = vec![("legal_name".to_string(), "cp_name".to_string(), 0.55)];
        let ha = compute_text_hash(&r, &s, Side::A); // uses "legal_name" → "A"
        let hb = compute_text_hash(&r, &s, Side::B); // uses "cp_name" → "B"
        assert_ne!(ha, hb);
    }

    #[test]
    fn hash_empty_specs_is_stable() {
        let r = rec(&[("legal_name", "X")]);
        let h1 = compute_text_hash(&r, &[], Side::A);
        let h2 = compute_text_hash(&r, &[], Side::A);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_missing_field_vs_empty_string_differ() {
        // Missing field → empty text → but both should produce same hash
        let r_missing = rec(&[]);
        let r_empty = rec(&[("legal_name", "")]);
        let s = vec![("legal_name".to_string(), "cp_name".to_string(), 0.55)];
        // Both produce the same hash (missing field treated as "")
        assert_eq!(
            compute_text_hash(&r_missing, &s, Side::A),
            compute_text_hash(&r_empty, &s, Side::A)
        );
    }

    // -----------------------------------------------------------------------
    // TextHashStore
    // -----------------------------------------------------------------------

    #[test]
    fn update_and_get() {
        let mut store = TextHashStore::new(specs());
        let r = rec(&[("legal_name", "Acme Corp"), ("short_name", "Acme")]);
        store.update("id1", &r, Side::A);
        assert!(store.get("id1").is_some());
    }

    #[test]
    fn update_with_empty_specs_is_noop() {
        let mut store = TextHashStore::empty();
        let r = rec(&[("legal_name", "Acme Corp")]);
        store.update("id1", &r, Side::A);
        assert!(store.get("id1").is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn remove_works() {
        let mut store = TextHashStore::new(specs());
        let r = rec(&[("legal_name", "Acme")]);
        store.update("id1", &r, Side::A);
        assert!(store.get("id1").is_some());
        store.remove("id1");
        assert!(store.get("id1").is_none());
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("test.index");

        let mut store = TextHashStore::new(specs());
        let r1 = rec(&[("legal_name", "Acme Corp"), ("short_name", "Acme")]);
        let r2 = rec(&[("legal_name", "Globex"), ("short_name", "GLX")]);
        store.update("id1", &r1, Side::A);
        store.update("id2", &r2, Side::A);

        store.save(&cache_path).unwrap();

        let loaded = TextHashStore::load(&cache_path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("id1"), store.get("id1"));
        assert_eq!(loaded.get("id2"), store.get("id2"));
        assert_eq!(loaded.emb_specs, store.emb_specs);
    }

    #[test]
    fn load_missing_sidecar_returns_empty() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("nonexistent.index");
        let store = TextHashStore::load(&cache_path).unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.emb_specs.is_empty());
    }

    #[test]
    fn sidecar_path_pattern() {
        let p = std::path::Path::new("bench")
            .join("cache")
            .join("a.combined_embedding_abc.index");
        let sp = texthash_sidecar_path(&p);
        let expected = std::path::Path::new("bench")
            .join("cache")
            .join("a.combined_embedding_abc.index.texthash");
        assert_eq!(sp, expected);
    }

    #[test]
    fn delete_texthash_removes_sidecar() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("test.index");
        let mut store = TextHashStore::new(specs());
        store.update("x", &rec(&[("legal_name", "X")]), Side::A);
        store.save(&cache_path).unwrap();
        assert!(texthash_sidecar_path(&cache_path).exists());
        delete_texthash(&cache_path);
        assert!(!texthash_sidecar_path(&cache_path).exists());
    }
}
