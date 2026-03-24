//! Cache manifest: detect config changes between runs.
//!
//! A `.manifest` sidecar sits alongside each cache file:
//!   `{cache_path}.manifest`   (JSON, human-readable)
//!
//! On load: compare stored hashes against the current config.
//! Any mismatch forces a cold rebuild with an informative log message.
//! On save: write (or overwrite) the manifest with current config state.
//!
//! This closes two correctness gaps:
//! 1. Blocking config changes (adding/removing a blocking field) used to
//!    silently reuse a stale index. Now they trigger a logged cold rebuild.
//! 2. Model changes are detected and reported before any scoring happens.

use std::path::Path;

use crate::config::schema::BlockingConfig;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Why a cached index needs to be rebuilt.
#[derive(Debug, Clone, PartialEq)]
pub enum StaleReason {
    /// Cache is present and config matches — load it.
    Fresh,
    /// No manifest file found (new cache or pre-manifest cache).
    Missing,
    /// Embedding field spec (fields / weights) changed.
    SpecChanged,
    /// Blocking config (enabled / operator / fields) changed.
    BlockingChanged,
    /// Embedding model name changed.
    ModelChanged,
}

impl std::fmt::Display for StaleReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaleReason::Fresh => write!(f, "fresh"),
            StaleReason::Missing => {
                write!(f, "no manifest (first run or pre-manifest cache)")
            }
            StaleReason::SpecChanged => write!(f, "embedding field spec changed"),
            StaleReason::BlockingChanged => write!(f, "blocking config changed"),
            StaleReason::ModelChanged => write!(f, "embedding model changed"),
        }
    }
}

/// Serialised manifest stored as a `.manifest` JSON sidecar.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheManifest {
    /// 8-char FNV hex of the embedding field spec (fields + weights).
    pub spec_hash: String,
    /// 8-char FNV hex of the blocking configuration.
    pub blocking_hash: String,
    /// Embedding model name (e.g. "all-MiniLM-L6-v2").
    pub model: String,
    /// Number of records in the index at build time (informational).
    pub record_count: usize,
    /// RFC3339 UTC timestamp of when the index was built (informational).
    pub built_at: String,
}

// ---------------------------------------------------------------------------
// blocking_hash
// ---------------------------------------------------------------------------

/// Compute an 8-character FNV-1a hex hash of the blocking configuration.
///
/// Input: `"{enabled}/{operator}/{sorted field pairs}"`
/// Pairs are sorted so that reordering them in the YAML doesn't invalidate
/// the cache.
pub fn blocking_hash(bc: &BlockingConfig) -> String {
    let mut pairs: Vec<String> = bc
        .fields
        .iter()
        .map(|p| format!("{}:{}", p.field_a, p.field_b))
        .collect();
    pairs.sort();
    let s = format!("{}/{}/{}", bc.enabled, bc.operator, pairs.join(","));
    fnv1a_8(&s)
}

// ---------------------------------------------------------------------------
// Read / write / check
// ---------------------------------------------------------------------------

/// Path of the manifest sidecar for a given cache base path.
///
/// Pattern: `{cache_path}.manifest`
/// Example: `bench/cache/a.combined_embedding_a3f7c2b1.index.manifest`
pub fn manifest_path(cache_path: &Path) -> std::path::PathBuf {
    let name = cache_path
        .file_name()
        .map(|n| format!("{}.manifest", n.to_string_lossy()))
        .unwrap_or_else(|| "cache.manifest".to_string());
    cache_path.with_file_name(name)
}

/// Write (or overwrite) the manifest sidecar.
pub fn write_manifest(cache_path: &Path, manifest: &CacheManifest) -> Result<(), std::io::Error> {
    let json =
        serde_json::to_string_pretty(manifest).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(manifest_path(cache_path), json)
}

/// Read the manifest sidecar.
///
/// Returns `None` if the file is absent (not an error — first run).
/// Returns an error only if the file exists but can't be parsed.
pub fn read_manifest(cache_path: &Path) -> Result<Option<CacheManifest>, std::io::Error> {
    let p = manifest_path(cache_path);
    if !p.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(&p)?;
    let m: CacheManifest = serde_json::from_str(&data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(Some(m))
}

/// Compare the stored manifest against the current config.
///
/// Returns `StaleReason::Fresh` only if the manifest exists and all hashes
/// match the current config. Any other outcome forces a cold rebuild.
pub fn check_manifest(
    cache_path: &Path,
    current_spec_hash: &str,
    current_blocking_hash: &str,
    current_model: &str,
) -> StaleReason {
    match read_manifest(cache_path) {
        Err(e) => {
            eprintln!(
                "Warning: could not read cache manifest ({}), treating as missing",
                e
            );
            StaleReason::Missing
        }
        Ok(None) => StaleReason::Missing,
        Ok(Some(m)) => {
            if m.model != current_model {
                StaleReason::ModelChanged
            } else if m.spec_hash != current_spec_hash {
                StaleReason::SpecChanged
            } else if m.blocking_hash != current_blocking_hash {
                StaleReason::BlockingChanged
            } else {
                StaleReason::Fresh
            }
        }
    }
}

/// Delete the manifest sidecar (if it exists).
///
/// Used by `cmd_cache_clear` to keep sidecars in sync with index files.
pub fn delete_manifest(cache_path: &Path) {
    let p = manifest_path(cache_path);
    if p.exists() {
        let _ = std::fs::remove_file(&p);
    }
}

/// Build a `CacheManifest` for the current config state.
pub fn make_manifest(
    spec_hash: String,
    blocking_hash: String,
    model: String,
    record_count: usize,
) -> CacheManifest {
    CacheManifest {
        spec_hash,
        blocking_hash,
        model,
        record_count,
        built_at: now_rfc3339(),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Re-export shared FNV-1a hash from util.
pub(crate) use crate::util::fnv1a_8;

/// Simple RFC3339 UTC timestamp using only `std::time`.
///
/// Algorithm from https://howardhinnant.github.io/date_algorithms.html
pub(crate) fn now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Civil date from days since 1970-01-01 (which is 0000-03-01 + 719468 days)
    let z = secs / 86400 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month primum [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    let hh = (secs / 3600) % 24;
    let mm = (secs / 60) % 60;
    let ss = secs % 60;

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, hh, mm, ss)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{BlockingConfig, BlockingFieldPair};
    use tempfile::tempdir;

    fn bc(enabled: bool, operator: &str, pairs: &[(&str, &str)]) -> BlockingConfig {
        BlockingConfig {
            enabled,
            operator: operator.to_string(),
            fields: pairs
                .iter()
                .map(|(a, b)| BlockingFieldPair {
                    field_a: a.to_string(),
                    field_b: b.to_string(),
                })
                .collect(),
            field_a: None,
            field_b: None,
        }
    }

    #[test]
    fn blocking_hash_deterministic() {
        let c = bc(true, "and", &[("country_a", "country_b")]);
        assert_eq!(blocking_hash(&c), blocking_hash(&c));
        assert_eq!(blocking_hash(&c).len(), 8);
    }

    #[test]
    fn blocking_hash_changes_on_enabled_toggle() {
        let c1 = bc(true, "and", &[("x", "y")]);
        let c2 = bc(false, "and", &[("x", "y")]);
        assert_ne!(blocking_hash(&c1), blocking_hash(&c2));
    }

    #[test]
    fn blocking_hash_changes_on_operator() {
        let c1 = bc(true, "and", &[("x", "y")]);
        let c2 = bc(true, "or", &[("x", "y")]);
        assert_ne!(blocking_hash(&c1), blocking_hash(&c2));
    }

    #[test]
    fn blocking_hash_stable_on_field_reorder() {
        let c1 = bc(true, "and", &[("aa", "bb"), ("cc", "dd")]);
        let c2 = bc(true, "and", &[("cc", "dd"), ("aa", "bb")]);
        assert_eq!(blocking_hash(&c1), blocking_hash(&c2));
    }

    #[test]
    fn manifest_roundtrip() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("a.combined_embedding_abc12345.index");
        let m = CacheManifest {
            spec_hash: "abc12345".to_string(),
            blocking_hash: "deadbeef".to_string(),
            model: "all-MiniLM-L6-v2".to_string(),
            record_count: 1000,
            built_at: "2026-01-01T00:00:00Z".to_string(),
        };
        write_manifest(&cache_path, &m).unwrap();
        let loaded = read_manifest(&cache_path).unwrap().unwrap();
        assert_eq!(loaded.spec_hash, m.spec_hash);
        assert_eq!(loaded.blocking_hash, m.blocking_hash);
        assert_eq!(loaded.model, m.model);
        assert_eq!(loaded.record_count, 1000);
    }

    #[test]
    fn check_missing() {
        let dir = tempdir().unwrap();
        let reason = check_manifest(&dir.path().join("none.index"), "a", "b", "m");
        assert_eq!(reason, StaleReason::Missing);
    }

    #[test]
    fn check_fresh() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("test.index");
        write_manifest(
            &p,
            &CacheManifest {
                spec_hash: "s".to_string(),
                blocking_hash: "b".to_string(),
                model: "m".to_string(),
                record_count: 1,
                built_at: "2026-01-01T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        assert_eq!(check_manifest(&p, "s", "b", "m"), StaleReason::Fresh);
    }

    #[test]
    fn check_model_changed() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("test.index");
        write_manifest(
            &p,
            &CacheManifest {
                spec_hash: "s".to_string(),
                blocking_hash: "b".to_string(),
                model: "old".to_string(),
                record_count: 1,
                built_at: "2026-01-01T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        assert_eq!(
            check_manifest(&p, "s", "b", "new"),
            StaleReason::ModelChanged
        );
    }

    #[test]
    fn check_spec_changed() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("test.index");
        write_manifest(
            &p,
            &CacheManifest {
                spec_hash: "old-spec".to_string(),
                blocking_hash: "b".to_string(),
                model: "m".to_string(),
                record_count: 1,
                built_at: "2026-01-01T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        assert_eq!(
            check_manifest(&p, "new-spec", "b", "m"),
            StaleReason::SpecChanged
        );
    }

    #[test]
    fn check_blocking_changed() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("test.index");
        write_manifest(
            &p,
            &CacheManifest {
                spec_hash: "s".to_string(),
                blocking_hash: "old-block".to_string(),
                model: "m".to_string(),
                record_count: 1,
                built_at: "2026-01-01T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        assert_eq!(
            check_manifest(&p, "s", "new-block", "m"),
            StaleReason::BlockingChanged
        );
    }

    #[test]
    fn delete_removes_sidecar() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("test.index");
        write_manifest(
            &p,
            &CacheManifest {
                spec_hash: "s".to_string(),
                blocking_hash: "b".to_string(),
                model: "m".to_string(),
                record_count: 1,
                built_at: "2026-01-01T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        assert!(manifest_path(&p).exists());
        delete_manifest(&p);
        assert!(!manifest_path(&p).exists());
    }

    #[test]
    fn now_rfc3339_looks_reasonable() {
        let ts = now_rfc3339();
        // Should be "YYYY-MM-DDTHH:MM:SSZ"
        assert_eq!(ts.len(), 20, "timestamp = {}", ts);
        assert!(ts.ends_with('Z'));
        assert!(ts.starts_with("20")); // year 2000+
    }
}
