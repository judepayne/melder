//! Safety checks for the disposable SQLite batch database.

use std::path::{Component, Path, PathBuf};

use crate::config::schema::{Config, MatchMethod};
use crate::error::ConfigError;

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::InvalidValue {
        field: "batch.db_path".into(),
        message: message.into(),
    }
}

// Resolve existing ancestors (including directory symlinks), retaining missing
// leaves. Paths in config are relative to the process cwd, not the YAML file.
fn resolve(path: &Path) -> std::io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            other => {
                resolved.push(other.as_os_str());
                match std::fs::canonicalize(&resolved) {
                    Ok(path) => resolved = path,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e),
                }
            }
        }
    }
    Ok(resolved)
}

// Missing leaves cannot be canonicalized on case-insensitive volumes.
// Reserve equivalent spellings conservatively, including on case-sensitive
// volumes, without writing probes in user directories.
fn collision_key(path: &Path) -> String {
    let n = icu_normalizer::DecomposingNormalizer::new_nfkd();
    let name = path.to_string_lossy();
    let folded = n.normalize(&name).to_uppercase().to_lowercase();
    n.normalize(&folded).into_owned()
}

fn family(path: &Path) -> Vec<PathBuf> {
    ["", "-wal", "-shm", "-journal"]
        .iter()
        .map(|suffix| {
            let mut name = path.as_os_str().to_os_string();
            name.push(suffix);
            PathBuf::from(name)
        })
        .collect()
}

/// Reject destructive scratch reuse and unsupported SQLite batch scoring.
pub(crate) fn validate_scratch(
    cfg: &Config,
    config_path: Option<&Path>,
) -> Result<(), ConfigError> {
    let Some(scratch) = cfg.batch.db_path.as_deref() else {
        return Ok(());
    };
    if scratch.trim().is_empty() {
        return Err(invalid(
            "must be a non-empty path to a new disposable database",
        ));
    }
    if cfg
        .match_fields
        .iter()
        .any(|f| f.method == MatchMethod::Embedding)
    {
        return Err(ConfigError::InvalidValue {
            field: "match_fields".into(),
            message: "embedding match fields are not supported with batch.db_path (SQLite batch has no embedding index); use memory batch mode".into(),
        });
    }
    let scratch_family = family(Path::new(scratch));
    for path in &scratch_family {
        match std::fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(invalid(format!(
                    "scratch database and sidecars must not already exist: {}",
                    path.display()
                )));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(invalid(format!("cannot inspect {}: {e}", path.display()))),
        }
    }

    let crossmap = Path::new(cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv"));
    let mut protected = vec![
        crossmap.with_extension("tmp"),
        PathBuf::from("bench/upsert.wal"),
        PathBuf::from(&cfg.datasets.a.path),
        PathBuf::from(&cfg.datasets.b.path),
        PathBuf::from(cfg.cross_map.path.as_deref().unwrap_or("crossmap.csv")),
    ];
    for path in [
        cfg.exclusions.path.as_deref(),
        cfg.synonym_dictionary.as_ref().map(|s| s.path.as_str()),
        cfg.live.match_log_path.as_deref(),
        Some(cfg.embeddings.model.as_str()),
    ]
    .into_iter()
    .flatten()
    {
        protected.push(PathBuf::from(path));
    }
    if let Some(path) = config_path {
        protected.push(path.to_path_buf());
    }
    for path in [cfg.output.db_path.as_deref(), cfg.live.db_path.as_deref()]
        .into_iter()
        .flatten()
    {
        protected.extend(family(Path::new(path)));
    }
    // SQLite batch emits CSV in cwd when csv_dir_path is absent, even with
    // only SQLite/Parquet output selected. Include atomic-writer temp names.
    for (dir, extension) in [
        (
            Some(cfg.output.csv_dir_path.as_deref().unwrap_or(".")),
            "csv",
        ),
        (cfg.output.parquet_dir_path.as_deref(), "parquet"),
    ] {
        if let Some(dir) = dir {
            for name in ["relationships", "unmatched", "candidates"] {
                let path = Path::new(dir).join(format!("{name}.{extension}"));
                protected.push(path.with_extension(format!("{extension}.tmp")));
                protected.push(path);
            }
        }
    }
    for scratch in scratch_family {
        let scratch = collision_key(&resolve(&scratch)?);
        for path in &protected {
            if !path.as_os_str().is_empty() && scratch == collision_key(&resolve(path)?) {
                return Err(invalid(format!(
                    "scratch database or sidecar overlaps protected path {}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equivalent_unicode_output_names_share_a_collision_key() {
        assert_eq!(
            collision_key(Path::new("RÉSUMÉ.db")),
            collision_key(Path::new("re\u{301}sume\u{301}.DB")),
            "different case and Unicode composition must not disguise a scratch/output collision"
        );
    }

    fn make_config(root: &Path) -> Config {
        let yaml = format!(
            r#"
job: {{ name: scratch_test }}
datasets:
  a: {{ path: a.csv, id_field: id }}
  b: {{ path: b.csv, id_field: id }}
cross_map: {{ backend: local, path: cm.csv, a_id_field: a, b_id_field: b }}
embeddings: {{ model: unused-model, a_cache_dir: cache }}
match_fields:
  - {{ field_a: name, field_b: name, method: exact, weight: 1.0 }}
thresholds: {{ auto_match: 0.85, review_floor: 0.6 }}
output: {{ csv_dir_path: output }}
batch: {{ db_path: '{}' }}
"#,
            root.join("scratch.db").display()
        );
        serde_yaml::from_str(&yaml).expect("valid test config")
    }

    #[test]
    fn rejects_existing_scratch_family_and_preserves_bytes() -> Result<(), ConfigError> {
        let dir = tempfile::tempdir()?;
        for suffix in ["", "-wal", "-shm", "-journal"] {
            let cfg = make_config(dir.path());
            let path = dir.path().join(format!("scratch.db{suffix}"));
            std::fs::write(&path, "original")?;
            assert!(
                validate_scratch(&cfg, None).is_err(),
                "reject existing {suffix}"
            );
            assert_eq!(
                std::fs::read(&path)?,
                b"original",
                "preserve original bytes"
            );
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_missing_protected_paths_and_empty_scratch() -> Result<(), ConfigError> {
        let dir = tempfile::tempdir()?;
        for target in [
            "output.db",
            "output.db-wal",
            "live.db",
            "live.db-shm",
            "out/unmatched.csv.tmp",
            "cm.tmp",
        ] {
            let mut cfg = make_config(dir.path());
            cfg.batch.db_path = Some(dir.path().join(target).to_string_lossy().into_owned());
            cfg.output.db_path = Some(dir.path().join("output.db").to_string_lossy().into_owned());
            cfg.live.db_path = Some(dir.path().join("live.db").to_string_lossy().into_owned());
            cfg.cross_map.path = Some(dir.path().join("cm.csv").to_string_lossy().into_owned());
            cfg.output.csv_dir_path = Some(dir.path().join("out").to_string_lossy().into_owned());
            assert!(
                validate_scratch(&cfg, None).is_err(),
                "reject missing protected {target}"
            );
            assert!(
                !dir.path().join(target).exists(),
                "validation must not create {target}"
            );
        }
        let mut cfg = make_config(dir.path());
        assert!(
            validate_scratch(&cfg, None).is_ok(),
            "fresh scratch is valid with an unused model"
        );
        cfg.batch.db_path = Some("  ".into());
        assert!(
            validate_scratch(&cfg, None).is_err(),
            "reject blank scratch"
        );
        Ok(())
    }

    #[test]
    fn rejects_only_sqlite_batch_embedding_fields() -> Result<(), ConfigError> {
        let dir = tempfile::tempdir()?;
        let mut cfg = make_config(dir.path());
        cfg.match_fields[0].method = MatchMethod::Embedding;
        assert!(
            validate_scratch(&cfg, None).is_err(),
            "SQLite cannot score embedding fields"
        );
        cfg.batch.db_path = None;
        assert!(
            validate_scratch(&cfg, None).is_ok(),
            "memory/live configs may use embeddings"
        );
        Ok(())
    }

    #[test]
    fn resolves_missing_relative_leaves_and_symlink_ancestors() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("missing/../scratch.db");
        assert_eq!(
            resolve(&path)?,
            dir.path().canonicalize()?.join("scratch.db"),
            "normalize missing leaves"
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path(), dir.path().join("alias"))?;
            assert_eq!(
                resolve(&dir.path().join("alias/scratch.db"))?,
                resolve(&path)?,
                "resolve symlink parents"
            );
        }
        Ok(())
    }
}
