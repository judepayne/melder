//! Build script for the `builtin-model` feature.
//!
//! When `--features builtin-model` is enabled, this script resolves the
//! embedding model specified by `MELDER_BUILTIN_MODEL` and copies the
//! required files into `OUT_DIR/builtin_model/` for `include_bytes!()`.
//!
//! The env var accepts either:
//! - A HuggingFace repo ID (e.g. `themelder/arctic-embed-xs-entity-resolution`)
//!   → downloaded via `hf-hub`
//! - A local directory path (e.g. `./models/my-model`) → files copied directly
//!
//! If `MELDER_BUILTIN_MODEL` is unset, defaults to
//! `themelder/arctic-embed-xs-entity-resolution`.

fn main() {
    #[cfg(feature = "builtin-model")]
    builtin_model::fetch();
}

#[cfg(feature = "builtin-model")]
mod builtin_model {
    use std::fs;
    use std::path::{Path, PathBuf};

    const DEFAULT_MODEL: &str = "themelder/arctic-embed-xs-entity-resolution";

    const REQUIRED_FILES: &[&str] = &[
        "model.onnx",
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ];

    pub fn fetch() {
        // Re-run if the env var changes.
        println!("cargo:rerun-if-env-changed=MELDER_BUILTIN_MODEL");

        let model_source =
            std::env::var("MELDER_BUILTIN_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
        let dest_dir = out_dir.join("builtin_model");
        fs::create_dir_all(&dest_dir).expect("failed to create builtin_model dir");

        let source_dir = if is_local_path(&model_source) {
            copy_from_local(&model_source, &dest_dir)
        } else {
            download_from_hub(&model_source, &dest_dir)
        };

        // Verify all required files are present.
        for fname in REQUIRED_FILES {
            let path = dest_dir.join(fname);
            if !path.exists() {
                panic!(
                    "builtin-model: required file '{}' not found in {} (source: {})",
                    fname,
                    dest_dir.display(),
                    source_dir.display(),
                );
            }
        }

        eprintln!(
            "builtin-model: embedded {} from {}",
            model_source,
            source_dir.display()
        );
    }

    fn is_local_path(model: &str) -> bool {
        model.starts_with('/')
            || model.starts_with("./")
            || model.starts_with("../")
            || model.starts_with('~')
            || Path::new(model).is_dir()
    }

    fn copy_from_local(path: &str, dest_dir: &Path) -> PathBuf {
        let src = Path::new(path);
        if !src.is_dir() {
            panic!(
                "builtin-model: MELDER_BUILTIN_MODEL='{}' is not a directory",
                path
            );
        }
        for fname in REQUIRED_FILES {
            let src_file = src.join(fname);
            let dst_file = dest_dir.join(fname);
            fs::copy(&src_file, &dst_file).unwrap_or_else(|e| {
                panic!(
                    "builtin-model: failed to copy {}: {}",
                    src_file.display(),
                    e
                )
            });
        }
        src.to_path_buf()
    }

    fn download_from_hub(repo_id: &str, dest_dir: &Path) -> PathBuf {
        // Check if we already have all files (avoid re-downloading on
        // incremental builds).
        let all_present = REQUIRED_FILES.iter().all(|f| dest_dir.join(f).exists());
        if all_present {
            return dest_dir.to_path_buf();
        }

        eprintln!(
            "builtin-model: downloading {} from HuggingFace Hub...",
            repo_id
        );

        let api =
            hf_hub::api::sync::Api::new().expect("builtin-model: failed to init HuggingFace API");
        let repo = api.model(repo_id.to_string());

        for fname in REQUIRED_FILES {
            let cached_path = repo.get(fname).unwrap_or_else(|e| {
                panic!(
                    "builtin-model: failed to download {}/{}: {}",
                    repo_id, fname, e
                )
            });
            let dst_file = dest_dir.join(fname);
            fs::copy(&cached_path, &dst_file).unwrap_or_else(|e| {
                panic!(
                    "builtin-model: failed to copy {} to {}: {}",
                    cached_path.display(),
                    dst_file.display(),
                    e
                )
            });
        }

        dest_dir.to_path_buf()
    }
}
