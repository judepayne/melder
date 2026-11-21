//! Single-collection enrollment ownership and legacy WAL recovery.

use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::config::enroll_schema::EnrollJobConfig;
use crate::config::schema::MatchMethod;
use crate::encoder::Encoder;
use crate::encoder::coordinator::EncoderCoordinator;
use crate::error::MelderError;
use crate::matching::exclusions::Exclusions;
use crate::models::{Record, Side};
use crate::state::live::LiveSideState;
use crate::state::match_log::{MatchLog, MatchLogEvent};
use crate::store::{RecordCollection, memory::MemoryCollection};

/// Enrollment resources. There is no paired store or matching membership.
pub struct EnrollState {
    pub config: EnrollJobConfig,
    pub collection: MemoryCollection,
    pub pool: LiveSideState,
    pub encoder_pool: Arc<dyn Encoder>,
    pub coordinator: Option<EncoderCoordinator>,
    pub exclusions: Exclusions,
    pub wal: MatchLog,
    pub synonym_dictionary: Option<Arc<crate::synonym::dictionary::SynonymDictionary>>,
    /// Serializes score/index/evidence mutations and administrative snapshots.
    pub gate: Mutex<()>,
}

/// Current pool records and the ordered evidence captured at the same instant.
pub type EnrollSnapshot = (Vec<(String, Record)>, Vec<MatchLogEvent>);

impl EnrollState {
    /// Load native single-pool resources without constructing a paired job.
    pub fn load_job(config: EnrollJobConfig) -> Result<Arc<Self>, MelderError> {
        let encoder = crate::state::state::build_encoder_resources(
            &config.embeddings.model,
            config.embeddings.remote_encoder_cmd.as_deref(),
            &config.performance,
            true,
        )?;
        Self::load_job_with_encoder(config, encoder)
    }

    /// Load a native enrollment job with a caller-provided encoder.
    pub fn load_job_with_encoder(
        config: EnrollJobConfig,
        encoder_pool: Arc<dyn Encoder>,
    ) -> Result<Arc<Self>, MelderError> {
        let blocking_hash = crate::vectordb::manifest::enroll_blocking_hash(&config.blocking);
        let backend_blocking = crate::vectordb::enroll_backend_blocking_config(&config.blocking);
        let specs = crate::vectordb::enroll_embedding_field_specs(&config);
        let (records, ids) = if config.dataset.path.is_empty() {
            (std::collections::HashMap::new(), Vec::new())
        } else {
            crate::data::load_dataset(
                Path::new(&config.dataset.path),
                &config.dataset.id_field,
                &config.required_fields,
                config.dataset.format.as_deref(),
            )?
        };
        let combined_index = crate::vectordb::build_or_load_combined_index_resources(
            &config.vector_backend,
            Some(&config.embeddings.cache_dir),
            &records,
            &ids,
            &specs,
            blocking_hash,
            backend_blocking.enabled.then_some(&backend_blocking),
            &config.performance,
            &config.embeddings.model,
            true,
            encoder_pool.as_ref(),
            true,
            (!config.dataset.path.is_empty()).then(|| Path::new(&config.dataset.path)),
        )?;
        let collection = MemoryCollection::from_enroll_records(records, &config.blocking);
        let exclusions = super::live::LiveMatchState::load_exclusions_resources(&config.exclusions);
        let wal_path = config
            .live
            .match_log_path
            .as_deref()
            .unwrap_or("bench/upsert.wal");
        let events = MatchLog::replay(Path::new(wal_path)).map_err(anyhow::Error::from)?;
        // Stable replay order also preserves insertion order in the ANN index.
        let mut touched = std::collections::BTreeSet::new();
        for event in &events {
            match event {
                MatchLogEvent::UpsertRecord {
                    side: Side::A,
                    record,
                } => {
                    if let Some(id) = record.get(&config.dataset.id_field) {
                        if let Some(old) = collection.get(id)? {
                            collection.blocking_remove(id, &old)?;
                        }
                        collection.insert(id, record)?;
                        collection.blocking_insert(id, record)?;
                        touched.insert(id.clone());
                    }
                }
                MatchLogEvent::RemoveRecord { side: Side::A, id } => {
                    if let Some(old) = collection.get(id)? {
                        collection.blocking_remove(id, &old)?;
                    }
                    collection.remove(id)?;
                    touched.insert(id.clone());
                }
                MatchLogEvent::Exclude { a_id, b_id } => exclusions.add(a_id, b_id),
                MatchLogEvent::Unexclude { a_id, b_id } => exclusions.remove(a_id, b_id),
                _ => {}
            }
        }
        if let Some(index) = combined_index.as_deref() {
            for id in touched {
                if let Some(record) = collection.get(&id)? {
                    let hash =
                        crate::vectordb::texthash::compute_text_hash(&record, &specs, Side::A);
                    if index.contains(&id) && index.text_hash_for(&id) == Some(hash) {
                        continue;
                    }
                    let vector = crate::vectordb::encode_combined_vector(
                        &record,
                        &specs,
                        encoder_pool.as_ref(),
                        true,
                    )?;
                    if !vector.is_empty() {
                        index
                            .upsert(&id, &vector, &record, Side::A)
                            .map_err(anyhow::Error::from)?;
                    }
                } else {
                    index.remove(&id).map_err(anyhow::Error::from)?;
                }
            }
        }
        let bm25_index = if config
            .match_fields
            .iter()
            .any(|f| f.method == MatchMethod::Bm25)
            && !config.bm25_fields.is_empty()
        {
            Some(crate::bm25::simple::SimpleBm25::try_build_enroll(
                &collection,
                &config.bm25_fields,
            )?)
        } else {
            None
        };
        let synonym_dictionary = config
            .synonym_dictionary
            .as_ref()
            .map(|cfg| {
                crate::synonym::dictionary::SynonymDictionary::load(Path::new(&cfg.path))
                    .map(Arc::new)
            })
            .transpose()?;
        let synonym_index = if config.synonym_fields.is_empty() {
            None
        } else {
            let index = crate::synonym::index::SynonymIndex::try_build_enroll_collection(
                &collection,
                &config.synonym_fields,
                synonym_dictionary.clone(),
            )?;
            Some(std::sync::RwLock::new(index))
        };
        let coordinator = None;
        let wal = MatchLog::open(Path::new(wal_path)).map_err(anyhow::Error::from)?;
        Ok(Arc::new(Self {
            config,
            collection,
            pool: LiveSideState {
                combined_index,
                bm25_index,
                synonym_index,
            },
            encoder_pool,
            coordinator,
            exclusions,
            wal,
            synonym_dictionary,
            gate: Mutex::new(()),
        }))
    }

    /// Start encoding coordination inside the server runtime.
    pub fn init_coordinator(&mut self) {
        let wait = self.config.performance.encoder_batch_wait_ms.unwrap_or(0);
        if wait > 0 {
            self.coordinator = Some(EncoderCoordinator::new(
                Arc::clone(&self.encoder_pool),
                std::time::Duration::from_millis(wait),
            ));
        }
    }

    /// Capture authoritative records and their corresponding WAL under one gate.
    pub fn snapshot(&self) -> Result<EnrollSnapshot, MelderError> {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.wal.flush().map_err(anyhow::Error::from)?;
        let mut records = Vec::new();
        self.collection
            .for_each_record(&mut |id, record| records.push((id.to_owned(), record.clone())))?;
        let events = MatchLog::replay(Path::new(
            self.config
                .live
                .match_log_path
                .as_deref()
                .unwrap_or("bench/upsert.wal"),
        ))
        .map_err(anyhow::Error::from)?;
        Ok((records, events))
    }

    /// Persist the single pool's disposable vector cache.
    pub fn save_combined_index_caches(&self) -> Result<(), MelderError> {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.save_cache_locked()
    }

    fn save_cache_locked(&self) -> Result<(), MelderError> {
        let specs = crate::vectordb::enroll_embedding_field_specs(&self.config);
        if specs.is_empty() {
            return Ok(());
        }
        if let Some(index) = &self.pool.combined_index {
            let quantization = self
                .config
                .performance
                .vector_quantization
                .as_deref()
                .unwrap_or("f32");
            let hash = crate::vectordb::spec_hash(&specs, quantization);
            let path =
                crate::vectordb::combined_cache_path(&self.config.embeddings.cache_dir, "a", &hash);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(anyhow::Error::from)?;
            }
            index.save(&path).map_err(anyhow::Error::from)?;
        }
        Ok(())
    }

    /// Compact the active segment while retaining ordered enrollment evidence.
    pub fn compact(&self) -> Result<(), MelderError> {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.wal
            .compact(&self.config.dataset.id_field, &self.config.dataset.id_field)
            .map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// Flush durable resources and save rebuildable caches before shutdown.
    pub fn shutdown(&self) -> Result<(), MelderError> {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.wal.flush().map_err(anyhow::Error::from)?;
        self.exclusions.flush()?;
        self.save_cache_locked()
    }

    /// Flush durable enrollment state without fabricating matching membership.
    pub fn flush(&self) -> Result<(), MelderError> {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.wal.flush().map_err(anyhow::Error::from)?;
        self.exclusions.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct UnusedEncoder;

    impl Encoder for UnusedEncoder {
        fn encode(&self, _: &[&str]) -> Result<Vec<Vec<f32>>, crate::error::EncoderError> {
            panic!("exact enrollment must not encode")
        }
        fn dim(&self) -> usize {
            0
        }
        fn pool_size(&self) -> usize {
            1
        }
        fn encode_batch_size(&self) -> usize {
            1
        }
    }

    #[test]
    fn snapshot_and_restart_keep_all_segments_and_ordered_invalidation() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let config_path = dir.path().join("config.yaml");
        std::fs::write(
            &config_path,
            r#"
job: {name: enrollment_recovery}
vector_backend: flat
embeddings: {model: all-MiniLM-L6-v2, cache_dir: cache}
match_fields:
  - {field: name, method: exact, weight: 1.0}
thresholds: {auto_match: 0.85, review_floor: 0.6}
output: {csv_dir_path: out}
"#,
        )
        .expect("config file");
        let mut config = crate::config::load_enroll_job_config(&config_path).expect("config");
        let base = dir.path().join("enroll.wal");
        config.live.match_log_path = Some(base.to_string_lossy().into_owned());
        let record = Record::from([("id".into(), "old".into()), ("name".into(), "seed".into())]);
        // The legacy file precedes every timestamped segment.
        std::fs::write(
            &base,
            MatchLog::serialize_event(&MatchLogEvent::UpsertRecord {
                side: Side::A,
                record,
            })
            .expect("serialize"),
        )
        .expect("legacy file");
        {
            let wal = MatchLog::open(&base).expect("first segment");
            wal.append(&MatchLogEvent::Exclude {
                a_id: "old".into(),
                b_id: "other".into(),
            })
            .expect("exclude");
            wal.flush().expect("flush first");
        }
        {
            let wal = MatchLog::open(&base).expect("second segment");
            wal.append(&MatchLogEvent::RemoveRecord {
                side: Side::A,
                id: "old".into(),
            })
            .expect("remove");
            wal.append(&MatchLogEvent::Unexclude {
                a_id: "old".into(),
                b_id: "other".into(),
            })
            .expect("unexclude");
            wal.flush().expect("flush second");
        }
        let mut restart_config =
            crate::config::load_enroll_job_config(&config_path).expect("restart config");
        restart_config.live.match_log_path = Some(base.to_string_lossy().into_owned());
        let state =
            EnrollState::load_job_with_encoder(config, Arc::new(UnusedEncoder)).expect("load");
        let (records, events) = state.snapshot().expect("snapshot");
        assert!(
            records.is_empty(),
            "later segment removal overrides legacy record"
        );
        assert_eq!(
            events.len(),
            4,
            "snapshot includes full legacy and multisegment prefix"
        );
        assert!(
            !state.exclusions.contains("old", "other"),
            "unexclude replays in order"
        );
        state.compact().expect("compact");
        state.shutdown().expect("shutdown");
        drop(state);
        let restored = EnrollState::load_job_with_encoder(restart_config, Arc::new(UnusedEncoder))
            .expect("native restart");
        assert!(
            restored.snapshot().expect("restored snapshot").0.is_empty(),
            "restart preserves removal"
        );
        assert_eq!(
            restored.snapshot().expect("history").1.len(),
            4,
            "compaction preserves old evidence"
        );
    }
}
