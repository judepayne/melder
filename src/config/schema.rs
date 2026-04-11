//! Config schema — Rust equivalent of Go `schema.go`.
//!
//! Users should be able to point `meld` at an existing `match` YAML config
//! with zero changes.

use serde::Deserialize;

/// Operating mode — determines which YAML schema to expect and how
/// the scoring pipeline behaves at runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mode {
    /// Two-sided matching: side A (pool) vs side B (query).
    #[default]
    Match,
    /// Single-pool enrollment: records scored against one growing pool.
    Enroll,
}

/// Scoring method for a match field.
///
/// Deserialized directly from YAML — invalid method names are rejected at
/// parse time (serde error) rather than at validation time. This eliminates
/// the class of bugs where a typo like `method: fussy` silently falls through
/// to a default scorer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMethod {
    Exact,
    Fuzzy,
    Embedding,
    Numeric,
    Bm25,
    Synonym,
}

impl MatchMethod {
    /// String representation matching the YAML/JSON field value.
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchMethod::Exact => "exact",
            MatchMethod::Fuzzy => "fuzzy",
            MatchMethod::Embedding => "embedding",
            MatchMethod::Numeric => "numeric",
            MatchMethod::Bm25 => "bm25",
            MatchMethod::Synonym => "synonym",
        }
    }
}

impl std::fmt::Display for MatchMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fuzzy string similarity scorer.
///
/// Deserialized from YAML — invalid scorer names are rejected at parse time.
/// `token_sort_ratio` is accepted as an alias for `token_sort` (the canonical
/// name) for backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzyScorer {
    Wratio,
    PartialRatio,
    #[serde(alias = "token_sort_ratio")]
    TokenSort,
    Ratio,
}

impl FuzzyScorer {
    /// String representation matching the YAML/JSON field value.
    pub fn as_str(&self) -> &'static str {
        match self {
            FuzzyScorer::Wratio => "wratio",
            FuzzyScorer::PartialRatio => "partial_ratio",
            FuzzyScorer::TokenSort => "token_sort",
            FuzzyScorer::Ratio => "ratio",
        }
    }
}

impl std::fmt::Display for FuzzyScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Top-level configuration parsed from YAML.
///
/// Note: `deny_unknown_fields` is intentionally not used because configs
/// may contain extension fields (e.g. `sidecar`) read by external tooling.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Operating mode. Set by the CLI subcommand, not from YAML.
    #[serde(skip)]
    pub mode: Mode,
    pub job: JobConfig,
    #[serde(default)]
    pub datasets: DatasetsConfig,
    #[serde(default)]
    pub cross_map: CrossMapConfig,
    #[serde(default)]
    pub exclusions: ExclusionsConfig,
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub blocking: BlockingConfig,
    /// Pre-blocking exact-match phase. All configured field pairs must match
    /// (AND semantics, all non-empty) for an immediate auto-confirm at 1.0.
    /// Runs before blocking and scoring — recovers cross-block matches that
    /// blocking would miss (e.g. records with wrong country but matching LEI).
    #[serde(default)]
    pub exact_prefilter: ExactPrefilterConfig,

    pub match_fields: Vec<MatchField>,
    #[serde(default)]
    pub output_mapping: Vec<FieldMapping>,
    pub thresholds: ThresholdsConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub scoring_log: ScoringLogConfig,
    #[serde(default)]
    pub batch: BatchConfig,
    #[serde(default)]
    pub live: LiveConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    #[serde(default)]
    pub hooks: HooksConfig,
    /// Vector storage backend: "flat" (brute-force, dev only) or "usearch"
    /// (per-block HNSW, included by default).
    #[serde(default = "default_vector_backend")]
    pub vector_backend: String,
    /// Maximum final results to return per record. Defaults to 5.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// How many candidates ANN retrieves from the full block. Defaults to 50.
    /// Must be >= top_n.
    #[serde(default)]
    pub ann_candidates: Option<usize>,
    /// How many candidates BM25 retrieves independently from the full block.
    /// Defaults to 10. Must be >= top_n.
    #[serde(default)]
    pub bm25_candidates: Option<usize>,
    /// How many BM25 upserts to buffer before committing the Tantivy index.
    ///
    /// Each Tantivy commit is expensive (~2–5 ms): it finalizes segments,
    /// builds FST term dictionaries, and garbage-collects old files.
    /// Batching amortizes that cost across multiple writes.
    ///
    /// - `1` (default): commit after every upsert. Maximum accuracy — every
    ///   BM25 query sees the very latest records. This is the safest option
    ///   but the slowest under high concurrency.
    /// - `N > 1`: commit after every N upserts. Newly inserted records may
    ///   not be visible to BM25 queries until the batch completes. The
    ///   embedding index is unaffected and still finds candidates immediately,
    ///   so the accuracy impact is typically negligible.
    ///
    /// Recommended values: 50–200 for high-throughput live workloads.
    #[serde(default)]
    pub bm25_commit_batch_size: Option<usize>,
    // Derived at load time (not in YAML). Populated by `compute_required_fields`.
    #[serde(skip)]
    pub required_fields_a: Vec<String>,
    #[serde(skip)]
    pub required_fields_b: Vec<String>,
    /// Which text fields the BM25 index concatenates per document.
    ///
    /// Each entry is a `(field_a, field_b)` pair. When omitted, derived
    /// automatically from fuzzy/embedding match field entries (backward
    /// compatible). When set explicitly, the user controls exactly which
    /// fields are indexed — useful for BM25-only configs that would
    /// otherwise need ghost fields with `weight: 0.0`.
    #[serde(default)]
    pub bm25_fields: Vec<Bm25FieldPair>,
    /// Which field pairs to build synonym indices for.
    ///
    /// When omitted, auto-derived from `method: synonym` entries in
    /// `match_fields` with default generators (acronym, min_length=3).
    /// When set explicitly, the user controls generator options.
    #[serde(default)]
    pub synonym_fields: Vec<SynonymFieldConfig>,
    /// Optional user-provided synonym dictionary CSV.
    ///
    /// Each row is an equivalence group of terms that should be treated as
    /// synonyms. Supplements the auto-generated acronym index with explicit
    /// term mappings (e.g. "HSBC" ↔ "Hongkong and Shanghai Banking Corporation").
    #[serde(default)]
    pub synonym_dictionary: Option<SynonymDictionaryConfig>,
}

#[derive(Debug, Deserialize, Default)]
pub struct JobConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct DatasetsConfig {
    #[serde(default)]
    pub a: DatasetConfig,
    #[serde(default)]
    pub b: DatasetConfig,
}

#[derive(Debug, Deserialize, Default)]
pub struct DatasetConfig {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub id_field: String,
    /// Optional shared identifier field (e.g. LEI). If set on one side,
    /// it must be set on both. Records sharing a common ID are
    /// auto-matched before any scoring takes place.
    #[serde(default)]
    pub common_id_field: Option<String>,
    /// "csv" | "parquet" | "jsonl"; inferred from extension if absent.
    #[serde(default)]
    pub format: Option<String>,
    /// For csv/jsonl; defaults to utf-8.
    #[serde(default)]
    pub encoding: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct CrossMapConfig {
    /// "local" (only supported backend). Defaults to "local".
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Path for local backend.
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub a_id_field: String,
    #[serde(default)]
    pub b_id_field: String,
}

/// Known non-matching pairs to exclude from scoring.
///
/// Pairs are loaded from CSV at startup and can be added/removed at runtime
/// via the API. The CSV is updated on shutdown with any runtime changes.
#[derive(Debug, Deserialize, Default)]
pub struct ExclusionsConfig {
    /// Path to the exclusions CSV file. Omit to disable.
    #[serde(default)]
    pub path: Option<String>,
    /// Column name for A-side IDs in the CSV.
    #[serde(default)]
    pub a_id_field: String,
    /// Column name for B-side IDs in the CSV.
    #[serde(default)]
    pub b_id_field: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct EmbeddingsConfig {
    /// HuggingFace model name or local ONNX path. Mutually exclusive with
    /// `remote_encoder_cmd` — exactly one must be set.
    #[serde(default)]
    pub model: String,
    /// Shell command to spawn a user-supplied encoder subprocess. When set,
    /// melder launches this command once per pool slot at startup and talks
    /// to it via the stdin/stdout protocol documented in
    /// `docs/remote-encoder.md`. Mutually exclusive with `model`.
    ///
    /// Example: `"python scripts/acme_encoder.py --env prod"`.
    #[serde(default)]
    pub remote_encoder_cmd: Option<String>,
    /// Directory for A-side combined embedding index cache. Created
    /// automatically on first run; loaded on subsequent runs to skip encoding.
    #[serde(default)]
    pub a_cache_dir: String,
    /// Directory for B-side combined embedding index cache. Optional — omit
    /// to skip B-side caching (vectors rebuilt from scratch each run).
    #[serde(default)]
    pub b_cache_dir: Option<String>,
}

/// Pre-blocking exact-match configuration.
///
/// All field pairs are evaluated with AND semantics: every pair must match
/// (both values non-empty and equal after trimming) for a record to be
/// auto-confirmed. Analogous to `BlockingConfig` but for confirmation
/// rather than candidate filtering.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct ExactPrefilterConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Field pairs that must ALL match (AND). Reuses `BlockingFieldPair`
    /// since the structure is identical.
    #[serde(default)]
    pub fields: Vec<BlockingFieldPair>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct BlockingConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Blocking operator. Only "and" is supported. Defaults to "and".
    #[serde(default = "default_operator")]
    pub operator: String,
    #[serde(default)]
    pub fields: Vec<BlockingFieldPair>,
    /// Legacy single-field syntax — promoted to `fields` vec at load time.
    #[serde(default)]
    pub field_a: Option<String>,
    #[serde(default)]
    pub field_b: Option<String>,
}

#[derive(Debug, Deserialize, serde::Serialize, Clone)]
pub struct BlockingFieldPair {
    pub field_a: String,
    pub field_b: String,
}

/// A field pair for BM25 indexing. The text values of `field_a` (on side A)
/// and `field_b` (on side B) are concatenated into each document's content.
#[derive(Debug, Deserialize, Clone)]
pub struct Bm25FieldPair {
    pub field_a: String,
    pub field_b: String,
}

/// Configuration for synonym generation on a field pair.
///
/// Controls which fields are indexed for synonym/acronym matching and which
/// generators to use. In the common case this is auto-derived from
/// `method: synonym` entries in `match_fields` — users only need to set this
/// explicitly to override generator options.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct SynonymFieldConfig {
    pub field_a: String,
    pub field_b: String,
    #[serde(default = "default_synonym_generators")]
    pub generators: Vec<SynonymGenerator>,
}

/// A synonym generator configuration (e.g. acronym generation).
#[derive(Debug, Deserialize, Clone)]
pub struct SynonymGenerator {
    /// Generator type: "acronym".
    #[serde(rename = "type")]
    pub gen_type: String,
    /// Minimum acronym length to produce. Defaults to 3.
    #[serde(default = "default_min_length")]
    pub min_length: usize,
}

/// Optional user-provided synonym dictionary CSV path.
#[derive(Debug, Deserialize, Clone)]
pub struct SynonymDictionaryConfig {
    /// Path to the CSV file. Each row is an equivalence group of 2+ terms.
    pub path: String,
}

pub fn default_synonym_generators() -> Vec<SynonymGenerator> {
    vec![SynonymGenerator {
        gen_type: "acronym".to_string(),
        min_length: 3,
    }]
}

fn default_min_length() -> usize {
    3
}

#[derive(Debug, Deserialize)]
pub struct MatchField {
    /// Empty for `method: bm25` when using inline `fields`.
    #[serde(default)]
    pub field_a: String,
    /// Empty for `method: bm25` when using inline `fields`.
    #[serde(default)]
    pub field_b: String,
    /// Scoring method. Invalid values are rejected at YAML parse time.
    pub method: MatchMethod,
    /// Fuzzy scorer variant. Only meaningful when `method == Fuzzy`.
    /// Defaults to `Wratio` at config load time if omitted.
    #[serde(default)]
    pub scorer: Option<FuzzyScorer>,
    pub weight: f64,
    /// For `method: bm25` only — which text fields the BM25 index covers.
    /// Preferred over the top-level `bm25_fields` section. When set here,
    /// the top-level `bm25_fields` must be absent (error if both present).
    /// When neither is set, fields are auto-derived from fuzzy/embedding entries.
    #[serde(default)]
    pub fields: Option<Vec<Bm25FieldPair>>,
}

#[derive(Debug, Deserialize)]
pub struct FieldMapping {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct ThresholdsConfig {
    #[serde(default)]
    pub auto_match: f64,
    #[serde(default)]
    pub review_floor: f64,
    /// Minimum score gap between the top and second-best candidate required
    /// to auto-confirm a match.
    ///
    /// When set, a top candidate that clears `auto_match` but whose margin
    /// over rank-2 is less than `min_score_gap` is downgraded to `review`
    /// instead of being auto-confirmed. Single-candidate results (no rank-2
    /// exists) are never downgraded.
    ///
    /// Default: `None` (disabled — no gap requirement).
    ///
    /// Example: `min_score_gap: 0.10` requires the top match to score at
    /// least 0.10 higher than the second-best candidate to auto-confirm.
    #[serde(default)]
    pub min_score_gap: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct OutputConfig {
    /// Directory for CSV output files (relationships.csv, unmatched.csv).
    #[serde(default)]
    pub csv_dir_path: Option<String>,
    /// Directory for Parquet output files (relationships.parquet, unmatched.parquet).
    #[serde(default)]
    pub parquet_dir_path: Option<String>,
    /// Path for the SQLite output database.
    #[serde(default)]
    pub db_path: Option<String>,
    /// Remove the batch match log after a successful build. Default false.
    #[serde(default)]
    pub cleanup_match_log: bool,
}

/// Scoring log configuration.
///
/// When enabled, records every scored record's full top_n candidate set with
/// per-field breakdowns. Enables `candidates.csv`, `field_scores` DB table,
/// and explainability views.
#[derive(Debug, Deserialize)]
pub struct ScoringLogConfig {
    /// Enable the scoring log. Default false for batch/live, true for enroll.
    #[serde(default)]
    pub enabled: bool,
    /// Compression: "zstd" (default) or "none".
    #[serde(default = "default_scoring_log_compression")]
    pub compression: String,
    /// Size-based rotation for long-lived servers (MB). Default 1024. Ignored in batch.
    #[serde(default)]
    pub rotation_size_mb: Option<u64>,
}

impl Default for ScoringLogConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            compression: "zstd".to_string(),
            rotation_size_mb: None,
        }
    }
}

fn default_scoring_log_compression() -> String {
    "zstd".to_string()
}

/// Batch-mode SQLite configuration.
///
/// When `db_path` is set, `meld run` stores records in SQLite instead of
/// in-memory DashMap. The database is created fresh each run and deleted
/// on completion. This enables batch matching on datasets that exceed
/// available RAM (e.g. 55M records).
#[derive(Debug, Deserialize, Default)]
pub struct BatchConfig {
    /// Path to the SQLite database file for batch mode.
    ///
    /// When set, batch mode uses `SqliteStore` instead of `MemoryStore`.
    /// The database is created fresh each run.
    ///
    /// Default: `None` (use in-memory storage).
    #[serde(default)]
    pub db_path: Option<String>,
    /// Number of read-only SQLite connections. Default: num_cpus.
    #[serde(default)]
    pub sqlite_read_pool_size: Option<u32>,
    /// SQLite page cache per read connection in megabytes. Default: 128.
    #[serde(default)]
    pub sqlite_pool_worker_cache_mb: Option<u64>,
    /// SQLite page cache for the write connection in megabytes. Default: 64.
    #[serde(default)]
    pub sqlite_cache_mb: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct LiveConfig {
    #[serde(default)]
    pub match_log_path: Option<String>,
    /// How often dirty CrossMap state is flushed to disk (seconds). Default 5.
    #[serde(default)]
    pub crossmap_flush_secs: Option<u64>,
    /// Path to the SQLite database file for durable live-mode storage.
    ///
    /// When set, live mode uses `SqliteStore` + `SqliteCrossMap` instead of
    /// in-memory `MemoryStore` + `MemoryCrossMap`. The database is created
    /// on first run (cold start) and reused on subsequent runs (warm start).
    ///
    /// Default: `None` (use in-memory storage, current behavior).
    #[serde(default)]
    pub db_path: Option<String>,
    /// Skip the initial matching pass at startup.
    ///
    /// When false (default), all unmatched B records are scored against A
    /// before the API starts listening. Set to true to skip this pass and
    /// start accepting requests immediately.
    #[serde(default)]
    pub skip_initial_match: bool,
    /// SQLite page cache size in megabytes for the write connection.
    ///
    /// Controls `PRAGMA cache_size` for the SQLite write connection. Larger
    /// caches keep more B-tree pages in memory, reducing disk I/O.
    ///
    /// Default: 64 (MB).
    #[serde(default)]
    pub sqlite_cache_mb: Option<u64>,
    /// Number of read-only SQLite connections in the pool.
    ///
    /// Read operations (get, contains, blocking_query, etc.) are served by
    /// a pool of read-only connections, allowing concurrent reads from
    /// multiple threads. Write operations use a single dedicated connection.
    ///
    /// Default: 4.
    #[serde(default)]
    pub sqlite_read_pool_size: Option<u32>,
    /// SQLite page cache size in megabytes for each read pool connection.
    ///
    /// Each read connection maintains its own page cache. Total read cache
    /// memory is `sqlite_read_pool_size × sqlite_pool_worker_cache_mb`.
    ///
    /// Default: 128 (MB).
    #[serde(default)]
    pub sqlite_pool_worker_cache_mb: Option<u64>,
}

/// Performance tuning — applies to both batch and live modes.
#[derive(Debug, Deserialize, Default)]
pub struct PerformanceConfig {
    /// Number of concurrent ONNX inference sessions. Default 1.
    #[serde(default)]
    pub encoder_pool_size: Option<usize>,
    /// Use the INT8-quantised ONNX model variant. ~2x faster encoding,
    /// negligible quality loss. Default false.
    #[serde(default)]
    pub quantized: bool,
    /// How long (ms) the encoding coordinator waits to collect concurrent
    /// requests before dispatching them as a single ONNX batch call.
    ///
    /// - `None` or `0`: coordinator disabled, each request encodes
    ///   independently (current behaviour, zero overhead).
    /// - `1–10`: recommended for concurrent workloads (c >= 4). Requests
    ///   arriving within this window are batched, amortising ONNX overhead.
    ///
    /// Only affects live mode (`meld serve`). Batch mode always encodes
    /// in large batches regardless of this setting.
    #[serde(default)]
    pub encoder_batch_wait_ms: Option<u64>,
    /// Scalar quantization for vectors stored in the usearch HNSW index.
    ///
    /// - `"f32"` (default): full 32-bit precision, 1,536 bytes per 384-dim vector.
    /// - `"f16"`: half precision, 768 bytes per vector. Negligible recall loss,
    ///   ~50% smaller indices, faster search due to reduced memory bandwidth.
    /// - `"bf16"`: brain float 16, same size as f16. Wider exponent range,
    ///   slightly less mantissa precision. Also negligible recall loss.
    ///
    /// Has no effect when `vector_backend` is `flat`.
    /// Changing this value invalidates the usearch index cache (forces rebuild).
    #[serde(default)]
    pub vector_quantization: Option<String>,
    /// How the usearch HNSW index is loaded from the cache file.
    ///
    /// - `"load"` (default): full in-memory load. Consistent search latency,
    ///   higher peak RAM. Safe for both `meld run` and `meld serve`.
    /// - `"mmap"`: memory-mapped — OS pages in/out on demand. Lower peak RAM
    ///   but unpredictable cold-cache latency. **Read-only: do not use with
    ///   `meld serve`** (upserts write to the index and will fail).
    ///
    /// Has no effect when `vector_backend` is `flat`.
    #[serde(default)]
    pub vector_index_mode: Option<String>,
    /// HNSW search beam width (`ef` parameter) for the usearch index.
    ///
    /// Controls how many graph nodes are explored when searching for the
    /// top-k nearest neighbors. Higher values improve recall (fewer missed
    /// true matches) at the cost of slower search.
    ///
    /// - `None` or `0` (default): use usearch's built-in default.
    /// - Typical values: 16–256. Must be >= `ann_candidates`.
    ///
    /// Has no effect when `vector_backend` is `flat`.
    #[serde(default)]
    pub expansion_search: Option<usize>,
    /// ONNX encoding device: `"cpu"` (default) or `"gpu"`.
    ///
    /// When set to `"gpu"`, uses CoreML on macOS and CUDA on Linux for
    /// ONNX embedding inference. Requires building with `--features gpu-encode`.
    ///
    /// Only effective in batch mode (`meld run`). Ignored in live mode
    /// (`meld serve`) where GPU encoding does not improve single-record latency.
    ///
    /// ## Tuning guide (GPU mode)
    ///
    /// GPU encoding benefits from multiple concurrent ONNX sessions
    /// (`encoder_pool_size`) to keep the GPU fed while CPU handles
    /// tokenisation. The optimal settings depend on your hardware:
    ///
    /// - **`encoder_pool_size`**: ~60% of CPU core count. Each session
    ///   needs CPU time for tokenisation; too many sessions starve CPU,
    ///   too few starve GPU. Example: 12 sessions on a 20-core M1 Ultra.
    ///
    /// - **`encoder_batch_size`**: 256 is optimal for most configurations.
    ///   Larger batches (512) cause GPU memory pressure when combined with
    ///   many concurrent sessions. Smaller batches (64) underutilise GPU.
    ///
    /// - Avoid `pool_size × batch_size` products above ~3,000. Beyond that,
    ///   concurrent GPU memory usage degrades throughput sharply.
    ///
    /// Benchmarked on M1 Ultra (20 cores, 64 GPU cores, 64 GB):
    /// - pool=12, batch=256: **1,828 rec/s** (8.7× sequential CPU)
    /// - pool=8, batch=256: 1,677 rec/s
    /// - pool=16, batch=512: 473 rec/s (memory pressure)
    #[serde(default)]
    pub encoder_device: Option<String>,
    /// Number of texts sent per ONNX inference call during batch encoding.
    ///
    /// - Default `64` for CPU (tuned for Apple Silicon cache locality).
    /// - Default `256` for GPU (amortises kernel launch overhead, avoids
    ///   GPU memory pressure at higher concurrent session counts).
    ///
    /// Only effective in batch mode.
    #[serde(default)]
    pub encoder_batch_size: Option<usize>,
    /// Per-call timeout (ms) for a `SubprocessEncoder` encode call. Only
    /// effective when `embeddings.remote_encoder_cmd` is set. If the
    /// subprocess does not return a response within this window the slot
    /// is killed and respawned; the in-flight encode call fails.
    ///
    /// Defaults to 60000ms (60s) when unset. A subprocess's own remote-service
    /// timeout must be strictly less than this value.
    #[serde(default)]
    pub encoder_call_timeout_ms: Option<u64>,
}

/// Pipeline hook configuration — a single long-running subprocess that
/// receives match events as newline-delimited JSON on stdin.
#[derive(Debug, Deserialize, Default)]
pub struct HooksConfig {
    /// Shell command to run as the hook subprocess. If absent, hooks are
    /// disabled. Example: `"python scripts/hook.py"`
    #[serde(default)]
    pub command: Option<String>,
}

impl Config {
    /// Returns `true` when the engine is running in single-pool enrollment mode.
    pub fn is_enroll_mode(&self) -> bool {
        self.mode == Mode::Enroll
    }
}

fn default_backend() -> String {
    "local".into()
}

pub fn default_vector_backend() -> String {
    "usearch".into()
}

pub fn default_operator() -> String {
    "and".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_bench_live_yaml() {
        let yaml =
            std::fs::read_to_string("benchmarks/live/10kx10k_inject3k_usearch/warm/config.yaml")
                .expect("failed to read bench_live.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize bench_live.yaml");

        assert_eq!(config.job.name, "live_10kx10k_usearch_warm");
        assert_eq!(config.datasets.a.path, "benchmarks/data/dataset_a_10k.csv");
        assert_eq!(config.datasets.b.id_field, "counterparty_id");
        assert_eq!(config.cross_map.backend, "local");
        assert_eq!(config.embeddings.model, "all-MiniLM-L6-v2");
        assert!(config.blocking.enabled);
        assert_eq!(config.match_fields.len(), 4);
        assert_eq!(config.match_fields[0].method, MatchMethod::Embedding);
        assert!((config.match_fields[0].weight - 0.55).abs() < f64::EPSILON);
        assert!((config.thresholds.auto_match - 0.85).abs() < f64::EPSILON);
        assert!((config.thresholds.review_floor - 0.60).abs() < f64::EPSILON);
        assert_eq!(config.top_n, Some(5));
        assert_eq!(config.performance.encoder_pool_size, Some(4));
    }

    #[test]
    fn deserialize_counterparty_recon_with_sidecar() {
        // sidecar section in YAML is silently ignored by serde (no deny_unknown_fields)
        let yaml = std::fs::read_to_string("tests/fixtures/counterparty_recon.yaml")
            .expect("failed to read counterparty_recon.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize counterparty_recon.yaml");

        assert_eq!(config.job.name, "counterparty_recon");
        assert_eq!(config.output_mapping.len(), 4);
        assert_eq!(config.output_mapping[0].from, "sector");
        assert_eq!(config.output_mapping[0].to, "ref_sector");
        assert_eq!(config.performance.encoder_pool_size, Some(2));
    }

    #[test]
    fn deserialize_bench1kx1k() {
        let yaml = std::fs::read_to_string("benchmarks/batch/10kx10k_flat/cold/config.yaml")
            .expect("failed to read bench1kx1k.yaml");
        let config: Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize bench1kx1k.yaml");

        assert_eq!(config.datasets.a.path, "benchmarks/data/dataset_a_10k.csv");
        // top_n: 20 is set explicitly in bench1kx1k.yaml
        assert_eq!(config.top_n, Some(20));
    }
}
