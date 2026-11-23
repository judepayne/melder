# Changelog

## v0.1.6 (unreleased)

- With batch scoring logging enabled, export every shortlisted pair’s rank, score and field breakdown to CSV, Parquet or SQLite on either storage backend. [Batch outputs](docs/batch-mode.md#sinks-and-scoring-evidence)

- Made graph assignment the default for batch matching. Greedy remains available. [Batch docs](docs/batch-mode.md#choosing-pairs-with-the-graph-optimizer)
- Sped up greedy batch CSV output replay by about 9× on the 100k benchmark. Same matches, less work.
- Reused BM25 block filters across batch queries. Warm scoring was 2.76× faster on the 1M reference benchmark. [Results](benchmarks/batch/usearch_scaling/comparison/findings.md)
- Sped up unblocked batch BM25 searches by dropping a full-pool scan per query. [Benchmarks](docs/performance.md#unblocked-batch-bm25-comparison)
- Made graph assignment the default at live startup. Incoming requests still default to greedy matching.
- Added optional deferred acceptance to live mode: a stronger match can replace an earlier one, at the cost of extra work per request. [Live docs](docs/live-mode.md)
- Made enroll mode faster by generating embeddings in parallel or in batches.
- Unified matching around record pools. Enrollment now takes a single-pool config and rejects paired settings. [Design rationale](DECISIONS.md#d-007--separate-searching-a-pool-from-assigning-a-match)
- Fixed Parquet output and preserved live matches and enrollment edges across exports.
- Added Decision log ([DECISIONS.md](DECISIONS.md)) and Changelog to project.
