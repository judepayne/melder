# melder — Start Here

You are picking up a Rust rewrite of a Go+Python fuzzy record matching tool.
Read this file first, then DESIGN.md in this same directory.

---

## Context

**melder** is a Rust rewrite of **match**, a config-driven fuzzy record matching
tool currently implemented in Go (orchestration, HTTP API, scoring) + Python
(embedding model, FAISS index, rapidfuzz). The Go+Python version works but hits
a ~150 req/s ceiling due to IPC serialization, single-threaded Python (GIL),
and cross-process state synchronization.

The Rust version eliminates all three bottlenecks by running everything —
encoding, vector search, fuzzy matching, scoring — in a single process.

## Where things are

```
/Users/jude/Dropbox/Projects/melder/           <-- THIS PROJECT (Rust rewrite)
  DESIGN.md                                     <-- Comprehensive design doc
  START_HERE.md                                 <-- This file

/Users/jude/Library/CloudStorage/Dropbox/Projects/match/   <-- REFERENCE (Go+Python)
  DESIGN.md                 Full Go+Python architecture doc
  ARCHITECTURE-REVIEW.md    Bottleneck analysis + 4 rewrite options
  OPTIMIZATIONS.md          History of 8 optimizations + 4 to-do
  match/                    Go module root
    internal/config/        schema.go (structs), loader.go (validate)
    internal/scoring/       exact.go, embedding.go, composite.go
    internal/matching/      blocking.go, candidates.go, engine.go
    internal/crossmap/      local.go
    internal/state/         state.go, upsertlog.go
    session/session.go      Live session (970 lines — the most complex file)
    api/handlers.go         HTTP handlers
    api/server.go           HTTP server
    pkg/models.go           Record, FieldScore, MatchResult
    sidecar/pipe_main.py    Python sidecar (DO NOT PORT — this is what we're replacing)
    bench/                  Stress tests, configs, test data
      bench_live.yaml       Live mode config (10K x 10K)
      live_stress_test.py   Sequential stress test
      live_concurrent_test.py  Concurrent stress test
      testdata/             Generated CSV datasets
    CLAUDE.md               Symmetry invariant rule
```

## What exists so far

**Nothing has been built yet.** This directory contains only design docs.
No `Cargo.toml`, no `src/`, no code. You are starting from scratch.

## What to build

A single Rust binary (`meld`) that:

1. Reads the same YAML config format as the Go version (backward compatible)
2. Exposes the same HTTP API (all endpoints, request/response shapes)
3. Provides the same CLI (run, serve, validate, cache, review, crossmap)
4. Produces identical matching results (same scoring, classification, semantics)

Key Rust crates:
- `fastembed` or `ort` + `tokenizers` — ONNX embedding inference
- `instant-distance` — HNSW vector search
- `rapidfuzz` — fuzzy string matching
- `axum` + `tokio` — HTTP server
- `clap` — CLI
- `serde` + `serde_yaml` — config
- `dashmap` — concurrent hash maps
- `rayon` — data parallelism

## Build order (phases from DESIGN.md)

### Phase 1: Config + Data + Scoring (start here)

1. `cargo init` in this directory
2. Add `serde`, `serde_yaml`, `clap`, `csv`, `serde_json`, `anyhow`, `thiserror`
3. Implement config structs in `src/config/schema.rs` — see DESIGN.md section 2
   for the exact Rust structs. Cross-reference with Go `schema.go`.
4. Implement config loader/validator in `src/config/loader.rs` — port all 15
   validation rules from Go `loader.go`.
5. Implement `src/models.rs` — Record, FieldScore, MatchResult types.
6. Implement `src/scoring/exact.rs` — trivial, 24 lines in Go.
7. Implement CLI skeleton with `clap` — just `validate` command first.
8. Write tests: load the Go project's `bench/bench_live.yaml`, validate it.

**Verification:** `meld validate --config <path-to-go-project>/match/bench/bench_live.yaml`
should succeed. Use the Go version's test configs as golden inputs.

### Phase 2: Encoder + Index

1. Add `fastembed` (or `ort` + `tokenizers`), `ndarray-npy`
2. Implement `src/encoder/` — load model, encode text, return Vec<f32>
3. Implement `src/index/flat.rs` — brute-force dot product search
4. Implement `src/scoring/embedding.rs` — cosine similarity
5. Implement `src/state/cache.rs` — save/load .npy files
6. Wire up `cache build` CLI command

**Verification:** Encode the same text with both Go+Python and Rust, compare
vectors (should be close but not identical — ONNX vs PyTorch).

### Phase 3: Matching Engine

1. Add `rapidfuzz`
2. Implement `src/scoring/fuzzy.rs`, `src/fuzzy/scorer.rs`
3. Implement `src/scoring/composite.rs` — weighted scoring
4. Implement `src/matching/blocking.rs` — AND/OR filter
5. Implement `src/matching/candidates.rs` — HNSW search
6. Implement `src/matching/engine.rs` — MatchTopN
7. Implement `src/crossmap/local.rs` — bidirectional HashMap + CSV
8. Implement `src/data/csv.rs` — CSV loader with column selection
9. Wire up `run` CLI command

**Verification:** `meld run --config bench_live.yaml` output should match
`match run` output (same records matched, similar scores).

### Phase 4: Live Server

1. Add `axum`, `tokio`, `dashmap`, `tracing`
2. Implement `src/api/` — all endpoints from section 4 of DESIGN.md
3. Implement `src/session/session.rs` — upsert flow (MUCH simpler than Go)
4. Implement `src/state/upsert_log.rs` — WAL
5. Wire up `serve` CLI command

**Verification:** Run Go project's `bench/live_stress_test.py` against `meld serve`.
Then `bench/live_concurrent_test.py` for throughput comparison.

## Key invariants

1. **Symmetry:** Every feature for A must exist identically for B.
2. **Config compatibility:** Existing Go YAML configs must work unchanged
   (ignoring `sidecar` section).
3. **API compatibility:** Same endpoints, same request/response JSON shapes.
4. **Scoring compatibility:** Same classification logic, same weight handling.
   Small float differences in embedding scores are acceptable.

## Running the Go version for comparison

```bash
cd /Users/jude/Library/CloudStorage/Dropbox/Projects/match/match
GONOSUMDB='*' GOFLAGS='-mod=mod' go build -o match ./cmd/match
./match validate --config bench/bench_live.yaml
./match serve --config bench/bench_live.yaml --port 8090

# Stress test (starts its own server on 8090 — don't start one manually)
cd bench && python live_stress_test.py --iterations 100

# IMPORTANT: Reset state before benchmarks
printf 'entity_id,counterparty_id\n' > bench/crossmap_live.csv
: > bench/live_upserts.ndjson
```

## Go project gotchas (learned the hard way)

- The stress test starts its own server — don't start one manually before running it
- `POST /api/v1/a/add` requires `{"record": {...}}` envelope, not bare fields
- Python sidecar stderr is suppressed in serve mode — add `--verbose` for debugging
- LSP false positives in `pipe_main.py` — ignore them
- Cache files can accumulate stale vectors — delete and rebuild if counts look wrong

## Performance targets

| Metric | Go+Python (current) | Rust (target) |
|---|---|---|
| Sequential (c=1) | ~72-111 req/s | **400+** |
| Concurrent (c=10) | ~150 req/s | **1000+** |
| Machine utilization | 13% | **>60%** |
