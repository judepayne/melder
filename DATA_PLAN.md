# Output Data Redesign — Implementation Plan

## BRANCHING AND GATE PROTOCOL — READ FIRST

**All work for this plan happens on a feature branch, not `main`.**

```bash
git checkout -b feature/output-data-model
```

This is a fundamental architectural change (batch mode becomes event-sourced, new output pipeline, new log format). It must not land on `main` until every phase is complete and all benchmarks pass.

**After every phase, the agent MUST:**

1. Run all 4 benchmarks (batch perf, live perf, live accuracy, enroll accuracy)
2. **STOP and report results to the user**, including:
   - Pass/fail for each benchmark
   - Throughput deltas vs Phase 0 baseline (percentage)
   - Any accuracy diffs
   - Any significant findings, edge cases, or deviations from the plan
3. **Wait for the user's explicit permission before starting the next phase**

Do not chain phases. Do not proceed after a benchmark failure without investigating, fixing, re-running, and reporting again. The user reviews each gate and gives the go-ahead.

---

## Prerequisites

**Read `DATA_DESIGN.md` (project root) before starting.** It contains the full architectural design: schema definitions (§6), relationship types and reasons (§6e), views (§6d), output matrix (§7), invariants (§10), and explicit deferrals (§12). This plan references it throughout. The design doc is the source of truth for *what* to build; this plan is the source of truth for *how* and *in what order*.

**Invariants that must never be violated** (from DATA_DESIGN.md §10):
1. Batch asymmetry (B queries A), live symmetry, enroll single-pool
2. One scoring pipeline — `pipeline::score_pool()` is the single scoring path
3. CrossMap bijection under single RwLock
4. Combined vector weighted cosine identity
5. Operational match log write semantics unchanged for live mode
6. Accuracy regression tests pass byte-identical before and after

---

## Context

Melder's output has evolved organically: batch writes 3 CSVs at end-of-run, live mode has a match log (currently called `upsert_log`) for crash recovery but no analytical output, enroll mode has zero persistent output, and `meld export` produces only two ID columns. The design in `DATA_DESIGN.md` unifies all three modes under a single architecture: the match log becomes the canonical input for output generation everywhere, a single `build_outputs()` function produces CSVs and/or a SQLite DB, and an opt-in scoring log enriches outputs with candidate-level detail.

### Terminology

| Term | File | Purpose |
|---|---|---|
| **Match log** | `src/state/match_log.rs` (renamed from `upsert_log.rs`) | Append-only ndjson recording the matching process: upserts, confirms, reviews, breaks, removes, excludes. Canonical input for output generation. Source of truth for live/enroll crash recovery. |
| **Scoring log** | `src/output/scoring_log.rs` (new) | Opt-in append-only ndjson recording every scored record's full top_n candidate set with per-field score breakdowns. Enrichment layer — outputs become richer when present. |

## Benchmark protocol

Run after **every** phase:

| # | Benchmark | Command | What it checks |
|---|-----------|---------|---------------|
| 1 | Batch perf | `python3 benchmarks/batch/10kx10k_usearch/cold/run_test.py --binary ./target/release/meld` | Throughput (rec/s) |
| 2 | Live perf | `python3 benchmarks/live/10kx10k_inject3k_usearch/cold/run_test.py --binary ./target/release/meld` | Throughput (req/s) |
| 3 | Live accuracy | `python3 benchmarks/accuracy/live_10kx10k_inject3k/run_test.py --binary ./target/release/meld` | Exact crossmap pair matching |
| 4 | Enroll accuracy | `python3 benchmarks/accuracy/enroll_5k_inject1k/run_test.py --binary ./target/release/meld` | Exact enroll edge matching |

**Regression thresholds:** Performance: >5% throughput drop vs baseline. Accuracy: any diff in expected output is a failure.

**Important:** Accuracy tests check crossmap/edge state via the HTTP API, NOT output CSV files. CSV format changes do not affect accuracy tests.

Build release binary before benchmarks: `cargo build --release --features usearch`

---

## Phase 0: Baseline

**Goal:** Record baseline benchmarks on `main`, then create the feature branch.

**Steps:**
1. `cargo build --release --features usearch`
2. Run all 4 benchmarks, record results
3. `git checkout -b feature/output-data-model`
4. Report baselines to user

---

## Phase 1: Rename + data model extensions

**Goal:** Rename `upsert_log` → `match_log` throughout the codebase. Extend the match log event vocabulary and `MatchResult` with `rank`, `reason`, and `NoMatchBelow`. Wire real values at all emit sites. This is the foundation everything else builds on.

**Risk:** LOW. Rename is mechanical. Additive fields with serde defaults. No behavioural change to scoring. Existing match log files parse without error (backwards-compatible deserialization via `#[serde(default)]`).

### Rename: `upsert_log` → `match_log`

**`src/state/upsert_log.rs`** → **`src/state/match_log.rs`**
- Rename file
- `UpsertLog` struct → `MatchLog`
- `append_upsert()` method → `append_record()` (the zero-clone upsert-specific method)
- All other public methods (`append`, `flush`, `replay`, `compact`, `open`) keep their names

**`src/state/mod.rs`** — update `pub mod upsert_log` → `pub mod match_log`

**`src/session/mod.rs`** — update all `use` paths and type references (`UpsertLog` → `MatchLog`, `upsert_log` → `match_log`)

**`src/cli/serve.rs`**, **`src/cli/enroll.rs`**, **`src/cli/export.rs`**, **`src/cli/run.rs`** — update imports

**`src/config/schema.rs`** — rename `LiveConfig.upsert_log` field to `match_log_path` (add `#[serde(alias = "upsert_log")]` for backwards compat with existing config files)

**All tests** referencing `UpsertLog` or `upsert_log` — update

**`WalEvent` enum** → **`MatchLogEvent`** in `src/state/match_log.rs`. Rename the enum type for consistency with the module rename. All `#[serde(rename = "...")]` tags on variants stay as-is (e.g. `#[serde(rename = "upsert_record")]` on `UpsertRecord`) — this preserves on-disk ndjson format. Existing match log files deserialise without error. Update all references across the codebase (`WalEvent` → `MatchLogEvent`).

### Data model extensions

**`src/state/match_log.rs`** (formerly upsert_log.rs)
- `CrossMapConfirm`: add `rank: Option<u8>` and `reason: Option<String>` with `#[serde(default, skip_serializing_if = "Option::is_none")]` (matches existing `score` pattern)
- `ReviewMatch`: add `rank: Option<u8>` and `reason: Option<String>` (same serde attrs)
- New variant: `NoMatchBelow { query_id: String, query_side: Side, best_candidate_id: Option<String>, best_score: Option<f64> }`
- Update `compact()` match arms for new variant (pass-through)

**`src/models.rs`**
- Add `pub reason: Option<String>` and `pub rank: Option<u8>` to `MatchResult`
- Update all MatchResult construction sites (lines 130, 211 in engine.rs; session scoring loops; pipeline)

**`src/matching/pipeline.rs`**
- In `apply_score_gap_check()` (line 422): when downgrading, set `results[0].reason = Some("downgraded".into())`
- In `score_pool()`: after sorting, enumerate results and set `rank: Some((i + 1) as u8)` on each

**`src/batch/engine.rs`**
- Common-ID pre-match (line 130): set `reason: Some("canonical".into()), rank: None`
- Exact prefilter (line 211): set `reason: Some("exact".into()), rank: None`
- Claim loop (line 419): pipeline now sets rank; propagate `reason` from MatchResult to RecordOutcome

**`src/session/mod.rs`**
- At each `CrossMapConfirm` emit (lines 465, 849, 993, 1427): pass `rank` and `reason` from `MatchResult` or context:
  - Scoring claim loops (465, 993): `rank: result.rank, reason: result.reason.clone()`
  - Common-ID pre-match (849): `rank: None, reason: Some("canonical")`
  - Manual confirm (1427): `rank: None, reason: Some("crossmap")`
- At each `ReviewMatch` emit (lines 479, 1007): pass `rank: result.rank, reason: result.reason.clone()`
- After no-match hook send (line 1045): emit `NoMatchBelow` match log event

**`src/cli/export.rs`**
- Update match log event destructuring to handle new fields (`..` on CrossMapConfirm/ReviewMatch)
- Add match arm for `NoMatchBelow` in replay: track in unmatched map with best_score/best_candidate_id

### Tests
- Deserialize old match log line (without rank/reason) — backwards compat
- Serialize+deserialize `NoMatchBelow` round-trip
- `apply_score_gap_check` sets reason = "downgraded"
- `score_pool()` returns results with rank 1..N set
- Pre-score MatchResults have correct reason strings
- Update all existing tests that construct MatchResult/MatchLogEvent to include new fields

### Expected benchmark impact
Zero. All new fields use `skip_serializing_if` — match log event size unchanged when None. Scoring logic identical. When values are populated (live mode), adds ~20-30 bytes per event — negligible.

---

## Phase 2: Batch event-sourcing (match log writer in batch mode)

**Goal:** The batch engine writes a match log during the scoring pass. This is **the highest-risk phase** — isolated deliberately. BatchResult vectors are kept in parallel (additive — both old and new paths produce output).

**Risk:** HIGH. Touches the Rayon parallel claim loop. Mitigation: purely additive — match log writes alongside existing vectors. Both paths produce output; diff them to verify correctness.

### Writer pattern: direct mutex, no channel

The match log uses `Mutex<BufWriter<File>>` (the existing `MatchLog` pattern). Each match log event is ~200 bytes. At 55k records/sec with 8 Rayon workers, worst-case mutex contention is ~3.5μs per record (~500ns hold × 7 waiters). Total: ~190ms/sec spread across 8 workers = ~24ms per worker per second. Negligible.

**No channel. No writer thread.** Pass `Arc<MatchLog>` into the Rayon closure. Workers call `match_log.append()` directly. The mutex serialises writes; the `BufWriter` absorbs them without syscalls most of the time.

**Backpressure:** Implicit via the mutex. If the `BufWriter`'s internal buffer fills and triggers a `write()` syscall (~10μs), other workers wait marginally longer. At ~11 MB/sec of match log data, the kernel page cache absorbs this. Self-regulating, zero configuration, zero new infrastructure.

### Changes

**`src/batch/engine.rs`** (major)
- `run_batch()` gains a `match_log_path: Option<&Path>` parameter (or derives it from config)
- If set: open a `MatchLog` at the path, wrap in `Arc<MatchLog>`
- Before scoring: emit `UpsertRecord` for all A-side and B-side records on main thread (sequential, not parallel — records are already loaded)
- Pre-score phases: emit `CrossMapConfirm` with `reason: Some("canonical")` / `Some("exact")` alongside existing `matched.push(mr)`
- Partition phase (line 238): for each `b_id` where `crossmap.has_b(b_id)`, emit `CrossMapConfirm { a_id: crossmap.get_b(b_id), b_id, score: None, rank: None, reason: Some("crossmap") }`
- Inside Rayon `filter_map` closure (line 419-457): after determining outcome, call `match_log.append(&event)`:
  - `RecordOutcome::Auto(mr)` → `CrossMapConfirm { a_id, b_id, score, rank, reason }`
  - `RecordOutcome::Review(mr)` → `ReviewMatch { id, side, candidate_id, score, rank, reason }`
  - `RecordOutcome::NoMatch(id, _, best_score)` → `NoMatchBelow { query_id, query_side, best_candidate_id, best_score }`
- After Rayon loop: `match_log.flush()`, log path + file size
- **Keep all existing BatchResult logic unchanged** — matched/review/unmatched vectors still populated

**`src/config/schema.rs`**
- Add `pub match_log_dir: Option<String>` to `OutputConfig` (default: same directory as results_path)

**`src/cli/run.rs`**
- Derive match log path from config, pass to `run_batch()`
- Log the match log file path after run_batch returns

### Edge cases to handle

**Claim contention and rank semantics.** When rank-1's `crossmap.claim()` fails (A-side already claimed by another thread), the claim loop continues to rank-2. If rank-2 claims successfully, the match log event should emit `rank: Some(2)`, not `Some(1)`. This works correctly without special handling because `pipeline::score_pool()` pre-sets rank on each `MatchResult` before the claim loop iterates them. The match log event reads `result.rank` directly. Per DATA_DESIGN.md §6e: "rank reflects the scoring order, not the outcome." Verify this in tests: construct a scenario where rank-1 is contended and rank-2 claims — check the match log event has `rank: 2`.

**Error handling inside Rayon closure.** `match_log.append()` returns `io::Result<()>`. Errors cannot easily propagate out of `filter_map`. Strategy: log the error with `tracing::error!` and continue processing. After the Rayon loop completes, check whether any errors occurred (use an `AtomicBool` flag set to true on first error). If errors occurred, log a summary warning: "N match log write errors — output may be incomplete." Do not abort the scoring run — the BatchResult vectors (still populated in this phase) provide a fallback. In Phase 4 when vectors are removed, this becomes a hard error.

**Event ordering.** Rayon workers complete in non-deterministic order. Match log lines will not be in B-record order. This is correct — the build pipeline (Phase 3) identifies events by record ID, not by line position. The match log is an unordered event set, not an ordered sequence. No sorting needed.

**UpsertRecord for records not in any relationship.** Every A-side and B-side record must be emitted as a `UpsertRecord` event, even if the record ends up unmatched. The build pipeline's `unmatched_a` / `unmatched_b` views work by set difference (records present in `a_records`/`b_records` but absent from `relationships`). If a record is missing from the match log, it won't appear in the unmatched view.

### Tests
- Unit: open MatchLog, append 10k events from multiple threads (simulating Rayon), verify file has 10k lines
- Unit: verify flush produces a complete readable file
- Unit: claim contention scenario — rank-1 contended, rank-2 claims — verify match log event has rank=2
- Unit: append error sets AtomicBool flag, post-loop code detects it
- Integration: run batch benchmark, verify match log exists with expected event counts
- **Critical verification:** diff match-log-derived counts (confirmed, review, unmatched) against BatchResult vector lengths — must match exactly

### Expected benchmark impact
<2% throughput regression. Match log writes ~11 MB/sec at 55k rec/s, fully buffered via BufWriter. Mutex contention adds ~24ms per worker per second at 8 cores. If >5% regression: increase BufWriter capacity from default to 4 MB.

---

## Phase 3: Build pipeline + config modernisation

**Goal:** Create `src/output/` module with `build_outputs()` that reads the batch match log and produces CSVs and/or SQLite DB. Wire into `src/cli/run.rs` as a **second** output path alongside old `write_outputs()`. Rewrite `meld export` to use same function. New config keys.

**Risk:** MEDIUM. New code with no hot-path interaction. Main risk is match log replay correctness.

### New files

**`src/output/mod.rs`** — module declarations  
**`src/output/build.rs`** — `build_outputs()` function:
```rust
pub fn build_outputs(
    match_log_path: &Path,
    scoring_log_path: Option<&Path>,  // None for this phase
    csv_dir: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
) -> Result<BuildReport>
```
Reads match log sequentially, builds in-memory state (records, relationships, unmatched), calls CSV/DB builders. Atomic writes via `.tmp` + rename.

**`src/output/manifest.rs`** — `OutputManifest` (mode, field lists, thresholds, job name) and `BuildReport` (counts, timings, warnings)

**`src/output/csv.rs`** — `write_relationships_csv()` and `write_unmatched_csv()`:
- `relationships.csv`: b_id, a_id, score, relationship_type, reason, rank + A-side field columns
- `unmatched.csv`: b_id + B-side fields + best_score + best_a_id

**`src/output/db.rs`** — SQLite DB builder using `rusqlite` (already in Cargo.toml):
- Tables: `a_records` (dynamic cols), `b_records` (dynamic cols), `relationships`, `field_scores` (empty without scoring log), `metadata`
- Indices per DATA_DESIGN.md §6b
- Load views from `src/output/views.sql` via `include_str!()`

**`src/output/views.sql`** — 11 views from DATA_DESIGN.md §6d (confirmed_matches, scored_matches, asserted_matches, review_queue, near_misses, runner_ups, unmatched_a, unmatched_b, broken_matches, summary, relationship_detail)

### Modified files

**`src/lib.rs`** — add `pub mod output;`

**`src/config/schema.rs`** — extend `OutputConfig`:
```rust
pub struct OutputConfig {
    // Old fields (deprecated, backwards-compat)
    pub results_path: String,
    pub review_path: String,
    pub unmatched_path: String,
    // New fields
    pub csv_dir_path: Option<String>,
    pub db_path: Option<String>,
    pub match_log_dir: Option<String>,  // from Phase 2
    pub cleanup_match_log: bool,  // default false
}
```

**`src/cli/run.rs`** — after existing `write_outputs()` call, also call `build_outputs()` if new config keys are set. Both paths run in parallel this phase.

**`src/cli/export.rs`** — refactor `export_memory()` to call `build_outputs()` with the live match log path. Keep `export_sqlite()` for backwards-compat with existing live-mode SQLite stores.

### Tests
- Unit: build_outputs with known match log fixture → verify CSV contents, DB tables, view results
- Unit: CrossMapBreak in match log correctly removes relationship from output
- Unit: RemoveRecord in match log correctly removes record
- Unit: atomic write — verify no partial files on simulated failure
- Unit: DB views return expected row counts
- Integration: batch run with new config → verify both old and new CSVs exist and contain equivalent data

### Expected benchmark impact
Negligible on scoring throughput. Build adds 1-5s wall-clock time after scoring. Accuracy tests unaffected (they check crossmap via API, not output files).

---

## Phase 4: Cutover — remove BatchResult vectors + old writers

**Goal:** Delete the in-memory result vectors, the old `write_outputs()`, and `src/batch/writer.rs`. All batch output flows through match log → `build_outputs()`.

**Risk:** MEDIUM. This is the point of no return for batch output. Mitigated by Phase 3's parallel-path verification. Pre-check: diff new CSVs against old CSVs from Phase 3 runs.

### Changes

**`src/batch/engine.rs`**
- Remove `matched`, `review`, `unmatched` vec declarations and the post-loop collection (lines 460-466)
- Simplify `BatchResult` to `{ stats: BatchStats }`
- `RecordOutcome` may be simplified — outcomes only needed for match log event generation and stats counting

**`src/cli/run.rs`**
- Remove `write_outputs()` function entirely (lines 377-398)
- `build_outputs()` becomes the sole output path
- Update `print_summary()` to read counts from `BuildReport`

**`src/batch/writer.rs`** — delete entire file  
**`src/batch/mod.rs`** — remove writer re-exports

**`src/config/schema.rs`**
- Batch mode validation: error if both `csv_dir_path` and `db_path` are unset
- Deprecation warning if old `results_path`/`review_path`/`unmatched_path` are set

**Benchmark configs** — update `benchmarks/batch/10kx10k_usearch/cold/config.yaml` (and warm) to use new output config keys. Test with both old-style (deprecation path) and new-style config.

### Tests
- All existing batch tests updated to use new BatchResult shape
- Integration: batch output matches expected (same pairs, same scores — format differs)
- Verify memory profile: no large allocations for result vectors during scoring

### Expected benchmark impact
Marginal improvement — no Vec allocation/growth during scoring. Memory reduction proportional to matched+review+unmatched count.

---

## Phase 5: Scoring log + admin endpoints

**Goal:** Implement the opt-in scoring log writer (ndjson, optional zstd), wire into batch/live/enroll, extend build pipeline to consume it, add admin HTTP endpoints for live/enroll.

**Risk:** MEDIUM. Admin endpoints are new HTTP surface but don't touch scoring. Scoring log writer is the one genuinely new piece of concurrency infrastructure.

### Scoring log writer pattern: channel + writer thread

Unlike the match log (which uses direct mutex writes and ~200 byte events), the scoring log carries ~2.5 KB per entry (all top_n candidates with field scores). JSON serialisation of a nested `ScoredRecord` takes 5-20μs — holding a mutex during serialisation would cause unacceptable contention at 55k/sec. Instead:

1. **Producer (Rayon worker or tokio handler):** serialises `ScoredRecord` to `Vec<u8>` on its own thread (parallel, no contention)
2. **Channel:** pre-serialised bytes sent through a bounded channel
3. **Writer thread:** receives bytes, calls `write_all()` on `BufWriter<File>` (optionally through zstd encoder), handles rotation

**Batch mode:** `crossbeam::channel::bounded(10_000)`. `tx.send(bytes)` blocks if channel is full.

**Live/enroll mode:** `tokio::sync::mpsc::channel(10_000)`. `tx.send(bytes).await` suspends the async task (not the thread) if channel is full — other tasks continue running.

**Backpressure mechanics:**

Channel capacity 10,000 entries × ~2.5 KB = ~25 MB buffer. At 55k/sec peak, the buffer holds ~180ms of scoring data.

Under normal operation, the writer drains faster than producers fill:
- `write_all(2.5 KB)` to `BufWriter` is ~500ns per entry → ~2M entries/sec capacity
- With zstd: compression at level 3 processes ~500 MB/sec → ~200k entries/sec
- Disk: NVMe 2-5 GB/s, SATA SSD ~500 MB/s → comfortable headroom at 137 MB/sec uncompressed

The channel fills only under sustained disk saturation (spinning disk + uncompressed + wide config). In that case, producers block on `send()`, scoring throughput self-regulates to match disk write speed. No data loss, no crash, just slower scoring.

**Writer failure handling:**

If the writer thread dies (disk error, panic), the channel fills and producers block indefinitely. To prevent hangs:
- **Batch:** writer thread's `JoinHandle` is checked after the Rayon loop. If it panicked, abort the run with an error.
- **Live/enroll:** writer task sets a shared `AtomicBool` flag on failure. Session checks the flag before `.send().await` and returns 5xx immediately rather than blocking on a dead channel.

### New dependencies

**`Cargo.toml`** — add `zstd = "0.13"` (or latest stable)

### New files

**`src/output/scoring_log.rs`**
- `ScoringLogWriter` struct with the channel/writer pattern described above
- Self-describing header line (first ndjson line): `{ type: "header", schema: 1, mode, job, thresholds, a_fields, b_fields, ... }`
- `ScoredRecord` entries: `{ type: "scored", query_id, query_side, timestamp, outcome, reason, candidates: [{rank, matched_id, score, field_scores: [...]}] }`
- File extension: `.ndjson` / `.ndjson.zst`
- Rotation by size for live/enroll (configurable, default 1 GB)

**`src/api/admin.rs`** (or integrate into handlers)
- `POST /admin/flush` — run `build_outputs()` in background task, return 202
- `POST /admin/shutdown` — drain, flush, build, exit, return 202
- `GET /admin/flush/{id}` — status polling

### Modified files

**`src/output/build.rs`** — extend to read scoring log when present:
- Populate `field_scores` DB table
- Add `candidate` rows (ranks 2..N) to `relationships`
- Add `near_miss` rows (rank-1 below review_floor) to `relationships`
- Produce `candidates.csv` in CSV output

**`src/output/csv.rs`** — add `write_candidates_csv()`: b_id, rank, a_id, score

**`src/batch/engine.rs`** — after `score_pool()` returns in Rayon loop, optionally serialise full `Vec<MatchResult>` to `ScoredRecord` bytes and send through scoring log channel

**`src/session/mod.rs`** — after scoring in upsert path, optionally send `ScoredRecord` to scoring log channel. Also for enroll path.

**`src/api/server.rs`** — add `/admin/flush`, `/admin/shutdown` routes (live/enroll only)

**`src/cli/serve.rs`** — wire scoring log writer (if configured), improve SIGTERM handler to run `build_outputs()`

**`src/cli/enroll.rs`** — wire scoring log writer (default on per DATA_DESIGN.md §5f)

**`src/config/schema.rs`** — add `ScoringLogConfig`:
```rust
pub struct ScoringLogConfig {
    pub enabled: bool,  // default false for batch/live, true for enroll
    pub compression: String,  // "zstd" | "none", default "zstd"
    pub rotation_size_mb: Option<u64>,  // default 1024
}
```

### Tests
- Unit: write scoring log header + 1000 entries, read back, verify structure
- Unit: zstd compression round-trip
- Unit: writer failure sets AtomicBool flag, sender detects it
- Integration: batch with scoring log on → verify candidates.csv exists, field_scores table populated
- Integration: `POST /admin/flush` produces output files
- Integration: `POST /admin/shutdown` exits cleanly
- All 4 benchmarks with scoring log OFF (must show no regression)
- Informational: benchmarks with scoring log ON (new capability, measure overhead)

### Expected benchmark impact
With scoring log OFF: zero (no channel created, no serialization). With scoring log ON: ~2-5% overhead from JSON serialization (parallel on workers) + zstd compression (writer thread).

---

## Verification — end-to-end

After all phases complete, on the feature branch:

1. `cargo test --features usearch,parquet-format,simd -- --test-threads=1` — all unit tests pass
2. `bash tests/restart/test_restart.sh` — restart consistency test passes
3. All 4 benchmarks pass with no regression vs Phase 0 baseline
4. Manual verification:
   - `meld run` with new config → `.match_log.ndjson` exists, `relationships.csv` and `unmatched.csv` correct, `results.db` has expected tables/views/indices
   - `meld run` with scoring log on → `candidates.csv` exists, `field_scores` table populated, `relationship_detail` view returns rows
   - `meld serve` + inject records + `meld export` → produces same output shape
   - `meld enroll` + enroll records + `POST /admin/flush` → output DB with symmetric relationships
   - Old-style config (deprecated keys) → deprecation warning, output still produced
5. `cargo clippy --features usearch,parquet-format,simd -- -D warnings` — no warnings

Then: merge to `main`.
