# Output Data Design

## Status

Final design. Supersedes all prior proposals. Covers output data handling across batch (`meld run`), live (`meld serve`), and enroll (`meld enroll`) modes.

---

## 1. Core architecture

Three principles, in priority order:

1. **The operational WAL is the canonical input for output generation in every mode.** Batch, live, and enroll all emit the same WAL event vocabulary during scoring. All downstream outputs (CSVs, SQLite DB) are derived from the WAL.

2. **The scoring log is an optional enrichment layer.** When enabled, it carries the per-record candidate sets and field-score breakdowns that the WAL does not. Outputs become richer; outputs are not structurally different.

3. **A single build function produces the final outputs.** `build_outputs(wal, scoring_log, csv_dir, db_path)` is called from four places: batch end-of-run, `meld export`, `POST /admin/flush`, and `POST /admin/shutdown`. One implementation, four callers. Same code path, same tests, same output shape.

The three modes differ only in **when** the build runs and **how long** the WAL lives:

| Mode | WAL lifetime | When `build_outputs` runs |
|---|---|---|
| Batch | Per-run, written during scoring pass | Automatically at end-of-run |
| Live | Long-lived, written per request | On-demand via `meld export` or HTTP admin endpoints |
| Enroll | Long-lived, written per `/enroll` call | On-demand via `meld export` or HTTP admin endpoints |

Everything else in this document is a consequence of those three principles.

---

## 2. The operational WAL

The existing WAL (`src/state/upsert_log.rs`) is the source of truth for mutations and serves as the canonical input for output generation. It stays small, append-only, buffered, and purpose-built.

### 2a. Event vocabulary

Current events (unchanged in shape, with small additions below):

- `UpsertRecord { side, record }`
- `CrossMapConfirm { a_id, b_id, score, rank, reason }` — score and rank become richer; see §2b
- `ReviewMatch { id, side, candidate_id, score, rank, reason }` — rank and reason added; see §2b
- `CrossMapBreak { a_id, b_id }`
- `RemoveRecord { side, id }`
- `Exclude { a_id, b_id }`
- `Unexclude { a_id, b_id }`

New event:

- `NoMatchBelow { query_id, query_side, best_candidate_id, best_score }` — emitted when a record's best candidate scored below `review_floor`. Preserves today's `unmatched.csv` `best_score` column in WAL-only builds.

### 2b. Extensions to existing events

`CrossMapConfirm` and `ReviewMatch` gain two fields:

```rust
CrossMapConfirm {
    a_id:   String,
    b_id:   String,
    score:  Option<f64>,   // None for pre-score paths
    rank:   Option<u8>,    // None for pre-score paths; otherwise rank in candidate list
    reason: Option<String>, // None = normal scored; "canonical" | "exact" | "crossmap" | "downgraded"
}

ReviewMatch {
    id:           String,
    side:         Side,
    candidate_id: String,
    score:        f64,
    rank:         u8,
    reason:       Option<String>, // None = normal; Some("downgraded") = min_score_gap demotion
}
```

Both additions are ~20–30 bytes per event against an existing ~100-byte payload. The WAL stays small. The purpose of the additions is to make the WAL-only output DB meaningful: without `rank` and `reason`, the `relationships.rank` and `relationships.reason` columns would always be NULL when the scoring log is off, and the `asserted_matches` / `scored_matches` views would be unusable.

### 2c. What the WAL does **not** carry

- Per-field score breakdowns (`Vec<FieldScore>`). These are bulky — ~400 bytes per candidate for typical configs — and belong to the scoring log.
- Candidates at ranks 2..N. These belong to the scoring log.
- Record content beyond `UpsertRecord` (no redundant copies).

### 2d. Write semantics

Unchanged from the current implementation. Mutex-wrapped `BufWriter<File>` with background flush, not synchronous before HTTP response. Live and enroll modes inherit current behaviour exactly. Batch mode is new — see §3.

### 2e. Replay

Unchanged for live/enroll startup state recovery. The build pipeline (§4) treats the WAL as a stream of events to be read sequentially; no semantic change.

---

## 3. Batch mode becomes event-sourced

The largest architectural change in this document. Today `src/batch/engine.rs` holds results in memory (`BatchResult { matched, review, unmatched }`) and writes CSVs at the end. Under this design, batch writes a WAL during the scoring pass and reads it back at end-of-run to produce CSVs and the DB.

### 3a. Why

- **Crash resilience.** A 6-hour batch run that crashes at hour 5 currently produces nothing. Post-refactor, the WAL is on disk and the build can resume or produce a partial output.
- **Memory profile.** Today's `BatchResult` holds all matched/review/unmatched vectors until the end of scoring. Streaming to WAL removes this peak memory cost.
- **Unification.** The build pipeline has one input format across all three modes. `meld export`, end-of-batch, and admin flush all call the same function.
- **Free once you have it.** The WAL machinery already exists and is battle-tested in live mode. Adding a writer handle to the batch engine is small relative to the architectural payoff.

### 3b. Events emitted by batch

During the scoring pass, batch emits:

1. `UpsertRecord { side: A, record }` for every A-side record as it's loaded into the pool.
2. `UpsertRecord { side: B, record }` for every B-side record as it's loaded.
3. For each scored B record, exactly one of:
   - `CrossMapConfirm { ... rank, reason }` — claim loop succeeded
   - `ReviewMatch { ... rank, reason }` — score landed in the review band
   - `NoMatchBelow { query_id, query_side, best_candidate_id, best_score }` — best candidate below `review_floor`
4. `CrossMapConfirm { reason: Some("crossmap"), score: None, rank: None }` — for each pre-loaded crossmap CSV pair.

Pre-score phases (common-ID, exact prefilter) emit `CrossMapConfirm` with `reason: Some("canonical")` or `Some("exact")` and `score: Some(1.0)`, `rank: None`.

### 3c. Concurrency

Batch is Rayon-parallel (`src/batch/engine.rs:314`). The WAL writer is single-consumer: workers send events through an mpsc channel (`crossbeam::channel` preferred for non-tokio batch context) to a writer thread that owns the `BufWriter<File>`. This matches the pattern already used for live-mode hook events, but with **blocking backpressure** (not drop-on-full) because the WAL is canonical data.

Throughput math: ~200 bytes per event × ~1 event per scored record × 55k records/sec peak = ~11 MB/sec. Well within any disk. Use a 4 MB buffer in the batch writer (larger than live's buffer, since batch has no response-latency constraint) to amortise syscalls.

### 3d. WAL file lifecycle in batch

- Written to a predictable path: `<output_dir>/<job_name>.wal.ndjson` (configurable).
- Kept after the build by default. Reasons: cheap, useful for debugging, enables rebuilding CSVs/DB with different options without rerunning scoring.
- Opt-out via `output.cleanup_wal: true` for users who want a clean output directory.

### 3e. `BatchResult` after the refactor

`BatchResult` still exists but its role shrinks to "scoring pass statistics":

```rust
pub struct BatchResult {
    pub stats: BatchStats,
}
```

The `matched`/`review`/`unmatched` vectors go away. Anything that needs them reads the WAL via the build pipeline.

---

## 4. The build pipeline

One function, four callers.

### 4a. Signature

```rust
pub fn build_outputs(
    wal_path: &Path,
    scoring_log_path: Option<&Path>,
    csv_dir_path: Option<&Path>,
    db_path: Option<&Path>,
    manifest: &OutputManifest,
) -> Result<BuildReport>
```

`OutputManifest` carries the config snapshot needed to build the output schema (mode, A/B field lists, thresholds, top_n, model, job name). It's constructed from `Config` at the call site.

`BuildReport` carries counts, timings, and any non-fatal warnings (e.g., "3 crossmap pairs skipped: referenced records not in store").

### 4b. Callers

1. **Batch end-of-run.** `src/cli/run.rs` calls `build_outputs` after `run_batch()` returns successfully.
2. **`meld export`.** `src/cli/export.rs` becomes a thin wrapper: parses args, builds manifest from config + current state, calls `build_outputs`.
3. **`POST /admin/flush`.** Triggers a build without shutting down the server. Returns 202 Accepted; build runs in a background task.
4. **`POST /admin/shutdown`.** Drains in-flight requests, flushes WAL and scoring log, calls `build_outputs`, exits cleanly. Returns 202 Accepted.

### 4c. Atomic writes

Every file produced by `build_outputs` is written to a `.tmp` sibling and renamed atomically on success. If the build crashes partway through, the user's previous output files remain intact. The WAL is untouched by the build (read-only), so a failed build can be retried freely.

### 4d. Reading the scoring log

If `scoring_log_path` is set and the file exists, the build iterates it in parallel with the WAL replay. For each WAL event that references a scoring event (by `query_id` + timestamp), the builder joins the scoring data to enrich the output. Events without a matching scoring log entry (e.g., pre-score canonical matches) degrade gracefully — the output row exists, the field-score detail is empty.

If `scoring_log_path` is not set, the builder skips scoring-log logic entirely and produces the baseline output shape.

---

## 5. The scoring log

Opt-in enrichment. Self-describing. Rust-side writer; downstream consumers are `build_outputs` and any external tool that wants to read the format.

### 5a. Purpose

Carries what the WAL deliberately does not: every scored record's full top_n candidate set with per-field score breakdowns. Enables:

- Candidate visibility (ranks 2..N, near-misses)
- Per-field explainability ("why did this pair land in review?")
- Tuning workflows (`meld tune --no-run` reading historical scoring data)
- Richer CSVs (`candidates.csv`) and a fully populated `field_scores` DB table

### 5b. Format

Append-only ndjson, optionally zstd-compressed.

**First line: header.** Self-describing snapshot of everything a downstream tool needs to interpret the log without reference to config files.

```json
{
  "type":"header",
  "schema":1,
  "mode":"match",
  "job":"acme-2026-04",
  "started_at":"2026-04-05T12:00:00Z",
  "model":"all-MiniLM-L6-v2",
  "top_n":5,
  "thresholds":{"auto_match":0.85,"review_floor":0.65,"min_score_gap":0.05},
  "a_fields":["legal_name","country_code","lei"],
  "b_fields":["counterparty_name","domicile","lei_code"]
}
```

**Subsequent lines: scored records.** One entry per scored query record:

```json
{
  "type":"scored",
  "query_id":"CP-044",
  "query_side":"b",
  "timestamp":"2026-04-05T12:03:17.482Z",
  "outcome":"review",
  "reason":null,
  "candidates":[
    {"rank":1,"matched_id":"ENT-019","score":0.74,"field_scores":[
      {"field_a":"legal_name","field_b":"counterparty_name","method":"embedding","score":0.68,"weight":0.55},
      {"field_a":"country_code","field_b":"domicile","method":"exact","score":1.0,"weight":0.20}
    ]},
    {"rank":2,"matched_id":"ENT-207","score":0.61,"field_scores":[...]}
  ]
}
```

No `matched_record` payload — consumers join to A-side / B-side record tables built from `UpsertRecord` WAL events.

### 5c. Write path

Shared with the WAL writer pattern but on a separate channel and file:

- Rayon workers (batch) or session handlers (live/enroll) serialise `ScoredRecord` entries into `Vec<u8>` buffers and send via `crossbeam::channel` (batch) or `tokio::sync::mpsc` (live/enroll).
- A single writer task drains the channel, calls `write_all` on a `BufWriter<File>` (≥1 MB buffer), and optionally feeds the buffer through a zstd encoder.
- Flush on rotation and clean shutdown. No per-write fsync.
- **Blocking backpressure.** If the channel is full, producers block. The scoring log is canonical (enrichment) data; silent drops are not acceptable.

### 5d. Compression

Optional but **strongly recommended on by default**. Config:

```yaml
scoring_log:
  enabled: true
  compression: zstd  # "none" | "zstd" (default)
```

At typical configs the scoring log compresses ~5–10×. For a 10M-record batch at `top_n=5`, this is the difference between 25 GB and 3–5 GB on disk. zstd level 3 costs ~20% of one core at peak throughput — negligible against Rayon-parallel scoring.

File extension: `.ndjson` uncompressed, `.ndjson.zst` compressed. The build pipeline detects the extension and decompresses transparently.

### 5e. Throughput and disk

At 55k records/sec peak (batch), `top_n=5`, ~5 fields per candidate:

- Per entry: ~2.5 KB uncompressed → ~137 MB/sec sustained
- After zstd: ~15–25 MB/sec sustained
- CPU cost: ~30% of one core for JSON serialisation (parallel across Rayon workers), ~20% of one core for compression (writer thread)

NVMe: 15–35× headroom. SATA SSD: ~3× headroom. Spinning disk: comfortable with compression on.

### 5f. Enrolment default

Scoring log defaults to **on in enroll mode**. Enroll mode has no other persistent relationship output; the scoring log is the only path to analytical output at all. Defaults to off in batch and live modes (user opts in).

### 5g. Rotation

Long-lived live/enroll servers need rotation to keep individual files manageable. Timestamped rotation files (same pattern as the existing WAL): `scores.ndjson`, `scores.2026-04-05T12-00-00.ndjson`, etc. Rotation is triggered by size (default: 1 GB) or on config reload.

### 5h. Failure mode

If the scoring log writer fails (disk full, permissions, I/O error), behaviour is **crash loud, not silent degradation**. The scoring log is opt-in; users who enabled it did so because they want the data. Silently continuing without it would be worse than failing. Log an error to stderr, terminate the writer task, and:

- **Batch mode:** abort the run with a clear error.
- **Live/enroll mode:** return 5xx on subsequent upserts until the scoring log is either repaired or explicitly disabled via admin API.

This is a deliberate inversion of the hook writer's drop-on-full semantics. Hooks are best-effort notifications; the scoring log is canonical data.

---

## 6. Output schemas

### 6a. CSV outputs

Written to `output_csv_dir_path` when set. File shapes:

**`relationships.csv`** — all confirmed matches and review-band pairs. Replaces today's `results.csv` + `review.csv` split.

| Column | Source | Notes |
|---|---|---|
| `b_id` | WAL | Query-side record |
| `a_id` | WAL | Matched reference record |
| `score` | WAL | Composite; NULL for pre-loaded crossmap pairs |
| `relationship_type` | WAL | `match` \| `review` |
| `reason` | WAL | NULL for normal scored; `canonical` \| `exact` \| `crossmap` \| `downgraded` |
| `rank` | WAL | Rank in candidate list; NULL for pre-score paths |
| *(A-side fields)* | WAL `UpsertRecord` | One column per configured A-side field, from `output_mapping` if set |

**`unmatched.csv`** — B-side records with no match or review relationship. Enriched over today: includes `best_a_id`.

| Column | Source | Notes |
|---|---|---|
| `b_id` | WAL | |
| *(B-side fields)* | WAL `UpsertRecord` | |
| `best_score` | WAL `NoMatchBelow` | NULL if no candidates reached scoring |
| `best_a_id` | WAL `NoMatchBelow` | NULL if no candidates reached scoring |

**`candidates.csv`** — produced only when scoring log is on. Ranks 2..N for every scored query record, long-format.

| Column | Source | Notes |
|---|---|---|
| `b_id` | Scoring log | |
| `rank` | Scoring log | 2..top_n |
| `a_id` | Scoring log | |
| `score` | Scoring log | |

Field-score detail is not expanded into CSV columns. It lives in the SQLite DB's `field_scores` table for users who want it.

### 6b. SQLite DB schema — match mode

Written to `output_db_path` when set.

```sql
-- A-side records. Dynamic columns per configured A-side field.
CREATE TABLE a_records (
    id TEXT PRIMARY KEY
    -- + legal_name TEXT, country_code TEXT, lei TEXT, ...
);

-- B-side records. Dynamic columns per configured B-side field.
CREATE TABLE b_records (
    id TEXT PRIMARY KEY
    -- + counterparty_name TEXT, domicile TEXT, lei_code TEXT, ...
);

-- Every relationship Melder knows about.
CREATE TABLE relationships (
    a_id              TEXT    NOT NULL,
    b_id              TEXT    NOT NULL,
    score             REAL,             -- NULL for pre-loaded crossmap pairs
    rank              INTEGER,          -- NULL for pre-score paths
    relationship_type TEXT    NOT NULL, -- 'match' | 'review' | 'candidate' | 'broken'
    reason            TEXT,             -- NULL for normal scored; 'canonical'|'exact'|'crossmap'|'downgraded'
    created_at        TEXT,             -- ISO-8601
    broken_at         TEXT,             -- non-NULL only for relationship_type='broken'
    PRIMARY KEY (a_id, b_id)
);

-- Per-field breakdowns. Populated only when scoring log is on.
CREATE TABLE field_scores (
    a_id    TEXT NOT NULL,
    b_id    TEXT NOT NULL,
    field_a TEXT NOT NULL,
    field_b TEXT NOT NULL,
    method  TEXT NOT NULL,  -- 'exact' | 'fuzzy' | 'embedding' | 'bm25' | 'numeric' | 'synonym'
    score   REAL NOT NULL,
    weight  REAL NOT NULL,
    FOREIGN KEY (a_id, b_id) REFERENCES relationships(a_id, b_id)
);

-- Run metadata.
CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Keys: job_name, config_hash, created_at, mode, auto_match_threshold,
--       review_floor_threshold, min_score_gap, model, top_n,
--       a_record_count, b_record_count, scoring_log_enabled, schema_version

-- Indices.
CREATE INDEX idx_relationships_a        ON relationships(a_id);
CREATE INDEX idx_relationships_b        ON relationships(b_id);
CREATE INDEX idx_relationships_type     ON relationships(relationship_type);
CREATE INDEX idx_relationships_b_rank   ON relationships(b_id, rank);
CREATE INDEX idx_fscores_relationship   ON field_scores(a_id, b_id);
```

Foreign key enforcement stays **off** (`PRAGMA foreign_keys = OFF`). The FK declarations are documentary; the build pipeline validates referential integrity at the application level before inserting and emits warnings for any skipped rows.

### 6c. SQLite DB schema — enroll mode

Same table shapes, symmetric relationships, canonical `id_1 < id_2` ordering:

```sql
CREATE TABLE records (
    id TEXT PRIMARY KEY
    -- + dynamic fields from config
);

CREATE TABLE relationships (
    id_1              TEXT    NOT NULL,
    id_2              TEXT    NOT NULL,
    score             REAL,
    rank              INTEGER,
    relationship_type TEXT    NOT NULL,  -- 'match' | 'review' | 'candidate'
    reason            TEXT,
    created_at        TEXT,
    PRIMARY KEY (id_1, id_2),
    CHECK (id_1 < id_2)
);

CREATE TABLE field_scores (
    id_1   TEXT NOT NULL,
    id_2   TEXT NOT NULL,
    field  TEXT NOT NULL,
    method TEXT NOT NULL,
    score  REAL NOT NULL,
    weight REAL NOT NULL,
    FOREIGN KEY (id_1, id_2) REFERENCES relationships(id_1, id_2)
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX idx_relationships_id1      ON relationships(id_1);
CREATE INDEX idx_relationships_id2      ON relationships(id_2);
CREATE INDEX idx_relationships_type     ON relationships(relationship_type);
CREATE INDEX idx_relationships_id1_rank ON relationships(id_1, rank);
CREATE INDEX idx_fscores_relationship   ON field_scores(id_1, id_2);
```

### 6d. Views

Shipped as a SQL resource file (`src/output/views.sql`) loaded at DB build time and applied after tables are populated. Keeping views in a separate file rather than inline Rust means:

- Views evolve independently of Rust code
- Users can modify views in-place on a built DB
- Schema changes touch one place

Views (same list for match and enroll modes, parameterised by id columns):

- `confirmed_matches` — all pairs with `relationship_type='match'`, scored and asserted
- `scored_matches` — `relationship_type='match' AND reason IS NULL`
- `asserted_matches` — `relationship_type='match' AND reason IS NOT NULL`
- `review_queue` — `relationship_type='review'`
- `near_misses` — `relationship_type='candidate' AND rank=1`
- `runner_ups` — `rank > 1`
- `unmatched_a` / `unmatched_b` — records with no `match` or `review` relationship
- `broken_matches` — `relationship_type='broken'`
- `summary` — aggregate counts by type and reason
- `relationship_detail` — join of `relationships` × `field_scores` for explainability queries

### 6e. Relationship types and reasons

| `relationship_type` | Meaning | Source |
|---|---|---|
| `match` | Confirmed 1:1 pair | `CrossMapConfirm` WAL event |
| `review` | Rank-1 in review band, awaiting decision | `ReviewMatch` WAL event |
| `candidate` | Scored but not acted on (ranks 2..N, near-misses) | Scoring log only |
| `broken` | Previously confirmed, explicitly broken | `CrossMapBreak` after prior `CrossMapConfirm` |

| `reason` | Applies to | Meaning |
|---|---|---|
| `canonical` | `match` | Matched via `common_id_field`. Score 1.0, rank NULL. |
| `exact` | `match` | Matched via `exact_prefilter`. Score 1.0, rank NULL. |
| `crossmap` | `match` | Pair came from pre-loaded crossmap CSV or manual `/crossmap/confirm`. |
| `downgraded` | `review` | Above `auto_match` but demoted by `min_score_gap`. Score ≥ auto_match intentionally. |
| NULL | any | Normal scored outcome. |

---

## 7. Output matrix

What each config setting produces. Users flip three knobs: `output_csv_dir_path`, `output_db_path`, `scoring_log.enabled`.

| Output | Scoring log OFF | Scoring log ON |
|---|---|---|
| `relationships.csv` | `b_id, a_id, score, relationship_type, reason, rank` + A-side fields | Same shape (field-score detail lives in DB) |
| `unmatched.csv` | `b_id` + B-side fields + `best_score`, `best_a_id` | Same |
| `candidates.csv` | **Not produced** | `b_id, rank, a_id, score` |
| DB: `a_records` / `b_records` / `records` | Populated from `UpsertRecord` | Same |
| DB: `relationships` (match/review/broken) | Populated from WAL | Same |
| DB: `relationships` (candidate, near-miss) | Empty | Populated from scoring log |
| DB: `field_scores` | **Empty** | Populated from scoring log |
| DB: views `confirmed_matches`, `review_queue`, `unmatched_*`, `broken_matches`, `summary` | Return rows | Return rows |
| DB: views `near_misses`, `runner_ups`, `relationship_detail` | Empty | Return rows |

The scoring log is the feature that turns Melder's output DB from "a list of confirmed pairs" into "the full entity resolution graph with explainability." Users who don't need candidate data or field-score detail get a perfectly useful DB without it.

---

## 8. API surface

New endpoints on live and enroll servers. All require admin authentication (mechanism existing or to be added — separate concern).

### 8a. `POST /admin/flush`

Triggers a build without shutting down. The server keeps accepting requests.

- Returns `202 Accepted` immediately with a build ID.
- Build runs in a background task on the tokio runtime.
- Status queryable via `GET /admin/flush/{build_id}`.
- On failure, build ID's status shows the error; server remains healthy.
- Multiple concurrent flush calls are serialised (one build at a time).

Use case: periodic snapshots of a running server for review/analysis without restart.

### 8b. `POST /admin/shutdown`

Graceful shutdown with final build.

- Returns `202 Accepted` immediately.
- Server moves to draining state: stops accepting new requests, completes in-flight.
- Flushes WAL and scoring log writers.
- Calls `build_outputs` if `output_csv_dir_path` or `output_db_path` is configured.
- Exits cleanly on success.
- On build failure: exits with non-zero code, logs loudly. The WAL remains on disk; user can rebuild with `meld export`.

Use case: orchestrated shutdowns (Kubernetes pre-stop hooks, systemd stop, CI teardown).

### 8c. SIGTERM handler

SIGTERM triggers the same sequence as `POST /admin/shutdown`. Standard process management tooling (Docker, Kubernetes, systemd) all send SIGTERM by default, so correct behaviour flows naturally.

---

## 9. Config

```yaml
output:
  # At least one of csv_dir_path or db_path must be set in batch mode.
  # Live and enroll modes may run with neither (WAL is still written; user calls meld export later).
  csv_dir_path: output/           # optional
  db_path: output/results.db      # optional

  # Batch mode only: clean up the per-run WAL after a successful build.
  # Default false — WAL is kept for debugging and rebuilds.
  cleanup_wal: false

scoring_log:
  # Off by default for batch and live; on by default for enroll.
  enabled: false

  # zstd strongly recommended. "none" for debugging or when downstream tools can't decompress.
  compression: zstd

  # Size-based rotation for long-lived live/enroll servers. Ignored in batch.
  rotation_size_mb: 1024
```

### 9a. Validation

- **Batch mode:** error at config load time if both `csv_dir_path` and `db_path` are unset. The user would get no output at all.
- **Live/enroll:** no minimum output requirement. A server may run purely to maintain the operational WAL and serve API queries.
- **`cleanup_wal`** is ignored outside batch mode (the live/enroll WAL is the source of truth for state recovery and cannot be cleaned up post-build).

### 9b. Deprecation of old config keys

Old `output.results_path` / `output.review_path` / `output.unmatched_path` keys are detected, map to `output.csv_dir_path` with a deprecation warning, and will be removed in a later version. No silent behaviour change.

---

## 10. Invariants

Preserved through this design:

1. **Batch asymmetry, live symmetry, enroll single-pool.** Unchanged. The WAL carries mode-neutral events; the build pipeline chooses match-mode or enroll-mode schema based on `OutputManifest.mode`.
2. **One scoring pipeline.** `pipeline::score_pool()` remains the single scoring path. Neither WAL writing nor scoring log writing changes how scores are computed.
3. **CrossMap bijection.** The output DB is a read-only projection; no write-back.
4. **Combined vector weighted cosine identity.** Unchanged.
5. **Operational WAL write semantics.** Live mode's hot path is untouched. WAL writes are still buffered and not synchronously flushed before HTTP response.
6. **Accuracy regression tests pass unchanged.** Nothing in this design changes scoring. Test fixtures comparing match counts and scores must be byte-identical before and after implementation.

---

## 11. Risk and performance

### 11a. Performance budget

| Change | Expected cost | Measurement point |
|---|---|---|
| Batch WAL write | ~11 MB/sec @ 55k records/sec, buffered | 10kx10k batch benchmark |
| Scoring log write (on) | ~137 MB/sec uncompressed, ~25 MB/sec zstd | 10kx10k batch benchmark with flag on |
| End-of-run build | 2–30 sec depending on size | Batch benchmark wall-clock delta |
| Live upsert path | Zero delta (WAL semantics unchanged) | 10kx10k_inject3k live benchmark |
| Enroll path | Scoring log channel send adds ~µs | 5k_inject1k enroll benchmark |

All four accuracy and performance benchmarks must show no regression vs baseline before this design is merged to `main`.

### 11b. Risk areas

| Area | Risk | Mitigation |
|---|---|---|
| Batch engine event-sourcing refactor | Touches claim loop and Rayon parallelism — highest-risk code path | Keep the refactor purely additive: WAL writes alongside existing vectors in the first commit; remove vectors in a follow-up commit; benchmark after each step |
| Scoring log writer backpressure | Blocking producers on full channel could stall Rayon workers | 10k-entry channel with 1 MB per entry ≈ 10 GB peak — effectively unbounded at realistic rates. Log a warning if channel reaches 50% capacity so we notice before it bites |
| Dynamic SQLite columns from config | Field names with special characters, very wide schemas | SQLite column names are quoted in the builder; field names come from config (already validated elsewhere) |
| `meld export` on long-lived WALs | Slow export for year-long live servers | Deferred — see §12 |
| Atomic write rename on network filesystems | Rename semantics vary | Document as a requirement; fail loudly if rename fails |

---

## 12. Explicit deferrals

Things that are out of scope for this design but will matter later:

1. **WAL compaction / checkpointing for long-lived live/enroll servers.** A year-old WAL with millions of events makes `meld export` take minutes. Classic event-sourcing checkpoint pattern — write a snapshot, truncate the tail, export reads snapshot + tail. Blocks commercial-scale live deployments but is not required for alpha.

2. **Incremental builds.** Remember the last-built WAL offset, read only new events on subsequent `meld export` calls. Nice optimisation for periodic snapshots, not required for correctness.

3. **Scoring log verbosity levels.** The design ships with one level (full: all top_n with field_scores). A later version could add `candidates` (no field_scores), `winner` (rank-1 only), etc. Defer until a user asks.

4. **Schema versioning for the output DB.** Currently the DB is always rebuilt from the WAL + scoring log — no migration needed. When commercial users start expecting stable output files across Melder versions, add a `schema_version` check in the metadata table and fail loudly on mismatch.

5. **DB browser / view server.** A lightweight HTTP interface for browsing the output DB in a web browser (the `scripts/serve-views.py` idea from the prior design). Not part of the core; ship separately when there's demand.

6. **Admin endpoint authentication.** `/admin/flush` and `/admin/shutdown` need auth. Mechanism is a separate concern (API keys, mTLS, etc.) and tracked elsewhere.

7. **Graceful degradation for missing scoring log at build time.** If the user configured a scoring log path that doesn't exist at build time, the builder currently errors. A future version could treat it as "scoring log off for this build" with a warning. Not critical for alpha.

---

## 13. Implementation order

Suggested sequence. Each step is independently benchmarkable against the four regression tests.

1. **WAL event extensions.** Add `rank` and `reason` to `CrossMapConfirm` and `ReviewMatch`. Add `NoMatchBelow` variant. Backwards-compatible at the deserialisation layer (old events still parse). Update live-mode call sites. Benchmark: no regression.

2. **`MatchResult.reason`.** Add optional `reason` field. Set in pre-score phases and `apply_score_gap_check`. Benchmark: no regression.

3. **Batch engine event-sourcing.** Add WAL writer to `src/batch/engine.rs`, emit `UpsertRecord` / `CrossMapConfirm` / `ReviewMatch` / `NoMatchBelow` during scoring. Keep existing `BatchResult` vectors for now (additive). Benchmark: confirm throughput unchanged.

4. **Build pipeline skeleton.** New module `src/output/build.rs` with `build_outputs()` reading WAL and producing `relationships.csv` + `unmatched.csv` + minimal DB (no scoring log support yet). Wire into `src/cli/run.rs` end-of-run. Switch `meld export` to use it.

5. **Remove `BatchResult` vectors.** Delete the in-memory result accumulation from `batch/engine.rs`. All output now flows through WAL → build pipeline. Benchmark: memory profile should improve.

6. **Scoring log writer.** New module `src/output/scoring_log.rs` with the header + ndjson format, channel writer, optional zstd compression. Wire into session (live/enroll) and batch engine behind the `scoring_log.enabled` flag. Benchmark with flag on and off.

7. **Scoring log reader in build pipeline.** Extend `build_outputs` to read the scoring log when present and populate `field_scores` table, `candidates.csv`, and candidate/near-miss rows in `relationships`.

8. **DB views resource file.** Add `src/output/views.sql`, load at build time.

9. **Admin endpoints.** Add `POST /admin/flush` and `POST /admin/shutdown` handlers. Wire SIGTERM handler to shutdown path.

10. **Docs.** Update `docs/batch-mode.md`, `docs/live-mode.md`, `docs/enroll-mode.md`, `docs/configuration.md`, `docs/cli-reference.md`, `docs/api-reference.md`. Add new `docs/output-data.md` covering the DB schema and views.

Work happens on a feature branch. No merge to `main` until all four benchmarks are green and all docs are updated.

---

## 14. What this design deliberately does not do

To keep it honest, some things that were considered and rejected:

- **A second SQLite DB that runs alongside the operational store.** Rejected: two databases, two sync problems, twice the failure modes. The operational WAL + on-demand build gives the same user-facing output with one authoritative input.

- **An async live-mode output DB writer that writes to the DB on every upsert.** Rejected: adds a mutex-guarded SQLite connection on the hot path, introduces channel backpressure at the HTTP layer, creates consistency questions between operational state and output state. The simpler model is: WAL is live, DB is built on demand.

- **CSV derivation inversion (primary SQLite DB → derived CSVs).** Rejected: forces every CSV output to round-trip through SQLite. Users who want CSVs get them directly from the WAL-driven builder. Users who want the DB get it too. Neither derives from the other; both derive from the WAL.

- **Graceful WAL-replay fallback when the scoring log is corrupted.** Rejected: the scoring log is enrichment, not canonical. If it's corrupt, rebuild without it. No fallback layer needed.

- **A Python-first builder script with later Rust port.** Rejected: the single-binary commercial story requires the builder in Rust from day one. The schema is stable enough to commit to; the previous "iterate in Python" argument is weaker than the "one binary, one command" argument enabled by this design.

- **Schema versioning scheme for the output DB before v1 ships.** Rejected: premature. The DB is always rebuilt from the WAL; there is nothing to migrate. Add versioning when there are users who care.
