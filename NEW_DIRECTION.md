# New Direction: Per-Block HNSW with All-Embedding Scoring

## The idea

Replace the current three-stage pipeline (blocking → candidate selection →
full scoring) with a two-stage pipeline (blocking → vector search) built
on per-block HNSW indices that persist across job runs.

The key insight: usearch's unfiltered HNSW search is nearly
scale-insensitive (~0.14ms at 10K, ~0.36ms at 100K) but its filtered
search is catastrophically slow when filter acceptance is low. Instead
of filtering inside one giant index, build one small HNSW index per
block. Every search is unfiltered within its block index, which is
exactly what HNSW is designed for.

Combined with all-embedding scoring (every match field uses vector
similarity instead of fuzzy string comparison), the entire scoring
pipeline becomes a fixed number of HNSW lookups per query — independent
of block size.

---

## The use case

Melder jobs are recurring. A counterparty reconciliation runs daily
against largely the same entity universe. Day 1 processes 100K records,
building vectors for every entity name. Day 2, 95% of the same entities
appear again. By day 30, the per-block HNSW indices contain nearly every
entity name the job has ever seen.

The per-block indices persist to disk between runs. On each run:

- Records with text already in the index skip ONNX encoding entirely
- New text (~5% of records on a typical day) gets encoded and inserted
  into the relevant block index
- Old vectors that no longer appear in today's data stay in the index
  — they cost nothing and may reappear tomorrow

Deletions are unnecessary because:

1. HNSW search is scale-insensitive — 10K or 50K vectors in a block,
   search is still sub-millisecond
2. Vectors are expensive to create (~3ms ONNX) and cheap to store
   (~1.5KB) — keeping old vectors is pure upside
3. Stale vectors are harmless noise — the search returns the closest
   match regardless of how many inactive vectors surround it
4. The same entity names recur across runs — a vector created on day 1
   serves every subsequent run for free

The only maintenance would be an occasional manual compaction
(`meld compact --job foo`) if an index grows unwieldy after years of
accumulation, but at 1.5KB per vector this threshold is very distant.

---

## Architecture

### Current pipeline (three stages)

```
record → blocking lookup → candidate IDs (could be 300K)
                                ↓
                         candidate selection
                         (wratio on each: O(block_size))
                                ↓
                         top 10 candidates
                                ↓
                         full scoring (4 fields × 10 candidates)
                                ↓
                         classify → result
```

The bottleneck at scale is candidate selection: 300K wratio calls at
~1-3μs each = 300ms–900ms per query.

### Proposed pipeline (two stages)

```
record → blocking lookup → block ID
                              ↓
                     per-block HNSW index
                     (one index per match field)
                              ↓
                     N HNSW searches (one per field, top-K each)
                     each ~0.3ms regardless of block size
                              ↓
                     merge results, weighted score, classify → result
```

No candidate selection stage. No wratio. No O(block_size) iteration.
The work per query is O(N_fields × log(block_size)), which is
effectively constant.

---

## System design: the processing pipeline

### Load phase

When a job starts, records from both A and B sides must be categorised
into blocks. This is the same blocking logic we have today, but the
output feeds into per-block HNSW indices rather than per-block ID sets.

```
┌─────────────────────────────────────────────────────────┐
│                     LOAD PHASE                          │
│                                                         │
│  ┌─────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Dataset │───▶│ Compute      │───▶│ Block Router   │  │
│  │ A / B   │    │ block key    │    │ (HashMap:      │  │
│  │ (csv)   │    │ per record   │    │  block_key →   │  │
│  └─────────┘    └──────────────┘    │  block_id)     │  │
│                                     └───────┬────────┘  │
│                                             │           │
│                    ┌────────────────────────┐│           │
│                    ▼                        ▼▼           │
│  ┌──────────────────────┐  ┌──────────────────────┐     │
│  │ Block "US/finance"   │  │ Block "GB/tech"      │     │
│  │                      │  │                      │     │
│  │  record_id → vec     │  │  record_id → vec     │     │
│  │  (per match field)   │  │  (per match field)   │     │
│  │                      │  │  HNSW index per      │     │
│  │  HNSW index per      │  │  match field         │     │
│  │  match field         │  │                      │     │
│  └──────────────────────┘  └──────────────────────┘     │
│            ...N blocks...                               │
└─────────────────────────────────────────────────────────┘
```

**Data structures maintained during load:**

- `BlockRouter`: `HashMap<BlockKey, BlockId>` — maps composite blocking
  key (e.g. `("US", "finance")`) to a block identifier. O(1) lookup.
- `RecordBlockMap`: `HashMap<RecordId, BlockId>` — maps each record to
  its block. Needed for the reverse lookup: given a record ID, which
  block's HNSW index contains its vector?
- `PerBlockIndices`: `Vec<BlockState>` where each `BlockState` holds one
  HNSW index per match field, plus a `HashMap<RecordId, Side>` tracking
  which records are A-side vs B-side within that block.

### Record ingestion (per record)

For each record (A or B side):

1. Compute block key from record fields (same as current blocking)
2. Look up or create the block via `BlockRouter`
3. Add record ID to `RecordBlockMap`
4. For each match field configured as embedding:
   - Compute embedding text from the record's field value
   - Check if this text already has a vector in the block's HNSW index
     (text hash → vector key lookup)
   - If yes: reuse the existing vector (skip ONNX encode)
   - If no: encode via ONNX, insert into the block's HNSW index
5. Store the record in the record store (DashMap, same as today)

### Query phase (matching)

For each query record (say B-side, matching against A-side):

1. Compute block key → find block ID via `BlockRouter`
2. For each match field:
   - Get or compute the query vector for this field
   - Search the block's HNSW index for this field (unfiltered, top-K)
   - Filter results to A-side records only (since the index contains
     both sides; this is a post-filter on K results, not a search
     filter on N vectors — trivially fast)
3. Merge results across fields: weighted combination of per-field
   vector similarities
4. Classify top result (auto/review/no-match)

### Persistence

Each block's HNSW indices persist to disk under a job-specific
directory:

```
~/.melder/jobs/{job_name}/blocks/
├── block_0/
│   ├── field_0.usearch       # HNSW index for match field 0
│   ├── field_0.keys          # text_hash → vector_key mapping
│   ├── field_1.usearch
│   ├── field_1.keys
│   └── meta.json             # block key, record count, last updated
├── block_1/
│   └── ...
└── manifest.json             # block router state, schema version
```

On job startup:

1. Load `manifest.json` → reconstruct `BlockRouter`
2. For each block, load HNSW indices from disk (usearch native
   save/load is fast: ~90ms for 100K vectors)
3. Ingest new records — only encode text not already in the indices
4. Save updated indices at job completion

---

## Threading model and concurrency

### Load phase: embarrassingly parallel with one contention point

```
                    ┌─────────────────────┐
                    │   Record Stream      │
                    │   (A or B dataset)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Blocking Key      │
                    │   Computation       │  ← pure, no shared state
                    │   (per record)      │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │      ONNX Encoding              │
              │      (encoder pool: N sessions) │  ← Mutex per session
              │                                 │     (existing design)
              │   Batched: group records by     │
              │   block, encode in batches of   │
              │   64-256 for ONNX throughput     │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │      Block HNSW Insertion       │
              │                                 │
              │   RwLock per block               │
              │   Multiple blocks can be        │  ← parallelism across
              │   written concurrently          │     blocks, serialised
              │                                 │     within a block
              └─────────────────────────────────┘
```

**Where contention exists:**

- **Encoder pool**: `Mutex` per ONNX session slot (existing design,
  proven at scale). N sessions = N concurrent encodes. This is the
  dominant cost and the serialisation is unavoidable — ONNX sessions
  are not thread-safe.

- **Per-block HNSW insert**: `RwLock` per block. HNSW insertion is
  O(log N) and takes ~0.3ms per vector. Since each block has its own
  lock, inserts into different blocks run fully in parallel. Within a
  single block, inserts serialise — but this is fine because the ONNX
  encode upstream is 10× slower, so the insert lock is never the
  bottleneck.

- **BlockRouter / RecordBlockMap**: written during load, read during
  query. A `DashMap` or a build-then-freeze pattern (populate during
  load behind a `Mutex`, then `Arc<HashMap>` for queries) eliminates
  contention during the query phase.

**Where contention does NOT exist:**

- Record store (DashMap) — lock-free reads, sharded writes
- Blocking key computation — pure function, no shared state
- Text hash lookup (is this text already encoded?) — read-only during
  query phase; during load, per-block so same lock as the HNSW insert

### Query phase: fully parallel, no write contention

```
              ┌────────────────────────────────────┐
              │     Query Records (Rayon par_iter)  │
              └──────────┬─────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Per query record:            │
         │                               │
         │  1. block key → BlockRouter   │  ← read-only HashMap
         │     (no lock)                 │
         │                               │
         │  2. per-field HNSW search     │  ← RwLock read on block
         │     (multiple fields in       │     (concurrent reads OK)
         │     sequence per record)      │
         │                               │
         │  3. merge scores, classify    │  ← pure computation
         │                               │
         │  4. crossmap update           │  ← Mutex (brief, same as
         │     (if auto-match)           │     today)
         └───────────────────────────────┘
```

**During the query phase, all block indices are read-only.** Multiple
query records can search the same block's HNSW index concurrently
because `RwLock` allows concurrent readers. The only write contention
is the crossmap update on auto-match, which is a brief `Mutex` lock on
a HashMap insert — identical to the current design.

This means **query throughput scales linearly with cores** up to the
point where memory bandwidth saturates (which, for HNSW's pointer-
chasing access pattern, is very high).

### Live mode: read-heavy with occasional writes

In live mode, records arrive one at a time via HTTP. The pattern is:

- **Add record**: encode (Mutex on encoder), insert into block HNSW
  index (RwLock write on that block), insert into record store
  (DashMap). Then search the same block for matches (RwLock read).
  The write lock is held only during HNSW insert (~0.3ms), not during
  search.

- **Match record**: blocking lookup (read-only), HNSW search per field
  (RwLock read on block). No writes except crossmap. Fully concurrent.

- **Remove record**: remove from record store (DashMap), remove from
  RecordBlockMap. Do NOT remove from HNSW index (orphan retention).
  No HNSW lock needed at all.

The write contention profile is better than today's design: currently
a single `RwLock<VecIndex>` serialises all writes across the entire
index. With per-block locks, writes to different blocks never contend.
For a job with 1000 blocks, contention drops by ~1000×.

---

## Performance projections

### Benchmark data (from usearch evaluation)

| Operation | 10K vectors | 100K vectors |
|-----------|-------------|--------------|
| HNSW unfiltered search (top-10) | 0.14ms | 0.36ms |
| HNSW insert | 0.36ms | 0.87ms |
| Flat brute-force search | 1.87ms | 18.5ms |

HNSW search scales as O(log N). From 10K to 100K (10× more data),
search time increased by only 2.6×. Extrapolating:

| Block size | HNSW search time (projected) |
|------------|------------------------------|
| 1K | ~0.08ms |
| 10K | ~0.14ms |
| 100K | ~0.36ms |
| 1M | ~0.7ms |

### Projected per-record query cost

Assume 4 match fields, all embedding-based, with typical block sizes:

**Well-designed blocking (1K–10K per block):**

| Component | Time | Notes |
|-----------|------|-------|
| Block lookup | <0.01ms | HashMap, O(1) |
| 4× HNSW search | 4 × 0.14ms = 0.56ms | unfiltered, per field |
| Score merge + classify | <0.01ms | arithmetic |
| **Total (warm)** | **~0.6ms** | **no encoding** |
| ONNX encode (cold) | +3ms per new field | amortised across runs |

Compare to the current pipeline on the same data:

| Component | Time (current) | Time (proposed) |
|-----------|---------------|-----------------|
| Blocking | <0.01ms | <0.01ms |
| Candidate selection (wratio on 5K blocked) | ~10ms | eliminated |
| Full scoring (10 candidates × 4 fields) | ~0.3ms | ~0.6ms |
| **Total** | **~10.3ms** | **~0.6ms** |

That's a **17× improvement** for warm queries, entirely from
eliminating the candidate selection bottleneck.

**Coarse blocking (100K per block):**

| Component | Time (current) | Time (proposed) |
|-----------|---------------|-----------------|
| Candidate selection (wratio on 100K) | ~200ms | eliminated |
| 4× HNSW search | N/A | 4 × 0.36ms = 1.44ms |
| **Total** | **~200ms** | **~1.5ms** |

**133× improvement.** This is where the design really shines — it
makes coarse blocking viable. Users no longer need to carefully tune
blocking keys to keep block sizes small.

### Projected batch throughput

At 1M × 1M (both sides have 1M records):

**Current design** (extrapolated from 100K × 100K at 306 rec/s):

The current pipeline scales roughly as O(avg_block_size) per query
due to candidate selection. At 1M with the same blocking keys, block
sizes grow 10×, so throughput drops ~10×: ~30 rec/s. For 1M B records,
that's ~9 hours.

**Proposed design** with 8 cores:

Per query: ~0.6ms (well-blocked) to ~1.5ms (coarse-blocked)
Single-threaded: 660–1,600 rec/s
8 cores (Rayon): 5,000–13,000 rec/s
1M B records: 1.3–3.3 minutes

Even with coarse blocking:
Per query: ~1.5ms
8 cores: ~5,000 rec/s
1M B records: ~3.3 minutes

**That's a ~160× improvement over the current projected 9 hours.**

### Projected scaling limits

| Scale | Block count | Avg block size | HNSW search | Total per query | 1M queries (8 cores) |
|-------|-------------|----------------|-------------|-----------------|----------------------|
| 100K | 500 | 200 | ~0.06ms | ~0.3ms | 23 seconds |
| 1M | 2,000 | 500 | ~0.08ms | ~0.4ms | 50 seconds |
| 10M | 5,000 | 2,000 | ~0.12ms | ~0.5ms | 63 seconds |
| 100M | 10,000 | 10,000 | ~0.14ms | ~0.6ms | 75 seconds |

The scaling is nearly flat because HNSW search grows logarithmically
with block size, and the number of HNSW searches per query is constant
(= number of match fields).

The physical limits at extreme scale become:

1. **Memory**: each HNSW index stores vectors (1.5KB/vec) plus graph
   edges (~1.5× overhead). At 100M vectors across all blocks:
   ~100M × 1.5KB × 2.5 = ~375GB. This requires sharding across
   machines, or quantised vectors (uint8: ~95GB), or mmap-backed
   indices.

2. **Disk**: persisted indices at 100M vectors: ~375GB. SSD read for
   cold load: ~4 seconds (at 100GB/s NVMe).

3. **ONNX encoding on cold start**: 100M records × 4 fields × 3ms =
   ~139 hours on CPU. This is a one-time cost, parallelisable across
   machines, and fully cached thereafter. GPU inference (50K texts/s)
   brings it to ~2 hours.

---

## All-embedding scoring: the shift

The current design uses a mix of scoring methods per field: embedding,
fuzzy, exact, numeric. The proposed design works best when all (or
most) match fields use embedding scoring, because each field needs its
own HNSW index for the direct vector lookup.

**What this means for each current method:**

- **embedding**: direct fit. One HNSW index per field per block.

- **fuzzy**: replaced by embedding similarity. For entity names, the
  embedding model captures both semantic similarity and character-level
  similarity (MiniLM-L6-v2 gives high scores for "Goldman Sachs" vs
  "Goldman Sachs International" just as wratio would). Edge cases where
  character-level similarity matters but semantic similarity doesn't
  (e.g. "AAA Corp" vs "AAB Corp") are rare in entity matching and can
  be handled by a post-scoring exact/fuzzy check on the top-K results.

- **exact**: doesn't need HNSW. Exact match fields (like currency code,
  country code) can be checked as a post-filter on the top-K results
  from the HNSW search, or incorporated into the blocking key (which is
  their natural home anyway — if two records must have the same currency
  to match, that's a blocking constraint, not a scoring signal).

- **numeric**: similar to exact — a post-filter or score adjustment on
  top-K results. Numeric fields (like amounts) aren't embeddable in a
  meaningful way; they remain as arithmetic comparisons on a small
  result set.

The pipeline becomes:

```
HNSW search (embedding fields) → top-K per field
         ↓
merge across fields (weighted combination)
         ↓
post-score exact/numeric fields on merged top-K
         ↓
final weighted score → classify
```

The exact/numeric post-scoring adds negligible cost because it operates
on only K results (typically 10–20), not the full block.

---

## What this does NOT change

- **Config format**: same YAML, same field definitions. The `method`
  field on each match field still controls scoring. The engine
  internally decides whether to use HNSW search or post-scoring based
  on the method.

- **API**: same HTTP endpoints, same request/response format. The
  internal pipeline change is invisible to callers.

- **Crossmap**: unchanged. Still tracks A↔B pairs, still persisted.

- **Record store**: still DashMap. Records are stored the same way.

- **Blocking config**: same fields, same operators. The blocking key
  now additionally determines which HNSW index to search, but the
  user-facing config is identical.

---

## Implementation phases

### Phase 1: Per-block index infrastructure

- `BlockRouter`: HashMap<BlockKey, BlockId> with build-then-freeze
  lifecycle
- `RecordBlockMap`: HashMap<RecordId, BlockId>
- `BlockState`: holds per-field HNSW indices + record membership
- Persistence: save/load block indices to job-specific directory
- Text hash deduplication: skip encoding for already-seen text

### Phase 2: All-embedding scoring

- Encode each match field separately (currently only one field is
  embedded — the "primary embedding text" concatenation)
- Build per-field HNSW indices within each block
- Search pipeline: N HNSW lookups → merge → post-score exact/numeric
- Benchmark against current pipeline at 10K, 100K, 1M

### Phase 3: Warm-cache optimisation

- Persist block indices between job runs
- Text hash → vector key mapping for encoding deduplication
- Incremental index updates (new text only)
- `meld compact` command for optional index maintenance

### Phase 4: Live mode integration

- Per-block RwLock for concurrent read/write
- HNSW insert on add, orphan retention on remove
- Query uses same search pipeline as batch

---

## Design considerations

### Block stability and the two-mode approach

usearch is a self-contained index: each `Index` stores its own copy of
every vector. When you `add(key, vector)`, the vector is `memcpy`-ed
into the index's internal storage. When you `save()`, vectors are
serialised alongside the HNSW graph into the same file.

This means per-block indices work efficiently **only when blocks are
disjoint partitions** — each record belongs to exactly one block, so
each vector is stored exactly once across all indices. With AND
blocking (the default), this holds: a record with `country_code=US`
and `sector=finance` maps to exactly one block `US/finance`. No
duplication.

But it also means **the block definition must be stable**. If a user
blocks on `country_code` for a month, accumulating vectors across
daily runs, then switches to `country_code AND sector`, every vector
is now in the wrong index. The entire usearch index set would need to
be rebuilt — re-inserting all vectors into new per-block indices.
Switching back a week later leaves the previous indices orphaned on
disk. Without discipline, database size inflates rapidly.

This leads to a two-mode design:

**Flat mode** (current, default): brute-force `VecIndex`, file-backed,
cheap to rebuild, no commitment to block structure. This is the
exploration and tuning mode. Users experiment with different blocking
configs, different match fields, run benchmarks. Fast to iterate,
no persistent state beyond the vector cache files.

**usearch mode** (opt-in): per-block HNSW indices, persistent on disk,
optimised for recurring jobs with a stable config. Users opt in once
they've settled on a blocking configuration that works for their data.
Controlled via a config flag:

```yaml
vector_backend: flat      # default — exploration mode
vector_backend: usearch   # opt-in — production mode
```

Or triggered explicitly via CLI: `meld optimise --job foo` to build
persistent usearch indices from the existing flat data.

On startup in usearch mode, melder checks the persisted manifest's
blocking config hash against the current config. If they match, load
the indices. If they don't, warn the user:

> "Blocking config has changed since usearch indices were built.
> Run `meld rebuild --job foo` or switch to `vector_backend: flat`."

### Record-to-block mapping as metadata

Rather than maintaining a separate `HashMap<RecordId, BlockId>`
alongside the record store, the block assignment becomes metadata
on the record itself, computed at ingestion time and carried along
for the record's lifetime.

At startup during dataset load, for each record:

1. Compute the block key from the record's blocking fields (same
   logic as today's `BlockingIndex`)
2. Look up or create the block ID via the `BlockRouter`
3. Attach the block ID as metadata on the record

In live mode, the same flow: a record arrives via HTTP, the block key
is computed, the block ID is assigned and attached. From that point on,
any operation involving the record — search, scoring, remove — knows
which block (and therefore which usearch index) it belongs to, without
a separate lookup table.

The record store evolves from `DashMap<String, Record>` to
`DashMap<String, StoredRecord>`:

```rust
struct RecordMeta {
    block_id: u32,
    // future: vector_cached, last_encoded, etc.
}

struct StoredRecord {
    fields: Record,        // the user's data (HashMap<String, String>)
    meta: RecordMeta,      // system metadata, not visible to the user
}
```

This keeps system metadata out of the user's field namespace and is
type-safe. The `block_id` is a cheap `u32` — no string allocation, no
hashing at lookup time.

For the reverse direction (block → records), the usearch index already
tracks which keys it contains. No separate reverse map needed.

### Block-to-index mapping and persistence

The block-to-usearch-index mapping is a `Vec<BlockIndices>` indexed
by `block_id`:

```rust
struct BlockIndices {
    block_key: BlockKey,           // e.g. ("US", "finance")
    field_indices: Vec<Index>,     // one usearch Index per match field
}
```

Small, fast, held in memory during operation. Persisted to disk
alongside a manifest:

```
~/.melder/jobs/{job_name}/
├── manifest.json
│     blocking_config_hash: "a3f8..."
│     embedding_model: "all-MiniLM-L6-v2"
│     embedding_dim: 384
│     match_fields: ["legal_name", "address", ...]
│     schema_version: 1
│     blocks: [
│       { id: 0, key: ["US", "finance"], records: 1234 },
│       { id: 1, key: ["GB", "tech"], records: 567 },
│       ...
│     ]
├── blocks/
│   ├── 0/
│   │   ├── field_0.usearch    # legal_name vectors + HNSW graph
│   │   ├── field_1.usearch    # address vectors + HNSW graph
│   │   └── ...
│   ├── 1/
│   │   └── ...
│   └── ...
└── block_router.bin           # serialised BlockKey → BlockId map
```

The manifest stores:
- **Blocking config hash** — detect config changes between runs
- **Embedding model and dimension** — detect model changes
- **Match field list** — detect field changes (which fields have
  indices)
- **Block directory** — block keys, IDs, record counts
- **Schema version** — forward compatibility

### Recomputing block assignments at startup

Block assignments are **recomputed at startup**, not persisted per
record. Loading records from csv/parquet, computing blocking keys,
and assigning block IDs is deterministic and cheap — it's HashMap
lookups on string fields, no ONNX encoding involved.

Recomputation guarantees consistency with the current config. If the
blocking config has changed (detected via the manifest hash), melder
can either:

- Warn and refuse to load stale usearch indices (safe default)
- Recompute block assignments and rebuild indices automatically
  (expensive but correct)
- Fall back to flat mode for this run

Recomputation also handles records that have been added or removed
between runs. New records get assigned to blocks and their vectors
inserted into the relevant usearch indices. Removed records leave
orphan vectors in the indices — harmless, as discussed earlier.

For live mode, block assignment happens inline: record arrives,
block key computed, block ID assigned, stored in `RecordMeta`,
vector inserted into the block's usearch index. No startup
recomputation needed for live-added records — they're assigned
on arrival.

### Index garbage collection

Over time, block definitions may shift. A user blocks on
`country_code` for three months, then refines to
`country_code AND sector`. The old single-field blocks become
orphans — they're on disk, consuming space, but no record points
to them any more. Without cleanup, this accumulates indefinitely.

The solution is a TTL-based garbage collector. Each block tracks
the last time it was accessed (searched or inserted into), persisted
as a timestamp in the manifest:

```json
{
  "blocks": [
    { "id": 0, "key": ["US", "finance"], "records": 1234, "last_accessed": "2026-03-08T14:30:00Z" },
    { "id": 1, "key": ["GB"],            "records": 0,    "last_accessed": "2026-01-15T09:00:00Z" },
    ...
  ]
}
```

Block 1 in the example above is a leftover from the old blocking
config — zero current records, last accessed two months ago.

**Collection strategy:**

A periodic cleanup job runs either as:

- A background thread in long-running live mode (checks once per
  hour, lightweight — just reads timestamps)
- A CLI command for batch users: `meld gc --job foo`
- Automatically at startup before loading indices

The job walks the manifest and deletes any block whose
`last_accessed` timestamp exceeds a configurable TTL:

```yaml
vector_backend: usearch
usearch:
  gc_ttl_days: 30    # delete block indices not accessed in 30 days
```

Deletion means removing the block's directory from disk
(`~/.melder/jobs/{job_name}/blocks/{id}/`) and its entry from the
manifest. The vectors are gone, but they were already orphaned —
no current record references them. If the same block key reappears
in future data, the index is simply rebuilt from scratch (the ONNX
encode cost is the real expense; the HNSW build is fast).

**Access timestamp updates:**

- **Batch mode**: at the start of a run, touch `last_accessed` on
  every block that has at least one record assigned to it during
  block recomputation. Blocks with zero records are not touched —
  they start aging toward the TTL.

- **Live mode**: touch `last_accessed` on a block whenever a search
  or insert hits it. To avoid write amplification (updating the
  manifest on every request), buffer timestamps in memory and flush
  to disk periodically (e.g. every 5 minutes, or at shutdown).

**Disk accounting:**

The manifest could also track per-block disk usage (sum of
`.usearch` file sizes) so that `meld gc --job foo --dry-run`
can report:

```
Blocks eligible for GC (not accessed in 30+ days):
  block 1 ("GB")           — 12.3 MB, last accessed 2026-01-15
  block 7 ("DE/auto")      — 4.1 MB,  last accessed 2026-02-01
  Total reclaimable: 16.4 MB

Use `meld gc --job foo` to delete.
```

This keeps storage under control without requiring the user to
understand the internal block structure. The TTL default of 30 days
is conservative — most stale blocks from config changes will be
caught within a month, while blocks from infrequently-run jobs
(e.g. monthly reconciliations) survive comfortably.

### Decision: VectorDB trait as the sole interface

The `VectorDB` trait (already defined in `src/vectordb/mod.rs`)
becomes the **only** interface the rest of melder uses for vector
operations. Everything behind it — whether flat brute-force or
per-block HNSW with manifests, garbage collection, and persistent
indices — is encapsulated.

```
                    ┌─────────────────────────┐
                    │   rest of melder         │
                    │   (session, batch,       │
                    │    pipeline, API)         │
                    └────────┬────────────────┘
                             │
                     VectorDB trait
                             │
              ┌──────────────┼──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │   FlatVectorDB    │       │   UsearchVectorDB     │
    │                   │       │                       │
    │   HashMap + Vec   │       │   BlockRouter         │
    │   brute-force     │       │   RecordBlockMap      │
    │   single file     │       │   per-block Indices   │
    │                   │       │   manifest.json       │
    │                   │       │   GC / TTL            │
    └───────────────────┘       └───────────────────────┘
```

The user switches between backends with a single config line:

```yaml
vector_backend: flat       # FlatVectorDB — exploration mode
vector_backend: usearch    # UsearchVectorDB — production mode
```

At startup, the factory picks the implementation:

```rust
let vector_db: Arc<dyn VectorDB> = match config.vector_backend {
    "flat"    => Arc::new(FlatVectorDB::new(dim)),
    "usearch" => Arc::new(UsearchVectorDB::new(dim, &config, job_dir)?),
};
```

Everything else — session, batch engine, pipeline, API — holds
`Arc<dyn VectorDB>` and calls trait methods. The blocking config,
manifest, GC, persistence, block routing — all encapsulated inside
`UsearchVectorDB`. The rest of melder never sees blocks.

**Trait interface evolution.** The current trait methods
(`search`, `upsert`, `get`, etc.) need to carry enough context for
the usearch backend to determine block membership without exposing
block concepts to callers. The approach:

- `upsert(id, record, vec)` — takes the full record so the backend
  can compute the block key internally. `FlatVectorDB` ignores the
  record; `UsearchVectorDB` extracts blocking fields from it.
- `search(query_vec, top_k)` and `search_filtered(query_vec, top_k,
  allowed_ids)` — for the usearch backend, the caller's block
  membership is determined from the query record (passed at a higher
  level and stored in internal state during upsert). The trait never
  exposes block IDs.
- `save()` / `load()` — each backend handles its own persistence
  format. Flat writes a single binary cache file. Usearch writes
  a manifest + per-block index directory.

**What this eliminates from the current codebase (~550 lines):**

The `index/` module (`flat.rs`, `cache.rs`, `mod.rs` — 696 lines)
is absorbed into `vectordb/`. `VecIndex` becomes an internal
implementation detail of `FlatVectorDB`, no longer a public type.
Cache serialisation moves inside the flat backend.

Three duplicate copies of `build_or_load_index` across `state.rs`,
`live.rs`, and `batch/engine.rs` (~270 lines) collapse into the
trait's `save`/`load` methods — each backend handles its own
persistence internally.

Manual `RwLock<VecIndex>` wrapping in `live.rs` (~40 lines) is
absorbed — backends manage their own concurrency. `FlatVectorDB`
already has an internal `RwLock`; `UsearchVectorDB` has per-block
locks.

All references to `VecIndex` across the codebase change to
`dyn VectorDB` or `Arc<dyn VectorDB>`:

| Location | Current | After |
|----------|---------|-------|
| `state/state.rs` | `index_a: VecIndex` | `index_a: Arc<dyn VectorDB>` |
| `state/live.rs` | `index: RwLock<VecIndex>` | `index: Arc<dyn VectorDB>` |
| `batch/engine.rs` | `index_a: &VecIndex` | `index_a: &dyn VectorDB` |
| `matching/pipeline.rs` | `pool_index: &VecIndex` | `pool_index: &dyn VectorDB` |
| `matching/candidates.rs` | `pool_index: &VecIndex` | `pool_index: &dyn VectorDB` |
| `session/mod.rs` | accesses via `LiveSideState.index` | unchanged interface |

**Combined simplification budget.** Together with other cleanup
(handler boilerplate, main.rs extraction, dead config fields), the
trait refactor brings the codebase from ~10,400 lines to ~9,200 —
a ~1,200 line reduction with cleaner architecture, no dead code,
and a single swappable vector backend.

---

## Risks and mitigations

**Risk: embedding similarity misses matches that wratio catches.**
Mitigation: run the current wratio pipeline in parallel on a test
dataset and compare recall. If gaps exist, add a hybrid mode that
falls back to wratio for the top-K HNSW results that score below a
confidence threshold.

**Risk: per-field encoding multiplies ONNX cost.**
Mitigation: batch encode all fields for a record in a single ONNX call
(concatenate field texts with separator tokens). Or accept the cost as
a one-time investment that's fully amortised by caching.

**Risk: thousands of small HNSW indices consume more memory than one
large flat index.**
Mitigation: HNSW overhead is ~1.5× per vector. With 1M total vectors
split across 2,000 blocks of 500, the overhead is the same 1.5× total.
The per-index fixed overhead (metadata, graph root) is negligible.
Monitor total memory and fall back to flat search for very small blocks
(< 100 vectors) where HNSW overhead isn't justified.

**Risk: usearch build time is slow for initial cold start.**
Mitigation: HNSW insert is O(log N) per vector and the bottleneck is
the ONNX encode upstream (3ms vs 0.3ms for insert). The HNSW build
is never the bottleneck. Once built, indices persist and the cost is
not repeated.

**Risk: block key changes between runs invalidate persisted indices.**
Mitigation: store the blocking config hash in the manifest. If the
config changes, rebuild all indices. This is the same staleness
approach used for the current VecIndex cache, just extended to blocks.

---

## Code cleanup (independent of new direction)

The following simplifications can be done now. They have no
behavioural impact and don't depend on the usearch or all-embedding
work — they're pure code quality improvements that reduce the
codebase and make subsequent changes easier.

### Collapse API handler boilerplate

`src/api/handlers.rs` is ~438 lines containing 14 handlers that
all follow the identical pattern:

```rust
pub async fn add_a(State(s): State<AppState>, Json(body): Json<Value>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        s.add(body, Side::A)
    }).await;
    match result {
        Ok(Ok(resp)) => { tracing::info!("add_a ok"); Json(resp).into_response() }
        Ok(Err(e))   => { tracing::error!("add_a: {}", e); /* error json */ }
        Err(e)       => { tracing::error!("add_a panic: {}", e); /* 500 */ }
    }
}
```

The only difference between handlers is the session method called
and the `Side` parameter. A generic helper eliminates the
repetition:

```rust
async fn blocking_handler<F, R>(
    session: AppState, op: &str, f: F,
) -> impl IntoResponse
where
    F: FnOnce(Arc<Session>) -> Result<R, SessionError> + Send + 'static,
    R: Serialize,
{
    // spawn_blocking + match + logging — written once
}

pub async fn add_a(State(s): State<AppState>, Json(body): Json<Value>) -> impl IntoResponse {
    blocking_handler(s, "add_a", |session| session.add(body, Side::A)).await
}
```

Each handler becomes a 2-3 line wrapper. The A/B side pairs share
the same logic with only a `Side` parameter difference. 438 lines
drops to ~150 lines. No behaviour change.

### Extract CLI commands from main.rs

`src/main.rs` is 1374 lines containing all CLI command
implementations inline: `cmd_run` (187 lines), `cmd_tune`
(215 lines), `cmd_review_import` (185 lines), etc. None of this
logic is independently testable.

Extract each command group into `src/cli/` modules:

```
src/cli/
├── mod.rs           # dispatcher
├── run.rs           # cmd_run (batch execution)
├── serve.rs         # cmd_serve (server startup)
├── tune.rs          # cmd_tune (threshold tuning)
├── cache.rs         # cmd_cache_info, cmd_cache_build
├── review.rs        # cmd_review_import, cmd_review_export, ...
└── crossmap.rs      # cmd_crossmap_*
```

`main.rs` becomes a ~50 line file: parse CLI args, dispatch to
the right module. Each command module can be unit-tested
independently.

### Remove dead config fields

| Field | Why it's dead |
|-------|---------------|
| `cross_map.redis_url` | No Redis CrossMap implementation exists |
| `Config.sidecar` | Go-era field, silently ignored |
| `Config.workers` (top-level) | Deprecated, migrated to `performance.workers` |
| `LiveConfig.encoder_pool_size` | Deprecated, migrated to `performance.encoder_pool_size` |

Remove the fields and migration shims. If a user has them in
their YAML, serde will give a clear parse error telling them what
to change. Better than silently ignoring dead config.

### Collapse single-file module shims

Several modules have a `mod.rs` that does nothing but re-export
from a single file:

| Module | `mod.rs` lines | Single file |
|--------|---------------|-------------|
| `crossmap/mod.rs` | 5 | `local.rs` (231 lines) |
| `encoder/mod.rs` | 10 | `pool.rs` (203 lines) |

Rename the single file to `mod.rs` in each case. Eliminates
unnecessary indirection.

### Merge small scoring files

`scoring/exact.rs` (58 lines — one function + tests) and
`scoring/embedding.rs` (122 lines — three math functions + tests)
are small enough to live in `scoring/mod.rs` without it becoming
unwieldy. The merged file would be ~430 lines.

### Summary of cleanup impact

| Change | Lines saved |
|--------|-------------|
| Handler boilerplate collapse | ~290 |
| main.rs extraction | structural (no line reduction, but testable) |
| Dead config fields | ~40 |
| Module shim collapse | ~25 |
| Scoring file merge | structural (fewer files) |
| **Total** | **~350 lines + structural improvements** |

Combined with the ~550 lines saved by the VectorDB trait refactor
(absorbing `index/`, deduplicating `build_or_load_index`, absorbing
`RwLock` wrapping), the total reduction is ~900 lines of code
removal plus significantly cleaner project structure.

---

## Summary

The current pipeline's bottleneck at scale is candidate selection:
O(block_size) string comparisons that dominate wall time once blocks
exceed a few thousand records. HNSW vector search eliminates this
entirely by replacing O(N) candidate scoring with O(log N) vector
lookup.

The key architectural move is **one HNSW index per block per field**,
not one giant filtered index. This plays to HNSW's strength (fast
unfiltered search) and avoids its weakness (slow filtered search).

Combined with persistent indices that accumulate vectors across job
runs, this creates a system where:

- Recurring jobs get faster over time (more cache hits, less encoding)
- Block size doesn't matter (HNSW is scale-insensitive)
- The pipeline is simpler (two stages instead of three)
- Query throughput is nearly constant regardless of dataset size

---

## Food for thought: do we need blocks at all?

The per-block design eliminates the candidate selection bottleneck by
making each HNSW search unfiltered within a block. But there's a
simpler question: if HNSW unfiltered search at 1M vectors takes ~0.7ms,
why not search the entire dataset directly?

The pipeline would collapse to its simplest possible form:

```
record → encode fields → N HNSW searches (full index) → merge → classify
```

No blocking. No candidate selection. No block router. No per-block
indices. No blocking config for the user to get wrong. One HNSW index
per match field per side, total.

Where blocking previously served as a hard constraint ("only match
within the same country"), that becomes a post-filter on the top-K
results: search the full index, get top-20, discard any where country
doesn't match. At 20 results, that filter is free.

This trades some raw throughput for radical simplicity. Cross-block
matches become possible — a record miscategorised by a coarse blocking
key still gets found by vector similarity. The entire blocking
configuration section disappears from the user's YAML.

### Projected performance: blockless design at 100K × 100K

Current measured performance (from README, Apple Silicon, warm caches):

| Metric | 1K × 1K | 10K × 10K | 100K × 100K |
|--------|--------:|----------:|------------:|
| Batch throughput | 30,000 rec/s | 4,200 rec/s | 306 rec/s |
| Batch total time | <0.1s | 2.4s | 5m 27s |

The current 100K × 100K result (306 rec/s, 5m 27s) is dominated by
candidate selection — wratio running across thousands of blocked
records per query. The actual full scoring on the final 10 candidates
is fast; it's the funnel that's expensive.

**Blockless all-embedding batch at 100K × 100K (projected):**

| Component | Time per query | Notes |
|-----------|---------------|-------|
| 4 HNSW searches (100K index) | 4 × 0.36ms = 1.44ms | measured benchmark data |
| Post-filter exact/numeric on top-20 | <0.01ms | trivial |
| Classify | <0.01ms | arithmetic |
| **Total per query** | **~1.5ms** | |

Single-threaded: ~660 queries/sec.
8 cores (Rayon): ~5,000 queries/sec.
100K B-side queries at 5,000/s: **~20 seconds**.

Compare to the current 5 minutes 27 seconds. That's a **16× improvement**
from eliminating candidate selection entirely, with a simpler config
and no blocking tuning required.

### Projected performance: blockless live mode at 100K × 100K

Current measured live performance (10K × 10K, from README):

| Metric | Sequential (c=1) | Concurrent (c=10) |
|--------|-----------------|-------------------|
| Throughput | 255 req/s | 760 req/s |
| p50 latency | 3.8ms | 11.6ms |

The live mode bottleneck is ONNX encoding (~2-3ms per request), which
doesn't change with dataset size. What does change is the scoring
pipeline cost downstream of encoding. At 10K × 10K, the scoring
pipeline (blocking + candidates + scoring) adds ~1ms. At 100K × 100K
with the current design, larger blocks make candidate selection
slower — wratio across bigger blocks could add 5-20ms depending on
blocking selectivity, dragging sequential throughput down toward
~100-150 req/s.

**Blockless all-embedding live at 100K × 100K (projected):**

| Component | Time | Notes |
|-----------|------|-------|
| ONNX encode | ~3ms | unchanged, still dominant |
| 4 HNSW searches (100K index) | 1.44ms | O(log N), nearly scale-free |
| Post-filter + classify | <0.01ms | |
| VecIndex upsert + WAL | ~0.1ms | unchanged |
| **Total** | **~4.5ms** | |

Sequential: ~220 req/s (slightly slower than current 10K due to HNSW
search cost, but crucially this does NOT degrade further at 1M).
Concurrent (c=10): ~650-700 req/s.

Compare to the projected current design at 100K where candidate
selection bloat would push sequential down to ~100-150 req/s. The
blockless HNSW approach holds steady because HNSW search time grows
logarithmically — going from 100K to 1M adds only ~0.3ms to the
search, invisible next to the 3ms encode.

| Metric | Current 10K×10K (measured) | Current 100K×100K (projected) | Blockless 100K×100K (projected) |
|--------|--------------------------|------------------------------|-------------------------------|
| Sequential throughput | 255 req/s | ~100-150 req/s | ~220 req/s |
| Concurrent (c=10) | 760 req/s | ~400-500 req/s | ~650-700 req/s |
| p50 latency (seq) | 3.8ms | ~7-10ms | ~4.5ms |

The key insight: **live mode throughput becomes encoding-bound, not
scoring-bound, at every dataset size.** The scoring pipeline shrinks
to a constant ~1.5ms regardless of whether the index holds 10K or 10M
vectors. The only way to push throughput higher is faster encoding
(GPU, quantised models, request coalescing) — which is a much better
problem to have than "redesign your blocking keys."

### What if we kept blocks? Per-block HNSW at 100K × 100K

The main proposal in this document uses per-block HNSW indices. Here's
what that looks like at 100K × 100K for direct comparison.

With well-designed blocking (e.g. `country_code AND sector`), 100K
records might split into ~500 blocks averaging 200 records each. With
coarser blocking (just `country_code`), maybe ~50 blocks averaging
2,000 records.

**Per-block batch at 100K × 100K (well-blocked, ~200 per block):**

| Component | Time per query | Notes |
|-----------|---------------|-------|
| Block lookup | <0.01ms | HashMap O(1) |
| 4 HNSW searches (200-vec index) | 4 × ~0.05ms = 0.20ms | tiny index, nearly instant |
| Post-filter + classify | <0.01ms | |
| **Total per query** | **~0.2ms** | |

Single-threaded: ~5,000 queries/sec.
8 cores (Rayon): ~30,000-40,000 queries/sec.
100K B-side queries: **~2.5-3 seconds.**

**Per-block batch at 100K × 100K (coarse-blocked, ~2K per block):**

| Component | Time per query | Notes |
|-----------|---------------|-------|
| Block lookup | <0.01ms | |
| 4 HNSW searches (2K-vec index) | 4 × ~0.10ms = 0.40ms | still small |
| Post-filter + classify | <0.01ms | |
| **Total per query** | **~0.4ms** | |

Single-threaded: ~2,500 queries/sec.
8 cores: ~15,000-20,000 queries/sec.
100K B-side queries: **~5-7 seconds.**

**Per-block live at 100K × 100K:**

| Component | Time | Notes |
|-----------|------|-------|
| ONNX encode | ~3ms | unchanged, dominant |
| Block lookup | <0.01ms | |
| 4 HNSW searches (200–2K index) | 0.20-0.40ms | block-size dependent |
| Post-filter + classify | <0.01ms | |
| VecIndex upsert + WAL | ~0.1ms | |
| **Total** | **~3.3-3.5ms** | |

Sequential: ~285-300 req/s.
Concurrent (c=10): ~800-900 req/s.

### Side-by-side: all three approaches at 100K × 100K

**Batch mode (warm caches, 8 cores):**

| Approach | Per-query cost | Throughput | 100K queries | vs current |
|----------|---------------|------------|--------------|------------|
| Current (wratio candidates) | ~3.3ms | 306 rec/s | 5m 27s | baseline |
| Blockless HNSW | ~1.5ms | ~5,000 rec/s | ~20s | **16×** |
| Per-block HNSW (coarse) | ~0.4ms | ~15,000 rec/s | ~7s | **47×** |
| Per-block HNSW (tight) | ~0.2ms | ~35,000 rec/s | ~3s | **114×** |

**Live mode (sequential, single client):**

| Approach | Per-request cost | Throughput | vs current |
|----------|-----------------|------------|------------|
| Current (wratio candidates) | ~3.9ms (10K), ~7-10ms (100K) | 255→~100-150 req/s | degrades |
| Blockless HNSW | ~4.5ms (any scale) | ~220 req/s | stable |
| Per-block HNSW | ~3.3-3.5ms (any scale) | ~285-300 req/s | **stable, fastest** |

**Live mode (concurrent, c=10):**

| Approach | Throughput | vs current |
|----------|------------|------------|
| Current (wratio candidates) | 760→~400-500 req/s | degrades |
| Blockless HNSW | ~650-700 req/s | stable |
| Per-block HNSW | ~800-900 req/s | **stable, fastest** |

The pattern is clear:

- **Batch mode**: per-block HNSW is dramatically faster because each
  search traverses a tiny graph. Tight blocking with 200-record blocks
  gives 114× improvement over current. Even coarse blocking gives 47×.
  Blockless gives 16× — still transformative, but leaving performance
  on the table.

- **Live mode**: the differences between approaches are smaller because
  ONNX encoding dominates (~3ms out of ~3.3-4.5ms total). Per-block is
  marginally faster than blockless (~300 vs ~220 req/s sequential), but
  both are stable across dataset sizes while the current design degrades.
  In live mode, the real win is **scale-insensitivity**, not raw speed.

The interesting question: is the 7× batch throughput difference between
blockless and tight-blocked worth the complexity of block infrastructure?
For a daily 100K × 100K job, 3 seconds vs 20 seconds is irrelevant —
both are effectively instant. At 1M × 1M, it becomes ~50 seconds vs
~6 minutes — still both acceptable. The answer depends on whether you're
running one job a day or thousands.

### The trade-off

The blockless design is ~4-7× slower in batch mode than the per-block
design because each HNSW search traverses a larger graph. But it's
still 16× faster than the current pipeline, and the simplicity gain is
substantial: no blocking config, no block infrastructure, no per-block
index lifecycle, fewer moving parts to build and maintain.

For users who need the absolute fastest batch throughput and are willing
to configure blocking, the per-block design remains available as an
optimisation. But blockless all-embedding scoring may be the right
default — fast enough for any realistic dataset, with zero tuning
required.

---

## Appendix: encoding performance findings

### CPU vs CoreML/GPU on Apple Silicon (M3)

Tested `all-MiniLM-L6-v2` via ONNX Runtime with CPU vs CoreML
execution providers. Release build, warm model, 500 iterations
single-text and 50 iterations batch-of-100.

| Mode | CPU | CoreML (ANE/GPU) | Speedup |
|------|-----|------------------|---------|
| Single text | **1.45ms** (689/s) | 7.23ms (138/s) | CPU is **5× faster** |
| Batch of 100 | **60.7ms** (1,647/s) | 130.8ms (764/s) | CPU is **2× faster** |

**CoreML is slower for this model.** MiniLM-L6-v2 is small (22M
params, 6 transformer layers). The overhead of dispatching to the
neural engine — serialising tensors, crossing the CPU/ANE boundary,
copying results back — exceeds the actual compute savings. CPU ONNX
Runtime with NEON SIMD on Apple Silicon handles the matrix multiplies
faster than the round-trip to the ANE.

CoreML/GPU acceleration matters for large models (LLMs, Stable
Diffusion, Whisper) where compute is heavy enough to amortise the
dispatch cost. For a 22M-param embedding model, CPU is optimal.

### Revised performance projections

The 1.45ms single-text encode (release build) is faster than the ~3ms
used throughout this document's projections (which came from debug
builds and stress test measurements with contention overhead). This
shifts the balance:

**Live mode at 100K × 100K with 1.45ms encode:**

| Approach | Total latency | Throughput (seq) |
|----------|--------------|------------------|
| Current (wratio, 10K blocks) | ~1.45 + ~5-15ms = ~7-17ms | ~60-140 req/s |
| Blockless HNSW | ~1.45 + 1.44ms = ~2.9ms | ~345 req/s |
| Per-block HNSW (tight) | ~1.45 + 0.20ms = ~1.65ms | ~600 req/s |

With encoding at 1.45ms instead of 3ms, the scoring pipeline's
architecture becomes more visible in the total latency. The per-block
approach achieves sub-2ms total latency — which is remarkable for a
full record-matching pipeline including neural network inference.

### Single-artifact deployment

The `fastembed` crate supports `UserDefinedEmbeddingModel` which
accepts raw bytes for the ONNX model and tokeniser files. Using
Rust's `include_bytes!()` macro, the model weights (~87MB ONNX +
~700KB tokeniser files) can be embedded directly in the binary at
compile time.

This would produce a ~120MB self-contained binary requiring zero
external dependencies at runtime: no Python, no model download, no
internet access, no cache directory. Deploy with `scp meld server:`
and run immediately.

The four required files:

| File | Size | Purpose |
|------|------|---------|
| `model.onnx` | 87MB | ONNX model weights |
| `tokenizer.json` | 455KB | Tokeniser vocabulary |
| `config.json` | 600B | Model architecture |
| `special_tokens_map.json` | 350B | Special token definitions |
| `tokenizer_config.json` | 350B | Tokeniser settings |

Implementation: switch `EncoderPool` from `TextEmbedding::try_new()`
(downloads from HuggingFace) to `TextEmbedding::try_new_from_user_defined()`
(loads from embedded bytes). The change is confined to encoder pool
initialisation — the rest of the codebase is unaffected
