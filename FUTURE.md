# Future

Ideas and architectural directions for melder.

---

## 1. Vector database

The current vector index is a flat brute-force scan stored in a custom
binary format. It works well up to ~100K records but scales linearly --
every search computes a dot product against every vector in the index.

Migration to a vector database would happen in two stages:

### Stage 1: In-process ANN index

Replace the flat `VecIndex` with an in-process approximate nearest
neighbour (ANN) index using HNSW (hierarchical navigable small world
graphs). Rust crates such as `hnsw_rs` or `instant-distance` provide
this.

**What changes:**

- Search complexity drops from O(N) to O(log N) per query.
- Insert complexity increases from O(1) to O(log N) -- the HNSW graph
  must be updated on each insertion.
- Memory usage increases ~1.5x due to graph edge storage.
- The binary cache format would change to the library's native format.
- Remove support would require soft-delete with periodic rebuild, since
  HNSW graphs do not support true node deletion.

**What stays the same:**

- Single-binary deployment -- no external services.
- Same API, same scoring pipeline, same config.
- The HNSW index is a drop-in replacement for `VecIndex` behind the
  same `search()` / `upsert()` interface.

**When it matters:**

At 10K records a flat scan takes ~0.1ms per query -- HNSW would not
produce a measurable improvement. At 100K+ records the flat scan
exceeds 1ms and at 1M it reaches ~10ms. HNSW would keep search
under 0.1ms at any scale.

**Estimated effort:** Medium. The `VecIndex` trait surface is small
(search, search_filtered, upsert, remove, get, contains, len). An
HNSW adapter implementing this interface would be ~200 lines. The
main complexity is cache serialisation and the soft-delete/rebuild
strategy for removals.

### Stage 2: External vector database

Replace the in-process index with a client that talks to a running
vector database (Qdrant, Milvus, Weaviate, Pinecone) over gRPC or
HTTP.

**What changes:**

- Deployment becomes binary + vector DB service (or managed cloud).
- Each vector search incurs a network round-trip (~1-5ms latency).
- Persistence is managed by the DB -- no more `.index` cache files,
  no staleness problem.
- Blocking/filtering could be pushed into the DB's native metadata
  filter syntax, potentially replacing the in-process BlockingIndex.
- Multiple melder instances could share the same vector index for
  horizontal scaling.

**What stays the same:**

- Scoring pipeline, crossmap, WAL, API -- all unchanged.
- The vector DB is only used for the "find similar vectors" step;
  everything else remains in-process.

**When it matters:**

At 1M+ records, or when multiple melder instances need to share
state, or when durable vector persistence matters more than
single-binary simplicity.

**Estimated effort:** Large. Requires a vector DB client abstraction,
async network calls in the scoring path, configuration for DB
connection, and a migration path from local indices. The blocking
filter integration varies by DB vendor.

---

## 2. Performance analysis and scaling roadmap

This section analyses where bottlenecks appear as dataset sizes grow
and what can be done to lift performance at each layer. The analysis
is grounded in the current architecture and benchmark results.

### Current performance budget

At 10K x 10K with warm caches, the time budget for a single live-mode
add is roughly:

| Phase | Time | Notes |
|-------|------|-------|
| ONNX encode | ~2-3ms | Single text, `all-MiniLM-L6-v2` |
| Blocking lookup | <0.01ms | HashMap lookup |
| Candidate selection (wratio, 10 candidates from ~1K blocked) | ~0.3ms | String comparisons |
| Full scoring (4 fields x 10 candidates) | ~0.3ms | 1 embedding lookup + 3 cheap scorers |
| Vector index upsert + WAL write | ~0.1ms | Append + memcpy |
| **Total** | **~3.9ms** | **~255 req/s sequential** |

Encoding is already ~65% of wall time. Everything else is noise. The
interesting question is where else ceilings appear as N grows, and what
can be done about each.

---

### Layer 1: ONNX encoding -- the dominant cost

#### Where we are

`fastembed` wraps ONNX Runtime. Each `encode()` call runs tokenisation,
attention, and pooling on a single ONNX session. The encoder pool
(`Vec<Mutex<TextEmbedding>>`) provides N independent sessions, but each
session is single-threaded. At `pool_size: 4` you get up to 4
concurrent encodes, but each one still takes ~2-3ms for a short entity
name.

For batch mode with warm caches, encoding is **completely eliminated**
-- vectors are loaded from `.index` files. So batch mode at 100K x 100K
is already CPU-saturated on scoring, not encoding. The encoding problem
is really a **live mode** and **cold-start** problem.

#### Scaling pressure

- Live mode today: ~250 req/s sequential, ~760 req/s at c=10 (limited
  by 4 encoder slots)
- The batch endpoints help (1.8x at size=50) because they amortise the
  per-call overhead in ONNX Runtime (session setup, memory allocation),
  but the actual matrix multiply time is irreducible
- At batch_size=50, encoding 50 short texts takes ~15-20ms -- which
  means the ONNX model can process roughly 2,500-3,000 texts/sec per
  session

#### How to lift it

**A. GPU inference.** ONNX Runtime supports CoreML (Apple Silicon),
CUDA, and TensorRT execution providers. `fastembed` supports
`ExecutionProviderDispatch` -- you can pass `CUDAExecutionProvider` at
init time. On an A100, MiniLM-L6 can encode ~50,000 texts/sec -- a
15-20x improvement over CPU. On Apple Silicon with CoreML/ANE, the gain
would be more modest (3-5x) but still meaningful. The key architectural
change: you'd want a **single GPU session** rather than a pool of N CPU
sessions, because GPU sessions parallelise internally. The encoder pool
abstraction already supports this -- you'd just init one slot with a GPU
provider and remove the pool_size config.

**B. Smaller/distilled models.** MiniLM-L6-v2 is already small (22M
params), but there are even faster options: `gte-small` (33M params but
faster ONNX graph), `e5-small-v2`, or quantised models. ONNX Runtime
supports INT8 quantisation -- a quantised MiniLM-L6 runs ~2x faster on
CPU with negligible quality loss for entity matching. The `fastembed`
crate supports quantised ONNX graphs natively -- you just point it at a
different model directory.

**C. Speculative pre-encoding.** In live mode, we could encode the text
*asynchronously* when the HTTP request arrives (before acquiring the
session lock for scoring). The scoring pipeline already accepts
`_with_vec` variants. If you had a small ring buffer of "recently
encoded texts", repeat-adds (which are ~20% of the stress test
workload) could skip encoding entirely. This is effectively a **vector
LRU cache** keyed on the embedding text. The session already detects
when the embedding text hasn't changed and skips re-encoding --
extending this to a shared cache would help with concurrent duplicate
adds.

**D. Batched inference with request coalescing.** This is the big one
for live mode. Instead of each HTTP request encoding independently, a
coordinator could collect requests that arrive within a short window
(e.g. 1-2ms) and submit them as a single ONNX batch. ONNX Runtime's
throughput per text improves significantly with batch sizes of 8-32 --
the GPU-like SIMD paths in the CPU EP amortise instruction dispatch. The
architecture would be:

```
HTTP request --> queue --> coordinator (drains every 1ms)
                              |
                 ONNX batch encode (N texts)
                              |
               fan-out results to individual scoring tasks
```

This is exactly what the Go version's `BatchUpsertAndQuery` was trying
to do but couldn't because of the Python GIL. In Rust, it would work
properly. The batch endpoints already demonstrate the principle --
coalescing just automates it for single-record requests.

---

### Layer 2: Vector search -- currently invisible, will become visible

#### Where we are

The flat `VecIndex` does brute-force dot product: O(N x D) per query.
At 10K records x 384 dims = 3.84M multiplies, which takes ~0.1ms. LLVM
auto-vectorises the inner loop to ARM NEON. This is invisible in the
profile.

#### Scaling pressure

| N | Dot products per query | Estimated time |
|---|------------------------|----------------|
| 10K | 3.8M | ~0.1ms |
| 100K | 38.4M | ~1ms |
| 1M | 384M | ~10ms |
| 10M | 3.84B | ~100ms |

At 100K you start to feel it. At 1M it's a real problem -- especially
because blocking may not reduce the candidate pool enough. If 10% of
records share a country code, blocking at 1M still leaves 100K
candidates for the flat scan.

But note: the flat scan only appears in the **candidates stage** when
using `method: embedding` for candidate selection (or when candidates
are disabled entirely). With `method: fuzzy` for candidates -- which is
our default config -- the vector search only happens during **full
scoring**, and by then we've already narrowed to 10 candidates. So the
cosine similarity lookup is just 10 x O(D) = 3,840 multiplies.
Negligible.

The real question is: what role does the VecIndex play at scale?

1. **Batch cold-start encoding**: O(N) encodes, stored in VecIndex.
   Already parallelised across encoder pool slots. Cached to disk. Not
   a search problem.
2. **Live `search_filtered`**: Only used if candidates are
   embedding-based. With fuzzy candidates (default), this path isn't
   hit during scoring.
3. **Live `upsert`**: O(1) -- append or overwrite. Fine at any scale.

So with the current default config, VecIndex search doesn't actually
bottleneck. The HNSW plan (section 1 above) is insurance against
configs that use `candidates.method: embedding` -- which is a valid and
sometimes better choice.

#### How to lift it (when it matters)

**A. HNSW in-process.** As described in section 1. Drop-in replacement,
O(log N) search. The `hnsw_rs` crate is solid. Main complication:
removals require a tombstone + periodic rebuild (HNSW doesn't support
node deletion). The rebuild could run on a background thread -- snapshot
the current state, build a new graph, atomically swap.

**B. SIMD-explicit dot product.** The current `dot_product_f32` loop
does auto-vectorise, but you could get ~2x more throughput with explicit
NEON intrinsics (processing 4 or 8 floats per cycle with `vfmaq_f32`).
The `simsimd` crate does this. At 100K records this would halve the
search time from ~1ms to ~0.5ms. Small gain but free.

**C. Quantised vectors.** Store vectors as `u8` (product quantisation
or scalar quantisation) instead of `f32`. 4x less memory, 4x faster
scan (integer dot products with NEON `vdotq_u32`). Quality loss is
negligible for top-10 candidate retrieval. This is what FAISS
`IndexIVFPQ` does -- you could implement a simple scalar quantisation
(map each f32 to uint8 in [0, 255]) and get most of the benefit with
~50 lines of code.

---

### Layer 3: Blocking -- O(1) but cardinality-sensitive

#### Where we are

The `BlockingIndex` is a `HashMap<Vec<String>, HashSet<String>>` --
composite key to set of IDs. Lookup is O(1). Insert/remove is O(1).
This is already optimal in terms of algorithmic complexity.

#### Scaling pressure

The problem isn't lookup speed -- it's **block size**. If you block on
`country_code` and 30% of records are in the US, then at 1M records you
have 300K candidates in the "US" block. All 300K go into candidate
selection. With fuzzy candidate selection that's 300K wratio comparisons
-- ~3s per query. Unacceptable.

At 10K this is fine (3K US records x wratio = ~3ms). At 100K it's
marginal (~30ms). At 1M it breaks.

#### How to lift it

**A. Multi-field blocking with AND.** Already supported. Blocking on
`country_code AND sector` splits the "US" block into "US/finance",
"US/tech", etc. Each sub-block is much smaller. The user controls this
in config -- add more blocking fields to get tighter blocks.

**B. Locality-sensitive hashing (LSH) blocking.** Instead of exact
field equality, use hash-based blocking on the embedding vector itself.
Two records with similar embeddings are likely to hash to the same
bucket. This replaces the current exact-field blocking with an
approximate but more flexible approach. Libraries like `lsh-rs` provide
this. The advantage: you don't need to know which fields to block on.
The disadvantage: tuning the hash parameters (number of hash tables,
number of bits) affects recall/precision.

**C. Sorted neighbourhood blocking.** Sort records by a blocking key
(e.g. first 3 characters of the name), then only compare records within
a sliding window. This is simpler than LSH and well-understood in the
record linkage literature. Works well when there's a natural sort order.

**D. Canopy clustering.** Two-pass approach: first pass uses a cheap
distance (e.g. TF-IDF on tokenised names) to create overlapping
clusters (canopies). Second pass does exact scoring within each canopy.
This is the approach used by Apache Mahout's entity resolution.

The key insight: at 1M+ records, blocking is no longer just an
optimisation -- it's a **correctness requirement** because without it,
the pipeline simply can't finish in time. The user needs to think
carefully about block design, and melder should warn (or suggest) when
block sizes exceed a threshold.

---

### Layer 4: Candidate selection -- the hidden bottleneck at scale

#### Where we are

Fuzzy candidate selection runs `wratio` on `short_name` vs
`counterparty_name` for every blocked candidate. `wratio` itself calls
`ratio`, `partial_ratio`, and `token_sort_ratio` -- roughly 3
edit-distance computations per pair. For typical entity names (10-50
chars), each `wratio` takes ~1-3us.

#### Scaling pressure

| Block size | wratio calls | Time |
|------------|-------------|------|
| 1K | 1K | ~2ms |
| 10K | 10K | ~20ms |
| 100K | 100K | ~200ms |
| 1M | 1M | ~2s |

At 100K block size this dominates the pipeline. Larger than encoding.

#### How to lift it

**A. Switch to embedding candidates at scale.** If you have vectors
cached, `candidates.method: embedding` does a single vector search
instead of N string comparisons. At 10K blocked records, a flat scan
(10K dot products) takes ~0.1ms vs ~20ms for wratio. 200x faster. The
trade-off: embedding candidates may miss cases where the
character-level similarity is high but semantic similarity is low (e.g.
"AAA Corp" vs "AAB Corp" -- nearly identical strings but embeddings
might diverge). For entity matching this is rarely an issue.

**B. Pre-sorted candidate lists.** If the blocking index stored
candidates sorted by a pre-computed key (e.g. first 4 characters of the
candidate field), you could skip wratio entirely for candidates that
don't share a prefix. This is a cheap pre-filter before the expensive
scoring.

**C. Parallelise candidate scoring.** The `select_candidates` function
currently scores sequentially. Since each candidate score is
independent, this could trivially use Rayon `par_iter` within the
function. At 100K candidates, using 8 cores would drop the time from
200ms to ~25ms. The code change is minimal -- just `.par_iter()` instead
of `.iter()` on the `candidate_ids` slice.

---

### Layer 5: Memory -- the physical constraint

#### Where we are

Each record is a `HashMap<String, String>` in a `DashMap`. Each vector
is 384 x 4 bytes = 1.5KB. At 10K records, total memory for one side is
roughly:

- Records: ~10K x 500 bytes (average) = 5MB
- Vectors: ~10K x 1.5KB = 15MB
- Blocking index: ~1MB
- Total per side: ~21MB
- Both sides + overhead: ~50MB
- Encoder pool (4 sessions): ~400MB
- **Total: ~450MB**

#### Scaling pressure

| N per side | Records | Vectors | Encoder | Total |
|------------|---------|---------|---------|-------|
| 10K | 5MB | 15MB | 400MB | ~420MB |
| 100K | 50MB | 150MB | 400MB | ~600MB |
| 1M | 500MB | 1.5GB | 400MB | ~2.4GB |
| 10M | 5GB | 15GB | 400MB | ~20GB |

At 1M per side you need 2.4GB -- fine on a modern server. At 10M you
need 20GB -- still feasible but getting tight. The vector storage
dominates.

#### How to lift it

**A. Quantised vector storage.** Store vectors as `[u8; 384]` instead
of `[f32; 384]`. 4x less memory. At 10M records: vectors drop from
15GB to 3.75GB.

**B. Memory-mapped vector index.** Instead of loading all vectors into
a `Vec<f32>` in the heap, `mmap` the cache file directly. The OS pages
in only what's needed. This eliminates the startup memory spike and lets
the OS manage the working set. The `memmap2` crate makes this
straightforward. The `VecIndex::from_parts` constructor would change to
take a `&[f32]` slice backed by an mmap region rather than an owned
`Vec<f32>`.

**C. Record storage compression.** `HashMap<String, String>` is
memory-inefficient -- each entry has 3 allocations (key String, value
String, HashMap node). A column-oriented approach (store each field as a
`Vec<String>` with ID-to-index mapping) would halve memory usage. Or
use a pre-interned string table for field names.

**D. Sharding.** Split the dataset by blocking key. Each shard holds
its own records + vectors + blocking index. Different shards can live on
different machines or just in different memory regions. Search fans out
to the relevant shard(s). This is the path to 100M+ records.

---

### Layer 6: Live mode concurrency -- the architectural ceiling

#### Where we are

Live mode runs on `tokio` with `spawn_blocking` for each request. The
scoring pipeline uses `DashMap` for lock-free reads and
`RwLock<VecIndex>` for vector operations. The crossmap uses
`RwLock<CrossMap>`. Concurrent writes serialise on the RwLock write
path.

#### Scaling pressure

At high concurrency (c=100+), the write lock on `VecIndex` becomes a
bottleneck. Every add/update needs to:

1. Acquire write lock on the vector index (to upsert the new vector)
2. Acquire write lock on the blocking index (to update the entry)
3. Acquire write lock on the crossmap (if auto-matched)

While one request holds the VecIndex write lock, all other requests --
even reads -- are blocked (RwLock's write lock is exclusive).

#### How to lift it

**A. Lock-free vector index.** Use a concurrent append-only structure
for vectors. New vectors are appended to a `Vec<AtomicU8>` or similar
lock-free structure. Searches read the current length and scan up to
that point. Updates overwrite in-place (safe for f32 on aligned
addresses with `AtomicU32` reinterpretation). This eliminates the write
lock entirely.

**B. Read-copy-update (RCU) for the index.** Keep two copies of the
index. Reads always go to the "current" copy (lock-free). Writes go to
a staging area. Periodically (every 10ms), the staging area is merged
into a new copy and swapped in atomically. This is how Linux kernel data
structures handle concurrent reads with rare writes.

**C. Per-shard locks.** If the index is sharded by blocking key, each
shard has its own lock. Writes to different shards don't contend. At 10
shards, contention drops by 10x.

---

### The 10M+ regime

At 10M records per side, the architecture needs fundamental changes:

1. **Encoding is no longer per-request.** You can't encode 10M records
   at startup in reasonable time (10M x 3ms = 8+ hours). You need
   persistent, incremental encoding -- encode once, store forever,
   update incrementally. The `.index` cache already does this for batch
   mode. For live mode, the WAL replay already handles incremental
   updates. The main gap is that the cache staleness check is
   size-based -- it needs to be content-hash-based.

2. **Vector search must be sublinear.** HNSW is non-negotiable at this
   scale. Or an external vector database (Qdrant scales to 100M+
   vectors, handles persistence, replication, and filtering natively).

3. **Blocking becomes a distributed problem.** You need to route each
   query to the right shard(s) before any work happens. This is
   essentially a distributed hash table.

4. **The single-binary model breaks.** At 10M x 20GB memory, you need
   a cluster. The natural split: stateless scoring workers (they receive
   a record + pre-encoded vector, pull candidates from a shared vector
   DB, score locally, return results) + a stateful coordination layer
   (crossmap, WAL, shard routing). The scoring workers can be replicated
   horizontally. The coordination layer needs consensus or a database.

5. **Batch mode becomes a Spark/Dask job.** At 10M x 10M = 100 trillion
   potential pairs, even with blocking you might have billions of
   candidate pairs. This is the scale where you need distributed compute
   with shuffle -- map each record to its blocking key, shuffle to
   co-locate all records with the same key, score within each partition.
   This is exactly what `splink` (the Python record linkage library)
   does on Spark.

---

### Is encoding the ultimate ceiling?

**Yes and no.**

- **For live mode at any scale**: yes. Encoding is 65% of wall time
  today and will remain dominant because everything downstream (search,
  scoring) can be made sublinear or parallel, but the neural network
  forward pass is irreducibly O(sequence_length x model_params). The
  only escapes are GPU inference, quantisation, or model distillation.

- **For batch mode with warm caches**: no. Encoding is eliminated by
  caching. The ceiling is the **scoring pipeline throughput** x
  **available cores**. At 100K x 100K you're already saturating CPU,
  which is the best possible outcome -- you've hit the machine's
  physical limit. To go further you need more machines or faster
  machines.

- **For cold-start (first run, no caches)**: encoding is the absolute
  bottleneck. 100K records x 3ms = 5 minutes. 1M records = 50 minutes.
  GPU inference cuts this to 3 minutes for 1M. Quantised models cut it
  further. But this is a one-time cost that's fully amortised by
  caching.

The **true ultimate ceiling** is actually the interaction between
**blocking cardinality** and **candidate selection cost**. At very large
scale, if your blocking keys don't provide enough selectivity (e.g. you
block on country and 40% of your data is "US"), the candidate selection
phase processes hundreds of thousands of pairs per query, and no amount
of encoding optimisation helps because encoding isn't the problem --
you're drowning in string comparisons. The fix is better blocking (more
discriminating keys, LSH, or embedding-based candidates instead of fuzzy
candidates).

---

### Summary: performance roadmap

| Scale | Bottleneck | Fix |
|-------|-----------|-----|
| 10K-100K | Encoding (live) | Batch endpoints, request coalescing, GPU |
| 100K-1M | Candidate selection (block size) | Better blocking keys, embedding candidates, parallel scoring |
| 1M-10M | Vector search + memory | HNSW, quantised vectors, mmap |
| 10M+ | Everything | Distributed architecture, external vector DB, Spark-style batch |

The architecture is solid through the 1M mark without structural
changes -- just config tuning (better blocking, embedding candidates)
and a few tactical improvements (HNSW, SIMD dot product, INT8
quantisation). Beyond 1M, you're looking at a different system.
