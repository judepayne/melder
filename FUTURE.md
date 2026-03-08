# Future

Ideas and architectural directions for melder.

---

## 1. Vector database

The current vector index is a flat brute-force scan stored in a custom
binary format. It works well up to ~100K records but scales linearly —
every search computes a dot product against every vector in the index.

Migration to a vector database would happen in two stages:

### Stage 1: In-process ANN index

Replace the flat `VecIndex` with an in-process approximate nearest
neighbour (ANN) index using HNSW (hierarchical navigable small world
graphs). Rust crates such as `hnsw_rs` or `instant-distance` provide
this.

**What changes:**

- Search complexity drops from O(N) to O(log N) per query.
- Insert complexity increases from O(1) to O(log N) — the HNSW graph
  must be updated on each insertion.
- Memory usage increases ~1.5x due to graph edge storage.
- The binary cache format would change to the library's native format.
- Remove support would require soft-delete with periodic rebuild, since
  HNSW graphs do not support true node deletion.

**What stays the same:**

- Single-binary deployment — no external services.
- Same API, same scoring pipeline, same config.
- The HNSW index is a drop-in replacement for `VecIndex` behind the
  same `search()` / `upsert()` interface.

**When it matters:**

At 10K records a flat scan takes ~0.1ms per query — HNSW would not
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
- Persistence is managed by the DB — no more `.index` cache files,
  no staleness problem.
- Blocking/filtering could be pushed into the DB's native metadata
  filter syntax, potentially replacing the in-process BlockingIndex.
- Multiple melder instances could share the same vector index for
  horizontal scaling.

**What stays the same:**

- Scoring pipeline, crossmap, WAL, API — all unchanged.
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
