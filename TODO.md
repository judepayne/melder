# Maybe

Ideas that remain genuinely undone, filtered from FUTURE.md, notes.md, and TODO.md.
The usearch HNSW backend, batch endpoints, embedding-based candidates,
multi-field blocking, skip-re-encode, CLI extraction, and WAL-based live output
are all already shipped. Items are ranked by assessed usefulness.

---

**1. GC / TTL for stale live-mode blocks.** *(Live mode)*
The blocking index accumulates entries for records that are added then removed in
live mode. No eviction mechanism exists. A `meld gc` command (or background task)
that evicts blocks older than `gc_ttl_days` would prevent unbounded growth over
long server uptimes.

**2. Single-artifact deployment.** *(Operational)*
Embedding model weights are downloaded and cached at runtime. `include_bytes!()`
on the ONNX weights at build time would produce a fully self-contained binary —
no network access, no cache directory, no first-run latency. Increases binary
size significantly but simplifies deployment in air-gapped or containerised
environments.

**3. Pipeline hooks.** *(Pipeline)*
User-defined logic at specific pipeline points. The interesting hook sites are:
pre-score (after blocking, before scoring), post-score (after composite score,
before classification), and on-confirm (when a cross-map entry is created).
Form options range in complexity: HTTP callout to a user-provided endpoint
(simplest, async-friendly), external subprocess with JSON on stdin/stdout
(portable, no dependency), or WASM plugin (sandboxed, in-process). A
callout-style approach covers most real use cases (audit logging, downstream
notifications, LEI enrichment) without the complexity of an in-process plugin
system. Design-heavy but high real-world value for production deployments.

**4. Memory-mapped index (flat backend).** *(Vector index)*
`mmap` the `.index` cache file via `memmap2` instead of loading into a heap
`Vec<f32>`. Eliminates the startup memory spike; the OS pages in only what's
touched. `VecIndex::from_parts` would take a `&[f32]` backed by the mmap region
rather than an owned vec. Mainly useful at large scale. Applies to the flat
backend only — usearch manages its own memory mapping internally.

**5. SIMD-explicit dot product.** *(Vector index) — maybe*
The flat scan auto-vectorises today, but explicit NEON intrinsics (`vfmaq_f32`)
or the `simsimd` crate would give ~2x throughput on the inner loop. At 100K
records this halves search time from ~1ms to ~0.5ms. Marginal in practice since
the flat scan only appears in full-scoring (10 candidates) with default config.
Only applies to the flat backend.

**6. LSH blocking.** *(Blocking) — maybe*
Hash-based blocking on the embedding vector instead of exact field equality. Two
records with similar embeddings hash to the same bucket. No need to know which
fields to block on; tuning the hash parameters (tables, bits) controls recall vs.
selectivity. Useful when exact-field blocks are too coarse, but complex to
implement and tune correctly. Caveat: hash instability across runs means the same
record may not land in the same block on subsequent runs, which complicates
reproducibility and incremental matching.

**7. External vector database.** *(Vector index) — maybe*
Qdrant / Milvus / Weaviate over gRPC. The `VectorDB` trait surface is already
the right abstraction — a client implementation would be a drop-in replacement.
Buys: no staleness problem, durable persistence, multiple melder instances sharing
one index. Costs: network round-trip (~1–5ms) per search, external service
dependency. Only worth it at 1M+ records or when horizontal scaling is needed.

---

*10M+ regime: stateless scoring workers + stateful coordination layer (crossmap,
WAL, shard routing). Batch mode becomes a Spark-style shuffle job. Not worth
designing until the need is real. See `splink` for prior art.*
