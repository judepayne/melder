# Maybe

Ideas that remain genuinely undone, filtered from FUTURE.md, notes.md, and TODO.md.
The usearch HNSW backend, batch endpoints, embedding-based candidates,
multi-field blocking, skip-re-encode, CLI extraction, and WAL-based live output
are all already shipped. Items are ranked by assessed usefulness.

---

**1. Incremental encoding (text-hash dedup).** *(Encoding)*
Currently a stale cache triggers a full re-encode of all records. For recurring
batch jobs where ~95% of records are unchanged day-over-day, this is wasteful.
Keying each cached vector by a hash of the input text would let a re-run skip
all unchanged records and only encode the diff. Also fixes a correctness gap:
records deleted and replaced with the same count pass the size-based staleness
check today, producing wrong cached vectors silently.

**2. Config hash in manifest.** *(Operational)*
Blocking config changes between runs (e.g. adding a blocking field) silently
invalidate the index without any warning. Storing a hash of the blocking config
in the cache manifest and comparing on load would catch this and trigger a
rebuild or at least a loud warning. Tiny effort, prevents real wrong-results bugs.

**3. Quantised ONNX models.** *(Encoding)* ✓ DONE
`performance.quantized: true` in config selects the INT8 quantised ONNX variant
for any model that has one (AllMiniLML6V2Q, AllMiniLML12V2Q). Encode throughput
~2x faster; ~2% of borderline pairs shift classification bucket. Models without
a quantised variant error clearly. Default: false.

**4. Live mode review queue.** *(Live mode)*
`crossmap/confirm` already lets callers confirm any of the top-N matches, but
there is no explicit review workflow for borderline scores. Two gaps: the `/add`
response doesn't signal when a match landed in the review band — a
`"classification": "review"` field would let callers handle this without
re-interpreting raw scores against thresholds. Beyond that, borderline matches
could be held in a server-side queue accessible via a `/review` endpoint family
(list, accept, reject), mirroring the batch-mode CLI review commands.

**5. Parallel candidate scoring (live mode).** *(Candidates)* ✓ DONE
Both flat-scan paths in `select_candidates` now use `par_iter` (rayon). The
no-embeddings path and the dot-product scoring path are both parallelised.
At 100K blocked candidates this drops from ~200ms to ~25ms on 8 cores.

**6. GC / TTL for stale live-mode blocks.** *(Live mode)*
The blocking index accumulates entries for records that are added then removed in
live mode. No eviction mechanism exists. A `meld gc` command (or background task)
that evicts blocks older than `gc_ttl_days` would prevent unbounded growth over
long server uptimes.

**7. Single-artifact deployment.** *(Operational)*
Embedding model weights are downloaded and cached at runtime. `include_bytes!()`
on the ONNX weights at build time would produce a fully self-contained binary —
no network access, no cache directory, no first-run latency. Increases binary
size significantly but simplifies deployment in air-gapped or containerised
environments.

**8. Pipeline hooks.** *(Pipeline)*
User-defined logic at specific pipeline points. The interesting hook sites are:
pre-score (after blocking, before scoring), post-score (after composite score,
before classification), and on-confirm (when a cross-map entry is created).
Form options range in complexity: HTTP callout to a user-provided endpoint
(simplest, async-friendly), external subprocess with JSON on stdin/stdout
(portable, no dependency), or WASM plugin (sandboxed, in-process). A
callout-style approach covers most real use cases (audit logging, downstream
notifications, LEI enrichment) without the complexity of an in-process plugin
system. Design-heavy but high real-world value for production deployments.

**9. Request coalescing.** *(Encoding)*
Requests arriving within a short window (~1ms) are batched into a single ONNX
call before fan-out. ONNX Runtime throughput improves substantially at batch
sizes of 8–32; the SIMD paths get properly amortised. Architecture:
HTTP → queue → coordinator → batch encode → fan-out. The batch endpoints already
demonstrate the principle; coalescing automates it for single-record traffic.
High ceiling gain but architecturally non-trivial.

**10. `workers` field — wire up or remove.** *(Operational)* ✓ DONE
Removed entirely: from all YAML configs, both schema fields (`Config.workers`
and `PerformanceConfig.workers`), the loader defaulting/sync/validation logic,
and the validate.rs print. Rayon thread count is controlled via the
`RAYON_NUM_THREADS` env var if needed.

**11. API handler boilerplate.** *(Code quality)* ✓ DONE
Extracted 7 private helpers (`upsert_handler`, `match_handler`, `remove_handler`,
`query_handler`, `add_batch_handler`, `match_batch_handler`, `remove_batch_handler`),
each parameterised by `Side`. The 14 public A/B handlers are now one-liners.
439 → 290 lines.

**12. Quantised vector storage.** *(Vector index)*
Store vectors as `[u8; 384]` instead of `[f32; 384]`. 4x less memory, 4x faster
scan with integer dot products. At 1M records this takes vectors from 1.5GB to
~375MB. Simple scalar quantisation (map f32 → uint8 in [0, 255]) captures most
of the benefit in ~50 lines. Only starts to matter above 500K records.

**13. Memory-mapped index.** *(Vector index)*
`mmap` the `.index` cache file via `memmap2` instead of loading into a heap
`Vec<f32>`. Eliminates the startup memory spike; the OS pages in only what's
touched. `VecIndex::from_parts` would take a `&[f32]` backed by the mmap region
rather than an owned vec. Mainly useful at large scale.

**14. SIMD-explicit dot product.** *(Vector index)*
The flat scan auto-vectorises today, but explicit NEON intrinsics (`vfmaq_f32`)
or the `simsimd` crate would give ~2x throughput on the inner loop. At 100K
records this halves search time from ~1ms to ~0.5ms. Marginal in practice since
the flat scan only appears in full-scoring (10 candidates) with default config.

**15. LSH blocking.** *(Blocking)*
Hash-based blocking on the embedding vector instead of exact field equality. Two
records with similar embeddings hash to the same bucket. No need to know which
fields to block on; tuning the hash parameters (tables, bits) controls recall vs.
selectivity. Useful when exact-field blocks are too coarse, but complex to
implement and tune correctly.

**16. Lock-free / RCU vector index.** *(Concurrency)*
At high concurrency (c=100+), the `RwLock<VecIndex>` write path serialises all
adds. Two options: (A) append-only lock-free structure with atomic length —
searches read up to the current length, updates overwrite in place; (B)
read-copy-update — writes go to a staging area, merged and atomically swapped in
every ~10ms. The current architecture handles c=10–20 fine; this only matters if
concurrency targets go significantly higher.

**17. External vector database.** *(Vector index)*
Qdrant / Milvus / Weaviate over gRPC. The `VectorDB` trait surface is already
the right abstraction — a client implementation would be a drop-in replacement.
Buys: no staleness problem, durable persistence, multiple melder instances sharing
one index. Costs: network round-trip (~1–5ms) per search, external service
dependency. Only worth it at 1M+ records or when horizontal scaling is needed.

**18. Windows compatibility — tidy up `--socket` flag.** *(Operational)* ✓ DONE
The `--socket` field in the `Serve` subcommand and its match arm are now gated
with `#[cfg(unix)]`. On Windows the flag no longer appears in `--help` and is
not parsed. Build requirement on Windows: MSVC C++ toolchain (needed by
`usearch` and ONNX Runtime).

---

*10M+ regime: stateless scoring workers + stateful coordination layer (crossmap,
WAL, shard routing). Batch mode becomes a Spark-style shuffle job. Not worth
designing until the need is real. See `splink` for prior art.*
