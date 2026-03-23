---
name: Melder project overview
description: Rust record-matching/entity-resolution engine — batch and live modes, configurable scoring pipeline
type: project
---

Melder is a high-performance record matching engine in Rust (binary: `meld`, crate: `melder`). Matches records between datasets A and B using exact, fuzzy, BM25, and embedding similarity scoring.

**Key architectural facts:**
- Single scoring pipeline (`score_pool()` in `src/matching/pipeline.rs`) — used by all modes
- CrossMap enforces 1:1 bijection under single RwLock
- Combined vector index uses sqrt(w) scaling for weighted cosine identity
- RecordStore trait with MemoryStore (DashMap) and SqliteStore backends
- Feature flags: `usearch` (HNSW ANN), `bm25` (Tantivy), `parquet-format`
- Batch mode: Rayon-parallelised; Live mode: Axum HTTP server with WAL

**Why:** Entity resolution for financial data — matching vendor records to internal reference masters (counterparties, instruments, issuers).

**How to apply:** Follow AGENTS.md bootstrap protocol. Read `vault/project_overview.md` at session start. Check `vault/decisions/Key Decisions.md` and `vault/ideas/Discarded Ideas.md` before proposing architectural changes.
