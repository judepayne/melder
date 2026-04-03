---
type: overview
module: general
status: active
tags: [overview, index, onboarding]
related_code: []
---

# Melder — Project Overview

> Read this at the start of every session. Follow `AGENTS.md` for all coding rules and conventions. The 4 principles in §2 are inviolable — violating them is a bug.

_Last updated: 2026-04-03_

---

## 1. What Melder Is

**Melder** is a Rust record-matching (entity resolution) engine. Given datasets A and B, it finds which B records correspond to which A records.

- **Binary**: `meld` | **Crate**: `melder` | **Mode**: `Match` (two-sided) or `Enroll` (single-pool)
- **Scoring**: composite of exact, fuzzy (wratio, token_sort, partial_ratio), BM25, embedding, numeric, synonym
- **Production config**: Arctic-embed-xs R22 + 50% BM25 + synonym 0.20 (zero overlap, 100% recall)

**Run modes:**
- `meld run` — batch, B queries A, output to CSV
- `meld serve` — live HTTP server, symmetric A↔B
- `meld enroll` — single-pool entity resolution

---

## 2. Inviolable Principles

These principles are non-negotiable. Violating them is a bug.

| # | Principle | What it means |
|---|-----------|---------------|
| 1 | **Batch = B queries A, Live = A↔B symmetric, Enroll = single pool** | Batch: B is query side, A is reference. Live: both sides equal. Enroll: one pool only. |
| 2 | **One Scoring Pipeline** | All score computation flows through `score_pool()` → `score_pair()`. Never create separate scoring code paths for new features — always use the existing pipeline. |
| 3 | **CrossMap Bijection (1:1)** | One A maps to one B, enforced atomically via `claim()` under single RwLock |
| 4 | **Combined Vector = Weighted Cosine** | Don't normalize scores to 0-1 and add them. Scale each field vector by `sqrt(weight)`, concatenate. Identity: `dot(combined_A, combined_B) = Σ(w_i × cos(a_i, b_i))`. Per-field cosines recovered via `decompose_emb_scores()` — no second ONNX call. |

See [[decisions/key_decisions]] for the tradeoffs behind each principle.

---

## 3. Build & Test

```bash
cargo build --release --features usearch          # standard
cargo build --release --features usearch,gpu-encode
cargo test --all-features                         # REQUIRED before commit
```

**Quality gates before commit:**
1. `cargo fmt -- --check` — fix any formatting violations
2. `cargo clippy --all-features 2>&1` — fix all clippy warnings/errors

Run these after tests pass. Tackle every issue before committing — don't submit code with lint warnings.

Feature flags: `usearch`, `parquet-format`, `simd`, `gpu-encode`, `builtin-model`. BM25 always compiled.

---

## 4. Quick Reference

| Item | Value |
|------|-------|
| Config schema | [[architecture/config_reference]] |
| API routes (live/enroll) | [[architecture/api_reference]] |
| Scoring details | [[architecture/scoring_algorithm]] |
| Current backlog | [[todo]] |
| Benchmarks & experiments | [[benchmarks_and_experiments]] |
| Build/test commands | §3 above |
| Error types | `src/error.rs` |
| Core types | `src/models.rs` |
| Enroll mode | `docs/enroll-mode.md` |

---

## 5. Vault Index

### When to read each vault document

| Document | When to read |
|----------|---------------|
| [[decisions/key_decisions]] | When touching CrossMap, pipeline, vector index, or revisiting tradeoffs |
| [[decisions/key_decisions_implementation]] | When debugging specific implementation issues (SQLite wiring, concurrency fixes, refactoring history) |
| [[architecture/module_map]] | When exploring unfamiliar code areas |
| [[architecture/config_reference]] | When adding/modifying config fields |
| [[architecture/scoring_algorithm]] | When changing how scores are computed |
| [[architecture/state_and_persistence]] | When modifying startup, WAL, or flush logic |
| [[architecture/business_logic_flow]] | When understanding batch/live pipeline flow |
| [[architecture/api_reference]] | When adding new HTTP endpoints |
| [[benchmarks_and_experiments]] | When running or interpreting benchmarks |
| [[architecture/training_loop]] | When running or modifying the fine-tuning pipeline |
| [[training/fine_tuning_guide]] | When fine-tuning embeddings |
| [[ideas/discarded_ideas]] | Before implementing a new approach |
| [[todo]] | Before starting any work — what's already done |
| [[business/use_cases]] | When explaining the product to stakeholders |

### Vault structure

```
vault/
├── project_overview.md         # THIS FILE — START HERE
├── todo.md                     # Backlog & completed
├── benchmarks_and_experiments.md # Benchmarks & experiments
├── architecture/               # Code architecture (8 files)
├── decisions/                  # Tradeoffs & experiments (3 files)
├── ideas/                      # Open issues & discarded (2 files)
├── training/                   # Fine-tuning guide
└── business/                   # Use cases
```

---

## 6. Before You Start

1. **Check [[todo]]** — scan completed items so you don't re-implement solved problems.
2. **Read relevant vault docs** — use the vault index (§5) to find what you need before touching code.

## 6b. Before You Commit

1. **Run tests** — `cargo test --all-features`. No exceptions.
2. **Run linters** — `cargo fmt -- --check` and `cargo clippy --all-features`. Fix every issue.
3. **Preserve invariants** — the 4 principles in §2 are non-negotiable.
4. **Update user docs** — if your change affects how users configure or run melder, update the relevant file in `docs/`.

### docs/ Folder

User-facing documentation in `docs/`:

| File | What it covers |
|------|----------------|
| `configuration.md` | YAML config fields, all options |
| `scoring.md` | How scoring works (exact, fuzzy, BM25, embedding, synonym) |
| `batch-mode.md` | Running `meld run` |
| `live-mode.md` | Running `meld serve`, API endpoints |
| `enroll-mode.md` | Running `meld enroll` |
| `cli-reference.md` | All CLI subcommands |
| `api-reference.md` | HTTP API routes |
| `accuracy-and-tuning.md` | Threshold tuning, improving accuracy |
| `performance.md` | Performance optimization |
| `vector-caching.md` | Cache management |
| `hooks.md` | Pipeline hooks |
| `building.md` | Building from source |

**IMPORTANT:** If your change affects how users configure or run melder, update the relevant doc. If unsure, ask.

---

## 7. Recent Changes

See [[todo]] for the most recent completed work and current backlog.
