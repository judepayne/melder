← [Back to README](../README.md) | [Configuration](configuration.md) | [Performance](performance.md)

# How Vector Caching Works

## Overview

The melder uses a sentence-transformer model (default:
`all-MiniLM-L6-v2`) to convert each record's text fields into dense
numeric vectors — fingerprints that capture meaning rather than
characters. Two records about the same entity produce vectors that point
in nearly the same direction, even if the wording differs completely.
This is how `method: embedding` scoring works.

Encoding is expensive: running 10,000 records through the ONNX model
takes around 8 seconds. To avoid repeating this work, the melder caches
the encoded vectors to disk after the first run.

## The combined embedding index

Rather than storing a separate vector index for every embedding field,
the melder builds a single *combined index* per side. For each record,
the vectors for all embedding fields are scaled by the square root of
their weights and concatenated into one long vector. This combined
vector has a useful property: searching for the nearest combined vectors
is exactly equivalent to finding the records with the highest weighted
cosine similarity across all embedding fields — the same ranking that
full scoring would produce. A single nearest-neighbour search therefore
retrieves the right candidates without needing multiple per-field
lookups.

The cache filename encodes a hash of the field names, their order, and
their weights. Changing any of these produces a different filename, so
the old cache is ignored and a fresh one is built automatically.
`meld cache clear` (without `--all`) uses this same hash to identify
and delete only the now-unreachable stale files.

## Config options

```yaml
embeddings:
  model: all-MiniLM-L6-v2
  a_cache_dir: cache                 # required
  b_cache_dir: cache                 # optional — omit to skip B-side caching
```

`a_cache_dir` is required — the A-side combined index is always cached
to disk. On first run the directory is created and populated; on
subsequent runs the index loads in milliseconds (~170ms for 100k
records).

`b_cache_dir` is optional. When set, the B-side combined index is also
saved to disk. When omitted, B vectors are re-encoded from scratch on
every run.

## Batch mode lifecycle

```
First run:
  A index: check manifest → missing → cold build → save index + manifest + texthash
  B index: same

Subsequent runs (same data, same config):
  A index: manifest fresh → load index + texthash → diff: 0 changed → return (ms)
  B index: same

Subsequent runs (some records changed):
  A index: manifest fresh → load → diff: N records changed → re-encode N → save
  B index: same

Config changed (model / spec / blocking):
  A index: manifest mismatch → log reason → cold build
  B index: same

Scoring (per B record):
  1. Look up B record's combined vector in the B index
  2. Search the A combined index for the top_n nearest neighbours (O(log N)
     with usearch, O(N) with flat)
  3. Score each candidate across all match fields
  4. Classify and output
```

### Setting b_cache_dir for threshold tuning

Setting `b_cache_dir` is especially valuable when tuning thresholds or
weights — encoding is done once and the score distribution can be
explored cheaply on subsequent runs.

## Live mode lifecycle

```
Startup:
  A index: load from a_cache_dir (or encode + build if stale)
  B index: load from b_cache_dir (or encode; only saved if configured)
  Both indices live in memory for fast concurrent access.

During operation (upsert / try-match):
  Encode the record's embedding fields into a combined vector.
  Upsert the combined vector into the in-memory index.
  Search the opposite side's index for top_n nearest neighbours.

Shutdown:
  Save combined indices to their cache directories.
```

## Staleness and invalidation

Cache validation is multi-layered and runs automatically on every
startup. No manual intervention is needed.

### Layer 1 — Config hash (manifest check)

A `.manifest` sidecar is stored alongside each cache file. It records a
hash of the embedding field spec (field names, order, weights), the
blocking configuration, and the model name. On load, these hashes are
compared against the current config. Any mismatch triggers an immediate
cold rebuild with a clear log message explaining what changed:

```
Warning: A combined index cache invalidated (blocking config changed), rebuilding from scratch.
Warning: B combined index cache invalidated (embedding model changed), rebuilding from scratch.
```

### Layer 2 — Text-hash deduplication (incremental encoding)

After the manifest check passes, the engine computes a FNV-1a hash of
each record's source text and compares it against the stored hashes in a
`.texthash` sidecar. Records whose text has not changed are skipped —
their cached vectors are reused. Only records whose text actually
changed (or that are new) are re-encoded through the ONNX model.

This means recurring batch jobs where most records are stable only
re-encode the changed minority. If more than 90% of records change in a
single run, a full cold rebuild is triggered instead (more efficient
batching outweighs the incremental overhead at that point).

### Layer 3 — Spec hash in the filename

If you change a field's weight, rename a field, or add/remove an
embedding field, the spec hash embedded in the cache filename changes
and the old cache file becomes unreachable. The engine builds a fresh
index automatically; `meld cache clear` (smart mode) finds and removes
the now-unreachable old file along with its sidecars.

## Cache files produced per index

| File | Contents |
|------|----------|
| `*.index` | Flat backend: combined vectors (binary). Usearch: key mapping manifest only. |
| `*.usearchdb/` | Usearch backend: HNSW graph files, one per block. |
| `*.index.manifest` | Config hashes, model name, record count, build timestamp. |
| `*.index.texthash` | Per-record FNV-1a hashes of source text. |

## meld cache status

`meld cache status` prints the model, spec hash, blocking hash,
record count, and build timestamp from each manifest:

```
  A cache          benchmarks/batch/100kx100k_usearch/warm/cache (1 index files, 52.3 MB)
    model=all-MiniLM-L6-v2 spec=a3f7c2b1 blocking=deadbeef records=100000 built=2026-03-10T14:22:05Z
```

## meld cache clear

`meld cache clear` and `meld cache clear --all` both delete the
sidecars alongside the index files they belong to. See
[CLI Reference](cli-reference.md) for full usage.
