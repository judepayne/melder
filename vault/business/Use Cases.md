---
type: business
module: general
status: active
tags: [use-cases, batch-mode, live-mode, enterprise]
related_code: [cli/run.rs, cli/serve.rs, api/handlers.rs]
---

# Use Cases

Melder is a general-purpose record matching engine. The three use cases below are representative of common enterprise data management problems, particularly in financial services. They are illustrative, not exhaustive.

## 1. Overnight Batch Reconciliation Against a Reference Data Master

### The Problem

Large organisations maintain authoritative reference data masters -- golden sources for clients, legal entities, instruments, counterparties, or securities. Periodically (typically overnight), they receive vendor files: Bloomberg instrument feeds, Markit counterparty lists, KYC provider extracts, or internal system dumps. These files refer to the same real-world entities as the master but use different identifiers, different name spellings, different field structures, and often no shared key.

The task is to match every record in the vendor file against the master, identifying which vendor records correspond to which master records, which are new (no match), and which are ambiguous (close but not certain). This must complete within an overnight batch window -- typically a few hours at most -- and produce outputs that feed downstream enrichment, exception handling, and onboarding workflows.

Without automated matching, operations teams manually eyeball spreadsheets, a process that is slow, error-prone, and does not scale beyond a few hundred records.

### How Melder Solves It

**Mode: Batch (`meld run`)**

The reference data master is loaded as dataset A (the reference pool). The vendor file is loaded as dataset B (the query set). Melder matches every B record against the A-side pool and writes three output files:

- **results.csv** -- confirmed matches (score >= `auto_match` threshold). These can be fed directly into downstream enrichment: for each matched pair, the vendor record (B) is linked to the master record (A), and master attributes (LEI, classification, domicile, etc.) can be stamped onto the vendor record.
- **review.csv** -- borderline pairs (score between `review_floor` and `auto_match`). These are routed to a human review queue for manual decision.
- **unmatched.csv** -- vendor records with no viable match. These are candidates for new master record creation or further investigation.

The [[Constitution#1 Batch Asymmetry Live Symmetry|batch asymmetry]] is intentional here: B records are the ones being enriched with A-side data, so B is the query side.

### Configuration Considerations

- **Blocking** is critical for performance at scale. If both datasets share a categorical field (country, currency, asset class), blocking on it eliminates the vast majority of impossible pairs cheaply. At 100k records per side with country blocking, each B record typically searches ~5k candidates instead of 100k.
- **Common ID pre-match**: If both datasets share a stable identifier (ISIN, LEI, CUSIP), configure `common_id_field` on both sides. Records with identical common IDs are matched instantly with score 1.0 before any scoring runs, dramatically reducing the workload for the remaining records.
- **Caching**: Set `a_cache_dir` (required) and `b_cache_dir` (optional but recommended for recurring jobs). The first run encodes all records through the ONNX model and caches the vectors to disk. Subsequent runs with the same data load in milliseconds. When the vendor file changes daily but the master is mostly stable, incremental text-hash diffing re-encodes only the changed records. See [[Key Decisions#Three-Layer Cache Invalidation]].
- **Threshold tuning**: Use `meld tune` before the first production run to examine the score distribution and set appropriate thresholds. A well-tuned job typically auto-matches 50-70% of records, sends 15-25% to review, and flags 10-20% as unmatched. See the tuning guide in `TUNE.md`.
- **Scheduling**: The job is stateless and idempotent. Run it via cron, Airflow, or any scheduler. The cross-map accumulates confirmed matches across runs, so re-running after importing review decisions progressively shrinks the review queue.

### Scale Reference

On a 100k x 100k dataset with usearch backend and warm caches, batch scoring completes in ~12 seconds wall time (~9,886 records/sec). See [[Performance Baselines]] and [[Benchmarks]].

---

## 2. Live Duplicate Detection Over a Reference Data Master

### The Problem

When a front-office user, onboarding team, or automated feed attempts to create a new record in a reference data master (new client, new instrument, new counterparty), there is a risk of creating a duplicate of an entity that already exists under a slightly different name or identifier. Duplicate records cause downstream problems: split positions, duplicated risk exposure, reconciliation breaks, and regulatory reporting errors.

The traditional approach is a manual search -- the user types a name into a search box and eyeballs the results. This is unreliable for several reasons: users may not search thoroughly, exact-match search misses spelling variations ("JP Morgan" vs "JPMorgan Chase"), and the search may not cover all relevant fields (name, LEI, country, short name).

What is needed is a real-time similarity search that takes the proposed new record, scores it against all existing records using semantic and fuzzy matching, and immediately returns the most likely duplicates with confidence scores -- before the record is committed.

### How Melder Solves It

**Mode: Live (`meld serve`) using the `/match` endpoint**

The reference data master is loaded as dataset A on server startup. When a user or system wants to check for duplicates before creating a new record, it sends the proposed record to the **`/b/match`** endpoint. This scores the record against the entire A-side pool using the same pipeline as batch mode (see [[Constitution#2 One Scoring Pipeline]]) but does **not** store the record or modify any state. It is a pure read-only similarity search.

The response returns the top-N most similar existing records with scores and per-field breakdowns. The calling system can then:

- **Auto-reject** if a match scores above `auto_match` (the entity almost certainly already exists)
- **Flag for review** if a match scores in the review band (possible duplicate, needs human judgement)
- **Allow creation** if no match scores above `review_floor` (genuinely new entity)

If the record is confirmed as new and should be added to the master, a follow-up call to `/a/add` inserts it into the live index so that future duplicate checks will find it.

### Why Live Mode

Batch mode cannot serve this use case because the check must happen in real-time, at the moment the user is attempting the creation. Latency requirements are tight -- the check should complete in under 50ms to feel instantaneous in a UI. Melder's live mode delivers p50 latency of 2-3ms at 10k records and 3-7ms at 100k records (see [[Benchmarks]]), well within this budget.

### Configuration Considerations

- **`top_n`**: Set to 5-10 for duplicate detection. You want to show the user a short list of the most plausible duplicates, not a long ranked list.
- **Blocking**: Use with care. Blocking on country is sensible (a UK client is not a duplicate of a US client), but over-aggressive blocking may miss cross-subsidiary duplicates. Consider OR-mode multi-field blocking if multiple routes to the same entity exist.
- **The `/match` vs `/add` distinction is critical**: `/match` is read-only and safe to call speculatively. `/add` modifies state. The duplicate-check workflow should use `/match` first, and only `/add` after human or system confirmation.
- **Text-hash skip**: In this use case, the A-side records are relatively stable (the master changes slowly). The text-hash optimisation means that periodic bulk reloads of the master (e.g., nightly full refresh) only re-encode records whose text actually changed. See [[Key Decisions#Text-Hash Skip Optimization]].
- **WAL and crash recovery**: Configure `live.upsert_log` so that any records added via `/a/add` during the day survive a server restart. On startup, the WAL is replayed to restore state. See [[State & Persistence#WAL]] and [[Config Reference#live]].

### Integration Pattern

A typical integration is a thin API gateway or middleware layer between the user-facing application and Melder:

```
User UI  -->  Middleware  -->  POST /api/v1/b/match  -->  Melder
                                                           |
              <-- "Possible duplicates found" <-----------+
              or
              <-- "No duplicates, proceed" <--------------+
                    |
                    v
              POST /api/v1/a/add  -->  Melder  (only if confirmed new)
```

The middleware interprets the scores and applies business rules (auto-reject threshold, mandatory review threshold, etc.) that may differ from Melder's built-in thresholds.

---

## 3. Continuous Synchronisation of Overlapping Reference Masters

### The Problem

In large organisations -- particularly those formed through mergers, acquisitions, or organic siloing -- it is common to have two or more reference data masters that describe the same domain (e.g., customers, counterparties, instruments) but were built independently. Each system has its own identifiers, its own data model, its own coverage, and its own data quality characteristics. There is partial overlap (many entities exist in both systems) but no reliable common key to join on.

The business need is to continuously synchronise these systems: identify which records in System X correspond to which records in System Y, detect when new records appear in either system, and maintain an ongoing mapping between the two. This is not a one-off migration -- both systems continue to receive new records and updates, and the mapping must stay current.

Manual reconciliation is impractical at scale and becomes stale immediately. Traditional ETL joins fail because there is no common key. Even when partial keys exist (e.g., both systems have a "name" field), the values differ enough that exact matching misses a large fraction of true pairs.

### How Melder Solves It

**Mode: Live (`meld serve`) using the `/add` endpoints**

Both datasets are loaded at startup -- System X as dataset A, System Y as dataset B. The [[Constitution#1 Batch Asymmetry Live Symmetry|live mode symmetry]] is essential here: both sides are equal participants, and records flow in from both directions.

The synchronisation operates as a continuous loop:

1. **Initial bulk load**: On first startup, Melder loads both datasets from their source files, builds embedding indices, and scores all records. The cross-map captures the initial set of confirmed matches. The review queue captures ambiguous pairs for human resolution.

2. **Ongoing change feed**: As new records appear in either system (or existing records are updated), they are pushed to Melder via `/a/add` or `/b/add`. Each addition immediately triggers scoring against the opposite side. If a match is found above `auto_match`, it is confirmed in the cross-map automatically. If it falls in the review band, it is surfaced for human decision.

3. **Cross-map as the synchronisation ledger**: The [[Constitution#3 CrossMap Bijection 1 1 Under One Lock|cross-map]] is the authoritative record of which System X entity corresponds to which System Y entity. It enforces 1:1 pairing, which is the correct model for entity resolution: one real-world entity should map to exactly one record in each system. The cross-map can be exported at any time via `meld crossmap export` and fed to downstream integration pipelines.

4. **Bidirectional detection**: When a new record is added to System X (via `/a/add`), Melder immediately checks whether a corresponding record already exists in System Y. Conversely, a new System Y record (via `/b/add`) is checked against System X. This catches the common scenario where the same real-world entity is onboarded into both systems independently at different times.

5. **Removals**: When a record is decommissioned in one system, `/a/remove` or `/b/remove` removes it from the index and breaks any existing cross-map pair. The orphaned record on the other side is returned to the unmatched pool and will be re-evaluated when new records arrive.

### Why Live Mode (Not Batch)

Batch mode could handle the initial matching, but it cannot maintain the mapping over time. New records appear continuously in both systems, and each must be checked immediately -- not on the next nightly batch run. A stale mapping means new duplicates, missed linkages, and integration drift.

Live mode provides:
- **Immediate matching** on every record change (p50 < 10ms)
- **Symmetric treatment** of both sides (batch mode is asymmetric -- B queries A only)
- **Persistent state** via the cross-map and WAL, surviving restarts
- **Real-time cross-map** that downstream systems can poll or export

### Configuration Considerations

- **Blocking**: Choose fields that exist in both systems and are reasonably reliable. If both systems have a country field, block on it. If coverage is patchy, use OR-mode with multiple blocking fields so records have multiple routes to find their match.
- **`common_id_field`**: If both systems share any common identifier (even a partial one -- e.g., both have an LEI field but not all records have it populated), configure it. Records with matching common IDs are paired instantly, reducing the scoring workload.
- **Cross-map persistence**: The cross-map is flushed to disk periodically (default every 5 seconds) and on shutdown. For production deployments, back up the cross-map CSV regularly -- it is the most valuable output of the synchronisation process.
- **WAL**: Essential for this use case. Configure `live.upsert_log` so that all record additions and cross-map changes are journalled. On restart, the WAL replays to restore full state without re-processing the change feed. See [[State & Persistence]].
- **Review workflow**: Integrate with `meld review list` and `meld review import` to process ambiguous pairs. Accepted pairs are added to the cross-map. Rejected pairs are removed from the review queue. Over time, the fraction of records requiring human review should decrease as the cross-map grows and only genuinely new entities remain unmatched.
- **Monitoring**: Use the `/status` endpoint to monitor record counts, uptime, and cross-map size. Alert if the unmatched pool grows unexpectedly -- this may indicate a data quality issue in one of the source systems.

### Architecture Pattern

```
System X (change feed)                          System Y (change feed)
       |                                               |
       v                                               v
  POST /api/v1/a/add                          POST /api/v1/b/add
       |                                               |
       +-------------------+   +-------------------+
                           |   |
                           v   v
                         Melder (live)
                           |
                    Cross-map (1:1 ledger)
                           |
                           v
                  Integration pipeline
                  (export, sync, enrich)
```

The integration pipeline periodically exports the cross-map and uses it to synchronise records between the two systems -- copying missing fields, flagging discrepancies, or feeding a unified view to downstream consumers.

---

## Summary

| Use Case | Mode | Dataset A | Dataset B | Key Endpoint |
|---|---|---|---|---|
| Overnight batch reconciliation | Batch (`meld run`) | Reference master | Vendor file | N/A (file output) |
| Live duplicate detection | Live (`meld serve`) | Reference master | Proposed new records | `/b/match` (read-only) |
| Continuous sync of two masters | Live (`meld serve`) | System X | System Y | `/a/add`, `/b/add` (bidirectional) |

See also: [[Constitution]], [[Business Logic Flow]], [[Benchmarks]]
