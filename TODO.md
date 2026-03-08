# TODO

Items to discuss and design before implementing.

---

## 1. Hooks

User-defined logic that runs at specific points in the pipeline. Open
questions:

- Which hook points? Candidates include: pre-score (after blocking,
  before scoring), post-score (after composite score, before
  classification), post-match (after classification, before output/
  response), on-confirm (when a cross-map entry is created).
- What form do hooks take? Options: external process (subprocess with
  JSON on stdin/stdout), WASM plugin, Lua script, HTTP callout to a
  user-provided endpoint.
- What can a hook do? Read-only inspection (logging, metrics, side
  effects) vs mutation (adjust scores, override classification, enrich
  records, veto a match).
- Should hooks be sync (blocking the pipeline) or async (fire-and-forget)?
- Config syntax -- something like:
  ```yaml
  hooks:
    post_score:
      - command: ./scripts/override_lei_matches.sh
        timeout: 5s
    on_confirm:
      - url: https://internal.example.com/webhooks/match-confirmed
  ```

---

## 2. Data output in live mode

Currently live mode returns match results in the HTTP response but does
not persist them to disk the way batch mode writes results/review/
unmatched csvs. Questions:

- Should the server write a running log of match results (append-only
  csv or JSONL)? If so, what triggers a write -- every `/add` response,
  only confirmed matches, or both?
- Should there be an endpoint to dump current results/review/unmatched
  state on demand (e.g. `GET /api/v1/export`)?
- Should the server periodically snapshot its state to the same output
  paths that batch mode uses, so the files are always up to date?
- How does this interact with the WAL? The WAL captures record additions
  and cross-map changes for crash recovery, but it does not capture
  match scores or classifications.

---

## 3. Review and N-matches in live mode

In batch mode, each B record gets exactly one best match (or lands in
review/unmatched). Live mode returns `top_n` matches per request but
has no review queue or multi-match workflow. Questions:

- Should live mode maintain a review queue analogous to batch mode?
  Borderline matches (between `review_floor` and `auto_match`) could
  be held in a queue accessible via API rather than auto-returned.
- How should N-matches work? Currently `/add` returns up to `top_n`
  scored candidates. Should the caller be able to confirm any of the N
  (not just the top one)? That already works via `/crossmap/confirm`,
  but there is no UI or workflow around it.
- Should there be a `/review` endpoint family (list, accept, reject)
  mirroring the CLI review commands but operating on the live review
  queue?
- If a record is added and its best match scores in the review band,
  should the response indicate that explicitly (e.g.
  `"needs_review": true`) rather than just returning the scores and
  leaving classification interpretation to the caller?
