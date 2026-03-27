← [Back to README](../README.md) | [Live Mode](live-mode.md) | [Enroll Mode](enroll-mode.md)

# Pipeline Hooks

Hooks let you run a command that reacts to matching decisions in
real time. When a match is confirmed, a pair enters the review queue,
a record has no match, or a match is broken, Melder sends a JSON event
to your script.

## How it works

Melder starts your hook command **once** when the server starts. The
command runs as a long-lived subprocess. Melder writes events to its
stdin as newline-delimited JSON — one event per line. Your script reads
lines in a loop and does whatever it needs to.

This is the same model used by many log shippers and stream processors.
There is no per-event process spawn — just a single pipe write per
event.

## Configuration

Add a `hooks` section to your config YAML:

```yaml
hooks:
  command: "python scripts/hook.py"
```

That's it. If `command` is absent, no subprocess is spawned and hooks
are disabled. The command is run via `sh -c` on Unix and `cmd /C` on
Windows.

## Event types

### `on_confirm` — match confirmed

Fired when a record is auto-matched (score >= `auto_match` threshold)
or when a match is manually confirmed via `POST /crossmap/confirm`.

```json
{
  "type": "on_confirm",
  "a_id": "ENT-001",
  "b_id": "CP-042",
  "score": 0.94,
  "source": "auto",
  "field_scores": [
    { "field_a": "legal_name", "field_b": "counterparty_name", "method": "embedding", "score": 0.97, "weight": 0.55 },
    { "field_a": "country_code", "field_b": "domicile", "method": "exact", "score": 1.0, "weight": 0.20 }
  ]
}
```

`source` is `"auto"` for threshold-based matches or `"manual"` for API
confirms. Manual confirms have `score: 0.0` and no `field_scores`.

### `on_review` — pair enters review queue

Fired when a record's best match falls in the review band
(`review_floor` <= score < `auto_match`).

```json
{
  "type": "on_review",
  "a_id": "ENT-001",
  "b_id": "CP-099",
  "score": 0.73,
  "field_scores": [...]
}
```

### `on_nomatch` — no match found

Fired when a record's best candidate scores below `review_floor`, or
when there are no candidates at all.

```json
{
  "type": "on_nomatch",
  "side": "b",
  "id": "CP-200",
  "best_score": 0.42,
  "best_candidate_id": "ENT-055"
}
```

`best_score` and `best_candidate_id` are absent when there are zero
candidates.

### `on_break` — confirmed match broken

Fired when a previously confirmed match is broken via
`POST /crossmap/break`.

```json
{
  "type": "on_break",
  "a_id": "ENT-001",
  "b_id": "CP-042"
}
```

## Example hook script

Here is a complete, working example. Save this as `scripts/hook.py`
and point your config at it:

```python
#!/usr/bin/env python3
"""Example Melder hook script.

Reads match events from stdin (newline-delimited JSON) and prints a
human-readable summary to stdout.

IMPORTANT: Every print() call uses flush=True. Without this, Python
buffers stdout when it is not connected to a terminal (which is
always the case when Melder spawns the script). Without flushing, you
will see NO output until the buffer fills (typically 8KB) or the
process exits. This is the #1 cause of "my hook doesn't print
anything" issues.
"""

import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        print(f"[hook] bad JSON: {line}", flush=True)
        continue

    kind = event.get("type", "unknown")

    if kind == "on_confirm":
        source = event.get("source", "?")
        score = event.get("score", 0)
        print(
            f"[hook] MATCHED ({source})  "
            f"{event['a_id']} <-> {event['b_id']}  "
            f"score={score:.2f}",
            flush=True,  # <-- CRITICAL: flush every line
        )

    elif kind == "on_review":
        score = event.get("score", 0)
        print(
            f"[hook] REVIEW  "
            f"{event['a_id']} <-> {event['b_id']}  "
            f"score={score:.2f}",
            flush=True,
        )

    elif kind == "on_nomatch":
        best = event.get("best_score")
        if best is not None:
            print(
                f"[hook] NO MATCH  {event['id']}  "
                f"(best: {event.get('best_candidate_id', '?')} "
                f"at {best:.2f})",
                flush=True,
            )
        else:
            print(
                f"[hook] NO MATCH  {event['id']}  (no candidates)",
                flush=True,
            )

    elif kind == "on_break":
        print(
            f"[hook] BROKEN  {event['a_id']} <-> {event['b_id']}",
            flush=True,
        )

    else:
        print(f"[hook] unknown event: {event}", flush=True)
```

> **Why `flush=True` matters:** When Python's stdout is connected to a
> pipe (not a terminal), it uses block buffering by default — output
> accumulates in an 8KB buffer and is only written when the buffer
> fills. This means your hook script appears to produce no output for
> minutes, even though events are being processed. Adding `flush=True`
> to every `print()` call forces immediate output. Alternatively, run
> Python with `-u` (unbuffered): `python -u scripts/hook.py`.

## Failure handling

### Process dies

If your hook process exits (crash, unhandled exception, killed):

1. Melder logs a warning with the exit code
2. Events arriving during the restart window are **dropped** (not
   buffered)
3. Melder respawns the process with exponential backoff (1s, 2s, 4s,
   8s, ..., capped at 60s)
4. If the process has been running stably for 60+ seconds, the failure
   counter resets

### Repeated failures

After 5 consecutive failures (process dies within 60 seconds each
time), hooks are **disabled** for the rest of the server's lifetime.
Melder logs a warning and stops trying. This prevents a broken script
from consuming resources with endless spawn attempts.

### Pipe full

The OS pipe between Melder and your script has a buffer (typically
16-64KB). If your script reads slower than Melder writes, events queue
in the pipe buffer. If the buffer fills completely (script is severely
stalled), Melder drops events and logs a warning. The scoring pipeline
is never blocked.

### Command not found

If the hook command fails to start (bad path, Python not installed),
Melder treats it as a process failure — same backoff and failure
counting as above.

## Performance

Hooks have **zero impact on the scoring pipeline**:

- The scoring thread sends events via a non-blocking channel send
  (~10 nanoseconds)
- A dedicated background task reads from the channel, serializes
  JSON, and writes to the pipe
- If the channel is full (1000 events buffered), new events are
  silently dropped
- If hooks are not configured, there is no channel, no task, no
  subprocess — zero overhead

## Enroll mode

Hooks work in enroll mode (`meld enroll`). Since enroll mode has no
crossmap and no classification, only `on_nomatch` applies — fired when
an enrolled record has zero edges above `review_floor`. The
`on_confirm`, `on_review`, and `on_break` events are not emitted in
enroll mode.

## Limitations

- **Best-effort delivery.** Events may be dropped if the hook process
  is down, the pipe is full, or the internal channel is full. Hooks
  are for notifications, not guaranteed delivery.
- **No ordering guarantee under concurrency.** When multiple requests
  are processed concurrently, hook events may arrive at the subprocess
  in a different order than they were generated.
- **No batching.** Each event is written individually. If your script
  needs batching (e.g., bulk database inserts), implement it in the
  script (accumulate N events, then flush).
- **Single subprocess.** All events go to one process. If you need
  fan-out (e.g., different destinations for different event types),
  route inside your script.
