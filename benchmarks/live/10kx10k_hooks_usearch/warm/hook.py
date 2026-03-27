#!/usr/bin/env python3
"""Hook script for the hooks benchmark — prints all events to stdout.

IMPORTANT: flush=True on every print(). Without it, Python buffers
stdout when connected to a pipe and you see nothing until the buffer
fills or the process exits.
"""

import json
import sys

counts = {"on_confirm": 0, "on_review": 0, "on_nomatch": 0, "on_break": 0}

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
    counts[kind] = counts.get(kind, 0) + 1

    if kind == "on_confirm":
        source = event.get("source", "?")
        score = event.get("score", 0)
        print(
            f"[hook] MATCHED ({source:6s})  "
            f"{event['a_id']:>20s} <-> {event['b_id']:<20s}  "
            f"score={score:.2f}",
            flush=True,
        )

    elif kind == "on_review":
        score = event.get("score", 0)
        print(
            f"[hook] REVIEW          "
            f"{event['a_id']:>20s} <-> {event['b_id']:<20s}  "
            f"score={score:.2f}",
            flush=True,
        )

    elif kind == "on_nomatch":
        best = event.get("best_score")
        cand = event.get("best_candidate_id", "none")
        if best is not None:
            print(
                f"[hook] NO MATCH        {event['id']:>20s}  best={cand} at {best:.2f}",
                flush=True,
            )
        else:
            print(
                f"[hook] NO MATCH        {event['id']:>20s}  (no candidates)",
                flush=True,
            )

    elif kind == "on_break":
        print(
            f"[hook] BROKEN          {event['a_id']:>20s} <-> {event['b_id']:<20s}",
            flush=True,
        )

    else:
        print(f"[hook] unknown: {event}", flush=True)

# Print summary on EOF (server shutdown closes stdin)
print("\n--- Hook summary ---", flush=True)
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}", flush=True)
