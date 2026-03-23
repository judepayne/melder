---
name: Table format preference
description: User prefers a specific holdout results table format with auto-matched/review/unmatched buckets and clean/heavy noise/not-a-match sub-lines
type: feedback
---

Always present training experiment results in the fixed table format defined in `benchmarks/accuracy/science/AGENTS.md`. Key points:

- Holdout only (no train columns)
- One column per round, Round 0 labelled "(base)"
- Three main buckets: Auto-matched, Review, Unmatched
- Each bucket has sub-lines: Clean, Heavy noise, Not a match (or Missed clean/heavy noise for Unmatched)
- Bottom section: Precision, Recall (vs ceiling), Combined recall
- Numbers formatted with commas, right-aligned
- Use `output_table.py` script to generate programmatically

**Why:** User shares these tables with non-technical stakeholders. Clear labelling matters — avoid jargon like "false positive", "phantom", "leaked". Use "Not a match" consistently.

**How to apply:** When user asks to see results or "show me the table", use this format. Run `output_table.py` first, then present in the conversation.
