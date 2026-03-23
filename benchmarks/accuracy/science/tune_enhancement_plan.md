# Plan: Ground-Truth Analysis in `meld tune`

## Context

The science experiments built Python scripts (score_chart.py, overlap.py, evaluate.py) that analyse score distributions by splitting them into **known match** vs **known non-match** populations. This two-population view is far more useful than the current `meld tune` output, which lumps all scores together and can only suggest percentile-based thresholds.

The config already supports `common_id_field` on both datasets. Currently this auto-matches records at score 1.0 before scoring. In tune mode, we repurpose it as **ground truth** — score everything normally, then classify results against the common ID mapping.

## Approach

### 1. Skip common_id pre-matching in tune mode

**File:** `src/batch/engine.rs`

- Add `skip_common_id_prematch: bool` parameter to `run_batch()`
- Wrap the common_id pre-match block (~lines 86-144) with `if !skip_common_id_prematch`
- Also wrap the exact_prefilter block (~lines 147-215) — both short-circuit scoring

**File:** `src/cli/run.rs` (2 call sites, lines 137 and 270)
- Pass `false` (preserve existing behaviour)

**File:** `src/cli/tune.rs` (1 call site, line 80)
- Pass `true` when `common_id_field` is configured on both datasets, `false` otherwise

### 2. Build ground-truth lookup

**File:** `src/cli/tune.rs`

Before calling `run_batch`, when both `common_id_field`s are set:
- Iterate A records, build `HashMap<common_id_value, a_record_id>`
- Iterate B records, look up each B's common_id in the A-side map
- Produce `HashMap<b_id, a_id>` — the ground truth ("this B should match this A")
- Store as `Option<HashMap<String, String>>` — `None` = no ground truth, use old output

### 3. Classify scores into two populations

**File:** `src/cli/tune.rs`

After `run_batch`, iterate all results (matched + review + unmatched):
- Look up each B record's ID in the ground-truth map
- If present → **known match**, record its score in `match_scores: Vec<f64>`
- If absent → **known non-match**, record its score in `nonmatch_scores: Vec<f64>`
- For unmatched records with `None` best score, use 0.0

Also track precision/recall counters:
- `auto_tp`: in matched, ground-truth A == `matched_id`
- `auto_fp`: in matched, no ground-truth or wrong A
- `review_tp` / `review_fp`: same for review
- `fn_count`: known matches that ended up in unmatched or matched to wrong A

### 4. New output sections (when ground truth available)

**a) Two-population ASCII histogram**
- `█` known match, `░` known non-match
- `▼` threshold annotations on review_floor and auto_match rows
- Skip empty edge buckets (unless they fall within user-specified range)

**b) Numeric distribution table**
- Per-bucket counts: matched, unmatched, total
- Skip buckets with zero total (unless within user-specified range)
- Same bucket_width and range as the histogram

**Display parameters for both (a) and (b)** — exposed as CLI flags on `meld tune`:

| Flag | Default | Description |
|---|---|---|
| `--bucket-width` | 0.04 | Width of each score bucket |
| `--min-score` | auto (first non-empty bucket) | Lower bound of display range |
| `--max-score` | auto (last non-empty bucket) | Upper bound of display range |
| `--bar-width` | 40 | Max bar width in characters (histogram only) |

Defaults produce sensible output for most cases. The user can zoom in with e.g. `--min-score 0.50 --max-score 0.70 --bucket-width 0.02` to inspect the overlap zone at high resolution.

**c) Overlap coefficient**
- `Σ min(P_match(bucket), P_nonmatch(bucket))` with bucket_width=0.02 (fixed, not affected by display params)
- Print single line: `Overlap: 0.1234  (0 = perfect separation, 1 = identical)`

**d) Ground-truth accuracy**
```
Auto-matched:      1,234
  Correct (TP):    1,200
  Incorrect (FP):     34
Review:              567
  Correct (TP):      500
  Incorrect (FP):     67
Missed (FN):          89

Precision:         97.2%
Recall (auto):     63.2%
Combined recall:   89.5%
```

**e) Per-field stats and threshold analysis** — kept unchanged from current output

### 5. `--no-run` flag for fast re-analysis

**Problem:** The scoring pipeline (embeddings, ANN, scoring) takes minutes. The analysis (bucketing, charts, overlap) takes milliseconds. Without caching, every change to `--bucket-width` or `--min-score` re-runs the full pipeline.

**Solution:**
- `meld tune` writes output files (results.csv, review.csv, unmatched.csv) to the paths specified in config — same as `meld run` does
- Add `--no-run` CLI flag: skips the scoring pipeline entirely, reads existing output files, and runs only the analysis/display
- If `--no-run` is set but output files don't exist → error with clear message

**Workflow:**
```bash
# First run: full pipeline + analysis (slow)
meld tune --config config.yaml

# Subsequent runs: just re-analyse with different view params (instant)
meld tune --config config.yaml --no-run --min-score 0.50 --max-score 0.70 --bucket-width 0.02
meld tune --config config.yaml --no-run --bucket-width 0.01
```

### 6. Backwards compatibility

- No `common_id_field` configured → output is identical to today
- `common_id_field` configured but zero matching records → warning, fall back to old output
- `--no-run` without `common_id_field` → still works, shows old-style analysis from cached output

## Files to modify

| File | Changes |
|---|---|
| `src/batch/engine.rs` | Add `skip_common_id_prematch` param, guard 2 blocks |
| `src/cli/run.rs` | Pass `false` at 2 call sites |
| `src/cli/tune.rs` | Ground-truth map, score classification, new output sections, `--no-run` flag, write output CSVs |
| `src/batch/writer.rs` | Reuse existing CSV writing for tune output (if not already shared) |

## Verification

1. `cargo build --release --features usearch` — compiles
2. Run `meld tune --config config_without_common_id.yaml` — output unchanged
3. Run `meld tune --config config_with_common_id.yaml` — new two-population output appears
4. Run `meld tune --config config_with_common_id.yaml --no-run` — instant, same output
5. Run with `--no-run --bucket-width 0.02 --min-score 0.5 --max-score 0.7` — zoomed view
6. Compare histogram/overlap against Python scripts on same dataset for consistency
