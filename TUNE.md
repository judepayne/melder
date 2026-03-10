# Tuning guide

Getting a new matching job right involves three things: choosing the right
fields and methods, setting their weights, and placing the thresholds.
The `meld tune` command runs the full batch pipeline without writing any
output files and produces a report that gives you the evidence to make
all three decisions without guesswork.

---

## Running the tune command

```bash
meld tune --config config.yaml
```

No output files are written. The full batch pipeline runs and a report
is printed to stdout. On the 1,000x1,000 benchmark dataset this takes
about 8 seconds (warm cache).

```bash
# Show startup timings and current threshold values
meld tune --config config.yaml --verbose
```

---

## What the report contains

The report has four sections, each answering a different question.

### 1. Score distribution

```
=== Score Distribution ===

  [0.0-0.1)    27 (  2.7%) ##
  [0.1-0.2)     0 (  0.0%)
  [0.2-0.3)     0 (  0.0%)
  [0.3-0.4)     3 (  0.3%)
  [0.4-0.5)    37 (  3.7%) ###
  [0.5-0.6)    61 (  6.1%) #####
  [0.6-0.7)    41 (  4.1%) ####
  [0.7-0.8)   123 ( 12.3%) ###########
  [0.8-0.9)   295 ( 29.5%) ##########################
  [0.9-1.0]   413 ( 41.3%) ########################################
```

**What to look for:** a healthy score distribution is bimodal -- a
cluster of high-scoring records (likely matches) and a cluster of
low-scoring ones (likely non-matches), with a relatively thin band in
the middle. The `#` bars are proportional to the largest bucket, so you
can see the shape at a glance.

A distribution that is mostly flat or heavily concentrated in the middle
suggests your fields are not discriminating well. Look to the per-field
section to find out why.

### 2. Per-field score statistics

```
=== Per-Field Score Statistics ===

  Field                                      Min    Max   Mean Median StdDev
  --------------------------------------------------------------------------
  country_code_domicile                    1.000  1.000  1.000  1.000  0.000
  legal_name_counterparty_name             0.087  1.000  0.791  0.842  0.193
  lei_lei_code                             0.000  1.000  0.542  1.000  0.498
  short_name_counterparty_name             0.118  1.000  0.900  1.000  0.183
```

**What to look for:** `StdDev` is the discriminating-power indicator. A
field with `StdDev ~ 0` scores identically for every record -- it adds
no information and its weight is wasted (or worse, uniformly biases
scores). A field with high `StdDev` varies across records and is doing
real work.

`Mean` tells you the typical score. A high `Mean` combined with low
`StdDev` (e.g. a field that almost always scores 0.9) means the field is
a weak separator -- it can only help push borderline records down, not
up.

### 3. Threshold analysis

```
=== Threshold Analysis (current settings) ===

  auto_match  >= 0.85:   569 (56.9%)
  review      >= 0.60:   303 (30.3%)
  no_match     < 0.60:   128 (12.8%) (incl. 27 with no candidates)
  total:            1000
```

This shows how the current thresholds split your records. The "no
candidates" count tells you how many B records found no A candidate at
all after blocking -- these are invisible to scoring and no amount of
weight tuning will recover them.

### 4. Suggested thresholds

```
=== Suggested Thresholds ===

  Suggested auto_match  (90th percentile): 0.9312
  Suggested review_floor (50th percentile): 0.8410
```

These are percentile-based suggestions. The 90th percentile means "only
the top 10% of scores auto-match" and the 50th percentile means
"roughly half of scored pairs go to review." These are starting points
-- adjust based on how large a review queue you can handle and how much
recall risk you can tolerate.

---

## Worked example: the 1kx1k benchmark

The benchmark config (`testdata/configs/bench1kx1k.yaml`) matches
1,000 synthetic counterparty records (dataset B) against 1,000 entity
master records (dataset A) using four fields:

| Field | Method | Weight |
|---|---|---|
| `legal_name` / `counterparty_name` | embedding | 0.55 |
| `short_name` / `counterparty_name` | fuzzy (partial_ratio) | 0.20 |
| `country_code` / `domicile` | exact | 0.20 |
| `lei` / `lei_code` | exact | 0.05 |

Blocking is on `country_code` / `domicile`. Thresholds: `auto_match=0.85`,
`review_floor=0.60`.

### Score distribution analysis

The distribution is heavily right-skewed -- 70.8% of records score above
0.80, which indicates the pipeline is successfully identifying the clear
majority of matches. The dead zone at 0.1--0.3 is a good sign: scores
are genuinely bimodal, not a continuous smear.

The 27 records at 0.0--0.1 are records that found zero candidates after
blocking -- meaning their `domicile` value had no corresponding
`country_code` in dataset A at all. They cannot match regardless of how
scores are tuned.

The review queue (0.60--0.85) contains 303 records (30.3% of all B
records). Whether that is acceptable depends on the use case, but it is
large relative to the unmatched pool. This is a sign that the thresholds
and/or weights may be pulling records upward into review that should be
unmatched.

### Per-field analysis: the country_code problem

```
  country_code_domicile     1.000  1.000  1.000  1.000  0.000
```

`StdDev = 0.000`. This field scores 1.0 for every single record because
`country_code`/`domicile` is also the blocking field. The blocking
filter ensures that every candidate pair already has a matching country
-- so the exact match on that field is guaranteed before scoring even
begins. Including it as a match field is not just wasteful: its 0.20
weight lifts every composite score by 0.20 uniformly, which pushes
borderline records into review that should be unmatched. **This weight
should be redistributed.**

### Per-field analysis: short_name's weak discrimination

`short_name` scores 0.964 for auto-matched records and 0.932 for review
records -- a gap of only 0.032. This means `short_name` is almost
identical in score for records that eventually auto-match versus those
that need human review. The only population where it differs
meaningfully is unmatched (0.438), meaning it mainly detects obvious
non-matches rather than discriminating quality within the match band.

With `partial_ratio` as the scorer and the B field being
`counterparty_name` (a longer form of the name), very high scores are
almost always returned even for near-matches -- the scorer is too
permissive for this use case.

### Per-field analysis: lei is underweighted

The `lei` field shows strong separation across all three populations
(0.71 auto, 0.39 review, 0.08 unmatched) despite having the lowest
weight of all fields (0.05). The high `StdDev` (0.498) confirms it is
bimodal: when present in both datasets, it is a near-certain match
signal. When absent from B (as is common), it scores 0 and contributes
nothing. This is the correct behaviour for a sparse but reliable field
-- but its weight of 0.05 means it barely moves the composite score even
when it fires. Given the clear separation it shows, it merits a
substantially higher weight.

### Per-field analysis: legal_name is the primary signal

`legal_name` shows the cleanest separation of all four fields: each
population sits in a clearly distinct band (0.92 auto, 0.68 review,
0.40 unmatched) with low within-band variance (stddev ~ 0.085--0.090
across all three). This is the field doing the real discriminating work.
Its 0.55 weight is appropriate in direction but given that
`country_code` is contributing nothing, that freed weight should flow
primarily here.

---

## Suggested configuration changes

Based on the analysis above, the following changes would produce a
tighter, more accurate result:

**1. Remove `country_code` as a match field.** It contributes no
information because blocking already guarantees a country match.
Removing it frees 0.20 weight.

**2. Redistribute the freed weight to `legal_name` and `lei`.** The
embedding field is the primary signal; `lei` punches above its weight.
A rebalanced set might be:

| Field | Old weight | Suggested weight | Rationale |
|---|---|---|---|
| `legal_name` (embedding) | 0.55 | 0.65 | Primary signal, best separation |
| `short_name` (fuzzy) | 0.20 | 0.10 | Weak auto/review separation |
| `country_code` (exact) | 0.20 | -- | Remove; redundant with blocking |
| `lei` (exact) | 0.05 | 0.25 | Strong when present; underweighted |

**3. Consider raising `review_floor` to 0.65--0.70.** The 0.60--0.70
band contains only 41 records (4.1%) and the per-field analysis shows
these are genuinely low-confidence. Moving them to unmatched would
shrink the review queue without losing real matches.

**4. Run `meld tune` again after each change** to see how the
distribution shifts before committing to a full run.

---

## Iterating on weights

The typical tuning loop is:

1. Run `meld tune` to see the current state
2. Identify which fields have near-zero stddev or flat scores across
   populations
3. Adjust weights in the YAML (reduce or remove weak fields, increase
   strong ones)
4. Run `meld tune` again -- watch whether the dead zone widens and the
   high-score cluster grows
5. Once the distribution looks clean, adjust thresholds to set the
   review queue size you want
6. Run `meld run --dry-run` to confirm the final counts, then drop
   `--dry-run`

If the distribution remains flat after reweighting, the underlying
fields may simply not have enough signal for confident automated
matching -- you may need to add a new field (e.g. a postcode, a
registration number) or switch the method on a field (e.g. from `fuzzy`
to `embedding` for a name field with many abbreviations).

---

## Multi-field blocking

Blocking supports multiple fields combined with AND or OR logic:

```yaml
blocking:
  enabled: true
  operator: or          # "and" | "or"
  fields:
    - field_a: country_code
      field_b: domicile
    - field_a: lei
      field_b: lei_code
```

**AND** -- a B record only reaches A candidates that match on *all*
blocking fields. Tightest filtering, fastest runtime, highest recall
risk if any blocking field is noisy.

**OR** -- a B record reaches A candidates that match on *any* blocking
field. Larger candidate sets, slower, but recovers records where one
blocking field is wrong or missing.

If a single-field AND config is blocking out records you expect to
match, adding a second field with OR logic is the recommended fix -- it
gives the engine a second route to find the correct A record without
disabling blocking entirely.

To disable blocking altogether, set `enabled: false` or omit the
`blocking` section. Every record will then be considered as a candidate,
which is thorough but slower on large datasets.

---

## Disabling the candidate pass

The candidate pass is the fuzzy pre-filter that picks the top N A
records per B record before the full scoring pipeline runs. It is the
right default for most datasets, but there are situations where it is
better to skip it:

- **Highly selective blocking** -- if your blocking key is a unique or
  near-unique identifier (LEI, registration number, account ID), each B
  record already maps to a tiny A-candidate pool. Running a fuzzy top-N
  pass on a pool of 1--3 records adds overhead with no benefit.
- **Small datasets** -- when A contains a few hundred records and
  blocking is enabled, the blocked pool may be 10--20 records. Scoring
  all of them directly is faster.
- **Exact-only scoring** -- if all match fields use `method: exact`, the
  candidate pass (which is fuzzy-based) has no role in quality and can
  be skipped.

```yaml
candidates:
  enabled: false
```

When `enabled: false`, melder skips the fuzzy candidate search and
treats every A record that survives blocking as a candidate. The
`scorer` and `n` settings are ignored. All scoring phases (exact, fuzzy,
embedding, composite, classification) run normally on the full blocked
pool.

**Warning:** if blocking is also disabled (or the blocking pool is very
large), disabling candidates means a full O(|A|x|B|) cross-product of
scoring work. On a 10,000x10,000 dataset without blocking that is 100
million scored pairs -- use with caution. The combination of tight
blocking + disabled candidates is safe and can be faster than loose
blocking + enabled candidates.
