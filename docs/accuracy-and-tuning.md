← [Back to Index](./) | [Scoring Methods](scoring.md) | [Configuration](configuration.md)

# Accuracy & Tuning

Getting record matching right is an empirical process. You configure
fields, methods, and weights — then measure how well the pipeline
separates true matches from non-matches. This page covers the tools
melder provides for measurement, and walks through a real worked example
showing the journey from a naive configuration to near-perfect accuracy.

## The accuracy problem

Every record-matching pipeline produces a score distribution: true
matches cluster at high scores, non-matches cluster at low scores. The
quality of your configuration is measured by how cleanly these two
populations separate.

In the ideal case, there is a clear gap between the lowest-scoring true
match and the highest-scoring non-match. You place your thresholds in
that gap and every decision is correct. In practice, the two populations
overlap — and that overlap zone is where errors live.

The **overlap coefficient** measures how much the two distributions
intersect: 0.0 means perfect separation, 1.0 means identical
distributions. This single number tells you how much room for
improvement exists.

## Using `meld tune`

The `meld tune` command runs the full batch pipeline and produces a
diagnostic report showing how well your configuration separates true
matches from non-matches.

```bash
meld tune --config config.yaml
```

### Ground truth with `common_id_field`

If both datasets share a business identifier (LEI, DUNS number, internal
ID), configure it as `common_id_field` on both datasets:

```yaml
datasets:
  a:
    path: entities.csv
    id_field: entity_id
    common_id_field: lei         # field in A containing the shared ID
  b:
    path: counterparties.csv
    id_field: counterparty_id
    common_id_field: lei_code    # field in B containing the same shared ID
```

When configured, `meld tune` uses this as ground truth: B records whose
`common_id_field` value matches an A record's value are **expected to
match** (labelled "with common_id"). B records with no matching value
are **not expected to match** (labelled "no common_id"). This enables
two-population analysis, overlap measurement, and accuracy metrics.

Without `common_id_field`, tune still works — you get a single-population
histogram and per-field statistics, but no overlap coefficient or
accuracy metrics.

### The report: with ground truth

Here is the full output from a tuned configuration (10k x 10k
counterparty matching with fine-tuned embeddings, BM25, and synonym
matching), zoomed into the overlap zone:

```bash
meld tune --config config.yaml --bucket-width 0.01 --min-score 0.52 --max-score 0.72
```

#### Score distribution

```
=== Score Distribution ===

  █ known match  ░ known non-match

  0.52 ░░░░░  ▼ review_floor (0.52)
  0.53 ░░░
  0.54 ░░░░░
  0.55 
  0.56 ░░
  0.57 
  0.58 ░░░
  0.59 
  0.60 
  0.61 ██
  0.62 ███░░
  0.63 █████
  0.64 ████████  ▼ auto_match (0.64)
  0.65 ████████████
  0.66 ██░░
  0.67 █████████████
  0.68 ████████████████████
  0.69 ██████████████████
  0.70 ███████████████████████
  0.71 ████████████████████████████████████████████████
  0.72 ██████████████████████████████████████████████████

  auto_match    >= 0.64:   7,037 (70.4%)
  review_floor  >= 0.52:      18 (0.2%)
  no_match       < 0.52:   2,945 (29.4%)
  total:                10,000
```

The histogram shows two populations: `█` for records with a common_id
match (expected to match) and `░` for records without (not expected to
match). Threshold positions are annotated with `▼`.

**What to look for:**

- **Clean gap between populations.** In this example, non-matches (`░`)
  cluster at 0.52-0.58 and matches (`█`) start at 0.61 — a clean gap.
  Your thresholds should sit in this gap.
- **Mixed buckets.** Buckets containing both `█` and `░` (like 0.62 and
  0.66) are where classification errors occur. If a bucket has `█░░░░`,
  the matches there are outnumbered by non-matches — human review is
  needed. If it has `████░`, the non-matches are rare — auto-matching
  is safe.
- **Threshold placement.** The review_floor should be below where
  matches start appearing. The auto_match should be above where
  non-matches stop appearing. The counts below the histogram tell you
  the practical impact: 70.4% auto-matched, 0.2% to review, 29.4%
  discarded.

**Zooming in.** Use `--bucket-width`, `--min-score`, and `--max-score`
to zoom into the overlap zone for fine-grained threshold placement:

```bash
# Wide view (default)
meld tune --config config.yaml

# Zoom into overlap zone
meld tune --config config.yaml --no-run --bucket-width 0.01 --min-score 0.50 --max-score 0.70
```

The `--no-run` flag skips the pipeline and re-analyses cached output
files — instant, so you can iterate on display parameters.

#### Overlap coefficient

```
  Overlap: 0.0007  (0 = perfect separation, 1 = identical)
```

A single number summarising how much the two populations overlap. Use
this to track improvement across configuration changes: lower is better.
A value below 0.01 indicates near-perfect separation.

#### Ground-truth accuracy

```
=== Ground-Truth Accuracy ===

  Auto-matched:      7,037
    Correct (TP):    7,036
    Incorrect (FP):      1
  Review:               18
    Correct (TP):        6
    Incorrect (FP):     12
  Missed (FN):           0

  Precision:         100.0%
  Recall (auto):      99.9%
  Combined recall:   100.0%
```

**What to look for:**

- **Incorrect (FP) in Auto-matched** — these are records that were
  auto-confirmed but whose common_id doesn't match. Zero or near-zero
  is essential. If this number is high, your auto_match threshold is
  too low.
- **Missed (FN)** — records that have a common_id match but scored
  below review_floor. These are lost matches. If this is non-zero,
  your review_floor is too high, or you need a better scoring method
  for the problematic record types (see the Overlap Zone section).
- **Combined recall** — the percentage of expected matches that ended
  up in either auto-match or review. 100% means nothing was missed.
- **Review queue size** — 18 records (0.2%) is the human review
  workload. Trade off against recall: lowering review_floor catches
  more edge cases but grows the queue.

#### Per-field analysis

```
=== Per-Field Analysis ===

  bm25_bm25
                                                  Min    Max   Mean Median StdDev
    All:                                        0.256  1.000  0.933  0.985  0.097
    With common_id (expect to match):           0.462  1.000  0.934  0.985  0.094
    No common_id (don't expect to match):       0.256  0.493  0.388  0.386  0.064
    Mean gap: 0.546 (strong separation)

  legal_name_counterparty_name
                                                  Min    Max   Mean Median StdDev
    All:                                        0.000  1.000  0.463  0.419  0.457
    With common_id (expect to match):           0.000  1.000  0.464  0.420  0.458
    No common_id (don't expect to match):       0.000  0.835  0.285  0.195  0.296
    Mean gap: 0.178 (strong separation)

  registered_address_counterparty_address
                                                  Min    Max   Mean Median StdDev
    All:                                        0.471  1.000  0.975  1.000  0.043
    With common_id (expect to match):           0.634  1.000  0.976  1.000  0.041
    No common_id (don't expect to match):       0.471  0.929  0.687  0.660  0.127
    Mean gap: 0.289 (strong separation)
```

Each field is broken down by population. The **Mean gap** tells you how
well that field separates matches from non-matches.

**What to look for:**

- **Weak separation** (gap < 0.05) — the field scores similarly for
  matches and non-matches. It's consuming weight budget without
  contributing to discrimination. Consider reducing its weight or
  removing it.
- **Strong separation** (gap > 0.15) — the field is earning its weight.
  In this example, BM25 has the strongest separation (0.546) — it
  scores 0.93 for matches but only 0.39 for non-matches.
- **High StdDev on non-matches** — the non-match population is spread
  out, meaning some non-matches score dangerously high on this field.
  That's where false positives come from. In this example, the name
  field has StdDev 0.296 for non-matches — some non-matching names
  score up to 0.835.
- **Near-zero StdDev** — the field scores identically for every pair.
  This usually means you're scoring a blocking field (e.g. country_code
  when blocking is already filtering on country). Remove it and
  redistribute the weight.

#### Overlap zone

```
=== Overlap Zone (0.52 - 0.64) ===

  With common_id: expect to match (6 records in overlap):

    CP-008957 -> ENT-008957  score: 0.619
      scores: legal_name_counterparty_name: 0.35  registered_address: 0.81  bm25: 0.64
      query:  SP  |  1862 CT.NEY SQUARES DICKERSONPORT, NE...
      match:  Stevens-Hunter PLC  |  1862 Courtney Squares, Dickersonport,...

    CP-007587 -> ENT-007587  score: 0.630
      scores: legal_name_counterparty_name: 0.30  registered_address: 0.85  bm25: 0.67
      query:  GH  |  USS Hoffman, FPO AP
      match:  Gibson-Edwards Holdings  |  USS Hoffman, FPO AP 52057
    ...

  No common_id: don't expect to match (12 records in overlap):

    CP-007215 -> ENT-003217  score: 0.626
      scores: legal_name_counterparty_name: 0.51  registered_address: 0.85  bm25: 0.45
      query:  Jones-Morales & Co  |  USS Williams, FPO AE 77381
      match:  Hall, Parks and Lee Partners  |  USS Williams, FPO AE 64653

    CP-000256 -> ENT-005913  score: 0.563
      scores: legal_name_counterparty_name: 0.70  registered_address: 0.57  bm25: 0.42
      query:  Smith LLC Capital  |  PSC 7332, Box 4437, APO AP 49346
      match:  Henderson LLC Capital  |  PSC 9233, Box 8335, APO AP 67296
    ...
```

This is the most actionable section. It shows the actual records in the
overlap zone — the danger area between review_floor and auto_match where
classification errors occur.

For each record you see: the composite score, per-field score breakdown,
the actual field values from the query (B) record, and what it matched
against (the A record). This tells you exactly *why* each record scored
where it did.

**What to look for:**

- **"With common_id" records** are true matches that scored too low for
  auto-match. In this example, all 6 are **two-letter acronyms** ("SP",
  "GH", "KL") — the B side has an abbreviation so short that synonym
  matching's min_length=3 filter excludes it. The name scores are
  0.30-0.51 while address scores are high (0.81-0.93). Action: these
  are inherently difficult cases; lowering the synonym min_length or
  adding them to a synonym dictionary CSV would help.

- **"No common_id" records** are non-matches that scored too high. In
  this example, two patterns are visible: (1) **shared military
  addresses** ("USS Williams, FPO AE") — different entities stationed
  at the same base score high on address; (2) **common-word names**
  ("Smith LLC Capital" vs "Henderson LLC Capital") — shared suffixes
  inflate the name score. Action: increase BM25 weight to penalise
  common words, or add address fields to the blocking filter to prevent
  cross-base matches.

Use `--overlap-limit` to control how many records are shown per
population (default: 5).

### The report: without ground truth

Without `common_id_field`, `meld tune` produces a simpler report. You
see the score distribution as a single population — all scored pairs
together — plus per-field statistics and threshold counts.

```
  NOTE: no common_id_field in config. Add common_id_field to both datasets
  for two-population analysis, overlap coefficient, and accuracy metrics.

=== Score Distribution ===

  █ scored pairs

  0.24 
  0.28 █
  0.32 ██████
  0.36 ███████████████████
  0.40 █████████████████████████
  0.44 ████████████████
  0.48 ████████
  0.52 ███  ▼ review_floor (0.52)
  0.56 █
  0.60 ██
  0.64 ██████  ▼ auto_match (0.64)
  0.68 ███████
  0.72 ███
  0.76 ██
  0.80 ██████
  0.84 ████████████
  0.88 ███████████████████
  0.92 █████████████████████████████████
  0.96 ██████████████████████████████████████████████████
  1.00 █████████████████████████████████████████████

  auto_match    >= 0.64:   6,953 (69.5%)
  review_floor  >= 0.52:     221 (2.2%)
  no_match       < 0.52:   2,826 (28.3%)
  total:                10,000
```

Without ground truth, you cannot distinguish true matches from false
positives within a score bucket — a cluster at 0.70 might be all
matches, all non-matches, or a mix. The histogram is still useful for
threshold placement (look for gaps or thin regions between clusters),
but you're working blind to accuracy.

If at all possible, provide a `common_id_field` — even a partial one
that covers only 30% of records — to unlock the two-population analysis.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--bucket-width` | 0.04 | Width of each histogram bucket |
| `--min-score` | auto | Lower bound of display range |
| `--max-score` | auto | Upper bound of display range |
| `--bar-width` | 50 | Maximum bar width in characters |
| `--overlap-limit` | 5 | Records shown per population in overlap zone |
| `--no-run` | off | Skip pipeline, re-analyse cached output (instant) |

### The tuning loop

1. Run `meld tune` to see the current state
2. Look at the histogram — are the two populations separated?
3. Check per-field analysis — which fields have weak separation?
4. Adjust weights (reduce weak fields, increase strong ones)
5. Run `meld tune` again — watch the overlap coefficient
6. Inspect the overlap zone — what's causing the remaining errors?
7. Adjust thresholds to balance auto-match precision vs review volume
8. Run `meld run --dry-run` to confirm counts, then drop `--dry-run`

### Blocking and match field interaction

Blocking is the single most important performance and accuracy
trade-off in the melder. When enabled, it eliminates candidates that
don't share a blocking key value before any scoring runs — typically
removing 95%+ of pairs. This gives an order-of-magnitude speedup
(10× in measured benchmarks at 100k scale), but every record excluded
by blocking is **permanently unreachable**: if the blocking key is
wrong, missing, or inconsistent on either side, the true match will
never be found. Disabling blocking removes this ceiling entirely, but
scoring throughput drops from ~8,500 rec/s to ~800 rec/s at 100k.
There is no free lunch — choose your blocking fields carefully.

Because blocking quality depends entirely on key quality, normalising
blocking fields before they reach the melder is one of the
highest-value things you can do. Country codes are a common example:
one dataset might use `GB` while the other uses `UK`, `GBR`, or
`United Kingdom`. A simple lookup table applied during data
preparation eliminates this mismatch at zero runtime cost. The same
applies to currency codes, sector classifications, or any categorical
field used for blocking — clean the key once, benefit on every run.

A common pitfall: using the same field for both blocking and scoring.
If you block on `country_code`, every candidate pair already has a
matching country — so an `exact` match field on country will score 1.0
for every pair, contributing zero information while consuming weight
budget. Remove it and redistribute the weight to fields that actually
discriminate.

### Multi-field blocking

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

**AND** — a B record only reaches A candidates that match on *all*
blocking fields. Tightest filtering, fastest runtime, highest recall
risk if any blocking field is noisy.

**OR** — a B record reaches A candidates that match on *any* blocking
field. Larger candidate sets, slower, but recovers records where one
blocking field is wrong or missing.

---

## Worked example: counterparty matching

What follows is a real case study. We matched 10,000 entities (A)
against 10,000 counterparties (B) using a synthetic but realistic
dataset with the kinds of data quality problems you encounter in
production: abbreviations, misspellings, truncations, acronyms, and
records that look similar but refer to different real-world entities.

### The messy data problem

Before diving into results, it helps to understand what "messy data"
actually looks like. Here are real pairs from the dataset, grouped by
the type of challenge they represent.

**Clean matches — the easy cases.** Some pairs are straightforward.
The names and addresses are nearly identical, differing only in
casing or minor formatting:

| A (entity master) | B (counterparty) | Challenge |
|---|---|---|
| Bolton, Brown and Perez Capital | bolton, brown and perez capital | Casing only |

These score 0.97+ and every method handles them correctly.

**Abbreviations and truncations — the bread and butter.** Most real
datasets have systematic differences in how names are recorded.
One system spells out "Limited", the other writes "Ltd". One
truncates after 30 characters:

| A (entity master) | B (counterparty) | Challenge |
|---|---|---|
| Anderson LLC AB | Anderson L.L.C. AB | Punctuated abbreviation |
| Walker-White Partners | Walker-White Ptnrs | Suffix abbreviation |
| Beck-Russell Capital | Beck-Russell cap | Truncated suffix |
| Lutz and Sons Holdings | Lutz and Sons hldgs | Informal abbreviation |
| Melendez, Martinez and Owen SAS | Melendez, Martinez and Owen S.A.S | Dotted legal form |
| Butler LLC & Co | Butler L.L.C. | Missing suffix entirely |

Embeddings handle these well (scores 0.80-0.85) because the semantic
meaning is preserved. Fuzzy matching also works — the edit distance
is small relative to the string length.

**Address noise — numbers transposed, formats differ.** Addresses
are particularly messy. Suite numbers get dropped, street types are
abbreviated, zip codes go missing:

| A address | B address | Challenge |
|---|---|---|
| 3287 Scott Island Suite 923, New Emilyhaven, MN 47866 | 3287 scott island ste 923, new emilyhaven, mn | Casing + "Suite"→"ste" + zip dropped |
| 296 Mark Knoll, West Madison, ME 15023 | 296 Mark Knoll, W. Madison, ME | "West"→"W." + zip dropped |
| 383 Benjamin Wells Suite 651, North Lauraton, KS 45593 | 338 Benjamin Wells Ste 651, North Lauraton, KS 45593 | Transposed digits (383→338) |
| 3177 Mendoza Squares Suite 883, North Kevinburgh, ME 94958 | 3717 Mendoza Squares North Kevinburgh, ME 94958 | Transposed (3177→3717) + "Suite 883" dropped |

These are challenging because transposed digits are invisible to
embeddings but meaningful to exact matching. The melder handles this
by using embeddings for the name field and a separate address field,
so address noise affects only part of the composite score.

**Acronyms — the blind spot.** The hardest category. One system
stores the full legal name, the other stores an acronym that no
scoring method can resolve through similarity alone:

| A (entity master) | B (counterparty) | Score without synonym |
|---|---|---|
| Jones, Duncan and Bentley Inc | JDBI | 0.56 |
| Kidd, Gomez and Thomas SAS | KGTS | 0.59 |
| Wood, Santana and Boyd Holdings | WSBH | 0.52 |
| Roberts PLC SA | RP | 0.44 |
| Barber-Hernandez GmbH | BG | 0.42 |

There is zero character overlap between "JDBI" and "Jones, Duncan and
Bentley Inc". No embedding model can bridge this gap — the strings
share no semantic signal. These pairs only survive at all because the
address field carries them (addresses are often identical even when
names differ completely). This is the problem that synonym matching
was built to solve.

**False matches — the dangerous cases.** Some pairs look similar but
refer to different entities entirely:

| A (entity master) | B (counterparty) | Why it scores high |
|---|---|---|
| Smith PLC BV | Smith Ltd Corp | Common surname "Smith" |
| Ward-Thomas SRL | Abbott, Moore and Horn SRL | Shared military address (USCGC Lucas) |
| Henderson LLC Capital | Smith LLC Capital | Shared suffix "LLC Capital" |
| Garcia, Miller and Richards & Co | Ward, Miller and Ross & Co | Shared middle name "Miller" + same suffix |

These are the records that pollute the review queue. They score
0.58-0.64 — high enough to clear the review floor, low enough to
never auto-match. The common-word problem ("Smith", "LLC",
"Capital", "Holdings") inflates their scores because an untrained
embedding model treats these words as meaningful when they are
actually noise. BM25's IDF weighting directly addresses this.

### Starting point: off-the-shelf embeddings

We started with the default `bge-base-en-v1.5` model, embedding
similarity on name and address fields, country blocking, and thresholds
at `auto_match=0.88`, `review_floor=0.60`.

The score distribution told the story immediately:

```
  R0 (untrained) — overlap: 0.165

  0.56 █░
  0.60 █░░░                                ▼ review_floor
  0.64 ██████░░░░░░░░░░░░░░░░░░░░░░░░
  0.68 ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.72 ███████████░░░░░░░░░░░░░░░░░░░░░░░
  0.76 ██░░░░░░░░
  0.80 ██░░

  █ = true match    ░ = non-match
```

Both populations were jammed into the 0.64-0.80 range with massive
overlap (coefficient 0.165). The model saw everything as similarly
similar — "Smith PLC BV" scored almost as high against "Smith Ltd Corp"
(a false match) as "Bolton, Brown and Perez Capital" scored against
its true counterpart. The review queue contained 4,316 records, of
which 2,958 were non-matches — human reviewers would spend 69% of
their time rejecting false positives.

### Fine-tuning the embedding model

General-purpose embedding models are trained on web text, not entity
names. They don't know that "Holdings" and "Capital" are noise words in
a corporate context, or that "Garcia, Garcia and Weaver PLC" is a
completely different entity from "Garcia Group AG" despite sharing a
surname. Fine-tuning teaches the model your domain.

The melder's own crossmap output provides the training data: confirmed
matches become positive pairs, and in-batch negatives from MNRL
(Multiple Negatives Ranking Loss) provide the contrastive signal. We
used LoRA (Low-Rank Adaptation) to update only ~1% of model parameters,
which prevents catastrophic forgetting — the model retains its
general language understanding while learning domain-specific patterns.

The key finding: **small models plateau**. BGE-small (33M parameters)
reached an overlap floor of 0.081 regardless of training strategy.
BGE-base (110M parameters, 768 dimensions) broke through to 0.046
after 17 rounds of progressive LoRA training. The extra capacity lets
the model form more nuanced representations of entity names.

After fine-tuning (BGE-base, LoRA, 17 rounds, batch size 128):

```
  R17 (fine-tuned) — overlap: 0.046

  0.32 ░
  0.36 ░░
  0.40 ░░░░░░░░░░░░░░░
  0.44 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.48 █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0.52 █░░░░░░░░░░░░░░░░░░░░░░░░
  0.56 █████░░░░░░░░░
  0.60 ██████████████░░░               ▼ review_floor
  0.64 ██████████████████░
  0.68 ██████████░
  0.72 ███
  0.76 █████
  0.80 ███████
```

The two populations pulled apart dramatically. Non-matches shifted
down to the 0.36-0.52 range; true matches concentrated at 0.60+. The
overlap coefficient dropped 72% (0.165 → 0.046). But look at the
0.48-0.60 zone — there is still a thin neck where the populations
meet. Zooming into that zone revealed two distinct problems:

1. **High-scoring non-matches** at 0.58-0.65: pairs like "Smith PLC BV"
   vs "Smith Ltd Corp" where common surnames and legal suffixes inflate
   the embedding similarity.
2. **Low-scoring true matches** at 0.48-0.56: exclusively acronym cases
   like "JDBI" vs "Jones, Duncan and Bentley Inc" where the address
   field alone isn't enough to pull them above 0.60.

These are fundamentally different problems requiring different solutions.

### Adding BM25

BM25 (Best Matching 25) scores how many words two records share,
weighted by how rare each word is across the entire dataset. "Smith"
appears in hundreds of records, so BM25 gives it almost no weight.
"Stellantis" appears once, so it carries enormous weight. This is
exactly the tool needed for problem #1 — the common-word false matches.

We added BM25 at 20% weight alongside the embeddings:

```
  + BM25 at 20% — overlap: 0.005

  0.32 ░░░░
  0.34 ░░░░░░░░
  0.36 ░░░░░░░░░░░░░░░
  0.38 ░░░░░░░░░░░░░░░░
  0.40 ░░░░░░░░░░░░░░░░░
  0.42 ░░░░░░░░░░░░░░
  0.44 ░░░░░░░░
  0.46 █░░░░░░
  0.48 █░░░
  0.50 █░░
  0.52 █░
  0.56 █░
  0.58 █░
  0.60 ██░                                 ▼ review_floor
  0.64 ███████
  0.66 ████████
  0.68 ██████
  0.70 █████
  0.72 ███
  0.80 ███
  0.84 █████
  0.88 █████████                           ▼ auto_match
  0.92 ███████████████████
  0.96 ███████████████████████████████
  0.98 ████████████████████████████████████████████████████████████
```

The effect was surgical. BM25's IDF weighting pushed down exactly the
false matches that were polluting the overlap zone — pairs sharing
"Smith", "Holdings", "Capital", and similar common words. The overlap
coefficient dropped from 0.046 to 0.005 — a 89% reduction.

The review queue cleaned up dramatically: false positives in review
dropped from 84 to just 2. Combined recall actually *improved* slightly
(98.5% → 99.2%) because BM25 helped some borderline true matches that
share distinctive words.

### Adding synonym matching

BM25 solved the common-word false match problem, but the acronym true
matches remained stuck. "JDBI" and "Jones, Duncan and Bentley Inc"
share no words at all — BM25 cannot help any more than embeddings can.

Synonym matching addresses this with a purpose-built mechanism:

1. At startup, generate acronyms from each record's name field (e.g.
   "Jones, Duncan and Bentley Inc" → "JDBI")
2. Build a bidirectional index mapping acronyms to record IDs
3. When scoring, check both directions: is the query name an acronym
   of any indexed name, or vice versa?
4. Score 1.0 if a match is found, 0.0 otherwise
5. Add a flat bonus to the composite (e.g. +0.20)

With synonym matching at weight 0.20, the final distribution:

```
  + Synonym at 0.20 — overlap: 0.003

  0.26 ░
  0.28 ░
  0.30 ░
  0.32 ░░
  0.34 ░░░░░░
  0.36 ░░░░░░░░░░
  0.38 ░░░░░░░░░░░░░░░
  0.40 ░░░░░░░░░░░░░░░░░
  0.42 ░░░░░░░░░░░░░░░░░
  0.44 ░░░░░░░░░░░░░
  0.46 ░░░░░░░░
  0.48 ░░░░░░
  0.50 ░░░░
  0.52 ░░░
  0.54 █░
  0.56 █░
  0.58 █░
  0.60 █░
  0.62 ██░
  0.64 ███
  0.66 █████
  0.68 ████
  0.70 ████
  0.72 ███
  ...
  0.98 ████████████████████████████████████████████████████████████
```

The acronym pairs that were stuck at 0.48-0.56 received the +0.20
boost and moved to 0.68-0.76 — comfortably within the review band.
The non-match population was completely unaffected (synonym doesn't
fire on pairs with no acronym relationship).

With adjusted thresholds (`auto_match=0.64`, `review_floor=0.52`),
tuned to the cleaner separation:

- **100% combined recall** — zero missed matches
- **Zero false positives** in auto-match
- **Review queue of 221** (2.2% of B records) — down from 4,316

### The progression

| Stage | Overlap | Combined recall | Review queue | FP in review |
|---|---|---|---|---|
| Off-the-shelf embeddings | 0.165 | 99.4% | 4,316 | 2,958 |
| Fine-tuned (LoRA, 17 rounds) | 0.046 | 98.5% | 1,652 | 84 |
| + BM25 at 20% | 0.005 | 99.2% | 1,662 | 2 |
| + Synonym at 0.20, tuned thresholds | 0.003 | 100.0% | 221 | 4 |

Each step addressed a specific, identifiable problem:

- **Fine-tuning** taught the model that "Holdings" and "Capital" are
  noise in a corporate context, dramatically improving separation.
- **BM25** used IDF weighting to push down the residual common-word
  false matches that fine-tuning alone couldn't eliminate.
- **Synonym matching** rescued the acronym pairs that no
  similarity-based method can resolve.

---

## Guidelines for your own dataset

The experiments above provide a template. Not every dataset needs every
technique — start simple and add complexity only when measurement tells
you to.

### 1. Start with `meld tune`

Configure your fields, run `meld tune`, and look at the histogram. If
the distribution is bimodal with a clear gap, you may only need to
adjust thresholds. If the populations overlap heavily, read on.

### 2. Check your per-field statistics

Fields with near-zero StdDev are wasting weight. A common cause:
using the same field for both blocking and scoring (the country_code
trap — see the tuning loop section above). Remove or downweight these
fields and redistribute to your strongest discriminator.

### 3. Add BM25 if common words dominate the overlap zone

If your review queue is full of false positives driven by shared
generic terms ("Holdings", "International", "Smith", "LLC"), add
`method: bm25` at 10-20% weight. BM25's IDF weighting selectively
penalises these common terms without affecting distinctive matches.

### 4. Consider fine-tuning for persistent overlap

If the overlap coefficient remains above ~0.05 after weight tuning,
the embedding model may not understand your domain well enough. Fine-
tuning with LoRA is safe (no catastrophic forgetting) and uses your
own crossmap as training data. See [Fine Tuning Embeddings](../vault/ideas/Fine%20Tuning%20Embeddings.md)
for a step-by-step guide.

Key findings from our experiments:
- **Use LoRA**, not full fine-tuning. Full fine-tuning causes
  catastrophic forgetting after 1-2 rounds.
- **Larger models help.** BGE-base (110M params) pushed overlap 27%
  lower than BGE-small (33M params) with the same training strategy.
- **Batch size matters.** Batch 128 outperformed batch 32 for MNRL
  training — more in-batch negatives produce stronger contrastive
  signal.

### 5. Add synonym matching for acronym patterns

If you see true matches stuck at low scores where one side is an
acronym or abbreviation of the other, add `method: synonym`. The
weight determines the flat bonus: 0.10 for a modest boost, 0.20 for
a stronger one. See [Scoring Methods — Synonym](scoring.md#synonym)
for configuration details.

### 6. Use exact prefilter for shared identifiers

If both datasets share a unique identifier (LEI, ISIN, DUNS number),
configure it as an `exact_prefilter`. These pairs are confirmed at
score 1.0 before any scoring runs — roughly 40% of matchable records
in typical datasets. This is the single highest-impact configuration
change you can make if the data supports it.

### 7. Iterate

After each change, run `meld tune` again. Watch the overlap
coefficient, the review queue size, and the false positive count.
When you're satisfied, run `meld run --dry-run` to confirm the
final counts, then drop `--dry-run` for the production run.

---

## Reference

For the full experimental record with detailed per-round metrics,
score distributions, and observations, see
[benchmarks/accuracy/science/experiments.md](../benchmarks/accuracy/science/experiments.md).
