← [Back to README](../README.md) | [Scoring Methods](scoring.md) | [Configuration](configuration.md)

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

The `meld tune` command runs the full batch pipeline without writing
output files and produces a diagnostic report:

```bash
meld tune --config config.yaml
```

### What the report contains

**1. Score distribution.** A histogram showing how scores are
distributed across all scored pairs:

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

A healthy distribution is bimodal — a cluster of high scores (matches)
and a cluster of low scores (non-matches), with a thin band in the
middle. A flat or unimodal distribution means your fields are not
discriminating well.

**2. Per-field statistics.** Min, max, mean, median, and standard
deviation for each match field:

```
=== Per-Field Score Statistics ===

  Field                                      Min    Max   Mean Median StdDev
  --------------------------------------------------------------------------
  country_code_domicile                    1.000  1.000  1.000  1.000  0.000
  legal_name_counterparty_name             0.087  1.000  0.791  0.842  0.193
  lei_lei_code                             0.000  1.000  0.542  1.000  0.498
```

`StdDev` is the key indicator: a field with near-zero standard deviation
scores identically for every pair — it adds no information. A field
with high standard deviation varies across pairs and is doing real work.

**3. Threshold analysis.** How your current thresholds split the scored
pairs:

```
=== Threshold Analysis (current settings) ===

  auto_match  >= 0.85:   569 (56.9%)
  review      >= 0.60:   303 (30.3%)
  no_match     < 0.60:   128 (12.8%) (incl. 27 with no candidates)
```

**4. Suggested thresholds.** Percentile-based starting points — the 90th
percentile for auto_match (only the top 10% auto-confirm) and the 50th
for review_floor (roughly half go to review). These are starting points;
adjust based on your tolerance for review volume vs missed matches.

### The tuning loop

1. Run `meld tune` to see the current state
2. Identify fields with near-zero StdDev — their weight is wasted
3. Adjust weights (reduce weak fields, increase strong ones)
4. Run `meld tune` again — watch whether the dead zone widens
5. Adjust thresholds to set the review queue size you want
6. Run `meld run --dry-run` to confirm counts, then drop `--dry-run`

### Blocking and match field interaction

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
