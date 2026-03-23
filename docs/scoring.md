← [Back to README](../README.md) | [Configuration](configuration.md) | [Accuracy & Tuning](accuracy-and-tuning.md)

# Scoring Methods

The melder supports six comparison methods. Each one takes two field
values and returns a score between 0.0 and 1.0.

## Exact

Binary string equality. Case-insensitive (ASCII fast path, full Unicode
fallback). Returns 1.0 on match, 0.0 otherwise. Both fields empty
returns 0.0 — absence of data is not evidence of a match.

**Best for:** identifiers, codes, and categorical fields where partial
similarity is meaningless — country codes, LEIs, ISINs, currency codes.

**Trade-off:** zero tolerance for typos or formatting differences. Very
fast — single string comparison with no allocation.

## Fuzzy

Edit-distance-based string similarity, built on normalised Levenshtein
distance. Four scorers are available, selected via the `scorer` config
field:

| Scorer | What it does | Good at |
|--------|-------------|---------|
| `ratio` | Normalised Levenshtein similarity | General string comparison |
| `partial_ratio` | Best substring alignment (sliding window) | Short names within longer strings |
| `token_sort_ratio` | Sort tokens alphabetically, then ratio | Reordered words ("Goldman Sachs" vs "Sachs Goldman") |
| `wratio` (default) | Max of ratio, partial_ratio, token_sort | Robust catch-all when you don't know the error pattern |

All scorers normalise input (lowercase, trim whitespace) before
comparison. A score of 1.0 means identical strings after normalisation.

**Best for:** names, descriptions, and free-text fields where
character-level similarity matters. Use `partial_ratio` when short names
may appear within longer legal names. Use `token_sort_ratio` when word
order varies.

**Trade-off:** `wratio` runs all three sub-scorers and takes the max, so
it costs roughly 3x a single scorer — but still sub-millisecond for
typical entity names. Pure edit distance cannot handle synonyms or
abbreviations ("Corp" vs "Corporation", "JPM" vs "J.P. Morgan") — use
embedding for that.

## Embedding

Neural sentence embedding with cosine similarity. Each text value is fed
through a transformer model (default: `all-MiniLM-L6-v2`) that converts
it into a dense numeric vector capturing its semantic meaning. Two texts
about the same entity produce vectors pointing in nearly the same
direction, even if the wording differs completely.

Supported models (configured via `embeddings.model`):

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `all-MiniLM-L6-v2` | 384 | Default. Good balance of speed and quality |
| `all-MiniLM-L12-v2` | 384 | Slightly better quality, ~2x slower |
| `bge-small-en-v1.5` | 384 | BAAI family, English-optimised |
| `bge-base-en-v1.5` | 768 | Higher capacity, ~2x more memory |
| `bge-large-en-v1.5` | 1024 | Highest quality, ~4x slower than MiniLM-L6 |

**Best for:** the primary matching field (usually the entity or company
name). Captures semantic similarity that edit distance cannot: "JP Morgan
Chase" vs "JPMorgan" scores high despite low character overlap. Handles
abbreviations, word reordering, and minor language variations.

**Trade-off:** requires a model download (~90MB, auto-downloaded on first
run). Each ONNX inference takes ~1-3ms per text. Each encoder pool slot
uses ~50-100MB of RAM.

> [!NOTE]
> For domain-specific use cases (e.g. counterparty reconciliation),
> general-purpose models can be fine-tuned on your own matched pairs to
> improve accuracy. The melder's own crossmap output is the training data
> source. See [Fine Tuning Embeddings](../vault/ideas/Fine%20Tuning%20Embeddings.md) for a guide,
> and [Accuracy & Tuning](accuracy-and-tuning.md) for a worked example showing
> the impact of fine-tuning.

## BM25

IDF-weighted token overlap across indexed text fields. BM25 scores how
many words two records share, weighted by how rare each word is. Common
words like "Holdings", "International", and "Group" contribute almost
nothing; distinctive words like "Stellantis" or "Berkshire" contribute
heavily.

Specify which fields BM25 indexes via the inline `fields` key on the
BM25 match_fields entry. When omitted, fields are derived automatically
from your fuzzy/embedding match fields.

```yaml
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.80
  - method: bm25
    weight: 0.20
    fields:
      - field_a: legal_name
        field_b: counterparty_name
      - field_a: registered_address
        field_b: counterparty_address
```

**Best for:** suppressing common-word noise from untrained embedding
models. Also works as the sole candidate filter when no embedding fields
are configured (fast start, no ONNX model, no vector index).

**Trade-off:** requires `cargo build --release --features bm25`. Adds
a Tantivy index build at startup (~25ms for 10k records). Cannot
understand meaning — "JPM" and "J.P. Morgan" share no tokens.

## Synonym

Acronym and abbreviation matching. Detects when one side uses an acronym
(e.g. "HWAG") and the other uses the full name ("Harris, Watkins and
Goodwin BV"). No other scoring method can bridge this gap — there is no
character overlap and no semantic similarity between an acronym and its
expansion.

### How it works

1. **Index build.** At startup, the melder generates acronyms from each
   record's configured name fields. For "Harris, Watkins and Goodwin BV",
   it generates "HWAG" (initial letters of significant words, skipping
   common suffixes like BV, Ltd, GmbH). These acronyms are stored in a
   bidirectional HashMap index — both name-to-acronym and acronym-to-name
   lookups are supported.

2. **Candidate generation.** When scoring a B record, the synonym index
   is queried in both directions: is the B name an acronym of any indexed
   A name? Is any A name an acronym of the B name? Matches are added to
   the candidate pool alongside ANN and BM25 candidates.

3. **Scoring.** The synonym scorer is binary: 1.0 if a synonym
   relationship exists between the pair, 0.0 otherwise.

### Additive weight semantics

Unlike other scoring methods, synonym weight is **not** included in the
normalisation denominator. This makes it a flat additive bonus:

- **Synonym fires** (score = 1.0): composite = `baseline_score + synonym_weight`
- **Synonym doesn't fire** (score = 0.0): composite = `baseline_score` (unchanged)

The composite is clamped to [0.0, 1.0] after the addition.

This design is necessary because synonym is a sparse binary scorer — it
fires on perhaps 1% of pairs. If its weight participated in
normalisation, it would dilute the embedding signal for the 99% of pairs
where it contributes nothing.

As a consequence, synonym weight is **not** counted toward the
weights-must-sum-to-1.0 validation. You add it on top of your existing
weights:

```yaml
match_fields:
  - field_a: legal_name
    field_b: counterparty_name
    method: embedding
    weight: 0.48

  - field_a: registered_address
    field_b: counterparty_address
    method: embedding
    weight: 0.32

  - method: bm25
    weight: 0.20                   # embedding + bm25 = 1.0

  - field_a: legal_name
    field_b: counterparty_name
    method: synonym
    weight: 0.20                   # additive bonus, not counted in 1.0
```

With `weight: 0.20`, an acronym match adds a flat +0.20 to the composite
score. A record scoring 0.55 without synonym matching would score 0.75
with it — enough to move from borderline to confident review.

**Best for:** datasets where one side uses abbreviated or acronym names.
Common in counterparty reconciliation, where internal systems may store
"HSBC" while the reference master has "Hongkong and Shanghai Banking
Corporation".

**Trade-off:** the acronym generator uses a simple heuristic
(initial letters of significant words). It will produce false positives
for short, common acronyms. Blocking on country code or another
categorical field helps constrain these. Minimum acronym length is 3
characters.

### Synonym dictionary

The auto-generated acronym index covers many cases, but some equivalences
cannot be derived algorithmically — "HSBC" is not an acronym of "Hongkong
and Shanghai Banking Corporation" by initial-letter rules. For these, you
can provide a **synonym dictionary**: a CSV file where each row lists
terms that should be treated as equivalent.

```yaml
synonym_dictionary:
  path: data/synonyms.csv
```

The CSV has no header row. Each row is an equivalence group:

```csv
HSBC,Hongkong and Shanghai Banking Corporation,Hong Kong Shanghai Bank
JPM,JP Morgan,JPMorgan Chase,J.P. Morgan & Co
IBM,International Business Machines
DB,Deutsche Bank AG,Deutsche Bank
```

Rules:

- **Variable columns.** Rows can have 2, 3, 4, or more terms.
- **Minimum 2 terms per row.** Single-term rows are skipped with a warning.
- **Case-insensitive.** All terms are normalised to uppercase internally.
- **Whitespace trimmed.** Leading/trailing whitespace on each term is stripped.
- **Transitive merging.** If row 1 has `A,B` and row 2 has `B,C`, then
  A, B, and C are all treated as equivalent.
- **Bidirectional.** Looking up any term in a group returns all other
  terms. Both candidate generation and scoring use the dictionary.

The dictionary supplements (not replaces) the acronym generator. A pair
can match via acronym generation, dictionary equivalence, or both.

## Numeric

Numeric equality. Parses both values as floating-point numbers and
returns 1.0 if equal (within machine epsilon), 0.0 otherwise. No range
or tolerance matching — this is currently a stub. Use `exact` for
numeric identifiers. A future version may add tolerance-based comparison.
