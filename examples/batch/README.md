# Batch Mode Worked Example

Batch mode is **asymmetric**: the A side is a pre-indexed reference set
and each B record is scored against it one by one. A records are encoded
once into a vector index (and optionally cached to disk); B records are
encoded on the fly and compared to that index. This means the roles of A
and B are not interchangeable -- A is the stable reference, B is the
incoming data you want to match against it.

This example matches 10 reference entities (A, in `reference.csv`)
against 25 incoming counterparty records (B, in `incoming.csv`) in a
single batch run. It walks you through validation, running, and
interpreting the results.

## Prerequisites

Build melder from the project root:

```bash
cargo build --release
```

All commands below assume you are in the **project root** directory
(the one containing `Cargo.toml`).

## The data

Open the two csv files in this directory to get a feel for the data:

- `reference.csv` -- 10 reference entities with fields `entity_id`,
  `legal_name`, `short_name`, `country_code`, `lei`
- `incoming.csv` -- 25 incoming counterparty records with fields
  `counterparty_id`, `counterparty_name`, `domicile`, `lei_code`

The field names on each side are different (e.g. `country_code` vs
`domicile`). The config maps them together.

Some counterparties are obvious matches (e.g. "ACME Corp." / "Acme
Corporation"), some are ambiguous (e.g. "Sakura Holdings" / "Sakura
Financial Group"), and some have no match on the A side at all (e.g.
"Osaka Electronics Co").

---

## Step 1: Validate the config

The config file has a deliberate mistake. Try validating it:

```bash
./target/release/meld validate -c examples/batch/config.yaml
```

You should see an error like:

```
Config error: weights sum to 1.05, expected 1.0
```

Melder requires that all match field weights sum to exactly 1.0. Open
`examples/batch/config.yaml` and find the `lei` match field near the
bottom:

```yaml
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.10      # <-- this is the problem
```

Change `0.10` to `0.05` so the weights sum to 1.0
(0.55 + 0.20 + 0.20 + 0.05 = 1.00):

```yaml
  - field_a: lei
    field_b: lei_code
    method: exact
    weight: 0.05
```

Save the file and validate again:

```bash
./target/release/meld validate -c examples/batch/config.yaml
```

You should see:

```
Config valid: job="batch_example"
  datasets: A=examples/batch/reference.csv, B=examples/batch/incoming.csv
  match_fields: 4 fields
  thresholds: auto_match=0.85, review_floor=0.6
  ...
```

> **What just happened?** `meld validate` parses the YAML, checks that
> all required fields are present, verifies that match methods are valid
> (`exact`, `fuzzy`, `embedding`, `numeric`), confirms that weights sum
> to 1.0, and checks that thresholds are sensible. It does not load
> data or connect to anything -- it is purely a config syntax check.

---

## Step 2: Run the match

```bash
./target/release/meld run -c examples/batch/config.yaml --verbose
```

You will see output like this (timings will vary):

```
Job: batch_example (Worked example — 10 entities vs 25 counterparties)
Datasets: A=examples/batch/reference.csv B=examples/batch/incoming.csv
Thresholds: auto_match=0.85, review_floor=0.6
Initializing encoder pool (model=all-MiniLM-L6-v2, pool_size=2)...
Encoder ready (dim=384), took 0.1s
Loaded dataset A: 10 records in 0.0s
Building A combined embedding index (10 records, dim=384, 1 field(s))...
  A combined index: encoded 10/10
A combined embedding index built: 10 vecs in 0.0s (...)
Saved A combined index cache to examples/batch/cache/a.combined_embedding_XXXXXXXX.index
...
```

Let's break down what happens:

1. **Encoder init** -- the ONNX sentence-transformer model is loaded.
   On the very first run it downloads to `~/.cache/fastembed/` (~30MB).
   Subsequent runs use the cached model.

2. **A index build** -- each A record's `legal_name` is fed through the
   model to produce a 384-dimensional vector. These vectors are saved
   to `examples/batch/cache/a.combined_embedding_XXXXXXXX.index` (the
   hash in the filename encodes the field spec and quantization so the
   cache is automatically invalidated when config changes).

3. **B index build** -- same for B records, saved alongside the A cache.

4. **Blocking** -- melder builds an index on `country_code` / `domicile`.
   Only records in the same country are compared, avoiding wasted work.

5. **Scoring** -- for each B record, melder applies the scoring
   pipeline: blocking filter (same country only), embedding candidate
   selection (top 10 nearest vectors by cosine similarity), then full
   scoring across all four match fields.

6. **Classification** -- each B record's best score is classified:
   - score >= 0.85 --> **auto-match** (written to results.csv, added
     to crossmap)
   - 0.60 <= score < 0.85 --> **review** (written to review.csv)
   - score < 0.60 --> **no match** (written to unmatched.csv)

---

## Step 3: Check the cache

After the run, look at the cache directory:

```bash
ls -lh examples/batch/cache/
```

You should see files like:

```
a.combined_embedding_XXXXXXXX.index          ~15K
a.combined_embedding_XXXXXXXX.manifest       manifest metadata
a.combined_embedding_XXXXXXXX.texthash       per-record text hashes
b.combined_embedding_XXXXXXXX.index          ~40K
b.combined_embedding_XXXXXXXX.manifest
b.combined_embedding_XXXXXXXX.texthash
```

The `.index` files contain the encoded vectors. The `.manifest` and
`.texthash` sidecars enable incremental cache updates -- if you add a
record to the csv and re-run, only the new record is encoded.

Try running again:

```bash
./target/release/meld run -c examples/batch/config.yaml --verbose
```

Notice the log now says "Loaded A combined embedding index from cache"
-- the encoding step is skipped entirely.

> **Why does this matter?** Encoding is the slowest part of a batch
> run. At 10 records the difference is negligible, but at 10,000
> records encoding takes ~8 seconds. Caching drops this to ~5ms.
> This is especially useful when tuning -- you can change thresholds
> or weights and re-run without re-encoding.

---

## Step 4: Examine the outputs

Three csv files are written to `examples/batch/output/`:

### results.csv -- auto-matched pairs

```bash
head -5 examples/batch/output/results.csv
```

Each row is a confirmed match. Key columns:

| Column | Meaning |
|--------|---------|
| `a_id` | The A-side record ID (e.g. `ENT-001`) |
| `b_id` | The B-side record ID (e.g. `CP-001`) |
| `score` | Composite score (0.0 to 1.0) |
| `classification` | Always `auto` in this file |
| `<fieldA>_<fieldB>_score` | One column per match field showing that field's score (e.g. `legal_name_counterparty_name_score`) |

### review.csv -- uncertain matches

```bash
head -5 examples/batch/output/review.csv
```

Same format as results.csv, but these scored between 0.60 and 0.85 --
close enough to be plausible but not confident enough to auto-confirm.
In production, a human would review these and decide.

### unmatched.csv -- no match found

```bash
head -5 examples/batch/output/unmatched.csv
```

B records that scored below 0.60 against all A candidates. These are
counterparties with no corresponding entity on the A side (e.g.
"Osaka Electronics Co" has no match because there is no Japanese
electronics company in the entities file).

---

## Step 5: Check the crossmap

The crossmap tracks confirmed pairs. After the run:

```bash
head examples/batch/output/crossmap.csv
```

You will see rows like:

```
entity_id,counterparty_id
ENT-001,CP-001
ENT-002,CP-003
...
```

These are the auto-matched pairs. On a subsequent run, these B records
would be **skipped** because the crossmap already records their match.

> **The crossmap is the system of record.** Auto-matched pairs are
> written here automatically. Review decisions (accepted or rejected)
> are also recorded here. The crossmap persists across runs -- it is
> the cumulative result of all matching activity.

---

## Step 6: Experiment with thresholds

Try lowering the auto-match threshold to see how the results change.
Edit `examples/batch/config.yaml`:

```yaml
thresholds:
  auto_match: 0.75    # was 0.85
  review_floor: 0.50  # was 0.60
```

Clear the crossmap so all records are re-evaluated:

```bash
rm examples/batch/output/crossmap.csv
```

Run again:

```bash
./target/release/meld run -c examples/batch/config.yaml --verbose
```

Notice that more records are now auto-matched (lower bar) and fewer
fall into review. The B index loads from cache -- no re-encoding.

> **Tip:** Use `meld tune` on labelled data to find optimal thresholds
> automatically. See [TUNE.md](../../TUNE.md) for details.

When you are done experimenting, change the thresholds back to 0.85 /
0.60.

---

## Step 7: Clean up

To remove all cached data and start fresh:

```bash
./target/release/meld cache clear -c examples/batch/config.yaml
rm -f examples/batch/output/*.csv
```

This deletes stale cache files from `cache/` and all output files.
The next run will re-encode everything from scratch. Add `--all` to
`cache clear` to delete the current cache too (by default it only
removes files from old configs).

---

## Config reference

Here is what each section of `config.yaml` does:

| Section | Purpose |
|---------|---------|
| `job` | Metadata -- name and description for your reference |
| `datasets` | Paths to the csv files and the ID column in each |
| `cross_map` | Where to persist confirmed matches |
| `embeddings` | Model name and cache directories |
| `vector_backend` | `flat` (brute-force) or `usearch` (ANN). Flat is fine for small datasets |
| `top_n` | How many candidates to retrieve from the embedding index before full scoring |
| `blocking` | Pre-filter: only compare records sharing the same country |
| `match_fields` | The scoring rules: which fields to compare, how, and how much weight each carries |
| `thresholds` | Score boundaries for auto-match vs review vs no-match |
| `output` | Where to write the three output csv files |
| `performance` | Encoder pool size, ONNX quantization, vector index quantization |

For full configuration documentation, see the main [README](../../README.md).
