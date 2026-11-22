# Isolated exact scorer probe

Run from repository root: `bash benchmarks/batch/usearch_scaling/exact_probe/run.sh`.
Requires Rust, Python 3 and macOS `/usr/bin/time -l` or Linux GNU time `-v`.
No production changes: `ascii_fast` includes `src/scoring/exact.rs` directly;
`original_446d45e` reproduces that revision's trim / empty-empty / ASCII-equality /
allocating lowercase fallback. Rebuild and revalidate after scorer changes.

The runner builds in a fresh temporary directory (`EXACT_TARGET` overrides).
Defaults: 100,000 records, 32 candidates/query, four passes, three processes/cell.
`EXACT_N`, `EXACT_ROUNDS`, `EXACT_REPEATS`, `EXACT_MODES` (space separated) and
`EXACT_OUT` (CSV path) override defaults. Results include compiler/platform and
per-process RSS in one CSV; alternating repeats reverse method order. The caller
must arrange exclusive timing. For direct pairing use
`EXACT_MODES='original_446d45e ascii_fast'` (same corpus/repeat, not simultaneous).
The output path is overwritten: use a fresh path to preserve previous results.

## Correctness: contextual Unicode, not case folding

Both direct scorers use Rust Unicode `trim()`, reject empty/empty, then accept
`eq_ignore_ascii_case` **even for strings containing non-ASCII**, before Unicode
lowercase fallback. Cache keys here use whole-string `trim().to_lowercase()`.
This is not full case folding, NFKC, or character-wise lowercase.

`verify` tests every pair in each 256-record corpus plus an adversarial matrix
covering ASCII case changes around sigma, cased/case-ignorable marks, punctuation,
Unicode whitespace, dotted I expansion, Kelvin sign and sharp S. All four
alternatives must match the original's individual f64 bits; hash validation also
forces every hash to collide. Every measured shortlist pair is checked before
timing. Examples asserted explicitly:

- `AΣ` / `aΣ`: ASCII bypass accepts; both whole-string lowercase to `aς`.
- `AΣ` / `aς`: Unicode fallback accepts; `AΣ` / `aσ` rejects.
- `AΣ`.chars().flat_map(char::to_lowercase) produces `aσ`, **not** `aς`.
  Thus character-wise normalization is unsafe and excluded.

No disagreement was found for these whole-string keys. The bypass is compatible
with them: ASCII case substitutions preserve Unicode Cased/Case_Ignorable
properties, so they cannot change the Final_Sigma context. The fallback already
is whole-string lowercase equality. This reasoning is specific to Rust's default
Unicode lowercase, not arbitrary locale-sensitive transforms; finite fixtures
are not an exhaustive proof over Unicode versions. ß is not SS; σ is not ς;
İ becomes `i` + combining dot. Do not generalize to ASCII-only caches, full case
folding, per-character lowercasing, or separately normalized chunks.

Normalized mode compares nonempty keys. Hash mode uses DefaultHasher **and checks
key equality** on hash matches. Intern mode uses one dictionary shared across
both sides, with zero reserved for empty; independent ID dictionaries are invalid.
It retains normalized strings; temporary dictionary allocations count toward RSS.

## Workload and recorded evidence

Short ASCII codes, 20-character LEI-shaped identifiers (not validated LEIs), and
long mixed Unicode strings include trim/empty/Unicode fixtures. Each query gets
two neighboring duplicate opportunities and 30 deterministic modular distractors;
most fail, with more collisions in codes. There is no production retrieval,
field dispatch, row lookup, parallelism, mutation, or ingestion.

`prep_s` is cold preparation after raw data creation; `score_s` is warm scoring
only. Validation warms caches and is excluded from both timings. Whole-process
peak RSS includes raw strings, validation and preparation temporaries, not just
retained cache size; independent process peaks cannot reliably be subtracted.
No disk-cache/allocator cold-start guarantee or confidence interval is claimed.

`results/paired_original_ascii.csv`: 18 measurements, three paired repeats per
corpus, 12.8M comparisons each; scheduled after the caller's `micro2-go` grant.

| Corpus | Original median score s | ASCII-fast median score s | Ratio |
|---|---:|---:|---:|
| codes | 1.0396 | .3134 | 3.32× |
| LEI | 1.8336 | .7845 | 2.34× |
| Unicode | 10.2324 | 10.0536 | 1.02× |

`results/historical.csv` consolidates the first 36 measurements and RSS in bytes,
with original filenames and environment provenance. Historical `production`
is that run's scorer snapshot, **not** verified `original_446d45e`. Its medians:

| Corpus | production score s | normalized | verified hash | intern |
|---|---:|---:|---:|---:|
| codes | .2763 | .1085 | .0407 | .0381 |
| LEI | .6721 | .0698 | .0396 | .0379 |
| Unicode | 9.7800 | .3455 | .0425 | .0388 |

Do not infer original-versus-ASCII-fast gains from this historical table.

## Recommendation

Keep the allocation-free ASCII mismatch shortcut as the low-complexity option.
For repeated Unicode-heavy comparisons, investigate whole-string cached keys
first; verified hashes help negative comparisons, while shared intern IDs add
lifecycle complexity for relatively small warm-score gains here. None of these
isolated timings establishes end-to-end gains (see the parent comparison).

Production must precompute **both A and B**, per configured exact field, or charge
query-side normalization once per query; charging only reference preparation
while assuming free query keys overstates benefits. This probe reuses one combined
corpus on both sides, not two independently sized datasets: prep/RSS are not a
production two-pool cost estimate. Include ingestion, retained memory, cache load,
update invalidation, dictionary synchronization, concurrent access and restart
costs, and measure amortization against actual shortlist reuse before adopting.
Stable shared intern IDs especially complicate live mutation; hashes alone are
never sufficient equality evidence. No new timed runs were made for this audit.
