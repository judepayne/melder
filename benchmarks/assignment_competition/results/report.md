# Five-candidate competition experiment

**Result:** wider shortlists substantially help the experiment on the deliberately
constructed repair cases. The required width tracks the escape's actual rank.
At `top_n: 5`, all 300 repair-family queries were correctly matched and identical
across every experimental run. Main averaged 219.4 correct out of those 300.
The WALs confirm **311 repairs that actually used a fifth-ranked alternative**.

This demonstrates the repair mechanism, not a general real-world accuracy claim.
The groups are calibrated synthetic diagnostics with cloned company-name patterns,
concatenated aliases and colliding identifier strings. Ambiguous and unrepairable
controls remain problematic, and the original background still varies.

## What was run

- 10,000 original A records and 10,000 original B records, plus 500 per side:
  100 groups of five A/five B, twenty groups per family.
- Original matching-field values preserved. Added `competition_group` AND-blocking
  isolates each custom group while retaining the original background country pools.
- Real all-MiniLM-L6-v2 embeddings and existing weighted scoring: embedding 0.55,
  short-name partial ratio 0.20, country exact 0.20, identifier exact 0.05;
  auto threshold 0.85, review floor 0.60. No common-ID or exact-prefilter bypass.
- `ann_candidates: 100` throughout; `top_n: 1, 2, 3, 5` on both binaries.
- Twenty seeded B permutations, shared across all eight combinations. Width order
  rotated; binary order alternated. **160 successful measured processes**, sequential,
  eight Rayon threads, same prepared cache bytes, fresh crossmaps and outputs.
- One untimed combined-data preparation/calibration. All measured runs warm, with
  scoring logs disabled. No measured run discarded or repeated.
- Main commit `b9a83a41d8c238569f915531420877a81ab2b0d7`, binary SHA256
  `5aaac6df9ae2095be1f13093863b61987bcef511687cb61ea86b6f74d2abae81`.
- Uncommitted full-shortlist experiment binary SHA256
  `fc395c5100ed5fc4640c763ba4116047e89d66f7837bd34c3e43d118fb8271fe`.
- Same Apple M3/eight cores/16 GiB machine as the previous study. Production Rust
  source, original benchmark data and older experiment evidence were unchanged.

CSV order is not guaranteed processing order. Results reflect fresh-process
HashMap iteration/scheduling as well as shuffled B input. This is a descriptive
first pass; cloned groups and pairwise comparisons are not independent inferential
samples. No population confidence intervals or general determinism claims are made.

## Real-score calibration

The custom-only probe and combined 10,500-record dataset were checked using actual
baseline/ONNX/shared-pipeline scoring logs, not invented vectors or a surrogate
scorer. The combined gate scored all 10,500 B records and verified all 500 custom
queries had exactly five in-group candidates, the declared qualifying edges,
exclusive constrained queries and actual escape ranks 2/3/5. No protected/bypass
claims or baseline reassignments were present.

Custom-only probe scores for flexible repair candidates ranged from 0.868547 to
0.935683. The highest rejected edge was 0.760994; the unrepairable flexible query's
unrelated candidate scored 0.418454. Threshold margins are not borderline.
The combined matrix is retained separately and passed the same graph/rank gate.

Each repair group has one flexible B that qualifies for all five A records and
four constrained B records that each qualify only for their own A. The remaining
A is therefore the flexible query's unique complete one-to-one solution. This
**fixture oracle follows the unique complete feasible assignment**; it is not
independently established real-world identity evidence. Ambiguous groups instead
allow all 120 bijections, not one arbitrary identity labeling.

Ambiguous candidates tie within each query and share identical ranked prefixes.
Tiny score differences between otherwise identical queries are audited separately;
they do not change those prefixes or the threshold graph.

## 1. Increasing width unlocks the intended repairs

Each family contains 100 B queries. Values below are experimental **mean correctly
matched queries out of 100**, averaged over twenty runs.

| Free escape rank | `top_n: 1` | `top_n: 2` | `top_n: 3` | `top_n: 5` |
|---|---:|---:|---:|---:|
| Second | 70.20 | **100** | **100** | **100** |
| Third | 69.70 | 65.90 | **100** | **100** |
| Fifth | 69.25 | 67.10 | 64.80 | **100** |

Below the required width, every group accepts four pairs rather than five. Those
four pairs need not all be correct: a flexible query can occupy another B's true
A, leaving that constrained B unmatched. Consequently 80 accepted matches can
mean only roughly 65–70 correct matches in a 100-query cohort.

Once the escape is shortlisted, the experiment produced the unique correct full
assignment in **every run of every eligible repair group**. The candidate is not
merely present in configuration: serialized WAL replacement ranks show actual use.

## 2. Main versus experiment at width five

Means per 100-query family:

| Family | Main accepted | Experiment accepted | Main correct | Experiment correct | Main mean changed B between runs | Experiment mean changed B |
|---|---:|---:|---:|---:|---:|---:|
| Escape second | 90.00 | 100 | 80.00 | **100** | 20.23 | **0** |
| Escape third | 86.25 | 100 | 72.50 | **100** | 27.65 | **0** |
| Escape fifth | 83.45 | 100 | 66.90 | **100** | 35.58 | **0** |

Each experimental repair-family cohort had **one final mapping outcome across all
twenty runs**, versus twenty on main. Main sometimes gets the full solution
through favorable ordinary claim order; it is not required to fail every time.
Wider lists alone do not remove its first-successful-claim dependence.

Across the measured matrix, there were 1,678 experimental reassignments: 1,455
custom repairs and 223 background repairs. Custom repairs split as 589 rank-two,
555 rank-three and 311 rank-five cases. Independent audit confirmed that every
custom repair replaced an incorrect flexible-query pair with **two fixture-correct
pairs**, adding one accepted pair and two correct pairs. Those are aggregate move
counts across repeated runs, not unique business entities.

The 311 fifth-rank repairs are the direct evidence missing from the original
10k-only benchmark, whose chosen replacements never needed ranks beyond three.

## 3. Controls show why larger lists are not universally better

### Fully ambiguous groups

Both binaries accepted exactly 20, 40, 60 and 100 of the 100 ambiguous B queries
at widths 1, 2, 3 and 5 respectively. All accepted assignments were allowed by
the ambiguity-aware oracle. There were **no local repairs** in this control.

At width five, both still had twenty different mapping outcomes. Mean changed B
between runs was 80.26 on main and 80.34 experimentally. More available candidates
improved coverage in both versions but did not determine which equally plausible
identity pairing to choose. No unique-quality oracle is imposed here.

### No qualifying free escape

Both binaries accepted exactly 80 of the 100 unrepairable-control queries at every
width, and performed **zero repairs** in the control. The fifth A never qualified.

Larger lists increased opportunities for the similar known-negative query to take
someone else's A. At width one, main/experiment averaged 10.70/10.20 such false
acceptances out of twenty known negatives; at width five they averaged 16.30/16.10.
Correct known-positive matches out of eighty decreased from 69.30/69.80 to
63.70/63.90.

Here “unrepairable” means no legal **one-to-two augmentation**, not that no better
quality assignment exists. Replacing a wrong pair with one better pair could help,
but count-preserving takeovers/score-only swaps are outside this implementation.
This control cautions against assuming wider lists must improve accuracy.

## 4. Original background, kept separate

The background had no comparable step-change. Means for its original 10,000 B
records:

| Width | Main accepted | Experiment accepted | Main known-correct | Experiment known-correct |
|---|---:|---:|---:|---:|
| 1 | 5,162.00 | 5,162.00 | 4,517.90 | 4,514.10 |
| 2 | 5,173.60 | 5,177.50 | 4,517.25 | 4,513.30 |
| 3 | 5,174.05 | 5,177.80 | 4,515.80 | 4,515.40 |
| 5 | 5,173.75 | 5,177.75 | 4,513.80 | 4,511.45 |

At width five its mean changed-B count was 460.42/455.31, with twenty different
outcomes on both binaries. Differences are descriptive, not significance claims.
Its 2,000 ambiguous records have no verified alternative-identity oracle and are
not automatically counted as proven errors.

ANN100 and the new frozen graph differ from earlier studies; do not pool or
interpret cross-study timing/accuracy differences as isolated shortlist effects.

## 5. Performance

Full 10,500 × 10,500 run medians; ratios are paired experiment/main:

| Width | Main wall s | Experiment wall s | Median paired wall ratio |
|---|---:|---:|---:|
| 1 | 1.3172 | 1.3190 | 1.0056× |
| 2 | 1.3107 | 1.3243 | 1.0078× |
| 3 | 1.3170 | 1.3279 | 1.0054× |
| 5 | 1.3098 | 1.3327 | 1.0124× |

Observed paired wall overhead was roughly **0.5–1.2%**. Peak process RSS medians
were about 688–691 MiB on main versus 652–653 MiB experimentally. These short warm
measurements include all outliers; no formal performance inference or per-cohort
CPU attribution is claimed. Cache copies, calibration and integrity analysis were
outside subprocess timing.

## Scope and limitations

- The repair families share one five-city asset-manager naming template; twenty
  matching-text clones per family scale the workload. This is not 100 independent
  linguistic examples or representative business-accuracy evidence.
- Flexible names concatenate aliases, and synthetic identifier strings deliberately
  collide. They are not trustworthy unique legal-entity identifiers. Both versions
  receive exactly the same noisy inputs and scoring settings.
- Private five-A blocks guarantee candidate exposure. This tests **retaining and
  using deeper shortlisted alternatives**, not ANN recall in a large crowded block.
  Holding ANN100 fixed isolates shortlist width; it does not prove that increasing
  ANN retrieval from 20 to 100 improves discovery on ordinary data.
- Perfect repair-family mappings are specific to the designed unique full solution.
  Whole-dataset outputs remain variable because background/control cases vary.
- More accepted pairs, more stable pairings and more correct identities are distinct
  objectives. The controls intentionally demonstrate their differences.

## Validation and evidence

Final deep Partner Review was **clean**, with no findings. The review covered
the completed competition packet and did not rerun measurements or tests.

All 160 processes exited successfully, with no skipped B records. Active match/
review/unmatched counts reconciled with all 10,500 queries. Every crossmap was
bijective. An independent parent verifier replayed all **160 WALs to the exact
final CSV mappings**, checked group/country isolation and source/cache/binary
hashes, and independently reproduced all cohort stability means via per-record
frequency counts rather than pairwise distance enumeration.

It also verified every custom repair's old/new truth, qualification score and
actual replacement rank against the real combined calibration matrix. Source
vector/index bytes did not change; only per-run source-fingerprint manifest files
could refresh. Original matching-field values, fixture/data identities and pinned
Rust source/binaries remained unchanged. **26 competition and 28 shared Python
tests passed**; no Rust source changed or new Cargo test run was needed.

- [Complete measurements and normalized assignments](measurements.json.gz).
- [Independent validation](validation.json) and [retained verifier](../verify_results.py).
- [Combined real-score matrix and bypass proof](combined_calibration.json).
- [Generated full cohort tables](generated_summary.md).
- [Execution identities](execution_identity.json), [premeasurement source snapshot](source_snapshot.tar.gz)
  and [prespecified protocol](../protocol.json).
- [Fixture rows/oracles](../fixture.json), [custom probe evidence](../calibration.json),
  [raw custom scoring log](../calibration.scoring_log.ndjson), and [instructions](../README.md).
- Raw measured directories/preparation cache/WALs:
  `/var/folders/g6/zfm6_wzj20g4dq7x42bv4_l40000gn/T/melder-competition-3cj_c9ft/matrix/`.

Durable measurements contain full assignments, fixture and oracle, so descriptive
statistical reanalysis does not require raw execution artifacts. Archive-only
reanalysis was run successfully and reproduced every original summary value.
Exact execution
conditions and original independent WAL replay still require retained temporary
binaries, frozen cache and raw files (and the same model weights); hashes alone
cannot reconstruct an ANN graph after cleanup. Even with those artifacts, process
randomness means exact complete assignment outcomes are not promised.

This is not a sweep from the earlier stock setting of twenty to a larger width.
The dataset now deliberately contains useful late-ranked escapes, whereas chosen
repairs in the earlier corpus stayed within the first three. These results do not
predict a substantial gain from merely raising the original `top_n: 20`.

**Conclusion:** the proposed experiment now directly demonstrates a substantial
benefit when a qualifying free alternative lies deeper in the shortlist. It also
shows where wider lists do not solve ambiguity or accuracy. Nothing was committed
or merged, and this diagnostic result alone is not a deployment recommendation.
