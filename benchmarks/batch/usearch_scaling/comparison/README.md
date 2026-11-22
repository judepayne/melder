# Paired warm-cache end-to-end comparison

Requires macOS, Python 3.11+, PyYAML, and both built binaries. No builds or sampling
are performed. Run from the repository root:

```sh
RAYON_NUM_THREADS=20 ORT_NUM_THREADS=2 python3 benchmarks/batch/usearch_scaling/comparison/run.py \
  --seed /absolute/path/to/completed-scaling-experiment \
  --baseline /absolute/path/to/baseline/meld \
  --candidate /absolute/path/to/candidate/meld \
  --out /tmp/melder-paired-new \
  --sizes 10000 100000 250000 500000 1000000 \
  --repeats 3 --variants baseline exact-mixed
```

`--seed` is the required output directory of the original scaling runner. This comparison uses its fixed
`a-SIZE.csv`, `b-10000.csv`, `SIZE/cache`, and actual `SIZE/warm-1/config.yaml`
recipes (including the CPU preparation override). It does not use original full
source datasets or regenerate samples. Baseline preserves the existing recipe;
exact-mixed adds real `lei`/`lei_code` exact scoring at 0.05 and deducts 0.05 from
the BM25 field. Embedding fields, weights, model and index settings stay intact.
This is not an exact-only or synthetic equality workload: normal BM25, synonym,
embedding and retrieval work remains. Existing exact-prefilter behavior remains.
`exact-shortlist` is a separate stress recipe: keep the two embedding fields for real indexed retrieval, assign the remaining 0.50 to exact LEI, use 200 ANN candidates, and omit BM25 and synonym scoring. An exact-only job without retrieval would not exercise the intended comparisons.

Every invocation receives a fresh private cache, using macOS `cp -cR` APFS COW
clones or a full independent copy if cloning fails. No hardlinks, shared writable
cache, crossmap reuse, or seed writes. Cache copy is excluded from timing. Copies
are retained for inspection: plan disk space, especially on non-APFS storage.
New output directories are mandatory. Binary order alternates across successive
pairs (including variant/size boundaries); all runs are serial. Two cache-hit log
messages are mandatory; failures stay in raw results rather than being silently
included as warm observations. No parity assumption is made.

Outputs:

- `metadata.json`: binary/sample SHA-256, seed metadata, arguments, platform,
  thread environment and policies.
- `results.csv`: one row per invocation, scoring throughput from meld, process
  wall seconds and maximum RSS **bytes** from `/usr/bin/time -l`, external elapsed
  time, order, cache status, exit code/errors and artifact location. No aggregation
  or discarded outliers. Inspect errors before summarizing rates.
- `comparisons.jsonl`: paired semantic equality (`null` if either run failed),
  per-file canonical hashes, row counts and columns. Differences are reported,
  not treated as parity or automatically excused.
- `SIZE/VARIANT/REPEAT-POSITION-BINARY/`: exact config, private cache, `meld.log`,
  `time.txt`, crossmap, original semantic output CSVs/logs and `semantics.json`.

Canonical hashes include **every CSV cell**, scores and ranks included. Columns
and rows are sorted, duplicate rows are retained; numeric text is not rounded or
normalized. Only output CSVs are compared, not timestamped operational logs.
Hashes measure exported precision, not hidden full-precision scores. Pre-existing
score ties, ranking ambiguity and assignment choices can produce genuine output
differences even within one binary; compare repeated baseline runs and inspect
CSV differences before attributing a change to the candidate. Hash equality is
not proof of identity accuracy. No performance speedup implies semantic parity.

Lightweight validation (does not launch binaries):

```sh
python3 -m unittest discover -s benchmarks/batch/usearch_scaling/comparison -p 'test_*.py'
```
