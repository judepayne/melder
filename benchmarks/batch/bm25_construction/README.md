# Initial BM25 construction: share terms and build before publishing

M1 Ultra (20 cores, 64 GiB), macOS. Default-feature locked release; baseline `c4ae4c4` (same engine as `4ca17c2`). Three counterbalanced runs per arm per size, one process at a time; medians. USearch combined indexes warm, BM25 built fresh every time. A records are the nested samples in [the scaling experiment](../usearch_scaling/README.md); same 10k B queries, legal name/address BM25, 20 country blocks, other matching and output settings unchanged. Not a 1M × 1M timing.

| A records | Build approach | BM25 build (s) | Process wall (s) | Scoring queries/s | Process peak MiB |
|---:|---|---:|---:|---:|---:|
|100k|Baseline|0.529|4.588|5,609|2,345|
|100k|Fewer copies|0.478|4.558|5,609|2,358|
|100k|Serial builder + shared terms|0.455|4.460|5,732|2,314|
|1M|Baseline|6.704|29.715|1,548|11,281|
|1M|Fewer copies|6.092|28.662|1,568|11,449|
|1M|Serial builder + shared terms|5.317|27.626|1,571|11,017|

**Recommendation:** prefer the serial builder for batch construction. It took 21% less build time at 1M and 14% less at 100k. The 1M process peak was 263 MiB lower; this is not a measurement of the BM25 index alone. Build savings are once per index, not once per query. We would not promise a material speedup on a 1M-query job from this alone. No clear scoring regression on this fixture. A mostly unique vocabulary may benefit less; live insert throughput and live/enroll startup wall time were not benchmarked.

The [control](clone-control.patch) moves strings into their frequency map and then moves the map into document metadata instead of cloning it. It changes the order of metadata publication and is only a benchmark control, not a proposed live update change. The branch builds with single-writer maps, shares the spelling of each repeated term between documents, and appends already sorted postings without binary search. The result is still mutable, with the same tokenization, scoring and block partition rules. This three-arm study does not isolate the individual effects of sharing, locking, and appending. Across all nine runs at each size, the sorted accepted pair files matched; not a claim of identical unrounded scores or identity accuracy.

[Raw measurements](results.csv). To repeat, build the baseline and branch (`cargo build --release --locked`), and optionally a private baseline with `git apply clone-control.patch`. Give each release binary a distinct path. Prepare the warm inputs using the linked scaling experiment, then run:

```sh
RAYON_NUM_THREADS=20 ORT_NUM_THREADS=2 python3 benchmarks/batch/bm25_construction/run.py \
  --source /path/to/scaling/experiment --bin baseline=/path/to/baseline \
  --bin simple=/path/to/control --bin intern=/path/to/branch \
  --sizes 100k 1M --repeats 3 --limit 10000 --out /new/private/directory
```
