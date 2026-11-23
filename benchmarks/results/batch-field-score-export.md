# Batch field-score CSV export: Air check

Local approximate measurements, 2026-11-23; Apple M3. Baseline `git archive HEAD` at `38436a62ea3a2d77c499a2d0c376d559bcde5be2`; candidate uncommitted output writers/dispatch. Both `cargo build --release --locked` with default features, shared private target `/tmp/melder-live-postboot.fhBQnt/target`, copied to distinct binaries. No commits.

Fixture: existing `benchmarks/batch/10kx10k_bm25only/cold/config.yaml`, copied 10k A/B CSVs into private `/tmp`, same matching config (BM25/fuzzy/exact, country AND blocking, top_n=20, graph/approximate), fresh absent crossmap/cache/output per run. Off = scoring_log omitted; on = `scoring_log.enabled: true` (default zstd). No encoder used. Three rotating rounds with five-second rests, no cargo/rustc processes at each start. All runtime data outside Dropbox. Timing occurred after candidate release build; not alongside our builds/tests. Desktop background activity was not controlled.

`/usr/bin/time -l <binary> run --config <private-config>`; whole elapsed is Python monotonic around that command (includes startup/shutdown/output); RSS is Darwin maximum resident set bytes / 2^20. Throughput is CLI scoring throughput, not end-to-end. Output bytes sum **all** output-directory files including WAL/scoring log and candidates, not only CSV exports.

| Round | Variant | Scoring records/s | Whole seconds | Peak RSS MiB | Output bytes |
|---|---|---:|---:|---:|---:|
| 1 | baseline-off | 46640 | 9.597 | 584.9 | 10898100 |
| 1 | candidate-off | 42829 | 1.247 | 580.0 | 10898105 |
| 1 | candidate-on | 37483 | 1.363 | 807.8 | 54028612 |
| 2 | candidate-off | 36231 | 0.742 | 580.0 | 10898119 |
| 2 | candidate-on | 39787 | 1.340 | 794.0 | 54013287 |
| 2 | baseline-off | 45720 | 0.693 | 577.5 | 10898121 |
| 3 | candidate-on | 35887 | 1.401 | 795.3 | 54021611 |
| 3 | baseline-off | 50773 | 0.706 | 580.1 | 10898101 |
| 3 | candidate-off | 43809 | 0.695 | 577.3 | 10898037 |

## Interpretation and limits

Median whole time: baseline off 0.706s, candidate off 0.742s, candidate on 1.363s. First baseline launch was a 9.597s startup outlier (not discarded from table); first candidate off also slower. Tiny scoring phase and launch/cache effects make these unsuitable for a firm regression claim. Median scoring throughput: 46,640 / 42,829 / 37,483 records/s respectively. Candidate on adds roughly 0.62s whole time versus candidate off; peak RSS rises from ~580 to ~795 MiB; output grows from ~10.9MB to ~54.0MB. This measures scoring-log capture + replay + all enabled exports together, **not isolated companion CSV writer cost**. CSV only; no Parquet timing.

Validation: companion absent in all off outputs; all on companions parse with expected seven columns, 606,752 rows, numeric scores in [0,1], and evidence covering every accepted pair. Accepted-pair comparison details follow. Fixture input is fixed but retrieval exhibits small run-to-run retained-edge/review/group variation (including baseline), so do not call every output byte deterministic.

## Repeat on macOS

From the repository root with the candidate checked out (requires Python 3 and PyYAML):

```sh
set -eu
export WORK=$(mktemp -d /tmp/meld-field-scores.XXXXXX)
export CARGO_TARGET_DIR="$WORK/target"
mkdir "$WORK/base"
git archive 38436a6 | tar -x -C "$WORK/base"
(cd "$WORK/base" && cargo build --locked --release)
cp "$CARGO_TARGET_DIR/release/meld" "$WORK/baseline"
cargo build --locked --release
cp "$CARGO_TARGET_DIR/release/meld" "$WORK/candidate"
python3 - <<'PY'
import os, shutil, subprocess, time
from pathlib import Path
import yaml
w = Path(os.environ['WORK'])
fixture = Path('benchmarks/batch/10kx10k_bm25only/cold/config.yaml')
variants = ['baseline-off', 'candidate-off', 'candidate-on']
for i in range(3):
    for variant in variants[i:] + variants[:i]:
        arm, detail = variant.split('-'); run = w / f'{i+1}-{variant}'; run.mkdir()
        cfg = yaml.safe_load(fixture.read_text())
        for side, dataset in cfg['datasets'].items():
            p = w / f'{side}.csv'
            if not p.exists(): shutil.copyfile(dataset['path'], p)
            dataset['path'] = str(p)
        cfg['cross_map']['path'] = str(run/'crossmap.csv')
        cfg['embeddings']['a_cache_dir'] = str(run/'cache')
        cfg['output']['csv_dir_path'] = str(run/'output')
        if detail == 'on': cfg['scoring_log'] = {'enabled': True}
        (run/'config.yaml').write_text(yaml.safe_dump(cfg)); time.sleep(5)
        with (run/'log').open('w') as log:
            start = time.monotonic()
            subprocess.run(['/usr/bin/time','-l',w/arm,'run','--config',run/'config.yaml'], cwd=run, stdout=log, stderr=log, check=True)
            elapsed = time.monotonic() - start
        print(run.name, elapsed, sum(p.stat().st_size for p in (run/'output').rglob('*') if p.is_file()))
PY
# Read scoring throughput and process peak RSS from each log.
grep -E 'Throughput:|maximum resident set size' "$WORK"/*/log
```

Accepted pairs: counts [6092, 6092, 6092, 6092, 6092, 6092, 6092, 6092, 6092]; symmetric differences versus first baseline [0, 0, 2, 2, 2, 2, 2, 0, 2].
