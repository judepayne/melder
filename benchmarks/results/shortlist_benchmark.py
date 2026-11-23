#!/usr/bin/env python3
"""macOS reproducible 10k BM25 shortlist benchmark; Python 3 + PyYAML.
Run from repository root. Builds archived baseline, waits for supplied candidate.
All bulky artifacts are private /tmp; retained JSON contains evidence counts.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import yaml

ROOT = Path.cwd()
WORK = Path(os.environ.get('BENCH_WORK', tempfile.mkdtemp(prefix='melder-shortlist-', dir='/tmp')))
TARGET = os.environ.get('CARGO_TARGET_DIR', str(WORK/'target'))
CANDIDATE = Path(os.environ['BENCH_CANDIDATE']).resolve()
BASE = 'bfd16b4f75dca4510b1a79cc7988b4317bd2c2f5'
RESULT = ROOT / 'benchmarks/results/shortlist-results.json'

def quiet():
    while True:
        ps = subprocess.check_output(['ps', '-axo', 'comm='], text=True)
        if not any(Path(p.strip()).name in ('cargo', 'rustc') for p in ps.splitlines()):
            return
        print('Waiting for cargo/rustc to finish', flush=True)
        time.sleep(10)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

WORK.mkdir(exist_ok=True)
print('WORK', WORK, flush=True)
archive = WORK / 'baseline-source'
archive.mkdir(exist_ok=True)
with (WORK / 'baseline.tar').open('wb') as f:
    subprocess.run(['git', 'archive', BASE], stdout=f, check=True)
subprocess.run(['tar', '-xf', str(WORK/'baseline.tar'), '-C', str(archive)], check=True)
(WORK/'baseline.tar').unlink()
quiet()
with (WORK/'build.log').open('w') as log:
    subprocess.run(['cargo', 'build', '--release', '--locked'], cwd=archive,
                   env={**os.environ, 'CARGO_TARGET_DIR': TARGET}, stdout=log, stderr=log, check=True)
shutil.copy2(Path(TARGET)/'release/meld', WORK/'baseline')
print('Baseline built from', BASE, sha(WORK/'baseline'), flush=True)
while not CANDIDATE.exists():
    print('Waiting for', CANDIDATE, flush=True)
    time.sleep(15)
shutil.copy2(CANDIDATE, WORK/'candidate')
config = yaml.safe_load((ROOT/'benchmarks/batch/10kx10k_bm25only/cold/config.yaml').read_text())
for side, ds in config['datasets'].items():
    p = WORK / f'{side}.csv'
    shutil.copyfile(ROOT/ds['path'], p)
    ds['path'] = str(p)
results = {'baseline_commit': BASE, 'work': str(WORK),
           'binary_sha256': {a: sha(WORK/a) for a in ('baseline', 'candidate')},
           'fixture_sha256': {s: sha(WORK/f'{s}.csv') for s in ('a', 'b')}, 'runs': []}
# Reverse alternating rounds and rotate to counterbalance launch/order effects.
conditions = [(storage, arm, logging) for storage in ('memory','sqlite')
              for arm in ('baseline','candidate') for logging in ('off','on')]
for repeat in range(3):
    order = conditions[repeat*3:] + conditions[:repeat*3]
    if repeat % 2: order.reverse()
    for storage, arm, logging in order:
        quiet()
        time.sleep(3)
        run = WORK/f'{repeat+1}-{storage}-{arm}-{logging}'
        run.mkdir()
        cfg = yaml.safe_load(yaml.safe_dump(config))
        cfg['cross_map']['path'] = str(run/'crossmap.csv')
        cfg['embeddings']['a_cache_dir'] = str(run/'cache')
        cfg['output'] = {'csv_dir_path': str(run/'output'), 'db_path': str(run/'output.sqlite')}
        if storage == 'sqlite': cfg['batch'] = {'db_path': str(run/'scratch.sqlite')}
        if logging == 'on': cfg['scoring_log'] = {'enabled': True}
        (run/'config.yaml').write_text(yaml.safe_dump(cfg))
        with (run/'log').open('w') as f:
            start = time.monotonic()
            subprocess.run(['/usr/bin/time','-l',str(WORK/arm),'run','--config',str(run/'config.yaml')], cwd=run, stdout=f, stderr=f, check=True)
            wall = time.monotonic()-start
        log = (run/'log').read_text()
        db = sqlite3.connect(run/'output.sqlite')
        counts = {t: db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('relationships','field_scores')}
        evidence_edges = db.execute('SELECT COUNT(*) FROM (SELECT DISTINCT a_id,b_id FROM field_scores)').fetchone()[0]
        if arm == 'candidate' and logging == 'on':
            counts['candidate_edges'] = db.execute('SELECT count(*) FROM candidate_edges').fetchone()[0]
            assert counts['candidate_edges'] == evidence_edges and evidence_edges > 100000
        csv_counts = {}
        for p in (run/'output').glob('*.csv'):
            with p.open(newline='') as f: csv_counts[p.name] = max(0, sum(1 for _ in csv.reader(f))-1)
        if 'candidate_edges' in counts:
            assert csv_counts['candidate_edges.csv'] == counts['candidate_edges'] and csv_counts['field_scores.csv'] == counts['field_scores']
        db.close()
        metrics = {'repeat':repeat+1,'storage':storage,'arm':arm,'logging':logging,'wall_seconds':wall,
                   'rss_bytes': int(re.search(r'(\d+)\s+maximum resident set size',log)[1]),
                   'scoring_records_per_sec':int(re.search(r'Throughput:\s*(\d+) records/sec',log)[1]),
                   'persistent_bytes':sum(p.stat().st_size for p in (run/'output').rglob('*') if p.is_file())+(run/'output.sqlite').stat().st_size,
                   'sqlite_counts':counts,'evidence_edges':evidence_edges,'csv_rows':csv_counts}
        init = re.search(r'Encoder ready .*?, took ([\d.]+)s', log)
        metrics['encoder_init_s'] = float(init[1]) if init else 0
        results['runs'].append(metrics)
        RESULT.write_text(json.dumps(results,indent=2))
        print(json.dumps(metrics), flush=True)
        # Preserve config/log but bound disk use: each output's evidence was counted above.
        shutil.rmtree(run/'output')
        for p in run.glob('*.sqlite*'): p.unlink()
print('DONE', RESULT, flush=True)
