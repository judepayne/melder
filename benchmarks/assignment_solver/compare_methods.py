#!/usr/bin/env python3
"""Compare five assignment methods on identical frozen graphs in fresh processes."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from run import ROOT, PROTOTYPE, measured, save, sha256, source_identity

MODES = ['exact', 'score', 'count', 'best', 'hybrid']
LIMIT = 256  # active vertices per component; chosen before measurements


def run_suite(out, capture):
    out.mkdir(parents=True, exist_ok=False)
    built = subprocess.run(['cargo', 'build', '--release', '--bin', 'compare', '--message-format=json'], cwd=PROTOTYPE, capture_output=True, check=True)
    executable, = [a['executable'] for a in map(json.loads, built.stdout.splitlines()) if a.get('executable') and a.get('target', {}).get('name') == 'compare']
    binary = out / 'compare'
    shutil.copy2(executable, binary)
    identity = {'binary_sha256': sha256(binary), 'source_sha256': source_identity(), 'driver_sha256': sha256(Path(__file__)), 'rustc': subprocess.check_output(['rustc','-Vv'], text=True).strip()}
    report = {'identity': identity, 'exact_component_limit': LIMIT, 'graphs': {}}
    for label, source in [('capture10k', str(capture/'10k.graph')), ('capture100k', str(capture/'100k.graph')), ('giant10k','giant:10000:5:6221233'), ('giant100k','giant:100000:5:6221233')]:
        real = label.startswith('capture')
        graph = {'source': Path(source).name if real else source, 'sha256': sha256(Path(source)) if real else None, 'runs': []}
        report['graphs'][label] = graph
        by_mode = {}; digests = {}; exact_timed_out = False
        for rep, (workers, seed) in enumerate([(4,11),(4,22),(4,33),(1,44)]):
            for mode in MODES[rep:]+MODES[:rep]:
                # No ratio to an uncompleted exact solve. Do not retry the same timed-out treatment.
                if mode == 'exact' and exact_timed_out: continue
                stem = f'{label}-{mode}-{rep}'; pairs = out/(stem+'.pairs'); stdout = out/(stem+'.json'); stderr = out/(stem+'.stderr')
                command = list(map(str,[binary, source, mode, workers, LIMIT, seed, pairs]))
                resource = measured(command, 25, stdout, stderr)
                row = {'mode': mode, 'workers': workers, 'seed': seed, **resource}
                graph['runs'].append(row)
                if resource['timed_out']:
                    if mode != 'exact' or pairs.exists(): raise RuntimeError(stem + ' unexpected timeout/partial result')
                    exact_timed_out = True; row['status'] = 'timeout'; print(stem, 'timeout', flush=True); continue
                if resource['returncode'] != 0: raise RuntimeError(stderr.read_text())
                result = json.loads(stdout.read_text()); row.update(status='ok', result=result, pairs_sha256=sha256(pairs))
                if not int(result['objective_units']) <= int(result['upper_units']): raise ValueError('certificate violated')
                if digests.setdefault(mode, row['pairs_sha256']) != row['pairs_sha256']: raise ValueError('pair order or membership changed: '+stem)
                if mode not in by_mode: by_mode[mode] = {tuple(map(int,l.split())) for l in pairs.read_text().splitlines()}
                print(stem, round(result['solve_s'],5), flush=True)
        summary = graph['summary'] = {}
        exact = next((int(r['result']['objective_units']) for r in graph['runs'] if r['mode']=='exact' and r['status']=='ok'), None)
        for mode, pairs in by_mode.items():
            rows=[r for r in graph['runs'] if r['mode']==mode and r['status']=='ok' and r['workers']==4]; z=rows[0]['result']; score=int(z['objective_units']); upper=int(z['upper_units']);
            if exact is not None and not score <= exact <= upper: raise ValueError('incorrect bound')
            summary[mode]={'median_s':statistics.median(r['result']['solve_s'] for r in rows), 'peak_mib':max(r['peak_rss']['bytes'] for r in rows)/2**20, 'score':score, 'upper':upper,'pairs':len(pairs),'exact_only_pairs':len(by_mode['exact']-pairs) if exact is not None else None,'method_only_pairs':len(pairs-by_mode['exact']) if exact is not None else None}
        save(out/'report.json',report)
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--capture',type=Path,required=True); args=ap.parse_args(); run_suite(args.output.resolve(),args.capture.resolve())
