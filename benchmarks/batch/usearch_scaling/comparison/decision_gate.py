#!/usr/bin/env python3
"""Serial decision audit; no timing. Uses parity.py's retained-evidence checks."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import parity

HERE = Path(__file__).resolve().parent
# --root names the private comparison workspace, not the repository.


def audit(root):
    result = {}
    for size in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        for recipe in sorted(size.iterdir()):
            trials = [parity.load(t) for t in sorted(recipe.iterdir()) if t.is_dir()]
            assert len(trials) == 6
            groups = {}
            for a, b in itertools.combinations(trials, 2):
                arm = '/'.join(sorted([a['name'].split('-')[-1], b['name'].split('-')[-1]]))
                groups.setdefault(arm, []).append(parity.compare(a, b))
            result[f'{size.name}/{recipe.name}'] = {}
            for arm, cs in groups.items():
                metrics = {
                    'accepted_edge_diff': lambda c: c['match_symmetric_difference'],
                    'review_edge_diff': lambda c: c['review_symmetric_difference'],
                    'classification_changes': lambda c: c['query_category_changes'],
                    'unmatched_membership_diff': lambda c: c['unmatched_symmetric_difference'],
                    'unmatched_best_score_changes': lambda c: c['un']['scores']['changed'],
                    'unmatched_best_score_max_delta': lambda c: c['un']['scores']['max'],
                    'wal_score_changes': lambda c: c['wal']['scores']['changed'],
                    'wal_score_max_delta': lambda c: c['wal']['scores']['max'],
                    'wal_rank_changes': lambda c: c['wal']['rank_changes'],
                    'wal_reason_changes': lambda c: c['wal']['reason_changes'],
                }
                result[f'{size.name}/{recipe.name}'][arm] = dict(pairs=len(cs), ranges={k: [min(f(c) for c in cs), max(f(c) for c in cs)] for k, f in metrics.items()}, anomalous_pairs=[c for c in cs if c['match_symmetric_difference'] or c['review_symmetric_difference'] or c['query_category_changes']])
    return result


def synthetic(root):
    out = root / 'synthetic-decision-gate-v5'
    out.mkdir(exist_ok=True)
    results = {}
    for n in (31, 5101):
        data = out / str(n)
        data.mkdir(exist_ok=True)
        fields = ['entity_id', 'counterparty_id', 'name', 'address', 'region', 'segment']
        pool = [dict(entity_id=f'a{i}', name=f'unique{i} token{i} extra{i}', address=f'road{i} suite{i}', region='R', segment='S') for i in range(n)]
        # Composite keys, missing second component, missing first component, and fully missing.
        for i, pair in enumerate([('R',''), ('','S'), ('','')]):
            pool.append(dict(entity_id=f'm{i}', name=f'missing{i} uniqueMissing{i}', address=f'place{i}', region=pair[0], segment=pair[1]))
        queries = []
        for j, i in enumerate((0, 7, 19)):
            queries.append(dict(counterparty_id=f'q{j}', name=pool[i]['name'], address=pool[i]['address'], region='R', segment='S'))
        for i in range(3):
            queries.append(dict(pool[n+i], entity_id='', counterparty_id=f'qm{i}'))
        queries.append(dict(counterparty_id='unmatched', name='absentword', address='absentroad', region='R', segment='S'))
        for filename, records in [('a.csv', pool), ('b.csv', queries)]:
            with (data / filename).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
        for backend in ('memory', 'sqlite'):
            loaded = []
            for arm in ('baseline', 'candidate'):
                trial = data / backend / arm
                trial.mkdir(parents=True, exist_ok=False)
                config = dict(embeddings=dict(model='all-MiniLM-L6-v2', a_cache_dir=str(trial/'cache-a'), b_cache_dir=str(trial/'cache-b')), job=dict(name='synthetic_bm25'), datasets={side: dict(path=str(data / filename), id_field=idfield, format='csv') for side, filename, idfield in [('a','a.csv','entity_id'),('b','b.csv','counterparty_id')]}, cross_map=dict(backend='local', path=str(trial/'crossmap.csv'), a_id_field='entity_id', b_id_field='counterparty_id'), blocking=dict(enabled=True, operator='and', fields=[dict(field_a=f, field_b=f) for f in ('region','segment')]), bm25_fields=[dict(field_a=f,field_b=f) for f in ('name','address')], match_fields=[dict(method='bm25',weight=1)], thresholds=dict(auto_match=0.8,review_floor=0.2), output=dict(csv_dir_path=str(trial/'output'), db_path=str(trial/'results.db')), batch=dict(assignment=dict(strategy='graph',mode='exact')), top_n=5, bm25_candidates=10, ann_candidates=50)
                if backend == 'sqlite':
                    config['batch'].update(db_path=str(trial/'scratch.db'), sqlite_read_pool_size=1)
                (trial/'config.yaml').write_text(json.dumps(config))
                with (trial/'meld.log').open('w') as log:
                    subprocess.run([str(root/arm), 'run', '--config', str(trial/'config.yaml')], stdout=log, stderr=log, check=True, env=dict(os.environ, RAYON_NUM_THREADS='1'))
                if backend == 'memory':
                    loaded.append(parity.load(trial))
                else:
                    rel = parity.rows(trial/'output/relationships.csv')
                    un = parity.rows(trial/'output/unmatched.csv')
                    loaded.append(dict(name=arm, rel={(r['counterparty_id'],r['entity_id'],r['relationship_type']):r for r in rel}, un={r['counterparty_id']:r for r in un}, wal={}, counts={}, unmatched=len(un), crossmap=len(parity.rows(trial/'crossmap.csv'))))
            # parity.numeric expects numeric CSV cells; blank unmatched score means None.
            import copy
            normalized = copy.deepcopy(loaded)
            for run in normalized:
                for row in run['un'].values():
                    if row.get('best_score') == '':
                        row['best_score'] = None
            c = parity.compare(*normalized)
            # Compare complete rows and unrounded decision WAL, not rounded output scores alone.
            c['complete_evidence_equal'] = all(loaded[0][k] == loaded[1][k] for k in ('rel','un','wal'))
            import sqlite3
            dbs = []
            for arm in ('baseline', 'candidate'):
                with sqlite3.connect(data/backend/arm/'results.db') as db:
                    tables = [r[0] for r in db.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
                    dbs.append({t: sorted(db.execute('select * from "'+t+'"').fetchall(), key=repr) for t in tables})
            c['complete_sqlite_output_equal'] = dbs[0] == dbs[1]
            c['sqlite_tables'] = {k:len(v) for k,v in dbs[0].items()}
            c['wal_available'] = backend == 'memory'
            c['counts'] = {k: loaded[0][k] for k in ('counts','unmatched','crossmap')}
            results[f'{n}/{backend}'] = c
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()
    result = synthetic(args.root) if args.synthetic else audit(args.root/'final-paired')
    target = HERE / ('synthetic-decision-results.json' if args.synthetic else 'decision-summary.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(target)

if __name__ == '__main__':
    main()
