#!/usr/bin/env python3
"""Probe/freeze isolated competition fixtures using real baseline scoring logs only."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

BASELINE = Path('/var/folders/g6/zfm6_wzj20g4dq7x42bv4_l40000gn/T/melder-assignment-stability-t393kh8g/bin/main')
WORKSPACE = Path('/var/folders/g6/zfm6_wzj20g4dq7x42bv4_l40000gn/T/melder-competition-3cj_c9ft/calibration')
HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probe(a_rows, b_rows, directory, baseline=BASELINE):
    """Run production meld; return the unmodified parsed NDJSON records."""
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / 'a.csv', a_rows)
    write_csv(directory / 'b.csv', b_rows)
    template = (HERE / 'config.template.yaml').read_text()
    for key, value in {'DATA_A': directory / 'a.csv', 'DATA_B': directory / 'b.csv',
                       'CACHE': directory / 'cache', 'CROSSMAP': directory / 'crossmap.csv',
                       'OUTPUT': directory / 'output', 'TOP_N': 5}.items():
        template = template.replace('__' + key + '__', str(value))
    template += '\nscoring_log:\n  enabled: true\n  compression: none\n'
    config = directory / 'config.yaml'
    config.write_text(template)
    with (directory / 'run.log').open('w') as log:
        subprocess.run([str(baseline), 'run', '--config', str(config)], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    logs = list(directory.rglob('*.scoring_log.ndjson'))
    if len(logs) != 1:
        raise ValueError(f'Expected one raw scoring log: {logs}')
    return [json.loads(line) for line in logs[0].read_text().splitlines() if line.strip()]


def verify(fixture, records):
    """Assert the threshold graph and observed escape ranks, without rescoring."""
    headers = [r for r in records if r.get('type') == 'header']
    assert len(headers) == 1 and headers[0]['top_n'] == 5, 'Expected top_n=5 log'
    assert headers[0]['thresholds']['auto_match'] == .85, 'Wrong production threshold'
    scored_rows = [r for r in records if r.get('type') == 'scored']
    scored = {r['query_id']: r for r in scored_rows}
    assert len(scored_rows) == len(scored), 'Repeated query in scoring log'
    expected = {row['counterparty_id'] for row in fixture['b_rows']}
    if set(scored) != expected:
        raise ValueError('Raw log must contain exactly every custom B query')
    evidence = []
    for group in fixture['groups']:
        matrices = {}
        qualifying = {}
        for bid in group['b_ids']:
            candidates = scored[bid]['candidates']
            assert len(candidates) == 5, f'{bid}: expected full top_n=5'
            assert {c['matched_id'] for c in candidates} == set(group['a_ids']), bid
            assert [c['rank'] for c in candidates] == list(range(1, 6)), bid
            matrices[bid] = candidates
            qualifying[bid] = [c['matched_id'] for c in candidates if c['score'] >= .85]
        flexible = group['flexible_b_id']
        if group['kind'] == 'ambiguous':
            assert all(set(ids) == set(group['a_ids']) for ids in qualifying.values()), group['id']
        else:
            for bid, aid in group['truth'].items():
                if bid != flexible:
                    assert qualifying[bid] == [aid], f'{bid}: non-exclusive constrained edge'
            if group['kind'] == 'unrepairable':
                assert set(qualifying[flexible]) == set(group['truth'].values()), group['id']
            else:
                assert set(qualifying[flexible]) == set(group['a_ids']), group['id']
                escape = matrices[flexible][group['escape_rank'] - 1]['matched_id']
                assert escape == group['escape_a_id'], f'{group["id"]}: escape rank changed'
        # Enumerate actual complete feasible bijections to check uniqueness/ambiguity.
        complete = [dict(zip(group['b_ids'], permutation))
                    for permutation in itertools.permutations(group['a_ids'])
                    if all(aid in qualifying[bid]
                           for bid, aid in zip(group['b_ids'], permutation))]
        if group['kind'].startswith('rank'):
            assert complete == [group['truth']], f'{group["id"]}: truth not unique'
        elif group['kind'] == 'ambiguous':
            assert len(complete) == 120, group['id']
        else:
            assert not complete and len(set(group['truth'].values())) == 4, group['id']
        accepted = [c['score'] for cs in matrices.values() for c in cs if c['score'] >= .85]
        rejected = [c['score'] for cs in matrices.values() for c in cs if c['score'] < .85]
        assert min(accepted) >= .86, f'{group["id"]}: accepted margin below .01'
        assert not rejected or max(rejected) <= .84, f'{group["id"]}: rejected margin below .01'
        evidence.append({'id': group['id'], 'kind': group['kind'], 'matrix': matrices,
                         'qualifying': qualifying, 'complete_bijections': len(complete),
                         'min_accepted_score': min(accepted),
                         'max_rejected_score': max(rejected) if rejected else None})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=HERE / 'fixture.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    records = probe(fixture['a_rows'], fixture['b_rows'], args.output, args.baseline)
    groups = verify(fixture, records)
    evidence = {
        'schema_version': 1,
        'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        'baseline_path': str(args.baseline),
        'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'raw_scoring_log': str(next(args.output.rglob('*.scoring_log.ndjson'))),
        'config_path': str(args.output / 'config.yaml'),
        'config_sha256': hashlib.sha256((args.output / 'config.yaml').read_bytes()).hexdigest(),
        'raw_scoring_log_sha256': hashlib.sha256(
            next(args.output.rglob('*.scoring_log.ndjson')).read_bytes()).hexdigest(),
        'model': 'all-MiniLM-L6-v2',
        'minimum_threshold_margin': .01,
        'calibration_scope': 'custom groups only; parent must validate combined dataset',
        'threshold': .85, 'top_n': 5, 'ann_candidates': 100,
        'template_note': '20 exact matching-text clones per family; rank families share one '
                         'five-city asset-manager template. Unrepairable replaces York with '
                         'an unrelated agricultural repair company. Not 100 independent name '
                         'families or evidence of realistic ANN retrieval.',
        'groups': groups,
    }
    (args.output / 'calibration.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'Verified {len(groups)} groups; evidence: {args.output / "calibration.json"}')


if __name__ == '__main__':
    main()
