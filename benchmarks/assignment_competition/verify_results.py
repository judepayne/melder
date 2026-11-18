#!/usr/bin/env python3
"""Independently audit real competition runs and archive their complete evidence."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error('Verification requires assertions; run without Python optimization')
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads((args.run_root / 'measurements.json').read_text())
    identity = json.loads((here / 'results/execution_identity.json').read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert raw['status'] == 'complete' and len(raw['runs']) == 160
    for path, expected in identity['implementation_sha256'].items():
        if path.endswith(('.py', '.yaml', '.json')):
            assert sha(repo / path) == expected, path
    for path, expected in identity['binary_source_provenance']['experimental_source_sha256'].items():
        assert sha(repo / path) == expected, path
    for label, name in [('baseline', 'main'), ('experimental', 'experimental')]:
        expected = identity['binary_source_provenance']['binary_sha256'][name]
        assert raw['provenance']['binaries'][label]['sha256'] == expected
        assert raw['provenance']['binary_hashes_after'][label] == expected
        assert sha(raw['provenance']['binaries'][label]['path']) == expected
    paths = raw['provenance']['inputs']
    for entry in paths.values():
        assert sha(entry['path']) == entry['sha256']
    with Path(paths['dataset_a.csv']['path']).open() as stream:
        a_rows = {row['entity_id']: row for row in csv.DictReader(stream)}
    with Path(paths['dataset_b.csv']['path']).open() as stream:
        b_rows = {row['counterparty_id']: row for row in csv.DictReader(stream)}
    fixture = raw['fixture']
    groups = {g['id']: g for g in fixture['groups']}
    membership = {b: g for g in fixture['groups'] for b in g['b_ids']}
    custom_ids = set(membership)
    calibration = json.loads(Path(raw['preparation']['calibration']['path']).read_text())
    matrix = {b: {c['matched_id']: c for c in row} for b, row in calibration['matrix'].items()}
    assert len(matrix) == 500 and calibration['validated_queries'] == 500
    assert all(value['uniform_rank_order'] for value in calibration['ambiguous_ties'].values())
    seed = raw['preparation']['seed_cache_manifest']
    proof = {'runs': 160, 'independent_wal_csv_folds': 0, 'independent_active_output_checks': 0,
             'custom_repairs': 0, 'custom_repairs_net_true_pairs': 0, 'rank5_repairs': 0,
             'controls_repaired': 0, 'per_family_moves': Counter(), 'per_setting': {},
             'source_input_binary_hashes_unchanged': True, 'initial_cache_bytes_identical': True,
             'no_vector_or_index_bytes_changed': True, 'run_raw_sha256': {}}
    by_setting = defaultdict(list)
    for run in raw['runs']:
        directory = args.run_root / Path(run['directory']).name
        assert json.loads((directory / 'cache_before.json').read_text()) == seed
        after = json.loads((directory / 'cache_after.json').read_text())
        changed = [key for key in seed.keys() | after.keys() if seed.get(key) != after.get(key)]
        assert all(key.endswith('.manifest') for key in changed), changed
        forward, reverse = {}, {}
        def add(a, b):
            assert a in a_rows and b in b_rows
            assert a_rows[a]['competition_group'] == b_rows[b]['competition_group']
            assert a_rows[a]['country_code'] == b_rows[b]['domicile']
            assert a not in forward and b not in reverse
            forward[a], reverse[b] = b, a
        def remove(a, b):
            assert forward.get(a) == b and reverse.get(b) == a
            del forward[a], reverse[b]
        wal_paths = list((directory / 'output').glob('*.match_log*.ndjson'))
        assert len(wal_paths) == 1
        moves = 0
        for line in wal_paths[0].open():
            event = json.loads(line)
            kind = event['type']
            if kind == 'crossmap_confirm':
                add(event['a_id'], event['b_id'])
            elif kind == 'crossmap_break':
                remove(event['a_id'], event['b_id'])
            elif kind == 'crossmap_reassign':
                moves += 1
                old_a, old_b = event['old_a_id'], event['old_b_id']
                remove(old_a, old_b)
                group = membership.get(old_b)
                if group:
                    family = group['kind']
                    assert family.startswith('rank'), 'Control repaired'
                    assert old_b == group['flexible_b_id'] and old_a != group['escape_a_id']
                    assert run['top_n'] >= group['escape_rank']
                    assert all(group['truth'][p['b_id']] == p['a_id'] for p in event['replacements'])
                    flex = next(p for p in event['replacements'] if p['b_id'] == old_b)
                    assert flex['rank'] == group['escape_rank']
                    proof['custom_repairs'] += 1
                    proof['custom_repairs_net_true_pairs'] += 2
                    proof['rank5_repairs'] += flex['rank'] == 5
                    proof['per_family_moves'][family] += 1
                for pair in event['replacements']:
                    assert pair['score'] >= .85 and 1 <= pair['rank'] <= run['top_n']
                    if pair['b_id'] in membership:
                        actual = matrix[pair['b_id']][pair['a_id']]
                        assert actual['rank'] == pair['rank'] and abs(actual['score'] - pair['score']) < 1e-6
                    add(pair['a_id'], pair['b_id'])
            else:
                assert not kind.startswith('crossmap_'), kind
        assert moves == run['local_moves']
        with (directory / 'crossmap.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == len(forward) == len(reverse)
        assert {(r['entity_id'], r['counterparty_id']) for r in csv_rows} == set(forward.items())
        assignment = dict.fromkeys(b_rows) | reverse
        assert assignment == run['assignment']
        proof['independent_wal_csv_folds'] += 1
        matches, reviews = {}, set()
        with (directory / 'output/relationships.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['relationship_type'] == 'match':
                    assert row['counterparty_id'] not in matches
                    matches[row['counterparty_id']] = row['entity_id']
                elif row['relationship_type'] == 'review':
                    reviews.add(row['counterparty_id'])
        with (directory / 'output/unmatched.csv').open() as stream:
            unmatched = {row['counterparty_id'] for row in csv.DictReader(stream)}
        assert matches == reverse and not set(matches) & reviews
        assert unmatched == b_rows.keys() - matches.keys() - reviews
        assert len(matches) + len(reviews) + len(unmatched) == 10500
        proof['independent_active_output_checks'] += 1
        for g in fixture['groups']:
            count = sum(assignment[b] is not None for b in g['b_ids'])
            if g['kind'].startswith('rank'):
                if run['top_n'] < g['escape_rank']:
                    assert count == 4
                elif run['binary'] == 'experimental':
                    assert all(assignment[b] == g['truth'][b] for b in g['b_ids'])
                else:
                    assert count in (4, 5)
            elif g['kind'] == 'ambiguous':
                assert count == run['top_n'] and all(assignment[b] is None or assignment[b] in g['allowed'][b] for b in g['b_ids'])
            else:
                assert count == 4
        proof['run_raw_sha256'][directory.name] = {p.name: sha(p) for p in [wal_paths[0], directory / 'crossmap.csv', directory / 'stdout.log', directory / 'stderr.log']}
        by_setting[run['top_n'], run['binary']].append(run)
    for (n, label), runs in by_setting.items():
        assert len(runs) == 20
        recorded = raw['summary']['by_top_n_binary'][f'{n}/{label}']
        for family, cohort in recorded['cohorts'].items():
            ids = [b for b, o in raw['oracle'].items() if o['family'] == family]
            unequal = 0
            for b in ids:
                counts = Counter(run['assignment'][b] for run in runs)
                unequal += 190 - sum(value * (value - 1) // 2 for value in counts.values())
            mean = unequal / 190
            assert abs(mean - cohort['stability']['mean_pairwise_changed_b']) < 1e-10
        proof['per_setting'][f'{n}/{label}'] = {'runs': 20, 'independent_frequency_stability_matches': True}
    for index in range(1, 21):
        runs = [run for run in raw['runs'] if run['shuffle_index'] == index]
        assert len(runs) == 8 and len({json.dumps(run['permutation'], sort_keys=True) for run in runs}) == 1
        expected = [1, 2, 3, 5]
        offset = (index - 1) % 4
        expected = expected[offset:] + expected[:offset]
        labels = ['baseline', 'experimental'] if index % 2 else ['experimental', 'baseline']
        assert [(run['top_n'], run['binary']) for run in runs] == [(n, label) for n in expected for label in labels]
    assert len({run['permutation']['permutation_sha256'] for run in raw['runs']}) == 20
    proof['script_sha256'] = sha(Path(__file__))
    raw['independent_validation'] = proof
    raw['execution_identity'] = identity
    with (output / 'measurements.json.gz').open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
            zipped.write((json.dumps(raw, separators=(',', ':'), sort_keys=True) + '\n').encode())
    (output / 'validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    shutil.copy2(raw['preparation']['calibration']['path'], output / 'combined_calibration.json')
    shutil.copy2(args.run_root / 'summary.md', output / 'generated_summary.md')
    print(f"Verified {proof['independent_wal_csv_folds']} WAL/CSV folds, {proof['custom_repairs']} correct custom repairs, {proof['rank5_repairs']} fifth-rank repairs.")


if __name__ == '__main__':
    main()
