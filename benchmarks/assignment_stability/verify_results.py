#!/usr/bin/env python3
"""Independently verify raw runs and retain cohort/repair diagnostics.

Requires the retained raw output of run_test.py and matching binary/source
provenance. Never launches meld, modifies datasets, or changes measured runs.
"""
import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def verify_input_orders(results):
    """Validate paired bytes/paths and campaign-wide input-order controls."""
    requested = results['provenance']['requested_runs']
    mode = results.get('input_order', results['provenance'].get('input_order', 'shuffle'))
    assert mode in ('shuffle', 'fixed')
    assert len(results['runs']) == 2 * requested
    for index in range(1, requested + 1):
        runs = [run for run in results['runs'] if run['shuffle_index'] == index]
        assert len(runs) == 2 and runs[0]['permutation'] == runs[1]['permutation']
        expected = ['baseline', 'experimental'] if index % 2 else ['experimental', 'baseline']
        assert [run['binary'] for run in runs] == expected
        if 'input_path' in runs[0]:
            assert runs[0]['input_path'] == runs[1]['input_path']
            for run in runs:
                assert run['input_order'] == mode
                assert run['permutation']['input_path'] == run['input_path']
    if mode == 'fixed':
        original = results['provenance']['datasets']['dataset_b_10k.csv']
        for run in results['runs']:
            assert run['input_order'] == 'fixed'
            assert run['input_path'] == original['copy']
            assert run['permutation'] == {'csv_sha256': original['sha256'],
                                          'input_path': original['copy']}
        assert len({run['input_path'] for run in results['runs']}) == 1
    else:
        assert len({run['permutation']['permutation_sha256'] for run in results['runs']}) == requested
    return mode


def repair_rank_profile(effects):
    return {
        'total_moves': sum(item['moves'] for item in effects),
        'moves_using_rank_beyond_three': sum(any(rank is not None and rank > 3 for rank in move['replacement_ranks'])
            for item in effects for move in item['moves_detail']),
        'maximum_replacement_rank': max((rank for item in effects
            for move in item['moves_detail'] for rank in move['replacement_ranks'] if rank is not None),
            default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Raw directory containing measurements.json and shuffle_* directories")
    parser.add_argument("--provenance", type=Path,
                        default=Path(__file__).with_name("binary_provenance.json"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for compressed complete measurements and validation JSON")
    parser.add_argument("--prefix", default="verified")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Verification requires assertions; run without Python optimization")
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("--prefix must be a single nonempty filename component")
    REPO = Path(__file__).resolve().parents[2]
    RUNS = args.run_root.resolve(strict=True)
    DEST = args.output.resolve()
    DEST.mkdir(parents=True, exist_ok=True)
    r = json.loads((RUNS / 'measurements.json').read_text())
    p = json.loads(args.provenance.read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    requested_runs = r['provenance']['requested_runs']
    assert r['status'] == 'complete' and len(r['runs']) == 2 * requested_runs
    input_order = verify_input_orders(r)
    for path, expected in p['experimental_source_sha256'].items():
        assert sha(REPO / path) == expected, path
    for path, expected in p['original_input_sha256'].items():
        assert sha(REPO / path) == expected, path
    for label, name in [('baseline', 'main'), ('experimental', 'experimental')]:
        assert r['provenance']['binaries'][label]['sha256'] == p['binary_sha256'][name]
        assert r['provenance']['binary_hashes_after'][label] == p['binary_sha256'][name]
        assert sha(p['binary_paths'][name]) == p['binary_sha256'][name]
    with (REPO / 'benchmarks/data/dataset_b_10k.csv').open() as f:
        b = {row['counterparty_id']: row for row in csv.DictReader(f)}
    with (REPO / 'benchmarks/data/ground_truth_crossmap_10k.csv').open() as f:
        truth = {(row['entity_id'], row['counterparty_id']) for row in csv.DictReader(f)}
    cohorts = {group: {key for key, row in b.items() if row['_match_type'] == group}
               for group in ['matched', 'ambiguous', 'unmatched']}
    assert len(truth) == 7000
    seed = r['preparation']['seed_cache_manifest']
    validation = {'runs': len(r['runs']), 'input_order': input_order,
                  'input_rows': len(b), 'source_and_input_hashes_unchanged': True,
                  'identical_seed_cache_copies': True, 'wal_equals_final_crossmap_runs': 0,
                  'all_run_logs_and_wal_sha256': {}, 'cache_changed_files': {},
                  'per_run_cohorts': [], 'within_run_repair_effects': []}
    for run in r['runs']:
        d = RUNS / Path(run['directory']).name
        if 'input_path' in run:
            input_path = Path(run['input_path'])
            assert sha(input_path) == run['permutation']['csv_sha256']
            # Check the actual config used by meld, not just reported metadata.
            config = (d / 'config.yaml').read_text()
            assert f'  b:\n    path: {json.dumps(str(input_path))}\n' in config
            assert sha(d / 'config.yaml') == run['config_sha256']
        assert json.loads((d / 'cache_before.json').read_text()) == seed
        after = json.loads((d / 'cache_after.json').read_text())
        changed = sorted(key for key in seed.keys() | after.keys() if seed.get(key) != after.get(key))
        assert all(key.endswith('.manifest') for key in changed), changed
        validation['cache_changed_files'][d.name] = changed
        validation['all_run_logs_and_wal_sha256'][d.name] = {
            f.name: sha(f) for f in [d / 'stdout.log', d / 'stderr.log', d / 'resources.log']}
        a_to_b, b_to_a = {}, {}
        before = None
        moves = []
        for wal in sorted((d / 'output').glob('*.ndjson')):
            assert sha(wal) == run['wal_sha256'][str(wal.relative_to(d))]
            for line in wal.open():
                event = json.loads(line)
                t = event['type']
                if t == 'crossmap_confirm':
                    assert before is None, 'ordinary claims appeared after repair'
                    a, query = event['a_id'], event['b_id']
                    assert a not in a_to_b and query not in b_to_a
                    a_to_b[a], b_to_a[query] = query, a
                elif t == 'crossmap_break':
                    a, query = event['a_id'], event['b_id']
                    assert a_to_b.get(a) == query and b_to_a.get(query) == a
                    del a_to_b[a], b_to_a[query]
                elif t == 'crossmap_reassign':
                    if before is None:
                        before = dict(b_to_a)
                    old = (event['old_a_id'], event['old_b_id'])
                    assert a_to_b.get(old[0]) == old[1] and b_to_a.get(old[1]) == old[0]
                    del a_to_b[old[0]], b_to_a[old[1]]
                    new = [(x['a_id'], x['b_id']) for x in event['replacements']]
                    assert len(new) == 2
                    for a, query in new:
                        assert a not in a_to_b and query not in b_to_a
                        a_to_b[a], b_to_a[query] = query, a
                    moves.append({'old_pair': old, 'new_pairs': new,
                                  'replacement_ranks': [x.get('rank') for x in event['replacements']],
                                  'net_oracle_tp': sum(pair in truth for pair in new) - (old in truth),
                                  'lost_old_oracle_true_pair': old in truth})
                else:
                    assert not t.startswith('crossmap_'), t
        assert dict.fromkeys(b) | b_to_a == run['assignment']
        with (d / 'crossmap.csv').open() as f:
            csv_pairs = {(row['entity_id'], row['counterparty_id']) for row in csv.DictReader(f)}
        assert csv_pairs == set(a_to_b.items())
        assert len(moves) == run['local_moves']
        validation['wal_equals_final_crossmap_runs'] += 1
        before = b_to_a if before is None else before
        tp_before = sum((a, query) in truth for query, a in before.items())
        tp_after = sum((a, query) in truth for query, a in b_to_a.items())
        validation['within_run_repair_effects'].append({
            'binary': run['binary'], 'shuffle_index': run['shuffle_index'],
            'moves': len(moves), 'net_oracle_tp': tp_after - tp_before,
            'net_oracle_nontruth_pairs': len(moves) - (tp_after - tp_before),
            'lost_old_oracle_true_pairs': sum(move['lost_old_oracle_true_pair'] for move in moves),
            'precision_before': tp_before / len(before), 'precision_after': tp_after / len(b_to_a),
            'moves_detail': moves})
        validation['per_run_cohorts'].append({
            'binary': run['binary'], 'shuffle_index': run['shuffle_index'],
            'groups': {group: {'records': len(ids),
                               'automatically_matched': sum(run['assignment'][query] is not None for query in ids),
                               'oracle_true_pairs': sum((run['assignment'][query], query) in truth for query in ids)}
                       for group, ids in cohorts.items()}})
    validation['cohort_stability'] = {
        label: {group: {'records': len(ids), 'variable_b': sum(s['stability']['per_b_variety'][key] > 1 for key in ids)}
                for group, ids in cohorts.items()}
        for label, s in r['summary']['by_binary'].items()}
    validation['repair_rank_profile'] = repair_rank_profile(validation['within_run_repair_effects'])
    validation['analysis_script_sha256'] = sha(Path(__file__))
    validation['runner_sha256_after'] = sha(REPO / 'benchmarks/assignment_stability/run_test.py')
    validation['data_profile'] = p['data_profile']
    r['parent_validation'] = validation
    r['binary_source_provenance'] = p
    compressed = DEST / f'{args.prefix}-measurements.json.gz'
    with compressed.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
            zipped.write((json.dumps(r, indent=2, sort_keys=True) + '\n').encode())
    (DEST / f'{args.prefix}-validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Independent validation:', validation['wal_equals_final_crossmap_runs'], 'WAL/crossmap folds; all seeds, sources and inputs unchanged')
    print('Cohort stability:', json.dumps(validation['cohort_stability']))
    for label in ['baseline', 'experimental']:
        runs = [run for run in r['runs'] if run['binary'] == label]
        print(label, 'cohort auto medians', {group: statistics.median(
            item['groups'][group]['automatically_matched'] for item in validation['per_run_cohorts'] if item['binary'] == label)
            for group in cohorts})
        print(label, 'TP median', statistics.median(run['truth']['tp'] for run in runs),
              'matches median', statistics.median(run['truth']['matches'] for run in runs),
              'precision median', statistics.median(run['truth']['precision'] for run in runs),
              'recall median', statistics.median(run['truth']['recall'] for run in runs))
    print('Experimental within-run repairs:', [{key:value for key,value in item.items() if key not in ['moves_detail','binary']}
        for item in validation['within_run_repair_effects'] if item['binary'] == 'experimental'])
    print('Durable complete measurements:', compressed, compressed.stat().st_size, 'bytes')


if __name__ == "__main__":
    main()
