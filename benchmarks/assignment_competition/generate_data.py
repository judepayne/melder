#!/usr/bin/env python3
"""Append the calibrated, isolated fixture without changing original field values."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'data'
FIXTURE = HERE / 'fixture.json'


def sha256(path):
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    """Return CSV fields and dictionaries, preserving all string values."""
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)


def validate_fixture(fixture):
    """Validate the stable fixture contract and aggregate oracle counts."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(fixture['schema_version'] == 1, 'Unsupported fixture schema')
    groups = fixture['groups']
    require(Counter(g['kind'] for g in groups) ==
            Counter(dict.fromkeys(['rank2', 'rank3', 'rank5', 'ambiguous', 'unrepairable'], 20)),
            'Expected 20 groups of each family')
    a = {r['entity_id']: r for r in fixture['a_rows']}
    b = {r['counterparty_id']: r for r in fixture['b_rows']}
    require(len(a) == len(fixture['a_rows']) == 500, 'Expected 500 unique A rows')
    require(len(b) == len(fixture['b_rows']) == 500, 'Expected 500 unique B rows')
    require(all(i.startswith('COMP-') for i in list(a) + list(b)), 'Non-COMP ID')
    require(len({g['id'] for g in groups}) == 100, 'Duplicate group IDs')
    require(Counter(i for g in groups for i in g['a_ids']) == Counter(a.keys()), 'A membership')
    require(Counter(i for g in groups for i in g['b_ids']) == Counter(b.keys()), 'B membership')
    for g in groups:
        require(len(g['a_ids']) == len(g['b_ids']) == 5, 'Expected 5A/5B per group')
        require(set(g['allowed']) == set(g['b_ids']), 'Incomplete allowed oracle')
        require(all(a[i]['competition_group'] == g['id'] for i in g['a_ids']), 'A isolation')
        require(all(b[i]['competition_group'] == g['id'] for i in g['b_ids']), 'B isolation')
        require({a[i]['country_code'] for i in g['a_ids']} ==
                {b[i]['domicile'] for i in g['b_ids']} == {'GB'}, 'Country must be real GB')
        for bid, aid in g['truth'].items():
            require(bid in g['b_ids'] and aid in g['a_ids'], 'Truth outside group')
            require(g['allowed'][bid] == [aid] and b[bid]['_true_a_id'] == aid and
                    b[bid]['_match_type'] == 'matched', 'Truth row mismatch')
        require(len(set(g['truth'].values())) == len(g['truth']), 'Non-bijective truth')
        if g['kind'] == 'ambiguous':
            require(not g['truth'] and not g['known_unmatched_b'] and
                    g['flexible_b_id'] is None and g['escape_a_id'] is None and
                    g['escape_rank'] is None and g['max_matches'] == 5, 'Ambiguous metadata')
            require(all(set(g['allowed'][bid]) == set(g['a_ids']) and
                        b[bid]['_true_a_id'] == '' and b[bid]['_match_type'] == 'ambiguous'
                        for bid in g['b_ids']), 'Ambiguous oracle')
        elif g['kind'] == 'unrepairable':
            bid = g['flexible_b_id']
            require(g['known_unmatched_b'] == [bid] and g['allowed'][bid] == [] and
                    len(g['truth']) == 4 and g['max_matches'] == 4 and
                    b[bid]['_true_a_id'] == '' and b[bid]['_match_type'] == 'unmatched' and
                    g['escape_a_id'] is None and g['escape_rank'] is None, 'Negative oracle')
        else:
            require(g['escape_rank'] == int(g['kind'][-1]) and
                    g['truth'][g['flexible_b_id']] == g['escape_a_id'] and
                    len(g['truth']) == 5 and not g['known_unmatched_b'] and
                    g['max_matches'] == 5, 'Repair metadata')
    require(sum(len(g['truth']) for g in groups) == 380, 'Expected 380 custom positives')
    return fixture


def build_dataset(output, fixture_path=FIXTURE, original_dir=ORIGINAL):
    """Write a new master dataset directory; never overwrite any existing directory.

    Returns the JSON-serializable data_identity dictionary. The output filenames
    are dataset_a.csv, dataset_b.csv, ground_truth.csv, and data_identity.json.
    """
    output, fixture_path, original_dir = Path(output), Path(fixture_path), Path(original_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    fixture = validate_fixture(json.loads(fixture_path.read_text(encoding='utf-8')))
    sources = {'a': original_dir / 'dataset_a_10k.csv',
               'b': original_dir / 'dataset_b_10k.csv',
               'truth': original_dir / 'ground_truth_crossmap_10k.csv'}
    af, a = read_csv(sources['a'])
    bf, b = read_csv(sources['b'])
    tf, truth = read_csv(sources['truth'])
    if len(a) != 10000 or len(b) != 10000:
        raise ValueError('Original input must have exactly 10000 rows per side')
    if 'competition_group' in af or 'competition_group' in bf:
        raise ValueError('Original inputs unexpectedly already contain competition_group')
    for rows, custom, id_field in [(a, fixture['a_rows'], 'entity_id'),
                                    (b, fixture['b_rows'], 'counterparty_id')]:
        if len({r[id_field] for r in rows + custom}) != 10500:
            raise ValueError('Original and custom IDs overlap or contain duplicates')
    a = [dict(row, competition_group='background') for row in a] + fixture['a_rows']
    b = [dict(row, competition_group='background') for row in b] + fixture['b_rows']
    truth += [{'entity_id': aid, 'counterparty_id': bid}
              for g in fixture['groups'] for bid, aid in g['truth'].items()]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'dataset_a.csv', af + ['competition_group'], a)
    write_csv(output / 'dataset_b.csv', bf + ['competition_group'], b)
    write_csv(output / 'ground_truth.csv', tf, truth)
    identity = {
        'schema_version': 1,
        'fixture_sha256': sha256(fixture_path),
        'sources': {key: {'path': str(path.resolve()), 'sha256': sha256(path)}
                    for key, path in sources.items()},
        'files': {name: {'sha256': sha256(output / name), 'rows': count}
                  for name, count in [('dataset_a.csv', len(a)), ('dataset_b.csv', len(b)),
                                      ('ground_truth.csv', len(truth))]},
        'counts': {'a': len(a), 'b': len(b), 'ground_truth': len(truth),
                   'custom_a': 500, 'custom_b': 500, 'custom_known_positive': 380,
                   'custom_ambiguous': 100, 'custom_known_unmatched': 20},
        'isolation': 'AND(country_code=domicile, competition_group=same); '
                     'original rows background; custom rows per-group',
    }
    (output / 'data_identity.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory (must not exist)')
    args = parser.parse_args()
    identity = build_dataset(args.output)
    print(json.dumps(identity['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
