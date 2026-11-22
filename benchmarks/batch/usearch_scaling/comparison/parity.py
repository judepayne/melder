#!/usr/bin/env python3
"""Read retained CSV/WAL only; never launches meld or modifies trial inputs."""
import argparse
import collections
import csv
import itertools
import json
from pathlib import Path


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def load(trial):
    rel = rows(trial / 'output/relationships.csv')
    unmatched = rows(trial / 'output/unmatched.csv')
    # Natural identities exclude score/rank; multiplicity is checked separately.
    relationships = {(r['counterparty_id'], r['entity_id'], r['relationship_type']): r for r in rel}
    assert len(relationships) == len(rel), 'duplicate relationship identity'
    un = {r['counterparty_id']: r for r in unmatched}
    assert len(un) == len(unmatched), 'duplicate unmatched query'
    wal = {}
    events = collections.Counter()
    for path in sorted((trial / 'output').glob('*.ndjson')):
        with path.open() as f:
            for line in f:
                e = json.loads(line)
                t = e['type']
                events[t] += 1
                if t == 'crossmap_confirm':
                    key, score = (e['b_id'], e['a_id'], 'match'), e['score']
                elif t == 'review_match':
                    key, score = (e['id'], e['candidate_id'], 'review'), e['score']
                elif t == 'no_match_below':
                    key, score = (e['query_id'], '', 'unmatched'), e.get('best_score')
                else:
                    continue
                assert key not in wal, 'duplicate WAL decision'
                wal[key] = dict(score=score, rank=e.get('rank'), reason=e.get('reason'))
    csv_keys = set(relationships) | {(q, '', 'unmatched') for q in un}
    assert csv_keys == set(wal), 'CSV/WAL decision discrepancy'
    cross = rows(trial / 'crossmap.csv')
    assert {(r['counterparty_id'], r['entity_id']) for r in cross} == {
        (q, a) for q, a, t in relationships if t == 'match'}, 'CrossMap differs'
    return dict(name=trial.name, rel=relationships, un=un, wal=wal,
                events=dict(events), counts=dict(collections.Counter(r['relationship_type'] for r in rel)),
                unmatched=len(un), crossmap=len(cross))


def numeric(a, b, field):
    deltas = [abs(float(a[k][field]) - float(b[k][field])) for k in a.keys() & b.keys()
              if a[k].get(field) is not None and b[k].get(field) is not None]
    nonzero = sorted(x for x in deltas if x)
    return dict(common=len(deltas), changed=len(nonzero), max=max(nonzero, default=0),
                mean=sum(nonzero) / len(nonzero) if nonzero else 0,
                over_1e_7=sum(x > 1e-7 for x in nonzero), over_1e_6=sum(x > 1e-6 for x in nonzero))


def compare(a, b):
    result = dict(left=a['name'], right=b['name'])
    for kind in ('rel', 'un', 'wal'):
        x, y = a[kind], b[kind]
        common = x.keys() & y.keys()
        result[kind] = dict(left_only=len(x.keys() - y.keys()), right_only=len(y.keys() - x.keys()),
                            changed=sum(x[k] != y[k] for k in common))
        result[kind]['scores'] = numeric(x, y, 'best_score' if kind == 'un' else 'score')
        if kind != 'un':
            result[kind]['rank_changes'] = sum(x[k].get('rank') != y[k].get('rank') for k in common)
            result[kind]['reason_changes'] = sum(x[k].get('reason') != y[k].get('reason') for k in common)
    for t in ('match', 'review', 'unmatched'):
        x = {k for k in a['wal'] if k[2] == t}
        y = {k for k in b['wal'] if k[2] == t}
        result[t + '_symmetric_difference'] = len(x ^ y)
    # Query-level category transitions, distinct from candidate-edge changes.
    def categories(run):
        d = collections.defaultdict(set)
        for q, _, t in run['wal']:
            d[q].add(t)
        return d
    x, y = categories(a), categories(b)
    result['query_category_changes'] = sum(x[q] != y[q] for q in x.keys() | y.keys())
    examples = []
    for k in sorted(a['wal'].keys() & b['wal'].keys()):
        x, y = a['wal'][k], b['wal'][k]
        if x['score'] != y['score']:
            examples.append(dict(key=k, left=x, right=y))
    examples.sort(key=lambda e: abs(e['left']['score'] - e['right']['score']), reverse=True)
    result['wal_score_examples'] = examples[:3]
    result['wal_relationship_scores'] = numeric(
        {k: v for k, v in a['wal'].items() if k[2] != 'unmatched'},
        {k: v for k, v in b['wal'].items() if k[2] != 'unmatched'}, 'score')
    result['decision_set_examples'] = dict(
        left_only=[dict(key=k, value=a['wal'][k]) for k in sorted(a['wal'].keys() - b['wal'].keys())],
        right_only=[dict(key=k, value=b['wal'][k]) for k in sorted(b['wal'].keys() - a['wal'].keys())])
    result['unmatched_changed_cells'] = dict(collections.Counter(
        field for q in a['un'].keys() & b['un'].keys() for field in a['un'][q]
        if a['un'][q][field] != b['un'][q][field]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path, help='private exact-e2e results directory')
    p.add_argument('--out', type=Path, default=Path(__file__).with_name('parity-results.json'))
    args = p.parse_args()
    output = dict(root=str(args.root), sizes={})
    for size in (10000, 100000, 1000000):
        trials = [load(t) for t in sorted((args.root / str(size) / 'exact-mixed').iterdir()) if t.is_dir()]
        assert len(trials) == 6
        comparisons = []
        for a, b in itertools.combinations(trials, 2):
            c = compare(a, b)
            arms = sorted([a['name'].split('-')[-1], b['name'].split('-')[-1]])
            c['arms'] = '/'.join(arms)
            comparisons.append(c)
        output['sizes'][size] = dict(trials=[{k: v for k, v in t.items() if k not in ('rel', 'un', 'wal')} for t in trials], comparisons=comparisons)
        print(size, [(t['name'], t['counts'], t['unmatched']) for t in trials])
        for arms in ('baseline/baseline', 'candidate/candidate', 'baseline/candidate'):
            cs = [c for c in comparisons if c['arms'] == arms]
            def extent(f):
                values = [f(c) for c in cs]
                return [min(values), max(values)]
            print(arms, 'CSV rel changed', extent(lambda c: c['rel']['changed']),
                  'unmatched changed', extent(lambda c: c['un']['changed']),
                  'WAL changed', extent(lambda c: c['wal']['changed']),
                  'max delta', extent(lambda c: c['wal']['scores']['max']),
                  'decision/set/rank', max(c['query_category_changes'] + c['rel']['left_only'] + c['rel']['right_only'] + c['un']['left_only'] + c['un']['right_only'] + c['wal']['rank_changes'] for c in cs))
    args.out.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
