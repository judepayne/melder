#!/usr/bin/env python3
"""Capture actual pre-claim Melder top_n graphs; never run an assignment solver.

Only the two existing ordinary warm usearch configurations are supported. Original
crossmaps remain protected; original datasets/configs/caches are never written.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]
SCALE = 1_000_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def inventory(directory):
    return {str(p.relative_to(directory)): sha(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def quantize(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('score outside finite [0,1]')
    # Rust f64::round is halfway away from zero, unlike Python round. Avoid
    # floor(x + .5), which can introduce an extra rounding near a half-integer.
    value = score * SCALE
    integer = math.floor(value)
    return integer + (value - integer >= 0.5)


def ids(path, field):
    with path.open(newline='') as stream:
        values = [row[field] for row in csv.DictReader(stream)]
    if len(values) != len(set(values)):
        raise ValueError('duplicate dataset IDs')
    return sorted(values)


def retained(entry, header, protected_a):
    """Reconstruct per-candidate classification omitted by scoring-log schema 1."""
    candidates = entry['candidates']
    thresholds = header['thresholds']
    gap = thresholds.get('min_score_gap')
    demoted = (len(candidates) >= 2 and
               candidates[0]['score'] >= thresholds['auto_match'] and
               gap is not None and
               candidates[0]['score'] - candidates[1]['score'] < gap)
    expected = ('review' if demoted else 'auto' if candidates[0]['score'] >= thresholds['auto_match']
                else 'review' if candidates[0]['score'] >= thresholds['review_floor'] else 'no_match')
    if entry['outcome'] != expected or (entry.get('reason') == 'downgraded') != demoted:
        raise ValueError('scoring-log classification/gap reconstruction disagrees')
    for index, candidate in enumerate(candidates):
        if candidate['rank'] != index + 1:
            raise ValueError('non-contiguous ranks')
        quantize(candidate['score'])
        if index and (-candidates[index-1]['score'], candidates[index-1]['matched_id']) > (-candidate['score'], candidate['matched_id']):
            raise ValueError('non-canonical pipeline ranking')
        if (candidate['score'] >= thresholds['auto_match'] and not demoted
                and candidate['matched_id'] not in protected_a):
            yield candidate


def convert(directory, a_ids, b_ids, protected_a, protected_b, destination):
    a_ord = {value: i for i, value in enumerate(a_ids)}
    b_ord = {value: i for i, value in enumerate(b_ids)}
    logs = list((directory / 'output').glob('*.scoring_log.ndjson'))
    if len(logs) != 1:
        raise ValueError('expected exactly one uncompressed scoring log')
    entries = {}
    edges = []
    stats = Counter()
    with logs[0].open() as stream:
        header = json.loads(next(stream))
        if header['type'] != 'header' or header['schema'] != 1 or header['top_n'] != 20:
            raise ValueError('unexpected scoring header')
        for line in stream:
            entry = json.loads(line)
            b = entry['query_id']
            if entry['type'] != 'scored' or entry['query_side'] != 'b' or b in entries or b in protected_b or b not in b_ord:
                raise ValueError('invalid or duplicate query')
            if not 1 <= len(entry['candidates']) <= 20:
                raise ValueError('invalid top_n candidate count')
            candidates = entry['candidates']
            if [c['rank'] for c in candidates] != list(range(1, len(candidates) + 1)):
                raise ValueError('noncanonical candidate ranks')
            if len({c['matched_id'] for c in candidates}) != len(candidates):
                raise ValueError('duplicate candidate pair')
            if candidates != sorted(candidates, key=lambda c: (-c['score'], c['matched_id'])):
                raise ValueError('noncanonical score/ID order')
            entries[b] = entry
            stats['scored_queries'] += 1
            stats['candidate_rows'] += len(entry['candidates'])
            stats['gap_demotions'] += entry.get('reason') == 'downgraded'
            for candidate in entry['candidates']:
                if candidate['matched_id'] not in a_ord:
                    raise ValueError('unknown A ID')
                if candidate['score'] >= header['thresholds']['auto_match']:
                    stats['raw_threshold_auto'] += 1
                    stats['threshold_auto_to_protected_a'] += candidate['matched_id'] in protected_a
            for candidate in retained(entry, header, protected_a):
                edges.append((b_ord[b], a_ord[candidate['matched_id']], quantize(candidate['score'])))
    # ScoringLogSender deliberately omits empty results. Establish complete B
    # coverage through independent match-log terminal events, NOT CSV matches.
    match_logs = list((directory / 'output').glob('*.match_log*.ndjson'))
    if len(match_logs) != 1:
        raise ValueError('expected exactly one match log')
    terminals = {}
    protected_seen = set()
    with match_logs[0].open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event['type']
            if kind == 'crossmap_confirm':
                b = event['b_id']
                if event.get('reason') == 'crossmap':
                    if b not in protected_b or event['a_id'] not in protected_a:
                        raise ValueError('unrecognized protected pair')
                    protected_seen.add(b)
                    continue
            elif kind in ('review_match', 'no_match_below'):
                b = event.get('query_id', event.get('id'))
            else:
                continue
            if b in terminals or b in protected_b or b not in b_ord:
                raise ValueError('duplicate/invalid terminal B')
            terminals[b] = event
    free_b = set(b_ids) - protected_b
    if set(terminals) != free_b or protected_seen != protected_b:
        raise ValueError('incomplete query coverage')
    for b in free_b - entries.keys():
        event = terminals[b]
        if event['type'] != 'no_match_below' or event.get('best_score') is not None:
            raise ValueError('missing nonempty scoring entry')
        stats['empty_candidate_queries'] += 1
    edges.sort()
    if len({(b, a) for b, a, _ in edges}) != len(edges):
        raise ValueError('duplicate graph pair')
    with destination.open('w') as stream:
        stream.write(f'{len(b_ids)} {len(a_ids)}\n')
        for b, a, weight in edges:
            stream.write(f'{b} {a} {weight}\n')
    save(directory / 'ordinals.json', {'a': a_ids, 'b': b_ids,
                                      'protected_a': sorted(protected_a), 'protected_b': sorted(protected_b)})
    return {'n_b': len(b_ids), 'n_a': len(a_ids), 'free_b': len(free_b),
            'free_a': len(a_ids) - len(protected_a), 'edges': len(edges),
            'active_b': len({b for b, _, _ in edges}), 'active_a': len({a for _, a, _ in edges}),
            'scale': SCALE, 'graph_sha256': sha(destination), 'stats': dict(stats),
            'scoring_header': header, 'coverage_verified': True,
            'scoring_log_sha256': sha(logs[0]), 'match_log_sha256': sha(match_logs[0])}


def capture(size, output, binary, timeout, unseeded=False):
    original = REPO / f'benchmarks/batch/{size}x{size}_usearch/warm'
    directory = output / size
    directory.mkdir()
    shutil.copytree(original / 'cache', directory / 'cache')
    if unseeded:
        (directory / 'crossmap.csv').write_text('entity_id,counterparty_id\n')
    else:
        shutil.copy2(original / 'crossmap.csv', directory / 'crossmap.csv')
    shutil.copy2(original / 'config.yaml', directory / 'original-config.yaml')
    before = inventory(directory / 'cache')
    text = (original / 'config.yaml').read_text()
    # Deliberately limited replacement rather than a dependency on PyYAML.
    prefix = str(original.relative_to(REPO))
    text = text.replace(prefix + '/crossmap.csv', str(directory / 'crossmap.csv'))
    text = text.replace(prefix + '/cache', str(directory / 'cache'))
    text = text.replace(prefix + '/output', str(directory / 'output'))
    text += '\nscoring_log:\n  enabled: true\n  compression: none\n'
    (directory / 'config.yaml').write_text(text)
    a_path = REPO / f'benchmarks/data/dataset_a_{size}.csv'
    b_path = REPO / f'benchmarks/data/dataset_b_{size}.csv'
    a_ids = ids(a_path, 'entity_id')
    b_ids = ids(b_path, 'counterparty_id')
    with (directory / 'crossmap.csv').open(newline='') as stream:
        pairs = list(csv.DictReader(stream))
    pa = {p['entity_id'] for p in pairs}
    pb = {p['counterparty_id'] for p in pairs}
    if len(pa) != len(pairs) or len(pb) != len(pairs) or not pa <= set(a_ids) or not pb <= set(b_ids):
        raise ValueError('invalid protected crossmap')
    sources = {'config': sha(original / 'config.yaml'), 'a': sha(a_path), 'b': sha(b_path),
               'protected_crossmap': sha(directory / 'crossmap.csv'), 'cache': before}
    save(directory / 'inputs.json', sources)
    command = [str(binary), 'run', '--config', str(directory / 'config.yaml'), '--log-format', 'json']
    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
        result = subprocess.run(command, cwd=REPO, env={**os.environ, 'RAYON_NUM_THREADS': '4'},
                                stdout=stdout, stderr=stderr, timeout=timeout)
    if result.returncode:
        raise ValueError(f'Melder failed: {directory}/stderr.log')
    after = inventory(directory / 'cache')
    save(directory / 'cache-after.json', after)
    if before != after or before != inventory(original / 'cache'):
        raise ValueError('warm cache bytes changed')
    metadata = convert(directory, a_ids, b_ids, pa, pb, output / f'{size}.graph')
    metadata.update({'command': command, 'input_sha256': sources, 'warm_cache_unchanged': True,
                     'protected_pairs': len(pairs), 'ann_candidates': 20, 'top_n': 20})
    save(directory / 'metadata.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/melder-approx-capture'))
    parser.add_argument('--binary', type=Path, default=REPO / 'target/release/meld')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--unseeded', action='store_true', help='Start a new batch with an empty private crossmap; do not change the original.')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = output / 'meld'
    shutil.copy2(args.binary, binary)
    source_paths = ['src/batch/engine.rs', 'src/matching/pipeline.rs', 'src/matching/candidates.rs',
                    'src/output/scoring_log.rs', 'src/config/schema.rs', 'src/models.rs']
    source_hashes = {}
    for name in source_paths:
        path = REPO / name
        if path.is_file():
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            source_hashes[name] = sha(path)
    provenance = {'started_utc': datetime.now(timezone.utc).isoformat(), 'binary_sha256': sha(binary),
                  'capture_sha256': sha(Path(__file__)), 'source_sha256': source_hashes,
                  'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True),
                  'binary_build_provenance': 'Release binary pinned by hash; source snapshots are contextual, not a reproducible-build attestation. See capture/README.md for this campaign build command.',
                  'scope': ('Fresh private crossmap' if args.unseeded else 'Original warm crossmaps protected') + '; full top_n pre-claim logs, not accepted-link CSVs. No solver timings.'}
    save(output / 'provenance.json', provenance)
    summaries = {size: capture(size, output, binary, args.timeout, args.unseeded) for size in ('10k', '100k')}
    save(output / 'summary.json', summaries)
    print(json.dumps({size: {key: value for key, value in summary.items()
                            if key not in ('input_sha256', 'scoring_header')}
                      for size, summary in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
