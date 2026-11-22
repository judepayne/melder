#!/usr/bin/env python3
"""Paired warm-cache binary comparison; never modifies the seed experiment."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import yaml

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_csv(path):
    """Hash every cell, including scores/ranks; sort rows, never deduplicate."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(set(header)) != len(header):
            raise ValueError(f'duplicate columns: {path}')
        order = sorted(range(len(header)), key=header.__getitem__)
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f'ragged CSV: {path}')
            rows.append([row[i] for i in order])
    payload = [[header[i] for i in order], sorted(rows)]
    return {'rows': len(rows), 'columns': payload[0], 'sha256': hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}


def recipe(seed, size, variant):
    # Use the actual executed recipe (including CPU override), not stale defaults.
    cfg = yaml.safe_load((seed / str(size) / 'warm-1/config.yaml').read_text())
    if cfg['vector_backend'] != 'usearch':
        raise ValueError('seed must use usearch')
    if variant == 'exact-mixed':
        fields = cfg['match_fields']
        bm25 = [field for field in fields if field['method'] == 'bm25']
        if len(bm25) != 1 or bm25[0]['weight'] <= .05:
            raise ValueError('expected one BM25 field with weight > 0.05')
        bm25[0]['weight'] = round(bm25[0]['weight'] - .05, 12)
        fields.append({'field_a': 'lei', 'field_b': 'lei_code',
                       'method': 'exact', 'weight': .05})
    elif variant == 'exact-shortlist':
        # Keep the two existing vector fields/weights so the identical warm indexes
        # supply real candidates. Isolate LEI scoring; no lexical or synonym work.
        fields = [f for f in cfg['match_fields'] if f['method'] == 'embedding']
        fields.append({'field_a': 'lei', 'field_b': 'lei_code', 'method': 'exact',
                       'weight': round(1 - sum(f['weight'] for f in fields), 12)})
        cfg['match_fields'] = fields
        cfg.pop('bm25_fields', None)
        cfg['ann_candidates'] = 200
    return cfg


def clone_cache(source, target):
    # APFS clone files have independent inodes: writes cannot corrupt seed/sibling.
    result = subprocess.run(['/bin/cp', '-cR', str(source), str(target)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return 'apfs-cow'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return 'full-copy'


def parse_metrics(text, timing):
    rate = re.search(r'Throughput:\s*([\d.]+) records/sec', text)
    wall = re.search(r'([\d.]+)\s+real\b', timing)
    rss = re.search(r'(\d+)\s+maximum resident set size', timing)
    hits = text.count('loaded combined index from cache')
    if not (rate and wall and rss) or hits != 2:
        raise ValueError(f'missing metrics or expected exactly two cache hits (got {hits})')
    return float(rate[1]), float(wall[1]), int(rss[1]), hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='must not exist')
    parser.add_argument('--seed', type=Path, required=True, help='completed usearch_scaling run directory')
    parser.add_argument('--sizes', type=int, nargs='+', default=[10000, 100000, 250000, 500000, 1000000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'exact-mixed', 'exact-shortlist'],
                        default=['baseline', 'exact-mixed'])
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('requires macOS /usr/bin/time -l (RSS is bytes)')
    if args.repeats < 1 or not args.sizes or min(args.sizes) < 1:
        parser.error('sizes and repeats must be positive')
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.variants)) != len(args.variants):
        parser.error('sizes and variants must be distinct')
    seed = args.seed.resolve()
    binaries = {name: getattr(args, name).resolve() for name in ('baseline', 'candidate')}
    for binary in binaries.values():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f'not executable: {binary}')
    # Validate everything before creating output or launching any workload.
    configs = {}
    inputs = {'b': seed / 'b-10000.csv'}
    for size in args.sizes:
        inputs[str(size)] = seed / f'a-{size}.csv'
        if not (seed / str(size) / 'cache').is_dir():
            parser.error(f'missing seed cache for {size}')
        for variant in args.variants:
            configs[size, variant] = recipe(seed, size, variant)
    for path in inputs.values():
        if not path.is_file():
            parser.error(f'missing fixed sample: {path}')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = {'args': vars(args), 'platform': platform.platform(),
            'binary_sha256': {k: digest(v) for k, v in binaries.items()},
            'input_sha256': {k: digest(v) for k, v in inputs.items()},
            'seed_metadata': json.loads((seed / 'metadata.json').read_text()),
            'cache_policy': 'fresh private COW clone or full copy per invocation; no hard links',
            'hash_policy': 'all CSV cells as strings; sorted columns/rows, duplicates preserved'}
    # Store only performance-relevant environment, not credentials.
    meta['environment'] = {k: v for k, v in os.environ.items()
                           if k.startswith(('RAYON_', 'ORT_', 'OMP_', 'VECLIB_'))}
    (out / 'metadata.json').write_text(json.dumps(meta, default=str, indent=2))
    columns = ['size', 'variant', 'repeat', 'position', 'binary', 'score_records_s',
               'process_wall_s', 'max_rss_bytes', 'elapsed_s', 'cache_hits',
               'cache_copy', 'returncode', 'error', 'artifacts']
    pair_index = 0
    with (out / 'results.csv').open('x', newline='') as stream, (out / 'comparisons.jsonl').open('x') as comparisons:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        stream.flush()
        for size in args.sizes:
            for variant in args.variants:
                for repeat in range(1, args.repeats + 1):
                    order = ['baseline', 'candidate'] if pair_index % 2 == 0 else ['candidate', 'baseline']
                    pair_index += 1
                    semantics = {}
                    for position, name in enumerate(order, 1):
                        run = out / str(size) / variant / f'{repeat}-{position}-{name}'
                        run.mkdir(parents=True)
                        mode = clone_cache(seed / str(size) / 'cache', run / 'cache')
                        cfg = copy.deepcopy(configs[size, variant])
                        for side, path in [('a', inputs[str(size)]), ('b', inputs['b'])]:
                            cfg['datasets'][side]['path'] = str(path)
                            cfg['embeddings'][side + '_cache_dir'] = str(run / 'cache')
                        cfg['cross_map']['path'] = str(run / 'crossmap.csv')
                        cfg['output']['csv_dir_path'] = str(run / 'output')
                        config = run / 'config.yaml'
                        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
                        timing = run / 'time.txt'
                        start = time.monotonic()
                        with (run / 'meld.log').open('x') as log:
                            result = subprocess.run(['/usr/bin/time', '-l', '-o', str(timing),
                                                     str(binaries[name]), 'run', '--config', str(config)],
                                                    stdout=log, stderr=subprocess.STDOUT)
                        row = dict(size=size, variant=variant, repeat=repeat, position=position,
                                   binary=name, elapsed_s=time.monotonic() - start, cache_copy=mode,
                                   returncode=result.returncode, artifacts=str(run.relative_to(out)))
                        try:
                            if result.returncode:
                                raise ValueError(f'binary exited {result.returncode}')
                            rate, wall, rss, hits = parse_metrics((run / 'meld.log').read_text(), timing.read_text())
                            row.update(score_records_s=rate, process_wall_s=wall,
                                       max_rss_bytes=rss, cache_hits=hits)
                            files = sorted((run / 'output').glob('*.csv'))
                            if not files:
                                raise ValueError('no semantic CSV outputs')
                            semantics[name] = {p.name: canonical_csv(p) for p in files}
                            relationships = semantics[name].get('relationships.csv', {})
                            if not {'score', 'rank'}.issubset(relationships.get('columns', [])):
                                raise ValueError('relationships CSV must include scores and ranks')
                            (run / 'semantics.json').write_text(json.dumps(semantics[name], indent=2))
                        except Exception as exc:
                            row['error'] = str(exc)
                            semantics.pop(name, None)
                        writer.writerow(row)
                        stream.flush()
                        print(size, variant, repeat, name, row.get('error', 'ok'), flush=True)
                    comparison = dict(size=size, variant=variant, repeat=repeat,
                                      equal=semantics['baseline'] == semantics['candidate'] if len(semantics) == 2 else None,
                                      semantics=semantics)
                    comparisons.write(json.dumps(comparison) + '\n')
                    comparisons.flush()


if __name__ == '__main__':
    main()
