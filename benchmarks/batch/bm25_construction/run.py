#!/usr/bin/env python3
"""Warm-cache BM25 construction comparison (requires PyYAML, macOS/Linux).

Example: python3 run.py --source /path/to/scaling/experiment --bin old=/path/release/meld --bin new=/path/release/meld \
    --sizes 100k 1M --repeats 4 --seed 42 --out /tmp/bm25-comparison
Use --limit 0 for construction-only; throughput is then deliberately blank.
No priming/encoding runs are allowed: missing evidence is a fatal error.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[3]
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def inventory(path):
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    require(files, f'empty/missing input: {path}')
    return {str(p.relative_to(path) if path.is_dir() else p.name):
            {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}


def count_rows(path, fields):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(set(fields) <= set(reader.fieldnames or []), f'missing fields: {path}')
        return sum(1 for _ in reader)


def size_arg(value):
    aliases = {'100k': 100000, '1m': 1000000}
    n = aliases.get(value.lower())
    if n is None:
        n = int(value)
    if n not in (100000, 1000000):
        raise argparse.ArgumentTypeError('sizes must be 100k/100000 or 1M/1000000')
    return n


def parse_log(text, size, limit):
    text = ANSI.sub('', text)
    def one(pattern):
        matches = re.findall(pattern, text, re.M)
        require(len(matches) == 1, f'expected one log match for {pattern!r}, got {matches}')
        return matches[0]
    require(int(one(r'Loaded dataset A: (\d+) records')) == size, 'wrong loaded A count')
    require(int(one(r'Loaded (\d+) B records')) == 10000, 'wrong loaded B count')
    build_count, build_ms = one(
        r'BM25 index build complete[^\n]*a_records=(\d+)[^\n]*elapsed_ms="?([\d.]+)')
    require(int(build_count) == size, 'wrong BM25 corpus count')
    for side, count in [('A', size), ('B', 10000)]:
        hits = re.findall(r'loaded combined index from cache, source unchanged[^\n]*'
                          r'side="?' + side + r'"?\s+vecs=(\d+)', text)
        require(hits == [str(count)], f'missing/invalid warm cache hit for {side}: {hits}')
    require(int(one(r'Total B records:\s*(\d+)')) == limit, 'wrong processed B count')
    require(int(one(r'Skipped \(crossmap\):\s*(\d+)')) == 0, 'crossmap was not fresh')
    rate = float(one(r'Throughput:\s*([\d.]+) records/sec')) if limit else ''
    return {'bm25_build_ms': float(build_ms), 'score_records_per_sec': rate if limit else '',
            'corpus_records': int(build_count), 'loaded_b': 10000, 'processed_b': limit,
            'warm_a': True, 'warm_b': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='prepared usearch_scaling experiment directory')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bin', action='append', required=True, metavar='NAME=PATH')
    ap.add_argument('--sizes', type=size_arg, nargs='+', default=[100000, 1000000])
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, choices=[0, 10000], default=10000)
    args = ap.parse_args()
    SOURCE = args.source.resolve()
    require(platform.system() in ('Darwin', 'Linux'), 'RSS units supported only on macOS/Linux')
    require(args.repeats > 0, 'repeats must be positive')
    require(len(set(args.sizes)) == len(args.sizes), 'duplicate sizes')
    bins = {}
    for spec in args.bin:
        name, sep, value = spec.partition('=')
        require(sep and re.fullmatch(r'[A-Za-z0-9_-]+', name), 'use NAME=PATH with a simple NAME')
        path = Path(value).resolve()
        require(name not in bins and path.is_file() and os.access(path, os.X_OK), f'invalid binary: {spec}')
        bins[name] = path
    require(len(bins) >= 2, 'provide at least two binary arms')
    out = args.out.resolve()
    require(not out.exists(), '--out must be a new directory')
    require(not out.is_relative_to(SOURCE), '--out cannot be inside the source experiment')

    # All copying, hashing and config generation are outside process measurements.
    provenance = {'seed': args.seed, 'limit': args.limit, 'repeats': args.repeats,
                  'platform': platform.platform(), 'python': sys.version,
                  'argv': sys.argv, 'source': str(SOURCE),
                  'runner_sha256': digest(Path(__file__)),
                  'environment': {k: os.environ.get(k) for k in ('RAYON_NUM_THREADS', 'ORT_NUM_THREADS')}, 'inputs': {}, 'binaries': {}}
    for name, path in bins.items():
        provenance['binaries'][name] = {'path': str(path), 'sha256': digest(path)}
    originals = {}
    configs = {}
    for size in args.sizes:
        template = SOURCE / str(size) / 'warm-1/config.yaml'
        config = yaml.safe_load(template.read_text())
        require(config['vector_backend'] == 'usearch', 'not a USearch recipe')
        require(config['performance']['vector_index_mode'] == 'load', 'not warm load mode')
        require(config['bm25_fields'] == [
            {'field_a': 'legal_name', 'field_b': 'counterparty_name'},
            {'field_a': 'registered_address', 'field_b': 'counterparty_address'}],
            'unexpected BM25 fields')
        for side, expected, fields in [('a', size, ['legal_name', 'registered_address']),
                                       ('b', 10000, ['counterparty_name', 'counterparty_address'])]:
            path = SOURCE / (f'a-{size}.csv' if side == 'a' else 'b-10000.csv')
            require(count_rows(path, fields) == expected, f'wrong input count: {path}')
            config['datasets'][side]['path'] = str(path)
            originals[path] = inventory(path)
        cache = SOURCE / str(size) / 'cache'
        for side, expected in [('a', size), ('b', 10000)]:
            manifests = list(cache.glob(f'{side}.*.index.manifest'))
            require(len(manifests) == 1, f'missing/ambiguous cache manifest: {cache}/{side}')
            manifest = manifests[0]
            require(json.loads(manifest.read_text())['record_count'] == expected, 'cache corpus mismatch')
            stem = manifest.name.removesuffix('.index.manifest')
            require((cache / f'{stem}.index.texthash').is_file(), 'missing text hash')
            require(any((cache / f'{stem}.usearchdb').glob('*.usearch')), 'missing USearch shards')
        originals[cache] = inventory(cache)
        originals[template] = inventory(template)
        configs[size] = config
    provenance['inputs'] = {str(p): v for p, v in originals.items()}
    out.mkdir(parents=True)
    for name in bins:
        for size in args.sizes:
            shutil.copytree(SOURCE / str(size) / 'cache', out / 'caches' / name / str(size))
    rng = random.Random(args.seed)
    base = list(bins)
    rng.shuffle(base)
    schedule = []
    # Rotate each shuffled block: every arm occupies every position once per
    # complete block, without conflating binary identity with run position.
    for repeat in range(args.repeats):
        if repeat and repeat % len(base) == 0:
            rng.shuffle(base)
        sizes = list(args.sizes)
        rng.shuffle(sizes)
        order = base[repeat % len(base):] + base[:repeat % len(base)]
        for size in sizes:
            schedule.extend({'repeat': repeat + 1, 'size': size, 'arm': name} for name in order)
    provenance['schedule'] = schedule
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with (out / 'raw.csv').open('x', newline='') as raw:
        writer = None
        for index, entry in enumerate(schedule):
            name, size = entry['arm'], entry['size']
            run = out / f'{index:04d}-{name}-{size}-r{entry["repeat"]}'
            run.mkdir()
            config = copy.deepcopy(configs[size])
            config['cross_map']['path'] = str(run / 'crossmap.csv')
            config['output']['csv_dir_path'] = str(run / 'output')
            for side in ('a', 'b'):
                config['embeddings'][f'{side}_cache_dir'] = str(out / 'caches' / name / str(size))
            cfg = run / 'config.yaml'
            cfg.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [str(bins[name]), 'run', '--config', str(cfg), '--limit', str(args.limit)]
            (run / 'command.json').write_text(json.dumps(command))
            env = dict(os.environ, RUST_LOG='info', NO_COLOR='1')
            with (run / 'meld.log').open('wb') as log:
                start = time.perf_counter()
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                _, status, usage = os.wait4(process.pid, 0)
                wall = time.perf_counter() - start
                process.returncode = os.waitstatus_to_exitcode(status)
            require(process.returncode == 0, f'process failed: {run}/meld.log')
            row = dict(index=index, **entry, **parse_log((run / 'meld.log').read_text(), size, args.limit),
                       process_wall_s=wall,
                       peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                       log=str(run / 'meld.log'))
            require(row['peak_rss_bytes'] > 0, 'missing RSS measurement')
            if writer is None:
                writer = csv.DictWriter(raw, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            raw.flush()
    for path, before in originals.items():
        require(inventory(path) == before, f'source changed during experiment: {path}')
    (out / 'complete.json').write_text(json.dumps({'runs': len(schedule), 'sources_unchanged': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f'error: {error}')
