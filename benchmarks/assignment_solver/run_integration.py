#!/usr/bin/env python3
"""Run fresh-crossmap warm-cache CLI timings with private copies of frozen caches."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from capture import GREEDY_BATCH_YAML
from run import save, sha256

REPO = Path(__file__).resolve().parents[2]


def run_suite(binary, capture, out, repeats, modes=('greedy', 'approximate', 'exact')):
    out.mkdir(parents=True, exist_ok=False)
    pinned = out / 'meld'
    shutil.copy2(binary, pinned)
    report = dict(binary_sha256=sha256(pinned), repeats=repeats, runs=[])
    for size in ('10k', '100k'):
        old = capture / size
        base = (old / 'config.yaml').read_text().removesuffix(GREEDY_BATCH_YAML)
        cache = out / f'{size}-cache'
        shutil.copytree(old / 'cache', cache)
        fingerprint = lambda: {p.name: sha256(p) for p in cache.iterdir() if p.is_file()}
        before = fingerprint()
        for rep in range(repeats):
            offset = rep % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                directory = out / f'{size}-{mode}-{rep}'
                directory.mkdir()
                config = directory / 'config.yaml'
                text = base.replace(str(old.resolve() / 'cache'), str(cache))
                for name in ('crossmap.csv', 'output'):
                    text = text.replace(str(old.resolve() / name), str(directory / name))
                if str(old.resolve()) in text or str(directory / 'crossmap.csv') not in text:
                    raise ValueError('capture paths did not resolve cleanly')
                if mode == 'greedy':
                    text += GREEDY_BATCH_YAML
                else:
                    text += f'\nbatch:\n  assignment:\n    strategy: graph\n    mode: {mode}\n'
                config.write_text(text)
                start = time.perf_counter()
                with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                    subprocess.run([str(pinned), 'run', '--config', str(config)], cwd=REPO,
                                   env={**os.environ, 'RAYON_NUM_THREADS': '4'}, stdout=stdout, stderr=stderr,
                                   timeout=360, check=True)
                elapsed = time.perf_counter() - start
                with (directory / 'crossmap.csv').open(newline='') as f:
                    rows = sorted(tuple(r) for r in list(csv.reader(f))[1:])
                with (directory / 'output' / 'relationships.csv').open(newline='') as f:
                    exported = sorted((r['entity_id'], r['counterparty_id']) for r in csv.DictReader(f) if r['relationship_type'] == 'match')
                if rows != exported:
                    raise ValueError(f'{directory}: output relationships do not match crossmap')
                if fingerprint() != before:
                    raise ValueError(f'{directory}: warm cache changed')
                stdout = (directory / 'stdout.log').read_text()
                summary = [l.strip() for l in stdout.splitlines() if re.search('Groups:|Graph solver time:|Retained edges:|Result:|Total score:', l)]
                report['runs'].append(dict(size=size, mode=mode, rep=rep, seconds=elapsed, pairs=len(rows), pairs_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(), summary=summary))
                save(out/'report.json', report)
                print(size, mode, rep, f'{elapsed:.3f}s', len(rows), *summary, flush=True)
    for size in ('10k', '100k'):
        for mode in modes:
            values = [r['seconds'] for r in report['runs'] if r['size']==size and r['mode']==mode]
            print(size, mode, f'median {statistics.median(values):.3f} s')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, choices=range(1, 10), default=3)
    p.add_argument('--modes', nargs='+', choices=('greedy', 'approximate', 'exact'), default=('greedy', 'approximate', 'exact'))
    args = p.parse_args()
    if len(set(args.modes)) != len(args.modes):
        p.error('--modes must not repeat a mode')
    run_suite(args.binary.resolve(), args.capture.resolve(), args.out.resolve(), args.repeats, args.modes)
