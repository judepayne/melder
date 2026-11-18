#!/usr/bin/env python3
"""Measure exact vs best on single, connected, balanced synthetic groups."""
import argparse
import gzip
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
from pathlib import Path

SIZES = (64, 128, 256, 512, 768, 1024, 2048, 4096)
DEGREES = (5, 20)
PROFILES = ('eligible', 'strong_diagonal', 'broad')
SEEDS = (3101, 3102, 3103, 3104, 3105)
REPS = 3
SCALE = 10**9


def fixture(vertices, degree, seed):
    n = vertices // 2
    rng = random.Random(seed)
    rows = []
    for b in range(n):
        # Diagonal plus cyclic successor makes a single connected component.
        neighbors = {b, (b + 1) % n}
        while len(neighbors) < degree:
            neighbors.add(rng.randrange(n))
        for a in sorted(neighbors):
            rows.append((b, a, rng.randrange(SCALE)))
    return rows


def score(b, a, raw, profile):
    if profile == 'eligible':
        return 850000000 + raw * 150000000 // (SCALE - 1)
    if profile == 'strong_diagonal' and a == b:
        return 970000000 + raw * 30000000 // (SCALE - 1)
    if profile == 'strong_diagonal':
        return 850000000 + raw * 100000000 // (SCALE - 1)
    return raw + 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.out.resolve() / 'compare'
    shutil.copy2(args.binary, binary)
    graph = args.out / 'current.graph'
    pairs = args.out / 'current.pairs'
    cases = [(v, d, seed, profile) for v in SIZES for d in DEGREES for seed in SEEDS for profile in PROFILES]
    random.Random(9701).shuffle(cases)
    records = []
    for i, (v, d, seed, profile) in enumerate(cases):
        n = v // 2
        weights = {(b, a): score(b, a, x, profile) for b, a, x in fixture(v, d, seed)}
        text = f'{n} {n}\n' + ''.join(f'{b} {a} {w}\n' for (b, a), w in weights.items())
        graph.write_text(text)
        by_mode = {}; runs = []
        for rep in range(REPS):
            order = ['exact','best'] if (rep+i)%2==0 else ['best','exact']
            for mode in order:
                command = [str(binary), str(graph), mode, '1', '256', str(41+rep), str(pairs)]
                result = json.loads(subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout)
                assert result['components'] == 1 and result['largest_component_vertices'] == v and result['edges'] == n*d
                data = pairs.read_bytes(); chosen = [tuple(map(int,line.split())) for line in data.splitlines()]; total = sum(weights[e] for e in chosen)
                assert total == int(result['objective_units']) and len(chosen)==result['selected']
                assert len({a for _,a in chosen}) == len({b for b,_ in chosen}) == len(chosen)
                signature = (total, hashlib.sha256(data).hexdigest())
                assert by_mode.setdefault(mode, signature) == signature
                runs.append([rep,mode,result['solve_s']])
        exact, best = by_mode['exact'][0], by_mode['best'][0]
        assert exact >= best > 0
        if profile == 'strong_diagonal':
            assert best == exact == sum(weights[b,b] for b in range(n))
        records.append(dict(vertices=v, degree=d, profile=profile, seed=seed, graph_sha256=hashlib.sha256(text.encode()).hexdigest(), outcomes=by_mode, timings=runs))
    report = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=platform.python_version(), platform=platform.platform(), records=records)
    (args.out/'raw.json.gz').write_bytes(gzip.compress(json.dumps(report,separators=(',',':')).encode(),mtime=0))
    for v in SIZES:
        for profile in PROFILES:
            vals=[]
            for row in records:
                if (row['vertices'],row['profile'])!=(v,profile): continue
                e,b = [statistics.median(t for _,mode,t in row['timings'] if mode==m)*1000 for m in ('exact','best')]; loss=100*(1-row['outcomes']['best'][0]/row['outcomes']['exact'][0]);vals.append((e,b,e/b,loss,e-b))
            print(v,profile,*(f'{statistics.median(col):.3f}' for col in zip(*vals)),f'maxloss={max(x[3] for x in vals):.3f}')


if __name__ == '__main__':
    main()
