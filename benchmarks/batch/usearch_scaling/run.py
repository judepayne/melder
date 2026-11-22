#!/usr/bin/env python3
"""Compare warm scoring on nested stratified pools using one editable YAML recipe."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sample(source, field, sizes, folder, label, seed):
    # Random ranks within each country; interleave by fractional rank so every
    # prefix approximates the country distribution. Emit in source order.
    groups = defaultdict(list)
    with source.open(newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            groups[row[field]].append(i)
    total = sum(map(len, groups.values()))
    if max(sizes) > total:
        raise ValueError(f'{source}: only {total} rows')
    rng = random.Random(seed)
    ranks = [0.] * total
    for country in sorted(groups):
        ids = groups[country]
        rng.shuffle(ids)
        for j, idx in enumerate(ids):
            ranks[idx] = (j + .5) / len(ids)
    order = sorted(range(total), key=ranks.__getitem__)
    position = [0] * total
    for j, idx in enumerate(order):
        position[idx] = j
    paths = {n: folder / f'{label}-{n}.csv' for n in sizes}
    from contextlib import ExitStack
    with ExitStack() as stack:
        reader = csv.reader(stack.enter_context(source.open(newline='')))
        header = next(reader)
        writers = {n: csv.writer(stack.enter_context(p.open('x', newline=''))) for n,p in paths.items()}
        for w in writers.values():
            w.writerow(header)
        for i, row in enumerate(reader):
            for n, w in writers.items():
                if position[i] < n:
                    w.writerow(row)
    return paths


def counts(path, field):
    with path.open(newline='') as f:
        return Counter(row[field] for row in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, default=HERE/'reference.yaml')
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='new directory; never reuse old crossmaps')
    ap.add_argument('--a', type=Path, default=ROOT/'benchmarks/data/dataset_a_1000k.csv')
    ap.add_argument('--b', type=Path, default=ROOT/'benchmarks/data/dataset_b_1000k.csv')
    ap.add_argument('--sizes', type=int, nargs='+', default=[10000,100000,250000,500000,1000000])
    ap.add_argument('--queries', type=int, default=10000)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--encoder-device', choices=['cpu','gpu'], help='override preparation device, not search settings')
    args = ap.parse_args()
    if min(*args.sizes,args.queries,args.repeats)<1 or len(set(args.sizes))!=len(args.sizes):
        ap.error('sizes, queries and repeats must be positive and sizes distinct')
    recipe=args.reference.read_text(); binary=args.binary.resolve(); out=args.out.resolve(); out.mkdir(parents=True)
    if yaml.safe_load(recipe)['vector_backend']!='usearch':
        ap.error('reference must use usearch')
    (out/'reference.yaml').write_text(recipe)
    meta={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'args':vars(args), 'binary_sha256':digest(binary),'a_sha256':digest(args.a),'b_sha256':digest(args.b),'rayon_threads':os.getenv('RAYON_NUM_THREADS'),'ort_threads':os.getenv('ORT_NUM_THREADS')}
    (out/'metadata.json').write_text(json.dumps(meta,default=str,indent=2))
    pools=sample(args.a,'country_code',args.sizes,out,'a',args.seed)
    b=sample(args.b,'domicile',[args.queries],out,'b',args.seed+1)[args.queries]; qb=counts(b,'domicile')
    with (out/'measurements.csv').open('x') as results:
        w=csv.writer(results); w.writerow(['pool','queries','trial','query_weighted_mean_block','scoring_records_s','wall_s']); results.flush()
        for n,a in pools.items():
            ca=counts(a,'country_code'); block=sum(count*ca[k] for k,count in qb.items())/args.queries
            cache=out/str(n)/'cache'; rates=[]
            for trial in range(args.repeats+1):
                p=out/str(n)/('prime' if not trial else f'warm-{trial}'); p.mkdir(parents=True)
                cfg=yaml.safe_load(recipe); cfg['job']['name']=f'scale_{n}'; cfg['job'].pop('description',None)
                for side,path in [('a',a),('b',b)]:
                    cfg['datasets'][side]['path']=str(path); cfg['embeddings'][side+'_cache_dir']=str(cache)
                cfg['cross_map']['path']=str(p/'crossmap.csv'); cfg['output']['csv_dir_path']=str(p/'output')
                if args.encoder_device: cfg['performance']['encoder_device']=args.encoder_device
                path=p/'config.yaml'; path.write_text(yaml.safe_dump(cfg,sort_keys=False)); log=p/'meld.log'; start=time.monotonic()
                with log.open('w') as f: subprocess.run([binary,'run','--config',path],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                wall=time.monotonic()-start; text=log.read_text(); rate=float(re.search(r'Throughput:\s*([\d.]+) records/sec',text)[1]); print(n,p.name,rate,round(wall,2),flush=True)
                if trial and text.count('loaded combined index from cache')<2: raise RuntimeError(f'expected two warm cache hits: {log}')
                w.writerow([n,args.queries,p.name,round(block,2),rate,round(wall,3)]); results.flush()
                if trial: rates.append(rate)
            print('warm median',n,statistics.median(rates),flush=True)


if __name__=='__main__': main()
