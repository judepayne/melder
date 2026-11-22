#!/usr/bin/env python3
"""Regenerate the blocked batch before/after chart from retained warm trials."""
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def main():
    with (HERE/'results/paired.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['experiment'] == 'final-paired' and r['variant'] == 'baseline']
    sizes = sorted({int(r['size']) for r in rows})
    fig, (rate, cost) = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for arm, color in [('baseline', '#9a5324'), ('candidate', '#006a70')]:
        samples = [[float(r['score_records_s']) for r in rows if int(r['size']) == n and r['binary'] == arm] for n in sizes]
        assert all(len(s) == 3 for s in samples)
        for ax in (rate, cost):
            data = samples if ax is rate else [[1000/v for v in s] for s in samples]
            ys = list(map(median, data))
            ax.errorbar(sizes, ys, yerr=[[y-min(s) for s,y in zip(data,ys)], [max(s)-y for s,y in zip(data,ys)]], fmt='o-', capsize=3, label=arm, color=color)
    rate.set_xscale('log'); rate.set_yscale('log'); rate.set_xlabel('Reference records (log scale)'); rate.set_ylabel('Queries scored per second (log scale)')
    cost.set_xlabel('Reference records (linear scale)'); cost.set_ylabel('Aggregate scoring seconds per 1,000 queries')
    for ax in (rate, cost):
        ax.set_xticks(sizes, ['10k','100k','250k','500k','1M'], rotation=25); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle('Reusing blocked BM25 eligibility · same 10,000 queries · M1 Ultra')
    fig.savefig(HERE/'blocked-bm25.png', dpi=150)


if __name__ == '__main__':
    main()
