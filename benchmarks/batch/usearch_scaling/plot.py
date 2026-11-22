#!/usr/bin/env python3
"""Plot warm scoring only. Error bars show all repetitions, not confidence intervals."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def fit(x, y):
    a, b = np.polyfit(x, y, 1)
    return a, b, 1 - np.sum((y - (a*x+b))**2) / np.sum((y-y.mean())**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('chart', type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.measurements.open() as f:
        for r in csv.DictReader(f):
            if r['trial'].startswith('warm-'):
                groups[int(r['pool'])].append(float(r['scoring_records_s']))
    n = np.array(sorted(groups)); rates = np.array([np.median(groups[k]) for k in n]); cost = 1000/rates
    a,b,log_r2 = fit(np.log10(n),cost); c,d,linear_r2=fit(n,cost)
    print(f'Cost fit R²: log10(N)={log_r2:.4f}, N={linear_r2:.4f}')
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    fig.suptitle('Blocked USearch + BM25: warm batch scoring', weight='bold')
    for ax, is_cost in zip(axes,[False,True]):
        y=cost if is_cost else rates
        ranges=[1000/np.asarray(groups[k]) if is_cost else groups[k] for k in n]
        ax.errorbar(n,y,yerr=[y-[min(r) for r in ranges],[max(r) for r in ranges]-y],fmt='o' if is_cost else 'o-',capsize=4,color='#086788',label='Warm median (min–max)')
        if is_cost:
            ax.set_xscale('log'); grid=np.geomspace(n.min(),n.max(),200)
            ax.plot(grid,a*np.log10(grid)+b,'--',color='#bd5b2b',label=f'Fit to log(N) · R² {log_r2:.3f}')
            ax.plot(grid,c*grid+d,':',color='#394457',label=f'Fit to N · R² {linear_r2:.3f}')
            ax.set_ylim(bottom=0); ax.legend(frameon=False)
        ax.set_xlabel('Reference pool (log scale)' if is_cost else 'Reference pool (linear scale)'); ax.set_xticks(n,[f'{k//1000}k' if k<1000000 else f'{k/1000000:g}M' for k in n],rotation=30)
        ax.set_ylabel('Scoring seconds per 1,000 queries' if is_cost else 'Queries scored per second');ax.grid(alpha=.22);ax.set_axisbelow(True)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
    fig.savefig(args.chart,dpi=160)


if __name__=='__main__': main()
