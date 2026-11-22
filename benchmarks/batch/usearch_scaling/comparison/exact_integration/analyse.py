#!/usr/bin/env python3
"""Recompute mixed parity details from retained private CSV/WAL outputs."""
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parity

root = Path(sys.argv[1])
output = {}
for size in (100000, 1000000):
    trials = [parity.load(t) for t in sorted((root / str(size) / 'exact-mixed').iterdir()) if t.is_dir()]
    assert len(trials) == 6
    comparisons = []
    for a, b in itertools.combinations(trials, 2):
        c = parity.compare(a, b)
        c['arms'] = '/'.join(sorted([a['name'].split('-')[-1], b['name'].split('-')[-1]]))
        comparisons.append(c)
    output[size] = dict(trials=[{k: v for k, v in t.items() if k not in ('rel', 'un', 'wal')} for t in trials], comparisons=comparisons)
Path(__file__).with_name('parity-details.json').write_text(json.dumps(output, indent=2))
