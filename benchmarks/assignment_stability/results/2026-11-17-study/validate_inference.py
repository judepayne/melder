#!/usr/bin/env python3
"""Independent frequency-identity checks of the retained study inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    identity = json.loads((here / 'execution_identity.json').read_text())
    root = args.study_root or Path(identity['study_root'])
    analysis = json.loads((root / 'inference/analysis.json').read_text())
    assert __debug__, 'Run without Python optimization'
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[4]
    for path, expected in identity['implementation_sha256'].items():
        # Documentation may acquire final result links after measurements.
        if path.endswith('.py') or path.endswith('config.template.yaml') or path.endswith('protocol.json'):
            assert sha(repo / path) == expected, path
    for path, expected in identity['binary_source_provenance']['experimental_source_sha256'].items():
        assert sha(repo / path) == expected, path
    checks = {'method': 'Per-record frequency identity, independent of pairwise distance matrices',
              'bootstrap_samples_checked_per_mode': [0, 1, 2, 9, 99, 499, 999, 1999, 2999, 4999],
              'modes': {}}
    for mode, folder in [('shuffle', 'shuffled'), ('fixed', 'fixed')]:
        raw = json.loads((root / folder / 'measurements.json').read_text())
        stored = analysis['modes'][mode]
        n = stored['n_paired_runs']
        ordered = {label: sorted((run for run in raw['runs'] if run['binary'] == label),
                                 key=lambda run: run['shuffle_index'])
                   for label in ['baseline', 'experimental']}
        b_ids = sorted(ordered['baseline'][0]['assignment'])
        pairs = n * (n - 1) // 2
        observed = {}
        for label, runs in ordered.items():
            unequal = 0
            for b_id in b_ids:
                counts = Counter(run['assignment'][b_id] for run in runs)
                unequal += pairs - sum(value * (value - 1) // 2 for value in counts.values())
            observed[label] = unequal / pairs
            assert abs(observed[label] - stored['primary'][f'{label}_mean_changed_b']) < 1e-10
        reps = stored['bootstrap']['replicates']
        assert abs(observed['baseline'] - observed['experimental'] - stored['primary']['difference_changed_b']) < 1e-10
        for bs, es, delta, relative in zip(reps['baseline_changed_b'], reps['experimental_changed_b'],
                                            reps['difference_changed_b'], reps['relative_reduction']):
            assert abs(bs - es - delta) < 1e-10
            assert relative is not None and abs(relative - delta / bs) < 1e-10
        rng = random.Random(stored['bootstrap']['seed'])
        sampled = set(checks['bootstrap_samples_checked_per_mode'])
        for replicate in range(stored['bootstrap']['repetitions']):
            weights = Counter(rng.randrange(n) for _ in range(n))
            if replicate not in sampled:
                continue
            for label, runs in ordered.items():
                unequal = 0
                for b_id in b_ids:
                    counts = Counter()
                    for index, weight in weights.items():
                        counts[runs[index]['assignment'][b_id]] += weight
                    unequal += pairs - sum(value * (value - 1) // 2 for value in counts.values())
                actual = stored['bootstrap']['replicates'][f'{label}_changed_b'][replicate]
                assert abs(unequal / pairs - actual) < 1e-10, (mode, label, replicate)
        for point, values, recorded in [
            (stored['primary']['difference_changed_b'],
             stored['bootstrap']['replicates']['difference_changed_b'],
             stored['primary']['basic_95_ci_changed_b']),
            (stored['primary']['relative_reduction'],
             stored['bootstrap']['replicates']['relative_reduction'],
             stored['primary']['relative_basic_95_ci']),
        ]:
            quantiles = statistics.quantiles(values, n=40, method='inclusive')
            expected = [2 * point - quantiles[-1], 2 * point - quantiles[0]]
            assert all(abs(a - b) < 1e-10 for a, b in zip(expected, recorded)), mode
        checks['modes'][mode] = {'paired_runs': n, 'within_binary_pair_count': pairs,
                                 'independent_mean_changed_b': observed,
                                 'independent_basic_ci_matches': True,
                                 'weighted_frequency_bootstrap_matches': True}
    checks['script_sha256'] = sha(Path(__file__))
    (here / 'independent_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Independent frequency means, 40 bootstrap values and both 95% intervals agree.')


if __name__ == '__main__':
    main()
