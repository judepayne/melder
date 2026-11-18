"""Cheap fixture, generator, and frozen production-score evidence checks (no inference)."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from calibrate import verify
from generate_data import FIXTURE, HERE, ORIGINAL, build_dataset, read_csv, sha256, validate_fixture


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text())

    def test_contract_and_counts(self):
        validate_fixture(self.fixture)
        groups = self.fixture['groups']
        self.assertEqual(sum(len(g['truth']) for g in groups), 380)
        self.assertEqual(sum(len(g['known_unmatched_b']) for g in groups), 20)
        self.assertEqual(sum(len(g['b_ids']) for g in groups if g['kind'] == 'ambiguous'), 100)
        self.assertEqual(len({r['entity_id'] for r in self.fixture['a_rows']} |
                             {r['counterparty_id'] for r in self.fixture['b_rows']}), 1000)

    def test_reject_broken_contract(self):
        broken = copy.deepcopy(self.fixture)
        broken['b_rows'][0]['competition_group'] = 'background'
        with self.assertRaisesRegex(ValueError, 'B isolation'):
            validate_fixture(broken)

    def test_real_scoring_evidence(self):
        evidence = json.loads((HERE / 'calibration.json').read_text())
        raw = HERE / 'calibration.scoring_log.ndjson'
        records = [json.loads(line) for line in raw.read_text().splitlines()]
        self.assertEqual(evidence['fixture_sha256'], sha256(FIXTURE))
        self.assertEqual(evidence['raw_scoring_log_sha256'], sha256(raw))
        self.assertEqual(evidence['config_sha256'], sha256(HERE / 'calibration.config.yaml'))
        self.assertEqual(verify(self.fixture, records), evidence['groups'])
        production_fields = [('legal_name', 'counterparty_name', 'embedding', .55),
                             ('short_name', 'counterparty_name', 'fuzzy', .20),
                             ('country_code', 'domicile', 'exact', .20),
                             ('lei', 'lei_code', 'exact', .05)]
        for record in records:
            if record.get('type') == 'scored':
                for candidate in record['candidates']:
                    self.assertEqual([(f['field_a'], f['field_b'], f['method'], f['weight'])
                                      for f in candidate['field_scores']], production_fields)

    def test_clone_disclosure_and_graph_properties(self):
        evidence = json.loads((HERE / 'calibration.json').read_text())
        self.assertIn('clones', evidence['template_note'])
        expected = {'rank2': 1, 'rank3': 1, 'rank5': 1, 'ambiguous': 120, 'unrepairable': 0}
        for group in evidence['groups']:
            self.assertEqual(group['complete_bijections'], expected[group['kind']])
            self.assertGreaterEqual(group['min_accepted_score'], .86)
            if group['max_rejected_score'] is not None:
                self.assertLessEqual(group['max_rejected_score'], .84)

    def test_generator_preserves_original_values(self):
        sources = {name: ORIGINAL / name for name in
                   ['dataset_a_10k.csv', 'dataset_b_10k.csv', 'ground_truth_crossmap_10k.csv']}
        before = {name: sha256(path) for name, path in sources.items()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'new'
            identity = build_dataset(output)
            self.assertEqual(identity['counts']['a'], 10500)
            self.assertEqual(identity['counts']['b'], 10500)
            for side, id_field in [('a', 'entity_id'), ('b', 'counterparty_id')]:
                original_fields, original_rows = read_csv(sources[f'dataset_{side}_10k.csv'])
                fields, rows = read_csv(output / f'dataset_{side}.csv')
                self.assertEqual(fields, original_fields + ['competition_group'])
                self.assertEqual([{k: row[k] for k in original_fields} for row in rows[:10000]],
                                 original_rows, 'Original field values/order changed')
                self.assertTrue(all(r['competition_group'] == 'background' for r in rows[:10000]))
                self.assertEqual(len({r[id_field] for r in rows}), 10500)
                self.assertEqual(Counter(r['competition_group'] for r in rows[10000:]),
                                 Counter({g['id']: 5 for g in self.fixture['groups']}))
            _, original_truth = read_csv(sources['ground_truth_crossmap_10k.csv'])
            _, truth = read_csv(output / 'ground_truth.csv')
            self.assertEqual(truth[:len(original_truth)], original_truth)
            self.assertEqual(len(truth), len(original_truth) + 380)
            self.assertEqual({(r['counterparty_id'], r['entity_id']) for r in truth[len(original_truth):]},
                             {(bid, aid) for g in self.fixture['groups'] for bid, aid in g['truth'].items()})
            self.assertEqual(json.loads((output / 'data_identity.json').read_text()), identity)
            for name, metadata in identity['files'].items():
                self.assertEqual(metadata['sha256'], sha256(output / name))
            with self.assertRaises(FileExistsError):
                build_dataset(output)
            sentinel = Path(temporary) / 'sentinel'
            sentinel.write_text('keep me')
            with self.assertRaises(FileExistsError):
                build_dataset(sentinel)
            self.assertEqual(sentinel.read_text(), 'keep me')
        self.assertEqual(before, {name: sha256(path) for name, path in sources.items()})


if __name__ == '__main__':
    unittest.main()
