"""Focused capture checks plus independent replay of retained real artifacts."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'capture.py'
SPEC = importlib.util.spec_from_file_location('capture_graph', SCRIPT)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    def test_rust_round_positive_half_and_neighbors(self):
        for score, expected in [(0.0, 0), (1.0, 1000000000),
                                (0.0000000005, 1),
                                (math.nextafter(0.0000000005, 0), 0),
                                (0.8500000005, 850000001)]:
            self.assertEqual(capture.quantize(score), expected, f'round({score})')
        for score in [float('nan'), float('inf'), -0.01, 1.01]:
            with self.assertRaises(ValueError):
                capture.quantize(score)

    def test_downgraded_list_cannot_promote_lower_auto(self):
        header = {'thresholds': {'auto_match': 0.85, 'review_floor': 0.6,
                                  'min_score_gap': 0.05}}
        entry = {'outcome': 'review', 'reason': 'downgraded', 'candidates': [
            {'rank': 1, 'matched_id': 'A0', 'score': 0.94},
            {'rank': 2, 'matched_id': 'A1', 'score': 0.92},
            {'rank': 3, 'matched_id': 'A2', 'score': 0.91},
            {'rank': 4, 'matched_id': 'A3', 'score': 0.80}]}
        self.assertEqual([c['matched_id'] for c in capture.retained(entry, header, {'A2'})], [])
        header['thresholds']['min_score_gap'] = None
        entry.update(outcome='auto', reason=None)
        self.assertEqual([c['matched_id'] for c in capture.retained(entry, header, set())],
                         ['A0', 'A1', 'A2'])

    def test_single_candidate_has_no_gap_demotion(self):
        header = {'thresholds': {'auto_match': 0.85, 'review_floor': 0.6, 'min_score_gap': 0.1}}
        entry = {'outcome': 'auto', 'reason': None,
                 'candidates': [{'rank': 1, 'matched_id': 'A', 'score': 0.9}]}
        self.assertEqual(len(list(capture.retained(entry, header, set()))), 1)

    def test_preclaim_list_not_accepted_links_and_missing_log_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory / 'output'
            output.mkdir()
            header = {'type': 'header', 'schema': 1, 'top_n': 20,
                      'thresholds': {'auto_match': 0.85, 'review_floor': 0.6, 'min_score_gap': None}}
            entry = {'type': 'scored', 'query_id': 'B', 'query_side': 'b',
                     'outcome': 'auto', 'reason': None, 'candidates': [
                         {'rank': 1, 'matched_id': 'A0', 'score': 0.95},
                         {'rank': 2, 'matched_id': 'A1', 'score': 0.90}]}
            score_log = output / 'job.scoring_log.ndjson'
            score_log.write_text(json.dumps(header) + '\n' + json.dumps(entry) + '\n')
            match_log = output / 'job.match_log.ndjson'
            match_log.write_text(json.dumps({'type': 'crossmap_confirm', 'a_id': 'A0',
                                            'b_id': 'B', 'score': 0.95}) + '\n')
            graph = directory / 'out.graph'
            result = capture.convert(directory, ['A0', 'A1'], ['B'], set(), set(), graph)
            self.assertEqual(result['edges'], 2, 'unaccepted rank 2 must remain in retained graph')
            score_log.write_text(json.dumps(header) + '\n')
            with self.assertRaisesRegex(ValueError, 'missing nonempty scoring entry'):
                capture.convert(directory, ['A0', 'A1'], ['B'], set(), set(), graph)
            match_log.write_text(json.dumps({'type': 'no_match_below', 'query_id': 'B',
                                            'query_side': 'b', 'best_score': None}) + '\n')
            result = capture.convert(directory, ['A0', 'A1'], ['B'], set(), set(), graph)
            self.assertEqual(result['stats']['empty_candidate_queries'], 1,
                             'only independently witnessed empty results may be omitted')

    def test_real_capture_replay(self):
        root = Path('/tmp/melder-approx-capture')
        if not (root / 'summary.json').exists():
            self.skipTest('real capture artifacts not present')
        for size in ('10k', '100k'):
            directory = root / size
            ordinals = json.loads((directory / 'ordinals.json').read_text())
            original = json.loads((directory / 'metadata.json').read_text())
            with tempfile.TemporaryDirectory() as temp:
                graph = Path(temp) / 'retained.graph'
                actual = capture.convert(directory, ordinals['a'], ordinals['b'],
                                         set(ordinals['protected_a']), set(ordinals['protected_b']), graph)
                self.assertEqual(actual['graph_sha256'], original['graph_sha256'], size)
                self.assertEqual(actual['stats'], original['stats'], size)
                self.assertEqual(graph.read_bytes(), (root / f'{size}.graph').read_bytes(), size)
                self.assertTrue(actual['coverage_verified'], size)


if __name__ == '__main__':
    unittest.main()
