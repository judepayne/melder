"""Lightweight runner tests: no meld workloads or cache copies."""
import csv
from pathlib import Path
import tempfile
import unittest

import run


class RunnerTests(unittest.TestCase):
    def test_canonical_duplicates_scores_and_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.csv'
            def save(rows):
                with path.open('w', newline='') as stream:
                    csv.writer(stream).writerows(rows)
                return run.canonical_csv(path)
            first = save([['id', 'score', 'rank'], ['a', '.7', '1'], ['b', '.8', '2'], ['a', '.7', '1']])
            reordered = save([['rank', 'id', 'score'], ['2', 'b', '.8'], ['1', 'a', '.7'], ['1', 'a', '.7']])
            self.assertEqual(first, reordered)
            removed = save([['id', 'score', 'rank'], ['a', '.7', '1'], ['b', '.8', '2']])
            self.assertNotEqual(first['sha256'], removed['sha256'])
            changed = save([['id', 'score', 'rank'], ['a', '.7', '2'], ['b', '.8', '2'], ['a', '.7', '1']])
            self.assertNotEqual(first['sha256'], changed['sha256'])

    def test_metrics(self):
        text = 'loaded combined index from cache\n' * 2 + 'Throughput: 123.45 records/sec'
        timing = ' 1.25 real 2.0 user 0.1 sys\n 987654 maximum resident set size\n'
        self.assertEqual(run.parse_metrics(text, timing), (123.45, 1.25, 987654, 2))
        with self.assertRaises(ValueError):
            run.parse_metrics(text.replace('loaded combined index from cache', ''), timing)

    def test_recipe_preserves_embedding(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory)
            folder = seed / '10/warm-1'
            folder.mkdir(parents=True)
            cfg = {'vector_backend': 'usearch', 'match_fields': [
                {'method': 'embedding', 'weight': .3, 'field_a': 'name'},
                {'method': 'bm25', 'weight': .5}], 'embeddings': {'model': 'original'}}
            (folder / 'config.yaml').write_text(yaml.safe_dump(cfg))
            mixed = run.recipe(seed, 10, 'exact-mixed')
            self.assertEqual(mixed['embeddings'], cfg['embeddings'])
            self.assertEqual(mixed['match_fields'][0], cfg['match_fields'][0])
            self.assertEqual(mixed['match_fields'][1]['weight'], .45)
            self.assertEqual(mixed['match_fields'][2]['field_b'], 'lei_code')
            self.assertEqual(run.recipe(seed, 10, 'baseline'), cfg)


if __name__ == '__main__':
    unittest.main()
