"""Focused stdlib-only tests; no meld/Cargo processes or benchmark measurements."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("assignment_stability_runner", Path(__file__).with_name("run_test.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
VERIFY_SPEC = importlib.util.spec_from_file_location("assignment_stability_verifier", Path(__file__).with_name("verify_results.py"))
verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verifier)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def csv(self, name, fields, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def crossmap(self, pairs, name="crossmap.csv"):
        return self.csv(name, ("entity_id", "counterparty_id"),
                        [{"entity_id": a, "counterparty_id": b} for a, b in pairs])

    def test_shuffle_reproducibility_preservation_and_round_trip(self):
        rows = [{"counterparty_id": str(i), "name": f'Unicode é, "quoted"\n{i}'} for i in range(40)]
        first = runner.shuffled_rows(rows, 20261117)
        self.assertEqual(first, runner.shuffled_rows(rows, 20261117))
        self.assertNotEqual(first, runner.shuffled_rows(rows, 20261118))
        self.assertEqual(rows, sorted(first, key=lambda row: int(row["counterparty_id"])))
        fields = ["counterparty_id", "name"]
        one = runner.write_shuffle(self.root / "one.csv", fields, rows, 7)
        two = runner.write_shuffle(self.root / "two.csv", fields, rows, 7)
        self.assertEqual(one, two)
        self.assertEqual(rows[0]["counterparty_id"], "0", "Shuffle must not mutate the original list")
        self.assertEqual(runner.read_csv(self.root / "one.csv", fields)[1], runner.shuffled_rows(rows, 7))

    def test_fixed_input_preserves_exact_original_bytes_and_path(self):
        original = self.root / "original.csv"
        original.write_bytes(b'counterparty_id,name\r\nb1,"original quotes"\r\n')
        fields, rows = runner.read_csv(original, ('counterparty_id',))
        with mock.patch.object(runner, 'write_shuffle', side_effect=AssertionError('must not shuffle')):
            first = runner.pair_input('fixed', original, self.root / 'shuffles', fields, rows, 7, 1)
            second = runner.pair_input('fixed', original, self.root / 'shuffles', fields, rows, 8, 2)
        self.assertEqual(first, second)
        self.assertEqual(first[0], original)
        self.assertEqual(first[1], {'csv_sha256': runner.sha256(original), 'input_path': str(original)})
        self.assertFalse((self.root / 'shuffles').exists())
        self.assertEqual(original.read_bytes(), b'counterparty_id,name\r\nb1,"original quotes"\r\n')
        shuffled, metadata = runner.pair_input('shuffle', original, self.root / 'shuffles', fields, rows, 7, 1)
        self.assertNotEqual(shuffled, original)
        self.assertEqual(metadata['seed'], 7)
        self.assertIn('permutation_sha256', metadata)

    def test_reuse_seed_has_no_cold_run_and_preserves_source(self):
        cache = self.root / 'seed'
        cache.mkdir()
        (cache / 'vectors').write_bytes(b'original vector bytes')
        before = runner.file_manifest(cache)
        with mock.patch.object(runner, 'run_binary', side_effect=AssertionError('no cold rebuild')):
            source, manifest, preparation = runner.reuse_seed_cache(cache, self.root / 'preparation')
        self.assertEqual(source, cache.resolve())
        self.assertEqual(manifest, before)
        self.assertEqual(runner.file_manifest(cache), before)
        self.assertTrue(preparation['excluded_from_timing_summaries'])
        self.assertTrue(preparation['reused_without_cold_rebuild'])
        self.assertEqual(preparation['seed_cache_manifest'], before)
        self.assertEqual(preparation['seed_cache_path'], str(cache.resolve()))
        self.assertEqual(json.loads((self.root / 'preparation/seed_cache_manifest.json').read_text()), before)
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(ValueError):
            runner.reuse_seed_cache(empty, self.root / 'invalid_preparation')

    def test_warm_run_refuses_changed_seed_before_launch(self):
        binary = self.root / 'binary'
        binary.write_bytes(b'pinned binary')
        cache = self.root / 'seed'
        cache.mkdir()
        vectors = cache / 'vectors'
        vectors.write_bytes(b'original')
        expected = runner.file_manifest(cache)
        vectors.write_bytes(b'changed outside harness')
        with mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError, 'Seed cache changed'):
                runner.run_binary(binary, runner.sha256(binary), self.root / 'run', '',
                                  self.root / 'a', self.root / 'b', {}, (cache, expected))
        self.assertEqual(vectors.read_bytes(), b'changed outside harness', 'Do not repair or overwrite user cache')

    def test_checkpoint_references_full_outcomes_without_summaries_or_rewrites(self):
        directory = self.root / 'run'
        directory.mkdir()
        run = {'directory': str(directory), 'binary': 'baseline', 'shuffle_index': 1,
               'assignment': {'b1': 'a1'}, 'assignment_sha256': 'outcome'}
        reference = runner.persist_run(run)
        path = Path(reference['path'])
        self.assertEqual(json.loads(path.read_text()), run)
        self.assertEqual(reference['sha256'], runner.sha256(path))
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            runner.persist_run(dict(run, assignment={'b1': None}))
        results = {'status': 'failed', 'runs': [run], 'error': 'second run failed',
                   'active_run': {'binary': 'experimental'}}
        with mock.patch.object(runner, 'summarize', side_effect=AssertionError('not at checkpoints')):
            runner.save_checkpoint(self.root, results, [reference])
            runner.save_checkpoint(self.root, results, [reference])
        checkpoint = json.loads((self.root / 'checkpoint.json').read_text())
        self.assertEqual(checkpoint['run_results'], [reference])
        self.assertEqual(checkpoint['completed_runs'], 1)
        self.assertNotIn('runs', checkpoint)
        self.assertNotIn('summary', checkpoint)
        self.assertEqual(checkpoint['error'], 'second run failed')
        self.assertEqual(path.read_bytes(), original)

    def test_main_reuse_fixed_campaign_and_failure_checkpoint(self):
        here = self.root / 'benchmarks/assignment_stability'
        here.mkdir(parents=True)
        (here / 'config.template.yaml').write_text((runner.HERE / 'config.template.yaml').read_text())
        self.csv('benchmarks/data/dataset_a_10k.csv', ['entity_id'], [{'entity_id': 'a1'}])
        self.csv('benchmarks/data/dataset_b_10k.csv', ['counterparty_id'], [{'counterparty_id': 'b1'}])
        self.crossmap([('a1', 'b1')], 'benchmarks/data/ground_truth_crossmap_10k.csv')
        binary = self.root / 'binary'
        binary.write_bytes(b'pinned')
        cache = self.root / 'seed'
        cache.mkdir()
        (cache / 'vectors').write_bytes(b'seed bytes')
        seed_manifest = runner.file_manifest(cache)
        analyzed = {'assignment': {'b1': 'a1'}, 'assignment_sha256': runner.assignment_hash({'b1': 'a1'}),
                    'truth': runner.truth_metrics({'b1': 'a1'}, {'b1': 'a1'}), 'local_moves': 0}
        for fail in (False, True):
            output = self.root / ('failed' if fail else 'complete')
            calls = []

            def fake_run(binary, binary_hash, directory, template, a, b, env, seed_cache=None):
                self.assertEqual(seed_cache, (cache.resolve(), seed_manifest), 'Never cold-prepare on reuse')
                calls.append((b, b.read_bytes()))
                if fail and len(calls) == 3:
                    raise ValueError('injected failure')
                directory.mkdir()
                return {'directory': str(directory), 'wall_seconds': 1.0}, ''

            argv = ['run_test.py', '--baseline-binary', str(binary), '--experimental-binary', str(binary),
                    '--output', str(output), '--runs', '3', '--input-order', 'fixed', '--seed-cache', str(cache)]
            with mock.patch.object(runner, 'ROOT', self.root), mock.patch.object(runner, 'HERE', here), \
                    mock.patch.object(runner.sys, 'argv', argv), mock.patch.object(runner.sys, 'platform', 'darwin'), \
                    mock.patch.object(runner.os, 'access', return_value=True), \
                    mock.patch.object(runner, 'run_binary', side_effect=fake_run), \
                    mock.patch.object(runner, 'analyze_run', return_value=analyzed), \
                    mock.patch.object(runner, 'summarize', wraps=runner.summarize) as summary, \
                    contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(ValueError, 'injected failure'):
                        runner.main()
                    self.assertEqual(summary.call_count, 0)
                else:
                    runner.main()
                    self.assertEqual(summary.call_count, 1)
            self.assertEqual(len(set(calls)), 1, 'Every run must use identical B path/bytes')
            checkpoint = json.loads((output / 'checkpoint.json').read_text())
            self.assertEqual(checkpoint['completed_runs'], 2 if fail else 6)
            for reference in checkpoint['run_results']:
                self.assertEqual(reference['sha256'], runner.sha256(Path(reference['path'])))
            if fail:
                self.assertEqual(checkpoint['status'], 'failed')
                self.assertIn('injected failure', checkpoint['error'])
                self.assertIn('active_run', checkpoint)
                self.assertFalse((output / 'measurements.json').exists())
            else:
                results = json.loads((output / 'measurements.json').read_text())
                self.assertEqual(len(results['runs']), 6)
                self.assertEqual(verifier.verify_input_orders(results), 'fixed')
                self.assertIn('summary', results)
                results['runs'][-1]['input_path'] = '/different/path.csv'
                with self.assertRaises(AssertionError):
                    verifier.verify_input_orders(results)
            self.assertEqual(runner.file_manifest(cache), seed_manifest)

    def test_verification_modes_and_empty_repair_ranks(self):
        runs = []
        for index in (1, 2):
            order = ('baseline', 'experimental') if index == 1 else ('experimental', 'baseline')
            for label in order:
                runs.append({'binary': label, 'shuffle_index': index,
                             'permutation': {'permutation_sha256': str(index)}})
        results = {'provenance': {'requested_runs': 2}, 'runs': runs}
        self.assertEqual(verifier.verify_input_orders(results), 'shuffle', 'Old schema defaults to shuffle')
        runs[-1]['permutation'] = runs[0]['permutation']
        with self.assertRaises(AssertionError):
            verifier.verify_input_orders(results)
        self.assertEqual(verifier.repair_rank_profile([]), {
            'total_moves': 0, 'moves_using_rank_beyond_three': 0, 'maximum_replacement_rank': None})
        self.assertIsNone(verifier.repair_rank_profile([{'moves': 1, 'moves_detail': [
            {'replacement_ranks': [None, None]}]}])['maximum_replacement_rank'])

    def test_normalized_hash_independent_of_csv_row_order(self):
        a, b = {"a1", "a2"}, {"b1", "b2", "b3"}
        one = runner.load_assignment(self.crossmap([("a1", "b2"), ("a2", "b1")]), a, b)
        two = runner.load_assignment(self.crossmap([("a2", "b1"), ("a1", "b2")], "other.csv"), a, b)
        self.assertEqual(runner.assignment_hash(one), runner.assignment_hash(two))
        self.assertIsNone(one["b3"])
        self.assertEqual(runner.assignment_hash(one), runner.assignment_hash(dict(reversed(list(one.items())))))
        two["b3"] = "a3"
        self.assertNotEqual(runner.assignment_hash(one), runner.assignment_hash(two))
        self.assertEqual(runner.compare_assignments(one, two), {"changed_b": 1, "jaccard": 2 / 3})

    def test_crossmap_bijection_and_valid_ids(self):
        invalid = [[("a1", "b1"), ("a1", "b2")],
                   [("a1", "b1"), ("a2", "b1")],
                   [("a1", "b1"), ("a1", "b1")], [("bad", "b1")], [("a1", "bad")]]
        for pairs in invalid:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                runner.load_assignment(self.crossmap(pairs), {"a1", "a2"}, {"b1", "b2"})
        empty = runner.load_assignment(self.crossmap([]), {"a1"}, {"b1"})
        self.assertEqual(empty, {"b1": None})
        with self.assertRaises(ValueError):
            runner.record_ids([{"id": "x"}, {"id": "x"}], "id")
        with self.assertRaises(ValueError):
            runner.record_ids([{"id": ""}], "id")

    def test_truth_and_pairwise_stats_include_unmatched(self):
        a = {"b1": "a1", "b2": None, "b3": "a3"}
        b = {"b1": "a1", "b2": "a2", "b3": None}
        c = {"b1": "a1", "b2": None, "b3": None}
        summary = runner.stability([a, b, c])
        self.assertEqual(summary["distinct_outcomes"], 3)
        self.assertEqual(summary["changed_b"], {"min": 1, "median": 1, "max": 2})
        self.assertEqual(summary["jaccard"], {"min": 1 / 3, "median": 0.5, "max": 0.5})
        self.assertEqual(summary["per_b_variety"], {"b1": 1, "b2": 2, "b3": 2})
        self.assertEqual(summary["shared_stable_mappings"], {"b1": "a1"})
        truth = {"b1": "a1", "b2": "a2", "b3": None}
        self.assertEqual(runner.truth_metrics(a, truth),
                         {"tp": 1, "fp": 1, "fn": 1, "matches": 2, "precision": 0.5, "recall": 0.5})
        self.assertIsNone(runner.truth_metrics(dict.fromkeys(a), truth)["precision"])
        self.assertEqual(runner.compare_assignments(dict.fromkeys(a), dict.fromkeys(a))["jaccard"], 1.0)
        self.assertIsNone(runner.stability([a])["changed_b"])
        with self.assertRaises(ValueError):
            runner.compare_assignments(a, {"different_b": None})

    def test_data_copy_refuses_differences_and_cache_manifest_checks_bytes(self):
        source = self.root / "original.csv"
        source.write_bytes(b"id,name\r\n1,unchanged\r\n")
        copied = runner.copy_unchanged(source, self.root / "data/copy.csv")
        self.assertEqual(source.read_bytes(), copied.read_bytes())
        self.assertEqual(copied, runner.copy_unchanged(source, copied))
        before = runner.file_manifest(copied.parent)
        copied.write_bytes(b"different")
        self.assertNotEqual(before, runner.file_manifest(copied.parent))
        with self.assertRaises(ValueError):
            runner.copy_unchanged(source, copied)
        self.assertEqual(copied.read_bytes(), b"different", "Do not overwrite differing existing data")

    def test_template_json_quoting_and_missing_placeholders(self):
        template = "\n".join(f"{key}: {token}" for key, token in
                             zip("abcde", ("__DATA_A__", "__DATA_B__", "__CACHE__", "__CROSSMAP__", "__OUTPUT__")))
        paths = [self.root / f'path with "quotes" {i}' for i in range(5)]
        rendered = runner.render_config(template, *paths)
        self.assertEqual([json.loads(line.split(": ", 1)[1]) for line in rendered.splitlines()],
                         [str(p.resolve()) for p in paths])
        with self.assertRaises(ValueError):
            runner.render_config("incomplete", *paths)

    def test_warm_cache_evidence_and_darwin_resources(self):
        log = ('INFO side="A" loaded combined index from cache, source unchanged\n'
               'INFO side=B loaded combined index from cache, all fresh')
        self.assertEqual(len(runner.warm_evidence(log)["cache_hits"]), 2)
        for bad in ("", log.splitlines()[0], log + "\nbuilding combined embedding index",
                    log + "\nupdating embedding cache"):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                runner.warm_evidence(bad)
        self.assertEqual(runner.parse_resources(" 2.12 real 3.5 user 0.7 sys\n 123456 maximum resident set size\n"),
                         {"time_real_seconds": 2.12, "user_seconds": 3.5, "sys_seconds": 0.7,
                          "maximum_rss_bytes": 123456})
        with self.assertRaises(ValueError):
            runner.parse_resources("unknown format")

    def fixture_output(self):
        self.crossmap([("a1", "b1")])
        self.csv("output/relationships.csv", ("entity_id", "counterparty_id", "relationship_type"),
                 [{"entity_id": "a1", "counterparty_id": "b1", "relationship_type": "match"},
                  {"entity_id": "a2", "counterparty_id": "b2", "relationship_type": "review"},
                  {"entity_id": "a2", "counterparty_id": "b1", "relationship_type": "broken"}])
        self.csv("output/unmatched.csv", ("counterparty_id",), [{"counterparty_id": "b3"}])
        (self.root / "output/events.ndjson").write_text(
            '{"type":"crossmap_confirm"}\n{"type":"crossmap_reassign"}\n', encoding="utf-8")
        return "Total B records: 3\nSkipped (crossmap): 0\nAuto-matched: 1\nReview: 1\nNo match: 1\n"

    def test_output_reconciliation_and_wal_moves(self):
        log = self.fixture_output()
        args = (self.root, {"a1", "a2"}, {"b1", "b2", "b3"}, {"b1": "a1", "b2": "a2", "b3": None})
        result = runner.analyze_run(*args, log)
        self.assertEqual(result["local_moves"], 1)
        self.assertEqual(result["truth"]["tp"], 1)
        self.assertEqual(result["truth"]["recall"], 0.5)
        self.assertEqual(result["batch_totals"]["reviews"], 1)
        with self.assertRaises(ValueError):
            runner.analyze_run(*args, log.replace("Auto-matched: 1", "Auto-matched: 2"))
        self.csv("output/unmatched.csv", ("counterparty_id",), [{"counterparty_id": "b2"}])
        with self.assertRaises(ValueError):
            runner.analyze_run(*args, log)

    def test_partial_results_are_atomic_and_include_all_completed_runs(self):
        runs = []
        for label, wall, assignment in (("baseline", 2.0, {"b1": "a1"}),
                                         ("experimental", 3.0, {"b1": None})):
            runs.append({"binary": label, "shuffle_index": 1, "wall_seconds": wall,
                         "assignment": assignment, "assignment_sha256": runner.assignment_hash(assignment),
                         "truth": runner.truth_metrics(assignment, {"b1": "a1"}), "local_moves": 0})
        results = {"status": "running", "runs": runs[:1]}
        runner.save_results(self.root, results)
        self.assertEqual(json.loads((self.root / "measurements.json").read_text())["runs"], runs[:1])
        self.assertEqual(results["summary"]["paired"], [])
        results.update(status="failed", error="test failure", runs=runs)
        runner.save_results(self.root, results)
        pair = results["summary"]["paired"][0]
        self.assertEqual(pair["changed_b"], 1)
        self.assertEqual(pair["experimental_over_baseline_wall"], 1.5)
        self.assertEqual(pair["experimental_minus_baseline"]["matches"], -1)
        self.assertIn("test failure", (self.root / "summary.md").read_text())
        self.assertIn(runner.QUALITY_CAVEAT, (self.root / "summary.md").read_text())
        self.assertFalse((self.root / "measurements.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
