#!/usr/bin/env python3
"""Focused toy-only harness tests. Never launches meld or builds Rust."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import analyze_results as analysis
import run_test as runner

shared = analysis.shared


def make_fixture():
    groups, a_rows, b_rows, scored = [], [], [], []
    for kind in analysis.FAMILIES[:-1]:
        a = [f"{kind}-a{i}" for i in range(5)]
        b = [f"{kind}-b{i}" for i in range(5)]
        rank = int(kind[-1]) if kind.startswith("rank") else None
        if rank:
            escape = a[rank - 1]
            truth = {b[0]: escape, **dict(zip(b[1:], [x for x in a if x != escape]))}
        elif kind == "unrepairable":
            escape, truth = None, dict(zip(b[1:], a[:4]))
        else:
            escape, truth = None, {}
        g = {"id": kind, "kind": kind, "a_ids": a, "b_ids": b,
             "flexible_b_id": b[0] if kind != "ambiguous" else None,
             "escape_a_id": escape, "escape_rank": rank,
             "truth": truth, "allowed": {x: (a[:] if kind == "ambiguous" else [truth[x]] if x in truth else []) for x in b},
             "known_unmatched_b": [b[0]] if kind == "unrepairable" else [],
             "max_matches": 4 if kind == "unrepairable" else 5}
        groups.append(g)
        a_rows.extend({"entity_id": x, "competition_group": kind, "country_code": "GB"} for x in a)
        b_rows.extend({"counterparty_id": x, "competition_group": kind, "domicile": "GB",
                       "_match_type": "ambiguous" if kind == "ambiguous" else "matched" if x in truth else "unmatched",
                       "_true_a_id": truth.get(x, "")} for x in b)
        for x in b:
            if kind == "ambiguous":
                values = [(y, .9) for y in a]
            elif x == b[0]:
                values = [(y, .95 - i * .01 if kind != "unrepairable" or i < 4 else .5) for i, y in enumerate(a)]
            else:
                values = [(y, .93 if y == truth[x] else .7) for y in a]
                values.sort(key=lambda item: -item[1])
            scored.append({"type": "scored", "query_id": x, "query_side": "b", "reason": None,
                           "candidates": [{"rank": i + 1, "matched_id": y, "score": s,
                                           "field_scores": [{"field_a": "name", "field_b": "name", "score": s, "weight": 1}]} for i, (y, s) in enumerate(values)]})
    return {"schema_version": 1, "groups": groups, "a_rows": a_rows, "b_rows": b_rows}, scored


def make_confirm(a, b, **extra):
    return {"type": "crossmap_confirm", "a_id": a, "b_id": b, "score": .95, "rank": 1, **extra}


def make_move(group="rank2", rank=2):
    return {"type": "crossmap_reassign", "old_a_id": f"{group}-a0", "old_b_id": f"{group}-b0",
            "replacements": [{"a_id": f"{group}-a0", "b_id": f"{group}-b1", "score": .95, "rank": 1},
                             {"a_id": f"{group}-a1", "b_id": f"{group}-b0", "score": .94, "rank": rank}]}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.rows = make_fixture()

    def test_saved_real_scoring_log_uses_serialized_side(self):
        fixture = json.loads((runner.HERE / "fixture.json").read_text())
        rows = [json.loads(line) for line in (runner.HERE / "calibration.scoring_log.ndjson").read_text().splitlines()]
        proof = runner.validate_calibration(fixture, rows)
        self.assertEqual(proof["validated_queries"], 500)
        self.assertTrue(proof["all_ambiguous_tied"])

    def test_full_toy_matrix(self):
        proof = runner.validate_calibration(self.fixture, self.rows)
        self.assertEqual(proof["validated_queries"], 25)
        self.assertTrue(proof["all_ambiguous_tied"])
        self.assertEqual(len(proof["matrix"]), 25)

    def test_fail_closed_matrix_mutations(self):
        cases = {
            "missing query": lambda rows: rows.pop(),
            "duplicate query": lambda rows: rows.append(copy.deepcopy(rows[0])),
            "missing candidate": lambda rows: rows[0]["candidates"].pop(),
            "foreign candidate": lambda rows: rows[0]["candidates"][0].update(matched_id="foreign"),
            "wrong escape rank": lambda rows: rows[0]["candidates"][1].update(rank=3),
            "subthreshold flex": lambda rows: rows[0]["candidates"][-1].update(score=.84),
            "extra constrained qualification": lambda rows: rows[1]["candidates"][1].update(score=.9),
            "missing field scores": lambda rows: rows[0]["candidates"][0].update(field_scores=[]),
            "bypass": lambda rows: rows[0].update(reason="exact"),
            "unrepairable escape qualifies": lambda rows: rows[20]["candidates"][-1].update(score=.86),
            "ambiguous subthreshold": lambda rows: rows[15]["candidates"][-1].update(score=.84),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(self.rows)
                mutate(rows)
                with self.assertRaises(ValueError):
                    runner.validate_calibration(self.fixture, rows)

    def test_non_tied_ambiguous_explicitly_reported(self):
        self.rows[15]["candidates"][0]["score"] = .91
        proof = runner.validate_calibration(self.fixture, self.rows)
        self.assertFalse(proof["all_ambiguous_tied"])
        self.assertGreater(proof["ambiguous_ties"]["ambiguous"]["maximum_within_row_score_spread"], 0)

    def test_real_log_header_and_baseline_bypass_audit(self):
        a_ids = {r["entity_id"] for r in self.fixture["a_rows"]}
        b_ids = {r["counterparty_id"] for r in self.fixture["b_rows"]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "output"
            output.mkdir()
            header = {"type": "header", "top_n": 5, "thresholds": {"auto_match": .85, "review_floor": .60, "min_score_gap": None}}
            score_path = output / "toy.scoring_log.ndjson"
            score_path.write_text("\n".join(json.dumps(r) for r in [header, *self.rows]) + "\n")
            (output / "toy.match_log.ndjson").write_text("")
            proof = runner.audit_preparation(directory, self.fixture, a_ids, b_ids)
            self.assertEqual(proof["baseline_bypass_proof"]["all_b_scored"], 25)
            self.assertTrue((directory / "calibration_matrix.json").is_file())
            header["top_n"] = 3
            score_path.write_text("\n".join(json.dumps(r) for r in [header, *self.rows]) + "\n")
            with self.assertRaisesRegex(ValueError, "header"):
                runner.audit_preparation(directory, self.fixture, a_ids, b_ids)


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.fixture, _ = make_fixture()
        self.truth = {b: a for g in self.fixture["groups"] for b, a in g["truth"].items()}
        self.rows = self.fixture["b_rows"] + [
            {"counterparty_id": "background-ambiguous", "_match_type": "ambiguous", "_true_a_id": "arbitrary"},
            {"counterparty_id": "background-positive", "_match_type": "matched", "_true_a_id": "correct"},
            {"counterparty_id": "background-negative", "_match_type": "unmatched", "_true_a_id": ""}]
        self.truth["background-positive"] = "correct"
        self.oracle = analysis.cohort_oracle(self.fixture, self.rows, self.truth)

    def test_ambiguous_bijection_not_false_positive(self):
        assignment = dict.fromkeys(self.oracle)
        g = next(g for g in self.fixture["groups"] if g["kind"] == "ambiguous")
        assignment.update(zip(g["b_ids"], reversed(g["a_ids"])))
        q = analysis.quality(assignment, self.oracle, "ambiguous")
        self.assertEqual(q["allowed_ambiguous"], 5)
        self.assertEqual(q["incorrect"], 0)
        self.assertEqual(q["known_correct"], 0)
        assignment[g["b_ids"][0]] = "foreign"
        self.assertEqual(analysis.quality(assignment, self.oracle, "ambiguous")["incorrect"], 1)

    def test_background_caveat_and_negative(self):
        assignment = dict.fromkeys(self.oracle)
        assignment.update({"background-ambiguous": "not-arbitrary", "background-positive": "correct", "background-negative": "wrong"})
        q = analysis.quality(assignment, self.oracle, "background")
        self.assertEqual(q["unverified_ambiguous"], 1)
        self.assertEqual(q["known_correct"], 1)
        self.assertEqual(q["incorrect"], 1)
        self.assertEqual(q["matched_known_negative"], 1)

    def test_stability_normalizes_and_counts_none(self):
        a = {"b1": "a1", "b2": None}
        b = {"b1": None, "b2": "a2"}
        s = analysis.normalized_stability([a, b, a])
        self.assertEqual(s["distinct_cohort_mapping_outcomes"], 2)
        self.assertAlmostEqual(s["mean_pairwise_changed_b"], 4 / 3)
        self.assertAlmostEqual(s["mean_pairwise_changed_b_fraction"], 2 / 3)
        self.assertIsNone(analysis.normalized_stability([a])["mean_pairwise_changed_b"])
        self.assertEqual(analysis.normalized_stability([{"b": None}, {"b": None}])["mean_pairwise_changed_b"], 0)
        with self.assertRaises(ValueError):
            analysis.normalized_stability([a, {"b1": "a1"}])


class OrderingTests(unittest.TestCase):
    def test_balanced_factorial_order(self):
        slots, first = Counter(), Counter()
        for index in range(20):
            order = runner.execution_order(index)
            self.assertEqual(set(order), {(n, label) for n in runner.WIDTHS for label in ("baseline", "experimental")})
            for slot in range(4):
                n, label = order[slot * 2]
                slots[n, slot] += 1
                first[n, label] += 1
        self.assertEqual(set(slots.values()), {5})
        self.assertEqual(set(first.values()), {10})

    def test_shared_permutation_bytes_and_no_overwrite(self):
        rows = [{"counterparty_id": f"b{i}", "name": f"comma,{i}"} for i in range(25)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, permutation = shared.pair_input("shuffle", directory / "unused", directory / "shuffles",
                                                  ["counterparty_id", "name"], rows, 20261117, 1)
            for _ in runner.execution_order(0):
                self.assertEqual(shared.sha256(path), permutation["csv_sha256"])
            _, actual = shared.read_csv(path, ("counterparty_id",))
            self.assertEqual(actual, shared.shuffled_rows(rows, 20261117))
            self.assertNotEqual(actual, shared.shuffled_rows(rows, 20261118))
            with self.assertRaises(FileExistsError):
                shared.write_shuffle(path, ["counterparty_id", "name"], rows, 20261117)

    def test_numeric_replacement_precedes_shared_renderer(self):
        template = (runner.HERE / "config.template.yaml").read_text()
        runner.validate_template(template)
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            for n in runner.WIDTHS:
                config = shared.render_config(runner.width_template(template, n), p / "a", p / "b", p / "cache", p / "crossmap", p / "output")
                self.assertIn(f"top_n: {n}\n", config)
                self.assertNotIn("__", config)
                self.assertIn("scoring_log:\n  enabled: false", config)
            self.assertIn("enabled: true\n  compression: none", runner.width_template(template, 5, True))


class WalTests(unittest.TestCase):
    def setUp(self):
        self.fixture, _ = make_fixture()
        self.a_ids = {r["entity_id"] for r in self.fixture["a_rows"]}
        self.b_ids = {r["counterparty_id"] for r in self.fixture["b_rows"]}

    def fold(self, events):
        return analysis.fold_wal(events, self.a_ids, self.b_ids, self.fixture)

    def test_atomic_fold_counts_and_high_ranks(self):
        result = self.fold([make_confirm("rank5-a0", "rank5-b0"), make_move("rank5", 5)])
        self.assertEqual(result["assignment"]["rank5-b0"], "rank5-a1")
        self.assertEqual(result["assignment"]["rank5-b1"], "rank5-a0")
        self.assertEqual(result["local_moves"], 1)
        self.assertEqual(result["moves_by_group"], {"rank5": 1})
        self.assertEqual(result["rank_gt3_count"], 1)
        self.assertEqual(result["replacement_ranks_by_group"], {"rank5": {"1": 1, "5": 1}})

    def test_serialized_lowercase_review_and_no_match_sides(self):
        events = [
            {"type": "review_match", "side": "b", "id": "rank2-b0", "candidate_id": "rank2-a0"},
            {"type": "no_match_below", "query_side": "b", "query_id": "rank2-b1", "best_candidate_id": "rank2-a1"},
        ]
        self.assertEqual(self.fold(events)["local_moves"], 0)
        events[0]["side"] = "a"
        with self.assertRaisesRegex(ValueError, "non-B"):
            self.fold(events)

    def test_protected_incumbents(self):
        for extra in ({"reason": "canonical"}, {"reason": "exact"}, {"reason": "crossmap"}, {"score": None}):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "protected"):
                    self.fold([make_confirm("rank2-a0", "rank2-b0", **extra), make_move()])

    def test_controls_never_repair(self):
        for group in ("ambiguous", "unrepairable"):
            with self.subTest(group=group):
                with self.assertRaisesRegex(ValueError, "control"):
                    self.fold([make_confirm(f"{group}-a0", f"{group}-b0"), make_move(group)])

    def test_invalid_wal_edges_and_counts(self):
        bad_moves = []
        for field, value in (("rank", None), ("score", .84)):
            move = make_move()
            move["replacements"][0][field] = value
            bad_moves.append(move)
        move = make_move()
        move["replacements"][1]["a_id"] = "rank2-a0"
        bad_moves.append(move)
        move = make_move()
        move["replacements"][1]["b_id"] = "rank3-b0"
        bad_moves.append(move)
        for move in bad_moves:
            with self.subTest(move=move):
                with self.assertRaises(ValueError):
                    self.fold([make_confirm("rank2-a0", "rank2-b0"), move])
        for events in ([make_move()],
                       [make_confirm("rank2-a0", "rank3-b0")],
                       [make_confirm("rank2-a0", "rank2-b0"), make_confirm("rank2-a0", "rank2-b1")],
                       [{"type": "crossmap_break", "a_id": "rank2-a0", "b_id": "rank2-b0"}]):
            with self.subTest(events=events):
                with self.assertRaises(ValueError):
                    self.fold(events)

    def test_break_and_reclaim(self):
        result = self.fold([make_confirm("rank2-a0", "rank2-b0"),
                            {"type": "crossmap_break", "a_id": "rank2-a0", "b_id": "rank2-b0"},
                            make_confirm("rank2-a0", "rank2-b1")])
        self.assertIsNone(result["assignment"]["rank2-b0"])
        self.assertEqual(result["assignment"]["rank2-b1"], "rank2-a0")


class PersistenceTests(unittest.TestCase):
    def test_bad_combined_calibration_stops_before_measurement(self):
        fixture, rows = make_fixture()
        rows[0]["candidates"][-1]["score"] = .84
        a_ids = {r["entity_id"] for r in fixture["a_rows"]}
        b_ids = {r["counterparty_id"] for r in fixture["b_rows"]}
        truth = {b: a for g in fixture["groups"] for b, a in g["truth"].items()}
        oracle = analysis.cohort_oracle(fixture, fixture["b_rows"], truth)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "fake-binary"
            binary.write_text("never executed")
            binary.chmod(0o700)
            args = argparse.Namespace(runs=20, rayon_threads=8, output=root / "results", seed=20261117,
                                      baseline_binary=binary, experimental_binary=binary, data=root / "data", fixture=root / "fixture.json")

            def fake_prepare(binary, digest, directory, template, a, b, env):
                directory.mkdir()
                (directory / "cache").mkdir()
                (directory / "cache/combined").write_bytes(b"toy cache")
                (directory / "output").mkdir()
                header = {"type": "header", "top_n": 5, "thresholds": {"auto_match": .85, "review_floor": .60, "min_score_gap": None}}
                (directory / "output/toy.scoring_log.ndjson").write_text("\n".join(json.dumps(r) for r in [header, *rows]))
                (directory / "output/toy.match_log.ndjson").write_text("")
                self.assertIn("enabled: true\n  compression: none", template)
                return {"directory": str(directory)}, ""

            with mock.patch.object(runner.sys, "platform", "darwin"), \
                 mock.patch.object(runner, "load_inputs", return_value=(fixture, [], fixture["b_rows"], a_ids, b_ids, truth, oracle, {})), \
                 mock.patch.object(runner.shared, "run_binary", side_effect=fake_prepare) as launch:
                with self.assertRaisesRegex(ValueError, "qualify all five"):
                    runner.run_benchmark(args)
                self.assertEqual(launch.call_count, 1)
            checkpoint = json.loads((args.output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["status"], "failed")
            self.assertEqual(checkpoint["completed_runs"], 0)
            self.assertNotIn("oracle", checkpoint)
            self.assertNotIn("fixture", checkpoint)
            self.assertFalse((args.output / "measurements.json").exists())
            self.assertFalse((args.output / "shuffles").exists())
            with mock.patch.object(runner.sys, "platform", "darwin"):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    runner.run_benchmark(args)

    def test_mocked_full_factorial_has_one_preparation_and_shared_b(self):
        fixture, rows = make_fixture()
        a_ids = {r["entity_id"] for r in fixture["a_rows"]}
        b_ids = {r["counterparty_id"] for r in fixture["b_rows"]}
        truth = {b: a for g in fixture["groups"] for b, a in g["truth"].items()}
        oracle = analysis.cohort_oracle(fixture, fixture["b_rows"], truth)
        assignment = dict.fromkeys(b_ids)
        invocations = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "fake-binary"
            binary.write_text("never executed")
            binary.chmod(0o700)
            args = argparse.Namespace(runs=2, rayon_threads=8, output=root / "results", seed=20261117,
                                      baseline_binary=binary, experimental_binary=binary, data=root / "data", fixture=root / "fixture.json")

            def fake_run(binary, digest, directory, template, a, b, env, seed_cache=None):
                directory.mkdir()
                output = directory / "output"
                output.mkdir()
                (output / "toy.match_log.ndjson").write_text("")
                if seed_cache is None:
                    (directory / "cache").mkdir()
                    (directory / "cache/combined").write_bytes(b"one cold cache")
                    header = {"type": "header", "top_n": 5, "thresholds": {"auto_match": .85, "review_floor": .60, "min_score_gap": None}}
                    (output / "toy.scoring_log.ndjson").write_text("\n".join(json.dumps(r) for r in [header, *rows]))
                else:
                    shutil.copytree(seed_cache[0], directory / "cache")
                    self.assertEqual(shared.file_manifest(directory / "cache"), seed_cache[1])
                    self.assertFalse((directory / "crossmap.csv").exists())
                config = shared.render_config(template, a, b, directory / "cache", directory / "crossmap.csv", output)
                (directory / "config.yaml").write_text(config)
                invocations.append({"directory": directory, "b": b, "b_hash": shared.sha256(b) if seed_cache else None,
                                    "seed": seed_cache is None, "config": config})
                return {"directory": str(directory), "wall_seconds": 1, "config_sha256": shared.sha256(directory / "config.yaml"),
                        "resources": {"maximum_rss_bytes": 100}}, ""

            def fake_analyze(*unused):
                return {"assignment": assignment.copy(), "assignment_sha256": shared.assignment_hash(assignment),
                        "local_moves": 0, "truth": {"unused": True}}

            with mock.patch.object(runner.sys, "platform", "darwin"), \
                 mock.patch.object(runner, "load_inputs", return_value=(fixture, list(fixture["b_rows"][0]), fixture["b_rows"], a_ids, b_ids, truth, oracle, {})), \
                 mock.patch.object(runner.shared, "run_binary", side_effect=fake_run), \
                 mock.patch.object(runner.shared, "analyze_run", side_effect=fake_analyze):
                summary_path = runner.run_benchmark(args)
            self.assertTrue(summary_path.is_file())
            self.assertEqual(sum(i["seed"] for i in invocations), 1)
            self.assertEqual(len(invocations), 17)
            for offset in (1, 9):
                per_permutation = invocations[offset:offset + 8]
                self.assertEqual(len({i["b"] for i in per_permutation}), 1)
                self.assertEqual(len({i["b_hash"] for i in per_permutation}), 1)
                self.assertTrue(all("scoring_log:\n  enabled: false" in i["config"] for i in per_permutation))
            result = json.loads((args.output / "measurements.json").read_text())
            checkpoint = json.loads((args.output / "checkpoint.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["runs"]), 16)
            self.assertEqual(checkpoint["completed_runs"], 16)
            self.assertEqual(len(list(args.output.glob("shuffle_*/run_result.json"))), 16)
            self.assertNotIn("runs", checkpoint)
            self.assertTrue(all("top_n" in r for r in checkpoint["run_results"]))
            self.assertEqual(shared.file_manifest(args.output / "preparation/cache"), {"combined": shared.sha256(args.output / "preparation/cache/combined")})

    def test_full_outcome_once_light_checkpoint_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_dir = output / "run"
            run_dir.mkdir()
            run = {"directory": str(run_dir), "binary": "baseline", "top_n": 1, "shuffle_index": 1,
                   "assignment": {"b": None}, "assignment_sha256": shared.assignment_hash({"b": None})}
            reference = shared.persist_run(run)
            self.assertEqual(json.loads(Path(reference["path"]).read_text()), run)
            self.assertEqual(reference["sha256"], shared.sha256(Path(reference["path"])))
            with self.assertRaises(ValueError):
                shared.persist_run(run)
            shared.save_checkpoint(output, {"runs": [run], "status": "running"}, [reference])
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertNotIn("runs", checkpoint)
            self.assertNotIn("assignment", checkpoint["run_results"][0])
            self.assertEqual(checkpoint["completed_runs"], 1)
            self.assertFalse(list(output.rglob("*.tmp")))
            shared.atomic_json(output / "checkpoint.json", {"status": "complete"})
            self.assertEqual(json.loads((output / "checkpoint.json").read_text()), {"status": "complete"})

    def test_summary_complete_factorial_toy(self):
        fixture, _ = make_fixture()
        truth = {b: a for g in fixture["groups"] for b, a in g["truth"].items()}
        oracle = analysis.cohort_oracle(fixture, fixture["b_rows"], truth)
        assignment = {b: truth.get(b) for b in oracle}
        # Give every ambiguous query an allowed bijection.
        g = next(g for g in fixture["groups"] if g["kind"] == "ambiguous")
        assignment.update(zip(g["b_ids"], g["a_ids"]))
        runs = []
        for index in (1, 2):
            for n, label in runner.execution_order(index - 1):
                runs.append({"binary": label, "top_n": n, "shuffle_index": index, "assignment": assignment.copy(),
                             "wall_seconds": 1 if label == "baseline" else 2,
                             "resources": {"maximum_rss_bytes": 100},
                             "wal_audit": {"moves_by_group": {}, "replacement_ranks_by_group": {}, "rank_gt3_count": 0}})
        results = {"schema_version": 1, "status": "complete", "fixture": fixture, "oracle": oracle, "runs": runs}
        summary = analysis.summarize(results)
        self.assertEqual(len(summary["by_top_n_binary"]), 8)
        self.assertEqual(len(summary["paired_binary"]), 8)
        self.assertEqual(len(summary["paired_width"]), 12)
        self.assertEqual(len(summary["factorial_interactions"]), 6)
        self.assertEqual(summary["paired_binary"][0]["wall_ratio"], 2)
        self.assertEqual(summary["by_top_n_binary"]["1/baseline"]["cohorts"]["ambiguous"]["quality_mean"]["incorrect"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            analysis.save_results(output, results)
            self.assertEqual(json.loads((output / "measurements.json").read_text())["status"], "complete")
            self.assertIn("not a significance", (output / "summary.md").read_text())
            self.assertFalse(list(output.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
