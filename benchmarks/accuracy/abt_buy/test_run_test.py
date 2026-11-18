"""Small offline regressions: python3 -m unittest discover -s benchmarks/accuracy/abt_buy."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "abt_buy_runner", Path(__file__).with_name("run_test.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ScoringLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.patch = patch.object(runner, "TEST_DIR", str(self.root))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.current = self.output / "accuracy_abt_buy.graph.match_log.20260101_120000.ndjson"
        self.companion = self.output / f"scores.{self.current.stem}.scoring_log.ndjson"
        self.legacy = self.output / "accuracy_abt_buy.scoring_log.ndjson"
        self.stderr = f"Other diagnostic\nBatch graph log: {self.current}\n"

    def write_scores(self, path):
        events = [
            {"type": "manifest"},
            {"type": "scored", "query_side": "a", "query_id": "a2",
             "candidates": [{"rank": 1, "matched_id": "wrong", "score": 1}]},
            {"type": "scored", "query_side": "b", "query_id": "b1",
             "candidates": [{"rank": 2, "matched_id": "wrong", "score": 0.99},
                            {"rank": 1, "matched_id": "a1", "score": 0.8}]},
            {"type": "scored", "query_side": "b", "query_id": "b2",
             "candidates": [{"rank": 1, "matched_id": "a2", "score": 0.6}]},
            {"type": "scored", "query_side": "b", "query_id": "b3",
             "candidates": []},
        ]
        path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")

    def test_graph_uses_exact_companion_not_stale_or_newer_logs(self):
        self.write_scores(self.companion)
        self.legacy.write_text("invalid stale evidence", encoding="utf-8")
        for stamp in ("20200101_000000", "20990101_000000"):
            (self.output / f"scores.accuracy_abt_buy.graph.match_log.{stamp}.scoring_log.ndjson").write_text(
                "invalid other-run evidence", encoding="utf-8"
            )
        selected = runner.scoring_log_for_run(self.stderr)
        self.assertEqual(selected, self.companion)
        self.assertEqual(runner.load_rank_one_candidates(selected),
                         {"b1": ("a1", 0.8), "b2": ("a2", 0.6)})

    def test_missing_graph_companion_never_falls_back(self):
        self.write_scores(self.legacy)
        (self.output / f"scores.{self.current.stem}.scoring_log.pending").touch()
        with self.assertRaisesRegex(FileNotFoundError, "Scoring log for this run"):
            runner.scoring_log_for_run(self.stderr)

    def test_greedy_or_old_binary_uses_job_wide_log(self):
        self.write_scores(self.legacy)
        self.assertEqual(runner.scoring_log_for_run("legacy diagnostics\n"), self.legacy)

    def test_missing_legacy_log_fails(self):
        with self.assertRaises(FileNotFoundError):
            runner.scoring_log_for_run("")

    def test_ambiguous_run_identity_fails(self):
        with self.assertRaises(ValueError):
            runner.scoring_log_for_run(self.stderr + self.stderr)

    def test_evaluation_uses_selected_scores_and_final_relationships(self):
        self.write_scores(self.companion)
        truth = self.root / "ground_truth.csv"
        truth.write_text("abt_id,buy_id\na1,b1\na2,b2\n", encoding="utf-8")
        (self.output / "relationships.csv").write_text(
            "abt_id,buy_id,relationship_type\na1,b1,match\na2,b2,review\n",
            encoding="utf-8",
        )
        report = io.StringIO()
        with patch.object(runner, "GROUND_TRUTH", str(truth)), contextlib.redirect_stdout(report):
            runner.evaluate(runner.scoring_log_for_run(self.stderr))
        lines = report.getvalue().splitlines()
        for label, value in [("Precision (auto)", "100.0%"),
                             ("Recall (auto)", "50.0%"),
                             ("Combined recall (auto + review)", "100.0%"),
                             ("Oracle F1 (top-1, best threshold)", "100.0%"),
                             ("at threshold", "0.600")]:
            self.assertTrue(any(label in line and line.endswith(value) for line in lines),
                            f"Missing metric {label}: {value}")

    def test_main_captures_run_identity_and_cleans_stale_outputs(self):
        self.legacy.write_text("stale", encoding="utf-8")

        def fake_run(command, **kwargs):
            self.assertFalse(self.legacy.exists(), "Output must be cleaned before running")
            self.assertEqual(kwargs, {"stderr": subprocess.PIPE, "text": True})
            self.write_scores(self.companion)
            return subprocess.CompletedProcess(command, 0, stderr=self.stderr)

        with patch.object(runner.sys, "argv", ["run_test.py", "--binary", __file__]), \
                patch.object(runner, "fetch_data"), \
                patch.object(runner.subprocess, "run", side_effect=fake_run), \
                patch.object(runner, "evaluate") as evaluate, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            runner.main()
        evaluate.assert_called_once_with(self.companion)


if __name__ == "__main__":
    unittest.main()
