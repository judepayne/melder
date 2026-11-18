"""Offline unit/static checks; no Rust binary, network, models or cargo."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run


class WorkloadTests(unittest.TestCase):
    def test_mirrors_and_truthful_competition(self):
        gadgets = run.workload(3)
        self.assertEqual(len(gadgets), 6)
        seen = set()
        for g in gadgets:
            self.assertNotEqual(g["side"], g["receiver_side"])
            records = [g[role] for role in ("receiver", "incumbent", "arrival")]
            self.assertEqual(len({r["group"] for r in records}), 1)
            self.assertEqual(len({r["name"] for r in records}), 1)
            self.assertEqual(len({r["stable"] for r in records}), 1)
            self.assertNotEqual(g["incumbent"]["competitive"], g["receiver"]["competitive"])
            self.assertEqual(g["arrival"]["competitive"], g["receiver"]["competitive"])
            for r in records:
                self.assertNotIn(r["id"], seen)
                seen.add(r["id"])
            self.assertNotEqual(run.pair(g, "incumbent"), run.pair(g, "arrival"))
        self.assertEqual({g["side"] for g in gadgets}, {"a", "b"})

    def test_config_isolated_and_no_shortcuts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            enabled = run.config(root, "deferred_acceptance", 32)
            greedy = run.config(root, "greedy", 32)
            self.assertEqual(enabled["live"].pop("assignment")["strategy"], "deferred_acceptance")
            self.assertEqual(greedy["live"].pop("assignment")["strategy"], "greedy")
            self.assertEqual(enabled, greedy, "only assignment policy differs")
            self.assertNotIn("exact_prefilter", enabled)
            self.assertNotIn("model", enabled["embeddings"])
            self.assertTrue(enabled["blocking"]["enabled"])
            fields = enabled["match_fields"]
            self.assertEqual([f["method"] for f in fields], ["bm25", "numeric", "numeric"])
            self.assertAlmostEqual(sum(f["weight"] for f in fields), 1)
            self.assertNotIn("field_a", fields[0], "BM25 must omit scalar field mappings")
            self.assertLess(enabled["thresholds"]["auto_match"], 0.6,
                            "incumbent must qualify even without BM25 contribution")
            for side in ("a", "b"):
                self.assertTrue(Path(enabled["datasets"][side]["path"]).is_relative_to(root))
                self.assertNotIn("common_id_field", enabled["datasets"][side])

    def test_percentiles(self):
        self.assertIsNone(run.percentile([], 99))
        self.assertEqual(run.percentile([5], 50), 5)
        self.assertEqual(run.percentile(list(range(1, 101)), 95), 95)
        self.assertEqual(run.metrics([1, 2, 3], 2)["requests_per_second"], 1.5)

    def test_encoder_guard_never_returns_vectors(self):
        result = subprocess.run([sys.executable, str(run.HERE / "unused_encoder.py")],
                                input='{"type":"encode","texts":["x"]}\n',
                                text=True, capture_output=True, check=True)
        handshake, response = map(json.loads, result.stdout.splitlines())
        self.assertEqual(handshake["protocol_version"], 1)
        self.assertIn("error", response)
        self.assertNotIn("vector_count", response)
        self.assertIn("UNEXPECTED_ENCODE", result.stderr)

    def test_ascii_python_sources(self):
        # The scripts use ASCII fixtures. Markdown reports can contain UTF-8.
        for path in run.HERE.glob("*.py"):
            path.read_bytes().decode("ascii")


if __name__ == "__main__":
    unittest.main()
