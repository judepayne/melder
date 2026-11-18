"""Cheap stdlib tests for the predeclared whole-run study analysis."""

import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_study as study


METADATA = {"p": {"_match_type": "matched", "_true_a_id": "a"},
            "u": {"_match_type": "unmatched", "_true_a_id": ""},
            "x": {"_match_type": "ambiguous", "_true_a_id": "c"}}
TRUTH = {"p": "a"}
A = {"p": "a", "u": None, "x": None}
B = {"p": "b", "u": None, "x": None}
C = {"p": None, "u": "a", "x": "c"}


def make_raw(baseline, experimental=None, mode="shuffle"):
    experimental = baseline if experimental is None else experimental
    runs = []
    for label, assignments in zip(study.LABELS, (baseline, experimental)):
        for i, assignment in enumerate(assignments, 1):
            permutation = {"csv_sha256": str(i) if mode == "shuffle" else "fixed",
                           "input_path": str(i) if mode == "shuffle" else "fixed"}
            if mode == "shuffle":
                permutation["permutation_sha256"] = str(i)
            runs.append({"binary": label, "shuffle_index": i, "assignment": dict(assignment),
                         "permutation": permutation, "wall_seconds": 1 if label == "baseline" else 2,
                         "resources": {"maximum_rss_bytes": 1024}, "local_moves": 0})
    return {"status": "complete", "provenance": {"input_order": mode, "requested_runs": len(baseline)}, "runs": runs}


def analyze(raw, repetitions=100, seed=17):
    return study.analyze_mode(raw, raw["provenance"]["input_order"], METADATA, TRUTH, repetitions, seed)


class StudyTests(unittest.TestCase):
    def test_quantile_and_basic(self):
        self.assertEqual(study.quantile([0, 10, 20, 30], .25), 7.5)
        self.assertEqual(study.basic_ci(10, [0, 10, 20]), [.5, 19.5])
        self.assertIsNone(study.basic_ci(None, [None]))

    def test_known_u_and_unmatched_hamming(self):
        matrix = study.distance_matrix([A, B, C], sorted(A))
        self.assertEqual(matrix, [[0, 1, 3], [1, 0, 3], [3, 3, 0]])
        self.assertEqual(study.weighted_u(matrix, [1, 1, 1]), 7 / 3)
        self.assertEqual(study.weighted_u(matrix, [3, 0, 0]), 0)

    def test_weighted_matches_explicit_duplicate_index_sample(self):
        matrix = study.distance_matrix([A, B, C], sorted(A))
        for indices in itertools.product(range(3), repeat=3):
            counts = [indices.count(i) for i in range(3)]
            explicit = sum(matrix[i][j] for i, j in itertools.combinations(indices, 2)) / 3
            self.assertEqual(study.weighted_u(matrix, counts), explicit, indices)
        # [0,0,1] has only two nonzero positional edges, denominator stays 3.
        self.assertEqual(study.weighted_u(matrix, [2, 1, 0]), 2 / 3)

    def test_same_binary_zero_effect_and_ci(self):
        result = analyze(make_raw([A, B, C]))
        self.assertEqual(result["primary"]["difference_changed_b"], 0)
        self.assertEqual(result["primary"]["basic_95_ci_changed_b"], [0, 0])
        self.assertEqual(result["primary"]["decision"], "inconclusive")
        self.assertEqual(result["median_paired_wall_ratio_experimental_over_baseline"], 2)

    def test_directional_positive_effect(self):
        result = analyze(make_raw([A, B, C], [A, A, A]))
        primary = result["primary"]
        self.assertGreater(primary["difference_changed_b"], 0)
        self.assertGreater(primary["basic_95_ci_changed_b"][0], 0)
        self.assertEqual(primary["relative_reduction"], 1)
        self.assertEqual(primary["difference_percentage_points"], 100 * (7 / 3) / 3)
        self.assertIn("positive-CI", primary["decision"])

    def test_zero_baseline_relative_undefined(self):
        for experimental in ([A, A, A], [A, B, C]):
            result = analyze(make_raw([A, A, A], experimental))
            self.assertIsNone(result["primary"]["relative_reduction"])
            self.assertIsNone(result["primary"]["relative_basic_95_ci"])
            self.assertEqual(result["bootstrap"]["replicates"]["relative_reduction"], [None] * 100)

    def test_deterministic_and_paired_whole_run_resampling(self):
        raw = make_raw([A, B, C], [B, C, A])
        first = analyze(raw)
        self.assertEqual(first, analyze(raw))
        self.assertNotEqual(first["bootstrap"], analyze(raw, seed=18)["bootstrap"])
        reps = first["bootstrap"]["replicates"]
        baseline = study.distance_matrix([A, B, C], sorted(A))
        experimental = study.distance_matrix([B, C, A], sorted(A))
        quality_delta = [-1, 0, 1]  # paired experimental-minus-baseline TP
        for r, counts in enumerate(study.bootstrap_counts(3, 100, 17)):
            self.assertEqual(sum(counts), 3)
            self.assertEqual(reps["baseline_changed_b"][r], study.weighted_u(baseline, counts))
            self.assertEqual(reps["experimental_changed_b"][r], study.weighted_u(experimental, counts))
            self.assertEqual(reps["quality_experimental_minus_baseline"]["tp"][r],
                             sum(c * q for c, q in zip(counts, quality_delta)) / 3)

    def test_quality_cohorts_not_all_nonoracle_proven_errors(self):
        value = study.quality({"p": "wrong", "u": "auto", "x": "ambiguous"}, METADATA, TRUTH)
        self.assertEqual(value["tp"], 0)
        self.assertEqual(value["known_positives_assigned_wrong_a"], 1)
        self.assertEqual(value["known_unmatched_b_automatched"], 1)
        self.assertEqual(value["ambiguous_b_automatched"], 1)
        self.assertEqual(value["strict_precision"], 0)
        value = study.quality(C, METADATA, TRUTH)
        self.assertEqual(value["known_positives_assigned_wrong_a"], 0)
        self.assertEqual(value["ambiguous_b_automatched"], 1)
        self.assertEqual(value["matches"], 2)

    def test_invalid_universes_and_indices(self):
        for alteration in (lambda raw: raw["runs"][0]["assignment"].pop("u"),
                           lambda raw: raw["runs"].pop(),
                           lambda raw: raw["runs"][0].update(shuffle_index=4),
                           lambda raw: raw["runs"].append(copy.deepcopy(raw["runs"][0])),
                           lambda raw: raw.update(status="failed")):
            raw = make_raw([A, B, C])
            alteration(raw)
            with self.assertRaises(ValueError):
                analyze(raw)
        with self.assertRaises(ValueError):
            analyze(make_raw([A]))

    def test_input_hash_validation_and_fixed(self):
        raw = make_raw([A, B, C])
        for run in raw["runs"]:
            run["permutation"]["csv_sha256"] = "same"
        with self.assertRaisesRegex(ValueError, "not distinct"):
            analyze(raw)
        raw = make_raw([A, B, C], mode="fixed")
        self.assertEqual(analyze(raw)["mode"], "fixed")
        raw["runs"][0]["permutation"]["csv_sha256"] = "different"
        with self.assertRaises(ValueError):
            analyze(raw)
        raw = make_raw([A, B, C], mode="fixed")
        for run in raw["runs"]:
            run["permutation"].pop("input_path")
        with self.assertRaisesRegex(ValueError, "missing"):
            analyze(raw)

    def test_cross_mode_identity_validation(self):
        a, b = make_raw([A, B]), make_raw([A, B], mode="fixed")
        a["provenance"]["template_sha256"] = "a"
        b["provenance"]["template_sha256"] = "b"
        with self.assertRaisesRegex(ValueError, "template"):
            study.cross_validate(a, b)

    def test_cli_schema_and_mode_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shuffled, fixed, output = root / "shuffled", root / "fixed", root / "output"
            for directory, count, mode in ((shuffled, 100, "shuffle"), (fixed, 50, "fixed")):
                directory.mkdir()
                raw = make_raw([A, B] * (count // 2), mode=mode)
                (directory / "measurements.json").write_text(json.dumps(raw), encoding="utf-8")
            with mock.patch.object(study, "load_metadata", return_value=(METADATA, TRUTH, {})), mock.patch("builtins.print"):
                study.main(["--shuffled", str(shuffled), "--fixed", str(fixed),
                            "--output", str(output), "--bootstrap", "5", "--seed", "20261118"])
            result = json.loads((output / "analysis.json").read_text())
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(result["modes"]["shuffle"]["bootstrap"]["seed"], 20261118)
            self.assertEqual(result["modes"]["fixed"]["bootstrap"]["seed"], 20261119)
            self.assertEqual(result["modes"]["shuffle"]["within_binary_pair_count"], 4950)
            self.assertEqual(result["modes"]["fixed"]["within_binary_pair_count"], 1225)
            self.assertIn("not determinism", (output / "summary.md").read_text())
            self.assertEqual(len(result["modes"]["shuffle"]["bootstrap"]["replicates"]["difference_changed_b"]), 5)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "existing output"):
                study.main(["--shuffled", "absent", "--fixed", "absent", "--output", folder])


if __name__ == "__main__":
    unittest.main()
