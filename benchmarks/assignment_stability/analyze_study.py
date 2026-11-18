#!/usr/bin/env python3
"""Predeclared paired whole-run analysis; standard library only, inputs read-only."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
LABELS = ("baseline", "experimental")
CAVEATS = [
    "Inference is conditional on this fixed corpus, cache, binaries and thread settings, assuming independent fresh processes across paired runs.",
    "Shuffled CSV order is not necessarily processing order (HashMap iteration and scheduling can vary). Fixed mode repeats identical B input.",
    "Whole paired runs, not individual B records or distance pairs, are resampled. Shared indices preserve input and temporal pairing covariance.",
    "Each mode answers a separate question with its own 95% interval; modes are not pooled and no familywise claim is made.",
    "A positive difference CI supports stability improvement on this corpus, not determinism. An interval containing zero is inconclusive.",
    "The strict-ID oracle excludes the 2000 ambiguous B records. Non-oracle pairs, especially ambiguous assignments, are not all proven errors.",
    "Quality intervals are secondary descriptive uncertainty summaries, not multiplicity-adjusted confirmatory claims. Performance and ever-variable B are descriptive only; no runs are excluded.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, probability):
    """Linear interpolation between sorted order statistics (index (m-1)*p)."""
    require(bool(values) and 0 <= probability <= 1, "Invalid quantile")
    values = sorted(values)
    index = (len(values) - 1) * probability
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def basic_ci(point, replicates):
    """Return a two-sided 95% basic bootstrap interval, or undefined."""
    if point is None or any(value is None for value in replicates):
        return None
    return [2 * point - quantile(replicates, .975),
            2 * point - quantile(replicates, .025)]


def distance_matrix(assignments, keys):
    """Align vectors once; Hamming includes transitions to and from unmatched None."""
    require(all(set(a) == set(keys) for a in assignments), "Assignment B universes differ")
    vectors = [tuple(a[k] for k in keys) for a in assignments]
    matrix = [[0] * len(vectors) for _ in vectors]
    for i, left in enumerate(vectors):
        for j in range(i):
            matrix[i][j] = matrix[j][i] = sum(a != b for a, b in zip(left, vectors[j]))
    return matrix


def weighted_u(matrix, counts):
    """Mean changed-record count over resampled positions; duplicate indices have zero distance."""
    n = sum(counts)
    require(n >= 2 and len(counts) == len(matrix), "Invalid bootstrap counts")
    return sum(counts[i] * counts[j] * matrix[i][j]
               for i in range(len(counts)) for j in range(i)) / (n * (n - 1) / 2)


def bootstrap_counts(n, repetitions, seed):
    """Draw n paired run indices per replicate, retaining original-index multiplicities."""
    rng = random.Random(seed)
    for _ in range(repetitions):
        counts = [0] * n
        for _ in range(n):
            counts[rng.randrange(n)] += 1
        yield counts


def validate_runs(raw, mode, expected_n=None):
    """Validate complete paired runs and per-mode input identities, without exclusions."""
    require(raw.get("status") == "complete", "Raw study is not complete")
    require(raw.get("provenance", {}).get("input_order") == mode, "Wrong provenance input_order")
    by_label = {label: {} for label in LABELS}
    for run in raw["runs"]:
        label, index = run["binary"], run["shuffle_index"]
        require(label in LABELS and isinstance(index, int) and index >= 1, "Invalid binary/index")
        require(index not in by_label[label], "Duplicate binary/index")
        by_label[label][index] = run
    indices = sorted(by_label["baseline"])
    require(len(indices) >= 2 and indices == sorted(by_label["experimental"]), "Unpaired indices")
    require(indices == list(range(1, len(indices) + 1)), "Incomplete paired indices")
    requested = raw["provenance"].get("requested_runs", expected_n or len(indices))
    require(len(indices) == requested and (expected_n is None or len(indices) == expected_n), "Incomplete run count")
    ordered = {label: [by_label[label][i] for i in indices] for label in LABELS}
    keys = sorted(ordered["baseline"][0]["assignment"])
    require(bool(keys), "Empty B universe")
    hashes, permutations, paths = [], [], []
    for left, right in zip(*[ordered[label] for label in LABELS]):
        for run in (left, right):
            require(set(run["assignment"]) == set(keys), "Assignment B universes differ")
            require(all(a is None or isinstance(a, str) for a in run["assignment"].values()), "Invalid assignment")
            values = [a for a in run["assignment"].values() if a is not None]
            require(len(values) == len(set(values)), "Assignment is not bijective")
            require(run.get("input_order", mode) == mode, "Run input order differs")
        lp, rp = left["permutation"], right["permutation"]
        require(lp == rp, "Paired input identities differ")
        hashes.append(lp["csv_sha256"])
        permutations.append(lp.get("permutation_sha256"))
        paths.append(lp.get("input_path", left.get("input_path")))
    if mode == "shuffle":
        require(len(set(hashes)) == len(indices) and None not in permutations and
                len(set(permutations)) == len(indices), "Shuffled inputs are not distinct")
    else:
        require(len(set(hashes)) == 1, "Fixed B hashes differ")
        require(None not in paths and len(set(paths)) == 1, "Fixed B paths differ or are missing")
    for label in LABELS:
        hashes_after = {r.get("binary_sha256_after") for r in ordered[label]} - {None}
        require(len(hashes_after) <= 1, "Binary changed within mode")
        expected = raw["provenance"].get("binaries", {}).get(label, {}).get("sha256")
        if expected is not None and hashes_after:
            require(hashes_after == {expected}, "Run binary differs from provenance")
    cache_hashes = {r.get("seed_cache_manifest_sha256") for r in raw["runs"]} - {None}
    require(len(cache_hashes) <= 1, "Seed cache differs within mode")
    return ordered, keys


def cross_validate(shuffled, fixed):
    """Compare available binary/input/template/cache/thread provenance across modes."""
    a, b = shuffled["provenance"], fixed["provenance"]
    for key in ("template_sha256", "environment_overrides"):
        if key in a and key in b:
            require(a[key] == b[key], f"Cross-mode {key} differs")
    for label in LABELS:
        av = a.get("binaries", {}).get(label, {}).get("sha256")
        bv = b.get("binaries", {}).get(label, {}).get("sha256")
        if av is not None and bv is not None:
            require(av == bv, f"Cross-mode {label} binary differs")
        ah = {r["binary_sha256_after"] for r in shuffled["runs"] if r["binary"] == label and "binary_sha256_after" in r}
        bh = {r["binary_sha256_after"] for r in fixed["runs"] if r["binary"] == label and "binary_sha256_after" in r}
        if ah and bh:
            require(ah == bh, "Cross-mode run binary differs")
    for name in set(a.get("datasets", {})) & set(b.get("datasets", {})):
        require(a["datasets"][name]["sha256"] == b["datasets"][name]["sha256"], "Cross-mode input differs")
    for source in ("seed_cache_manifest",):
        av = shuffled.get("preparation", {}).get(source)
        bv = fixed.get("preparation", {}).get(source)
        if av is not None and bv is not None:
            require(av == bv, "Cross-mode seed cache differs")
    ah = {r["seed_cache_manifest_sha256"] for r in shuffled["runs"] if "seed_cache_manifest_sha256" in r}
    bh = {r["seed_cache_manifest_sha256"] for r in fixed["runs"] if "seed_cache_manifest_sha256" in r}
    if ah and bh:
        require(ah == bh, "Cross-mode seed cache hash differs")


def load_metadata(keys):
    """Read the original 10k B metadata and strict positive oracle, never modify them."""
    names = ("dataset_b_10k.csv", "ground_truth_crossmap_10k.csv")
    for folder in (HERE / "data", HERE.parent / "data"):
        if all((folder / name).is_file() for name in names):
            break
    else:
        raise ValueError("Original metadata/oracle missing")
    with (folder / names[0]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata = {row["counterparty_id"]: row for row in rows}
    require(len(rows) == len(metadata) == 10000 and set(metadata) == set(keys), "Original 10k metadata universe differs")
    require(Counter(r["_match_type"] for r in rows)["ambiguous"] == 2000, "Expected 2000 ambiguous records")
    with (folder / names[1]).open(newline="", encoding="utf-8") as stream:
        truth_rows = list(csv.DictReader(stream))
    truth = {row["counterparty_id"]: row["entity_id"] for row in truth_rows}
    require(len(truth) == len(truth_rows) and set(truth) <= set(keys), "Invalid positive oracle")
    require(all(metadata[b]["_match_type"] == "matched" and metadata[b]["_true_a_id"] == a
                for b, a in truth.items()), "Oracle/metadata mismatch")
    require(set(truth) == {b for b, r in metadata.items() if r["_match_type"] == "matched"}, "Incomplete positive oracle")
    provenance = {name: {"path": str(folder / name), "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()} for name in names}
    return metadata, truth, provenance


def quality(assignment, metadata, truth):
    """Separate known identity errors from ambiguous automatic assignments."""
    tp = sum(assignment[b] == a for b, a in truth.items())
    matches = sum(a is not None for a in assignment.values())
    return {"tp": tp,
            "known_positives_assigned_wrong_a": sum(assignment[b] is not None and assignment[b] != a for b, a in truth.items()),
            "known_unmatched_b_automatched": sum(assignment[b] is not None and r["_match_type"] == "unmatched" for b, r in metadata.items()),
            "ambiguous_b_automatched": sum(assignment[b] is not None and r["_match_type"] == "ambiguous" for b, r in metadata.items()),
            "strict_precision": tp / matches if matches else None,
            "strict_recall": tp / len(truth) if truth else None, "matches": matches}


def mean_median(values):
    if not values or any(v is None for v in values):
        return {"mean": None, "median": None}
    return {"mean": statistics.mean(values), "median": statistics.median(values)}


def analyze_mode(raw, mode, metadata, truth, repetitions, seed, expected_n=None):
    """Analyze one mode with a shared paired whole-run bootstrap for every endpoint."""
    require(repetitions > 0, "Bootstrap repetitions must be positive")
    ordered, keys = validate_runs(raw, mode, expected_n)
    require(set(keys) == set(metadata), "Metadata B universe differs")
    n, size = len(ordered["baseline"]), len(keys)
    matrices = {label: distance_matrix([r["assignment"] for r in ordered[label]], keys) for label in LABELS}
    point = {label: weighted_u(matrices[label], [1] * n) for label in LABELS}
    delta = point["baseline"] - point["experimental"]
    relative = delta / point["baseline"] if point["baseline"] > 0 else None
    quality_rows = {label: [quality(r["assignment"], metadata, truth) for r in ordered[label]] for label in LABELS}
    for label in LABELS:
        for run, computed in zip(ordered[label], quality_rows[label]):
            recorded = run.get("truth", {})
            for name, key in (("tp", "tp"), ("matches", "matches"),
                              ("precision", "strict_precision"), ("recall", "strict_recall")):
                if name in recorded:
                    require(recorded[name] == computed[key], "Recorded truth metrics differ from original oracle")
    quality_deltas = {key: [(b[key] - a[key]) if a[key] is not None and b[key] is not None else None
                            for a, b in zip(quality_rows["baseline"], quality_rows["experimental"])]
                      for key in quality_rows["baseline"][0]}
    replicates = {"baseline_changed_b": [], "experimental_changed_b": [], "difference_changed_b": [], "relative_reduction": [],
                  "quality_experimental_minus_baseline": {key: [] for key in quality_deltas}}
    # Sparse nonzero entries are reused; matrices are computed exactly once.
    edges = [(i, j, matrices["baseline"][i][j], matrices["experimental"][i][j])
             for i in range(n) for j in range(i) if matrices["baseline"][i][j] or matrices["experimental"][i][j]]
    denominator = n * (n - 1) / 2
    for counts in bootstrap_counts(n, repetitions, seed):
        bs = es = 0
        for i, j, bd, ed in edges:
            weight = counts[i] * counts[j]
            bs += weight * bd
            es += weight * ed
        bs, es = bs / denominator, es / denominator
        replicates["baseline_changed_b"].append(bs)
        replicates["experimental_changed_b"].append(es)
        replicates["difference_changed_b"].append(bs - es)
        replicates["relative_reduction"].append((bs - es) / bs if bs > 0 and relative is not None else None)
        for key, values in quality_deltas.items():
            value = sum(c * v for c, v in zip(counts, values)) / n if all(v is not None for v in values) else None
            replicates["quality_experimental_minus_baseline"][key].append(value)
    ci = basic_ci(delta, replicates["difference_changed_b"])
    result = {"mode": mode, "n_paired_runs": n, "within_binary_pair_count": int(denominator), "b_count": size,
              "primary": {"baseline_mean_changed_b": point["baseline"], "experimental_mean_changed_b": point["experimental"],
                          "baseline_fraction": point["baseline"] / size, "experimental_fraction": point["experimental"] / size,
                          "difference_changed_b": delta, "difference_fraction": delta / size,
                          "difference_percentage_points": 100 * delta / size, "basic_95_ci_changed_b": ci,
                          "basic_95_ci_fraction": [v / size for v in ci], "basic_95_ci_percentage_points": [100 * v / size for v in ci],
                          "relative_reduction": relative, "relative_basic_95_ci": basic_ci(relative, replicates["relative_reduction"]),
                          "relative_undefined_bootstrap_replicates": sum(v is None for v in replicates["relative_reduction"]),
                          "decision": "positive-CI evidence on this corpus" if ci[0] > 0 else ("negative-CI evidence on this corpus" if ci[1] < 0 else "inconclusive")},
              "bootstrap": {"seed": seed, "repetitions": repetitions, "rng": "random.Random(seed).randrange(n), n draws per replicate",
                            "replicates": replicates}, "quality_paired_effects": {}, "by_binary": {}}
    for key, values in quality_deltas.items():
        observed = mean_median(values)["mean"]
        result["quality_paired_effects"][key] = {"experimental_minus_baseline_mean": observed,
            "basic_95_ci": basic_ci(observed, replicates["quality_experimental_minus_baseline"][key])}
    for label in LABELS:
        runs = ordered[label]
        assignments = [r["assignment"] for r in runs]
        variety = {b: len({a[b] for a in assignments}) for b in keys}
        stable = {b: assignments[0][b] for b in keys if variety[b] == 1}
        distances = [matrices[label][i][j] for i in range(n) for j in range(i)]
        result["by_binary"][label] = {
            "quality": {key: mean_median([row[key] for row in quality_rows[label]]) for key in quality_rows[label][0]},
            "distinct_whole_assignment_outcomes": len({tuple(a[b] for b in keys) for a in assignments}),
            "ever_variable_b": sum(v > 1 for v in variety.values()),
            "stable_pairs": {b: a for b, a in stable.items() if a is not None},
            "stable_unmatched_b": [b for b, a in stable.items() if a is None],
            "pairwise_changed_b": mean_median(distances),
            "distance_upper_triangle_row_order": [matrices[label][i][j] for i in range(n) for j in range(i + 1, n)],
            "cohort_variation": {cohort: {"b_count": sum(r["_match_type"] == cohort for r in metadata.values()),
                "ever_variable_b": sum(variety[b] > 1 and r["_match_type"] == cohort for b, r in metadata.items())}
                for cohort in sorted({r["_match_type"] for r in metadata.values()})},
            "wall_seconds_median": statistics.median(r["wall_seconds"] for r in runs),
            "maximum_rss_bytes_median": statistics.median(r["resources"]["maximum_rss_bytes"] for r in runs),
            "local_moves": mean_median([r["local_moves"] for r in runs])}
    result["median_paired_wall_ratio_experimental_over_baseline"] = statistics.median(
        b["wall_seconds"] / a["wall_seconds"] for a, b in zip(ordered["baseline"], ordered["experimental"]))
    return result


def markdown(analysis):
    lines = ["# Assignment stability study", "", "## Predeclared method", "", analysis["method"]["formula"], "",
             analysis["method"]["bootstrap"], "", "## Primary results", "",
             "|Mode|Paired runs|Baseline changed B|Experimental changed B|Difference (pp)|95% basic CI (pp)|Decision|",
             "|---|---:|---:|---:|---:|---|---|"]
    for mode, result in analysis["modes"].items():
        p = result["primary"]
        lines.append(f"|{mode}|{result['n_paired_runs']}|{p['baseline_mean_changed_b']:.4f}|{p['experimental_mean_changed_b']:.4f}|{p['difference_percentage_points']:.6f}|{p['basic_95_ci_percentage_points']}|{p['decision']}|")
    for mode, result in analysis["modes"].items():
        lines += ["", f"### {mode}: secondary quality and descriptive performance", "", "```json",
                  json.dumps({"quality_paired_effects": result["quality_paired_effects"],
                              "by_binary": {label: {k: v for k, v in result["by_binary"][label].items()
                                                    if k not in ("stable_pairs", "stable_unmatched_b", "distance_upper_triangle_row_order")}
                                            for label in LABELS},
                              "median_paired_wall_ratio": result["median_paired_wall_ratio_experimental_over_baseline"]}, indent=2), "```", ""]
    lines += ["## Caveats", ""] + ["- " + caveat for caveat in CAVEATS]
    lines += ["", "Full normalization, seed, replicate arrays, quality, stable pairs and distance matrices: `analysis.json`."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Refusing existing output directory")
    require(args.bootstrap > 0, "Bootstrap repetitions must be positive")
    raw = {mode: json.loads((folder / "measurements.json").read_text(encoding="utf-8"))
           for mode, folder in (("shuffle", args.shuffled), ("fixed", args.fixed))}
    _, keys = validate_runs(raw["shuffle"], "shuffle", 100)
    validate_runs(raw["fixed"], "fixed", 50)
    cross_validate(raw["shuffle"], raw["fixed"])
    metadata, truth, provenance = load_metadata(keys)
    for study in raw.values():
        for name, entry in provenance.items():
            recorded = study["provenance"].get("datasets", {}).get(name, {}).get("sha256")
            require(recorded is None or recorded == entry["sha256"], "Metadata hash differs from raw provenance")
    analysis = {"schema_version": 1, "metadata": provenance,
                "raw_inputs": {mode: {"path": str(folder / "measurements.json"),
                    "sha256": hashlib.sha256((folder / "measurements.json").read_bytes()).hexdigest()}
                    for mode, folder in (("shuffle", args.shuffled), ("fixed", args.fixed))},
                "method": {"formula": "U_b = sum(i<j) Hamming(A_bi,A_bj) / choose(n,2) / Bcount; primary = U_baseline - U_experimental (positive is improvement).",
                           "bootstrap": "Resample n paired whole-run indices with replacement. With counts wi, sum(i<j) wi*wj*dij / choose(n,2) / Bcount; duplicate-index diagonals are zero. Never divide by the sum of off-diagonal weights. Basic 95% CI = [2*observed - q.975, 2*observed - q.025], linear-interpolation quantiles. Same method for relative reduction difference/baseline when baseline > 0; if any bootstrap baseline is zero its relative interval is undefined (no dropped replicates). Paired quality means use sum(wi*(experimental_i-baseline_i))/n with the identical counts.",
                           "units": "Matrices and bootstrap changed_b arrays are record counts; divide by Bcount for fractions, multiply fractions by 100 for percentage points. Relative reductions are fractions.",
                           "normalization": "Fixed choose(n,2) and fixed B universe, including unmatched records.",
                           "predeclared": True}, "caveats": CAVEATS,
                "modes": {mode: analyze_mode(raw[mode], mode, metadata, truth, args.bootstrap, args.seed + offset, expected)
                          for mode, offset, expected in (("shuffle", 0, 100), ("fixed", 1, 50))}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(analysis, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(analysis), encoding="utf-8")
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
