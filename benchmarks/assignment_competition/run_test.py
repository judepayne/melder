#!/usr/bin/env python3
"""Run the paired 20-permutation x four-width x two-binary competition benchmark.

Stdlib only; Darwin resource accounting. Preparation is a single new combined
baseline top-five cache build and scoring-log audit, excluded from measurements.
No binaries are built and no dataset/calibration files are generated here.
"""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("assignment_competition_analysis", HERE / "analyze_results.py")
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)
shared = analysis.shared
require = shared.require
validate_calibration = analysis.validate_calibration
ROOT = shared.ROOT
WIDTHS = (1, 2, 3, 5)
ORDER_POLICY = ("Zero-based permutation index i: rotate [1,2,3,5] left by i%4; "
                "baseline first for even i, experimental first for odd i, at every width. "
                "For 20 permutations every width occupies every slot five times; each binary is first ten times per width.")


def execution_order(index):
    """Balanced width rotation and alternating within-width binary order."""
    offset = index % len(WIDTHS)
    widths = WIDTHS[offset:] + WIDTHS[:offset]
    labels = ("baseline", "experimental") if index % 2 == 0 else ("experimental", "baseline")
    return [(n, label) for n in widths for label in labels]


def width_template(template, top_n, preparation=False):
    """Replace the numeric token before the shared path renderer is called."""
    require(top_n in WIDTHS, "Invalid top_n")
    require(template.count("__TOP_N__") == 1, "Expected one top_n placeholder")
    require(not re.search(r"^scoring_log\s*:", template, re.MULTILINE), "Template must not override scoring log policy")
    rendered = template.replace("__TOP_N__", str(top_n))
    return rendered + ("\nscoring_log:\n  enabled: true\n  compression: none\n" if preparation else
                       "\nscoring_log:\n  enabled: false\n")


def validate_template(template):
    """Audit the owned-by-parent static config without adding a YAML dependency."""
    required = ("ann_candidates: 100", "top_n: __TOP_N__", "vector_backend: usearch",
                "auto_match: 0.85", "review_floor: 0.60", "operator: and",
                "field_a: competition_group", "field_b: competition_group",
                "field_a: country_code", "field_b: domicile", "weight: 0.55", "weight: 0.20", "weight: 0.05")
    require(all(s in template for s in required), "Template changes benchmark parameters")
    require(re.search(r"^exact_prefilter:\s*\n\s+enabled: false\s*$", template, re.MULTILINE), "Exact prefilter must be disabled")
    require("common_id_field" not in template and "local_reassignment" not in template,
            "No common-ID bypass or per-case local-reassignment config allowed")


def load_inputs(data, fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    identity_path = data / "data_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    require(identity["schema_version"] == 1 and fixture["schema_version"] == 1, "Unsupported data/fixture schema")
    require(identity["fixture_sha256"] == shared.sha256(fixture_path), "Dataset fixture identity differs")
    paths = {name: data / name for name in ("dataset_a.csv", "dataset_b.csv", "ground_truth.csv", "data_identity.json")}
    hashes = {name: shared.sha256(path) for name, path in paths.items()}
    for name in ("dataset_a.csv", "dataset_b.csv", "ground_truth.csv"):
        require(identity["files"][name]["sha256"] == hashes[name], f"Dataset identity hash mismatch: {name}")
    _, a_rows = shared.read_csv(paths["dataset_a.csv"], ("entity_id", "competition_group", "country_code"))
    fields, b_rows = shared.read_csv(paths["dataset_b.csv"], ("counterparty_id", "competition_group", "domicile", "_match_type", "_true_a_id"))
    a_ids, b_ids = shared.record_ids(a_rows, "entity_id"), shared.record_ids(b_rows, "counterparty_id")
    require(len(a_ids) == len(b_ids) == 10500, "Expected 10500 combined rows on each side")
    require(Counter(g["kind"] for g in fixture["groups"]) == Counter(dict.fromkeys(analysis.FAMILIES[:-1], 20)), "Expected 20 groups per custom family")
    require(len(fixture["a_rows"]) == len(fixture["b_rows"]) == 500, "Expected 500 fixture rows per side")
    require(len({g["id"] for g in fixture["groups"]}) == 100, "Duplicate group IDs")
    for rows, custom, id_field, membership, country in ((a_rows, fixture["a_rows"], "entity_id", "a_ids", "country_code"),
                                                        (b_rows, fixture["b_rows"], "counterparty_id", "b_ids", "domicile")):
        indexed = {r[id_field]: r for r in rows}
        require(Counter(i for g in fixture["groups"] for i in g[membership]) == Counter(r[id_field] for r in custom), "Fixture membership differs")
        for row in custom:
            require(row[id_field] in indexed and all(indexed[row[id_field]].get(k) == v for k, v in row.items()), "Combined custom row differs from fixture")
        custom_ids = {r[id_field] for r in custom}
        require(all(r["competition_group"] == "background" for r in rows if r[id_field] not in custom_ids), "Background isolation differs")
        for g in fixture["groups"]:
            require(len(g[membership]) == 5 and all(indexed[i]["competition_group"] == g["id"] for i in g[membership]), "Invalid custom group")
            require(len({indexed[i][country] for i in g[membership]}) == 1, "Group countries differ")
    for g in fixture["groups"]:
        require({r["country_code"] for r in a_rows if r["entity_id"] in g["a_ids"]} ==
                {r["domicile"] for r in b_rows if r["counterparty_id"] in g["b_ids"]}, "A/B country blocks differ")
    truth = shared.load_assignment(paths["ground_truth.csv"], a_ids, b_ids)
    oracle = analysis.cohort_oracle(fixture, b_rows, truth)
    require(Counter(o["state"] for o in oracle.values()) == Counter(positive=7380, ambiguous=2100, negative=1020), "Unexpected known-positive/ambiguous/negative counts")
    require(sum(a is not None for a in truth.values()) == 7380, "Expected 7380 positive truth pairs")
    for b, o in oracle.items():
        require(truth[b] == o["truth"], f"Combined positive truth differs from fixture: {b}")
    provenance = {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()}
    provenance["fixture.json"] = {"path": str(fixture_path), "sha256": shared.sha256(fixture_path)}
    return fixture, fields, b_rows, a_ids, b_ids, truth, oracle, provenance


def verify_sources(provenance):
    for item in provenance.values():
        require(shared.sha256(Path(item["path"])) == item["sha256"], f"Source changed: {item['path']}")


def audit_preparation(directory, fixture, a_ids, b_ids):
    paths = list((directory / "output").glob("*.scoring_log.ndjson"))
    require(len(paths) == 1, "Expected one real uncompressed preparation scoring log")
    with paths[0].open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(rows and rows[0].get("type") == "header", "Missing scoring log header")
    header, scored = rows[0], rows[1:]
    require(header["top_n"] == 5 and header["thresholds"]["auto_match"] == .85 and
            header["thresholds"]["review_floor"] == .60 and header["thresholds"]["min_score_gap"] is None,
            "Preparation scoring header does not reflect benchmark parameters")
    require(len(scored) == len(b_ids) and {r.get("query_id") for r in scored} == b_ids,
            "Preparation must score every combined B record exactly once")
    require(all(r.get("type") == "scored" and r.get("query_side") == "b" and
                r.get("reason") not in ("canonical", "exact", "crossmap") for r in scored), "Preparation bypass detected")
    calibration = validate_calibration(fixture, scored)
    require(all(audit["uniform_rank_order"] for audit in calibration["ambiguous_ties"].values()),
            "Ambiguous controls need common ranked prefixes to assert no repair below full width")
    wal = analysis.fold_wal(analysis.read_wal(directory), a_ids, b_ids, fixture)
    require(wal["local_moves"] == 0 and wal["protected_pairs"] == 0, "Baseline preparation performed local repair or protected bypass")
    calibration.update(header=header, scoring_log_path=str(paths[0]), scoring_log_sha256=shared.sha256(paths[0]),
                       baseline_bypass_proof={"exact_prefilter_enabled": False, "common_id_field_configured": False,
                                              "all_b_scored": len(scored), "protected_wal_pairs": wal["protected_pairs"],
                                              "baseline_reassign_events": wal["local_moves"]})
    shared.atomic_json(directory / "calibration_matrix.json", calibration)
    return {"path": str(directory / "calibration_matrix.json"), "sha256": shared.sha256(directory / "calibration_matrix.json"),
            "validated_custom_queries": calibration["validated_queries"], "header": header,
            "all_ambiguous_tied": calibration["all_ambiguous_tied"], "baseline_bypass_proof": calibration["baseline_bypass_proof"]}


def run_benchmark(args):
    require(args.runs > 0 and args.rayon_threads > 0, "Runs and Rayon threads must be positive")
    require(sys.platform == "darwin", "Requires macOS /usr/bin/time -l (RSS bytes)")
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Store generated benchmark results outside the project tree")
    require(not output.exists(), f"Refusing to overwrite: {output}")
    binaries = {"baseline": args.baseline_binary.resolve(strict=True), "experimental": args.experimental_binary.resolve(strict=True)}
    require(all(os.access(p, os.X_OK) for p in binaries.values()), "Binary is not executable")
    template_path = HERE / "config.template.yaml"
    template = template_path.read_text(encoding="utf-8")
    validate_template(template)
    fixture, fields, b_rows, a_ids, b_ids, truth, oracle, inputs = load_inputs(args.data.resolve(), args.fixture.resolve())
    output.mkdir(parents=True, exist_ok=False)
    references = []
    results = {"schema_version": 1, "status": "running", "runs": [], "fixture": fixture, "oracle": oracle,
               "caveat": analysis.CAVEAT, "order_policy": ORDER_POLICY,
               "provenance": {"project_root": str(ROOT), "seed": args.seed, "requested_permutations": args.runs,
                              "requested_measured_runs": args.runs * 8, "widths": list(WIDTHS), "inputs": inputs,
                              "template_sha256": shared.sha256(template_path),
                              "environment_overrides": {"RAYON_NUM_THREADS": str(args.rayon_threads), "NO_COLOR": "1"},
                              "binaries": {label: {"path": str(path), "sha256": shared.sha256(path)} for label, path in binaries.items()}}}
    # Checkpoints remain small: persist the fixture/oracle once, reference it thereafter.
    shared.atomic_json(output / "inputs.json", {"fixture": fixture, "oracle": oracle})
    results["inputs_reference"] = {"path": str(output / "inputs.json"), "sha256": shared.sha256(output / "inputs.json")}
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.rayon_threads), NO_COLOR="1")

    def checkpoint():
        shared.save_checkpoint(output, {k: v for k, v in results.items() if k not in ("fixture", "oracle")}, references)

    try:
        checkpoint()
        results["active_preparation"] = {"binary": "baseline", "top_n": 5, "excluded_from_timing_summaries": True}
        checkpoint()
        preparation, _ = shared.run_binary(binaries["baseline"], results["provenance"]["binaries"]["baseline"]["sha256"],
                                            output / "preparation", width_template(template, 5, True),
                                            args.data.resolve() / "dataset_a.csv", args.data.resolve() / "dataset_b.csv", env)
        seed_cache = output / "preparation/cache"
        seed_manifest = shared.file_manifest(seed_cache)
        require(bool(seed_manifest), "Preparation produced empty combined cache")
        shared.atomic_json(output / "preparation/seed_cache_manifest.json", seed_manifest)
        calibration = audit_preparation(output / "preparation", fixture, a_ids, b_ids)
        preparation.update(excluded_from_timing_summaries=True, seed_cache_source="fresh_combined_baseline_top5",
                           seed_cache_manifest=seed_manifest, calibration=calibration)
        results["preparation"] = preparation
        results.pop("active_preparation")
        verify_sources(inputs)
        checkpoint()
        for index in range(args.runs):
            shuffled, permutation = shared.pair_input("shuffle", args.data.resolve() / "dataset_b.csv", output / "shuffles",
                                                      fields, b_rows, args.seed + index, index + 1)
            for position, (n, label) in enumerate(execution_order(index), 1):
                directory = output / f"shuffle_{index + 1:02d}_n{n}_{label}"
                results["active_run"] = {"directory": str(directory), "binary": label, "top_n": n,
                                         "shuffle_index": index + 1, "position": position, "permutation": permutation}
                checkpoint()
                verify_sources(inputs)
                require(shared.sha256(template_path) == results["provenance"]["template_sha256"], "Template changed")
                require(shared.sha256(shuffled) == permutation["csv_sha256"], "B permutation changed")
                run, log = shared.run_binary(binaries[label], results["provenance"]["binaries"][label]["sha256"], directory,
                                             width_template(template, n), args.data.resolve() / "dataset_a.csv", shuffled,
                                             env, (seed_cache, seed_manifest))
                require(shared.sha256(shuffled) == permutation["csv_sha256"], "B permutation changed during run")
                require(not list((directory / "output").glob("*.scoring_log*")), "Measured run wrote scoring log")
                run.update(shared.analyze_run(directory, a_ids, b_ids, truth, log))
                # Do not expose the old strict-ID FP oracle for ambiguous cohorts.
                run.pop("truth")
                audit = analysis.fold_wal(analysis.read_wal(directory), a_ids, b_ids, fixture)
                require(audit.pop("assignment") == run["assignment"], "Independent WAL fold differs from final CSV")
                require(audit["local_moves"] == run["local_moves"], "WAL move counts disagree")
                require(audit["protected_pairs"] == 0, "Measured run bypassed scoring with protected claims")
                require(label != "baseline" or audit["local_moves"] == 0, "Baseline unexpectedly repaired assignments")
                require(all(int(rank) <= n for ranks in audit["replacement_ranks_by_group"].values() for rank in ranks), "Replacement outside configured width")
                require(shared.file_manifest(seed_cache) == seed_manifest, "Source seed cache changed")
                verify_sources(inputs)
                run.update(binary=label, top_n=n, shuffle_index=index + 1, execution_position=position,
                           binary_sha256_before=results["provenance"]["binaries"][label]["sha256"],
                           permutation=permutation, input_path=str(shuffled), wal_audit=audit,
                           input_sha256={name: item["sha256"] for name, item in inputs.items()},
                           source_seed_cache_unchanged=True,
                           quality={f: analysis.quality(run["assignment"], oracle, f) for f in analysis.FAMILIES})
                reference = shared.persist_run(run)
                reference["top_n"] = n
                references.append(reference)
                results["runs"].append(run)
                results.pop("active_run")
                checkpoint()
        require(len(results["runs"]) == args.runs * 8, "Incomplete factorial experiment")
        results["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        after = {label: shared.sha256(p) if p.is_file() else None for label, p in binaries.items()}
        results["provenance"]["binary_hashes_after"] = after
        if any(after[label] != results["provenance"]["binaries"][label]["sha256"] for label in binaries):
            results["status"] = "failed"
            results["error"] = "Binary changed"
        checkpoint()
        if results["status"] == "complete":
            analysis.save_results(output, results)
    require(results["status"] == "complete", results.get("error", "Incomplete benchmark"))
    return output / "summary.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--experimental-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the project tree")
    parser.add_argument("--runs", type=int, default=20, help="Number of permutations; measured runs = 8 x this value")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--rayon-threads", type=int, default=8)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--fixture", type=Path, default=HERE / "fixture.json")
    args = parser.parse_args()
    print(run_benchmark(args))


if __name__ == "__main__":
    main()
