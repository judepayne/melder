#!/usr/bin/env python3
"""Paired assignment stability benchmark with shuffled or fixed input (stdlib only).

Preparation is untimed. CSV order is not guaranteed processing order: Rust
HashMap iteration and scheduling can randomize it. Shuffle mode varies B order;
fixed mode repeats identical B bytes/path. Run from any cwd; meld runs from the
project root so its existing model cache is reused. No existing benchmark is modified.
"""

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CAVEAT = ("Shuffled-input stability, not isolated same-input determinism. CSV order "
          "is not guaranteed processing order: Rust HashMap iteration and scheduling randomize it.")
FIXED_CAVEAT = ("Identical-input repeat stability: B bytes and path are fixed. CSV order is not "
                "guaranteed processing order: Rust HashMap iteration and scheduling randomize it.")
QUALITY_CAVEAT = ("Quality uses the supplied strict-ID positive oracle. It excludes the intentionally "
                  "ambiguous cohort; a non-oracle pair there is not necessarily a proven identity error.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(required) <= set(reader.fieldnames),
                f"Missing CSV headers {required}: {path}")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"Duplicate headers: {path}")
        rows = list(reader)
        require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
                f"Malformed CSV rows: {path}")
        return reader.fieldnames, rows


def record_ids(rows, field):
    ids = [row[field] for row in rows]
    require(all(ids) and len(ids) == len(set(ids)), f"Empty or duplicate {field}")
    return set(ids)


def copy_unchanged(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as stream, source.open("rb") as original:
            shutil.copyfileobj(original, stream)
    require(sha256(source) == sha256(destination), f"Different existing data copy: {destination}")
    return destination


def shuffled_rows(rows, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    require(Counter(tuple(sorted(row.items())) for row in shuffled) ==
            Counter(tuple(sorted(row.items())) for row in rows), "Shuffle changed records")
    return shuffled


def write_shuffle(path, fields, rows, seed):
    shuffled = shuffled_rows(rows, seed)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shuffled)
    actual_fields, actual = read_csv(path, fields)
    require(actual_fields == fields and actual == shuffled, "Shuffle CSV round-trip changed records")
    return {"seed": seed, "csv_sha256": sha256(path),
            "permutation_sha256": json_hash([row["counterparty_id"] for row in shuffled])}


def pair_input(mode, original, directory, fields, rows, seed, index):
    if mode == "fixed":
        # Preserve the original bytes AND path across every run; no shuffle/seed claim.
        return original, {"csv_sha256": sha256(original), "input_path": str(original)}
    require(mode == "shuffle", f"Unknown input order: {mode}")
    directory.mkdir(exist_ok=True)
    path = directory / f"b_{index:02d}.csv"
    return path, dict(write_shuffle(path, fields, rows, seed), input_path=str(path))


def reuse_seed_cache(source, preparation_directory):
    source = source.resolve(strict=True)
    manifest = file_manifest(source)
    require(bool(manifest), "Provided seed cache is empty")
    preparation_directory.mkdir()
    atomic_json(preparation_directory / "seed_cache_manifest.json", manifest)
    return source, manifest, {
        "directory": str(preparation_directory), "excluded_from_timing_summaries": True,
        "seed_cache_manifest": manifest, "seed_cache_source": "provided",
        "seed_cache_path": str(source), "reused_without_cold_rebuild": True}


def load_assignment(path, a_ids, b_ids):
    _, rows = read_csv(path, ("entity_id", "counterparty_id"))
    assignment = dict.fromkeys(sorted(b_ids))
    seen_a = set()
    for row in rows:
        a, b = row["entity_id"], row["counterparty_id"]
        require(a in a_ids and b in b_ids, f"Invalid crossmap IDs: {a}, {b}")
        require(a not in seen_a and assignment[b] is None, f"Crossmap not bijective: {a}, {b}")
        seen_a.add(a)
        assignment[b] = a
    return assignment


def assignment_hash(assignment):
    # Explicit None entries make matched/unmatched transitions visible.
    return json_hash(sorted(assignment.items()))


def truth_metrics(assignment, truth):
    predicted = {(a, b) for b, a in assignment.items() if a is not None}
    expected = {(a, b) for b, a in truth.items() if a is not None}
    tp = len(predicted & expected)
    return {"tp": tp, "fp": len(predicted) - tp, "fn": len(expected) - tp,
            "matches": len(predicted),
            "precision": tp / len(predicted) if predicted else None,
            "recall": tp / len(expected) if expected else None}


def compare_assignments(left, right):
    require(left.keys() == right.keys(), "Assignment B universes differ")
    a = {(v, k) for k, v in left.items() if v is not None}
    b = {(v, k) for k, v in right.items() if v is not None}
    return {"changed_b": sum(left[k] != right[k] for k in left),
            "jaccard": len(a & b) / len(a | b) if a | b else 1.0}


def distribution(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None


def stability(assignments):
    require(bool(assignments), "No assignments to summarize")
    keys = assignments[0].keys()
    require(all(item.keys() == keys for item in assignments), "Assignment B universes differ")
    # Build matched-pair sets once, not once per pair of runs.
    matched = [{(a, b) for b, a in item.items() if a is not None} for item in assignments]
    pairs = []
    for i, j in itertools.combinations(range(len(assignments)), 2):
        intersection = len(matched[i] & matched[j])
        union = len(matched[i]) + len(matched[j]) - intersection
        pairs.append(dict(left=i + 1, right=j + 1,
                          changed_b=sum(assignments[i][b] != assignments[j][b] for b in keys),
                          jaccard=intersection / union if union else 1.0))
    variety = {b: len({item[b] for item in assignments}) for b in sorted(keys)}
    stable = {b: assignments[0][b] for b in sorted(keys) if variety[b] == 1}
    return {"distinct_outcomes": len({assignment_hash(a) for a in assignments}),
            "per_b_variety": variety, "variety_histogram": dict(Counter(variety.values())),
            "shared_stable_mappings": {b: a for b, a in stable.items() if a is not None},
            "shared_stable_unmatched": [b for b, a in stable.items() if a is None],
            "pairwise": pairs,
            "changed_b": distribution([p["changed_b"] for p in pairs]),
            "jaccard": distribution([p["jaccard"] for p in pairs])}


def file_manifest(directory):
    require(directory.is_dir(), f"Missing cache: {directory}")
    paths = sorted(directory.rglob("*"))
    require(not any(p.is_symlink() for p in paths), f"Symlink in cache: {directory}")
    return {str(p.relative_to(directory)): sha256(p) for p in paths if p.is_file()}


def render_config(template, a, b, cache, crossmap, output):
    for token, path in {"__DATA_A__": a, "__DATA_B__": b, "__CACHE__": cache,
                        "__CROSSMAP__": crossmap, "__OUTPUT__": output}.items():
        require(token in template, f"Missing template placeholder: {token}")
        template = template.replace(token, json.dumps(str(path.resolve())))
    require(not re.search(r"__[A-Z_]+__", template), "Unresolved template placeholder")
    return template


def parse_resources(text):
    result = {}
    match = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
    require(match is not None, "Cannot parse /usr/bin/time -l CPU report")
    result.update(zip(("time_real_seconds", "user_seconds", "sys_seconds"), map(float, match.groups())))
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    require(rss is not None, "Cannot parse /usr/bin/time -l RSS report")
    result["maximum_rss_bytes"] = int(rss.group(1))  # Darwin reports bytes, not KiB.
    return result


def warm_evidence(log):
    forbidden = ("building combined embedding index", "cold rebuild more efficient",
                 "combined index cache invalidated", "combined index cache load failed",
                 "updating embedding cache", "combined index updated")
    require(not any(message in log for message in forbidden), "Warm run encoded/rebuilt its cache")
    hits = [line for line in log.splitlines() if "loaded combined index from cache" in line]
    require(all(any(re.search(rf'\bside[=:]\s*"?{side}\b', line) for line in hits)
                for side in ("A", "B")), "Missing explicit A/B warm cache hit evidence")
    return {"cache_hits": hits, "no_encoding_or_rebuild_observed": True}


def analyze_run(directory, a_ids, b_ids, truth, log):
    assignment = load_assignment(directory / "crossmap.csv", a_ids, b_ids)
    _, relationships = read_csv(directory / "output/relationships.csv",
                                ("entity_id", "counterparty_id", "relationship_type"))
    _, unmatched = read_csv(directory / "output/unmatched.csv", ("counterparty_id",))
    active = {"match": [], "review": []}
    for row in relationships:
        a, b = row["entity_id"], row["counterparty_id"]
        require(a in a_ids and b in b_ids, "Invalid relationship IDs")
        kind = row["relationship_type"]
        require(kind in ("match", "review", "broken"), f"Unknown relationship_type: {kind}")
        if kind in active:
            active[kind].append((a, b))
    matches = {(a, b) for b, a in assignment.items() if a is not None}
    require(len(active["match"]) == len(matches) and set(active["match"]) == matches,
            "Active match relationships disagree with crossmap")
    reviewed = {b for _, b in active["review"]}
    matched = {b for _, b in matches}
    no_match = record_ids(unmatched, "counterparty_id") if unmatched else set()
    require(not (matched & reviewed) and no_match == b_ids - matched - reviewed,
            "Unmatched/review/matched B coverage inconsistent")
    totals = {}
    for key, label in {"total_b": "Total B records", "skipped": r"Skipped \(crossmap\)",
                       "matches": "Auto-matched", "reviews": "Review", "unmatched": "No match"}.items():
        found = re.findall(rf"^\s*{label}:\s*(\d+)\s*$", log, re.MULTILINE)
        require(len(found) == 1, f"Missing/ambiguous batch total: {key}")
        totals[key] = int(found[0])
    require(totals == {"total_b": len(b_ids), "skipped": 0, "matches": len(matches),
                       "reviews": len(reviewed), "unmatched": len(no_match)},
            f"Batch totals disagree with active outputs: {totals}")
    wal_paths = sorted((directory / "output").glob("*.ndjson"))
    require(bool(wal_paths), "Missing NDJSON match WAL")
    moves = 0
    for path in wal_paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                moves += event.get("type") in ("crossmap_reassign", "CrossMapReassign")
    return {"assignment": assignment, "assignment_sha256": assignment_hash(assignment),
            "truth": truth_metrics(assignment, truth), "batch_totals": totals,
            "local_moves": moves, "wal_sha256": {str(p.relative_to(directory)): sha256(p) for p in wal_paths}}


def run_binary(binary, binary_hash, directory, template, a, b, env, seed_cache=None):
    require(sha256(binary) == binary_hash, f"Binary changed before run: {binary}")
    directory.mkdir()
    cache = directory / "cache"
    if seed_cache is None:
        cache.mkdir()
        before = {}
    else:
        source, expected = seed_cache
        require(file_manifest(source) == expected, "Seed cache changed")
        shutil.copytree(source, cache)
        before = file_manifest(cache)
        require(before == expected, "Copied cache differs from seed")
        atomic_json(directory / "cache_before.json", before)
    output = directory / "output"
    output.mkdir()
    config = directory / "config.yaml"
    config.write_text(render_config(template, a, b, cache, directory / "crossmap.csv", output), encoding="utf-8")
    command = ["/usr/bin/time", "-l", "-o", str(directory / "resources.log"),
               str(binary), "run", "--config", str(config), "--verbose"]
    atomic_json(directory / "invocation.json", {"command": command, "cwd": str(ROOT),
                                               "overrides": {"RAYON_NUM_THREADS": env["RAYON_NUM_THREADS"],
                                                             "NO_COLOR": "1"}})
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        start = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
        wall = time.perf_counter() - start
    binary_after = sha256(binary)
    atomic_json(directory / "execution.json", {"returncode": process.returncode, "wall_seconds": wall,
                                               "binary_sha256_before": binary_hash,
                                               "binary_sha256_after": binary_after})
    require(binary_after == binary_hash, f"Binary changed during run: {binary}")
    require(process.returncode == 0, f"meld failed ({process.returncode}); see {directory}")
    log = "\n".join((directory / name).read_text(encoding="utf-8", errors="replace")
                    for name in ("stdout.log", "stderr.log"))
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    result = {"directory": str(directory), "wall_seconds": wall,
              "resources": parse_resources((directory / "resources.log").read_text()),
              "config_sha256": sha256(config), "binary_sha256_after": binary_after}
    if seed_cache is not None:
        result["warm_verification"] = warm_evidence(log)
        result["seed_cache_manifest_sha256"] = json_hash(before)
    atomic_json(directory / "cache_after.json", file_manifest(cache))
    return result, log


def summarize(results):
    runs = results["runs"]
    summary = {"by_binary": {}, "paired": [],
               "all_runs_stability": stability([r["assignment"] for r in runs]) if runs else None}
    for label in ("baseline", "experimental"):
        selected = [r for r in runs if r["binary"] == label]
        if selected:
            summary["by_binary"][label] = {
                "completed_runs": len(selected),
                "wall_seconds": distribution([r["wall_seconds"] for r in selected]),
                "stability": stability([r["assignment"] for r in selected])}
    for index in sorted({r["shuffle_index"] for r in runs}):
        pair = {r["binary"]: r for r in runs if r["shuffle_index"] == index}
        if len(pair) != 2:
            continue
        a, b = pair["baseline"], pair["experimental"]
        delta = {key: (b["truth"][key] - a["truth"][key]
                       if a["truth"][key] is not None and b["truth"][key] is not None else None)
                 for key in ("tp", "fp", "precision", "recall", "matches")}
        summary["paired"].append({"shuffle_index": index,
                                  **compare_assignments(a["assignment"], b["assignment"]),
                                  "experimental_minus_baseline": dict(delta, local_moves=b["local_moves"] - a["local_moves"]),
                                  "experimental_over_baseline_wall": b["wall_seconds"] / a["wall_seconds"]})
    summary["paired_wall_ratio"] = distribution([p["experimental_over_baseline_wall"] for p in summary["paired"]])
    return summary


def persist_run(run):
    """Write a completed full outcome exactly once and return its lightweight reference."""
    path = Path(run["directory"]) / "run_result.json"
    require(not path.exists(), f"Run result already exists: {path}")
    atomic_json(path, run)
    return {"path": str(path), "sha256": sha256(path),
            "binary": run["binary"], "shuffle_index": run["shuffle_index"],
            "assignment_sha256": run["assignment_sha256"]}


def save_checkpoint(output, results, references):
    """Checkpoint references only: no assignments or quadratic stability work."""
    checkpoint = {key: value for key, value in results.items() if key not in ("runs", "summary")}
    checkpoint["run_results"] = references
    checkpoint["completed_runs"] = len(references)
    atomic_json(output / "checkpoint.json", checkpoint)


def save_results(output, results):
    results["summary"] = summarize(results)
    atomic_json(output / "measurements.json", results)
    lines = ["# Assignment stability", "", f"Status: **{results['status']}**", "", results.get("caveat", CAVEAT), "",
             "All completed outcomes are included; preparation is excluded from timing summaries.", "",
             QUALITY_CAVEAT, "",
             "|Shuffle|Binary|Wall s|Matches|TP|FP|Precision|Recall|Moves|Assignment SHA256|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for run in results["runs"]:
        t = run["truth"]
        lines.append(f"|{run['shuffle_index']}|{run['binary']}|{run['wall_seconds']:.3f}|{t['matches']}|{t['tp']}|{t['fp']}|{t['precision']}|{t['recall']}|{run['local_moves']}|{run['assignment_sha256']}|")
    for label, value in results["summary"]["by_binary"].items():
        s = value["stability"]
        lines += ["", f"## {label}", f"Distinct assignment outcomes: {s['distinct_outcomes']}",
                  f"Wall seconds min/median/max: {value['wall_seconds']}",
                  f"Pairwise changed-B min/median/max: {s['changed_b']}",
                  f"Pairwise Jaccard min/median/max: {s['jaccard']}",
                  f"Shared stable mappings: {len(s['shared_stable_mappings'])}; stable unmatched: {len(s['shared_stable_unmatched'])}",
                  f"Per-B assignment variety histogram (including unmatched): {s['variety_histogram']}"]
    combined = results["summary"]["all_runs_stability"]
    if combined:
        lines += ["", "## Across all completed runs (both binaries)",
                  f"Distinct outcomes: {combined['distinct_outcomes']}; shared stable mappings: {len(combined['shared_stable_mappings'])}; stable unmatched: {len(combined['shared_stable_unmatched'])}",
                  f"Changed-B min/median/max: {combined['changed_b']}; Jaccard: {combined['jaccard']}"]
    lines += ["", "## Paired experimental vs baseline", "", "```json",
              json.dumps(results["summary"]["paired"], indent=2), "```", "",
              f"Wall ratio min/median/max: {results['summary']['paired_wall_ratio']}", "",
              "Full assignments, per-B variety, pairwise comparisons, truth metrics, resources and provenance: measurements.json."]
    if "error" in results:
        lines += ["", f"Failure: {results['error']}"]
    temporary = output / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--experimental-binary", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--input-order", choices=("shuffle", "fixed"), default="shuffle")
    parser.add_argument("--seed-cache", type=Path,
                        help="Reuse this prepared seed cache unchanged; skip cold preparation")
    parser.add_argument("--rayon-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.runs > 0 and args.rayon_threads > 0, "Runs and Rayon threads must be positive")
    require(sys.platform == "darwin", "This harness requires macOS /usr/bin/time -l (RSS bytes)")
    binaries = {"baseline": args.baseline_binary.resolve(strict=True),
                "experimental": args.experimental_binary.resolve(strict=True)}
    require(all(os.access(p, os.X_OK) for p in binaries.values()), "Binary is not executable")
    template_path = HERE / "config.template.yaml"
    template = template_path.read_text(encoding="utf-8")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never reuse/overwrite an existing output directory.
    references = []
    results = {"status": "running", "input_order": args.input_order,
               "caveat": FIXED_CAVEAT if args.input_order == "fixed" else CAVEAT,
               "quality_caveat": QUALITY_CAVEAT, "runs": [],
               "provenance": {"project_root": str(ROOT), "seed": args.seed, "requested_runs": args.runs,
                              "input_order": args.input_order,
                              "template_sha256": sha256(template_path),
                              "environment_overrides": {"RAYON_NUM_THREADS": str(args.rayon_threads), "NO_COLOR": "1"},
                              "binaries": {k: {"path": str(p), "sha256": sha256(p)} for k, p in binaries.items()}}}
    # Inherit environment for normal model-cache/runtime behavior; never serialize it.
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.rayon_threads), NO_COLOR="1")
    try:
        save_checkpoint(output, results, references)
        data = {}
        for name in ("dataset_a_10k.csv", "dataset_b_10k.csv", "ground_truth_crossmap_10k.csv"):
            source = ROOT / "benchmarks/data" / name
            copied = copy_unchanged(source, HERE / "data" / name)
            data[name] = {"source": str(source), "copy": str(copied), "sha256": sha256(copied)}
        results["provenance"]["datasets"] = data
        a = Path(data["dataset_a_10k.csv"]["copy"])
        b = Path(data["dataset_b_10k.csv"]["copy"])
        _, a_rows = read_csv(a, ("entity_id",))
        fields, b_rows = read_csv(b, ("counterparty_id",))
        a_ids, b_ids = record_ids(a_rows, "entity_id"), record_ids(b_rows, "counterparty_id")
        truth = load_assignment(Path(data["ground_truth_crossmap_10k.csv"]["copy"]), a_ids, b_ids)
        if args.seed_cache is not None:
            seed_cache, seed_manifest, preparation = reuse_seed_cache(
                args.seed_cache, output / "preparation")
        else:
            preparation, _ = run_binary(binaries["baseline"], results["provenance"]["binaries"]["baseline"]["sha256"],
                                        output / "preparation", template, a, b, env)
            seed_cache = output / "preparation/cache"
            seed_manifest = file_manifest(seed_cache)
            require(bool(seed_manifest), "Preparation produced an empty cache")
            atomic_json(output / "preparation/seed_cache_manifest.json", seed_manifest)
            preparation.update(excluded_from_timing_summaries=True, seed_cache_manifest=seed_manifest,
                               seed_cache_source="cold_preparation", seed_cache_path=str(seed_cache),
                               reused_without_cold_rebuild=False)
        results["preparation"] = preparation
        save_checkpoint(output, results, references)
        for index in range(args.runs):
            shuffled, permutation = pair_input(args.input_order, b, output / "shuffles",
                                                fields, b_rows, args.seed + index, index + 1)
            order = ("baseline", "experimental") if index % 2 == 0 else ("experimental", "baseline")
            for position, label in enumerate(order):
                directory = output / f"shuffle_{index + 1:02d}_{label}"
                results["active_run"] = {"directory": str(directory), "binary": label,
                                         "shuffle_index": index + 1, "permutation": permutation}
                save_checkpoint(output, results, references)
                require(sha256(shuffled) == permutation["csv_sha256"], "Shuffle changed before run")
                run, log = run_binary(binaries[label], results["provenance"]["binaries"][label]["sha256"],
                                      directory, template, a, shuffled, env, (seed_cache, seed_manifest))
                require(sha256(shuffled) == permutation["csv_sha256"], "Shuffle changed during run")
                run.update(analyze_run(directory, a_ids, b_ids, truth, log))
                run.update(binary=label, shuffle_index=index + 1, pair_position=position + 1,
                           permutation=permutation, input_order=args.input_order,
                           input_path=str(shuffled))
                references.append(persist_run(run))
                results["runs"].append(run)
                results.pop("active_run")
                save_checkpoint(output, results, references)
        require(file_manifest(seed_cache) == seed_manifest, "Seed cache changed after runs")
        require(all(sha256(Path(item["copy"])) == item["sha256"] and
                    sha256(Path(item["source"])) == item["sha256"] for item in data.values()),
                "Dataset changed during benchmark")
        results["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        after = {k: sha256(p) if p.is_file() else None for k, p in binaries.items()}
        results["provenance"]["binary_hashes_after"] = after
        if any(after[k] != results["provenance"]["binaries"][k]["sha256"] for k in binaries):
            results["status"] = "failed"
            results["error"] = "Binary hash changed"
        save_checkpoint(output, results, references)
        # Complete schema and summaries are materialized once, only on success.
        # On failure checkpoint.json still references every completed full result.
        if results["status"] == "complete":
            save_results(output, results)
    require(results["status"] == "complete", results.get("error", "Benchmark incomplete"))
    print(output / "summary.md")


if __name__ == "__main__":
    main()
