#!/usr/bin/env python3
"""Competition oracles, independent WAL audit and descriptive analysis (stdlib only)."""

import argparse
from collections import Counter
import importlib.util
import itertools
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("assignment_stability_shared", HERE.parent / "assignment_stability/run_test.py")
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
require = shared.require
FAMILIES = ("rank2", "rank3", "rank5", "ambiguous", "unrepairable", "background")
CAVEAT = ("Descriptive first pass, not a significance test. CSV permutations do not control Rust "
          "HashMap iteration or scheduling; local repair is not a global determinism guarantee. "
          "Expected rank-r repairability requires top_n >= r, but sampled order may already give "
          "five matches or leave four. The original background contains 2,000 ambiguous records with no verified identity oracle; "
          "their strict-ID nonagreement is not counted as an identity error.")


def validate_calibration(fixture, scoredrows):
    """Fail closed on the actual combined-dataset top-five score matrix.

    Accepts decoded real scoring-log rows (headers may be included). Ambiguous
    ties are reported explicitly, not assumed: all 25 edges must qualify.
    """
    require(fixture.get("schema_version") == 1, "Unsupported fixture schema")
    groups = fixture["groups"]
    expected = {b for g in groups for b in g["b_ids"]}
    by_b = {}
    for row in scoredrows:
        if row.get("type") != "scored" or row.get("query_id") not in expected:
            continue
        b = row["query_id"]
        require(b not in by_b, f"Duplicate calibration query: {b}")
        require(row.get("query_side") == "b", f"Calibration must score B: {b}")
        require(row.get("reason") not in ("canonical", "exact", "crossmap"), f"Calibration bypass: {b}")
        by_b[b] = row
    require(set(by_b) == expected, "Missing custom calibration queries")
    tie_audit = {}
    matrix = {}
    for g in groups:
        aid_set = set(g["a_ids"])
        flex = g["flexible_b_id"]
        group_scores = []
        row_spreads = []
        rank_orders = []
        for b in g["b_ids"]:
            candidates = by_b[b]["candidates"]
            require(len(candidates) == 5 and {c["matched_id"] for c in candidates} == aid_set,
                    f"Expected five distinct in-group candidates: {b}")
            require([c["rank"] for c in candidates] == [1, 2, 3, 4, 5], f"Invalid ranks: {b}")
            require(all(isinstance(c.get("field_scores"), list) and c["field_scores"] for c in candidates),
                    f"Missing real field scores: {b}")
            scores = [c["score"] for c in candidates]
            row_spreads.append(max(scores) - min(scores))
            rank_orders.append(tuple(c["matched_id"] for c in candidates))
            require(all(isinstance(s, (int, float)) and 0 <= s <= 1.000001 for s in scores), f"Invalid scores: {b}")
            require(all(x >= y for x, y in zip(scores, scores[1:])), f"Scores not ranked: {b}")
            qualified = {c["matched_id"] for c in candidates if c["score"] >= .85}
            if g["kind"] == "ambiguous":
                require(qualified == aid_set, f"Ambiguous edge below auto threshold: {b}")
            elif b != flex:
                require(qualified == {g["truth"][b]}, f"Constrained qualifications not only truth: {b}")
            elif g["kind"] == "unrepairable":
                own = set(g["truth"].values())
                require(len(own) == 4 and qualified == own, f"Unrepairable flex must qualify only four occupied As: {b}")
            else:
                require(qualified == aid_set, f"Rank-family flex must qualify all five: {b}")
                escape = next(c for c in candidates if c["matched_id"] == g["escape_a_id"])
                require(escape["rank"] == g["escape_rank"] == int(g["kind"][-1]), f"Wrong escape rank: {b}")
            matrix[b] = candidates
            group_scores.extend(scores)
        if g["kind"] == "ambiguous":
            tie_audit[g["id"]] = {
                "all_candidate_rows_tied": max(row_spreads) <= 1e-9,
                "uniform_rank_order": len(set(rank_orders)) == 1,
                "maximum_within_row_score_spread": max(row_spreads),
                "across_queries_score_spread": max(group_scores) - min(group_scores),
            }
    return {"validated_queries": len(by_b), "matrix": matrix, "ambiguous_ties": tie_audit,
            "all_ambiguous_tied": all(v["all_candidate_rows_tied"] for v in tie_audit.values()),
            "ambiguous_tie_policy": "All 25 edges qualify. Ties are within each query; between-query numeric drift and candidate ordering are audited separately."}


def cohort_oracle(fixture, b_rows, truth):
    """Normalize explicit positive, allowed-ambiguous and negative identities."""
    groups = {b: g for g in fixture["groups"] for b in g["b_ids"]}
    oracle = {}
    for row in b_rows:
        b = row["counterparty_id"]
        if b in groups:
            g = groups[b]
            state = "ambiguous" if g["kind"] == "ambiguous" else ("negative" if b in g["known_unmatched_b"] else "positive")
            oracle[b] = {"family": g["kind"], "group": g["id"], "state": state,
                         "truth": g["truth"].get(b), "allowed": g["allowed"][b]}
        else:
            tag = row["_match_type"]
            require(tag in ("matched", "ambiguous", "unmatched"), f"Unknown background oracle class: {tag}")
            state = {"matched": "positive", "ambiguous": "ambiguous", "unmatched": "negative"}[tag]
            require(state != "positive" or truth.get(b) == row["_true_a_id"], f"Background truth conflict: {b}")
            oracle[b] = {"family": "background", "group": "background", "state": state,
                         "truth": truth.get(b) if state == "positive" else None,
                         "allowed": None}  # Background ambiguous alternatives are not verified.
    return oracle


def quality(assignment, oracle, family):
    ids = [b for b, o in oracle.items() if o["family"] == family]
    result = dict.fromkeys(("records", "matches", "known_correct", "incorrect", "allowed_ambiguous",
                           "unverified_ambiguous", "ambiguous_records", "known_negative", "matched_known_negative", "known_positive"), 0)
    result["records"] = len(ids)
    for b in ids:
        a, o = assignment[b], oracle[b]
        result["known_negative"] += o["state"] == "negative"
        result["known_positive"] += o["state"] == "positive"
        result["ambiguous_records"] += o["state"] == "ambiguous"
        if a is None:
            continue
        result["matches"] += 1
        if o["state"] == "positive":
            result["known_correct"] += a == o["truth"]
            result["incorrect"] += a != o["truth"]
        elif o["state"] == "negative":
            result["matched_known_negative"] += 1
            result["incorrect"] += 1
        elif o["allowed"] is None:
            result["unverified_ambiguous"] += 1
        else:
            result["allowed_ambiguous"] += a in o["allowed"]
            result["incorrect"] += a not in o["allowed"]
    return result


def normalized_stability(assignments):
    """Mean changed B includes explicit None; normalized by cohort B count."""
    require(bool(assignments), "No assignments")
    keys = set(assignments[0])
    require(all(set(a) == keys for a in assignments), "Different B universes")
    changes = [sum(a[b] != c[b] for b in keys) for a, c in itertools.combinations(assignments, 2)]
    mean = statistics.mean(changes) if changes else None
    return {"distinct_cohort_mapping_outcomes": len({shared.assignment_hash(a) for a in assignments}),
            "pairwise_comparisons": len(changes), "mean_pairwise_changed_b": mean,
            "mean_pairwise_changed_b_fraction": mean / len(keys) if mean is not None and keys else None,
            "changed_b": shared.distribution(changes)}


def fold_wal(events, a_ids, b_ids, fixture):
    """Independently replay atomic claims/replacements, enforcing protection and isolation."""
    a_group = {a: g["id"] for g in fixture["groups"] for a in g["a_ids"]}
    b_group = {b: g["id"] for g in fixture["groups"] for b in g["b_ids"]}
    kinds = {g["id"]: g["kind"] for g in fixture["groups"]}
    assignment, reverse, protected = dict.fromkeys(b_ids), {}, set()
    moves, ranks = Counter(), Counter()
    counts = Counter()

    def check_edge(a, b):
        require(a in a_ids and b in b_ids, f"WAL unknown IDs: {a}/{b}")
        require(a_group.get(a, "background") == b_group.get(b, "background"), f"Out-of-group WAL edge: {a}/{b}")

    def claim(a, b):
        check_edge(a, b)
        require(assignment[b] in (None, a) and reverse.get(a) in (None, b), "WAL violates bijection")
        assignment[b], reverse[a] = a, b

    for e in events:
        kind = e["type"]
        counts[kind] += 1
        if kind == "crossmap_confirm":
            claim(e["a_id"], e["b_id"])
            if e.get("reason") in ("canonical", "exact", "crossmap") or e.get("score") is None:
                protected.add((e["a_id"], e["b_id"]))
        elif kind == "crossmap_break":
            a, b = e["a_id"], e["b_id"]
            check_edge(a, b)
            require(assignment[b] == a, "Breaking absent WAL pair")
            assignment[b] = None
            del reverse[a]
            protected.discard((a, b))
        elif kind == "crossmap_reassign":
            a, b = e["old_a_id"], e["old_b_id"]
            check_edge(a, b)
            require(assignment[b] == a and (a, b) not in protected, "Absent or protected reassignment incumbent")
            replacements = e["replacements"]
            require(len(replacements) == 2 and len({p["a_id"] for p in replacements}) == 2 and
                    len({p["b_id"] for p in replacements}) == 2, "Reassignment must be one-to-two")
            require(a in {p["a_id"] for p in replacements} and b in {p["b_id"] for p in replacements} and
                    all((p["a_id"], p["b_id"]) != (a, b) for p in replacements), "Reassignment must displace incumbent")
            group = b_group.get(b, "background")
            require(kinds.get(group) not in ("ambiguous", "unrepairable"), "Repair in protected custom control")
            require(all(b_group.get(p["b_id"], "background") == group for p in replacements), "Cross-group repair")
            assignment[b] = None
            del reverse[a]
            for p in replacements:
                require(p.get("score", 0) >= .85, "Subthreshold replacement")
                require(isinstance(p.get("rank"), int) and p["rank"] >= 1, "Missing replacement rank")
                claim(p["a_id"], p["b_id"])
                ranks[(group, p["rank"])] += 1
            moves[group] += 1
        elif kind in ("review_match", "no_match_below", "upsert_record"):
            if kind == "review_match":
                require(e["side"] == "b", "Unexpected non-B review")
                check_edge(e["candidate_id"], e["id"])
            elif kind == "no_match_below":
                require(e["query_side"] == "b" and e["query_id"] in b_ids, "Invalid no-match query")
                if e.get("best_candidate_id") is not None:
                    check_edge(e["best_candidate_id"], e["query_id"])
        else:
            raise ValueError(f"Unexpected batch WAL event: {kind}")
    return {"assignment": assignment, "moves_by_group": dict(moves),
            "replacement_ranks_by_group": {g: dict(Counter({str(r): n for (group, r), n in ranks.items() if group == g})) for g in moves},
            "rank_gt3_count": sum(n for (_, rank), n in ranks.items() if rank > 3),
            "protected_pairs": len(protected), "event_counts": dict(counts), "local_moves": sum(moves.values())}


def read_wal(directory):
    paths = sorted((Path(directory) / "output").glob("*.match_log*.ndjson"))
    require(len(paths) == 1, "Expected exactly one fresh batch match WAL")
    with paths[0].open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(results):
    oracle, fixture, runs = results["oracle"], results["fixture"], results["runs"]
    summary = {"by_top_n_binary": {}, "paired_binary": [], "paired_width": [], "factorial_interactions": []}
    for n in (1, 2, 3, 5):
        for label in ("baseline", "experimental"):
            selected = [r for r in runs if r["top_n"] == n and r["binary"] == label]
            if not selected:
                continue
            cohorts = {}
            for family in FAMILIES:
                ids = [b for b, o in oracle.items() if o["family"] == family]
                qs = [quality(r["assignment"], oracle, family) for r in selected]
                gs = [g for g in fixture["groups"] if g["kind"] == family]
                attained = [sum(sum(r["assignment"][b] is not None for b in g["b_ids"]) == g["max_matches"] for g in gs) for r in selected]
                cohorts[family] = {"quality": {k: shared.distribution([q[k] for q in qs]) for k in qs[0]},
                                   "quality_mean": {k: statistics.mean(q[k] for q in qs) for k in qs[0]},
                                   "stability": normalized_stability([{b: r["assignment"][b] for b in ids} for r in selected]),
                                   "groups_attaining_max": shared.distribution(attained) if gs else None,
                                   "groups_attaining_max_by_run": attained if gs else None,
                                   "expected_repair_width_available": n >= int(family[-1]) if family.startswith("rank") else None}
            group_kind = {g["id"]: g["kind"] for g in fixture["groups"]}
            moves_by_family = {f: sum(v for r in selected for g, v in r["wal_audit"]["moves_by_group"].items()
                                      if group_kind.get(g, "background") == f) for f in FAMILIES}
            rank_by_family = {f: Counter() for f in FAMILIES}
            for r in selected:
                for g, ranks in r["wal_audit"]["replacement_ranks_by_group"].items():
                    rank_by_family[group_kind.get(g, "background")].update(ranks)
            summary["by_top_n_binary"][f"{n}/{label}"] = {
                "runs": len(selected), "cohorts": cohorts,
                "wall_seconds": shared.distribution([r["wall_seconds"] for r in selected]),
                "maximum_rss_bytes": shared.distribution([r["resources"]["maximum_rss_bytes"] for r in selected]),
                "local_moves_by_family": moves_by_family, "replacement_ranks_by_family": {f: dict(c) for f, c in rank_by_family.items()},
                "replacement_rank_gt3_by_family": {f: sum(count for rank, count in c.items() if int(rank) > 3) for f, c in rank_by_family.items()},
                "replacement_rank_gt3_count": sum(r["wal_audit"]["rank_gt3_count"] for r in selected)}
    lookup = {(r["shuffle_index"], r["top_n"], r["binary"]): r for r in runs}

    def comparison(left, right):
        deltas = {}
        for f in FAMILIES:
            a, b = quality(left["assignment"], oracle, f), quality(right["assignment"], oracle, f)
            deltas[f] = {k: b[k] - a[k] for k in ("matches", "known_correct", "incorrect", "allowed_ambiguous", "matched_known_negative")}
        return {"cohort_deltas": deltas, "changed_b": shared.compare_assignments(left["assignment"], right["assignment"])["changed_b"],
                "wall_ratio": right["wall_seconds"] / left["wall_seconds"],
                "rss_ratio": right["resources"]["maximum_rss_bytes"] / left["resources"]["maximum_rss_bytes"]}

    for index in sorted({r["shuffle_index"] for r in runs}):
        for n in (1, 2, 3, 5):
            if all((index, n, label) in lookup for label in ("baseline", "experimental")):
                summary["paired_binary"].append({"shuffle_index": index, "top_n": n,
                    **comparison(lookup[index, n, "baseline"], lookup[index, n, "experimental"])})
            if n == 1:
                continue
            for label in ("baseline", "experimental"):
                if (index, 1, label) in lookup and (index, n, label) in lookup:
                    summary["paired_width"].append({"shuffle_index": index, "binary": label, "top_n": n,
                        **comparison(lookup[index, 1, label], lookup[index, n, label])})
            keys = [(index, width, label) for width in (1, n) for label in ("baseline", "experimental")]
            if all(k in lookup for k in keys):
                effects = {}
                for f in FAMILIES:
                    q = {(width, label): quality(lookup[index, width, label]["assignment"], oracle, f) for _, width, label in keys}
                    effects[f] = {metric: (q[n, "experimental"][metric] - q[n, "baseline"][metric]) -
                                 (q[1, "experimental"][metric] - q[1, "baseline"][metric])
                                  for metric in ("matches", "known_correct", "incorrect")}
                summary["factorial_interactions"].append({"shuffle_index": index, "top_n": n,
                    "repair_by_width_difference_in_differences": effects})
    def paired_effects(rows):
        return {
            "pairs": len(rows),
            "pairs_with_mapping_change": sum(r["changed_b"] > 0 for r in rows),
            "pairs_with_quality_or_coverage_change": sum(any(v for c in r["cohort_deltas"].values() for v in c.values()) for r in rows),
            "cohort_mean_deltas": {f: {metric: statistics.mean(r["cohort_deltas"][f][metric] for r in rows)
                                        for metric in ("matches", "known_correct", "incorrect", "allowed_ambiguous", "matched_known_negative")}
                                   for f in FAMILIES} if rows else {},
            "wall_ratio": shared.distribution([r["wall_ratio"] for r in rows]),
            "rss_ratio": shared.distribution([r["rss_ratio"] for r in rows])}

    summary["observed_effects"] = {key: paired_effects(summary[key]) for key in ("paired_binary", "paired_width")}
    summary["paired_binary_by_top_n"] = {str(n): paired_effects([r for r in summary["paired_binary"] if r["top_n"] == n]) for n in (1, 2, 3, 5)}
    summary["paired_width_by_binary_top_n"] = {f"{label}/{n}": paired_effects([r for r in summary["paired_width"] if r["binary"] == label and r["top_n"] == n])
                                             for label in ("baseline", "experimental") for n in (2, 3, 5)}
    return summary


def save_results(output, results):
    """Materialize the full final schema once; raw checkpoints contain references only."""
    results["summary"] = summarize(results)
    shared.atomic_json(output / "measurements.json", results)
    lines = ["# Assignment competition", "", CAVEAT, "", f"Status: **{results['status']}**; measured runs: {len(results['runs'])}",
             "Preparation and calibration excluded from timing. Ambiguous quality uses allowed mappings, not arbitrary strict IDs.",
             "", "|N|Binary|Family|Mean matches|Mean correct|Mean incorrect|Mean allowed ambiguous|Mean changed B|Distinct outcomes|Groups at max min/median/max|", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, item in results["summary"]["by_top_n_binary"].items():
        n, label = key.split("/")
        for family, cohort in item["cohorts"].items():
            q, s = cohort["quality_mean"], cohort["stability"]
            lines.append(f"|{n}|{label}|{family}|{q['matches']:.2f}|{q['known_correct']:.2f}|{q['incorrect']:.2f}|{q['allowed_ambiguous']:.2f}|{s['mean_pairwise_changed_b']}|{s['distinct_cohort_mapping_outcomes']}|{cohort['groups_attaining_max']}|")
    calibration = results.get("preparation", {}).get("calibration", {})
    if calibration:
        lines += ["", f"Combined baseline calibration validated {calibration['validated_custom_queries']} custom queries. "
                  f"All ambiguous candidate rows tied: **{calibration['all_ambiguous_tied']}**. "
                  "If false, ambiguous edges all qualify but some query rows are not tied; see the full calibration matrix."]
    lines += ["", "## Resources and local moves", "", "|N/Binary|Wall s min/median/max|RSS bytes min/median/max|Moves by family|Replacement ranks >3|", "|---|---|---|---|---:|"]
    for key, item in results["summary"]["by_top_n_binary"].items():
        lines.append(f"|{key}|{item['wall_seconds']}|{item['maximum_rss_bytes']}|{item['local_moves_by_family']}|{item['replacement_rank_gt3_count']}|")
    lines += ["", "## Observed paired effects (descriptive, not significance)", "", "```json",
              json.dumps(results["summary"]["observed_effects"], indent=2), "```", "",
              "measurements.json contains per-pair binary/width deltas and factorial interactions, full normalized cohort stability, quality, resource and rank audits.",
              "A mapping change alone is not proof of a beneficial repair; inspect known-correct/incorrect/coverage deltas and WAL moves. No global determinism claim."]
    path = output / "summary.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New analysis directory")
    args = parser.parse_args()
    results = json.loads(args.measurements.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    save_results(args.output, results)
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
