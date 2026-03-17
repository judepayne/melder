#!/usr/bin/env python3
"""Evaluate meld batch output against ground truth.

Compares results.csv (auto-matched) and review.csv against the ground truth
crossmap and dataset B metadata to compute precision, recall, and F1.

The theoretical ceiling is computed dynamically:
- Blocking ceiling: matched records reachable via blocking (correct country code)
- Combined ceiling: matched records reachable by ANY configured mechanism
  (blocking OR exact_prefilter). When exact_prefilter is configured, some
  previously-blocked records become reachable via exact field matching,
  raising the combined ceiling above the blocking ceiling.

Usage:
    python3 benchmarks/accuracy/eval.py \
        --results  benchmarks/accuracy/10kx10k_bm25/output/results.csv \
        --review   benchmarks/accuracy/10kx10k_bm25/output/review.csv \
        --unmatched benchmarks/accuracy/10kx10k_bm25/output/unmatched.csv \
        --ground-truth benchmarks/data/ground_truth_crossmap_10k.csv \
        --dataset-a benchmarks/data/dataset_a_10k.csv \
        --dataset-b benchmarks/data/dataset_b_10k.csv \
        --config benchmarks/accuracy/10kx10k_bm25/config.yaml  # optional
"""

import argparse
import csv
import sys


def load_pairs(
    path: str, a_col: str = "a_id", b_col: str = "b_id"
) -> set[tuple[str, str]]:
    """Load (a_id, b_id) pairs from a CSV file."""
    pairs = set()
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                pairs.add((row[a_col], row[b_col]))
    except FileNotFoundError:
        pass
    return pairs


def load_ground_truth(path: str) -> set[tuple[str, str]]:
    """Load ground truth pairs as (entity_id, counterparty_id)."""
    return load_pairs(path, "entity_id", "counterparty_id")


def load_b_metadata(path: str) -> dict[str, dict]:
    """Load dataset B with _match_type, _true_a_id and all fields for each record."""
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            meta[row["counterparty_id"]] = dict(row)
    return meta


def load_exact_prefilter_fields(config_path: str) -> list[tuple[str, str]]:
    """Read exact_prefilter field pairs from a meld config YAML.

    Returns list of (field_a, field_b) tuples, or [] if not configured.
    """
    try:
        import yaml
    except ImportError:
        return []
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ep = cfg.get("exact_prefilter", {})
        if not ep.get("enabled", False):
            return []
        return [(fp["field_a"], fp["field_b"]) for fp in ep.get("fields", [])]
    except Exception:
        return []


def compute_ceilings(
    dataset_a_path: str,
    b_meta: dict[str, dict],
    exact_fields: list[tuple[str, str]],
) -> tuple[int, int, int, int]:
    """Compute blocking and combined ceilings from actual data.

    Returns (total_true_matches, blocked_out, blocking_ceiling, combined_ceiling).

    blocking_ceiling: matched records reachable via blocking alone.
    combined_ceiling: matched records reachable by blocking OR exact_prefilter.
      When exact_fields is empty, combined_ceiling == blocking_ceiling.
    """
    a_records = {}
    with open(dataset_a_path) as f:
        for row in csv.DictReader(f):
            a_records[row["entity_id"]] = row

    total_matched = 0
    blocked_out = 0
    combined_unreachable = 0

    for meta in b_meta.values():
        if meta["_match_type"] != "matched":
            continue
        total_matched += 1
        true_a_id = meta["_true_a_id"]
        a_rec = a_records.get(true_a_id, {})

        is_blocked = a_rec.get("country_code") != meta.get("domicile")
        if is_blocked:
            blocked_out += 1

        # Check if exact prefilter can reach this blocked record
        if is_blocked and exact_fields:
            exact_reachable = all(
                a_rec.get(fa, "").strip().lower() == meta.get(fb, "").strip().lower()
                and a_rec.get(fa, "").strip() != ""
                and meta.get(fb, "").strip() != ""
                for fa, fb in exact_fields
            )
            if not exact_reachable:
                combined_unreachable += 1
        elif is_blocked:
            combined_unreachable += 1

    blocking_ceiling = total_matched - blocked_out
    combined_ceiling = total_matched - combined_unreachable
    return total_matched, blocked_out, blocking_ceiling, combined_ceiling


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results", required=True, help="Path to results.csv (auto-matched)"
    )
    parser.add_argument("--review", required=True, help="Path to review.csv")
    parser.add_argument("--unmatched", required=True, help="Path to unmatched.csv")
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth crossmap CSV"
    )
    parser.add_argument("--dataset-a", required=True, help="Path to dataset_a CSV")
    parser.add_argument(
        "--dataset-b", required=True, help="Path to dataset_b CSV (with _match_type)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to meld config YAML (optional, for exact_prefilter ceiling)",
    )
    args = parser.parse_args()

    # Load exact_prefilter fields from config if provided
    exact_fields = load_exact_prefilter_fields(args.config) if args.config else []

    # Load data
    gt = load_ground_truth(args.ground_truth)
    b_meta = load_b_metadata(args.dataset_b)
    auto_matched = load_pairs(args.results)
    review_pairs = load_pairs(args.review)

    # Compute ceilings dynamically
    total_true_matches, blocked_out, blocking_ceiling, combined_ceiling = (
        compute_ceilings(args.dataset_a, b_meta, exact_fields)
    )
    total_b = len(b_meta)

    # Count unmatched
    unmatched_count = 0
    try:
        with open(args.unmatched) as f:
            unmatched_count = sum(1 for _ in csv.DictReader(f))
    except FileNotFoundError:
        pass

    # --- Classification breakdown ---
    auto_tp = 0
    auto_fp = 0
    auto_fp_detail = {"matched_wrong": 0, "ambiguous": 0, "unmatched": 0}

    for a_id, b_id in auto_matched:
        if (a_id, b_id) in gt:
            auto_tp += 1
        else:
            auto_fp += 1
            mt = b_meta.get(b_id, {}).get("_match_type", "unknown")
            if mt == "matched":
                auto_fp_detail["matched_wrong"] += 1
            elif mt == "ambiguous":
                auto_fp_detail["ambiguous"] += 1
            elif mt == "unmatched":
                auto_fp_detail["unmatched"] += 1

    review_tp = 0
    review_fp = 0
    for a_id, b_id in review_pairs:
        if (a_id, b_id) in gt:
            review_tp += 1
        else:
            review_fp += 1

    # --- Auto-match metrics (vs blocking ceiling) ---
    precision = auto_tp / len(auto_matched) if auto_matched else 0.0
    recall_ceiling = auto_tp / blocking_ceiling if blocking_ceiling > 0 else 0.0
    recall_total = auto_tp / total_true_matches if total_true_matches > 0 else 0.0
    # --- Combined metrics (vs combined ceiling) ---
    combined_tp = auto_tp + review_tp
    combined_total = len(auto_matched) + len(review_pairs)
    combined_precision = combined_tp / combined_total if combined_total > 0 else 0.0
    combined_recall = combined_tp / combined_ceiling if combined_ceiling > 0 else 0.0

    # --- Output ---
    print()
    print("=" * 60)
    print("  ACCURACY EVALUATION")
    print("=" * 60)
    print()
    ceiling_note = f"{total_true_matches:,} matched - {blocked_out:,} blocked"
    if exact_fields and combined_ceiling != blocking_ceiling:
        recovered = combined_ceiling - blocking_ceiling
        ceiling_note += f" + {recovered:,} recovered by exact prefilter"
    print(f"  Blocking ceiling:         {blocking_ceiling:>6,}  ({ceiling_note})")
    if combined_ceiling != blocking_ceiling:
        print(
            f"  Combined ceiling:         {combined_ceiling:>6,}  (blocking + exact prefilter reachable)"
        )
    print(f"  Total true matches:       {total_true_matches:>6,}")
    print(f"  Total B records:          {total_b:>6,}")
    print()

    print("  --- Melder Output ---")
    print(f"  Auto-matched:             {len(auto_matched):>6,}")
    print(f"  Review:                   {len(review_pairs):>6,}")
    print(f"  Unmatched:                {unmatched_count:>6,}")
    print()

    print("  --- Auto-Match Accuracy ---")
    print(f"  True positives:           {auto_tp:>6,}")
    print(f"  False positives:          {auto_fp:>6,}")
    if auto_fp > 0:
        print(f"    - matched to wrong A:   {auto_fp_detail['matched_wrong']:>6,}")
        print(f"    - ambiguous records:     {auto_fp_detail['ambiguous']:>6,}")
        print(f"    - unmatched records:     {auto_fp_detail['unmatched']:>6,}")
    print()

    print("  --- Review Accuracy ---")
    print(f"  True positives in review: {review_tp:>6,}")
    print(f"  False positives in review:{review_fp:>6,}")
    print()

    print("  --- Metrics ---")
    print(f"  Precision (auto-match):   {precision:>9.4f}  ({precision * 100:.1f}%)")
    print(
        f"  Recall (vs ceiling):      {recall_ceiling:>9.4f}  ({recall_ceiling * 100:.1f}%)  [of {blocking_ceiling:,} reachable]"
    )
    print(
        f"  Recall (vs total):        {recall_total:>9.4f}  ({recall_total * 100:.1f}%)  [of {total_true_matches:,} true matches]"
    )
    print()

    print("  --- If All Review TPs Accepted ---")
    print(
        f"  Combined TP:              {combined_tp:>6,}  of {combined_ceiling:,} reachable"
    )
    print(
        f"  Combined precision:       {combined_precision:>9.4f}  ({combined_precision * 100:.1f}%)"
    )
    print(
        f"  Combined recall:          {combined_recall:>9.4f}  ({combined_recall * 100:.1f}%)  ***"
        if combined_recall >= 0.999
        else f"  Combined recall:          {combined_recall:>9.4f}  ({combined_recall * 100:.1f}%)"
    )
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
