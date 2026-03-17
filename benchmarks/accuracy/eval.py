#!/usr/bin/env python3
"""Evaluate meld batch output against ground truth.

Compares results.csv (auto-matched) and review.csv against the ground truth
crossmap and dataset B metadata to compute precision, recall, and F1.

The theoretical ceiling (max reachable matches given blocking) is computed
dynamically from the actual datasets, not hardcoded.

Usage:
    python3 benchmarks/accuracy/eval.py \
        --results  benchmarks/accuracy/10kx10k_bm25/output/results.csv \
        --review   benchmarks/accuracy/10kx10k_bm25/output/review.csv \
        --unmatched benchmarks/accuracy/10kx10k_bm25/output/unmatched.csv \
        --ground-truth benchmarks/data/ground_truth_crossmap_10k.csv \
        --dataset-a benchmarks/data/dataset_a_10k.csv \
        --dataset-b benchmarks/data/dataset_b_10k.csv
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
            reader = csv.DictReader(f)
            for row in reader:
                pairs.add((row[a_col], row[b_col]))
    except FileNotFoundError:
        pass
    return pairs


def load_ground_truth(path: str) -> set[tuple[str, str]]:
    """Load ground truth pairs as (entity_id, counterparty_id)."""
    return load_pairs(path, "entity_id", "counterparty_id")


def load_b_metadata(path: str) -> dict[str, dict]:
    """Load dataset B with _match_type and _true_a_id for each record."""
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            meta[row["counterparty_id"]] = {
                "match_type": row["_match_type"],
                "true_a_id": row["_true_a_id"],
                "domicile": row["domicile"],
            }
    return meta


def compute_ceiling(
    dataset_a_path: str, b_meta: dict[str, dict]
) -> tuple[int, int, int]:
    """Compute theoretical ceiling from actual data.

    Returns (total_true_matches, blocked_out, ceiling).
    A matched B record is 'blocked out' if its domicile differs from the
    true A record's country_code (blocking will prevent scoring).
    """
    a_country = {}
    with open(dataset_a_path) as f:
        for row in csv.DictReader(f):
            a_country[row["entity_id"]] = row["country_code"]

    total_matched = 0
    blocked = 0
    for meta in b_meta.values():
        if meta["match_type"] == "matched":
            total_matched += 1
            true_a_id = meta["true_a_id"]
            if a_country.get(true_a_id) != meta["domicile"]:
                blocked += 1

    return total_matched, blocked, total_matched - blocked


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
    args = parser.parse_args()

    # Load data
    gt = load_ground_truth(args.ground_truth)
    b_meta = load_b_metadata(args.dataset_b)
    auto_matched = load_pairs(args.results)
    review_pairs = load_pairs(args.review)

    # Compute ceiling dynamically
    total_true_matches, blocked_out, ceiling = compute_ceiling(args.dataset_a, b_meta)
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
            meta = b_meta.get(b_id, {})
            mt = meta.get("match_type", "unknown")
            if mt == "matched":
                auto_fp_detail["matched_wrong"] += 1
            elif mt == "ambiguous":
                auto_fp_detail["ambiguous"] += 1
            elif mt == "unmatched":
                auto_fp_detail["unmatched"] += 1

    # For review pairs
    review_tp = 0
    review_fp = 0
    for a_id, b_id in review_pairs:
        if (a_id, b_id) in gt:
            review_tp += 1
        else:
            review_fp += 1

    # --- Metrics ---
    precision = auto_tp / len(auto_matched) if auto_matched else 0.0
    recall_ceiling = auto_tp / ceiling if ceiling > 0 else 0.0
    recall_total = auto_tp / total_true_matches if total_true_matches > 0 else 0.0
    f1_ceiling = (
        2 * precision * recall_ceiling / (precision + recall_ceiling)
        if (precision + recall_ceiling) > 0
        else 0.0
    )
    f1_total = (
        2 * precision * recall_total / (precision + recall_total)
        if (precision + recall_total) > 0
        else 0.0
    )

    # --- Output ---
    print()
    print("=" * 60)
    print("  ACCURACY EVALUATION")
    print("=" * 60)
    print()
    print(
        f"  Theoretical ceiling:      {ceiling:>6,}  ({total_true_matches:,} matched - {blocked_out:,} blocked)"
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
        f"  Recall (vs ceiling):      {recall_ceiling:>9.4f}  ({recall_ceiling * 100:.1f}%)  [of {ceiling:,} reachable]"
    )
    print(
        f"  Recall (vs total):        {recall_total:>9.4f}  ({recall_total * 100:.1f}%)  [of {total_true_matches:,} true matches]"
    )
    print(f"  F1 (vs ceiling):          {f1_ceiling:>9.4f}")
    print(f"  F1 (vs total):            {f1_total:>9.4f}")
    print()

    # Combined: what if review TPs were also accepted?
    combined_tp = auto_tp + review_tp
    combined_total = len(auto_matched) + len(review_pairs)
    combined_precision = combined_tp / combined_total if combined_total > 0 else 0.0
    combined_recall_ceiling = combined_tp / ceiling if ceiling > 0 else 0.0
    combined_f1 = (
        2
        * combined_precision
        * combined_recall_ceiling
        / (combined_precision + combined_recall_ceiling)
        if (combined_precision + combined_recall_ceiling) > 0
        else 0.0
    )

    print("  --- If All Review TPs Accepted ---")
    print(f"  Combined TP:              {combined_tp:>6,}")
    print(
        f"  Combined precision:       {combined_precision:>9.4f}  ({combined_precision * 100:.1f}%)"
    )
    print(
        f"  Combined recall (ceil):   {combined_recall_ceiling:>9.4f}  ({combined_recall_ceiling * 100:.1f}%)"
    )
    print(f"  Combined F1:              {combined_f1:>9.4f}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
