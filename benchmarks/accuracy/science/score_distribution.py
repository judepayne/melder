#!/usr/bin/env python3
"""Score distribution by population (matched/ambiguous/unmatched).

Shows how scores distribute across the three ground-truth populations
for a given experiment round's holdout output.

Usage:
    python benchmarks/accuracy/science/score_distribution.py --experiment baseline --round 0
    python benchmarks/accuracy/science/score_distribution.py --experiment baseline --round 0 --bucket-width 0.05
"""

import argparse
import csv
import os

SCIENCE_DIR = "benchmarks/accuracy/science"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, help="Experiment name")
    p.add_argument("--round", type=int, required=True, help="Round number")
    p.add_argument(
        "--bucket-width", type=float, default=0.02, help="Bucket width (default: 0.02)"
    )
    p.add_argument(
        "--training", action="store_true",
        help="Use training data instead of holdout",
    )
    args = p.parse_args()

    exp_dir = os.path.join(SCIENCE_DIR, "results", args.experiment)
    if args.training:
        output_dir = os.path.join(exp_dir, "work", f"round_{args.round}", "output")
        b_path = os.path.join(SCIENCE_DIR, "rounds", f"round_{args.round}", "dataset_b.csv")
    else:
        output_dir = os.path.join(exp_dir, "holdout", f"round_{args.round}", "output")
        b_path = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

    a_path = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")

    # Load ground truth
    b_meta = {}
    with open(b_path) as f:
        for row in csv.DictReader(f):
            b_meta[row["counterparty_id"]] = row

    # Build buckets
    w = args.bucket_width
    n_buckets = int(1.0 / w) + 1  # extra bucket for 1.0
    matched = [0] * n_buckets
    ambiguous = [0] * n_buckets
    unmatched = [0] * n_buckets

    def add(score: float, b_id: str, a_id: str) -> None:
        b = b_meta.get(b_id, {})
        mt = b.get("_match_type", "unmatched")
        true_a = b.get("_true_a_id", "")
        bkt = min(int(score / w), n_buckets - 1)

        if mt == "matched" and a_id == true_a:
            matched[bkt] += 1
        elif mt == "ambiguous":
            ambiguous[bkt] += 1
        else:
            unmatched[bkt] += 1

    # Scored pairs (results + review)
    for fname in ["results.csv", "review.csv"]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                add(float(row["score"]), row["b_id"], row["a_id"])

    # Unmatched (below review_floor)
    seen_b = set()
    for fname in ["results.csv", "review.csv"]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                seen_b.add(row["b_id"])

    unmatched_path = os.path.join(output_dir, "unmatched.csv")
    if os.path.exists(unmatched_path):
        with open(unmatched_path) as f:
            for row in csv.DictReader(f):
                bid = row["counterparty_id"]
                if bid in seen_b:
                    continue
                score = float(row.get("score", "0"))
                b = b_meta.get(bid, {})
                mt = b.get("_match_type", "unmatched")
                bkt = min(int(score / w), n_buckets - 1)
                if mt == "matched":
                    matched[bkt] += 1
                elif mt == "ambiguous":
                    ambiguous[bkt] += 1
                else:
                    unmatched[bkt] += 1

    # Print
    source = "training" if args.training else "holdout"
    print(f"\n  Score distribution — {args.experiment} round {args.round} ({source})\n")
    print(f"  {'Bucket':<12} {'matched':>8} {'ambiguous':>10} {'unmatched':>10}  {'total':>7}")
    print("  " + "-" * 53)

    for i in range(n_buckets):
        t = matched[i] + ambiguous[i] + unmatched[i]
        if t == 0:
            continue
        lo = i * w
        hi = lo + w
        print(
            f"  {lo:.2f}-{hi:.2f}    {matched[i]:>8,} {ambiguous[i]:>10,} "
            f"{unmatched[i]:>10,}  {t:>7,}"
        )

    print("  " + "-" * 53)
    print(
        f"  {'Total':<12} {sum(matched):>8,} {sum(ambiguous):>10,} "
        f"{sum(unmatched):>10,}  {sum(matched) + sum(ambiguous) + sum(unmatched):>7,}"
    )
    print()


if __name__ == "__main__":
    main()
