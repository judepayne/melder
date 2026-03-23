#!/usr/bin/env python3
"""Compute overlap coefficient between matched and unmatched score distributions.

0 = perfect separation, 1 = identical distributions.

Usage:
    python benchmarks/accuracy/science/overlap.py --experiment baseline --round 0
    python benchmarks/accuracy/science/overlap.py --experiment baseline --rounds 0-4
"""

import argparse
import csv
import os

SCIENCE_DIR = "benchmarks/accuracy/science"


def load_scores(experiment: str, round_idx: int, training: bool):
    """Return (matched_scores, unmatched_scores) lists."""
    exp_dir = os.path.join(SCIENCE_DIR, "results", experiment)
    if training:
        output_dir = os.path.join(exp_dir, "work", f"round_{round_idx}", "output")
        b_path = os.path.join(SCIENCE_DIR, "rounds", f"round_{round_idx}", "dataset_b.csv")
    else:
        output_dir = os.path.join(exp_dir, "holdout", f"round_{round_idx}", "output")
        b_path = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

    b_meta = {}
    with open(b_path) as f:
        for row in csv.DictReader(f):
            b_meta[row["counterparty_id"]] = row

    matched_scores = []
    unmatched_scores = []

    seen_b = set()
    for fname in ["results.csv", "review.csv"]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                score = float(row["score"])
                b = b_meta.get(row["b_id"], {})
                mt = b.get("_match_type", "unmatched")
                true_a = b.get("_true_a_id", "")
                seen_b.add(row["b_id"])

                if mt == "matched" and row["a_id"] == true_a:
                    matched_scores.append(score)
                elif mt == "ambiguous":
                    matched_scores.append(score)
                else:
                    unmatched_scores.append(score)

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
                if mt == "matched" or mt == "ambiguous":
                    matched_scores.append(score)
                else:
                    unmatched_scores.append(score)

    return matched_scores, unmatched_scores


def overlap_coefficient(matched_scores, unmatched_scores, bucket_width=0.02):
    """Compute overlap coefficient between two score distributions."""
    n_buckets = int(1.0 / bucket_width) + 1

    m_hist = [0] * n_buckets
    u_hist = [0] * n_buckets

    for s in matched_scores:
        m_hist[min(int(s / bucket_width), n_buckets - 1)] += 1
    for s in unmatched_scores:
        u_hist[min(int(s / bucket_width), n_buckets - 1)] += 1

    m_total = sum(m_hist) or 1
    u_total = sum(u_hist) or 1

    overlap = sum(
        min(m / m_total, u / u_total) for m, u in zip(m_hist, u_hist)
    )

    return overlap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, help="Experiment name")
    p.add_argument("--round", type=int, default=None, help="Single round number")
    p.add_argument("--rounds", default=None, help="Round range, e.g. 0-4")
    p.add_argument("--training", action="store_true", help="Use training data")
    args = p.parse_args()

    if args.rounds:
        start, end = args.rounds.split("-")
        round_range = range(int(start), int(end) + 1)
    elif args.round is not None:
        round_range = [args.round]
    else:
        p.error("Specify --round N or --rounds N-M")

    source = "training" if args.training else "holdout"
    print(f"\n  Overlap coefficient — {args.experiment} ({source})")
    print(f"  0 = perfect separation, 1 = identical\n")

    for r in round_range:
        try:
            m, u = load_scores(args.experiment, r, args.training)
            ov = overlap_coefficient(m, u)
            label = "base" if r == 0 else f"R{r}"
            print(f"  Round {r} ({label}):  {ov:.4f}   ({len(m):,} matched, {len(u):,} unmatched)")
        except FileNotFoundError:
            print(f"  Round {r}: no data")

    print()


if __name__ == "__main__":
    main()
