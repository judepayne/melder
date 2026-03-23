#!/usr/bin/env python3
"""ASCII histogram of score distributions by population.

Shows matched, ambiguous, and unmatched populations as overlaid
bar charts in the terminal.

Usage:
    python benchmarks/accuracy/science/score_chart.py --experiment baseline --round 0
    python benchmarks/accuracy/science/score_chart.py --experiment baseline --round 0 --width 80
"""

import argparse
import csv
import os

import yaml

SCIENCE_DIR = "benchmarks/accuracy/science"

MATCHED_CHAR = "█"
AMBIGUOUS_CHAR = "▒"
UNMATCHED_CHAR = "░"


def load_distribution(experiment: str, round_idx: int, training: bool, bucket_width: float):
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

    w = bucket_width
    n_buckets = int(1.0 / w) + 1

    matched = [0] * n_buckets
    ambiguous = [0] * n_buckets
    unmatched = [0] * n_buckets

    def add(score, b_id, a_id):
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

    seen_b = set()
    for fname in ["results.csv", "review.csv"]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                add(float(row["score"]), row["b_id"], row["a_id"])
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

    return matched, ambiguous, unmatched


THRESHOLD_CHAR = "▼"


def render_chart(matched, ambiguous, unmatched, bucket_width, bar_width, split_ambiguous, max_score=None, thresholds=None):
    w = bucket_width
    n_buckets = len(matched)

    if not split_ambiguous:
        # Merge ambiguous into matched
        matched = [m + a for m, a in zip(matched, ambiguous)]
        ambiguous = [0] * n_buckets

    # Find range with data
    first = next((i for i in range(n_buckets) if matched[i] + ambiguous[i] + unmatched[i] > 0), 0)
    last = next((i for i in range(n_buckets - 1, -1, -1) if matched[i] + ambiguous[i] + unmatched[i] > 0), n_buckets - 1)

    if max_score is not None:
        last = min(last, int(max_score / w))

    # Include threshold buckets in visible range
    if thresholds:
        for val, _label in thresholds:
            bkt = int(val / w)
            if 0 <= bkt <= n_buckets - 1:
                first = min(first, bkt)
                last = max(last, bkt)

    max_count = max(
        max(matched[first:last+1]),
        max(ambiguous[first:last+1]) if split_ambiguous else 0,
        max(unmatched[first:last+1]),
        1,
    )

    scale = bar_width / max_count
    label_width = 7  # "  0.60 "

    # Pre-compute which buckets have threshold markers
    threshold_buckets = {}
    if thresholds:
        for val, tlabel in thresholds:
            bkt = int(val / w)
            if first <= bkt <= last:
                threshold_buckets[bkt] = (val, tlabel)

    lines = []
    for i in range(first, last + 1):
        lo = i * w
        m = matched[i]
        a = ambiguous[i]
        u = unmatched[i]

        m_len = int(m * scale + 0.5)
        a_len = int(a * scale + 0.5)
        u_len = int(u * scale + 0.5)

        # Ensure at least 1 char if count > 0
        if m > 0 and m_len == 0:
            m_len = 1
        if a > 0 and a_len == 0:
            a_len = 1
        if u > 0 and u_len == 0:
            u_len = 1

        bar_m = MATCHED_CHAR * m_len
        bar_a = AMBIGUOUS_CHAR * a_len if split_ambiguous else ""
        bar_u = UNMATCHED_CHAR * u_len

        label = f"  {lo:.2f} "

        if i in threshold_buckets:
            val, tlabel = threshold_buckets[i]
            bar = f"{bar_m}{bar_a}{bar_u}"
            marker = f"  {THRESHOLD_CHAR} {tlabel} ({val:.2f})"
            lines.append(f"{label}{bar}{marker}")
        else:
            lines.append(f"{label}{bar_m}{bar_a}{bar_u}")

    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, help="Experiment name")
    p.add_argument("--round", type=int, required=True, help="Round number")
    p.add_argument("--width", type=int, default=None, help="Max bar width (overrides --size)")
    p.add_argument("--bucket-width", type=float, default=None, help="Bucket width (overrides --size)")
    p.add_argument(
        "--size", choices=["big", "medium", "small"], default="medium",
        help="Chart size preset (default: medium)",
    )
    p.add_argument("--training", action="store_true", help="Use training data instead of holdout")
    p.add_argument("--split-ambiguous", action="store_true", help="Show ambiguous separately from matched")
    p.add_argument("--max-score", type=float, default=None, help="Upper score cutoff for display")
    p.add_argument("--config", default=None, help="Config YAML to read thresholds from (auto_match + review_floor)")
    p.add_argument("--auto-match", type=float, default=None, help="Auto-match threshold (overrides config)")
    p.add_argument("--review-floor", type=float, default=None, help="Review floor threshold (overrides config)")
    args = p.parse_args()

    size_presets = {
        "big":    {"width": 60, "bucket_width": 0.02},
        "medium": {"width": 40, "bucket_width": 0.04},
        "small":  {"width": 25, "bucket_width": 0.08},
    }
    preset = size_presets[args.size]
    bar_width = args.width or preset["width"]
    bucket_width = args.bucket_width or preset["bucket_width"]

    matched, ambiguous, unmatched = load_distribution(
        args.experiment, args.round, args.training, bucket_width
    )

    source = "training" if args.training else "holdout"
    print(f"\n  {args.experiment} round {args.round} ({source})")
    if args.split_ambiguous:
        print(f"  {MATCHED_CHAR} matched  {AMBIGUOUS_CHAR} ambiguous  {UNMATCHED_CHAR} unmatched")
    else:
        print(f"  {MATCHED_CHAR} matched+ambiguous  {UNMATCHED_CHAR} unmatched")
    print()

    # Load thresholds
    auto_match = args.auto_match
    review_floor = args.review_floor
    if args.config and (auto_match is None or review_floor is None):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        t = cfg.get("thresholds", {})
        if auto_match is None:
            auto_match = t.get("auto_match")
        if review_floor is None:
            review_floor = t.get("review_floor")

    thresholds = []
    if review_floor is not None:
        thresholds.append((review_floor, "review_floor"))
    if auto_match is not None:
        thresholds.append((auto_match, "auto_match"))

    lines = render_chart(matched, ambiguous, unmatched, bucket_width, bar_width, args.split_ambiguous, args.max_score, thresholds=thresholds or None)
    for line in lines:
        print(line)

    print()
    if args.split_ambiguous:
        print(f"  Totals: matched={sum(matched):,}  ambiguous={sum(ambiguous):,}  unmatched={sum(unmatched):,}")
    else:
        print(f"  Totals: matched+ambiguous={sum(matched) + sum(ambiguous):,}  unmatched={sum(unmatched):,}")
    print()


if __name__ == "__main__":
    main()
