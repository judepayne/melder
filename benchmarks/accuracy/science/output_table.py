#!/usr/bin/env python3
"""Print holdout accuracy table from a science experiment's metrics.csv.

Usage:
    python benchmarks/accuracy/science/output_table.py
    python benchmarks/accuracy/science/output_table.py --csv path/to/metrics.csv
"""

import argparse
import csv
import os
import sys

SCIENCE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCIENCE_DIR, "results", "metrics.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=RESULTS_CSV,
        help=f"Path to metrics.csv (default: {RESULTS_CSV})",
    )
    return p.parse_args()


def load_rounds(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def fmt(n: int) -> str:
    return f"{n:>,}"


def pct(v: float) -> str:
    return f"{v:.1%}"


def print_table(rounds: list[dict]) -> None:
    n = len(rounds)
    label_w = 36
    col_w = 10

    sep = "=" * (label_w + col_w * n)

    model = rounds[0].get("model", "unknown")
    if model.startswith("base (") and model.endswith(")"):
        model = model[6:-1]

    print(sep)
    print(f"  {model} — holdout results")
    print()

    header = " " * label_w
    for r in rounds:
        idx = int(r["round"])
        label = f"Round {idx}" + (" (base)" if idx == 0 else "")
        header += f"{label:>{col_w}}"
    print(header)
    print(sep)

    def row(label, values, indent=0, is_pct=False):
        pad = "  " * indent
        name = f"  {pad}{label}"
        line = f"{name:<{label_w}}"
        for v in values:
            line += f"{pct(v):>{col_w}}" if is_pct else f"{fmt(v):>{col_w}}"
        print(line)

    def blank():
        print()

    def h(field):
        return [int(r[f"holdout_{field}"]) for r in rounds]

    def hf(field):
        return [float(r[f"holdout_{field}"]) for r in rounds]

    auto_matched = h("auto_matched")
    auto_tp = h("auto_tp")
    auto_fp_ambiguous = h("auto_fp_ambiguous")
    auto_fp_unmatched = h("auto_fp_unmatched")

    review = h("review")
    review_tp = h("review_tp")
    review_fp_ambiguous = h("review_fp_ambiguous")
    review_fp_unmatched = h("review_fp_unmatched")

    fn = h("fn")
    fn_ambiguous = h("fn_ambiguous")
    tn = h("tn")
    unmatched = [f + fa + t for f, fa, t in zip(fn, fn_ambiguous, tn)]

    ceiling = h("blocking_ceiling")

    row("Ceiling (max reachable)", ceiling)
    blank()
    row("Auto-matched", auto_matched)
    row("Clean", auto_tp, indent=1)
    row("Heavy noise", auto_fp_ambiguous, indent=1)
    row("Not a match", auto_fp_unmatched, indent=1)
    blank()
    row("Review", review)
    row("Clean", review_tp, indent=1)
    row("Heavy noise", review_fp_ambiguous, indent=1)
    row("Not a match", review_fp_unmatched, indent=1)
    blank()
    row("Unmatched", unmatched)
    row("Missed (clean)", fn, indent=1)
    row("Missed (heavy noise)", fn_ambiguous, indent=1)
    row("Not a match", tn, indent=1)
    blank()
    row("Precision", hf("precision"), is_pct=True)
    row("Recall (vs ceiling)", hf("recall_ceiling"), is_pct=True)
    row("Combined recall", hf("combined_recall"), is_pct=True)

    print(sep)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: metrics CSV not found at '{args.csv}'")
        sys.exit(1)

    rounds = load_rounds(args.csv)
    if not rounds:
        print("Error: no data in metrics CSV")
        sys.exit(1)

    print_table(rounds)


if __name__ == "__main__":
    main()
