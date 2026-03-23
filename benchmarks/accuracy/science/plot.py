#!/usr/bin/env python3
"""Plot learning curves from results/metrics.csv.

Produces a 2×2 figure showing precision, recall, false positive count,
and review queue size over rounds — each with training and holdout lines.

Usage:
    python plot.py --metrics results/metrics.csv --output results/learning_curve.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def plot_learning_curve(metrics_csv: str, output_path: str) -> None:
    """Read metrics.csv and write learning_curve.png."""
    rows: list[dict] = []
    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No data in metrics.csv — skipping plot.")
        return

    rounds = [int(r["round"]) for r in rows]

    def col(key: str) -> list[float]:
        return [float(r.get(key, 0) or 0) for r in rows]

    has_holdout = "holdout_precision" in rows[0] and rows[0]["holdout_precision"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Melder Embedding Fine-Tuning — Learning Curve", fontsize=14, y=0.98)

    plot_cfg = [
        (axes[0, 0], "Precision (auto-match)",
         "train_precision",    "holdout_precision",    True),
        (axes[0, 1], "Recall vs Blocking Ceiling",
         "train_recall_ceiling", "holdout_recall_ceiling", True),
        (axes[1, 0], "False Positives (auto-match)",
         "train_auto_fp",      "holdout_auto_fp",      False),
        (axes[1, 1], "Review Queue Size",
         "train_review",       "holdout_review",       False),
    ]

    colors = ["steelblue", "darkorange", "firebrick", "mediumseagreen"]

    for (ax, title, train_key, holdout_key, is_ratio), color in zip(plot_cfg, colors):
        train_vals = col(train_key)
        ax.plot(rounds, train_vals, color=color, marker="o", linewidth=2,
                label="Training data")

        if has_holdout:
            holdout_vals = col(holdout_key)
            ax.plot(rounds, holdout_vals, color=color, marker="s",
                    linewidth=2, linestyle="--", alpha=0.75, label="Holdout")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round")
        if is_ratio:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curve saved → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default="results/metrics.csv")
    p.add_argument("--output",  default="results/learning_curve.png")
    args = p.parse_args()
    plot_learning_curve(args.metrics, args.output)


if __name__ == "__main__":
    main()
