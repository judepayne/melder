#!/usr/bin/env python3
"""Evaluate meld batch output against ground truth.

Compares results.csv, review.csv, and unmatched.csv against the ground truth
crossmap to produce a clean breakdown of matches, false positives, misses,
and correctly excluded records across all three decision buckets.

Usage:
    python3 benchmarks/accuracy/eval.py \
        --results   benchmarks/accuracy/10kx10k_embeddings/output/results.csv \
        --review    benchmarks/accuracy/10kx10k_embeddings/output/review.csv \
        --unmatched benchmarks/accuracy/10kx10k_embeddings/output/unmatched.csv \
        --ground-truth benchmarks/data/ground_truth_crossmap_10k.csv \
        --dataset-a benchmarks/data/dataset_a_10k.csv \
        --dataset-b benchmarks/data/dataset_b_10k.csv \
        --config    benchmarks/accuracy/10kx10k_embeddings/config.yaml
"""

import argparse
import csv


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_pairs(
    path: str, a_col: str = "a_id", b_col: str = "b_id"
) -> set[tuple[str, str]]:
    """Load (a_id, b_id) pairs from a CSV file."""
    pairs: set[tuple[str, str]] = set()
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
    """Load dataset B metadata keyed by counterparty_id."""
    meta: dict[str, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            meta[row["counterparty_id"]] = dict(row)
    return meta


def load_exact_prefilter_fields(config_path: str) -> list[tuple[str, str]]:
    """Read exact_prefilter field pairs from a meld config YAML."""
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


# ---------------------------------------------------------------------------
# Ceiling
# ---------------------------------------------------------------------------


def compute_ceiling(
    dataset_a_path: str,
    b_meta: dict[str, dict],
    exact_fields: list[tuple[str, str]],
) -> int:
    """Compute the combined ceiling: true matches reachable by blocking or exact prefilter."""
    a_records: dict[str, dict] = {}
    with open(dataset_a_path) as f:
        for row in csv.DictReader(f):
            a_records[row["entity_id"]] = row

    combined_unreachable = 0
    for meta in b_meta.values():
        if meta["_match_type"] != "matched":
            continue
        a_rec = a_records.get(meta["_true_a_id"], {})
        is_blocked = a_rec.get("country_code") != meta.get("domicile")
        if is_blocked:
            exact_reachable = exact_fields and all(
                a_rec.get(fa, "").strip().lower() == meta.get(fb, "").strip().lower()
                and a_rec.get(fa, "").strip() != ""
                and meta.get(fb, "").strip() != ""
                for fa, fb in exact_fields
            )
            if not exact_reachable:
                combined_unreachable += 1

    total_true = sum(1 for m in b_meta.values() if m["_match_type"] == "matched")
    return total_true - combined_unreachable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
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
    parser.add_argument("--dataset-b", required=True, help="Path to dataset_b CSV")
    parser.add_argument(
        "--config", default=None, help="Path to meld config YAML (optional)"
    )
    args = parser.parse_args()

    exact_fields = load_exact_prefilter_fields(args.config) if args.config else []

    gt = load_ground_truth(args.ground_truth)
    b_meta = load_b_metadata(args.dataset_b)
    auto = load_pairs(args.results)
    review = load_pairs(args.review)

    ceiling = compute_ceiling(args.dataset_a, b_meta, exact_fields)

    # --- Auto-matched ---
    auto_total = len(auto)
    auto_match = sum(1 for pair in auto if pair in gt)
    auto_fp = auto_total - auto_match

    # --- Review ---
    review_total = len(review)
    review_match = sum(1 for pair in review if pair in gt)
    review_fp = review_total - review_match

    # --- Unmatched ---
    # _match_type == "matched" → has a true match → missed (FN)
    # _match_type == "ambiguous" / "unmatched" → correctly excluded (TN)
    unmatched_missed = 0
    unmatched_excluded = 0
    try:
        with open(args.unmatched) as f:
            for row in csv.DictReader(f):
                if row.get("_match_type") == "matched":
                    unmatched_missed += 1
                else:
                    unmatched_excluded += 1
    except FileNotFoundError:
        pass
    unmatched_total = unmatched_missed + unmatched_excluded

    # --- Output ---
    L = 30  # label column width
    N = 8  # number column width

    def line(label: str, value: int, indent: int = 0) -> None:
        pad = "  " * indent
        print(f"  {pad}{label:<{L - len(pad)}} {value:>{N},}")

    print()
    print("=" * 60)
    print("  ACCURACY EVALUATION")
    print("=" * 60)
    print()
    line("Ceiling (max reachable)", ceiling)
    print()
    line("Auto-matched", auto_total)
    line("Match", auto_match, indent=1)
    line("False positive", auto_fp, indent=1)
    print()
    line("Review", review_total)
    line("Match", review_match, indent=1)
    line("False positive", review_fp, indent=1)
    print()
    line("Unmatched", unmatched_total)
    line("Missed", unmatched_missed, indent=1)
    line("Correctly excluded", unmatched_excluded, indent=1)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
