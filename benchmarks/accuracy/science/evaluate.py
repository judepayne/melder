#!/usr/bin/env python3
"""Ground-truth evaluation using _true_a_id and _match_type.

Classifies melder output against the hidden ground-truth fields embedded in
the B-side dataset — no human review or external crossmap required.

Returns a structured metrics dict and prints a summary.

Usage (standalone):
    python evaluate.py \\
        --round-dir benchmarks/accuracy/science/rounds/round_0 \\
        --config   benchmarks/accuracy/science/work/round_0/config.yaml
"""

import argparse
import csv
import os
import sys


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_dataset(path: str, id_field: str) -> dict[str, dict]:
    """Load a CSV keyed by id_field."""
    records = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            records[row[id_field]] = dict(row)
    return records


def load_result_pairs(path: str) -> list[tuple[str, str]]:
    """Load (a_id, b_id) pairs from a results/review CSV. Returns [] if absent."""
    pairs = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                pairs.append((row["a_id"], row["b_id"]))
    except FileNotFoundError:
        pass
    return pairs


# ---------------------------------------------------------------------------
# Blocking ceiling
# ---------------------------------------------------------------------------


def compute_blocking_ceiling(a_records: dict, b_meta: dict) -> int:
    """Count _match_type=matched B records whose country code agrees with A."""
    ceiling = 0
    for b in b_meta.values():
        if b.get("_match_type") != "matched":
            continue
        a = a_records.get(b.get("_true_a_id", ""), {})
        if a.get("country_code") == b.get("domicile"):
            ceiling += 1
    return ceiling


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_round(
    dataset_a: str,
    dataset_b: str,
    results: str,
    review: str,
    print_summary: bool = True,
) -> dict:
    """Evaluate one round and return a metrics dict.

    Parameters
    ----------
    dataset_a:     path to the A-side CSV (needs entity_id, country_code)
    dataset_b:     path to the B-side CSV (needs _true_a_id, _match_type)
    results:       path to meld's results.csv (auto-matched)
    review:        path to meld's review.csv
    print_summary: when True, print a human-readable summary to stdout
    """
    a_records = load_dataset(dataset_a, "entity_id")
    b_meta = load_dataset(dataset_b, "counterparty_id")

    auto_pairs = load_result_pairs(results)
    review_pairs = load_result_pairs(review)

    auto_matched_b = {b_id for _, b_id in auto_pairs}
    review_b = {b_id for _, b_id in review_pairs}
    seen_b = auto_matched_b | review_b

    total_true_matches = sum(
        1 for b in b_meta.values() if b.get("_match_type") == "matched"
    )
    blocking_ceiling = compute_blocking_ceiling(a_records, b_meta)

    # --- Classify auto-matched pairs ---
    auto_tp = auto_fp = 0
    auto_fp_wrong = auto_fp_ambiguous = auto_fp_unmatched = 0

    for a_id, b_id in auto_pairs:
        b = b_meta.get(b_id, {})
        mt = b.get("_match_type", "unknown")
        true_a = b.get("_true_a_id", "")

        if mt == "matched" and a_id == true_a:
            auto_tp += 1
        else:
            auto_fp += 1
            if mt == "matched":
                auto_fp_wrong += 1
            elif mt == "ambiguous":
                auto_fp_ambiguous += 1
            else:
                auto_fp_unmatched += 1

    # --- Classify review pairs ---
    review_tp = review_fp = 0
    review_fp_wrong = review_fp_ambiguous = review_fp_unmatched = 0
    for a_id, b_id in review_pairs:
        b = b_meta.get(b_id, {})
        mt = b.get("_match_type", "unmatched")
        if mt == "matched" and a_id == b.get("_true_a_id"):
            review_tp += 1
        else:
            review_fp += 1
            if mt == "matched":
                review_fp_wrong += 1
            elif mt == "ambiguous":
                review_fp_ambiguous += 1
            else:
                review_fp_unmatched += 1

    # --- Unmatched breakdown ---
    fn = sum(
        1
        for b in b_meta.values()
        if b.get("_match_type") == "matched" and b["counterparty_id"] not in seen_b
    )
    fn_ambiguous = sum(
        1
        for b in b_meta.values()
        if b.get("_match_type") == "ambiguous" and b["counterparty_id"] not in seen_b
    )
    tn = sum(
        1
        for b in b_meta.values()
        if b.get("_match_type") == "unmatched" and b["counterparty_id"] not in seen_b
    )

    # --- Metrics ---
    n_auto = len(auto_pairs)
    n_review = len(review_pairs)

    precision = auto_tp / n_auto if n_auto else 0.0
    recall_ceiling = auto_tp / blocking_ceiling if blocking_ceiling else 0.0
    recall_total = auto_tp / total_true_matches if total_true_matches else 0.0
    combined_tp = auto_tp + review_tp
    combined_recall = combined_tp / total_true_matches if total_true_matches else 0.0

    metrics = {
        "auto_matched": n_auto,
        "auto_tp": auto_tp,
        "auto_fp": auto_fp,
        "auto_fp_wrong": auto_fp_wrong,
        "auto_fp_ambiguous": auto_fp_ambiguous,
        "auto_fp_unmatched": auto_fp_unmatched,
        "review": n_review,
        "review_tp": review_tp,
        "review_fp": review_fp,
        "review_fp_wrong": review_fp_wrong,
        "review_fp_ambiguous": review_fp_ambiguous,
        "review_fp_unmatched": review_fp_unmatched,
        "fn": fn,
        "fn_ambiguous": fn_ambiguous,
        "tn": tn,
        "total_true_matches": total_true_matches,
        "blocking_ceiling": blocking_ceiling,
        "precision": round(precision, 4),
        "recall_ceiling": round(recall_ceiling, 4),
        "recall_total": round(recall_total, 4),
        "combined_recall": round(combined_recall, 4),
    }

    if print_summary:
        _print_summary(metrics)

    return metrics


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(m: dict) -> None:
    L, N = 28, 8

    def line(label: str, value: int, indent: int = 0) -> None:
        pad = "  " * indent
        print(f"  {pad}{label:<{L - len(pad)}} {value:>{N},}")

    print()
    print("=" * 60)
    print("  ACCURACY EVALUATION")
    print("=" * 60)
    print()
    line("Ceiling (max reachable)", m["blocking_ceiling"])
    print()
    line("Auto-matched", m["auto_matched"])
    line("matched", m["auto_tp"], indent=1)
    line("ambiguous", m["auto_fp_ambiguous"], indent=1)
    line("unmatched", m["auto_fp_unmatched"] + m["auto_fp_wrong"], indent=1)
    print()
    line("Review", m["review"])
    line("matched", m["review_tp"], indent=1)
    line("ambiguous", m.get("review_fp_ambiguous", 0), indent=1)
    line("unmatched", m.get("review_fp_unmatched", 0) + m.get("review_fp_wrong", 0), indent=1)
    print()
    line("Unmatched", m["fn"] + m["fn_ambiguous"] + m["tn"])
    line("matched", m["fn"], indent=1)
    line("ambiguous", m["fn_ambiguous"], indent=1)
    line("unmatched", m["tn"], indent=1)
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--round-dir",
        required=True,
        help="Round directory (contains dataset_*.csv and output/)",
    )
    p.add_argument("--config", default=None, help="Unused (accepted for compatibility)")
    args = p.parse_args()

    d = args.round_dir
    metrics = evaluate_round(
        dataset_a=os.path.join(d, "dataset_a.csv"),
        dataset_b=os.path.join(d, "dataset_b.csv"),
        results=os.path.join(d, "output", "results.csv"),
        review=os.path.join(d, "output", "review.csv"),
        print_summary=True,
    )
    print(
        f"\nPrecision: {metrics['precision']:.1%}  "
        f"Recall: {metrics['recall_ceiling']:.1%}  "
        f"FP: {metrics['auto_fp']}"
    )


if __name__ == "__main__":
    main()
