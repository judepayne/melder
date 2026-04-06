#!/usr/bin/env python3
"""Experiment 10 — BM25 composite scoring with Arctic-embed-xs.

Self-contained experiment that tests adding BM25 to the best fine-tuned
Arctic-embed-xs model from Experiment 9. Sweeps BM25 weights from 0% to 40%.

Requires experiment 9 to be complete (uses its best model).

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment10.py [--best-round N]
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.abspath("benchmarks/accuracy/science"))

# ---------------------------------------------------------------------------
# Paths (all relative to project root; run from there)
# ---------------------------------------------------------------------------

SCIENCE_DIR = "benchmarks/accuracy/science"
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_10")
MELD_BINARY = "./target/release/meld"

SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

BM25_WEIGHTS = [0.00, 0.10, 0.20, 0.30, 0.40]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_best_round(experiment: str) -> int:
    """Find the round with the lowest holdout overlap coefficient."""
    from overlap import load_scores, overlap_coefficient

    results_dir = os.path.join(SCIENCE_DIR, "results", experiment, "holdout")
    best_round = 0
    best_overlap = float("inf")

    for entry in sorted(os.listdir(results_dir)):
        if not entry.startswith("round_"):
            continue
        r = int(entry.split("_")[1])
        metrics_path = os.path.join(results_dir, entry, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        try:
            m, u = load_scores(experiment, r, training=False)
            ov = overlap_coefficient(m, u)
            print(f"  Round {r}: overlap {ov:.4f}")
            if ov < best_overlap:
                best_overlap = ov
                best_round = r
        except FileNotFoundError:
            continue

    print(f"  Best round: {best_round} (overlap {best_overlap:.4f})")
    return best_round


def build_config(
    model_path: str,
    bm25_weight: float,
    output_dir: str,
    cache_dir: str,
    crossmap_path: str,
    config_path: str,
) -> None:
    """Write a melder config with the given BM25 weight."""
    emb_scale = 1.0 - bm25_weight
    name_emb = round(0.60 * emb_scale, 4)
    addr_emb = round(0.40 * emb_scale, 4)

    cfg = {
        "job": {
            "name": f"experiment_10_bm25_{int(bm25_weight * 100)}pct",
            "description": f"BM25 composite — {int(bm25_weight * 100)}% BM25 with Arctic-embed-xs",
        },
        "datasets": {
            "a": {"path": SRC_DATASET_A, "id_field": "entity_id", "format": "csv"},
            "b": {
                "path": SRC_DATASET_B,
                "id_field": "counterparty_id",
                "format": "csv",
            },
        },
        "cross_map": {
            "backend": "local",
            "path": crossmap_path,
            "a_id_field": "entity_id",
            "b_id_field": "counterparty_id",
        },
        "embeddings": {
            "model": model_path,
            "a_cache_dir": os.path.join(cache_dir, "a"),
            "b_cache_dir": os.path.join(cache_dir, "b"),
        },
        "blocking": {
            "enabled": True,
            "operator": "and",
            "fields": [{"field_a": "country_code", "field_b": "domicile"}],
        },
        "match_fields": [
            {
                "field_a": "legal_name",
                "field_b": "counterparty_name",
                "method": "embedding",
                "weight": name_emb,
            },
            {
                "field_a": "registered_address",
                "field_b": "counterparty_address",
                "method": "embedding",
                "weight": addr_emb,
            },
        ],
        "thresholds": {"auto_match": 0.88, "review_floor": 0.60},
        "output": {
            "csv_dir_path": output_dir + "/",
        },
        "performance": {"encoder_pool_size": 4, "vector_index_mode": "load"},
        "vector_backend": "usearch",
        "top_n": 5,
        "ann_candidates": 50,
    }

    # Add BM25 fields and weight only if bm25_weight > 0
    if bm25_weight > 0:
        cfg["bm25_fields"] = [
            {"field_a": "legal_name", "field_b": "counterparty_name"},
            {"field_a": "registered_address", "field_b": "counterparty_address"},
        ]
        cfg["match_fields"].append({"method": "bm25", "weight": bm25_weight})

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_meld(config_path: str) -> None:
    result = subprocess.run([MELD_BINARY, "run", "--config", config_path, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")


def evaluate(bm25_pct: int, output_dir: str) -> dict:
    from evaluate import evaluate_round

    return evaluate_round(
        dataset_a=SRC_DATASET_A,
        dataset_b=SRC_DATASET_B,
        results=os.path.join(output_dir, "results.csv"),
        review=os.path.join(output_dir, "review.csv"),
        print_summary=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--best-round",
        type=int,
        default=None,
        help="Override: use this round from experiment 9 (default: auto-detect best)",
    )
    args = p.parse_args()

    if not os.path.exists(MELD_BINARY):
        print(f"Error: meld binary not found at '{MELD_BINARY}'")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT 10 — BM25 composite with Arctic-embed-xs")
    print(f"{'=' * 64}\n")

    # Find best round from experiment 9
    if args.best_round is not None:
        best_round = args.best_round
        print(f"Using specified round: {best_round}")
    else:
        print("Scanning experiment 9 for best overlap round...")
        best_round = find_best_round("experiment_9")

    # Locate the model
    model_dir = os.path.join(
        SCIENCE_DIR, "results", "experiment_9", "models", f"round_{best_round}"
    )
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"Error: model not found at '{model_path}'")
        print("Run experiment 9 first.")
        sys.exit(1)

    print(f"Model: {model_path}")

    # Sweep BM25 weights
    for bm25_weight in BM25_WEIGHTS:
        bm25_pct = int(bm25_weight * 100)
        run_dir = os.path.join(EXPERIMENT_DIR, f"bm25_{bm25_pct}pct")
        output_dir = os.path.join(run_dir, "output")
        cache_dir = os.path.join(run_dir, "cache")
        crossmap_path = os.path.join(run_dir, "crossmap.csv")
        config_path = os.path.join(run_dir, "config.yaml")

        print(f"\n{'=' * 64}")
        print(f"  BM25 weight: {bm25_pct}%")
        print(f"{'=' * 64}")

        # Clean previous output
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(crossmap_path):
            os.remove(crossmap_path)

        # Build config and run
        build_config(
            model_path=model_path,
            bm25_weight=bm25_weight,
            output_dir=output_dir,
            cache_dir=cache_dir,
            crossmap_path=crossmap_path,
            config_path=config_path,
        )

        run_meld(config_path)

        # Evaluate
        metrics = evaluate(bm25_pct, output_dir)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(
                {"bm25_pct": bm25_pct, "round": best_round, **metrics}, f, indent=2
            )

        print(
            f"  [{bm25_pct}% BM25] "
            f"prec={metrics['precision']:.1%}  "
            f"recall={metrics['recall_ceiling']:.1%}  "
            f"combined={metrics['combined_recall']:.1%}  "
            f"review_fp={metrics['review_fp_unmatched']}"
        )

    print(f"\n{'=' * 64}")
    print(f"  COMPLETE — experiment 10")
    print(f"  Results: {EXPERIMENT_DIR}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
