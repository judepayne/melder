#!/usr/bin/env python3
"""Experiment 11 — Synonym matching with Arctic-embed-xs + BM25.

Takes the best configuration from experiment 10 (Arctic-embed-xs R22 + 40% BM25)
and adds synonym/acronym matching. Runs two variants for comparison:

  1. Baseline: experiment 10's best (40% BM25, no synonym)
  2. With synonym: same + synonym at weight 0.20 (additive)

Synonym weight is additive — it does not reduce embedding or BM25 weights.
The synonym scorer is binary (1.0/0.0) and its weight is excluded from the
normalisation denominator when it scores 0.0.

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment11.py
"""

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
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_11")
MELD_BINARY = "./target/release/meld"

# Model from experiment 9 R22
MODEL_PATH = os.path.join(
    SCIENCE_DIR, "results", "experiment_9", "models", "round_22", "model.onnx"
)

SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

# Weights from experiment 10 best (40% BM25)
NAME_EMB_WEIGHT = 0.36
ADDR_EMB_WEIGHT = 0.24
BM25_WEIGHT = 0.40
SYNONYM_WEIGHT = 0.20


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_config(variant_dir, include_synonym):
    """Write a melder config. Returns (config_path, output_dir, crossmap_path)."""
    cache_dir = os.path.join(variant_dir, "cache")
    output_dir = os.path.join(variant_dir, "output")
    crossmap_path = os.path.join(variant_dir, "crossmap.csv")
    config_path = os.path.join(variant_dir, "config.yaml")

    os.makedirs(output_dir, exist_ok=True)

    match_fields = [
        {
            "field_a": "legal_name",
            "field_b": "counterparty_name",
            "method": "embedding",
            "weight": NAME_EMB_WEIGHT,
        },
        {
            "field_a": "registered_address",
            "field_b": "counterparty_address",
            "method": "embedding",
            "weight": ADDR_EMB_WEIGHT,
        },
        {"method": "bm25", "weight": BM25_WEIGHT},
    ]

    if include_synonym:
        match_fields.append(
            {
                "field_a": "legal_name",
                "field_b": "counterparty_name",
                "method": "synonym",
                "weight": SYNONYM_WEIGHT,
            }
        )

    label = "with synonym" if include_synonym else "baseline (no synonym)"

    cfg = {
        "job": {
            "name": "experiment_11",
            "description": f"Synonym + Arctic-xs + 40% BM25 — {label}",
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
            "model": MODEL_PATH,
            "a_cache_dir": os.path.join(cache_dir, "a"),
            "b_cache_dir": os.path.join(cache_dir, "b"),
        },
        "blocking": {
            "enabled": True,
            "operator": "and",
            "fields": [{"field_a": "country_code", "field_b": "domicile"}],
        },
        "bm25_fields": [
            {"field_a": "legal_name", "field_b": "counterparty_name"},
            {"field_a": "registered_address", "field_b": "counterparty_address"},
        ],
        "match_fields": match_fields,
        "thresholds": {"auto_match": 0.88, "review_floor": 0.60},
        "output": {
            "csv_dir_path": output_dir + "/",
        },
        "performance": {"encoder_pool_size": 4, "vector_index_mode": "load"},
        "vector_backend": "usearch",
        "top_n": 5,
        "ann_candidates": 50,
    }

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    weights_str = (
        f"name_emb={NAME_EMB_WEIGHT}  addr_emb={ADDR_EMB_WEIGHT}  bm25={BM25_WEIGHT}"
    )
    if include_synonym:
        weights_str += f"  synonym={SYNONYM_WEIGHT} (additive)"
    print(f"Config written: {config_path}")
    print(f"  {weights_str}")

    return config_path, output_dir, crossmap_path


def run_meld(config_path):
    result = subprocess.run([MELD_BINARY, "run", "--config", config_path, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")


def run_evaluate(variant_dir):
    from evaluate import evaluate_round

    return evaluate_round(
        dataset_a=SRC_DATASET_A,
        dataset_b=SRC_DATASET_B,
        results=os.path.join(variant_dir, "output", "results.csv"),
        review=os.path.join(variant_dir, "output", "review.csv"),
        print_summary=True,
    )


def run_variant(name, include_synonym):
    variant_dir = os.path.join(EXPERIMENT_DIR, name)
    os.makedirs(variant_dir, exist_ok=True)

    label = "WITH SYNONYM" if include_synonym else "BASELINE (no synonym)"
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"{'─' * 64}\n")

    config_path, output_dir, crossmap_path = build_config(variant_dir, include_synonym)

    # Clean previous output
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(crossmap_path):
        os.remove(crossmap_path)

    run_meld(config_path)

    print("\nEvaluating...")
    metrics = run_evaluate(variant_dir)

    metrics_path = os.path.join(variant_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(baseline, synonym):
    L = 32
    N = 10

    def row(label, b_val, s_val, indent=0):
        pad = "  " * indent
        delta = s_val - b_val
        sign = "+" if delta > 0 else ""
        delta_str = f"({sign}{delta:,})" if delta != 0 else ""
        print(
            f"  {pad}{label:<{L - len(pad)}} {b_val:>{N},} {s_val:>{N},}  {delta_str}"
        )

    def pct_row(label, b_val, s_val):
        delta = s_val - b_val
        sign = "+" if delta > 0 else ""
        delta_str = f"({sign}{delta:.1%})" if abs(delta) > 0.0001 else ""
        print(f"  {label:<{L}} {b_val:>{N}.1%} {s_val:>{N}.1%}  {delta_str}")

    print(f"\n{'=' * 72}")
    print(f"  EXPERIMENT 11 — COMPARISON")
    print(f"  {'Baseline':>{L + N}}  {'Synonym':>{N}}")
    print(f"  {'(40% BM25)':>{L + N}}  {'(40% BM25':>{N}}")
    print(f"  {'':>{L + N}}  {'+ synonym)':>{N}}")
    print(f"{'=' * 72}")
    print()

    row("Ceiling", baseline["blocking_ceiling"], synonym["blocking_ceiling"])
    print()
    row("Auto-matched", baseline["auto_matched"], synonym["auto_matched"])
    row("matched", baseline["auto_tp"], synonym["auto_tp"], indent=1)
    row(
        "ambiguous",
        baseline["auto_fp_ambiguous"],
        synonym["auto_fp_ambiguous"],
        indent=1,
    )
    row(
        "not a match",
        baseline["auto_fp_unmatched"] + baseline["auto_fp_wrong"],
        synonym["auto_fp_unmatched"] + synonym["auto_fp_wrong"],
        indent=1,
    )
    print()
    row("Review", baseline["review"], synonym["review"])
    row("matched", baseline["review_tp"], synonym["review_tp"], indent=1)
    row(
        "ambiguous",
        baseline.get("review_fp_ambiguous", 0),
        synonym.get("review_fp_ambiguous", 0),
        indent=1,
    )
    row(
        "not a match",
        baseline.get("review_fp_unmatched", 0) + baseline.get("review_fp_wrong", 0),
        synonym.get("review_fp_unmatched", 0) + synonym.get("review_fp_wrong", 0),
        indent=1,
    )
    print()
    row(
        "Unmatched",
        baseline["fn"] + baseline["fn_ambiguous"] + baseline["tn"],
        synonym["fn"] + synonym["fn_ambiguous"] + synonym["tn"],
    )
    row("missed (matched)", baseline["fn"], synonym["fn"], indent=1)
    row(
        "missed (ambiguous)",
        baseline["fn_ambiguous"],
        synonym["fn_ambiguous"],
        indent=1,
    )
    row("not a match", baseline["tn"], synonym["tn"], indent=1)
    print()
    pct_row("Precision", baseline["precision"], synonym["precision"])
    pct_row("Combined recall", baseline["combined_recall"], synonym["combined_recall"])
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not os.path.exists(MELD_BINARY):
        print(f"Error: meld binary not found at '{MELD_BINARY}'")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at '{MODEL_PATH}'")
        print("Run experiment 9 first.")
        sys.exit(1)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT 11 — Synonym matching with Arctic-xs + 40% BM25")
    print(f"{'=' * 64}")

    # Run baseline (40% BM25, no synonym)
    baseline = run_variant("baseline", include_synonym=False)

    # Run with synonym matching
    synonym = run_variant("with_synonym", include_synonym=True)

    # Print comparison
    print_comparison(baseline, synonym)

    print(f"\n{'=' * 64}")
    print(f"  COMPLETE")
    print(f"  Results: {EXPERIMENT_DIR}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
