#!/usr/bin/env python3
"""Experiment 7 — Synonym matching verification.

Tests the impact of adding synonym/acronym matching to the pipeline.
Uses the same fine-tuned model as experiment 6 (BGE-base LoRA R17) with
20% BM25 (the sweet spot from experiment 6) plus method: synonym on the
name field.

The goal is to verify that the 81 acronym cases identified in experiment 5's
overlap zone analysis are now recovered by the synonym candidate generator
and binary scorer.

Runs two configurations for comparison:
  1. Baseline: 20% BM25, no synonym (reproduces experiment 6's best config)
  2. With synonym: 20% BM25 + method: synonym on name field

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment7.py
"""

import json
import os
import shutil
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Paths (all relative to project root; run from there)
# ---------------------------------------------------------------------------

SCIENCE_DIR = "benchmarks/accuracy/science"
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_7")
MELD_BINARY = "./target/release/meld"

# Source assets
SRC_MODEL_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_6", "model")
SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

# Local copies inside experiment_7/
MODEL_DIR = os.path.join(EXPERIMENT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
DATASET_A = os.path.join(EXPERIMENT_DIR, "dataset_a.csv")
DATASET_B = os.path.join(EXPERIMENT_DIR, "dataset_b.csv")

EVALUATE_SCRIPT = os.path.join(SCIENCE_DIR, "evaluate.py")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def copy_assets():
    """Copy model weights and datasets into experiment_7/ if not present."""
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(SRC_MODEL_DIR):
            print(f"Error: experiment 6 model not found at '{SRC_MODEL_DIR}'")
            print("Experiment 7 requires the model from experiment 6.")
            print("Run experiment 6 first:")
            print("  python benchmarks/accuracy/science/run_experiment6.py")
            sys.exit(1)
        print(f"Copying model from {SRC_MODEL_DIR}...")
        shutil.copytree(SRC_MODEL_DIR, MODEL_DIR, dirs_exist_ok=True)
    else:
        print("Model already present.")

    if not os.path.exists(DATASET_A):
        print(f"Copying dataset A from {SRC_DATASET_A}...")
        shutil.copy2(SRC_DATASET_A, DATASET_A)
    else:
        print("Dataset A already present.")

    if not os.path.exists(DATASET_B):
        print(f"Copying dataset B from {SRC_DATASET_B}...")
        shutil.copy2(SRC_DATASET_B, DATASET_B)
    else:
        print("Dataset B already present.")


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_config(variant_dir, include_synonym):
    """Write a melder config. Returns the config path.

    Synonym weight is ADDITIVE — it does not reduce the embedding or BM25
    weights. The synonym scorer is binary (1.0/0.0) and its weight is excluded
    from the normalisation denominator when it scores 0.0. This means:
    - Non-acronym pairs: composite is identical to baseline (synonym contributes
      nothing and its weight is excluded from total_weight).
    - Acronym pairs: synonym adds 0.10 to the weighted sum and total_weight
      grows by 0.10, boosting the composite score.
    """
    bm25_weight = 0.20
    synonym_weight = 0.20 if include_synonym else 0.0
    # Keep embedding weights fixed — synonym is additive
    name_emb = 0.48  # 0.60 * 0.80 (20% given to BM25)
    addr_emb = 0.32  # 0.40 * 0.80

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
            "weight": name_emb,
        },
        {
            "field_a": "registered_address",
            "field_b": "counterparty_address",
            "method": "embedding",
            "weight": addr_emb,
        },
        {
            "method": "bm25",
            "weight": bm25_weight,
            "fields": [
                {"field_a": "legal_name", "field_b": "counterparty_name"},
                {"field_a": "registered_address", "field_b": "counterparty_address"},
            ],
        },
    ]

    if include_synonym:
        match_fields.append(
            {
                "field_a": "legal_name",
                "field_b": "counterparty_name",
                "method": "synonym",
                "weight": synonym_weight,
            }
        )

    label = "with synonym" if include_synonym else "baseline (no synonym)"

    cfg = {
        "job": {
            "name": "experiment_7",
            "description": f"Synonym matching verification — {label}",
        },
        "datasets": {
            "a": {"path": DATASET_A, "id_field": "entity_id", "format": "csv"},
            "b": {"path": DATASET_B, "id_field": "counterparty_id", "format": "csv"},
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
        "match_fields": match_fields,
        "thresholds": {"auto_match": 0.64, "review_floor": 0.52},
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

    weights_str = f"name_emb={name_emb}  addr_emb={addr_emb}  bm25={bm25_weight}"
    if include_synonym:
        weights_str += f"  synonym={synonym_weight}"
    print(f"Config written: {config_path}")
    print(f"  {weights_str}")

    return config_path, output_dir, crossmap_path


def run_meld(config_path):
    """Run meld batch scoring."""
    result = subprocess.run([MELD_BINARY, "run", "--config", config_path, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")


def run_evaluate(variant_dir):
    """Run evaluate.py on a variant's output and return metrics."""
    from evaluate import evaluate_round

    return evaluate_round(
        dataset_a=DATASET_A,
        dataset_b=DATASET_B,
        results=os.path.join(variant_dir, "output", "results.csv"),
        review=os.path.join(variant_dir, "output", "review.csv"),
        print_summary=True,
    )


def run_variant(name, include_synonym):
    """Run one variant end-to-end and return metrics."""
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

    # Run meld
    print("\nRunning meld...")
    run_meld(config_path)

    # Evaluate
    print("\nEvaluating...")
    metrics = run_evaluate(variant_dir)

    # Save metrics
    metrics_path = os.path.join(variant_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(baseline, synonym):
    """Print side-by-side comparison of baseline vs synonym results."""
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
    print(f"  EXPERIMENT 7 — COMPARISON")
    print(f"  {'Baseline':>{L + N}}  {'Synonym':>{N}}")
    print(f"  {'(20% BM25)':>{L + N}}  {'(20% BM25':>{N}}")
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

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT 7 — Synonym matching verification")
    print(f"{'=' * 64}")

    # Step 1: Copy assets
    copy_assets()

    # Step 2: Run baseline (20% BM25, no synonym)
    baseline = run_variant("baseline", include_synonym=False)

    # Step 3: Run with synonym matching
    synonym = run_variant("with_synonym", include_synonym=True)

    # Step 4: Print comparison
    print_comparison(baseline, synonym)

    print(f"\n{'=' * 64}")
    print(f"  COMPLETE")
    print(f"  Results: {EXPERIMENT_DIR}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
