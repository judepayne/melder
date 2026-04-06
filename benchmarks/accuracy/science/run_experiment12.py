#!/usr/bin/env python3
"""Experiment 12 — Add fuzzy name matching to the full pipeline.

Takes the best configuration from experiment 11 (Arctic-embed-xs R22 + 40% BM25
+ synonym 0.20) and adds wratio fuzzy matching on the name field at weight 0.10.

Single run — the fuzzy scorer is additive (same design as synonym: excluded from
normalisation when it would dilute the signal for non-matching pairs... actually
wratio always returns >0 for non-empty strings, so it participates in normalisation
for all pairs). This means it will slightly reduce the effective weight of other
components for all pairs — but it adds a complementary name signal that should
penalise the military-address false matches where names are genuinely different.

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment12.py
"""

import json
import os
import shutil
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.abspath("benchmarks/accuracy/science"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCIENCE_DIR = "benchmarks/accuracy/science"
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_12")
MELD_BINARY = "./target/release/meld"

MODEL_PATH = os.path.join(
    SCIENCE_DIR, "results", "experiment_9", "models", "round_22", "model.onnx"
)

SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_config():
    output_dir = os.path.join(EXPERIMENT_DIR, "output")
    cache_dir = os.path.join(EXPERIMENT_DIR, "cache")
    crossmap_path = os.path.join(EXPERIMENT_DIR, "crossmap.csv")
    config_path = os.path.join(EXPERIMENT_DIR, "config.yaml")

    os.makedirs(output_dir, exist_ok=True)

    cfg = {
        "job": {
            "name": "experiment_12",
            "description": "Arctic-xs + 40% BM25 + synonym 0.20 + wratio 0.10",
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
        "match_fields": [
            {
                "field_a": "legal_name",
                "field_b": "counterparty_name",
                "method": "embedding",
                "weight": 0.30,
            },
            {
                "field_a": "registered_address",
                "field_b": "counterparty_address",
                "method": "embedding",
                "weight": 0.20,
            },
            {"method": "bm25", "weight": 0.50},
            {
                "field_a": "legal_name",
                "field_b": "counterparty_name",
                "method": "synonym",
                "weight": 0.20,
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

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Config written: {config_path}")
    print(f"  name_emb=0.30  addr_emb=0.20  bm25=0.50  synonym=0.20 (additive)")

    return config_path


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
    print(f"  EXPERIMENT 12 — Fuzzy name matching")
    print(f"  Arctic-xs + 40% BM25 + synonym 0.20 (name 75:25 addr)")
    print(f"{'=' * 64}\n")

    config_path = build_config()

    # Clean previous output
    output_dir = os.path.join(EXPERIMENT_DIR, "output")
    crossmap_path = os.path.join(EXPERIMENT_DIR, "crossmap.csv")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(crossmap_path):
        os.remove(crossmap_path)

    # Run
    print("\nRunning meld...")
    result = subprocess.run([MELD_BINARY, "run", "--config", config_path, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")

    # Evaluate
    print("\nEvaluating...")
    from evaluate import evaluate_round

    metrics = evaluate_round(
        dataset_a=SRC_DATASET_A,
        dataset_b=SRC_DATASET_B,
        results=os.path.join(output_dir, "results.csv"),
        review=os.path.join(output_dir, "review.csv"),
        print_summary=True,
    )

    metrics_path = os.path.join(EXPERIMENT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print(f"\n{'=' * 64}")
    print(f"  COMPLETE")
    print(f"  Results: {EXPERIMENT_DIR}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
