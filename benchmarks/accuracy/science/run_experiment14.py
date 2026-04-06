#!/usr/bin/env python3
"""Experiment 14 — INT8 quantized Arctic-embed-xs R22.

Takes the production configuration from experiment 12 (Arctic-embed-xs R22 +
50% BM25 + synonym 0.20) and replaces the fp32 model with a dynamically
quantized INT8 version. Measures accuracy impact and speed difference.

The quantized model was produced by onnxruntime dynamic quantization:
    86 MB fp32 → 22 MB INT8 (3.9x compression).

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment14.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath("benchmarks/accuracy/science"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCIENCE_DIR = "benchmarks/accuracy/science"
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_14")
MELD_BINARY = "./target/release/meld"

# INT8 quantized model
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model", "model.onnx")

SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")


# ---------------------------------------------------------------------------
# Config — identical to experiment 12 except model path
# ---------------------------------------------------------------------------


def build_config():
    output_dir = os.path.join(EXPERIMENT_DIR, "output")
    cache_dir = os.path.join(EXPERIMENT_DIR, "cache")
    crossmap_path = os.path.join(EXPERIMENT_DIR, "crossmap.csv")
    config_path = os.path.join(EXPERIMENT_DIR, "config.yaml")

    os.makedirs(output_dir, exist_ok=True)

    cfg = {
        "job": {
            "name": "experiment_14",
            "description": "INT8 quantized Arctic-xs R22 + 50% BM25 + synonym 0.20",
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
                "weight": 0.35,
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
    print(f"  Model: {MODEL_PATH} (INT8 quantized)")
    print(f"  name_emb=0.30  addr_emb=0.20  bm25=0.50  synonym=0.20")

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
        print(f"Error: quantized model not found at '{MODEL_PATH}'")
        print("Run the quantization step first.")
        sys.exit(1)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT 14 — INT8 quantized model")
    print(f"  Arctic-xs R22 INT8 + 50% BM25 + synonym 0.20")
    print(f"{'=' * 64}\n")

    config_path = build_config()

    # Clean previous output
    output_dir = os.path.join(EXPERIMENT_DIR, "output")
    crossmap_path = os.path.join(EXPERIMENT_DIR, "crossmap.csv")
    cache_a = os.path.join(EXPERIMENT_DIR, "cache", "a")
    cache_b = os.path.join(EXPERIMENT_DIR, "cache", "b")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(cache_a, ignore_errors=True)
    os.makedirs(cache_a, exist_ok=True)
    shutil.rmtree(cache_b, ignore_errors=True)
    os.makedirs(cache_b, exist_ok=True)
    if os.path.exists(crossmap_path):
        os.remove(crossmap_path)

    # Run (cold — no cached embeddings)
    print("\nRunning meld (cold, INT8 model)...")
    t0 = time.time()
    result = subprocess.run([MELD_BINARY, "run", "--config", config_path, "--verbose"])
    wall_time = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")

    print(f"\nWall time: {wall_time:.1f}s")

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

    metrics["wall_time_secs"] = round(wall_time, 1)

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
