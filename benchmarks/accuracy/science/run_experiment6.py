#!/usr/bin/env python3
"""Experiment 6 — BM25 composite scoring.

Self-contained experiment that tests adding BM25 lexical matching
alongside the fine-tuned embedding model from Experiment 5 R17.

Copies required assets (model weights, datasets) into
results/experiment_6/ so the experiment directory is self-contained.

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment6.py
"""

import os
import shutil
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Paths (all relative to project root; run from there)
# ---------------------------------------------------------------------------

SCIENCE_DIR = "benchmarks/accuracy/science"
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_6")
MELD_BINARY = "./target/release/meld"

# Source assets (from earlier experiments / shared datasets)
SRC_MODEL_DIR = os.path.join(
    SCIENCE_DIR, "results", "experiment_5", "models", "round_17"
)
SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

# Local copies inside experiment_6/
MODEL_DIR = os.path.join(EXPERIMENT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
DATASET_A = os.path.join(EXPERIMENT_DIR, "dataset_a.csv")
DATASET_B = os.path.join(EXPERIMENT_DIR, "dataset_b.csv")

CACHE_DIR = os.path.join(EXPERIMENT_DIR, "cache")
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "output")
CONFIG_PATH = os.path.join(EXPERIMENT_DIR, "config.yaml")
CROSSMAP_PATH = os.path.join(EXPERIMENT_DIR, "crossmap.csv")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def copy_assets():
    """Copy model weights and datasets into experiment_6/ if not present."""
    # Model directory (entire dir — tokenizer, config, etc.)
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(SRC_MODEL_DIR):
            print(f"Error: experiment 5 model not found at '{SRC_MODEL_DIR}'")
            print(
                "Experiment 6 requires the fine-tuned model from experiment 5 round 17."
            )
            print("Run experiment 5 first:")
            print("  python benchmarks/accuracy/science/run.py \\")
            print("    --name experiment_5 --rounds 18 --loss mnrl \\")
            print("    --base-model BAAI/bge-base-en-v1.5 --batch-size 128 --epochs 1")
            sys.exit(1)
        print(f"Copying model from {SRC_MODEL_DIR}...")
        shutil.copytree(SRC_MODEL_DIR, MODEL_DIR, dirs_exist_ok=True)
    else:
        print("Model already present.")

    # Datasets
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


def build_config():
    """Write the melder config with BM25 at 10% weight."""
    # BM25 takes 10% — scale embedding weights down proportionally
    bm25_weight = 0.10
    emb_scale = 1.0 - bm25_weight
    name_emb = round(0.60 * emb_scale, 4)
    addr_emb = round(0.40 * emb_scale, 4)

    cfg = {
        "job": {
            "name": "experiment_6",
            "description": "BM25 composite scoring — 10% BM25 share",
        },
        "datasets": {
            "a": {"path": DATASET_A, "id_field": "entity_id", "format": "csv"},
            "b": {"path": DATASET_B, "id_field": "counterparty_id", "format": "csv"},
        },
        "cross_map": {
            "backend": "local",
            "path": CROSSMAP_PATH,
            "a_id_field": "entity_id",
            "b_id_field": "counterparty_id",
        },
        "embeddings": {
            "model": MODEL_PATH,
            "a_cache_dir": os.path.join(CACHE_DIR, "a"),
            "b_cache_dir": os.path.join(CACHE_DIR, "b"),
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
                "weight": name_emb,
            },
            {
                "field_a": "registered_address",
                "field_b": "counterparty_address",
                "method": "embedding",
                "weight": addr_emb,
            },
            {"method": "bm25", "weight": bm25_weight},
        ],
        "thresholds": {"auto_match": 0.88, "review_floor": 0.60},
        "output": {
            "csv_dir_path": OUTPUT_DIR + "/",
        },
        "performance": {"encoder_pool_size": 4, "vector_index_mode": "load"},
        "vector_backend": "usearch",
        "top_n": 5,
        "ann_candidates": 50,
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Config written to {CONFIG_PATH}")
    print(f"  name_emb={name_emb}  addr_emb={addr_emb}  bm25={bm25_weight}")


def run_meld():
    """Run meld batch scoring."""
    result = subprocess.run([MELD_BINARY, "run", "--config", CONFIG_PATH, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not os.path.exists(MELD_BINARY):
        print(f"Error: meld binary not found at '{MELD_BINARY}'")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT 6 — BM25 composite scoring")
    print(f"{'=' * 64}\n")

    # Step 1: Copy assets
    copy_assets()

    # Step 2: Build config
    build_config()

    # Step 3: Clean previous output
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(CROSSMAP_PATH):
        os.remove(CROSSMAP_PATH)

    # Step 4: Run meld
    print(f"\nRunning meld...")
    run_meld()

    print(f"\n{'=' * 64}")
    print(f"  COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
