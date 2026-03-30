#!/usr/bin/env python3
"""Experiment 13 — BM25-only vs 50/50 composite population separation.

Two runs against the holdout dataset:

  Run A: 50% embedding (30% name, 20% address) + 50% BM25
         Uses fine-tuned Arctic-embed-xs R22 from experiment 9.

  Run B: 100% BM25 only — no embeddings, no vector index.
         BM25 indexes both name and address fields.

Both runs are evaluated and their overlap coefficients / score distributions
compared to show how much the embedding adds on top of BM25.

Requires experiment 9 to be complete (Run A uses its best model).

Usage (from project root):
    python benchmarks/accuracy/science/run_experiment13.py
    python benchmarks/accuracy/science/run_experiment13.py --best-round 22
"""

import argparse
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
EXPERIMENT_DIR = os.path.join(SCIENCE_DIR, "results", "experiment_13")
MELD_BINARY = "./target/release/meld"

SRC_DATASET_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
SRC_DATASET_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

RUNS = {
    "composite_50_50": {
        "label": "50% embedding + 50% BM25 + synonym",
        "name_emb": 0.30,
        "addr_emb": 0.20,
        "bm25_weight": 0.50,
        "synonym_weight": 0.20,
        "needs_model": True,
    },
    "bm25_only": {
        "label": "100% BM25 only",
        "name_emb": 0.0,
        "addr_emb": 0.0,
        "bm25_weight": 1.00,
        "synonym_weight": 0.0,
        "needs_model": False,
    },
}


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
            if ov < best_overlap:
                best_overlap = ov
                best_round = r
        except FileNotFoundError:
            continue

    print(f"  Best round: {best_round} (overlap {best_overlap:.4f})")
    return best_round


def build_config(
    run_key: str,
    run_cfg: dict,
    model_path: str,
    output_dir: str,
    cache_dir: str,
    crossmap_path: str,
    config_path: str,
) -> None:
    """Write a melder config for the given run."""
    cfg = {
        "job": {
            "name": f"experiment_13_{run_key}",
            "description": f"Exp 13: {run_cfg['label']}",
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
        "blocking": {
            "enabled": True,
            "operator": "and",
            "fields": [{"field_a": "country_code", "field_b": "domicile"}],
        },
        "thresholds": {"auto_match": 0.88, "review_floor": 0.60},
        "output": {
            "results_path": os.path.join(output_dir, "results.csv"),
            "review_path": os.path.join(output_dir, "review.csv"),
            "unmatched_path": os.path.join(output_dir, "unmatched.csv"),
        },
        "performance": {"encoder_pool_size": 4},
        "top_n": 5,
    }

    match_fields = []

    # Embedding fields (only if this run uses embeddings).
    if run_cfg["needs_model"]:
        cfg["embeddings"] = {
            "model": model_path,
            "a_cache_dir": os.path.join(cache_dir, "a"),
            "b_cache_dir": os.path.join(cache_dir, "b"),
        }
        cfg["vector_backend"] = "usearch"
        cfg["ann_candidates"] = 50
        cfg["performance"]["vector_index_mode"] = "load"

        if run_cfg["name_emb"] > 0:
            match_fields.append(
                {
                    "field_a": "legal_name",
                    "field_b": "counterparty_name",
                    "method": "embedding",
                    "weight": run_cfg["name_emb"],
                }
            )
        if run_cfg["addr_emb"] > 0:
            match_fields.append(
                {
                    "field_a": "registered_address",
                    "field_b": "counterparty_address",
                    "method": "embedding",
                    "weight": run_cfg["addr_emb"],
                }
            )
    else:
        # BM25-only: still need an embeddings section for the config parser,
        # but no vector backend. Use a dummy model — melder won't encode
        # anything if there are no embedding match_fields.
        cfg["embeddings"] = {
            "model": "all-MiniLM-L6-v2",
            "a_cache_dir": os.path.join(cache_dir, "a"),
            "b_cache_dir": os.path.join(cache_dir, "b"),
        }
        cfg["bm25_candidates"] = 50

    # BM25 fields and weight.
    if run_cfg["bm25_weight"] > 0:
        cfg["bm25_fields"] = [
            {"field_a": "legal_name", "field_b": "counterparty_name"},
            {"field_a": "registered_address", "field_b": "counterparty_address"},
        ]
        match_fields.append({"method": "bm25", "weight": run_cfg["bm25_weight"]})

    # Synonym matching (additive weight, excluded from normalisation when 0).
    if run_cfg.get("synonym_weight", 0) > 0:
        cfg["synonym_fields"] = [
            {"field_a": "legal_name", "field_b": "counterparty_name"},
        ]
        match_fields.append(
            {
                "field_a": "legal_name",
                "field_b": "counterparty_name",
                "method": "synonym",
                "weight": run_cfg["synonym_weight"],
            }
        )

    cfg["match_fields"] = match_fields

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_meld(config_path: str) -> None:
    result = subprocess.run([MELD_BINARY, "run", "--config", config_path, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")


def evaluate_run(output_dir: str) -> dict:
    from evaluate import evaluate_round

    return evaluate_round(
        dataset_a=SRC_DATASET_A,
        dataset_b=SRC_DATASET_B,
        results=os.path.join(output_dir, "results.csv"),
        review=os.path.join(output_dir, "review.csv"),
        print_summary=True,
    )


def compute_overlap(run_key: str) -> float:
    """Compute overlap coefficient for a run's output."""
    from overlap import overlap_coefficient

    # Load scores from results.csv and review.csv, split by _match_type.
    import csv

    run_dir = os.path.join(EXPERIMENT_DIR, run_key)
    output_dir = os.path.join(run_dir, "output")

    # Load B dataset for ground truth labels.
    b_labels = {}
    with open(SRC_DATASET_B, newline="") as f:
        for row in csv.DictReader(f):
            b_labels[row["counterparty_id"]] = row["_match_type"]

    matched_scores = []
    unmatched_scores = []

    for fname in ["results.csv", "review.csv"]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                b_id = row.get("counterparty_id", row.get("b_id", ""))
                score = float(row.get("score", 0))
                label = b_labels.get(b_id, "unmatched")
                if label in ("matched", "ambiguous"):
                    matched_scores.append(score)
                else:
                    unmatched_scores.append(score)

    if matched_scores and unmatched_scores:
        return overlap_coefficient(matched_scores, unmatched_scores)
    return -1.0


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
        print("Build with: cargo build --release --features usearch,bm25")
        sys.exit(1)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT 13 — BM25-only vs 50/50 composite")
    print(f"{'=' * 64}\n")

    # Find best round from experiment 9.
    if args.best_round is not None:
        best_round = args.best_round
        print(f"Using specified round: {best_round}")
    else:
        print("Scanning experiment 9 for best overlap round...")
        best_round = find_best_round("experiment_9")

    # Locate the model (needed for the composite run).
    model_dir = os.path.join(
        SCIENCE_DIR, "results", "experiment_9", "models", f"round_{best_round}"
    )
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"Error: model not found at '{model_path}'")
        print("Run experiment 9 first.")
        sys.exit(1)

    print(f"Model: {model_path}\n")

    # Run both configurations.
    all_metrics = {}

    for run_key, run_cfg in RUNS.items():
        run_dir = os.path.join(EXPERIMENT_DIR, run_key)
        output_dir = os.path.join(run_dir, "output")
        cache_dir = os.path.join(run_dir, "cache")
        crossmap_path = os.path.join(run_dir, "crossmap.csv")
        config_path = os.path.join(run_dir, "config.yaml")

        print(f"{'=' * 64}")
        print(f"  Run: {run_cfg['label']}")
        print(f"{'=' * 64}")

        # Clean previous output.
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(crossmap_path):
            os.remove(crossmap_path)

        # Build config and run.
        build_config(
            run_key=run_key,
            run_cfg=run_cfg,
            model_path=model_path,
            output_dir=output_dir,
            cache_dir=cache_dir,
            crossmap_path=crossmap_path,
            config_path=config_path,
        )

        run_meld(config_path)

        # Evaluate.
        metrics = evaluate_run(output_dir)
        overlap = compute_overlap(run_key)
        metrics["overlap"] = overlap

        all_metrics[run_key] = metrics

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({"run": run_key, "round": best_round, **metrics}, f, indent=2)

        print(
            f"\n  [{run_cfg['label']}] "
            f"prec={metrics['precision']:.1%}  "
            f"recall={metrics['recall_ceiling']:.1%}  "
            f"combined={metrics['combined_recall']:.1%}  "
            f"review_fp={metrics.get('review_fp_unmatched', '?')}  "
            f"overlap={overlap:.4f}"
        )
        print()

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  COMPARISON")
    print(f"{'=' * 64}")
    print(f"  {'Metric':<30} {'50/50 Composite':>18} {'BM25 Only':>18}")
    print(f"  {'-' * 66}")

    for label, key in [
        ("Overlap", "overlap"),
        ("Precision", "precision"),
        ("Recall (vs ceiling)", "recall_ceiling"),
        ("Combined recall", "combined_recall"),
        ("Auto-matched", "auto_matched"),
        ("Review", "review_total"),
        ("Review FP (unmatched)", "review_fp_unmatched"),
        ("Missed (clean)", "missed_clean"),
        ("Missed (ambiguous)", "missed_ambiguous"),
    ]:
        v_comp = all_metrics.get("composite_50_50", {}).get(key, "?")
        v_bm25 = all_metrics.get("bm25_only", {}).get(key, "?")

        if isinstance(v_comp, float) and key in (
            "precision",
            "recall_ceiling",
            "combined_recall",
        ):
            v_comp_str = f"{v_comp:.1%}"
            v_bm25_str = f"{v_bm25:.1%}" if isinstance(v_bm25, float) else str(v_bm25)
        elif isinstance(v_comp, float) and key == "overlap":
            v_comp_str = f"{v_comp:.4f}"
            v_bm25_str = f"{v_bm25:.4f}" if isinstance(v_bm25, float) else str(v_bm25)
        else:
            v_comp_str = f"{v_comp:,}" if isinstance(v_comp, int) else str(v_comp)
            v_bm25_str = f"{v_bm25:,}" if isinstance(v_bm25, int) else str(v_bm25)

        print(f"  {label:<30} {v_comp_str:>18} {v_bm25_str:>18}")

    print(f"  {'-' * 66}")
    print(f"\n{'=' * 64}")
    print(f"  COMPLETE — experiment 13")
    print(f"  Results: {EXPERIMENT_DIR}")
    print(f"{'=' * 64}\n")

    # Print score distribution commands for follow-up analysis.
    print("Score distribution commands:")
    for run_key in RUNS:
        print(
            f"  python benchmarks/accuracy/science/score_chart.py "
            f"--results {EXPERIMENT_DIR}/{run_key}/output/results.csv "
            f"--review {EXPERIMENT_DIR}/{run_key}/output/review.csv "
            f"--dataset-b {SRC_DATASET_B} --max-score 0.8"
        )


if __name__ == "__main__":
    main()
