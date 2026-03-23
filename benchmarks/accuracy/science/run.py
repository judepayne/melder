#!/usr/bin/env python3
"""Scientific fine-tuning loop — fixed datasets, no regeneration.

Uses pre-generated datasets in science/rounds/round_N/dataset_b.csv and
science/holdout/dataset_b.csv. Every experiment sees identical data,
enabling controlled comparison of training approaches.

Run from the project root:
    python benchmarks/accuracy/science/run.py --rounds 5 --full-finetune --loss mnrl

Datasets must be pre-generated (see science/README or parent run.py).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

# ---------------------------------------------------------------------------
# Paths (all relative to project root; run from there)
# ---------------------------------------------------------------------------

SCIENCE_DIR = "benchmarks/accuracy/science"
BASE_CONFIG = os.path.join(SCIENCE_DIR, "config.yaml")

MASTER_A_PATH = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ---------------------------------------------------------------------------
# Bootstrap: add sibling modules to path
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.abspath(SCIENCE_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rounds", type=int, default=5, help="Number of rounds to run (default: 5)"
    )
    p.add_argument(
        "--base-model",
        default=BASE_MODEL_NAME,
        help="HuggingFace model name for fine-tuning base",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs per fine-tuning round (default: 3)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate (default: 2e-5)",
    )
    p.add_argument(
        "--lora-r", type=int, default=8, help="LoRA rank (default: 8)",
    )
    p.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA scaling factor (default: 16)",
    )
    p.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)",
    )
    p.add_argument(
        "--refeed-previous-generations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate pairs from all previous rounds (default: True)",
    )
    p.add_argument(
        "--full-finetune",
        action="store_true",
        default=False,
        help="Train all parameters instead of using LoRA (default: False)",
    )
    p.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Freeze the bottom N transformer layers (default: 0 = none frozen)",
    )
    p.add_argument(
        "--loss",
        choices=["mnrl", "cached_mnrl", "cosine", "posonly"],
        default="mnrl",
        help="Loss function: mnrl (ranking, default), cosine (absolute calibration), "
        "cached_mnrl (large-batch ranking), or posonly (push matches up only)",
    )
    p.add_argument(
        "--resume-from",
        type=int,
        default=0,
        metavar="N",
        help="Re-use existing round data for rounds < N",
    )
    p.add_argument(
        "--meld-binary", default="./target/release/meld", help="Path to the meld binary"
    )
    p.add_argument(
        "--name",
        required=True,
        help="Experiment name — results stored in results/{name}/",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_data_dir(idx: int) -> str:
    """Pre-generated data directory (read-only)."""
    return os.path.join(SCIENCE_DIR, "rounds", f"round_{idx}")


def round_work_dir(experiment: str, idx: int) -> str:
    """Per-round working directory for output/cache/config."""
    return os.path.join(SCIENCE_DIR, "results", experiment, "work", f"round_{idx}")


def holdout_work_dir(experiment: str, idx: int) -> str:
    """Per-round holdout working directory — preserved across rounds."""
    return os.path.join(SCIENCE_DIR, "results", experiment, "holdout", f"round_{idx}")


def model_dir_for_round(experiment: str, idx: int) -> str:
    return os.path.join(SCIENCE_DIR, "results", experiment, "models", f"round_{idx}")


def model_path_for_round(experiment: str, idx: int, base_model: str = BASE_MODEL_NAME) -> str:
    if idx == 0:
        return base_model
    return os.path.join(model_dir_for_round(experiment, idx), "model.onnx")


def results_csv_path(experiment: str) -> str:
    return os.path.join(SCIENCE_DIR, "results", experiment, "metrics.csv")


def learning_curve_path(experiment: str) -> str:
    return os.path.join(SCIENCE_DIR, "results", experiment, "learning_curve.png")


def patch_config(
    round_idx: int,
    dataset_b_path: str,
    cache_dir: str,
    output_dir: str,
    crossmap_path: str,
    model: str,
    out_config: str,
) -> None:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["job"]["name"] = f"science_round_{round_idx}"
    cfg["datasets"]["a"]["path"] = MASTER_A_PATH
    cfg["datasets"]["b"]["path"] = dataset_b_path
    cfg["embeddings"]["model"] = model
    cfg["embeddings"]["a_cache_dir"] = os.path.join(cache_dir, "a")
    cfg["embeddings"]["b_cache_dir"] = os.path.join(cache_dir, "b")
    cfg["output"]["results_path"] = os.path.join(output_dir, "results.csv")
    cfg["output"]["review_path"] = os.path.join(output_dir, "review.csv")
    cfg["output"]["unmatched_path"] = os.path.join(output_dir, "unmatched.csv")
    cfg["cross_map"]["path"] = crossmap_path

    with open(out_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_meld(config_path: str, binary: str) -> None:
    result = subprocess.run([binary, "run", "--config", config_path, "--verbose"])
    if result.returncode != 0:
        raise RuntimeError(f"meld run failed (exit {result.returncode})")


def run_and_evaluate(
    round_idx: int,
    work_dir: str,
    dataset_b_path: str,
    model: str,
    binary: str,
    label: str = "",
    print_summary: bool = True,
) -> dict:
    from evaluate import evaluate_round

    config_path = os.path.join(work_dir, "config.yaml")
    output_dir = os.path.join(work_dir, "output")
    cache_dir = os.path.join(work_dir, "cache")
    crossmap_path = os.path.join(work_dir, "crossmap.csv")

    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    if os.path.exists(crossmap_path):
        os.remove(crossmap_path)
    os.makedirs(output_dir, exist_ok=True)

    patch_config(
        round_idx=round_idx,
        dataset_b_path=dataset_b_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        crossmap_path=crossmap_path,
        model=model,
        out_config=config_path,
    )

    tag = f"[Round {round_idx}{(' ' + label) if label else ''}]"
    print(f"{tag} Running meld...", flush=True)
    t0 = time.time()
    run_meld(config_path, binary)
    print(f"{tag} Completed in {time.time() - t0:.1f}s", flush=True)

    print(f"{tag} Evaluating...", flush=True)
    return evaluate_round(
        dataset_a=MASTER_A_PATH,
        dataset_b=dataset_b_path,
        results=os.path.join(output_dir, "results.csv"),
        review=os.path.join(output_dir, "review.csv"),
        print_summary=print_summary,
    )


def append_metrics_csv(csv_path: str, row: dict) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Pre-flight checks
    if not os.path.exists(args.meld_binary):
        print(f"Error: meld binary not found at '{args.meld_binary}'")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)
    if not os.path.exists(BASE_CONFIG):
        print(f"Error: base config not found at '{BASE_CONFIG}'")
        sys.exit(1)
    if not os.path.exists(MASTER_A_PATH):
        print(f"Error: master dataset not found at '{MASTER_A_PATH}'")
        print("Generate datasets first (see science/README).")
        sys.exit(1)

    holdout_b_path = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")
    if not os.path.exists(holdout_b_path):
        print(f"Error: holdout dataset not found at '{holdout_b_path}'")
        sys.exit(1)

    experiment_name = args.name
    experiment_dir = os.path.join(SCIENCE_DIR, "results", experiment_name)
    metrics_csv = results_csv_path(experiment_name)
    curve_path = learning_curve_path(experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)

    # Clear results CSV for fresh experiment
    if args.resume_from == 0 and os.path.exists(metrics_csv):
        os.remove(metrics_csv)

    # Lazy imports
    from pairs import extract_pairs
    from finetune import finetune
    from export import export_onnx
    from plot import plot_learning_curve
    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Model: {args.base_model}  |  Loss: {args.loss}")
    print(f"  Rounds: {args.rounds}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"{'=' * 64}")

    pairs_paths: list[str] = []

    for round_idx in range(args.rounds):
        data_dir = round_data_dir(round_idx)
        work_dir = round_work_dir(experiment_name, round_idx)
        round_b_path = os.path.join(data_dir, "dataset_b.csv")
        metrics_path = os.path.join(work_dir, "metrics.json")
        pairs_path = os.path.join(work_dir, "pairs.csv")
        model = model_path_for_round(experiment_name, round_idx, args.base_model)

        if not os.path.exists(round_b_path):
            print(f"Error: round {round_idx} dataset not found at '{round_b_path}'")
            print(f"Only {round_idx} rounds of pre-generated data available.")
            break

        print(f"\n{'=' * 64}")
        model_label = (
            os.path.basename(model) if round_idx > 0 else f"base ({args.base_model})"
        )
        print(f"  ROUND {round_idx}  —  model: {model_label}")
        print(f"{'=' * 64}")

        # Resume check
        can_resume = (
            round_idx < args.resume_from
            and os.path.exists(metrics_path)
            and os.path.exists(pairs_path)
        )

        if can_resume:
            print(f"[Round {round_idx}] Resuming from cached data.")
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            os.makedirs(work_dir, exist_ok=True)

            # Run meld + evaluate (no data generation!)
            metrics = run_and_evaluate(
                round_idx=round_idx,
                work_dir=work_dir,
                dataset_b_path=round_b_path,
                model=model,
                binary=args.meld_binary,
                print_summary=True,
            )
            with open(metrics_path, "w") as f:
                json.dump({"round": round_idx, **metrics}, f, indent=2)

            # Extract training pairs
            print(f"[Round {round_idx}] Extracting training pairs...")
            config_path = os.path.join(work_dir, "config.yaml")
            n_pairs = extract_pairs(
                dataset_a=MASTER_A_PATH,
                dataset_b=round_b_path,
                results=os.path.join(work_dir, "output", "results.csv"),
                review=os.path.join(work_dir, "output", "review.csv"),
                config=config_path,
                out_path=pairs_path,
            )
            print(f"[Round {round_idx}] {n_pairs:,} training pairs extracted.")

        # Collect pairs
        if os.path.exists(pairs_path):
            if args.refeed_previous_generations:
                if pairs_path not in pairs_paths:
                    pairs_paths.append(pairs_path)
            else:
                pairs_paths = [pairs_path]

        # Holdout evaluation (per-round dir — preserved across rounds)
        h_work_dir = holdout_work_dir(experiment_name, round_idx)
        holdout_metrics_path = os.path.join(h_work_dir, "metrics.json")

        if can_resume and os.path.exists(holdout_metrics_path):
            print(f"[Round {round_idx}] Holdout: resuming from cached data.")
            with open(holdout_metrics_path) as f:
                holdout_metrics = json.load(f)
        else:
            print(f"[Round {round_idx}] Running holdout evaluation...")
            os.makedirs(h_work_dir, exist_ok=True)
            holdout_metrics = run_and_evaluate(
                round_idx=round_idx,
                work_dir=h_work_dir,
                dataset_b_path=holdout_b_path,
                model=model,
                binary=args.meld_binary,
                label="holdout",
                print_summary=False,
            )
            with open(holdout_metrics_path, "w") as f:
                json.dump(holdout_metrics, f, indent=2)

        # Append to results CSV
        row = {
            "round": round_idx,
            "model": model_label,
            **{f"train_{k}": v for k, v in metrics.items()},
            **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
        }
        if not can_resume:
            append_metrics_csv(metrics_csv, row)

        # Summary line
        print(
            f"\n[Round {round_idx}] "
            f"train   prec={metrics['precision']:.1%}  "
            f"recall={metrics['recall_ceiling']:.1%}  "
            f"fp={metrics['auto_fp']}"
        )
        print(
            f"[Round {round_idx}] "
            f"holdout prec={holdout_metrics['precision']:.1%}  "
            f"recall={holdout_metrics['recall_ceiling']:.1%}  "
            f"fp={holdout_metrics['auto_fp']}"
        )

        # Fine-tune for the next round (skip after last round)
        if round_idx < args.rounds - 1:
            next_model_dir = model_dir_for_round(experiment_name, round_idx + 1)
            next_onnx_path = os.path.join(next_model_dir, "model.onnx")

            if os.path.exists(next_onnx_path) and round_idx < args.resume_from:
                print(
                    f"[Round {round_idx}] Model for round {round_idx + 1} "
                    "already exists — skipping fine-tune."
                )
            else:
                os.makedirs(next_model_dir, exist_ok=True)
                total_pairs = sum(
                    max(0, sum(1 for _ in open(p)) - 1) for p in pairs_paths
                )
                epochs_this_round = max(
                    1,
                    args.epochs - (3 * round_idx // args.rounds),
                )
                print(
                    f"\n[Round {round_idx}] Fine-tuning for round {round_idx + 1} "
                    f"({len(pairs_paths)} round(s), ~{total_pairs:,} pairs, "
                    f"epochs={epochs_this_round})..."
                )
                # Fine-tune from previous round's model (not base)
                finetune_from = (
                    args.base_model if round_idx == 0
                    else model_dir_for_round(experiment_name, round_idx)
                )
                finetune(
                    pairs_paths=pairs_paths,
                    output_dir=next_model_dir,
                    base_model=finetune_from,
                    epochs=epochs_this_round,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    full_finetune=args.full_finetune,
                    loss_type=args.loss,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    freeze_layers=args.freeze_layers,
                )
                print(f"[Round {round_idx}] Exporting ONNX...")
                export_onnx(model_dir=next_model_dir)

        # Learning curve
        try:
            plot_learning_curve(metrics_csv, curve_path)
        except Exception as e:
            print(f"Warning: plot failed: {e}")

    # Final summary
    print(f"\n{'=' * 64}")
    print(f"  COMPLETE — {experiment_name} — {args.rounds} round(s)")
    print(f"  Results:       {metrics_csv}")
    print(f"  Holdout data:  {os.path.join(experiment_dir, 'holdout')}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
