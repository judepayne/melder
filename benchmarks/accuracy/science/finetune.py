#!/usr/bin/env python3
"""Fine-tune an embedding model on scored training pairs.

Always fine-tunes from the base HuggingFace checkpoint each round.

Two loss functions (--loss flag):

  mnrl (default) — MultipleNegativesRankingLoss:
    Ranking objective: "anchor should be closer to positive than to any other
    batch item." Uses only positive pairs (label > 0); in-batch items serve as
    negatives automatically. Teaches relative ordering without constraining
    absolute scores. Pairs CSV columns: sentence1, sentence2, label.

  cosine — CosineSimilarityLoss:
    Absolute calibration: "this pair should have cosine similarity ≈ label."
    Uses all pairs including explicit 0.0 negatives. Directly calibrates
    absolute scores but can compress the entire score distribution.

Two training modes:
  - LoRA (default): only Q/V projections trainable (~1% of parameters).
    Weights are merged before saving for ONNX compatibility.
  - Full fine-tune (--full-finetune): all parameters trainable.
    No LoRA overhead, more capacity for domain adaptation.

Usage:
    python finetune.py \\
        --pairs  rounds/round_0/pairs.csv rounds/round_1/pairs.csv \\
        --output models/round_2 \\
        [--base-model BAAI/bge-small-en-v1.5] [--epochs 3] [--batch-size 32] \\
        [--learning-rate 2e-5] [--full-finetune] [--loss mnrl] \\
        [--lora-r 8] [--lora-alpha 16] [--lora-dropout 0.1]
"""

import argparse
import csv
import os
import random
from collections import Counter

import torch
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    CosineSimilarityLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def detect_device() -> str:
    """Return best available compute device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(paths: list[str]) -> list[dict]:
    """Load and deduplicate scored training pairs from one or more pairs.csv files."""
    pairs: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["sentence1"], row["sentence2"], row["label"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "sentence1": row["sentence1"],
                        "sentence2": row["sentence2"],
                        "label": float(row["label"]),
                    }
                )

    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def finetune(
    pairs_paths: list[str],
    output_dir: str,
    base_model: str = "all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    full_finetune: bool = False,
    loss_type: str = "mnrl",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    freeze_layers: int = 0,
    device: str | None = None,
) -> None:
    """Fine-tune base_model on accumulated pairs and save to output_dir.

    Parameters
    ----------
    pairs_paths:    paths to pairs.csv files (all rounds accumulated so far)
    output_dir:     directory to save the fine-tuned model (HuggingFace format)
    base_model:     HuggingFace model name to start from each round
    epochs:         number of training epochs
    batch_size:     training batch size
    learning_rate:  AdamW learning rate
    full_finetune:  if True, train all parameters (no LoRA)
    loss_type:      "mnrl" (ranking, default) or "cosine" (absolute calibration)
    lora_r:         LoRA rank (ignored when full_finetune=True)
    lora_alpha:     LoRA scaling factor (ignored when full_finetune=True)
    lora_dropout:   dropout on LoRA layers (ignored when full_finetune=True)
    freeze_layers:  freeze the bottom N transformer layers (0 = none frozen)
    device:         compute device; auto-detected if None
    """
    device = device or detect_device()
    print(f"  Device:         {device}")

    pairs = load_pairs(pairs_paths)
    if not pairs:
        raise ValueError("No training pairs found — check pairs_paths")
    print(
        f"  Pairs:          {len(pairs):,} (deduplicated across {len(pairs_paths)} round(s))"
    )

    # Label distribution diagnostics
    label_counts = Counter(p["label"] for p in pairs)
    dist_parts = [f"{label}={count:,}" for label, count in sorted(label_counts.items())]
    print(f"  Label dist:     {', '.join(dist_parts)}")

    # Build HuggingFace Dataset for the ST v3 trainer.
    # posonly: only label>0 pairs with CosineSimilarityLoss — pure "push up" signal.
    # MNRL: positive pairs (label > 0) with optional hard negatives (label == 0).
    # Cosine: all pairs with continuous labels.
    if loss_type == "posonly":
        pos_pairs = [p for p in pairs if p["label"] > 0]
        print(
            f"  Pos-only mode:  {len(pos_pairs):,} positive pairs "
            f"(dropped {len(pairs) - len(pos_pairs):,} negatives — pure 'push matches up' signal)"
        )
        dataset = HFDataset.from_dict(
            {
                "sentence1": [p["sentence1"] for p in pos_pairs],
                "sentence2": [p["sentence2"] for p in pos_pairs],
                "label": [p["label"] for p in pos_pairs],
            }
        )
    elif loss_type in ("mnrl", "cached_mnrl"):
        positives = [p for p in pairs if p["label"] > 0]
        negatives = [p for p in pairs if p["label"] == 0.0]

        # Build anchor → [negative texts] lookup
        neg_by_anchor: dict[str, list[str]] = {}
        for p in negatives:
            neg_by_anchor.setdefault(p["sentence1"], []).append(p["sentence2"])

        # Build triplets: anchor, positive, negative (where available)
        anchors, pos_texts, neg_texts = [], [], []
        n_with_hard_neg = 0
        for p in positives:
            anchor = p["sentence1"]
            hard_negs = neg_by_anchor.get(anchor, [])
            anchors.append(anchor)
            pos_texts.append(p["sentence2"])
            if hard_negs:
                neg_texts.append(random.choice(hard_negs))
                n_with_hard_neg += 1
            else:
                neg_texts.append("")

        # If we have hard negatives, include them; otherwise anchor/positive only
        has_any_neg = n_with_hard_neg > 0
        if has_any_neg:
            # Filter out rows with empty negatives — MNRL needs all or none
            triplets = [
                (a, po, ne)
                for a, po, ne in zip(anchors, pos_texts, neg_texts)
                if ne
            ]
            # Also include positives without hard negatives as anchor/positive only
            pairs_only = [
                (a, po)
                for a, po, ne in zip(anchors, pos_texts, neg_texts)
                if not ne
            ]
            print(
                f"  MNRL mode:      {len(triplets):,} triplets (with hard negatives) "
                f"+ {len(pairs_only):,} pairs (in-batch negatives only)"
            )
            # Combine into one dataset — triplets get the negative column,
            # pairs get an empty string (sentence-transformers ignores empty)
            all_anchors = [t[0] for t in triplets] + [p[0] for p in pairs_only]
            all_positives = [t[1] for t in triplets] + [p[1] for p in pairs_only]
            all_negatives = [t[2] for t in triplets] + ["" for _ in pairs_only]
            dataset = HFDataset.from_dict(
                {
                    "anchor": all_anchors,
                    "positive": all_positives,
                    "negative": all_negatives,
                }
            )
        else:
            print(
                f"  MNRL mode:      {len(positives):,} positive pairs "
                f"(no hard negatives available, using in-batch only)"
            )
            dataset = HFDataset.from_dict(
                {
                    "anchor": [p["sentence1"] for p in positives],
                    "positive": [p["sentence2"] for p in positives],
                }
            )
    else:
        dataset = HFDataset.from_dict(
            {
                "sentence1": [p["sentence1"] for p in pairs],
                "sentence2": [p["sentence2"] for p in pairs],
                "label": [p["label"] for p in pairs],
            }
        )

    model = SentenceTransformer(base_model, device=device)

    if freeze_layers > 0:
        # Freeze the bottom N transformer layers (embeddings + layers 0..N-1)
        auto_model = model[0].auto_model
        # Freeze token/position/type embeddings
        for param in auto_model.embeddings.parameters():
            param.requires_grad = False
        # Freeze encoder layers 0..freeze_layers-1
        for i in range(freeze_layers):
            for param in auto_model.encoder.layer[i].parameters():
                param.requires_grad = False
        n_total_layers = len(auto_model.encoder.layer)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = frozen + trainable
        print(
            f"  Freeze layers:  bottom {freeze_layers}/{n_total_layers} frozen, "
            f"top {n_total_layers - freeze_layers} trainable"
        )
        print(f"  Parameters:     {trainable:,} trainable / {total:,} total ({100 * trainable / total:.1f}%)")

    if full_finetune or freeze_layers > 0:
        if freeze_layers == 0:
            # All parameters trainable — no LoRA, no freezing
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Full fine-tune: {trainable:,} trainable / {total:,} total")
    else:
        # Apply LoRA to Q and V projections of the underlying transformer.
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
        )
        model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
        trainable = sum(
            p.numel() for p in model[0].auto_model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model[0].auto_model.parameters())
        print(
            f"  LoRA r={lora_r} α={lora_alpha}: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)"
        )

    print(
        f"  Epochs:         {epochs}  |  batch: {batch_size}  |  lr: {learning_rate:.2e}"
    )

    if loss_type == "cached_mnrl":
        loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
        print(f"  Loss:           CachedMultipleNegativesRankingLoss (mini_batch=32, effective_batch={batch_size})")
    elif loss_type == "mnrl":
        loss_fn = MultipleNegativesRankingLoss(model)
        print("  Loss:           MultipleNegativesRankingLoss")
    else:
        loss_fn = CosineSimilarityLoss(model)
        print(f"  Loss:           CosineSimilarityLoss ({loss_type} mode)")

    # fp16 only on CUDA — MPS does not support it
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        bf16=False,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss_fn,
    )

    trainer.train()

    if not full_finetune:
        # Merge LoRA weights back into the base weights before saving.
        merged = model[0].auto_model.merge_and_unload()
        model[0].auto_model = merged

    model.save_pretrained(output_dir)
    print(f"  Saved model → {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="One or more pairs.csv files (all rounds accumulated)",
    )
    p.add_argument(
        "--output", required=True, help="Output directory for the fine-tuned model"
    )
    p.add_argument("--base-model", default="all-MiniLM-L6-v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument(
        "--loss",
        choices=["mnrl", "cached_mnrl", "cosine", "posonly"],
        default="mnrl",
        help="Loss function: mnrl (ranking, default), cosine (absolute calibration), "
        "cached_mnrl (large-batch ranking), or posonly (push matches up only)",
    )
    p.add_argument(
        "--full-finetune",
        action="store_true",
        default=False,
        help="Train all parameters instead of using LoRA (default: False)",
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--freeze-layers", type=int, default=0,
        help="Freeze the bottom N transformer layers (0 = none frozen)",
    )
    p.add_argument(
        "--device", default=None, help="Compute device (auto-detected if omitted)"
    )
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    finetune(
        pairs_paths=args.pairs,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        full_finetune=args.full_finetune,
        loss_type=args.loss,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_layers=args.freeze_layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
