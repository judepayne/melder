#!/usr/bin/env python3
"""Extract scored training pairs from a round's output.

Each pair teaches the model an absolute similarity target:

  sentence1 — A-side entity text (concatenated embedding fields)
  sentence2 — B-side entity text
  label     — float similarity target:
                1.0  true match (matched B record ↔ its true A)
                0.87  ambiguous / noisy match (same entity, heavy noise)
                0.0  non-match (false positive or in-bucket negative)

Three emission loops:

  1. **Matched B records** (_match_type == "matched"):
     - (a_text, b_text, 1.0)  — true positive
     - (a_text, neg_text, 0.0) — hard negative (meld FP or bucket sample)
     Positive pair emitted even if no negative found.

  2. **Ambiguous B records** (_match_type == "ambiguous"):
     - (true_a_text, ambiguous_b_text, 0.7)  — soft positive
     These are the same entity under heavy noise; 0.7 teaches the model
     to score them below auto_match (0.88) but well above non-matches.

  3. **Review-band hard negatives** (from review.csv):
     - (a_text, wrong_b_text, 0.0)  — pairs that scored 0.80–0.88 but
       are wrong matches.  Only emitted where the match is incorrect.

Pairs are deduplicated before writing (loops 1 and 3 may produce the
same pair).

Text construction mirrors melder's encoding: the embedding field values
(field_a for A records, field_b for B records) are read from config.yaml
and joined with a single space, skipping empty fields.

Usage (standalone):
    python pairs.py \\
        --round-dir benchmarks/accuracy/science/rounds/round_0 \\
        --config    benchmarks/accuracy/science/work/round_0/config.yaml \\
        --out       benchmarks/accuracy/science/work/round_0/pairs.csv
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_embedding_field_pairs(config_path: str) -> list[tuple[str, str]]:
    """Return [(field_a, field_b), ...] for all match_fields with method=embedding."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return [
        (mf["field_a"], mf["field_b"])
        for mf in cfg.get("match_fields", [])
        if mf.get("method") == "embedding"
    ]


def load_blocking_fields(config_path: str) -> tuple[str | None, str | None]:
    """Return (field_a, field_b) for the first blocking field pair, or (None, None)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    blocking = cfg.get("blocking", {})
    if not blocking.get("enabled", False):
        return None, None
    fields = blocking.get("fields", [])
    if fields:
        return fields[0]["field_a"], fields[0]["field_b"]
    return None, None


def make_text(record: dict, fields: list[str]) -> str:
    """Concatenate field values, skipping empty/missing ones."""
    parts = [record.get(f, "").strip() for f in fields]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_pairs(
    dataset_a: str,
    dataset_b: str,
    results: str,
    review: str,
    config: str,
    out_path: str,
) -> int:
    """Extract scored training pairs and write to out_path. Returns pair count."""
    emb_pairs = load_embedding_field_pairs(config)
    a_fields = [fa for fa, _ in emb_pairs]
    b_fields = [fb for _, fb in emb_pairs]
    block_a, block_b = load_blocking_fields(config)

    # --- Load datasets ---
    a_records: dict[str, dict] = {}
    with open(dataset_a) as f:
        for row in csv.DictReader(f):
            a_records[row["entity_id"]] = row

    b_records: dict[str, dict] = {}
    with open(dataset_b) as f:
        for row in csv.DictReader(f):
            b_records[row["counterparty_id"]] = row

    # --- Build in-bucket pool: bucket_value → [b_ids with no true A match] ---
    bucket_pool: dict[str, list[str]] = defaultdict(list)
    if block_b:
        for b_id, b_rec in b_records.items():
            if b_rec.get("_match_type") != "matched":
                val = b_rec.get(block_b, "")
                if val:
                    bucket_pool[val].append(b_id)

    # --- Collect meld false positives: a_id → {b_ids meld incorrectly paired} ---
    a_fps: dict[str, set[str]] = defaultdict(set)
    for path in [results, review]:
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    a_id, b_id = row["a_id"], row["b_id"]
                    b_rec = b_records.get(b_id, {})
                    mt = b_rec.get("_match_type", "unmatched")
                    true_a = b_rec.get("_true_a_id", "")
                    if mt != "matched" or a_id != true_a:
                        a_fps[a_id].add(b_id)
        except FileNotFoundError:
            pass

    # --- Deduplicated pair collector ---
    seen: set[tuple[str, str, float]] = set()
    pairs: list[dict] = []

    def emit(s1: str, s2: str, label: float) -> None:
        key = (s1, s2, label)
        if key not in seen:
            seen.add(key)
            pairs.append({"sentence1": s1, "sentence2": s2, "label": label})

    # --- Loop 1: Matched B records → true positives + hard negatives ---
    for b_id, b_rec in b_records.items():
        if b_rec.get("_match_type") != "matched":
            continue

        true_a = b_rec.get("_true_a_id", "")
        a_rec = a_records.get(true_a)
        if not a_rec:
            continue

        a_text = make_text(a_rec, a_fields)
        b_text = make_text(b_rec, b_fields)
        if not a_text or not b_text:
            continue

        # Always emit the positive pair
        emit(a_text, b_text, 1.0)

        # Emit all proven meld FPs as hard negatives (not just one).
        # Fall back to a single bucket sample only if no FPs exist.
        fps = [bid for bid in a_fps.get(true_a, set()) if bid != b_id]
        if fps:
            for neg_b_id in fps:
                neg_text = make_text(b_records[neg_b_id], b_fields)
                if neg_text:
                    emit(a_text, neg_text, 0.0)
        elif block_a:
            bucket_val = a_rec.get(block_a, "")
            candidates = [bid for bid in bucket_pool.get(bucket_val, []) if bid != b_id]
            if candidates:
                neg_b_id = random.choice(candidates)
                neg_text = make_text(b_records[neg_b_id], b_fields)
                if neg_text:
                    emit(a_text, neg_text, 0.0)

    # --- Loop 2: Ambiguous B records → soft positives (0.7) ---
    for b_id, b_rec in b_records.items():
        if b_rec.get("_match_type") != "ambiguous":
            continue

        true_a = b_rec.get("_true_a_id", "")
        a_rec = a_records.get(true_a)
        if not a_rec:
            continue

        a_text = make_text(a_rec, a_fields)
        b_text = make_text(b_rec, b_fields)
        if not a_text or not b_text:
            continue

        emit(a_text, b_text, 0.87)

    # --- Loop 3: Review-band hard negatives (incorrect matches in review) ---
    try:
        with open(review) as f:
            for row in csv.DictReader(f):
                a_id, b_id = row["a_id"], row["b_id"]
                b_rec = b_records.get(b_id, {})
                mt = b_rec.get("_match_type", "unmatched")
                true_a = b_rec.get("_true_a_id", "")

                # Only emit incorrect matches (skip correct ones in review band)
                if mt == "matched" and a_id == true_a:
                    continue

                a_rec = a_records.get(a_id)
                if not a_rec:
                    continue

                a_text = make_text(a_rec, a_fields)
                b_text = make_text(b_rec, b_fields)
                if not a_text or not b_text:
                    continue

                emit(a_text, b_text, 0.0)
    except FileNotFoundError:
        pass

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence1", "sentence2", "label"])
        writer.writeheader()
        writer.writerows(pairs)

    return len(pairs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = args.round_dir
    n = extract_pairs(
        dataset_a=os.path.join(d, "dataset_a.csv"),
        dataset_b=os.path.join(d, "dataset_b.csv"),
        results=os.path.join(d, "output", "results.csv"),
        review=os.path.join(d, "output", "review.csv"),
        config=args.config,
        out_path=args.out,
    )
    print(f"Extracted {n:,} training pairs → {args.out}")


if __name__ == "__main__":
    main()
