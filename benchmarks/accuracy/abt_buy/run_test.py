#!/usr/bin/env python3
"""Accuracy benchmark — Abt-Buy (Leipzig), real-world product matching.

The first accuracy benchmark on third-party data. Abt-Buy is the standard
small entity resolution benchmark from the Leipzig database group (Köpcke,
Thor & Rahm, 2010): 1,081 Abt.com products × 1,092 Buy.com products,
1,097 true pairs.

Phase 0 — Fetch data:
  - Download Abt-Buy.zip from the Leipzig benchmark site (skipped if the
    data is already present)
  - Convert Latin-1 → UTF-8 and write abt.csv, buy.csv, ground_truth.csv
    into data/ (gitignored — third-party data is not committed)

Phase 1 — Batch match:
  - meld run with config.yaml (A = Abt, B = Buy)

Phase 2 — Evaluate against the perfect mapping:
  - Precision / recall / F1 of auto-matched pairs
  - Combined recall (auto + review)
  - 1:1 ceiling: the ground truth contains some 1:N pairs, and the
    CrossMap bijection caps how many true pairs any melder run can find
  - Oracle F1: best single threshold over each B record's top candidate.
    Optimistic (the threshold is picked on the test set); use it only for
    rough comparison with published pairwise F1

Run from the project root:
    python3 benchmarks/accuracy/abt_buy/run_test.py
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from itertools import groupby

TEST_DIR = "benchmarks/accuracy/abt_buy"
DATA_DIR = f"{TEST_DIR}/data"
BINARY_DEFAULT = "./target/release/meld"
SOURCE_URL = "https://dbs.uni-leipzig.de/file/Abt-Buy.zip"
DATASET_A = f"{DATA_DIR}/abt.csv"
DATASET_B = f"{DATA_DIR}/buy.csv"
GROUND_TRUTH = f"{DATA_DIR}/ground_truth.csv"


# ---------------------------------------------------------------------------
# Phase 0 — data
# ---------------------------------------------------------------------------


def reencode(raw: bytes, dest: str, id_field: str) -> int:
    """Rewrite a Latin-1 CSV as UTF-8, renaming its `id` column.

    Both source files key on `id`. Distinct names keep the a/b ID columns
    in relationships.csv unambiguous. Returns the number of data rows.
    """
    rows = list(csv.reader(io.StringIO(raw.decode("latin-1"))))
    rows[0] = [id_field if c == "id" else c for c in rows[0]]
    with open(dest, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return len(rows) - 1


def fetch_data() -> None:
    """Download and convert Abt-Buy unless it is already present."""
    if all(os.path.exists(p) for p in (DATASET_A, DATASET_B, GROUND_TRUTH)):
        print(f"Phase 0: data present in {DATA_DIR}/")
        return

    print(f"Phase 0: downloading {SOURCE_URL}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))

    n_a = reencode(archive.read("Abt.csv"), DATASET_A, "abt_id")
    n_b = reencode(archive.read("Buy.csv"), DATASET_B, "buy_id")

    mapping = csv.DictReader(
        io.StringIO(archive.read("abt_buy_perfectMapping.csv").decode("latin-1"))
    )
    with open(GROUND_TRUTH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["abt_id", "buy_id"])
        n_gt = 0
        for row in mapping:
            writer.writerow([row["idAbt"], row["idBuy"]])
            n_gt += 1

    print(f"  abt.csv {n_a:,} rows, buy.csv {n_b:,} rows, {n_gt:,} true pairs")


# ---------------------------------------------------------------------------
# Phase 2 — evaluation
# ---------------------------------------------------------------------------


def load_ground_truth() -> set[tuple[str, str]]:
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        return {(r["abt_id"], r["buy_id"]) for r in csv.DictReader(f)}


def max_bipartite_matching(pairs: set[tuple[str, str]]) -> int:
    """Size of the largest 1:1 subset of pairs (augmenting paths)."""
    adj: dict[str, list[str]] = {}
    for a, b in pairs:
        adj.setdefault(b, []).append(a)
    owner: dict[str, str] = {}

    def augment(b: str, seen: set[str]) -> bool:
        for a in adj[b]:
            if a in seen:
                continue
            seen.add(a)
            if a not in owner or augment(owner[a], seen):
                owner[a] = b
                return True
        return False

    return sum(1 for b in adj if augment(b, set()))


def load_outputs() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return the final auto-match and review pairs."""
    auto: set[tuple[str, str]] = set()
    review: set[tuple[str, str]] = set()

    with open(f"{TEST_DIR}/output/relationships.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pair = (row["abt_id"], row["buy_id"])
            if row["relationship_type"] == "match":
                auto.add(pair)
            else:
                review.add(pair)

    return auto, review


def load_rank_one_candidates() -> dict[str, tuple[str, float]]:
    """Return each B record's genuine rank-one scored candidate."""
    path = f"{TEST_DIR}/output/accuracy_abt_buy.scoring_log.ndjson"
    top: dict[str, tuple[str, float]] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") != "scored" or event.get("query_side") != "b":
                continue

            rank_one = next(
                (
                    candidate
                    for candidate in event.get("candidates", [])
                    if candidate.get("rank") == 1
                ),
                None,
            )
            if rank_one is not None:
                top[event["query_id"]] = (
                    rank_one["matched_id"],
                    float(rank_one["score"]),
                )

    return top


def prf(predicted: set, gt: set) -> tuple[float, float, float]:
    tp = len(predicted & gt)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def oracle_f1(top: dict[str, tuple[str, float]], gt: set) -> tuple[float, float]:
    """Best F1 over any single threshold on top-1 candidates."""
    ranked = sorted(
        ((score, (a_id, b_id) in gt) for b_id, (a_id, score) in top.items()),
        key=lambda item: item[0],
        reverse=True,
    )
    best_f1, best_t = 0.0, 1.0
    true_positives = 0
    selected = 0

    for score, score_group in groupby(ranked, key=lambda item: item[0]):
        group = list(score_group)
        selected += len(group)
        true_positives += sum(correct for _, correct in group)
        precision = true_positives / selected
        recall = true_positives / len(gt)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if f1 > best_f1:
            best_f1, best_t = f1, score

    return best_f1, best_t


def evaluate() -> None:
    gt = load_ground_truth()
    ceiling = max_bipartite_matching(gt)
    auto, review = load_outputs()
    top = load_rank_one_candidates()

    auto_p, auto_r, auto_f1 = prf(auto, gt)
    combined = auto | review
    review_hits = len(review & gt)
    combined_hits = len(combined & gt)
    best_f1, best_t = oracle_f1(top, gt)

    L = 34  # label column width
    N = 9  # number column width

    def count(label: str, value: int, indent: int = 0) -> None:
        pad = "  " * indent
        print(f"  {pad}{label:<{L - len(pad)}} {value:>{N},}")

    def pct(label: str, value: float) -> None:
        print(f"  {label:<{L}} {value:>{N - 1}.1%}")

    print()
    print("=" * 60)
    print("  ACCURACY EVALUATION — Abt-Buy")
    print("=" * 60)
    print()
    count("True pairs (ground truth)", len(gt))
    count("Ceiling (1:1 max reachable)", ceiling)
    print()
    count("Auto-matched", len(auto))
    count("Match", len(auto & gt), indent=1)
    count("False positive", len(auto - gt), indent=1)
    print()
    count("Review", len(review))
    count("Match", review_hits, indent=1)
    count("False positive", len(review) - review_hits, indent=1)
    print()
    count("Missed (not in auto or review)", len(gt) - combined_hits)
    print()
    pct("Precision (auto)", auto_p)
    pct("Recall (auto)", auto_r)
    pct("F1 (auto)", auto_f1)
    pct("Recall vs ceiling (auto)", len(auto & gt) / ceiling)
    pct("Combined recall (auto + review)", combined_hits / len(gt))
    print()
    pct("Oracle F1 (top-1, best threshold)", best_f1)
    print(f"  {'  at threshold':<{L}} {best_t:>{N}.3f}")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=BINARY_DEFAULT, help="Path to the meld binary"
    )
    parser.add_argument(
        "--cold", action="store_true", help="Force cold run (delete cache)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release")
        sys.exit(1)

    fetch_data()

    # Clean output and crossmap (always fresh for accuracy measurement)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    if args.cold:
        shutil.rmtree(f"{TEST_DIR}/cache", ignore_errors=True)

    warm = "warm" if os.path.exists(f"{TEST_DIR}/cache") else "cold"
    print(f"\n=== Accuracy: Abt-Buy ({warm}) — {TEST_DIR} ===\n", flush=True)

    start = time.time()
    result = subprocess.run(
        [args.binary, "run", "--config", f"{TEST_DIR}/config.yaml", "--verbose"]
    )
    elapsed = time.time() - start
    print(f"\nBatch completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print("Batch run failed!")
        sys.exit(result.returncode)

    evaluate()


if __name__ == "__main__":
    main()
