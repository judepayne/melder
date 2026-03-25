#!/usr/bin/env python3
"""Generate the round training datasets from the committed master.

The master dataset (master/dataset_a.csv) and holdout (holdout/dataset_b.csv)
are committed to git. The per-round training B datasets are deterministically
generated from the master using fixed seeds.

Run from the project root:
    python benchmarks/accuracy/science/setup_datasets.py

Seeds:
    master/dataset_a.csv      — seed 0, 10,000 records
    holdout/dataset_b.csv     — seed 9999
    rounds/round_N/dataset_b.csv — seed 100 + N

All B datasets use the 1:1 generator (60% matched, 10% heavy noise,
30% unmatched). See benchmarks/data/generate.py for details.
"""

import os
import sys

SCIENCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCIENCE_DIR, "..", "..", ".."))
MASTER_A = os.path.join(SCIENCE_DIR, "master", "dataset_a.csv")
HOLDOUT_B = os.path.join(SCIENCE_DIR, "holdout", "dataset_b.csv")

# Number of rounds to generate (enough for the longest experiment: exp 9, 23 rounds)
NUM_ROUNDS = 23
SEED_OFFSET = 100
N_RECORDS = 10000


def main() -> None:
    # Pre-flight checks
    if not os.path.exists(MASTER_A):
        print(f"Error: master dataset not found at '{MASTER_A}'")
        print("This file should be committed to git. Check your clone.")
        sys.exit(1)

    if not os.path.exists(HOLDOUT_B):
        print(f"Error: holdout dataset not found at '{HOLDOUT_B}'")
        print("This file should be committed to git. Check your clone.")
        sys.exit(1)

    # Import generator
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmarks", "data"))
    from generate import generate_b_from_master_1to1

    # Generate round datasets
    generated = 0
    skipped = 0

    for round_idx in range(NUM_ROUNDS):
        out_dir = os.path.join(SCIENCE_DIR, "rounds", f"round_{round_idx}")
        out_path = os.path.join(out_dir, "dataset_b.csv")

        if os.path.exists(out_path):
            skipped += 1
            continue

        seed = SEED_OFFSET + round_idx
        generate_b_from_master_1to1(
            master_a_path=MASTER_A,
            b_seed=seed,
            n=N_RECORDS,
            include_addresses=True,
            out_dir=out_dir,
        )
        generated += 1

    print(f"\nDone: {generated} round datasets generated, {skipped} already existed.")
    print(f"Datasets in: {os.path.join(SCIENCE_DIR, 'rounds')}")


if __name__ == "__main__":
    main()
