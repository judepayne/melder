#!/usr/bin/env python3
"""Generate fixed datasets for the live accuracy regression test.

Creates:
  data/dataset_a_10k.csv  -- 10,000 A-side reference records
  data/dataset_b_10k.csv  -- 10,000 B-side records (7,000 matched, 1,000 ambiguous, 2,000 unmatched)
  data/inject_events.csv  -- 3,000 B-side records to inject via API (2,100 matched, 300 ambiguous, 600 unmatched)
  data/ground_truth.csv   -- all known A<->B pairs (from both initial load and injection)

Uses seed=42 for full reproducibility. Datasets use asymmetric column names:
  A-side: entity_id, legal_name, short_name, country_code, lei, ...
  B-side: counterparty_id, counterparty_name, domicile, lei_code, ...

Run from project root:
    python3 benchmarks/accuracy/live_10kx10k_inject3k/generate_data.py
"""

import csv
import os
import random
import sys

TEST_DIR = "benchmarks/accuracy/live_10kx10k_inject3k"
DATA_DIR = f"{TEST_DIR}/data"
SEED = 42
N_RECORDS = 10_000
N_INJECT = 3_000


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Import the shared generator
    sys.path.insert(0, "benchmarks/data")
    import generate

    # ---------------------------------------------------------------
    # Base 10k x 10k datasets
    # ---------------------------------------------------------------
    records_a, records_b = generate.generate_with_seed(
        seed=SEED,
        n=N_RECORDS,
        include_addresses=True,
        out_dir=DATA_DIR,
        n_exact=0,
    )

    # generate_with_seed writes dataset_a.csv / dataset_b.csv — rename to _10k
    for name in ["dataset_a", "dataset_b"]:
        src = os.path.join(DATA_DIR, f"{name}.csv")
        dst = os.path.join(DATA_DIR, f"{name}_10k.csv")
        if os.path.exists(src):
            os.rename(src, dst)

    matched = [r for r in records_b if r.get("_match_type") == "matched"]
    ambiguous = [r for r in records_b if r.get("_match_type") == "ambiguous"]
    unmatched = [r for r in records_b if r.get("_match_type") == "unmatched"]
    print(f"Base datasets: A={len(records_a)}, B={len(records_b)}")
    print(
        f"  matched={len(matched)}, ambiguous={len(ambiguous)}, unmatched={len(unmatched)}"
    )

    # ---------------------------------------------------------------
    # Injection records (3,000 B-side records referencing existing A pool)
    # ---------------------------------------------------------------
    # These simulate B records arriving via the live API after initial load.
    # They must reference the EXISTING A-side pool, not a separate one.
    #
    # Split: 2,100 matched (clear/moderate noise) / 300 ambiguous (heavy) / 600 unmatched
    rng = random.Random(SEED + 2000)

    # Re-seed the module-level generators for deterministic noise functions
    generate.rng.seed(SEED + 2000)
    generate.Faker.seed(SEED + 2000)

    a_ids = [r["entity_id"] for r in records_a]
    a_by_id = {r["entity_id"]: r for r in records_a}

    shuffled = list(a_ids)
    rng.shuffle(shuffled)
    match_a_ids = shuffled[:2100]
    ambig_a_ids = shuffled[2100:2400]
    # Remaining 600: unmatched (no A reference)

    inject_records = []
    ground_truth_inject = []

    # --- 2,100 matched injection records ---
    for i, a_id in enumerate(match_a_ids):
        a_rec = a_by_id[a_id]
        b_id = f"CP-INJ-{i:07d}"

        noise_level = rng.choices(["clear", "moderate"], weights=[0.6, 0.4])[0]
        noised_name = generate.add_name_noise(a_rec["legal_name"], noise_level)
        lei = a_rec["lei"] if rng.random() < 0.7 else ""
        addr = a_rec.get("registered_address", "")
        noised_addr = generate.add_address_noise(addr) if rng.random() < 0.5 else addr

        inject_records.append(
            {
                "counterparty_id": b_id,
                "counterparty_name": noised_name,
                "domicile": a_rec["country_code"],
                "lei_code": lei,
                "counterparty_address": noised_addr,
                "_true_a_id": a_id,
                "_match_type": "matched",
            }
        )
        ground_truth_inject.append((a_id, b_id))

    # --- 300 ambiguous injection records ---
    for i, a_id in enumerate(ambig_a_ids):
        a_rec = a_by_id[a_id]
        b_id = f"CP-INJ-{2100 + i:07d}"

        noised_name = generate.add_name_noise(a_rec["legal_name"], "heavy")
        country = (
            a_rec["country_code"]
            if rng.random() < 0.5
            else rng.choice(generate.COUNTRIES)
        )

        inject_records.append(
            {
                "counterparty_id": b_id,
                "counterparty_name": noised_name,
                "domicile": country,
                "lei_code": "",
                "counterparty_address": "",
                "_true_a_id": a_id,
                "_match_type": "ambiguous",
            }
        )
        # Ambiguous are NOT in ground truth

    # --- 600 unmatched injection records ---
    for i in range(600):
        b_id = f"CP-INJ-{2400 + i:07d}"
        country = rng.choice(generate.COUNTRIES)

        inject_records.append(
            {
                "counterparty_id": b_id,
                "counterparty_name": generate.company_name(country),
                "domicile": country,
                "lei_code": "",
                "counterparty_address": generate.generate_address(),
                "_true_a_id": "",
                "_match_type": "unmatched",
            }
        )

    # Shuffle injection records (deterministic)
    rng.shuffle(inject_records)

    # Write injection CSV (external fields only — strip internal metadata)
    inject_path = os.path.join(DATA_DIR, "inject_events.csv")
    ext_fields = [
        "counterparty_id",
        "counterparty_name",
        "domicile",
        "lei_code",
        "counterparty_address",
    ]
    with open(inject_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ext_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inject_records)
    print(f"  Wrote inject_events.csv ({len(inject_records)} records)")

    # Also write a lookup for the inject records (for test validation)
    inject_lookup_path = os.path.join(DATA_DIR, "inject_lookup.csv")
    with open(inject_lookup_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["counterparty_id", "_true_a_id", "_match_type"]
        )
        writer.writeheader()
        for rec in inject_records:
            writer.writerow(
                {
                    "counterparty_id": rec["counterparty_id"],
                    "_true_a_id": rec["_true_a_id"],
                    "_match_type": rec["_match_type"],
                }
            )
    print(f"  Wrote inject_lookup.csv ({len(inject_records)} records)")

    # ---------------------------------------------------------------
    # Ground truth CSV (base + injection matched pairs)
    # ---------------------------------------------------------------
    base_truth = []
    for rec in records_b:
        if rec.get("_true_a_id") and rec.get("_match_type") == "matched":
            base_truth.append((rec["_true_a_id"], rec["counterparty_id"]))

    all_truth = base_truth + ground_truth_inject
    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id", "counterparty_id"])
        for a_id, b_id in sorted(all_truth):
            writer.writerow([a_id, b_id])
    print(
        f"  Wrote ground_truth.csv ({len(all_truth)} pairs: {len(base_truth)} base + {len(ground_truth_inject)} inject)"
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\nSummary:")
    print(f"  Base A records:     {len(records_a):,}")
    print(f"  Base B records:     {len(records_b):,}")
    print(f"    matched:          {len(matched):,}")
    print(f"    ambiguous:        {len(ambiguous):,}")
    print(f"    unmatched:        {len(unmatched):,}")
    print(f"  Injection records:  {len(inject_records):,}")
    print(f"    matched:          2,100")
    print(f"    ambiguous:        300")
    print(f"    unmatched:        600")
    print(f"  Ground truth pairs: {len(all_truth):,}")
    print(f"  Files written to:   {DATA_DIR}/")


if __name__ == "__main__":
    main()
