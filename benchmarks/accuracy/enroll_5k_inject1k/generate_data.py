#!/usr/bin/env python3
"""Generate fixed datasets for the enroll accuracy regression test.

Creates:
  data/pool_5k.csv           -- 5,000 pool records (pre-loaded at startup)
  data/enroll_events.csv     -- 1,000 enroll events (600 matched, 200 ambiguous, 200 unmatched)
  data/remove_ids.csv        -- 50 pool IDs to remove after initial enrollment
  data/post_remove_events.csv -- 50 records enrolled after removals
  data/enroll_lookup.csv     -- ground truth metadata for enroll events
  data/post_remove_lookup.csv -- ground truth metadata for post-remove events

Enroll mode uses symmetric field names (no A/B distinction):
  entity_id, legal_name, short_name, country_code, lei

Uses seed=42 for full reproducibility.

Run from project root:
    python3 benchmarks/accuracy/enroll_5k_inject1k/generate_data.py
"""

import csv
import os
import random
import sys

TEST_DIR = "benchmarks/accuracy/enroll_5k_inject1k"
DATA_DIR = f"{TEST_DIR}/data"
SEED = 42
N_POOL = 5_000
N_ENROLL = 1_000
N_ENROLL_MATCHED = 600
N_ENROLL_AMBIGUOUS = 200
N_ENROLL_UNMATCHED = 200
N_REMOVE = 50
N_POST_REMOVE = 50

# Fields used in enroll mode (symmetric — no A/B distinction)
ENROLL_FIELDS = [
    "entity_id",
    "legal_name",
    "short_name",
    "country_code",
    "lei",
    "registered_address",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Import the shared generator
    sys.path.insert(0, "benchmarks/data")
    import generate

    # -------------------------------------------------------------------
    # Pool dataset (5,000 records) — pre-loaded at startup, no scoring
    # -------------------------------------------------------------------
    # Use generate_a_with_seed to create A-side records, then project
    # to the enroll field set.
    records_a = generate.generate_a_with_seed(
        seed=SEED,
        n=N_POOL,
        include_addresses=True,
        out_dir=DATA_DIR,
    )

    # Write pool CSV with only the enroll fields
    pool_path = os.path.join(DATA_DIR, "pool_5k.csv")
    with open(pool_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENROLL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records_a)
    print(f"  Wrote pool_5k.csv ({len(records_a)} records)")

    # Clean up the dataset_a.csv that generate_a_with_seed wrote
    dataset_a_path = os.path.join(DATA_DIR, "dataset_a.csv")
    if os.path.exists(dataset_a_path):
        os.remove(dataset_a_path)

    # -------------------------------------------------------------------
    # Enroll events (1,000 records)
    # -------------------------------------------------------------------
    # Re-seed for deterministic injection generation
    rng = random.Random(SEED + 3000)
    generate.rng.seed(SEED + 3000)
    generate.Faker.seed(SEED + 3000)

    a_by_id = {r["entity_id"]: r for r in records_a}
    a_ids = list(a_by_id.keys())

    shuffled = list(a_ids)
    rng.shuffle(shuffled)

    match_a_ids = shuffled[:N_ENROLL_MATCHED]
    ambig_a_ids = shuffled[N_ENROLL_MATCHED : N_ENROLL_MATCHED + N_ENROLL_AMBIGUOUS]
    # Remaining are not referenced (unmatched use fresh names)

    enroll_records = []
    enroll_lookup = []

    # --- 600 matched enroll records (clear/moderate noise) ---
    for i, a_id in enumerate(match_a_ids):
        a_rec = a_by_id[a_id]
        enr_id = f"ENR-{i:07d}"

        noise_level = rng.choices(["clear", "moderate"], weights=[0.6, 0.4])[0]
        noised_name = generate.add_name_noise(a_rec["legal_name"], noise_level)
        lei = a_rec["lei"] if rng.random() < 0.7 else ""
        address = a_rec.get("registered_address", "")
        address = generate.add_address_noise(address) if rng.random() < 0.5 else address

        enroll_records.append(
            {
                "entity_id": enr_id,
                "legal_name": noised_name,
                "short_name": generate.make_short_name(noised_name),
                "country_code": a_rec["country_code"],
                "lei": lei,
                "registered_address": address,
            }
        )
        enroll_lookup.append(
            {
                "entity_id": enr_id,
                "_true_pool_id": a_id,
                "_match_type": "matched",
            }
        )

    # --- 200 ambiguous enroll records (heavy noise) ---
    for i, a_id in enumerate(ambig_a_ids):
        a_rec = a_by_id[a_id]
        enr_id = f"ENR-{N_ENROLL_MATCHED + i:07d}"

        noised_name = generate.add_name_noise(a_rec["legal_name"], "heavy")
        # Ambiguous: sometimes use a different country to make matching harder
        country = (
            a_rec["country_code"]
            if rng.random() < 0.5
            else rng.choice(generate.COUNTRIES)
        )
        address = a_rec.get("registered_address", "")
        address = generate.add_address_noise(address) if rng.random() < 0.3 else ""

        enroll_records.append(
            {
                "entity_id": enr_id,
                "legal_name": noised_name,
                "short_name": generate.make_short_name(noised_name),
                "country_code": country,
                "lei": "",
                "registered_address": address,
            }
        )
        enroll_lookup.append(
            {
                "entity_id": enr_id,
                "_true_pool_id": a_id,
                "_match_type": "ambiguous",
            }
        )

    # --- 200 unmatched enroll records (completely new entities) ---
    for i in range(N_ENROLL_UNMATCHED):
        enr_id = f"ENR-{N_ENROLL_MATCHED + N_ENROLL_AMBIGUOUS + i:07d}"
        country = rng.choice(generate.COUNTRIES)

        enroll_records.append(
            {
                "entity_id": enr_id,
                "legal_name": generate.company_name(country),
                "short_name": generate.make_short_name(generate.company_name(country)),
                "country_code": country,
                "lei": generate.random_lei() if rng.random() < 0.3 else "",
                "registered_address": generate.generate_address(),
            }
        )
        enroll_lookup.append(
            {
                "entity_id": enr_id,
                "_true_pool_id": "",
                "_match_type": "unmatched",
            }
        )

    # Shuffle enroll events (deterministic)
    combined = list(zip(enroll_records, enroll_lookup))
    rng.shuffle(combined)
    enroll_records, enroll_lookup = zip(*combined)
    enroll_records = list(enroll_records)
    enroll_lookup = list(enroll_lookup)

    # Write enroll events CSV
    enroll_path = os.path.join(DATA_DIR, "enroll_events.csv")
    with open(enroll_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENROLL_FIELDS)
        writer.writeheader()
        writer.writerows(enroll_records)
    print(f"  Wrote enroll_events.csv ({len(enroll_records)} records)")

    # Write enroll lookup CSV (for test validation)
    lookup_path = os.path.join(DATA_DIR, "enroll_lookup.csv")
    with open(lookup_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["entity_id", "_true_pool_id", "_match_type"]
        )
        writer.writeheader()
        writer.writerows(enroll_lookup)
    print(f"  Wrote enroll_lookup.csv ({len(enroll_lookup)} records)")

    # -------------------------------------------------------------------
    # Remove IDs (50 pool records to remove)
    # -------------------------------------------------------------------
    # Re-seed for deterministic removal selection
    rng2 = random.Random(SEED + 4000)

    remove_candidates = list(a_ids)
    rng2.shuffle(remove_candidates)
    remove_ids = remove_candidates[:N_REMOVE]

    remove_path = os.path.join(DATA_DIR, "remove_ids.csv")
    with open(remove_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity_id"])
        writer.writeheader()
        for rid in remove_ids:
            writer.writerow({"entity_id": rid})
    print(f"  Wrote remove_ids.csv ({len(remove_ids)} IDs)")

    # -------------------------------------------------------------------
    # Post-remove enroll events (50 records)
    # -------------------------------------------------------------------
    # Re-seed for deterministic post-remove generation
    rng3 = random.Random(SEED + 5000)
    generate.rng.seed(SEED + 5000)
    generate.Faker.seed(SEED + 5000)

    removed_set = set(remove_ids)
    # Pool IDs that were NOT removed (still in pool)
    remaining_ids = [aid for aid in a_ids if aid not in removed_set]

    rng3.shuffle(remaining_ids)

    post_remove_records = []
    post_remove_lookup = []

    # 30 records that reference remaining pool records (should match)
    for i in range(30):
        a_id = remaining_ids[i]
        a_rec = a_by_id[a_id]
        pr_id = f"ENR-PR-{i:05d}"

        noise_level = rng3.choices(["clear", "moderate"], weights=[0.6, 0.4])[0]
        noised_name = generate.add_name_noise(a_rec["legal_name"], noise_level)
        lei = a_rec["lei"] if rng3.random() < 0.7 else ""
        address = a_rec.get("registered_address", "")
        address = (
            generate.add_address_noise(address) if rng3.random() < 0.5 else address
        )

        post_remove_records.append(
            {
                "entity_id": pr_id,
                "legal_name": noised_name,
                "short_name": generate.make_short_name(noised_name),
                "country_code": a_rec["country_code"],
                "lei": lei,
                "registered_address": address,
            }
        )
        post_remove_lookup.append(
            {
                "entity_id": pr_id,
                "_true_pool_id": a_id,
                "_match_type": "matched",
            }
        )

    # 10 records that reference REMOVED pool records (should NOT match)
    for i in range(10):
        a_id = remove_ids[i]
        a_rec = a_by_id[a_id]
        pr_id = f"ENR-PR-{30 + i:05d}"

        noise_level = rng3.choices(["clear", "moderate"], weights=[0.6, 0.4])[0]
        noised_name = generate.add_name_noise(a_rec["legal_name"], noise_level)
        lei = a_rec["lei"] if rng3.random() < 0.7 else ""
        address = a_rec.get("registered_address", "")
        address = (
            generate.add_address_noise(address) if rng3.random() < 0.5 else address
        )

        post_remove_records.append(
            {
                "entity_id": pr_id,
                "legal_name": noised_name,
                "short_name": generate.make_short_name(noised_name),
                "country_code": a_rec["country_code"],
                "lei": lei,
                "registered_address": address,
            }
        )
        post_remove_lookup.append(
            {
                "entity_id": pr_id,
                "_true_pool_id": a_id,
                "_match_type": "removed_target",
            }
        )

    # 10 completely unrelated records
    for i in range(10):
        pr_id = f"ENR-PR-{40 + i:05d}"
        country = rng3.choice(generate.COUNTRIES)

        post_remove_records.append(
            {
                "entity_id": pr_id,
                "legal_name": generate.company_name(country),
                "short_name": generate.make_short_name(generate.company_name(country)),
                "country_code": country,
                "lei": generate.random_lei() if rng3.random() < 0.3 else "",
                "registered_address": generate.generate_address(),
            }
        )
        post_remove_lookup.append(
            {
                "entity_id": pr_id,
                "_true_pool_id": "",
                "_match_type": "unmatched",
            }
        )

    # Shuffle post-remove events (deterministic)
    combined_pr = list(zip(post_remove_records, post_remove_lookup))
    rng3.shuffle(combined_pr)
    post_remove_records, post_remove_lookup = zip(*combined_pr)
    post_remove_records = list(post_remove_records)
    post_remove_lookup = list(post_remove_lookup)

    # Write post-remove events CSV
    pr_path = os.path.join(DATA_DIR, "post_remove_events.csv")
    with open(pr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENROLL_FIELDS)
        writer.writeheader()
        writer.writerows(post_remove_records)
    print(f"  Wrote post_remove_events.csv ({len(post_remove_records)} records)")

    # Write post-remove lookup CSV
    pr_lookup_path = os.path.join(DATA_DIR, "post_remove_lookup.csv")
    with open(pr_lookup_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["entity_id", "_true_pool_id", "_match_type"]
        )
        writer.writeheader()
        writer.writerows(post_remove_lookup)
    print(f"  Wrote post_remove_lookup.csv ({len(post_remove_lookup)} records)")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\nSummary:")
    print(f"  Pool records:         {len(records_a):,}")
    print(f"  Enroll events:        {len(enroll_records):,}")
    print(f"    matched:            {N_ENROLL_MATCHED:,}")
    print(f"    ambiguous:          {N_ENROLL_AMBIGUOUS:,}")
    print(f"    unmatched:          {N_ENROLL_UNMATCHED:,}")
    print(f"  Remove IDs:           {len(remove_ids):,}")
    print(f"  Post-remove events:   {len(post_remove_records):,}")
    print(f"    matched (remaining): 30")
    print(f"    removed_target:      10")
    print(f"    unmatched:           10")
    print(f"  Files written to:     {DATA_DIR}/")


if __name__ == "__main__":
    main()
