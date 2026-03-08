#!/usr/bin/env python3
"""
Generate synthetic datasets A and B for matchlib testing.

Dataset A: 100,000 reference entity records (the master / authoritative side)
Dataset B: 100,000 counterparty records (the incoming / fuzzy side)

Relationship:
  - 70,000 B records have a true match in A (with various levels of name noise)
  - 20,000 B records are near-matches (ambiguous, should land in review queue)
  - 10,000 B records have no match in A

Output formats: CSV, Parquet, JSONL
Output location: testdata/
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_A = 100_000
N_B = 100_000

N_MATCHED = 70_000  # B records with a clear A match
N_AMBIGUOUS = 20_000  # B records that are near-matches (review queue)
N_UNMATCHED = 10_000  # B records with no A match

COUNTRIES = [
    "GB",
    "US",
    "DE",
    "FR",
    "JP",
    "CH",
    "SG",
    "HK",
    "AU",
    "CA",
    "NL",
    "SE",
    "DK",
    "NO",
    "IT",
    "ES",
    "BE",
    "AT",
    "LU",
    "IE",
]

SECTORS = [
    "Banking",
    "Insurance",
    "Asset Management",
    "Hedge Fund",
    "Private Equity",
    "Technology",
    "Energy",
    "Healthcare",
    "Telecommunications",
    "Real Estate",
    "Industrials",
    "Consumer Goods",
    "Utilities",
    "Materials",
    "Transportation",
]

RATINGS = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "NR",
]

LEGAL_SUFFIXES = [
    "PLC",
    "Ltd",
    "Limited",
    "LLC",
    "Inc",
    "Corp",
    "Corporation",
    "AG",
    "GmbH",
    "SA",
    "SAS",
    "SRL",
    "NV",
    "BV",
    "AB",
    "& Co",
    "& Partners",
    "Group",
    "Holdings",
    "Capital",
    "Partners",
]

NOISE_ABBREV = {
    "PLC": ["plc", "P.L.C.", "Plc"],
    "Limited": ["Ltd", "ltd.", "LTD"],
    "Ltd": ["Limited", "ltd.", "LTD"],
    "Corporation": ["Corp", "Corp.", "corp"],
    "Inc": ["Inc.", "inc", "Incorporated"],
    "LLC": ["L.L.C.", "llc"],
    "Group": ["Grp", "GRP", "grp."],
    "Holdings": ["Hldgs", "Hldg", "hldgs"],
    "Capital": ["Cap", "Cap.", "cap"],
    "Partners": ["Ptnrs", "Ptrs"],
    "AG": ["A.G.", "ag"],
    "GmbH": ["gmbh", "G.m.b.H."],
    "SA": ["S.A.", "sa"],
    "NV": ["N.V.", "nv"],
    "BV": ["B.V.", "bv"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fake = Faker()
rng = random.Random(SEED)
Faker.seed(SEED)


def random_lei() -> str:
    """Generate a plausible 20-char LEI string (not GLEIF-valid, just realistic)."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    prefix = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))
    middle = "00"
    body = "".join(rng.choices(chars, k=12))
    check = f"{rng.randint(10, 99)}"
    return f"{prefix}{middle}{body}{check}"


def company_name(country: str) -> str:
    """Generate a realistic company name for the given country."""
    base = fake.company()
    # Strip any existing suffix faker may have added and add a realistic one
    # weighted towards the country's typical legal forms
    suffix_pool = LEGAL_SUFFIXES.copy()
    if country in ("GB",):
        suffix_pool = [
            "PLC",
            "Ltd",
            "Limited",
            "Group",
            "Holdings",
            "Capital",
            "Partners",
        ]
    elif country in ("DE", "AT", "CH"):
        suffix_pool = ["AG", "GmbH", "& Co", "Group", "Holdings"]
    elif country in ("FR",):
        suffix_pool = ["SA", "SAS", "Group", "Capital"]
    elif country in ("US",):
        suffix_pool = [
            "Inc",
            "Corp",
            "LLC",
            "Corporation",
            "Group",
            "Capital",
            "Partners",
        ]
    elif country in ("NL", "BE"):
        suffix_pool = ["NV", "BV", "Group", "Holdings"]
    elif country in ("SE", "DK", "NO"):
        suffix_pool = ["AB", "Group", "Holdings", "Capital"]
    suffix = rng.choice(suffix_pool)
    return f"{base} {suffix}"


def add_name_noise(name: str, level: str) -> str:
    """
    Add realistic noise to a company name.

    level:
      "clear"    — minor whitespace / punctuation variation
      "moderate" — abbreviations, word reordering, partial name
      "heavy"    — significant truncation or alias
    """
    if level == "clear":
        # Minor: change suffix abbreviation, add/remove comma, change case slightly
        for full, abbrevs in NOISE_ABBREV.items():
            if full in name:
                return name.replace(full, rng.choice(abbrevs), 1)
        return name.lower() if rng.random() < 0.3 else name

    if level == "moderate":
        words = name.split()
        if len(words) > 3:
            # Drop a word or two
            drop = rng.randint(1, min(2, len(words) - 2))
            words = words[:-drop]
        for full, abbrevs in NOISE_ABBREV.items():
            if full in name:
                words_str = " ".join(words)
                return words_str.replace(full, rng.choice(abbrevs), 1)
        return " ".join(words)

    if level == "heavy":
        words = name.split()
        # Return just the first 1-2 words
        keep = rng.randint(1, min(2, len(words)))
        return " ".join(words[:keep])

    return name


def make_short_name(legal_name: str) -> str:
    """Derive a short_name from the legal name (1-2 core words)."""
    words = legal_name.split()
    keep = min(2, len(words))
    return " ".join(words[:keep])


# ---------------------------------------------------------------------------
# Dataset A — reference entity master
# ---------------------------------------------------------------------------


def generate_dataset_a(n: int) -> list[dict]:
    records = []
    for i in range(n):
        country = rng.choice(COUNTRIES)
        lei = random_lei() if rng.random() < 0.85 else ""
        legal_name = company_name(country)
        rec = {
            "entity_id": f"ENT-{i:06d}",
            "legal_name": legal_name,
            "short_name": make_short_name(legal_name),
            "country_code": country,
            "sector": rng.choice(SECTORS),
            "credit_rating": rng.choice(RATINGS),
            "lei": lei,
            "incorporated_date": (
                datetime(1900, 1, 1) + timedelta(days=rng.randint(0, 45000))
            ).strftime("%Y-%m-%d"),
            "num_employees": rng.randint(10, 500_000),
            "annual_revenue_usd": rng.randint(100_000, 50_000_000_000),
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Dataset B — incoming counterparty records
# ---------------------------------------------------------------------------


def generate_dataset_b(records_a: list[dict]) -> list[dict]:
    records = []
    b_idx = 0

    # --- 70,000 matched records (clear to moderate noise) ---
    a_sample = rng.sample(range(len(records_a)), N_MATCHED)
    for a_i in a_sample:
        a = records_a[a_i]
        noise_level = rng.choices(["clear", "moderate"], weights=[0.6, 0.4])[0]
        country = a["country_code"]
        # Occasionally use a synonym for the country
        if rng.random() < 0.05:
            country = rng.choice(COUNTRIES)  # wrong country — blocking will miss
        lei = a["lei"] if rng.random() < 0.70 else ""  # 30% LEI absent in B

        rec = {
            "counterparty_id": f"CP-{b_idx:06d}",
            "counterparty_name": add_name_noise(a["legal_name"], noise_level),
            "domicile": country,
            "lei_code": lei,
            "onboarded_date": (
                datetime(2010, 1, 1) + timedelta(days=rng.randint(0, 5000))
            ).strftime("%Y-%m-%d"),
            "relationship_manager": fake.name(),
            "credit_limit_usd": rng.randint(100_000, 100_000_000),
            "internal_rating": rng.choice(["1A", "1B", "2A", "2B", "3A", "3B", "NR"]),
            # Ground truth (not used by matcher — for test validation only)
            "_true_a_id": a["entity_id"],
            "_match_type": "matched",
        }
        records.append(rec)
        b_idx += 1

    # --- 20,000 ambiguous (near-matches, heavy noise) ---
    a_sample2 = rng.sample(range(len(records_a)), N_AMBIGUOUS)
    for a_i in a_sample2:
        a = records_a[a_i]
        country = a["country_code"]
        rec = {
            "counterparty_id": f"CP-{b_idx:06d}",
            "counterparty_name": add_name_noise(a["legal_name"], "heavy"),
            "domicile": country,
            "lei_code": "",  # LEI always absent for ambiguous
            "onboarded_date": (
                datetime(2010, 1, 1) + timedelta(days=rng.randint(0, 5000))
            ).strftime("%Y-%m-%d"),
            "relationship_manager": fake.name(),
            "credit_limit_usd": rng.randint(100_000, 100_000_000),
            "internal_rating": rng.choice(["1A", "1B", "2A", "2B", "3A", "3B", "NR"]),
            "_true_a_id": a["entity_id"],
            "_match_type": "ambiguous",
        }
        records.append(rec)
        b_idx += 1

    # --- 10,000 unmatched ---
    for _ in range(N_UNMATCHED):
        country = rng.choice(COUNTRIES)
        rec = {
            "counterparty_id": f"CP-{b_idx:06d}",
            "counterparty_name": company_name(country),
            "domicile": country,
            "lei_code": random_lei() if rng.random() < 0.3 else "",
            "onboarded_date": (
                datetime(2010, 1, 1) + timedelta(days=rng.randint(0, 5000))
            ).strftime("%Y-%m-%d"),
            "relationship_manager": fake.name(),
            "credit_limit_usd": rng.randint(100_000, 100_000_000),
            "internal_rating": rng.choice(["1A", "1B", "2A", "2B", "3A", "3B", "NR"]),
            "_true_a_id": "",
            "_match_type": "unmatched",
        }
        records.append(rec)
        b_idx += 1

    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"  wrote {path} ({len(records):,} records)")


def write_parquet(records: list[dict], path: str) -> None:
    df = pd.DataFrame(records)
    # Ensure string columns are stored as str (not object/mixed)
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    print(f"  wrote {path} ({len(records):,} records)")


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  wrote {path} ({len(records):,} records)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic match test datasets"
    )
    parser.add_argument(
        "--out",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory (default: same directory as this script)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Generate small datasets (1,000 records each) for fast unit tests",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Generate datasets with exactly N records each (overrides --small)",
    )
    args = parser.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    global N_A, N_B, N_MATCHED, N_AMBIGUOUS, N_UNMATCHED
    if args.size is not None:
        N_A = args.size
        N_B = args.size
        N_MATCHED = int(N_A * 0.70)
        N_AMBIGUOUS = int(N_A * 0.20)
        N_UNMATCHED = N_A - N_MATCHED - N_AMBIGUOUS
        suffix = f"_{N_A}"
        print(f"Generating datasets ({N_A:,} records each) ...")
    elif args.small:
        N_A = 1_000
        N_B = 1_000
        N_MATCHED = 700
        N_AMBIGUOUS = 200
        N_UNMATCHED = 100
        suffix = "_1000"
        print("Generating SMALL datasets (1,000 records each) ...")
    else:
        suffix = ""
        print(f"Generating FULL datasets ({N_A:,} × A, {N_B:,} × B) ...")

    print("Generating dataset A ...")
    records_a = generate_dataset_a(N_A)

    print("Generating dataset B ...")
    records_b = generate_dataset_b(records_a)

    # Dataset A — all three formats
    print("Writing dataset A ...")
    write_csv(records_a, os.path.join(out, f"dataset_a{suffix}.csv"))
    write_parquet(records_a, os.path.join(out, f"dataset_a{suffix}.parquet"))
    write_jsonl(records_a, os.path.join(out, f"dataset_a{suffix}.jsonl"))

    # Dataset B — all three formats
    print("Writing dataset B ...")
    write_csv(records_b, os.path.join(out, f"dataset_b{suffix}.csv"))
    write_parquet(records_b, os.path.join(out, f"dataset_b{suffix}.parquet"))
    write_jsonl(records_b, os.path.join(out, f"dataset_b{suffix}.jsonl"))

    # Ground truth cross-map (only matched records, for seeding tests)
    print("Writing ground truth cross-map ...")
    gt_path = os.path.join(out, "ground_truth_crossmap.csv")
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id", "counterparty_id"])
        for rec in records_b:
            if rec["_match_type"] == "matched":
                writer.writerow([rec["_true_a_id"], rec["counterparty_id"]])
    print(f"  wrote {gt_path}")

    print("\nDone.")
    print(f"  Dataset A: {len(records_a):,} records")
    print(f"  Dataset B: {len(records_b):,} records")
    print(f"    matched:    {N_MATCHED:,}")
    print(f"    ambiguous:  {N_AMBIGUOUS:,}")
    print(f"    unmatched:  {N_UNMATCHED:,}")


if __name__ == "__main__":
    main()
