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
import re
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
N_AMBIGUOUS = 10_000  # B records that are near-matches (review queue)
N_UNMATCHED = 20_000  # B records with no A match

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

# Common financial word abbreviations used in "clear" noise.
# These represent realistic shorthand a counterparty ops team might use.
WORD_ABBREVS = {
    "International": ["Intl", "Int'l", "Internatl"],
    "Global": ["Glbl", "Gbl"],
    "Management": ["Mgmt", "Mgt"],
    "Financial": ["Fin", "Finl"],
    "Investment": ["Inv", "Invst"],
    "Investments": ["Invsts", "Invst"],
    "Asset": ["Asst", "Ast"],
    "Services": ["Svcs", "Svc"],
    "Technology": ["Tech"],
    "Securities": ["Secs", "Sec"],
    "Advisors": ["Advisers", "Advs"],
    "Advisers": ["Advisors", "Advs"],
    "Fund": ["Fd"],
    "Bank": ["Bnk", "Bk"],
    "Corporate": ["Corp"],
    "Private": ["Pvt", "Priv"],
}

# Words that can be swapped for similar alternatives in "moderate" noise.
# Simulates how counterparties refer to the same entity with a different
# qualifier — the most common real-world source of near-miss FPs.
QUALIFIER_SUBSTITUTES = {
    "International": ["Global", "Worldwide", "Continental", "European", "Americas"],
    "Global": ["International", "Worldwide", "Universal"],
    "Management": ["Advisors", "Advisers", "Capital", "Investments"],
    "Asset": ["Wealth", "Fund", "Investment"],
    "Wealth": ["Asset", "Private", "Investment"],
    "Holdings": ["Investments", "Ventures", "Enterprises", "Finance"],
    "Capital": ["Finance", "Financial", "Investments", "Partners"],
    "Partners": ["Associates", "Advisors", "Ventures", "Capital"],
    "Group": ["Holdings", "Investments", "Enterprises"],
    "Finance": ["Capital", "Financial", "Investments"],
    "Financial": ["Finance", "Capital", "Investments"],
    "Investment": ["Asset", "Wealth", "Fund"],
    "Bank": ["Bancorp", "Trust", "Banking"],
    "Private": ["Institutional", "Corporate", "Commercial"],
}

ADDRESS_ABBREV = {
    "Street": ["St", "St.", "Str"],
    "Avenue": ["Ave", "Ave.", "Av"],
    "Road": ["Rd", "Rd."],
    "Boulevard": ["Blvd", "Blvd."],
    "Drive": ["Dr", "Dr."],
    "Lane": ["Ln", "Ln."],
    "Court": ["Ct", "Ct."],
    "Place": ["Pl", "Pl."],
    "Suite": ["Ste", "Ste.", "STE"],
    "Floor": ["Fl", "Fl.", "FL"],
    "Building": ["Bldg", "Bldg."],
    "North": ["N", "N."],
    "South": ["S", "S."],
    "East": ["E", "E."],
    "West": ["W", "W."],
    "Apartment": ["Apt", "Apt.", "APT"],
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
    """Add realistic noise to a company name.

    level:
      "clear"    — minor abbreviation or punctuation variation; the match
                   should be easy but the strings aren't identical
      "moderate" — richer transformation: acronym, qualifier swap, word
                   drop, or reorder; designed to produce hard training pairs
      "heavy"    — severe truncation to 1–2 words; used for ambiguous records
                   that are intentionally near-unresolvable
    """
    if level == "clear":
        # Build a candidate pool of single-transformation options.
        candidates: list[str] = []

        # Suffix abbreviation (e.g. PLC → plc)
        for full, abbrevs in NOISE_ABBREV.items():
            if f" {full}" in name or name.endswith(full):
                candidates.append(name.replace(full, rng.choice(abbrevs), 1))

        # Common word abbreviation (e.g. International → Intl, Management → Mgmt)
        for full, abbrevs in WORD_ABBREVS.items():
            if f" {full}" in name or name.startswith(full):
                candidates.append(name.replace(full, rng.choice(abbrevs), 1))

        # Punctuation: & ↔ "and"
        if " & " in name:
            candidates.append(name.replace(" & ", " and ", 1))
        if " and " in name.lower():
            candidates.append(re.sub(r"\band\b", "&", name, count=1, flags=re.I))

        # Case variants
        candidates.append(name.lower())
        candidates.append(name.upper())

        if candidates:
            return rng.choice(candidates)
        return name.lower() if rng.random() < 0.3 else name

    if level == "moderate":
        words = name.split()
        strategies: list[str] = []

        # Qualifier substitution: most realistic — swap "International" → "Global" etc.
        for word in words:
            if word in QUALIFIER_SUBSTITUTES:
                strategies.append("qualifier_swap")
                break

        # Acronym: first letter of each significant word (e.g. GSAM, JPMC)
        sig_words = [
            w for w in words if len(w) > 2 and w.upper() not in {"THE", "AND", "OF"}
        ]
        if len(sig_words) >= 2:
            strategies.append("acronym")

        # Drop trailing word(s) — existing behaviour
        if len(words) > 3:
            strategies.append("drop_trailing")

        # Drop a middle word
        if len(words) > 3:
            strategies.append("drop_middle")

        # Reorder a pair of adjacent words
        if 3 <= len(words) <= 5:
            strategies.append("reorder")

        # Fallback: abbreviate a common word
        for full in WORD_ABBREVS:
            if f" {full}" in name or name.startswith(full):
                strategies.append("word_abbrev")
                break

        strategy = rng.choice(strategies) if strategies else "drop_trailing"

        if strategy == "qualifier_swap":
            new_words = words.copy()
            for i, w in enumerate(new_words):
                if w in QUALIFIER_SUBSTITUTES:
                    alts = [a for a in QUALIFIER_SUBSTITUTES[w] if a != w]
                    if alts:
                        new_words[i] = rng.choice(alts)
                        return " ".join(new_words)

        elif strategy == "acronym":
            letters = [
                w[0]
                for w in words
                if len(w) > 2 and w.upper() not in {"THE", "AND", "OF"}
            ]
            if len(letters) >= 2:
                return "".join(letters).upper()

        elif strategy == "drop_trailing":
            max_drop = max(1, len(words) - 2)
            drop = rng.randint(1, min(2, max_drop))
            result = " ".join(words[:-drop])
            for full, abbrevs in NOISE_ABBREV.items():
                if full in result:
                    return result.replace(full, rng.choice(abbrevs), 1)
            return result

        elif strategy == "drop_middle":
            mid = rng.randint(1, len(words) - 2)
            return " ".join(words[:mid] + words[mid + 1 :])

        elif strategy == "reorder":
            idx = rng.randint(0, len(words) - 2)
            new_words = words.copy()
            new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
            return " ".join(new_words)

        elif strategy == "word_abbrev":
            for full, abbrevs in WORD_ABBREVS.items():
                if f" {full}" in name or name.startswith(full):
                    return name.replace(full, rng.choice(abbrevs), 1)

        # Fallback
        return " ".join(words[:-1]) if len(words) > 2 else name

    if level == "heavy":
        words = name.split()
        # 70%: first 2–3 words (enough to be recognisable but clearly truncated)
        # 30%: acronym — a different but equally hard pattern
        if rng.random() < 0.30 and len(words) >= 3:
            letters = [w[0] for w in words if len(w) > 2]
            if len(letters) >= 2:
                return "".join(letters).upper()
        keep = min(rng.randint(2, 3), len(words))
        return " ".join(words[:keep])

    return name


def make_short_name(legal_name: str) -> str:
    """Derive a short_name from the legal name (1-2 core words)."""
    words = legal_name.split()
    keep = min(2, len(words))
    return " ".join(words[:keep])


def generate_address() -> str:
    """Generate a realistic street address using Faker."""
    return fake.address().replace("\n", ", ")


def add_address_noise(address: str) -> str:
    """
    Add realistic noise to an address. Guarantees at least one mutation.

    Randomly applies one or more of:
      - Abbreviate a street type (Street -> St, Avenue -> Ave, etc.)
      - Drop a secondary unit (Suite/Floor/Apt line)
      - Change comma/spacing
      - Swap to uppercase or lowercase
      - Swap two adjacent digits in the street number
      - Drop the zip/postal code
    """
    result = address
    changed = False

    # 1. Abbreviate a word (60% chance)
    if rng.random() < 0.60:
        for full, abbrevs in ADDRESS_ABBREV.items():
            if full in result:
                result = result.replace(full, rng.choice(abbrevs), 1)
                changed = True
                break

    # 2. Drop secondary unit info — remove "Suite/Apt/Floor NNN" (30% chance)
    if rng.random() < 0.30:
        new = re.sub(
            r",?\s*(Suite|Ste\.?|Apt\.?|Floor|Fl\.?|Unit)\s*#?\d+,?\s*",
            " ",
            result,
            flags=re.IGNORECASE,
        )
        new = " ".join(new.split())
        if new != result:
            result = new
            changed = True

    # 3. Change comma to newline-style or remove comma (25% chance)
    if rng.random() < 0.25:
        if ", " in result:
            result = result.replace(", ", " ", 1)
            changed = True

    # 4. Case change (20% chance)
    if rng.random() < 0.20:
        result = result.upper() if rng.random() < 0.5 else result.lower()
        changed = True

    # 5. Swap two adjacent digits in the street number (20% chance)
    if rng.random() < 0.20:
        m = re.match(r"^(\d{2,})", result)
        if m:
            num = list(m.group(1))
            if len(num) >= 2:
                i = rng.randint(0, len(num) - 2)
                num[i], num[i + 1] = num[i + 1], num[i]
                result = "".join(num) + result[m.end() :]
                changed = True

    # 6. Drop the zip/postal code at the end (15% chance)
    if rng.random() < 0.15:
        new = re.sub(r"\s+\d{5}(-\d{4})?$", "", result)
        if new != result:
            result = new
            changed = True

    # Fallback: if nothing changed, force a case change so noise is guaranteed
    if not changed:
        result = result.lower() if result[0].isupper() else result.upper()

    return result


# ---------------------------------------------------------------------------
# Dataset A — reference entity master
# ---------------------------------------------------------------------------


def generate_dataset_a(n: int, include_addresses: bool = False) -> list[dict]:
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
        if include_addresses:
            rec["registered_address"] = generate_address()
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Dataset B — incoming counterparty records
# ---------------------------------------------------------------------------


def generate_dataset_b(
    records_a: list[dict],
    include_addresses: bool = False,
    preserve_blocking: bool = False,
) -> list[dict]:
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
            if not preserve_blocking:
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
        if include_addresses:
            addr = a.get("registered_address", "")
            # 50% of the time add noise to the address
            rec["counterparty_address"] = (
                add_address_noise(addr) if rng.random() < 0.50 else addr
            )
        records.append(rec)
        b_idx += 1

    # --- 10,000 ambiguous (near-matches, heavy noise) ---
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
        if include_addresses:
            addr = a.get("registered_address", "")
            # Ambiguous: always noise the address
            rec["counterparty_address"] = add_address_noise(addr)
        records.append(rec)
        b_idx += 1

    # --- Unmatched: fully random entities in a random country ---
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
        if include_addresses:
            rec["counterparty_address"] = generate_address()
        records.append(rec)
        b_idx += 1

    rng.shuffle(records)
    return records


def generate_dataset_b_1to1(
    records_a: list[dict],
    include_addresses: bool = False,
    pct_matched: float = 0.60,
    pct_ambiguous: float = 0.10,
) -> list[dict]:
    """Generate one B record per A record — strict 1:1 mapping, no collisions.

    Each A record gets exactly one treatment:
      - matched (pct_matched, default 60%): clear/moderate noise, should auto-match
      - ambiguous (pct_ambiguous, default 10%): heavy noise, ideally review queue
      - unmatched (remainder, default 30%): completely different entity, should not match

    This eliminates crossmap collisions (two B records competing for the same A)
    and produces a cleaner evaluation signal for training runs.
    """
    records = []

    for b_idx, a in enumerate(records_a):
        roll = rng.random()

        if roll < pct_matched:
            # --- Matched: light noise, should auto-match ---
            noise_level = rng.choices(["clear", "moderate"], weights=[0.6, 0.4])[0]
            lei = a["lei"] if rng.random() < 0.70 else ""
            rec = {
                "counterparty_id": f"CP-{b_idx:06d}",
                "counterparty_name": add_name_noise(a["legal_name"], noise_level),
                "domicile": a["country_code"],
                "lei_code": lei,
                "onboarded_date": (
                    datetime(2010, 1, 1) + timedelta(days=rng.randint(0, 5000))
                ).strftime("%Y-%m-%d"),
                "relationship_manager": fake.name(),
                "credit_limit_usd": rng.randint(100_000, 100_000_000),
                "internal_rating": rng.choice(
                    ["1A", "1B", "2A", "2B", "3A", "3B", "NR"]
                ),
                "_true_a_id": a["entity_id"],
                "_match_type": "matched",
            }
            if include_addresses:
                addr = a.get("registered_address", "")
                rec["counterparty_address"] = (
                    add_address_noise(addr) if rng.random() < 0.50 else addr
                )

        elif roll < pct_matched + pct_ambiguous:
            # --- Ambiguous: heavy noise, ideally lands in review ---
            rec = {
                "counterparty_id": f"CP-{b_idx:06d}",
                "counterparty_name": add_name_noise(a["legal_name"], "heavy"),
                "domicile": a["country_code"],
                "lei_code": "",
                "onboarded_date": (
                    datetime(2010, 1, 1) + timedelta(days=rng.randint(0, 5000))
                ).strftime("%Y-%m-%d"),
                "relationship_manager": fake.name(),
                "credit_limit_usd": rng.randint(100_000, 100_000_000),
                "internal_rating": rng.choice(
                    ["1A", "1B", "2A", "2B", "3A", "3B", "NR"]
                ),
                "_true_a_id": a["entity_id"],
                "_match_type": "ambiguous",
            }
            if include_addresses:
                addr = a.get("registered_address", "")
                rec["counterparty_address"] = add_address_noise(addr)

        else:
            # --- Unmatched: completely different entity ---
            country = a["country_code"]
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
                "internal_rating": rng.choice(
                    ["1A", "1B", "2A", "2B", "3A", "3B", "NR"]
                ),
                "_true_a_id": "",
                "_match_type": "unmatched",
            }
            if include_addresses:
                rec["counterparty_address"] = generate_address()

        records.append(rec)

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


def generate_a_with_seed(
    seed: int,
    n: int,
    include_addresses: bool,
    out_dir: str,
) -> list[dict]:
    """Generate only the A-side (reference master) dataset with a given seed.

    Re-seeds both the module-level RNG and Faker so the same seed always
    produces identical output. Writes dataset_a.csv to out_dir and returns
    records_a.

    Use this to create a fixed reference master that persists across all
    training rounds — matching the real-world pattern where the A-side
    entity master is stable while vendor B files arrive daily.
    """
    global N_A

    rng.seed(seed)
    Faker.seed(seed)

    N_A = n

    os.makedirs(out_dir, exist_ok=True)
    records_a = generate_dataset_a(N_A, include_addresses=include_addresses)
    write_csv(records_a, os.path.join(out_dir, "dataset_a.csv"))
    return records_a


def generate_b_from_master(
    master_a_path: str,
    b_seed: int,
    n: int,
    include_addresses: bool,
    out_dir: str,
    preserve_blocking: bool = False,
) -> list[dict]:
    """Generate a fresh B-side (vendor file) dataset against a fixed A master.

    Loads A records from master_a_path, re-seeds the RNG with b_seed, and
    generates n B records (70% matched / 20% ambiguous / 10% unmatched).
    Writes dataset_b.csv to out_dir and returns records_b.

    When preserve_blocking is True, matched B records always keep their true
    A record's country code — no records will be structurally blocked. Use
    this for training runs where blocked records add noise to evaluation.

    Calling this with the same master_a_path but different b_seeds simulates
    successive daily vendor file deliveries against the same reference master —
    the real-world use case this training loop is designed to measure.
    """
    global N_B, N_MATCHED, N_AMBIGUOUS, N_UNMATCHED

    rng.seed(b_seed)
    Faker.seed(b_seed)

    with open(master_a_path, newline="") as f:
        records_a = list(csv.DictReader(f))

    N_B = n
    N_MATCHED = int(n * 0.70)
    N_AMBIGUOUS = int(n * 0.10)
    N_UNMATCHED = n - N_MATCHED - N_AMBIGUOUS

    os.makedirs(out_dir, exist_ok=True)
    records_b = generate_dataset_b(
        records_a,
        include_addresses=include_addresses,
        preserve_blocking=preserve_blocking,
    )
    write_csv(records_b, os.path.join(out_dir, "dataset_b.csv"))
    return records_b


def generate_b_from_master_1to1(
    master_a_path: str,
    b_seed: int,
    n: int,
    include_addresses: bool,
    out_dir: str,
    pct_matched: float = 0.60,
    pct_ambiguous: float = 0.10,
) -> list[dict]:
    """Generate a 1:1 B dataset against a fixed A master.

    Every A record gets exactly one B record. No two B records share the
    same true A — eliminates crossmap collisions in evaluation.

    Treatment split (per record):
      - pct_matched (default 60%): light noise, should auto-match
      - pct_ambiguous (default 10%): heavy noise, ideally review
      - remainder (default 30%): different entity, should not match

    Uses only the first n records from the master (or all if n >= len(A)).
    """
    rng.seed(b_seed)
    Faker.seed(b_seed)

    with open(master_a_path, newline="") as f:
        records_a = list(csv.DictReader(f))

    # Use first n records (or all)
    records_a = records_a[:n]

    os.makedirs(out_dir, exist_ok=True)
    records_b = generate_dataset_b_1to1(
        records_a,
        include_addresses=include_addresses,
        pct_matched=pct_matched,
        pct_ambiguous=pct_ambiguous,
    )
    write_csv(records_b, os.path.join(out_dir, "dataset_b.csv"))
    return records_b


def generate_with_seed(
    seed: int,
    n: int,
    include_addresses: bool,
    out_dir: str,
) -> tuple[list[dict], list[dict]]:
    """Generate A and B datasets with an explicit seed.

    Re-seeds both the module-level RNG and Faker so that the same seed always
    produces identical output regardless of prior calls. Writes dataset_a.csv
    and dataset_b.csv to out_dir and returns (records_a, records_b).

    Uses the standard 70 / 10 / 20 matched / ambiguous / unmatched split.
    """
    global N_A, N_B, N_MATCHED, N_AMBIGUOUS, N_UNMATCHED

    rng.seed(seed)
    Faker.seed(seed)

    N_A = n
    N_B = n
    N_MATCHED = int(n * 0.70)
    N_AMBIGUOUS = int(n * 0.10)
    N_UNMATCHED = n - N_MATCHED - N_AMBIGUOUS

    os.makedirs(out_dir, exist_ok=True)

    records_a = generate_dataset_a(N_A, include_addresses=include_addresses)
    records_b = generate_dataset_b(records_a, include_addresses=include_addresses)

    write_csv(records_a, os.path.join(out_dir, "dataset_a.csv"))
    write_csv(records_b, os.path.join(out_dir, "dataset_b.csv"))

    return records_a, records_b


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
    parser.add_argument(
        "--addresses",
        action="store_true",
        help="Include address fields (registered_address in A, counterparty_address in B)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible generation (default: %(default)s).",
    )
    args = parser.parse_args()

    rng.seed(args.seed)
    Faker.seed(args.seed)

    out = args.out
    os.makedirs(out, exist_ok=True)

    global N_A, N_B, N_MATCHED, N_AMBIGUOUS, N_UNMATCHED
    if args.size is not None:
        N_A = args.size
        N_B = args.size
        N_MATCHED = int(N_A * 0.70)
        N_AMBIGUOUS = int(N_A * 0.20)
        N_UNMATCHED = N_A - N_MATCHED - N_AMBIGUOUS
        suffix = f"_{N_A // 1000}k" if N_A % 1000 == 0 else f"_{N_A}"
        print(f"Generating datasets ({N_A:,} records each) ...")
    elif args.small:
        N_A = 1_000
        N_B = 1_000
        N_MATCHED = 700
        N_AMBIGUOUS = 100
        N_UNMATCHED = 200
        suffix = "_1k"
        print("Generating SMALL datasets (1,000 records each) ...")
    else:
        suffix = ""
        print(f"Generating FULL datasets ({N_A:,} × A, {N_B:,} × B) ...")

    print("Generating dataset A ...")
    records_a = generate_dataset_a(N_A, include_addresses=args.addresses)

    print("Generating dataset B ...")
    records_b = generate_dataset_b(records_a, include_addresses=args.addresses)

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
    gt_path = os.path.join(out, f"ground_truth_crossmap{suffix}.csv")
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
