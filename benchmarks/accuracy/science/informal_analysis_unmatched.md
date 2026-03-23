# Informal Analysis: Why Unmatched Records Score High

Sampled from baseline round 0 (BGE-small, untrained) holdout data. These are
B records with `_match_type=unmatched` — no true A counterpart exists — that
melder nonetheless paired with an A record and scored above review_floor (0.60).

---

## The 0.68-0.70 range (structural similarity)

Three random samples from the bulk of the unmatched population:

```
Pair 1 (score: 0.6971)
  A: Blake, Taylor and Williams NV
  B: Gilbert-Tucker NV
  A: 05479 Townsend Key, New Kevinville, MS 25256
  B: 6210 Melissa Keys, Mistyborough, TX 55538

Pair 2 (score: 0.6806)
  A: Taylor PLC Capital
  B: Conley-Taylor Corporation
  A: 6720 Joseph Mews Suite 731, East Brenda, PR 98927
  B: 38900 Sullivan Plains, Victoriaburgh, IA 73908

Pair 3 (score: 0.6971)
  A: Smith, Hall and Woodard NV
  B: Frazier-Poole NV
  A: 5279 Joshua Key, East James, WI 58682
  B: 19847 Hamilton Route, Matthewhaven, WA 47945
```

**Verdict:** Zero semantic similarity. The scores are driven entirely by format:
both sides are "Surname(s) + legal suffix" with Faker-generated US addresses.
The model says "these are both company names in the same country" — true but
useless for matching. A human would reject all three instantly.

---

## Spread across the 0.70-0.86 range (per-field scores visible)

Five samples chosen from different score bands to understand what drives
higher scores among non-matches:

```
Pair 1 (composite: 0.86, name: 0.85, addr: 0.87)
  A: Hoover and Sons AB
  B: Patterson and Sons AB
  A: PSC 7386, Box 9223, APO AE 52908
  B: PSC 6117, Box 8392, APO AP 29272

Pair 2 (composite: 0.76, name: 0.85, addr: 0.62)
  A: Jones, Martin and Mills Group
  B: Mills Ltd Group
  A: 4350 Jones Stravenue Apt. 494, New Susan, KS 81060
  B: 515 Hernandez Corner, Kimhaven, MT 62448

Pair 3 (composite: 0.74, name: 0.74, addr: 0.73)
  A: Collins, Williams and Hebert Limited
  B: Hobbs, Gibson and Burke Corp
  A: 703 Charles Pine Apt. 285, Pattonfurt, MT 72264
  B: 185 Vaughn Vista Apt. 432, Diazfurt, PW 68268

Pair 4 (composite: 0.72, name: 0.82, addr: 0.57)
  A: Lambert, Kennedy and Davenport Holdings
  B: Olsen, Zamora and Brown Holdings
  A: 8126 Nicholas Mission, Port Angelaton, AR 07970
  B: 2106 Erickson Shoal, East Carriechester, NJ 81969

Pair 5 (composite: 0.70, name: 0.74, addr: 0.65)
  A: Robinson, Jones and Huber BV
  B: Johnson, Gibson and Anthony BV
  A: 7867 Hill Crescent Suite 798, East Elizabeth, OR 53894
  B: 07554 Hernandez Divide, West Donnabury, ME 60911
```

### Two failure modes

**1. Structural matching (pairs 3, 4, 5):** "Surname, Surname and Surname
[suffix]" scores 0.70-0.74 regardless of which surnames appear. The model
encodes the template pattern, not the actual names. This is the bulk of the
review noise — all three populations overlap badly in this range.

**2. Partial token overlap (pairs 1, 2):** A shared word like "Mills", "and
Sons", or a rare address format (APO) pushes the score to 0.76-0.86. These
are harder cases — the model is seeing real shared meaning, it's just not
enough to conclude it's the same entity. Pair 1 is the worst: "and Sons AB"
matches on 2 of 3 tokens, and both addresses are military APO format (rare
in Faker output, so highly distinctive to the model).

### Implications

- The review_floor of 0.60 captures a lot of structural noise that has zero
  matching value. Everything below ~0.80 is essentially "these are both
  company names" — not actionable.
- Training successfully pushes these scores down (baseline R4 has zero
  unmatched above 0.66) but at the cost of dragging matched scores down too.
- The partial-overlap cases (pair 1, 2) are the ones training needs to handle
  carefully — they have genuine shared tokens that shouldn't be ignored, they
  just aren't sufficient for a match.
