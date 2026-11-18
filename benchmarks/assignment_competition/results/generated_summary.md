# Assignment competition

Descriptive first pass, not a significance test. CSV permutations do not control Rust HashMap iteration or scheduling; local repair is not a global determinism guarantee. Expected rank-r repairability requires top_n >= r, but sampled order may already give five matches or leave four. The original background contains 2,000 ambiguous records with no verified identity oracle; their strict-ID nonagreement is not counted as an identity error.

Status: **complete**; measured runs: 160
Preparation and calibration excluded from timing. Ambiguous quality uses allowed mappings, not arbitrary strict IDs.

|N|Binary|Family|Mean matches|Mean correct|Mean incorrect|Mean allowed ambiguous|Mean changed B|Distinct outcomes|Groups at max min/median/max|
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|1|baseline|rank2|80.00|69.90|10.10|0.00|19.557894736842105|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|baseline|rank3|80.00|70.45|9.55|0.00|20.03157894736842|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|baseline|rank5|80.00|69.75|10.25|0.00|20.221052631578946|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|baseline|ambiguous|20.00|0.00|0.00|20.00|32.126315789473686|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|baseline|unrepairable|80.00|69.30|10.70|0.00|19.557894736842105|20|{'min': 20, 'median': 20.0, 'max': 20}|
|1|baseline|background|5162.00|4517.90|32.35|0.00|452.69473684210527|20|None|
|1|experimental|rank2|80.00|70.20|9.80|0.00|20.08421052631579|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|experimental|rank3|80.00|69.70|10.30|0.00|20.231578947368423|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|experimental|rank5|80.00|69.25|10.75|0.00|19.905263157894737|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|experimental|ambiguous|20.00|0.00|0.00|20.00|32.06315789473684|20|{'min': 0, 'median': 0.0, 'max': 0}|
|1|experimental|unrepairable|80.00|69.80|10.20|0.00|20.25263157894737|20|{'min': 20, 'median': 20.0, 'max': 20}|
|1|experimental|background|5162.00|4514.10|31.95|0.00|449.2315789473684|20|None|
|2|baseline|rank2|90.25|80.50|9.75|0.00|19.778947368421054|20|{'min': 7, 'median': 10.0, 'max': 15}|
|2|baseline|rank3|80.00|66.45|13.55|0.00|27.957894736842107|20|{'min': 0, 'median': 0.0, 'max': 0}|
|2|baseline|rank5|80.00|67.05|12.95|0.00|27.742105263157896|20|{'min': 0, 'median': 0.0, 'max': 0}|
|2|baseline|ambiguous|40.00|0.00|0.00|40.00|56.42631578947368|20|{'min': 0, 'median': 0.0, 'max': 0}|
|2|baseline|unrepairable|80.00|67.00|13.00|0.00|29.00526315789474|20|{'min': 20, 'median': 20.0, 'max': 20}|
|2|baseline|background|5173.60|4517.25|37.55|0.00|464.13157894736844|20|None|
|2|experimental|rank2|100.00|100.00|0.00|0.00|0|1|{'min': 20, 'median': 20.0, 'max': 20}|
|2|experimental|rank3|80.00|65.90|14.10|0.00|26.626315789473683|20|{'min': 0, 'median': 0.0, 'max': 0}|
|2|experimental|rank5|80.00|67.10|12.90|0.00|27.57894736842105|20|{'min': 0, 'median': 0.0, 'max': 0}|
|2|experimental|ambiguous|40.00|0.00|0.00|40.00|56.05789473684211|20|{'min': 0, 'median': 0.0, 'max': 0}|
|2|experimental|unrepairable|80.00|66.40|13.60|0.00|27.86842105263158|20|{'min': 20, 'median': 20.0, 'max': 20}|
|2|experimental|background|5177.50|4513.30|39.60|0.00|454.94736842105266|20|None|
|3|baseline|rank2|89.50|79.00|10.50|0.00|20.31578947368421|20|{'min': 4, 'median': 9.5, 'max': 15}|
|3|baseline|rank3|86.30|72.60|13.70|0.00|27.973684210526315|20|{'min': 2, 'median': 6.0, 'max': 11}|
|3|baseline|rank5|80.00|64.55|15.45|0.00|31.121052631578948|20|{'min': 0, 'median': 0.0, 'max': 0}|
|3|baseline|ambiguous|60.00|0.00|0.00|60.00|71.5578947368421|20|{'min': 0, 'median': 0.0, 'max': 0}|
|3|baseline|unrepairable|80.00|64.60|15.40|0.00|31.6|20|{'min': 20, 'median': 20.0, 'max': 20}|
|3|baseline|background|5174.05|4515.80|39.30|0.00|461.89473684210526|20|None|
|3|experimental|rank2|100.00|100.00|0.00|0.00|0|1|{'min': 20, 'median': 20.0, 'max': 20}|
|3|experimental|rank3|100.00|100.00|0.00|0.00|0|1|{'min': 20, 'median': 20.0, 'max': 20}|
|3|experimental|rank5|80.00|64.80|15.20|0.00|31.352631578947367|20|{'min': 0, 'median': 0.0, 'max': 0}|
|3|experimental|ambiguous|60.00|0.00|0.00|60.00|72.41052631578947|20|{'min': 0, 'median': 0.0, 'max': 0}|
|3|experimental|unrepairable|80.00|65.05|14.95|0.00|33.23157894736842|20|{'min': 20, 'median': 20.0, 'max': 20}|
|3|experimental|background|5177.80|4515.40|40.25|0.00|456.5052631578947|20|None|
|5|baseline|rank2|90.00|80.00|10.00|0.00|20.231578947368423|20|{'min': 4, 'median': 11.0, 'max': 13}|
|5|baseline|rank3|86.25|72.50|13.75|0.00|27.647368421052633|20|{'min': 2, 'median': 5.5, 'max': 12}|
|5|baseline|rank5|83.45|66.90|16.55|0.00|35.578947368421055|20|{'min': 0, 'median': 3.5, 'max': 7}|
|5|baseline|ambiguous|100.00|0.00|0.00|100.00|80.2578947368421|20|{'min': 20, 'median': 20.0, 'max': 20}|
|5|baseline|unrepairable|80.00|63.70|16.30|0.00|34.45263157894737|20|{'min': 20, 'median': 20.0, 'max': 20}|
|5|baseline|background|5173.75|4513.80|38.85|0.00|460.42105263157896|20|None|
|5|experimental|rank2|100.00|100.00|0.00|0.00|0|1|{'min': 20, 'median': 20.0, 'max': 20}|
|5|experimental|rank3|100.00|100.00|0.00|0.00|0|1|{'min': 20, 'median': 20.0, 'max': 20}|
|5|experimental|rank5|100.00|100.00|0.00|0.00|0|1|{'min': 20, 'median': 20.0, 'max': 20}|
|5|experimental|ambiguous|100.00|0.00|0.00|100.00|80.34210526315789|20|{'min': 20, 'median': 20.0, 'max': 20}|
|5|experimental|unrepairable|80.00|63.90|16.10|0.00|32.48947368421052|20|{'min': 20, 'median': 20.0, 'max': 20}|
|5|experimental|background|5177.75|4511.45|40.55|0.00|455.30526315789473|20|None|

Combined baseline calibration validated 500 custom queries. All ambiguous candidate rows tied: **True**. If false, ambiguous edges all qualify but some query rows are not tied; see the full calibration matrix.

## Resources and local moves

|N/Binary|Wall s min/median/max|RSS bytes min/median/max|Moves by family|Replacement ranks >3|
|---|---|---|---|---:|
|1/baseline|{'min': 1.2952886669663712, 'median': 1.317196874995716, 'max': 1.4775937090162188}|{'min': 715866112, 'median': 724312064.0, 'max': 728383488}|{'rank2': 0, 'rank3': 0, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 0}|0|
|1/experimental|{'min': 1.3012314159423113, 'median': 1.3189959794981405, 'max': 1.6385020420420915}|{'min': 674316288, 'median': 683311104.0, 'max': 734117888}|{'rank2': 0, 'rank3': 0, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 0}|0|
|2/baseline|{'min': 1.2938613330479711, 'median': 1.3106965000042692, 'max': 1.3552089999429882}|{'min': 718012416, 'median': 721330176.0, 'max': 728023040}|{'rank2': 0, 'rank3': 0, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 0}|0|
|2/experimental|{'min': 1.2984871671069413, 'median': 1.3242870420217514, 'max': 1.3900043340399861}|{'min': 675627008, 'median': 684711936.0, 'max': 733233152}|{'rank2': 203, 'rank3': 0, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 74}|0|
|3/baseline|{'min': 1.2958697080612183, 'median': 1.317021312424913, 'max': 1.50218583305832}|{'min': 717553664, 'median': 723189760.0, 'max': 729219072}|{'rank2': 0, 'rank3': 0, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 0}|0|
|3/experimental|{'min': 1.3058747089235112, 'median': 1.3278615000308491, 'max': 1.4360781670548022}|{'min': 674283520, 'median': 684326912.0, 'max': 736985088}|{'rank2': 201, 'rank3': 283, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 73}|0|
|5/baseline|{'min': 1.2964552500052378, 'median': 1.309757457987871, 'max': 1.4852479160763323}|{'min': 710557696, 'median': 722198528.0, 'max': 730267648}|{'rank2': 0, 'rank3': 0, 'rank5': 0, 'ambiguous': 0, 'unrepairable': 0, 'background': 0}|0|
|5/experimental|{'min': 1.3016133329365402, 'median': 1.3327183960354887, 'max': 1.6603389999363571}|{'min': 677888000, 'median': 683540480.0, 'max': 739377152}|{'rank2': 185, 'rank3': 272, 'rank5': 311, 'ambiguous': 0, 'unrepairable': 0, 'background': 76}|311|

## Observed paired effects (descriptive, not significance)

```json
{
  "paired_binary": {
    "pairs": 80,
    "pairs_with_mapping_change": 80,
    "pairs_with_quality_or_coverage_change": 80,
    "cohort_mean_deltas": {
      "rank2": {
        "matches": 7.5625,
        "known_correct": 15.2,
        "incorrect": -7.6375,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "rank3": {
        "matches": 6.8625,
        "known_correct": 13.4,
        "incorrect": -6.5375,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "rank5": {
        "matches": 4.1375,
        "known_correct": 8.225,
        "incorrect": -4.0875,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "ambiguous": {
        "matches": 0,
        "known_correct": 0,
        "incorrect": 0,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "unrepairable": {
        "matches": 0,
        "known_correct": 0.1375,
        "incorrect": -0.1375,
        "allowed_ambiguous": 0,
        "matched_known_negative": -0.1375
      },
      "background": {
        "matches": 2.9125,
        "known_correct": -2.625,
        "incorrect": 1.075,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0.3
      }
    },
    "wall_ratio": {
      "min": 0.9098949696044304,
      "median": 1.0076244904084353,
      "max": 1.1214602172963293
    },
    "rss_ratio": {
      "min": 0.930162945624775,
      "median": 0.9476925571203421,
      "max": 1.0222632394494577
    }
  },
  "paired_width": {
    "pairs": 120,
    "pairs_with_mapping_change": 120,
    "pairs_with_quality_or_coverage_change": 120,
    "cohort_mean_deltas": {
      "rank2": {
        "matches": 14.958333333333334,
        "known_correct": 19.866666666666667,
        "incorrect": -4.908333333333333,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "rank3": {
        "matches": 8.758333333333333,
        "known_correct": 9.5,
        "incorrect": -0.7416666666666667,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "rank5": {
        "matches": 3.908333333333333,
        "known_correct": 2.2333333333333334,
        "incorrect": 1.675,
        "allowed_ambiguous": 0,
        "matched_known_negative": 0
      },
      "ambiguous": {
        "matches": 46.666666666666664,
        "known_correct": 0,
        "incorrect": 0,
        "allowed_ambiguous": 46.666666666666664,
        "matched_known_negative": 0
      },
      "unrepairable": {
        "matches": 0,
        "known_correct": -4.441666666666666,
        "incorrect": 4.441666666666666,
        "allowed_ambiguous": 0,
        "matched_known_negative": 4.441666666666666
      },
      "background": {
        "matches": 13.741666666666667,
        "known_correct": -1.5,
        "incorrect": 7.2,
        "allowed_ambiguous": 0,
        "matched_known_negative": 1.15
      }
    },
    "wall_ratio": {
      "min": 0.7989132045366831,
      "median": 1.0005797940703165,
      "max": 1.1663009255989498
    },
    "rss_ratio": {
      "min": 0.9253225625490856,
      "median": 0.9987452903951385,
      "max": 1.0910902155161941
    }
  }
}
```

measurements.json contains per-pair binary/width deltas and factorial interactions, full normalized cohort stability, quality, resource and rank audits.
A mapping change alone is not proof of a beneficial repair; inspect known-correct/incorrect/coverage deltas and WAL moves. No global determinism claim.
