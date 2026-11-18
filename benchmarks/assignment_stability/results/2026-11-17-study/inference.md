# Assignment stability study

## Predeclared method

U_b = sum(i<j) Hamming(A_bi,A_bj) / choose(n,2) / Bcount; primary = U_baseline - U_experimental (positive is improvement).

Resample n paired whole-run indices with replacement. With counts wi, sum(i<j) wi*wj*dij / choose(n,2) / Bcount; duplicate-index diagonals are zero. Never divide by the sum of off-diagonal weights. Basic 95% CI = [2*observed - q.975, 2*observed - q.025], linear-interpolation quantiles. Same method for relative reduction difference/baseline when baseline > 0; if any bootstrap baseline is zero its relative interval is undefined (no dropped replicates). Paired quality means use sum(wi*(experimental_i-baseline_i))/n with the identical counts.

## Primary results

|Mode|Paired runs|Baseline changed B|Experimental changed B|Difference (pp)|95% basic CI (pp)|Decision|
|---|---:|---:|---:|---:|---|---|
|shuffle|100|462.3784|455.7077|0.066707|[0.052193939393939334, 0.08257808080808092]|positive-CI evidence on this corpus|
|fixed|50|462.8106|455.8645|0.069461|[0.04610591836734784, 0.09520102040816365]|positive-CI evidence on this corpus|

### shuffle: secondary quality and descriptive performance

```json
{
  "quality_paired_effects": {
    "tp": {
      "experimental_minus_baseline_mean": -0.02,
      "basic_95_ci": [
        -2.570249999999996,
        2.64
      ]
    },
    "known_positives_assigned_wrong_a": {
      "experimental_minus_baseline_mean": 0.93,
      "basic_95_ci": [
        0.30000000000000004,
        1.57
      ]
    },
    "known_unmatched_b_automatched": {
      "experimental_minus_baseline_mean": 0.08,
      "basic_95_ci": [
        -0.5299999999999999,
        0.6900000000000001
      ]
    },
    "ambiguous_b_automatched": {
      "experimental_minus_baseline_mean": 2.76,
      "basic_95_ci": [
        0.1299999999999999,
        5.18
      ]
    },
    "strict_precision": {
      "experimental_minus_baseline_mean": -0.0006357587382315222,
      "basic_95_ci": [
        -0.001135164693672254,
        -0.0001193705868598959
      ]
    },
    "strict_recall": {
      "experimental_minus_baseline_mean": -2.8571428571377844e-06,
      "basic_95_ci": [
        -0.0003671785714285639,
        0.00037714285714286815
      ]
    },
    "matches": {
      "experimental_minus_baseline_mean": 3.75,
      "basic_95_ci": [
        3.4699999999999998,
        4.029999999999999
      ]
    }
  },
  "by_binary": {
    "baseline": {
      "quality": {
        "tp": {
          "mean": 4512.59,
          "median": 4513.0
        },
        "known_positives_assigned_wrong_a": {
          "mean": 11.75,
          "median": 12.0
        },
        "known_unmatched_b_automatched": {
          "mean": 27.11,
          "median": 27.0
        },
        "ambiguous_b_automatched": {
          "mean": 621.49,
          "median": 622.0
        },
        "strict_precision": {
          "mean": 0.8723452958707297,
          "median": 0.8724021254389314
        },
        "strict_recall": {
          "mean": 0.6446557142857143,
          "median": 0.6447142857142857
        },
        "matches": {
          "mean": 5172.94,
          "median": 5173.0
        }
      },
      "distinct_whole_assignment_outcomes": 100,
      "ever_variable_b": 924,
      "pairwise_changed_b": {
        "mean": 462.37838383838385,
        "median": 463.0
      },
      "cohort_variation": {
        "ambiguous": {
          "b_count": 2000,
          "ever_variable_b": 431
        },
        "matched": {
          "b_count": 7000,
          "ever_variable_b": 468
        },
        "unmatched": {
          "b_count": 1000,
          "ever_variable_b": 25
        }
      },
      "wall_seconds_median": 0.7000237085158005,
      "maximum_rss_bytes_median": 691232768.0,
      "local_moves": {
        "mean": 0,
        "median": 0.0
      }
    },
    "experimental": {
      "quality": {
        "tp": {
          "mean": 4512.57,
          "median": 4512.0
        },
        "known_positives_assigned_wrong_a": {
          "mean": 12.68,
          "median": 13.0
        },
        "known_unmatched_b_automatched": {
          "mean": 27.19,
          "median": 27.0
        },
        "ambiguous_b_automatched": {
          "mean": 624.25,
          "median": 625.0
        },
        "strict_precision": {
          "mean": 0.8717095371324982,
          "median": 0.8717156105100463
        },
        "strict_recall": {
          "mean": 0.6446528571428571,
          "median": 0.6445714285714286
        },
        "matches": {
          "mean": 5176.69,
          "median": 5177.0
        }
      },
      "distinct_whole_assignment_outcomes": 100,
      "ever_variable_b": 910,
      "pairwise_changed_b": {
        "mean": 455.7076767676768,
        "median": 456.0
      },
      "cohort_variation": {
        "ambiguous": {
          "b_count": 2000,
          "ever_variable_b": 426
        },
        "matched": {
          "b_count": 7000,
          "ever_variable_b": 459
        },
        "unmatched": {
          "b_count": 1000,
          "ever_variable_b": 25
        }
      },
      "wall_seconds_median": 0.7052506874897517,
      "maximum_rss_bytes_median": 646324224.0,
      "local_moves": {
        "mean": 4.01,
        "median": 4.0
      }
    }
  },
  "median_paired_wall_ratio": 1.0094768622161117
}
```


### fixed: secondary quality and descriptive performance

```json
{
  "quality_paired_effects": {
    "tp": {
      "experimental_minus_baseline_mean": 0.48,
      "basic_95_ci": [
        -3.6204999999999927,
        4.3004999999999995
      ]
    },
    "known_positives_assigned_wrong_a": {
      "experimental_minus_baseline_mean": 0.9,
      "basic_95_ci": [
        -0.020000000000000018,
        1.84
      ]
    },
    "known_unmatched_b_automatched": {
      "experimental_minus_baseline_mean": -0.96,
      "basic_95_ci": [
        -1.94,
        0.02049999999999974
      ]
    },
    "ambiguous_b_automatched": {
      "experimental_minus_baseline_mean": 3.7,
      "basic_95_ci": [
        0.0,
        7.62
      ]
    },
    "strict_precision": {
      "experimental_minus_baseline_mean": -0.0006016569373666591,
      "basic_95_ci": [
        -0.0013861578333762645,
        0.00014536480804477957
      ]
    },
    "strict_recall": {
      "experimental_minus_baseline_mean": 6.857142857142895e-05,
      "basic_95_ci": [
        -0.0005172142857142965,
        0.0006143571428571465
      ]
    },
    "matches": {
      "experimental_minus_baseline_mean": 4.12,
      "basic_95_ci": [
        3.62,
        4.62
      ]
    }
  },
  "by_binary": {
    "baseline": {
      "quality": {
        "tp": {
          "mean": 4513.44,
          "median": 4513.5
        },
        "known_positives_assigned_wrong_a": {
          "mean": 11.7,
          "median": 12.0
        },
        "known_unmatched_b_automatched": {
          "mean": 28.02,
          "median": 28.0
        },
        "ambiguous_b_automatched": {
          "mean": 619.44,
          "median": 619.0
        },
        "strict_precision": {
          "mean": 0.8725669249162327,
          "median": 0.8724988089967114
        },
        "strict_recall": {
          "mean": 0.6447771428571428,
          "median": 0.6447857142857143
        },
        "matches": {
          "mean": 5172.6,
          "median": 5173.0
        }
      },
      "distinct_whole_assignment_outcomes": 50,
      "ever_variable_b": 924,
      "pairwise_changed_b": {
        "mean": 462.810612244898,
        "median": 463
      },
      "cohort_variation": {
        "ambiguous": {
          "b_count": 2000,
          "ever_variable_b": 431
        },
        "matched": {
          "b_count": 7000,
          "ever_variable_b": 468
        },
        "unmatched": {
          "b_count": 1000,
          "ever_variable_b": 25
        }
      },
      "wall_seconds_median": 0.6823013544781134,
      "maximum_rss_bytes_median": 691109888.0,
      "local_moves": {
        "mean": 0,
        "median": 0.0
      }
    },
    "experimental": {
      "quality": {
        "tp": {
          "mean": 4513.92,
          "median": 4514.0
        },
        "known_positives_assigned_wrong_a": {
          "mean": 12.6,
          "median": 12.5
        },
        "known_unmatched_b_automatched": {
          "mean": 27.06,
          "median": 27.0
        },
        "ambiguous_b_automatched": {
          "mean": 623.14,
          "median": 623.5
        },
        "strict_precision": {
          "mean": 0.871965267978866,
          "median": 0.8719211810710732
        },
        "strict_recall": {
          "mean": 0.6448457142857142,
          "median": 0.6448571428571429
        },
        "matches": {
          "mean": 5176.72,
          "median": 5177.0
        }
      },
      "distinct_whole_assignment_outcomes": 50,
      "ever_variable_b": 910,
      "pairwise_changed_b": {
        "mean": 455.86448979591836,
        "median": 455
      },
      "cohort_variation": {
        "ambiguous": {
          "b_count": 2000,
          "ever_variable_b": 426
        },
        "matched": {
          "b_count": 7000,
          "ever_variable_b": 459
        },
        "unmatched": {
          "b_count": 1000,
          "ever_variable_b": 25
        }
      },
      "wall_seconds_median": 0.6903314375085756,
      "maximum_rss_bytes_median": 646586368.0,
      "local_moves": {
        "mean": 4.04,
        "median": 4.0
      }
    }
  },
  "median_paired_wall_ratio": 1.012223458961145
}
```

## Caveats

- Inference is conditional on this fixed corpus, cache, binaries and thread settings, assuming independent fresh processes across paired runs.
- Shuffled CSV order is not necessarily processing order (HashMap iteration and scheduling can vary). Fixed mode repeats identical B input.
- Whole paired runs, not individual B records or distance pairs, are resampled. Shared indices preserve input and temporal pairing covariance.
- Each mode answers a separate question with its own 95% interval; modes are not pooled and no familywise claim is made.
- A positive difference CI supports stability improvement on this corpus, not determinism. An interval containing zero is inconclusive.
- The strict-ID oracle excludes the 2000 ambiguous B records. Non-oracle pairs, especially ambiguous assignments, are not all proven errors.
- Quality intervals are secondary descriptive uncertainty summaries, not multiplicity-adjusted confirmatory claims. Performance and ever-variable B are descriptive only; no runs are excluded.

Full normalization, seed, replicate arrays, quality, stable pairs and distance matrices: `analysis.json`.
