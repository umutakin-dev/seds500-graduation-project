# smogn on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9049 | 0.7211 |
| GradientBoosting | 0.9061 | 0.7277 |
| LogisticRegression | 0.9017 | 0.7047 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9074 | 0.7276 |
| GradientBoosting | 0.9073 | 0.7314 |
| LogisticRegression | 0.9019 | 0.7079 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9071 | 0.7279 |
| GradientBoosting | 0.9068 | 0.7310 |
| LogisticRegression | 0.9015 | 0.7050 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9042 | 100.0% |
| Replacement | 0.9056 | 100.1% |
| Augmentation | 0.9051 | 100.1% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0072 |
| avg_jsd | 0.0770 |
| correlation_frobenius | 0.0235 |
| avg_cat_freq_diff | 0.0053 |

**Numerical:** 7 columns, avg Wasserstein=0.0072, avg JSD=0.0770, KS pass rate (p>0.05)=14%
**Categorical:** 9 columns, avg freq L1 diff=0.0053
