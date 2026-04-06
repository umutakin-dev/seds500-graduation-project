# our_tabddpm on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** minmax, clip=True
**Training time:** 2463.4s

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
| RandomForest | 0.8930 | 0.6167 |
| GradientBoosting | 0.8967 | 0.6297 |
| LogisticRegression | 0.8978 | 0.6451 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9040 | 0.7054 |
| GradientBoosting | 0.9019 | 0.6861 |
| LogisticRegression | 0.8995 | 0.6757 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9042 | 100.0% |
| Replacement | 0.8958 | 99.1% |
| Augmentation | 0.9018 | 99.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.5213 |
| avg_jsd | 0.0053 |
| correlation_frobenius | 3.6559 |
| avg_cat_freq_diff | 0.0269 |

**Numerical:** 7 columns, avg Wasserstein=0.5213, avg JSD=0.0053, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.0269
