# our_tabddpm on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** minmax, clip=True
**Training time:** 155.5s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7892 | 0.8003 |
| GradientBoosting | 0.8021 | 0.8206 |
| LogisticRegression | 0.7224 | 0.6839 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.5193 | 0.3090 |
| GradientBoosting | 0.4602 | 0.3459 |
| LogisticRegression | 0.5270 | 0.4230 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7918 | 0.8027 |
| GradientBoosting | 0.7686 | 0.7761 |
| LogisticRegression | 0.7044 | 0.6133 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7712 | 100.0% |
| Replacement | 0.5021 | 65.1% |
| Augmentation | 0.7549 | 97.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 82.7348 |
| avg_jsd | 0.1905 |
| correlation_frobenius | 20.0126 |
| avg_cat_freq_diff | 0.1271 |

**Numerical:** 24 columns, avg Wasserstein=82.7348, avg JSD=0.1905, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.1271
