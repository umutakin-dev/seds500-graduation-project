# smogn on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

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
| RandomForest | 0.7763 | 0.7862 |
| GradientBoosting | 0.7661 | 0.7750 |
| LogisticRegression | 0.7172 | 0.7018 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7763 | 0.7912 |
| GradientBoosting | 0.7815 | 0.7964 |
| LogisticRegression | 0.7147 | 0.6897 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7712 | 100.0% |
| Replacement | 0.7532 | 97.7% |
| Augmentation | 0.7575 | 98.2% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0130 |
| avg_jsd | 0.0112 |
| correlation_frobenius | 0.4541 |
| avg_cat_freq_diff | 0.0369 |

**Numerical:** 24 columns, avg Wasserstein=0.0130, avg JSD=0.0112, KS pass rate (p>0.05)=62%
**Categorical:** 3 columns, avg freq L1 diff=0.0369
