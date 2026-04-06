# ctgan on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** minmax, clip=True
**Training time:** 732.7s

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
| RandomForest | 0.8809 | 0.6633 |
| GradientBoosting | 0.8791 | 0.6701 |
| LogisticRegression | 0.8703 | 0.6733 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9041 | 0.7117 |
| GradientBoosting | 0.8990 | 0.6924 |
| LogisticRegression | 0.8962 | 0.6801 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9042 | 100.0% |
| Replacement | 0.8768 | 97.0% |
| Augmentation | 0.8998 | 99.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0534 |
| avg_jsd | 0.0465 |
| correlation_frobenius | 0.3219 |
| avg_cat_freq_diff | 0.2536 |

**Numerical:** 7 columns, avg Wasserstein=0.0534, avg JSD=0.0465, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.2536
