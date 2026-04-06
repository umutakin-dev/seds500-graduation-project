# vanilla_tabddpm on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** quantile, clip=False
**Training time:** 155.1s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7866 | 0.7936 |
| GradientBoosting | 0.8072 | 0.8224 |
| LogisticRegression | 0.7121 | 0.7144 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.2005 | 0.0477 |
| GradientBoosting | 0.2031 | 0.0586 |
| LogisticRegression | 0.2519 | 0.1568 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7892 | 0.7920 |
| GradientBoosting | 0.7841 | 0.7882 |
| LogisticRegression | 0.5604 | 0.4361 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7686 | 100.0% |
| Replacement | 0.2185 | 28.4% |
| Augmentation | 0.7112 | 92.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 1371.6743 |
| avg_jsd | 0.4444 |
| correlation_frobenius | 9.1143 |
| avg_cat_freq_diff | 0.2822 |

**Numerical:** 24 columns, avg Wasserstein=1371.6743, avg JSD=0.4444, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.2822
