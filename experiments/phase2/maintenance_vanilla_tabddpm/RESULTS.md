# vanilla_tabddpm on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** quantile, clip=False
**Training time:** 639.1s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9990 | 0.9923 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9940 | 0.9501 |
| GradientBoosting | 0.9950 | 0.9590 |
| LogisticRegression | 0.9790 | 0.7712 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9920 | 0.9313 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9893 | 99.0% |
| Augmentation | 0.9967 | 99.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 27.6489 |
| avg_jsd | 0.0100 |
| correlation_frobenius | 5.4693 |
| avg_cat_freq_diff | 0.0057 |

**Numerical:** 5 columns, avg Wasserstein=27.6489, avg JSD=0.0100, KS pass rate (p>0.05)=0%
**Categorical:** 6 columns, avg freq L1 diff=0.0057
