# ctgan on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** minmax, clip=True
**Training time:** 141.5s

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
| RandomForest | 0.9970 | 0.9761 |
| GradientBoosting | 0.9950 | 0.9602 |
| LogisticRegression | 0.9980 | 0.9843 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9970 | 0.9761 |
| GradientBoosting | 0.9975 | 0.9803 |
| LogisticRegression | 0.9980 | 0.9843 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9967 | 99.8% |
| Augmentation | 0.9975 | 99.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1255 |
| avg_jsd | 0.0315 |
| correlation_frobenius | 1.1420 |
| avg_cat_freq_diff | 0.1402 |

**Numerical:** 5 columns, avg Wasserstein=0.1255, avg JSD=0.0315, KS pass rate (p>0.05)=0%
**Categorical:** 6 columns, avg freq L1 diff=0.1402
