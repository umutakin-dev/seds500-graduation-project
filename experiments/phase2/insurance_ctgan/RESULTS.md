# ctgan on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** minmax, clip=True
**Training time:** 27.9s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8630 | 4611.7939 |
| GradientBoosting | 0.8803 | 4311.0199 |
| Ridge | 0.7822 | 5815.2469 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.3615 | 9956.1150 |
| GradientBoosting | 0.3140 | 10319.5989 |
| Ridge | 0.4490 | 9248.4909 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8219 | 5258.4987 |
| GradientBoosting | 0.7918 | 5685.0381 |
| Ridge | 0.6741 | 7113.1534 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8418 | 100.0% |
| Replacement | 0.3749 | 44.5% |
| Augmentation | 0.7626 | 90.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1535 |
| avg_jsd | 0.1826 |
| correlation_frobenius | 0.2924 |
| avg_cat_freq_diff | 0.0773 |

**Numerical:** 3 columns, avg Wasserstein=0.1535, avg JSD=0.1826, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.0773
