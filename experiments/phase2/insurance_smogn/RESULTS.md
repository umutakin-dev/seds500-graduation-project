# smogn on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** minmax, clip=True
**Training time:** 3.0s

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
| RandomForest | 0.8800 | 4316.8187 |
| GradientBoosting | 0.8817 | 4284.8160 |
| Ridge | 0.7819 | 5818.3767 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8720 | 4457.3201 |
| GradientBoosting | 0.8795 | 4325.7297 |
| Ridge | 0.7822 | 5815.5725 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8418 | 100.0% |
| Replacement | 0.8479 | 100.7% |
| Augmentation | 0.8445 | 100.3% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0208 |
| avg_jsd | 0.1213 |
| correlation_frobenius | 0.0121 |
| avg_cat_freq_diff | 0.0000 |

**Numerical:** 3 columns, avg Wasserstein=0.0208, avg JSD=0.1213, KS pass rate (p>0.05)=67%
**Categorical:** 3 columns, avg freq L1 diff=0.0000
