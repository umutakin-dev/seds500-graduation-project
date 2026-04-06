# vanilla_tabddpm on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** quantile, clip=False
**Training time:** 94.0s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8608 | 4648.4484 |
| GradientBoosting | 0.8789 | 4335.2532 |
| Ridge | 0.7748 | 5912.8294 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7976 | 5605.5705 |
| GradientBoosting | 0.8021 | 5542.3070 |
| Ridge | 0.7455 | 6285.2567 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8633 | 4606.7993 |
| GradientBoosting | 0.8617 | 4634.2116 |
| Ridge | 0.7650 | 6039.6354 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8382 | 100.0% |
| Replacement | 0.7818 | 93.3% |
| Augmentation | 0.8300 | 99.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 1.0777 |
| avg_jsd | 0.1910 |
| correlation_frobenius | 0.1660 |
| avg_cat_freq_diff | 0.1090 |

**Numerical:** 3 columns, avg Wasserstein=1.0777, avg JSD=0.1910, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.1090
