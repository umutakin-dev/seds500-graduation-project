# ctgan on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** minmax, clip=True
**Training time:** 29.9s

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
| RandomForest | 0.1953 | 11177.2215 |
| GradientBoosting | 0.3643 | 9934.7377 |
| Ridge | 0.4106 | 9565.8313 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8358 | 5049.3873 |
| GradientBoosting | 0.7805 | 5837.8337 |
| Ridge | 0.6769 | 7082.0356 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8418 | 100.0% |
| Replacement | 0.3234 | 38.4% |
| Augmentation | 0.7644 | 90.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1678 |
| avg_jsd | 0.1600 |
| correlation_frobenius | 0.2777 |
| avg_cat_freq_diff | 0.1595 |

**Numerical:** 3 columns, avg Wasserstein=0.1678, avg JSD=0.1600, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.1595

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5337** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0931 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

