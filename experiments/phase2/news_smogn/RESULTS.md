# smogn on Online News Popularity

**Dataset:** Online News Popularity (D9)
**Task:** regression
**Dimensions:** 44 num + 14 cat = 72 total
**Samples:** 31715 train / 7929 test
**Preprocessing:** minmax, clip=True
**Training time:** 14123.6s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0521 | 11267.3343 |
| GradientBoosting | -0.0448 | 11228.2499 |
| Ridge | 0.0275 | 10833.0801 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0568 | 11292.9959 |
| GradientBoosting | -0.0160 | 11072.4081 |
| Ridge | -0.0025 | 10998.7228 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.1463 | 11761.0339 |
| GradientBoosting | -0.0461 | 11235.3384 |
| Ridge | 0.0159 | 10897.4724 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | -0.0231 | 100.0% |
| Replacement | -0.0251 | 108.6% |
| Augmentation | -0.0588 | 254.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1953 |
| avg_jsd | 0.1027 |
| correlation_frobenius | 12.2863 |
| avg_cat_freq_diff | 0.0219 |

**Numerical:** 44 columns, avg Wasserstein=0.1953, avg JSD=0.1027, KS pass rate (p>0.05)=0%
**Categorical:** 14 columns, avg freq L1 diff=0.0219

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.9218** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.1583 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

