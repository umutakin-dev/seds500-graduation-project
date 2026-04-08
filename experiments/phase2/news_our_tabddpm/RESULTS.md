# our_tabddpm on Online News Popularity

**Dataset:** Online News Popularity (D9)
**Task:** regression
**Dimensions:** 44 num + 14 cat = 72 total
**Samples:** 31715 train / 7929 test
**Preprocessing:** minmax, clip=True
**Training time:** 3574.7s

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
| RandomForest | -0.2973 | 12511.9755 |
| GradientBoosting | -265.5863 | 179358.4541 |
| Ridge | -4686.9160 | 752129.9169 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0513 | 11263.3630 |
| GradientBoosting | -1.5776 | 17636.4136 |
| Ridge | -199.1725 | 155419.3650 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | -0.0231 | 100.0% |
| Replacement | -1650.9332 | 7143494.0% |
| Augmentation | -66.9338 | 289618.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 567.6021 |
| avg_jsd | 0.0931 |
| correlation_frobenius | 17.7459 |
| avg_cat_freq_diff | 0.2480 |

**Numerical:** 44 columns, avg Wasserstein=567.6021, avg JSD=0.0931, KS pass rate (p>0.05)=0%
**Categorical:** 14 columns, avg freq L1 diff=0.2480

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5015** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0009 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

