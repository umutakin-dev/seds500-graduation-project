# ctgan on Online News Popularity

**Dataset:** Online News Popularity (D9)
**Task:** regression
**Dimensions:** 44 num + 14 cat = 72 total
**Samples:** 31715 train / 7929 test
**Preprocessing:** minmax, clip=True
**Training time:** 1776.0s

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
| RandomForest | -0.0323 | 11161.1105 |
| GradientBoosting | -0.0268 | 11131.2744 |
| Ridge | 0.0172 | 10889.9800 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0207 | 11098.0987 |
| GradientBoosting | -0.0410 | 11207.8706 |
| Ridge | 0.0258 | 10842.6713 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | -0.0231 | 100.0% |
| Replacement | -0.0140 | 60.4% |
| Augmentation | -0.0120 | 51.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0495 |
| avg_jsd | 0.0504 |
| correlation_frobenius | 4.1735 |
| avg_cat_freq_diff | 0.0656 |

**Numerical:** 44 columns, avg Wasserstein=0.0495, avg JSD=0.0504, KS pass rate (p>0.05)=0%
**Categorical:** 14 columns, avg freq L1 diff=0.0656

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4977** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0008 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

