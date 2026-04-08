# our_tabddpm on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** minmax, clip=True
**Training time:** 93.4s

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
| RandomForest | 0.7585 | 6122.6546 |
| GradientBoosting | 0.7818 | 5820.5231 |
| Ridge | 0.7599 | 6105.3216 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8618 | 4632.1963 |
| GradientBoosting | 0.8644 | 4587.7986 |
| Ridge | 0.7750 | 5910.4196 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8418 | 100.0% |
| Replacement | 0.7667 | 91.1% |
| Augmentation | 0.8337 | 99.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0798 |
| avg_jsd | 0.1833 |
| correlation_frobenius | 0.0646 |
| avg_cat_freq_diff | 0.0280 |

**Numerical:** 3 columns, avg Wasserstein=0.0798, avg JSD=0.1833, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.0280

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5098** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9730 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

