# vanilla_tabddpm on Ames Housing

**Dataset:** Ames Housing (D11)
**Task:** regression
**Dimensions:** 32 num + 47 cat = 361 total
**Samples:** 2344 train / 586 test
**Preprocessing:** quantile, clip=False
**Training time:** 1341.3s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8941 | 29143.4130 |
| GradientBoosting | 0.9185 | 25557.5346 |
| Ridge | 0.9004 | 28254.6557 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -14431468451.1662 | 10756625401.5200 |
| GradientBoosting | -6198258437.7563 | 7049456299.6647 |
| Ridge | -876652544.0000 | 2651151427.4561 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8901 | 29677.9075 |
| GradientBoosting | -3757698.8280 | 173572862.1664 |
| Ridge | -17244228.0000 | 371828486.2506 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9043 | 100.0% |
| Replacement | -7168793144.3075 | -792709069354.2% |
| Augmentation | -7000641.9793 | -774115290.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 49713.1734 |
| avg_jsd | 0.5839 |
| correlation_frobenius | 6.6064 |
| avg_cat_freq_diff | 0.5369 |

**Numerical:** 32 columns, avg Wasserstein=49713.1734, avg JSD=0.5839, KS pass rate (p>0.05)=0%
**Categorical:** 47 columns, avg freq L1 diff=0.5369

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4866** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0000 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

