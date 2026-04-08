# ctgan on Supply Chain Pricing

**Dataset:** Supply Chain Pricing (D8)
**Task:** regression
**Dimensions:** 6 num + 18 cat = 9605 total
**Samples:** 4958 train / 1240 test
**Preprocessing:** minmax, clip=True
**Training time:** 603.7s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.4936 | 9880.2619 |
| GradientBoosting | 0.4100 | 10664.5131 |
| Ridge | 0.3697 | 11022.8570 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.1925 | 12477.1671 |
| GradientBoosting | 0.1648 | 12688.8565 |
| Ridge | -0.2813 | 15716.4943 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.4690 | 10117.7311 |
| GradientBoosting | 0.3396 | 11283.0859 |
| Ridge | 0.2727 | 11841.1763 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.4245 | 100.0% |
| Replacement | 0.0253 | 6.0% |
| Augmentation | 0.3604 | 84.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0525 |
| avg_jsd | 0.0521 |
| correlation_frobenius | 1.2570 |
| avg_cat_freq_diff | 0.9903 |

**Numerical:** 6 columns, avg Wasserstein=0.0525, avg JSD=0.0521, KS pass rate (p>0.05)=0%
**Categorical:** 18 columns, avg freq L1 diff=0.9903

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5093** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0014 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

