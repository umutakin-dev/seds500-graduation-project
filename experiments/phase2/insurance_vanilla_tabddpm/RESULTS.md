# vanilla_tabddpm on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** quantile, clip=False
**Training time:** 94.3s

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
| RandomForest | 0.8001 | 5571.0777 |
| GradientBoosting | 0.7746 | 5915.4451 |
| Ridge | 0.5211 | 8622.4307 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8654 | 4570.8999 |
| GradientBoosting | 0.8558 | 4731.5742 |
| Ridge | 0.5921 | 7957.5381 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8382 | 100.0% |
| Replacement | 0.6986 | 83.3% |
| Augmentation | 0.7711 | 92.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 7.9078 |
| avg_jsd | 0.2001 |
| correlation_frobenius | 0.9255 |
| avg_cat_freq_diff | 0.2698 |

**Numerical:** 3 columns, avg Wasserstein=7.9078, avg JSD=0.2001, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.2698

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5145** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9834 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

