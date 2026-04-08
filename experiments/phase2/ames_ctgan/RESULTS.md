# ctgan on Ames Housing

**Dataset:** Ames Housing (D11)
**Task:** regression
**Dimensions:** 32 num + 47 cat = 361 total
**Samples:** 2344 train / 586 test
**Preprocessing:** minmax, clip=True
**Training time:** 246.2s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8961 | 28861.7414 |
| GradientBoosting | 0.9167 | 25847.1748 |
| Ridge | 0.9023 | 27993.9879 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.7981 | 120067.1992 |
| GradientBoosting | -1.6501 | 145763.8426 |
| Ridge | -3.0681 | 180598.8318 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8698 | 32310.5010 |
| GradientBoosting | 0.8624 | 33212.5608 |
| Ridge | 0.8188 | 38111.0798 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9050 | 100.0% |
| Replacement | -1.8387 | -203.2% |
| Augmentation | 0.8503 | 94.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1442 |
| avg_jsd | 0.1810 |
| correlation_frobenius | 4.6253 |
| avg_cat_freq_diff | 0.3327 |

**Numerical:** 32 columns, avg Wasserstein=0.1442, avg JSD=0.1810, KS pass rate (p>0.05)=0%
**Categorical:** 47 columns, avg freq L1 diff=0.3327

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4766** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0018 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

