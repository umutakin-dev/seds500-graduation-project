# vanilla_tabddpm on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** quantile, clip=False
**Training time:** 464.6s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8042 | 0.5065 |
| GradientBoosting | 0.7756 | 0.5422 |
| Ridge | 0.5908 | 0.7323 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.6986 | 0.6284 |
| GradientBoosting | 0.6786 | 0.6489 |
| Ridge | 0.5283 | 0.7862 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7777 | 0.5397 |
| GradientBoosting | 0.7313 | 0.5934 |
| Ridge | 0.5452 | 0.7720 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7236 | 100.0% |
| Replacement | 0.6352 | 87.8% |
| Augmentation | 0.6848 | 94.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 5.8566 |
| avg_jsd | 0.0419 |
| correlation_frobenius | 6.5786 |

**Numerical:** 8 columns, avg Wasserstein=5.8566, avg JSD=0.0419, KS pass rate (p>0.05)=0%

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5032** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9916 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

