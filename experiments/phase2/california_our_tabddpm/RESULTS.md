# our_tabddpm on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** minmax, clip=True
**Training time:** 464.5s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8040 | 0.5068 |
| GradientBoosting | 0.7772 | 0.5404 |
| Ridge | 0.6572 | 0.6702 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7012 | 0.6258 |
| GradientBoosting | 0.7092 | 0.6173 |
| Ridge | 0.6525 | 0.6748 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7805 | 0.5363 |
| GradientBoosting | 0.7503 | 0.5720 |
| Ridge | 0.6532 | 0.6741 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7461 | 100.0% |
| Replacement | 0.6876 | 92.2% |
| Augmentation | 0.7280 | 97.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0568 |
| avg_jsd | 0.0216 |
| correlation_frobenius | 1.4335 |

**Numerical:** 8 columns, avg Wasserstein=0.0568, avg JSD=0.0216, KS pass rate (p>0.05)=0%

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5106** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0098 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

