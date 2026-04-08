# ctgan on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** minmax, clip=True
**Training time:** 218.3s

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
| RandomForest | 0.3262 | 0.9396 |
| GradientBoosting | 0.3740 | 0.9057 |
| Ridge | 0.3727 | 0.9067 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7796 | 0.5375 |
| GradientBoosting | 0.7086 | 0.6179 |
| Ridge | 0.5700 | 0.7506 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7461 | 100.0% |
| Replacement | 0.3577 | 47.9% |
| Augmentation | 0.6861 | 92.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0666 |
| avg_jsd | 0.0358 |
| correlation_frobenius | 1.3032 |

**Numerical:** 8 columns, avg Wasserstein=0.0666, avg JSD=0.0358, KS pass rate (p>0.05)=0%

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4932** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0089 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

