# ctgan on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** minmax, clip=True
**Training time:** 220.6s

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
| RandomForest | 0.3864 | 0.8967 |
| GradientBoosting | 0.3938 | 0.8913 |
| Ridge | 0.3958 | 0.8898 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7809 | 0.5358 |
| GradientBoosting | 0.7026 | 0.6243 |
| Ridge | 0.5770 | 0.7445 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7461 | 100.0% |
| Replacement | 0.3920 | 52.5% |
| Augmentation | 0.6868 | 92.1% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0446 |
| avg_jsd | 0.0306 |
| correlation_frobenius | 1.1863 |

**Numerical:** 8 columns, avg Wasserstein=0.0446, avg JSD=0.0306, KS pass rate (p>0.05)=0%
