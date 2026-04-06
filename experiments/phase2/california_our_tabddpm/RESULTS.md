# our_tabddpm on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** minmax, clip=True
**Training time:** 474.3s

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
| RandomForest | 0.7043 | 0.6225 |
| GradientBoosting | 0.7089 | 0.6176 |
| Ridge | 0.6432 | 0.6837 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7779 | 0.5395 |
| GradientBoosting | 0.7520 | 0.5700 |
| Ridge | 0.6481 | 0.6791 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7461 | 100.0% |
| Replacement | 0.6855 | 91.9% |
| Augmentation | 0.7260 | 97.3% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0838 |
| avg_jsd | 0.0121 |
| correlation_frobenius | 4.0147 |

**Numerical:** 8 columns, avg Wasserstein=0.0838, avg JSD=0.0121, KS pass rate (p>0.05)=0%
