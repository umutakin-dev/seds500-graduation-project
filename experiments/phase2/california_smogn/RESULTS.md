# smogn on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** minmax, clip=True
**Training time:** 552.6s

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
| RandomForest | 0.7863 | 0.5291 |
| GradientBoosting | 0.7375 | 0.5865 |
| Ridge | 0.5776 | 0.7440 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7962 | 0.5168 |
| GradientBoosting | 0.7546 | 0.5671 |
| Ridge | 0.5837 | 0.7386 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7461 | 100.0% |
| Replacement | 0.7005 | 93.9% |
| Augmentation | 0.7115 | 95.4% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.2158 |
| avg_jsd | 0.1185 |
| correlation_frobenius | 2.4109 |

**Numerical:** 8 columns, avg Wasserstein=0.2158, avg JSD=0.1185, KS pass rate (p>0.05)=0%
