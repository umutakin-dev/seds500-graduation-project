# vanilla_tabddpm on California Housing

**Dataset:** California Housing (D2)
**Task:** regression
**Dimensions:** 8 num + 0 cat = 8 total
**Samples:** 16512 train / 4128 test
**Preprocessing:** quantile, clip=False
**Training time:** 470.5s

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
| RandomForest | 0.6940 | 0.6332 |
| GradientBoosting | 0.6518 | 0.6754 |
| Ridge | 0.4075 | 0.8812 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.7847 | 0.5311 |
| GradientBoosting | 0.7160 | 0.6100 |
| Ridge | 0.4908 | 0.8169 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7236 | 100.0% |
| Replacement | 0.5844 | 80.8% |
| Augmentation | 0.6639 | 91.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 6.2512 |
| avg_jsd | 0.0540 |
| correlation_frobenius | 5.9300 |

**Numerical:** 8 columns, avg Wasserstein=6.2512, avg JSD=0.0540, KS pass rate (p>0.05)=0%
