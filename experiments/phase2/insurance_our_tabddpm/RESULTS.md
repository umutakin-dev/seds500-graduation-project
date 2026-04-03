# our_tabddpm on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** minmax, clip=True
**Training time:** 2.9s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8631 | 4609.9777 |
| GradientBoosting | 0.8803 | 4311.0199 |
| Ridge | 0.7822 | 5815.2472 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -9873457540.6759 | 1238079863.1838 |
| GradientBoosting | -8623570023.6559 | 1157064517.4360 |
| Ridge | -10862579712.0000 | 1298615285.6738 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8533 | 4771.7416 |
| GradientBoosting | -109212266.7519 | 130211619.7133 |
| Ridge | -1719257216.0000 | 516635746.8822 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8419 | 100.0% |
| Replacement | -9786535758.7773 | -1162492020664.5% |
| Augmentation | -609489827.2995 | -72398147656.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 48361.6829 |
| avg_jsd | 0.5904 |
| correlation_frobenius | 0.2597 |
| avg_cat_freq_diff | 0.4056 |

**Numerical:** 3 columns, avg Wasserstein=48361.6829, avg JSD=0.5904, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.4056
