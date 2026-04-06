# our_tabddpm on Supply Chain Pricing

**Dataset:** Supply Chain Pricing (D8)
**Task:** regression
**Dimensions:** 6 num + 18 cat = 9605 total
**Samples:** 4958 train / 1240 test
**Preprocessing:** minmax, clip=True
**Training time:** 910.9s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.4936 | 9880.2619 |
| GradientBoosting | 0.4100 | 10664.5131 |
| Ridge | 0.3697 | 11022.8570 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -12821299101.2298 | 1572162963.5779 |
| GradientBoosting | -1745029528.4981 | 580006599.9798 |
| Ridge | -31707320.0000 | 78182817.1364 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -6945273553.7846 | 1157114200.0453 |
| GradientBoosting | -4201371824.6428 | 899968076.7139 |
| Ridge | -71381114880.0000 | 3709566982.8021 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.4245 | 100.0% |
| Replacement | -4866011983.2426 | -1146382583375.0% |
| Augmentation | -27509253419.4758 | -6480898343517.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 51040.4401 |
| avg_jsd | 0.5774 |
| correlation_frobenius | 2.2508 |
| avg_cat_freq_diff | 0.1376 |

**Numerical:** 6 columns, avg Wasserstein=51040.4401, avg JSD=0.5774, KS pass rate (p>0.05)=0%
**Categorical:** 18 columns, avg freq L1 diff=0.1376
