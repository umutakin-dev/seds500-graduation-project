# our_tabddpm on Online News Popularity

**Dataset:** Online News Popularity (D9)
**Task:** regression
**Dimensions:** 44 num + 14 cat = 72 total
**Samples:** 31715 train / 7929 test
**Preprocessing:** minmax, clip=True
**Training time:** 4353.6s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0521 | 11267.3343 |
| GradientBoosting | -0.0448 | 11228.2499 |
| Ridge | 0.0275 | 10833.0801 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.5688 | 13758.9444 |
| GradientBoosting | -327.0127 | 198951.9748 |
| Ridge | -575.5116 | 263758.8076 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0537 | 11276.0065 |
| GradientBoosting | -3.9480 | 24435.2234 |
| Ridge | -45.5035 | 74910.9953 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | -0.0231 | 100.0% |
| Replacement | -301.0310 | 1302544.1% |
| Augmentation | -16.5017 | 71401.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 379.8108 |
| avg_jsd | 0.0710 |
| correlation_frobenius | 19.3068 |
| avg_cat_freq_diff | 0.2412 |

**Numerical:** 44 columns, avg Wasserstein=379.8108, avg JSD=0.0710, KS pass rate (p>0.05)=0%
**Categorical:** 14 columns, avg freq L1 diff=0.2412
