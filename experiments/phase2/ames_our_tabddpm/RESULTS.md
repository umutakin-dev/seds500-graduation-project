# our_tabddpm on Ames Housing

**Dataset:** Ames Housing (D11)
**Task:** regression
**Dimensions:** 32 num + 47 cat = 361 total
**Samples:** 2344 train / 586 test
**Preprocessing:** minmax, clip=True
**Training time:** 1348.9s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8961 | 28861.7414 |
| GradientBoosting | 0.9167 | 25847.1748 |
| Ridge | 0.9023 | 27993.9879 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -6092803797.4041 | 6989230750.7566 |
| GradientBoosting | -8443057373.8925 | 8227552320.5024 |
| Ridge | -208206192.0000 | 1292015152.7216 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8932 | 29259.8947 |
| GradientBoosting | -2102355.6712 | 129829678.3467 |
| Ridge | -5317942.0000 | 206486963.7456 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9050 | 100.0% |
| Replacement | -4914689121.0989 | -543053052164.9% |
| Augmentation | -2473432.2593 | -273304151.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 47447.1719 |
| avg_jsd | 0.5809 |
| correlation_frobenius | 7.1042 |
| avg_cat_freq_diff | 0.5249 |

**Numerical:** 32 columns, avg Wasserstein=47447.1719, avg JSD=0.5809, KS pass rate (p>0.05)=0%
**Categorical:** 47 columns, avg freq L1 diff=0.5249

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5085** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0000 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

