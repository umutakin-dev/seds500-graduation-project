# smogn on Ames Housing

**Dataset:** Ames Housing (D11)
**Task:** regression
**Dimensions:** 32 num + 47 cat = 361 total
**Samples:** 2344 train / 586 test
**Preprocessing:** minmax, clip=True
**Training time:** 63.5s

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
| RandomForest | 0.8905 | 29635.5237 |
| GradientBoosting | 0.8884 | 29906.8089 |
| Ridge | 0.8785 | 31215.8196 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.9054 | 27538.2710 |
| GradientBoosting | 0.9011 | 28159.3685 |
| Ridge | 0.8961 | 28865.5465 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9050 | 100.0% |
| Replacement | 0.8858 | 97.9% |
| Augmentation | 0.9009 | 99.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.2931 |
| avg_jsd | 0.1102 |
| correlation_frobenius | 11.8511 |
| avg_cat_freq_diff | 0.2047 |

**Numerical:** 32 columns, avg Wasserstein=0.2931, avg JSD=0.1102, KS pass rate (p>0.05)=3%
**Categorical:** 47 columns, avg freq L1 diff=0.2047

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.9082** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.1734 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

