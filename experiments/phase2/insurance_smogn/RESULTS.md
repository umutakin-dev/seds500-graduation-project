# smogn on Insurance Charges

**Dataset:** Insurance Charges (D3)
**Task:** regression
**Dimensions:** 3 num + 3 cat = 11 total
**Samples:** 1070 train / 268 test
**Preprocessing:** minmax, clip=True
**Training time:** 3.1s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8630 | 4611.7939 |
| GradientBoosting | 0.8803 | 4311.0199 |
| Ridge | 0.7822 | 5815.2469 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8718 | 4462.1070 |
| GradientBoosting | 0.8770 | 4368.9776 |
| Ridge | 0.7823 | 5813.2944 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.8663 | 4556.4521 |
| GradientBoosting | 0.8828 | 4265.1128 |
| Ridge | 0.7823 | 5813.0625 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8418 | 100.0% |
| Replacement | 0.8437 | 100.2% |
| Augmentation | 0.8438 | 100.2% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0198 |
| avg_jsd | 0.1146 |
| correlation_frobenius | 0.0085 |
| avg_cat_freq_diff | 0.0000 |

**Numerical:** 3 columns, avg Wasserstein=0.0198, avg JSD=0.1146, KS pass rate (p>0.05)=67%
**Categorical:** 3 columns, avg freq L1 diff=0.0000

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.8895** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.3317 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

