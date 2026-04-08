# smogn on Supply Chain Pricing

**Dataset:** Supply Chain Pricing (D8)
**Task:** regression
**Dimensions:** 6 num + 18 cat = 9605 total
**Samples:** 4958 train / 1240 test
**Preprocessing:** minmax, clip=True
**Training time:** 116.8s

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
| RandomForest | 0.5129 | 9690.4737 |
| GradientBoosting | 0.2357 | 12138.5620 |
| Ridge | -0.0212 | 14030.9196 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.5045 | 9773.2168 |
| GradientBoosting | 0.3306 | 11360.0489 |
| Ridge | 0.0061 | 13841.8687 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.4245 | 100.0% |
| Replacement | 0.2425 | 57.1% |
| Augmentation | 0.2804 | 66.1% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.3637 |
| avg_jsd | 0.1657 |
| correlation_frobenius | 3.2838 |
| avg_cat_freq_diff | 0.2817 |

**Numerical:** 6 columns, avg Wasserstein=0.3637, avg JSD=0.1657, KS pass rate (p>0.05)=0%
**Categorical:** 18 columns, avg freq L1 diff=0.2817

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.9102** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.1683 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

