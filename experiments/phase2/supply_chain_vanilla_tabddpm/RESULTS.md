# vanilla_tabddpm on Supply Chain Pricing

**Dataset:** Supply Chain Pricing (D8)
**Task:** regression
**Dimensions:** 6 num + 18 cat = 9605 total
**Samples:** 4958 train / 1240 test
**Preprocessing:** quantile, clip=False
**Training time:** 920.5s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.4906 | 9909.8521 |
| GradientBoosting | 0.4099 | 10666.1764 |
| Ridge | 0.3606 | 11102.1065 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -2603876958.1677 | 708503132.2527 |
| GradientBoosting | -5290485549.6624 | 1009902184.2900 |
| Ridge | -9688932.0000 | 43218482.7778 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -585980.6489 | 10628538.5725 |
| GradientBoosting | -19236917.4235 | 60897476.5554 |
| Ridge | -4442987520.0000 | 925484346.7204 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.4204 | 100.0% |
| Replacement | -2634683813.2767 | -626766895335.6% |
| Augmentation | -1487603472.6908 | -353887098471.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 51264.4666 |
| avg_jsd | 0.5859 |
| correlation_frobenius | 2.7121 |
| avg_cat_freq_diff | 0.8957 |

**Numerical:** 6 columns, avg Wasserstein=51264.4666, avg JSD=0.5859, KS pass rate (p>0.05)=0%
**Categorical:** 18 columns, avg freq L1 diff=0.8957

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4962** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0000 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

