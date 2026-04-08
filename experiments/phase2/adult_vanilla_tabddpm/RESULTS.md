# vanilla_tabddpm on Adult

**Dataset:** Adult (D10)
**Task:** classification
**Dimensions:** 6 num + 8 cat = 108 total
**Samples:** 39073 train / 9769 test
**Preprocessing:** quantile, clip=False
**Training time:** 2723.1s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8560 | 0.7915 |
| GradientBoosting | 0.8691 | 0.8029 |
| LogisticRegression | 0.8477 | 0.7756 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8456 | 0.7732 |
| GradientBoosting | 0.8503 | 0.7810 |
| LogisticRegression | 0.8427 | 0.7789 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8579 | 0.7944 |
| GradientBoosting | 0.8630 | 0.7971 |
| LogisticRegression | 0.8467 | 0.7787 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8576 | 100.0% |
| Replacement | 0.8462 | 98.7% |
| Augmentation | 0.8559 | 99.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 51.5942 |
| avg_jsd | 0.0372 |
| correlation_frobenius | 3.7605 |
| avg_cat_freq_diff | 0.1147 |

**Numerical:** 6 columns, avg Wasserstein=51.5942, avg JSD=0.0372, KS pass rate (p>0.05)=0%
**Categorical:** 8 columns, avg freq L1 diff=0.1147

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4960** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9968 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

