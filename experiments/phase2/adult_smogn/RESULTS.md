# smogn on Adult

**Dataset:** Adult (D10)
**Task:** classification
**Dimensions:** 6 num + 8 cat = 108 total
**Samples:** 39073 train / 9769 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8576 | 0.7942 |
| GradientBoosting | 0.8689 | 0.8030 |
| LogisticRegression | 0.8546 | 0.7854 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8608 | 0.7932 |
| GradientBoosting | 0.8662 | 0.7980 |
| LogisticRegression | 0.8540 | 0.7838 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8578 | 0.7942 |
| GradientBoosting | 0.8668 | 0.7990 |
| LogisticRegression | 0.8540 | 0.7844 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8604 | 100.0% |
| Replacement | 0.8603 | 100.0% |
| Augmentation | 0.8596 | 99.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0076 |
| avg_jsd | 0.0860 |
| correlation_frobenius | 0.0311 |
| avg_cat_freq_diff | 0.0062 |

**Numerical:** 6 columns, avg Wasserstein=0.0076, avg JSD=0.0860, KS pass rate (p>0.05)=17%
**Categorical:** 8 columns, avg freq L1 diff=0.0062

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.8086** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.3944 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

