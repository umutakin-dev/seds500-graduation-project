# smogn on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9990 | 0.9923 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9990 | 0.9923 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9990 | 0.9923 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9990 | 100.0% |
| Augmentation | 0.9990 | 100.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0068 |
| avg_jsd | 0.0057 |
| correlation_frobenius | 0.0357 |
| avg_cat_freq_diff | 0.0025 |

**Numerical:** 5 columns, avg Wasserstein=0.0068, avg JSD=0.0057, KS pass rate (p>0.05)=100%
**Categorical:** 6 columns, avg freq L1 diff=0.0025

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.8211** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.4291 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

