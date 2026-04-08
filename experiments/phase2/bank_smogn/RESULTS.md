# smogn on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9049 | 0.7211 |
| GradientBoosting | 0.9061 | 0.7277 |
| LogisticRegression | 0.9017 | 0.7047 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9048 | 0.7080 |
| GradientBoosting | 0.9071 | 0.7282 |
| LogisticRegression | 0.9024 | 0.7067 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9063 | 0.7246 |
| GradientBoosting | 0.9069 | 0.7292 |
| LogisticRegression | 0.9017 | 0.7040 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9042 | 100.0% |
| Replacement | 0.9048 | 100.1% |
| Augmentation | 0.9050 | 100.1% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0069 |
| avg_jsd | 0.0744 |
| correlation_frobenius | 0.0225 |
| avg_cat_freq_diff | 0.0039 |

**Numerical:** 7 columns, avg Wasserstein=0.0069, avg JSD=0.0744, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.0039

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.8179** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.3869 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

