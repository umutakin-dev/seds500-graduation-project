# ctgan on Adult

**Dataset:** Adult (D10)
**Task:** classification
**Dimensions:** 6 num + 8 cat = 108 total
**Samples:** 39073 train / 9769 test
**Preprocessing:** minmax, clip=True
**Training time:** 809.2s

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
| RandomForest | 0.8127 | 0.6863 |
| GradientBoosting | 0.8235 | 0.7055 |
| LogisticRegression | 0.7830 | 0.5498 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8572 | 0.7922 |
| GradientBoosting | 0.8585 | 0.7839 |
| LogisticRegression | 0.8493 | 0.7730 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8604 | 100.0% |
| Replacement | 0.8064 | 93.7% |
| Augmentation | 0.8550 | 99.4% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0390 |
| avg_jsd | 0.0510 |
| correlation_frobenius | 0.2677 |
| avg_cat_freq_diff | 0.5855 |

**Numerical:** 6 columns, avg Wasserstein=0.0390, avg JSD=0.0510, KS pass rate (p>0.05)=0%
**Categorical:** 8 columns, avg freq L1 diff=0.5855

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4949** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9999 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

