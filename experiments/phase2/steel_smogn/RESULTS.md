# smogn on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7892 | 0.8003 |
| GradientBoosting | 0.8021 | 0.8206 |
| LogisticRegression | 0.7224 | 0.6839 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7635 | 0.7604 |
| GradientBoosting | 0.7841 | 0.7920 |
| LogisticRegression | 0.6992 | 0.6732 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8046 | 0.8223 |
| GradientBoosting | 0.8021 | 0.8221 |
| LogisticRegression | 0.7018 | 0.6746 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7712 | 100.0% |
| Replacement | 0.7489 | 97.1% |
| Augmentation | 0.7695 | 99.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0135 |
| avg_jsd | 0.0105 |
| correlation_frobenius | 0.4576 |
| avg_cat_freq_diff | 0.0120 |

**Numerical:** 24 columns, avg Wasserstein=0.0135, avg JSD=0.0105, KS pass rate (p>0.05)=67%
**Categorical:** 3 columns, avg freq L1 diff=0.0120

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.8153** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.4005 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

