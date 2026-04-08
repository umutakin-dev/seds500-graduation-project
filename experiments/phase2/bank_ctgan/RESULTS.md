# ctgan on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** minmax, clip=True
**Training time:** 742.8s

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
| RandomForest | 0.8712 | 0.6845 |
| GradientBoosting | 0.8611 | 0.6832 |
| LogisticRegression | 0.8643 | 0.6789 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9018 | 0.7068 |
| GradientBoosting | 0.8998 | 0.7005 |
| LogisticRegression | 0.8968 | 0.6973 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9042 | 100.0% |
| Replacement | 0.8655 | 95.7% |
| Augmentation | 0.8995 | 99.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0520 |
| avg_jsd | 0.0446 |
| correlation_frobenius | 0.2696 |
| avg_cat_freq_diff | 0.2594 |

**Numerical:** 7 columns, avg Wasserstein=0.0520, avg JSD=0.0446, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.2594

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4930** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9994 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

