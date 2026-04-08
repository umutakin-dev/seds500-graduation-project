# ctgan on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** minmax, clip=True
**Training time:** 82.9s

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
| RandomForest | 0.5219 | 0.2555 |
| GradientBoosting | 0.5039 | 0.2572 |
| LogisticRegression | 0.5373 | 0.2799 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7841 | 0.7798 |
| GradientBoosting | 0.7738 | 0.7755 |
| LogisticRegression | 0.6581 | 0.5536 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7712 | 100.0% |
| Replacement | 0.5210 | 67.6% |
| Augmentation | 0.7386 | 95.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1263 |
| avg_jsd | 0.1044 |
| correlation_frobenius | 4.4456 |
| avg_cat_freq_diff | 0.2418 |

**Numerical:** 24 columns, avg Wasserstein=0.1263, avg JSD=0.1044, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.2418

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4503** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9930 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

