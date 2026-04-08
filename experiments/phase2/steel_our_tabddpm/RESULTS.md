# our_tabddpm on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** minmax, clip=True
**Training time:** 154.8s

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
| RandomForest | 0.5604 | 0.4473 |
| GradientBoosting | 0.5604 | 0.3623 |
| LogisticRegression | 0.5938 | 0.4815 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7866 | 0.8034 |
| GradientBoosting | 0.7712 | 0.7702 |
| LogisticRegression | 0.6915 | 0.6006 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7712 | 100.0% |
| Replacement | 0.5716 | 74.1% |
| Augmentation | 0.7498 | 97.2% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 84.8644 |
| avg_jsd | 0.1595 |
| correlation_frobenius | 17.8169 |
| avg_cat_freq_diff | 0.0735 |

**Numerical:** 24 columns, avg Wasserstein=84.8644, avg JSD=0.1595, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.0735

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4814** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9917 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

