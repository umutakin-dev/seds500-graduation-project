# vanilla_tabddpm on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** quantile, clip=False
**Training time:** 155.2s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7866 | 0.7936 |
| GradientBoosting | 0.8072 | 0.8224 |
| LogisticRegression | 0.7121 | 0.7144 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.2982 | 0.1477 |
| GradientBoosting | 0.2005 | 0.0477 |
| LogisticRegression | 0.2982 | 0.1819 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7943 | 0.7996 |
| GradientBoosting | 0.7841 | 0.7922 |
| LogisticRegression | 0.5527 | 0.4190 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7686 | 100.0% |
| Replacement | 0.2656 | 34.6% |
| Augmentation | 0.7104 | 92.4% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 1611.4284 |
| avg_jsd | 0.4353 |
| correlation_frobenius | 8.6310 |
| avg_cat_freq_diff | 0.4094 |

**Numerical:** 24 columns, avg Wasserstein=1611.4284, avg JSD=0.4353, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.4094

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4759** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0006 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

