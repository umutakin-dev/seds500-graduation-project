# vanilla_tabddpm on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** quantile, clip=False
**Training time:** 641.5s

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
| RandomForest | 0.9950 | 0.9590 |
| GradientBoosting | 0.9945 | 0.9546 |
| LogisticRegression | 0.9865 | 0.8727 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9930 | 0.9408 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9920 | 99.3% |
| Augmentation | 0.9970 | 99.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 26.6270 |
| avg_jsd | 0.0321 |
| correlation_frobenius | 3.9062 |
| avg_cat_freq_diff | 0.0131 |

**Numerical:** 5 columns, avg Wasserstein=26.6270, avg JSD=0.0321, KS pass rate (p>0.05)=0%
**Categorical:** 6 columns, avg freq L1 diff=0.0131

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5123** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9920 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

