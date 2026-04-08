# our_tabddpm on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** minmax, clip=True
**Training time:** 638.6s

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
| RandomForest | 0.9985 | 0.9883 |
| GradientBoosting | 0.9985 | 0.9883 |
| LogisticRegression | 0.9990 | 0.9923 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9985 | 0.9883 |
| GradientBoosting | 0.9985 | 0.9883 |
| LogisticRegression | 0.9990 | 0.9923 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9987 | 100.0% |
| Augmentation | 0.9987 | 100.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0304 |
| avg_jsd | 0.0173 |
| correlation_frobenius | 0.0601 |
| avg_cat_freq_diff | 0.0064 |

**Numerical:** 5 columns, avg Wasserstein=0.0304, avg JSD=0.0173, KS pass rate (p>0.05)=0%
**Categorical:** 6 columns, avg freq L1 diff=0.0064

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4814** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0011 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

