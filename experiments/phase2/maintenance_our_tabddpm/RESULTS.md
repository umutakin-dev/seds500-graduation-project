# our_tabddpm on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** minmax, clip=True
**Training time:** 647.5s

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
| LogisticRegression | 0.9985 | 0.9883 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9985 | 0.9883 |
| LogisticRegression | 0.9990 | 0.9923 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9985 | 99.9% |
| Augmentation | 0.9988 | 100.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0279 |
| avg_jsd | 0.0166 |
| correlation_frobenius | 0.0887 |
| avg_cat_freq_diff | 0.0027 |

**Numerical:** 5 columns, avg Wasserstein=0.0279, avg JSD=0.0166, KS pass rate (p>0.05)=20%
**Categorical:** 6 columns, avg freq L1 diff=0.0027
