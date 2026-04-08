# ctgan on AI4I Predictive Maintenance

**Dataset:** AI4I Predictive Maintenance (D4)
**Task:** classification
**Dimensions:** 5 num + 6 cat = 18 total
**Samples:** 8000 train / 2000 test
**Preprocessing:** minmax, clip=True
**Training time:** 149.2s

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
| RandomForest | 0.9955 | 0.9655 |
| GradientBoosting | 0.9935 | 0.9522 |
| LogisticRegression | 0.9990 | 0.9923 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9990 | 0.9923 |
| GradientBoosting | 0.9990 | 0.9923 |
| LogisticRegression | 0.9990 | 0.9923 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9990 | 100.0% |
| Replacement | 0.9960 | 99.7% |
| Augmentation | 0.9990 | 100.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0823 |
| avg_jsd | 0.0287 |
| correlation_frobenius | 1.1419 |
| avg_cat_freq_diff | 0.1031 |

**Numerical:** 5 columns, avg Wasserstein=0.0823, avg JSD=0.0287, KS pass rate (p>0.05)=0%
**Categorical:** 6 columns, avg freq L1 diff=0.1031

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5160** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9828 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

