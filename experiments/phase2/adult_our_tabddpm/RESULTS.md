# our_tabddpm on Adult

**Dataset:** Adult (D10)
**Task:** classification
**Dimensions:** 6 num + 8 cat = 108 total
**Samples:** 39073 train / 9769 test
**Preprocessing:** minmax, clip=True
**Training time:** 2739.3s

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
| RandomForest | 0.8525 | 0.7787 |
| GradientBoosting | 0.8549 | 0.7830 |
| LogisticRegression | 0.8482 | 0.7867 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8567 | 0.7927 |
| GradientBoosting | 0.8657 | 0.8001 |
| LogisticRegression | 0.8548 | 0.7889 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8604 | 100.0% |
| Replacement | 0.8519 | 99.0% |
| Augmentation | 0.8591 | 99.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0254 |
| avg_jsd | 0.0773 |
| correlation_frobenius | 0.1102 |
| avg_cat_freq_diff | 0.0156 |

**Numerical:** 6 columns, avg Wasserstein=0.0254, avg JSD=0.0773, KS pass rate (p>0.05)=0%
**Categorical:** 8 columns, avg freq L1 diff=0.0156

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4963** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9955 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

