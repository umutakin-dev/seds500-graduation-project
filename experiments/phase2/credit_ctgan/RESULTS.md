# ctgan on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** minmax, clip=True
**Training time:** 641.6s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8155 | 0.6770 |
| GradientBoosting | 0.8173 | 0.6758 |
| LogisticRegression | 0.8177 | 0.6764 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7275 | 0.6156 |
| GradientBoosting | 0.6848 | 0.5892 |
| LogisticRegression | 0.4480 | 0.4385 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8128 | 0.6784 |
| GradientBoosting | 0.8160 | 0.6714 |
| LogisticRegression | 0.8130 | 0.6610 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8168 | 100.0% |
| Replacement | 0.6201 | 75.9% |
| Augmentation | 0.8139 | 99.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0332 |
| avg_jsd | 0.0226 |
| correlation_frobenius | 1.1322 |
| avg_cat_freq_diff | 0.6631 |

**Numerical:** 14 columns, avg Wasserstein=0.0332, avg JSD=0.0226, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.6631

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5007** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0030 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

