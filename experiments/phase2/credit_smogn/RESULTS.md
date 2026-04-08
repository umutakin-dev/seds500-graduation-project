# smogn on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

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
| RandomForest | 0.8142 | 0.6667 |
| GradientBoosting | 0.8168 | 0.6748 |
| LogisticRegression | 0.8178 | 0.6769 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8143 | 0.6770 |
| GradientBoosting | 0.8168 | 0.6741 |
| LogisticRegression | 0.8182 | 0.6772 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8168 | 100.0% |
| Replacement | 0.8163 | 99.9% |
| Augmentation | 0.8164 | 100.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0049 |
| avg_jsd | 0.0073 |
| correlation_frobenius | 0.1037 |
| avg_cat_freq_diff | 0.0074 |

**Numerical:** 14 columns, avg Wasserstein=0.0049, avg JSD=0.0073, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.0074

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.8124** |
| Interpretation | CRITICAL — synthetic data is essentially copies of real data |
| Distance Ratio (train/test) | 0.4042 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

