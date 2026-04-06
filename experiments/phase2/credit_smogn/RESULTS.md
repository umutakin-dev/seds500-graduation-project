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
| RandomForest | 0.8138 | 0.6708 |
| GradientBoosting | 0.8165 | 0.6780 |
| LogisticRegression | 0.8165 | 0.6758 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8132 | 0.6759 |
| GradientBoosting | 0.8177 | 0.6784 |
| LogisticRegression | 0.8163 | 0.6748 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8168 | 100.0% |
| Replacement | 0.8156 | 99.9% |
| Augmentation | 0.8157 | 99.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0050 |
| avg_jsd | 0.0078 |
| correlation_frobenius | 0.1289 |
| avg_cat_freq_diff | 0.0055 |

**Numerical:** 14 columns, avg Wasserstein=0.0050, avg JSD=0.0078, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.0055
