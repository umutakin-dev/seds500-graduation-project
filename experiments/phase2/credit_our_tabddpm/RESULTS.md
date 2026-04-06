# our_tabddpm on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** minmax, clip=True
**Training time:** 1625.1s

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
| RandomForest | 0.8065 | 0.6225 |
| GradientBoosting | 0.8143 | 0.6684 |
| LogisticRegression | 0.8018 | 0.6179 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8150 | 0.6752 |
| GradientBoosting | 0.8170 | 0.6763 |
| LogisticRegression | 0.8120 | 0.6515 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8168 | 100.0% |
| Replacement | 0.8076 | 98.9% |
| Augmentation | 0.8147 | 99.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 331.4569 |
| avg_jsd | 0.1740 |
| correlation_frobenius | 4.0876 |
| avg_cat_freq_diff | 0.5701 |

**Numerical:** 14 columns, avg Wasserstein=331.4569, avg JSD=0.1740, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.5701
