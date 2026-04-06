# ctgan on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** minmax, clip=True
**Training time:** 632.4s

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
| RandomForest | 0.7877 | 0.5033 |
| GradientBoosting | 0.7880 | 0.4969 |
| LogisticRegression | 0.7957 | 0.5603 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8120 | 0.6692 |
| GradientBoosting | 0.8105 | 0.6444 |
| LogisticRegression | 0.8122 | 0.6490 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8168 | 100.0% |
| Replacement | 0.7904 | 96.8% |
| Augmentation | 0.8116 | 99.4% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0362 |
| avg_jsd | 0.0246 |
| correlation_frobenius | 1.0449 |
| avg_cat_freq_diff | 0.7433 |

**Numerical:** 14 columns, avg Wasserstein=0.0362, avg JSD=0.0246, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.7433
