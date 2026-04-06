# vanilla_tabddpm on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** quantile, clip=False
**Training time:** 1627.4s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8148 | 0.6756 |
| GradientBoosting | 0.8178 | 0.6761 |
| LogisticRegression | 0.8188 | 0.6799 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8027 | 0.6186 |
| GradientBoosting | 0.8008 | 0.6023 |
| LogisticRegression | 0.7788 | 0.4591 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8120 | 0.6692 |
| GradientBoosting | 0.8125 | 0.6568 |
| LogisticRegression | 0.8025 | 0.6195 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8172 | 100.0% |
| Replacement | 0.7941 | 97.2% |
| Augmentation | 0.8090 | 99.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 410.6809 |
| avg_jsd | 0.1250 |
| correlation_frobenius | 5.3692 |
| avg_cat_freq_diff | 0.4165 |

**Numerical:** 14 columns, avg Wasserstein=410.6809, avg JSD=0.1250, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.4165
