# vanilla_tabddpm on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** quantile, clip=False
**Training time:** 2461.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9085 | 0.7314 |
| GradientBoosting | 0.9055 | 0.7258 |
| LogisticRegression | 0.9027 | 0.7046 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8930 | 0.6006 |
| GradientBoosting | 0.8966 | 0.6219 |
| LogisticRegression | 0.8966 | 0.6229 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9029 | 0.6993 |
| GradientBoosting | 0.9012 | 0.6712 |
| LogisticRegression | 0.8988 | 0.6510 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9056 | 100.0% |
| Replacement | 0.8954 | 98.9% |
| Augmentation | 0.9010 | 99.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 1.1335 |
| avg_jsd | 0.0302 |
| correlation_frobenius | 2.3880 |
| avg_cat_freq_diff | 0.1237 |

**Numerical:** 7 columns, avg Wasserstein=1.1335, avg JSD=0.0302, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.1237
