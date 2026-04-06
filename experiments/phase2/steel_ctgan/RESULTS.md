# ctgan on Steel Plates Faults

**Dataset:** Steel Plates Faults (D5)
**Task:** classification
**Dimensions:** 24 num + 3 cat = 31 total
**Samples:** 1552 train / 389 test
**Preprocessing:** minmax, clip=True
**Training time:** 82.9s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7892 | 0.8003 |
| GradientBoosting | 0.8021 | 0.8206 |
| LogisticRegression | 0.7224 | 0.6839 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.4936 | 0.2362 |
| GradientBoosting | 0.4910 | 0.3304 |
| LogisticRegression | 0.4961 | 0.2648 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.7892 | 0.7952 |
| GradientBoosting | 0.7558 | 0.7234 |
| LogisticRegression | 0.6889 | 0.6080 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.7712 | 100.0% |
| Replacement | 0.4936 | 64.0% |
| Augmentation | 0.7446 | 96.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1522 |
| avg_jsd | 0.1143 |
| correlation_frobenius | 4.6355 |
| avg_cat_freq_diff | 0.0619 |

**Numerical:** 24 columns, avg Wasserstein=0.1522, avg JSD=0.1143, KS pass rate (p>0.05)=0%
**Categorical:** 3 columns, avg freq L1 diff=0.0619
