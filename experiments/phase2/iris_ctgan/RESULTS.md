# ctgan on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** minmax, clip=True
**Training time:** 15.5s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9000 | 0.8997 |
| GradientBoosting | 0.9000 | 0.8997 |
| LogisticRegression | 0.9333 | 0.9333 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9000 | 0.8977 |
| GradientBoosting | 0.8667 | 0.8660 |
| LogisticRegression | 0.8000 | 0.7802 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9333 | 0.9333 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.9333 | 0.9327 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9111 | 100.0% |
| Replacement | 0.8556 | 93.9% |
| Augmentation | 0.9333 | 102.4% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1622 |
| avg_jsd | 0.2392 |
| correlation_frobenius | 0.8049 |

**Numerical:** 4 columns, avg Wasserstein=0.1622, avg JSD=0.2392, KS pass rate (p>0.05)=50%
