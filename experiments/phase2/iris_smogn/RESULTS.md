# smogn on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.0s

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
| RandomForest | 0.9333 | 0.9333 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.9000 | 0.8997 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9000 | 0.8997 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.9333 | 0.9333 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9111 | 100.0% |
| Replacement | 0.9222 | 101.2% |
| Augmentation | 0.9222 | 101.2% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0450 |
| avg_jsd | 0.0835 |
| correlation_frobenius | 0.2211 |

**Numerical:** 4 columns, avg Wasserstein=0.0450, avg JSD=0.0835, KS pass rate (p>0.05)=100%
