# ctgan on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** minmax, clip=True
**Training time:** 12.7s

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
| RandomForest | 0.7667 | 0.7341 |
| GradientBoosting | 0.7667 | 0.7341 |
| LogisticRegression | 0.7667 | 0.7341 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9667 | 0.9666 |
| GradientBoosting | 0.9667 | 0.9666 |
| LogisticRegression | 0.9333 | 0.9327 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9111 | 100.0% |
| Replacement | 0.7667 | 84.1% |
| Augmentation | 0.9556 | 104.9% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1655 |
| avg_jsd | 0.2580 |
| correlation_frobenius | 0.8270 |

**Numerical:** 4 columns, avg Wasserstein=0.1655, avg JSD=0.2580, KS pass rate (p>0.05)=50%
