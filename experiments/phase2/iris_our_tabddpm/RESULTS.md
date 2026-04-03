# our_tabddpm on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** minmax, clip=True
**Training time:** 0.9s

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
| RandomForest | 0.3333 | 0.1667 |
| GradientBoosting | 0.3333 | 0.1667 |
| LogisticRegression | 0.3333 | 0.1667 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9333 | 0.9333 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.3333 | 0.1667 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9111 | 100.0% |
| Replacement | 0.3333 | 36.6% |
| Augmentation | 0.7333 | 80.5% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 46764.3149 |
| avg_jsd | 0.6231 |
| correlation_frobenius | 2.2474 |

**Numerical:** 4 columns, avg Wasserstein=46764.3149, avg JSD=0.6231, KS pass rate (p>0.05)=0%
