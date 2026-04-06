# vanilla_tabddpm on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** quantile, clip=False
**Training time:** 14.1s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9000 | 0.8997 |
| GradientBoosting | 0.9667 | 0.9666 |
| LogisticRegression | 1.0000 | 1.0000 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.3333 | 0.1667 |
| GradientBoosting | 0.3333 | 0.1667 |
| LogisticRegression | 0.7667 | 0.7511 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9000 | 0.8997 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.9000 | 0.8997 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9556 | 100.0% |
| Replacement | 0.4778 | 50.0% |
| Augmentation | 0.9111 | 95.3% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 2737.8726 |
| avg_jsd | 0.6002 |
| correlation_frobenius | 0.8724 |

**Numerical:** 4 columns, avg Wasserstein=2737.8726, avg JSD=0.6002, KS pass rate (p>0.05)=0%
