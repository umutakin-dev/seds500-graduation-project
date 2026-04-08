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
| RandomForest | 0.9333 | 0.9327 |
| GradientBoosting | 0.9333 | 0.9327 |
| LogisticRegression | 0.8667 | 0.8653 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9000 | 0.8997 |
| GradientBoosting | 0.9000 | 0.8997 |
| LogisticRegression | 0.9333 | 0.9333 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9111 | 100.0% |
| Replacement | 0.9111 | 100.0% |
| Augmentation | 0.9111 | 100.0% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.0601 |
| avg_jsd | 0.0801 |
| correlation_frobenius | 0.1860 |

**Numerical:** 4 columns, avg Wasserstein=0.0601, avg JSD=0.0801, KS pass rate (p>0.05)=100%

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.7917** |
| Interpretation | UNSAFE — significant privacy risk |
| Distance Ratio (train/test) | 0.5024 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

