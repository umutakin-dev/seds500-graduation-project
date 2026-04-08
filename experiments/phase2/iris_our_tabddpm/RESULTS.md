# our_tabddpm on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** minmax, clip=True
**Training time:** 15.6s

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
| LogisticRegression | 0.6667 | 0.5368 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9333 | 0.9333 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.9667 | 0.9666 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9111 | 100.0% |
| Replacement | 0.4444 | 48.8% |
| Augmentation | 0.9444 | 103.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 395.5953 |
| avg_jsd | 0.5538 |
| correlation_frobenius | 1.7424 |

**Numerical:** 4 columns, avg Wasserstein=395.5953, avg JSD=0.5538, KS pass rate (p>0.05)=0%

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4236** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0003 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

