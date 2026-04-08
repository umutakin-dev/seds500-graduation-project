# vanilla_tabddpm on Iris

**Dataset:** Iris (D1)
**Task:** classification
**Dimensions:** 4 num + 0 cat = 4 total
**Samples:** 120 train / 30 test
**Preprocessing:** quantile, clip=False
**Training time:** 13.5s

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
| RandomForest | 0.3000 | 0.1538 |
| GradientBoosting | 0.3333 | 0.1667 |
| LogisticRegression | 0.7000 | 0.6424 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9333 | 0.9333 |
| GradientBoosting | 0.9333 | 0.9333 |
| LogisticRegression | 0.9667 | 0.9666 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9556 | 100.0% |
| Replacement | 0.4444 | 46.5% |
| Augmentation | 0.9444 | 98.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 1968.0117 |
| avg_jsd | 0.5560 |
| correlation_frobenius | 1.7645 |

**Numerical:** 4 columns, avg Wasserstein=1968.0117, avg JSD=0.5560, KS pass rate (p>0.05)=0%

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4306** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9987 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

