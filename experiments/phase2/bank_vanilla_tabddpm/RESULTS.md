# vanilla_tabddpm on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** quantile, clip=False
**Training time:** 2492.3s

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
| RandomForest | 0.8954 | 0.6091 |
| GradientBoosting | 0.8963 | 0.6107 |
| LogisticRegression | 0.8961 | 0.6054 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9050 | 0.7093 |
| GradientBoosting | 0.9017 | 0.6809 |
| LogisticRegression | 0.9004 | 0.6594 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9056 | 100.0% |
| Replacement | 0.8959 | 98.9% |
| Augmentation | 0.9024 | 99.6% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.4402 |
| avg_jsd | 0.0234 |
| correlation_frobenius | 2.0919 |
| avg_cat_freq_diff | 0.0566 |

**Numerical:** 7 columns, avg Wasserstein=0.4402, avg JSD=0.0234, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.0566

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5032** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9971 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

