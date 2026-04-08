# our_tabddpm on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** minmax, clip=True
**Training time:** 1651.0s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8155 | 0.6770 |
| GradientBoosting | 0.8173 | 0.6758 |
| LogisticRegression | 0.8177 | 0.6764 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8082 | 0.6230 |
| GradientBoosting | 0.8137 | 0.6654 |
| LogisticRegression | 0.7868 | 0.5136 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8150 | 0.6760 |
| GradientBoosting | 0.8168 | 0.6741 |
| LogisticRegression | 0.8118 | 0.6576 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8168 | 100.0% |
| Replacement | 0.8029 | 98.3% |
| Augmentation | 0.8146 | 99.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 200.6011 |
| avg_jsd | 0.1203 |
| correlation_frobenius | 4.1948 |
| avg_cat_freq_diff | 0.3183 |

**Numerical:** 14 columns, avg Wasserstein=200.6011, avg JSD=0.1203, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.3183

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4959** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9974 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

