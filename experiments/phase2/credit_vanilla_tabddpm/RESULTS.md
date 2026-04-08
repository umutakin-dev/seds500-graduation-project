# vanilla_tabddpm on Credit Default

**Dataset:** Credit Default (D7)
**Task:** classification
**Dimensions:** 14 num + 9 cat = 91 total
**Samples:** 24000 train / 6000 test
**Preprocessing:** quantile, clip=False
**Training time:** 1652.1s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8148 | 0.6756 |
| GradientBoosting | 0.8178 | 0.6761 |
| LogisticRegression | 0.8188 | 0.6799 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8032 | 0.6167 |
| GradientBoosting | 0.8045 | 0.6179 |
| LogisticRegression | 0.7913 | 0.5300 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8128 | 0.6716 |
| GradientBoosting | 0.8143 | 0.6636 |
| LogisticRegression | 0.8062 | 0.6246 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.8172 | 100.0% |
| Replacement | 0.7997 | 97.9% |
| Augmentation | 0.8111 | 99.3% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 486.4718 |
| avg_jsd | 0.1090 |
| correlation_frobenius | 6.0812 |
| avg_cat_freq_diff | 0.5269 |

**Numerical:** 14 columns, avg Wasserstein=486.4718, avg JSD=0.1090, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.5269

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4918** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9978 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

