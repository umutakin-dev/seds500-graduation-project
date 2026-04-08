# our_tabddpm on Bank Marketing

**Dataset:** Bank Marketing (D6)
**Task:** classification
**Dimensions:** 7 num + 9 cat = 51 total
**Samples:** 36168 train / 9043 test
**Preprocessing:** minmax, clip=True
**Training time:** 2497.9s

## Utility

### Baseline
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9049 | 0.7211 |
| GradientBoosting | 0.9061 | 0.7277 |
| LogisticRegression | 0.9017 | 0.7047 |

### Replacement
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.8946 | 0.6234 |
| GradientBoosting | 0.8955 | 0.6173 |
| LogisticRegression | 0.8980 | 0.6495 |

### Augmentation
| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| RandomForest | 0.9047 | 0.7087 |
| GradientBoosting | 0.9012 | 0.6847 |
| LogisticRegression | 0.8991 | 0.6760 |

### Summary
| Scenario | Avg ACCURACY | % of Baseline |
| --- | --- | --- |
| Baseline | 0.9042 | 100.0% |
| Replacement | 0.8961 | 99.1% |
| Augmentation | 0.9017 | 99.7% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 0.1710 |
| avg_jsd | 0.0084 |
| correlation_frobenius | 2.0333 |
| avg_cat_freq_diff | 0.0144 |

**Numerical:** 7 columns, avg Wasserstein=0.1710, avg JSD=0.0084, KS pass rate (p>0.05)=0%
**Categorical:** 9 columns, avg freq L1 diff=0.0144

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5026** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9948 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

