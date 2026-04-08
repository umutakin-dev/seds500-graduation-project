# vanilla_tabddpm on Online News Popularity

**Dataset:** Online News Popularity (D9)
**Task:** regression
**Dimensions:** 44 num + 14 cat = 72 total
**Samples:** 31715 train / 7929 test
**Preprocessing:** quantile, clip=False
**Training time:** 3513.5s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0586 | 11302.4549 |
| GradientBoosting | -0.0121 | 11051.2255 |
| Ridge | 0.0232 | 10856.6923 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -507.5815 | 247732.6468 |
| GradientBoosting | -108576.0941 | 3619695.7234 |
| Ridge | -255464.7344 | 5552253.5906 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -0.0524 | 11269.4229 |
| GradientBoosting | -661.3791 | 282719.8278 |
| Ridge | -3385.5986 | 639270.7269 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | -0.0158 | 100.0% |
| Replacement | -121516.1366 | 768084280.8% |
| Augmentation | -1349.0101 | 8526879.2% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 2439.3509 |
| avg_jsd | 0.2900 |
| correlation_frobenius | 6.0819 |
| avg_cat_freq_diff | 0.2446 |

**Numerical:** 44 columns, avg Wasserstein=2439.3509, avg JSD=0.2900, KS pass rate (p>0.05)=0%
**Categorical:** 14 columns, avg freq L1 diff=0.2446

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.4989** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 0.9999 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

