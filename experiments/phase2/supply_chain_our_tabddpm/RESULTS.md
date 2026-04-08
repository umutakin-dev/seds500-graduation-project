# our_tabddpm on Supply Chain Pricing

**Dataset:** Supply Chain Pricing (D8)
**Task:** regression
**Dimensions:** 6 num + 18 cat = 9605 total
**Samples:** 4958 train / 1240 test
**Preprocessing:** minmax, clip=True
**Training time:** 917.2s

## Utility

### Baseline
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | 0.4936 | 9880.2619 |
| GradientBoosting | 0.4100 | 10664.5131 |
| Ridge | 0.3697 | 11022.8570 |

### Replacement
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -4734915901.0505 | 955405374.7577 |
| GradientBoosting | -7422319626.6137 | 1196193316.2649 |
| Ridge | -2897802.2500 | 23635564.7317 |

### Augmentation
| Model | R² | RMSE |
| --- | --- | --- |
| RandomForest | -7790345.1392 | 38753409.9512 |
| GradientBoosting | -415645647.0242 | 283069556.2219 |
| Ridge | -2482086144.0000 | 691735333.0298 |

### Summary
| Scenario | Avg R2 | % of Baseline |
| --- | --- | --- |
| Baseline | 0.4245 | 100.0% |
| Replacement | -4053377776.6381 | -954934287662.9% |
| Augmentation | -968507378.7211 | -228170418539.8% |

## Fidelity
### Statistical Fidelity Summary
| Metric | Value |
| --- | --- |
| avg_wasserstein | 51371.5463 |
| avg_jsd | 0.5811 |
| correlation_frobenius | 2.2750 |
| avg_cat_freq_diff | 1.0226 |

**Numerical:** 6 columns, avg Wasserstein=51371.5463, avg JSD=0.5811, KS pass rate (p>0.05)=0%
**Categorical:** 18 columns, avg freq L1 diff=1.0226

## Privacy (Membership Inference Attack)
| Metric | Value |
| --- | --- |
| Attack AUC | **0.5050** |
| Interpretation | SAFE — no membership information leaked |
| Distance Ratio (train/test) | 1.0000 |

*AUC ~0.50 = safe (random guessing), >0.60 = privacy concern, >0.80 = critical leak*

