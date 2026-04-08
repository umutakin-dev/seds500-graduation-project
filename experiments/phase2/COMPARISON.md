# Phase 2 Experiment Comparison Report

**Experiments:** 44 runs (4 methods × 11 datasets)
**Methods:** Our TabDDPM, Vanilla TabDDPM, CTGAN, SMOGN

## Dataset Overview

| # | Dataset | Samples | Num | Cat | Total Dims | Task |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Iris | 120+30 | 4 | 0 | 4 | classification |
| D2 | California Housing | 16512+4128 | 8 | 0 | 8 | regression |
| D3 | Insurance Charges | 1070+268 | 3 | 3 | 11 | regression |
| D4 | AI4I Predictive Maintenance | 8000+2000 | 5 | 6 | 18 | classification |
| D5 | Steel Plates Faults | 1552+389 | 24 | 3 | 31 | classification |
| D6 | Bank Marketing | 36168+9043 | 7 | 9 | 51 | classification |
| D7 | Credit Default | 24000+6000 | 14 | 9 | 91 | classification |
| D8 | Supply Chain Pricing | 4958+1240 | 6 | 18 | 9605 | regression |
| D9 | Online News Popularity | 31715+7929 | 44 | 14 | 72 | regression |
| D10 | Adult | 39073+9769 | 6 | 8 | 108 | classification |
| D11 | Ames Housing | 2344+586 | 32 | 47 | 361 | regression |

---

## Key Findings

**Replacement Scenario Wins:** Our TabDDPM: 0, Vanilla TabDDPM: 0, CTGAN: 0, SMOGN: 10, 

**Average Replacement Utility:**
- Our TabDDPM: **87.8%**
- Vanilla TabDDPM: **80.9%**
- CTGAN: **68.5%**
- SMOGN: **94.6%**

**Average Augmentation Utility:**
- Our TabDDPM: **99.6%**
- Vanilla TabDDPM: **97.0%**
- CTGAN: **95.5%**
- SMOGN: **96.1%**

---

## Utility — Replacement Scenario
*Train on synthetic data only, test on real data. Higher % = better.*

### Replacement (% of Baseline)

| Method | Iris (4d) | California (8d) | Insurance (11d) | Maintenance (18d) | Steel (31d) | Bank (51d) | Credit (91d) | Supply Chain (9605d) | News (72d) | Adult (108d) | Ames Housing (361d) | **Avg** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our TabDDPM** | 48.8% | 92.2% | 91.1% | 100.0% | 74.1% | 99.1% | 98.3% | — | — | 99.0% | — | **87.8%** |
| **Vanilla TabDDPM** | 46.5% | 87.8% | 83.3% | 99.3% | 34.6% | 98.9% | 97.9% | — | — | 98.7% | — | **80.9%** |
| **CTGAN** | 91.5% | 47.9% | 38.4% | 99.7% | 67.6% | 95.7% | 75.9% | 6.0% | — | 93.7% | — | **68.5%** |
| **SMOGN** | 100.0% | 94.1% | 100.2% | 100.0% | 97.1% | 100.1% | 99.9% | 57.1% | — | 100.0% | 97.9% | **94.6%** |
| **Best** | SMOGN | SMOGN | SMOGN | SMOGN | SMOGN | SMOGN | SMOGN | SMOGN | — | SMOGN | SMOGN | |

---

## Utility — Augmentation Scenario
*Train on real + synthetic data, test on real data. Higher % = better.*

### Augmentation (% of Baseline)

| Method | Iris (4d) | California (8d) | Insurance (11d) | Maintenance (18d) | Steel (31d) | Bank (51d) | Credit (91d) | Supply Chain (9605d) | News (72d) | Adult (108d) | Ames Housing (361d) | **Avg** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our TabDDPM** | 103.7% | 97.6% | 99.0% | 100.0% | 97.2% | 99.7% | 99.7% | — | — | 99.8% | — | **99.6%** |
| **Vanilla TabDDPM** | 98.8% | 94.6% | 92.0% | 99.8% | 92.4% | 99.6% | 99.3% | — | — | 99.8% | — | **97.0%** |
| **CTGAN** | 98.8% | 92.0% | 90.8% | 100.0% | 95.8% | 99.5% | 99.6% | 84.9% | — | 99.4% | 94.0% | **95.5%** |
| **SMOGN** | 100.0% | 95.4% | 100.2% | 100.0% | 99.8% | 100.1% | 100.0% | 66.1% | — | 99.9% | 99.5% | **96.1%** |
| **Best** | Our | Our | SMOGN | CTGAN | SMOGN | SMOGN | SMOGN | CTGAN | — | SMOGN | SMOGN | |

---

## Ablation: Our Improvements vs Vanilla TabDDPM
*Shows the impact of MinMaxScaler, outlier clipping, and capacity scaling.*

### Ablation: Our Improvements vs Vanilla TabDDPM

| | Iris (4d) | California (8d) | Insurance (11d) | Maintenance (18d) | Steel (31d) | Bank (51d) | Credit (91d) | Supply Chain (9605d) | News (72d) | Adult (108d) | Ames Housing (361d) | **Avg** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our TabDDPM** | 48.8% | 92.2% | 91.1% | 100.0% | 74.1% | 99.1% | 98.3% | — | — | 99.0% | — | **87.8%** |
| **Vanilla TabDDPM** | 46.5% | 87.8% | 83.3% | 99.3% | 34.6% | 98.9% | 97.9% | — | — | 98.7% | — | **80.9%** |
| **Improvement** | +2.3% | +4.4% | +7.7% | +0.7% | +39.6% | +0.2% | +0.4% | — | — | +0.3% | — | **+6.9%** |

---

## Statistical Fidelity (Avg Wasserstein Distance)
*Lower = better distribution match.*

| Method | Iris | California | Insurance | Maintenance | Steel | Bank | Credit | Supply Chain | News | Adult | Ames Housing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our TabDDPM** | 395.5953 | 0.0568 | 0.0798 | 0.0304 | 84.8644 | 0.1710 | 200.6011 | 51371.5463 | 567.6021 | 0.0254 | 47447.1719 |
| **Vanilla TabDDPM** | 1968.0117 | 5.8566 | 7.9078 | 26.6270 | 1611.4284 | 0.4402 | 486.4718 | 51264.4666 | 2439.3509 | 51.5942 | 49713.1734 |
| **CTGAN** | 0.2049 | 0.0666 | 0.1678 | 0.0823 | 0.1263 | 0.0520 | 0.0332 | 0.0525 | 0.0495 | 0.0390 | 0.1442 |
| **SMOGN** | 0.0601 | 0.2156 | 0.0198 | 0.0068 | 0.0135 | 0.0069 | 0.0049 | 0.3637 | 0.1953 | 0.0076 | 0.2931 |

---

## Privacy (Membership Inference Attack AUC)
*AUC ~0.50 = safe (random guessing). AUC > 0.60 = privacy concern. AUC > 0.80 = critical leak (data is just copies).*

| Method | Iris | California | Insurance | Maintenance | Steel | Bank | Credit | Supply Chain | News | Adult | Ames Housing | **Avg** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our TabDDPM** | 0.4236 | 0.5106 | 0.5098 | 0.4814 | 0.4814 | 0.5026 | 0.4959 | 0.5050 | 0.5015 | 0.4963 | 0.5085 | **0.4924** |
| **Vanilla TabDDPM** | 0.4306 | 0.5032 | 0.5145 | 0.5123 | 0.4759 | 0.5032 | 0.4918 | 0.4962 | 0.4989 | 0.4960 | 0.4866 | **0.4917** |
| **CTGAN** | **0.6208** | 0.4932 | 0.5337 | 0.5160 | 0.4503 | 0.4930 | 0.5007 | 0.5093 | 0.4977 | 0.4949 | 0.4766 | **0.5078** |
| **SMOGN** | **0.7917** | **0.9132** | **0.8895** | **0.8211** | **0.8153** | **0.8179** | **0.8124** | **0.9102** | **0.9218** | **0.8086** | **0.9082** | **0.8554** |

---

## Figures

![Replacement Comparison](figures/replacement_comparison.png)

![Scaling Curve](figures/scaling_curve.png)

![Ablation](figures/ablation.png)

![Heatmap Replacement](figures/heatmap_replacement.png)

![Heatmap Augmentation](figures/heatmap_augmentation.png)

