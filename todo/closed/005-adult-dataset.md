# Adult Dataset Experiment

**Source:** Experiment planning
**Depends on:** 001-multinomial-diffusion (categorical features required)

## Task

Run diffusion vs SMOTE comparison on Adult (Census Income) dataset.

## Why This Dataset

- **Mixed types:** Categorical + numerical features
- **Standard benchmark:** Used in TabDDPM, STaSy, TabSyn papers
- **Class imbalance:** ~75% <=50K, ~25% >50K
- **Real challenge:** Tests if diffusion handles mixed data better than SMOTE

## Dataset Info

- **Source:** UCI ML Repository / sklearn
- **Samples:** 48,842
- **Features:** 14 (6 numerical, 8 categorical)
- **Target:** Income >50K or <=50K (binary classification)

## Experiment Plan

1. Implement multinomial diffusion (todo 001)
2. Prepare Adult dataset with proper encoding
3. Train hybrid diffusion model (Gaussian + Multinomial)
4. Compare: Real, Diffusion, SMOTE, Augmented-Diffusion, Augmented-SMOTE
5. Generate report with confusion matrices

## Status
- [ ] Waiting for multinomial diffusion implementation
