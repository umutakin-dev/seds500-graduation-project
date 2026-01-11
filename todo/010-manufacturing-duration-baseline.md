# Experiment 010: Manufacturing Duration - Baseline

**Priority:** High
**Depends on:** Dataset preparation

## Objective

Establish baseline model performance on the manufacturing duration prediction task. This is the dataset where the AI engineer reported unsatisfactory results, making it the ideal candidate for showing augmentation benefits.

## Dataset

- **Source:** `Teklif Maliyet SSEK data.xlsx`
- **Rows:** 17,942
- **Target:** `Makine Süre (100.000 ADET) DK` (Machine duration for 100k units in minutes)
- **Analysis:** See `docs/manufacturing-duration-dataset.md`

## Features to Use

### Numeric (2)
- Çap (diameter, extracted from material description)
- Boy (length, extracted from material description)

### Categorical (4)
- Sartname (specification, 191 unique → group rare categories)
- Kısa tanım (machine recipe, 136 unique)
- ÜY (production location, 2 unique)
- İş Yeri (workplace, 144 unique → group rare categories)

### DO NOT USE (Data Leakage)
- All cost columns (Direkt İşicilik, Endirekt İşçilik, etc.)
- These are derived from the target variable

## Tasks

- [ ] Copy data to `data/manufacturing/` (gitignored)
- [ ] Create `src/prepare_manufacturing_data.py` script
  - [ ] Extract Çap and Boy from Malzeme Tanımı
  - [ ] Handle rare categories (group into "Other")
  - [ ] Train/test split (80/20)
  - [ ] Scale numeric features
  - [ ] Encode categorical features
- [ ] Create `src/evaluate_manufacturing_baseline.py` script
- [ ] Run baseline models:
  - [ ] Random Forest
  - [ ] Gradient Boosting
  - [ ] Ridge Regression
- [ ] Document results in `experiments/experiment-010-manufacturing-baseline/RESULTS.md`

## Expected Outcome

We expect baseline models to have moderate performance (R² ~ 0.5-0.7) given the feature-target correlations (~0.65). This leaves room for augmentation to potentially improve results.

## Success Criteria

- Establish baseline RMSE and R² scores
- Confirm this is a "hard" prediction problem where augmentation could help
- Prepare data in format ready for diffusion model training

## Notes

- This is a regression task (predicting continuous duration)
- Target ranges from 333 to 2,274 minutes
- Larger bolts (M24) take ~2.2x longer than smaller ones (M6)
