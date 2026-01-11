# Experiment Plan: Manufacturing Duration Prediction

**Date:** 2025-01-11
**Status:** Planning Complete

## Background

The original quotation dataset (`6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx`) used in Experiments 007-008 predicted `İlk Girilen Teklif Miktarı` (Initial Quote Amount) with high baseline accuracy (R² ~ 0.92). While diffusion still outperformed SMOGN, the narrative was "safe augmentation" rather than "augmentation rescues poor models."

After discussion with the AI engineer, we identified the correct problem: **predicting manufacturing duration for 100,000 units** - a task where baseline models reportedly performed poorly.

## New Dataset Discovery

We found a unified source dataset:
- **File:** `Teklif Maliyet SSEK data.xlsx` (17,942 rows × 15 columns)
- **Location:** Both `standart-muhendislik` and `ozel-muhendislik` folders (identical copies)
- **Target:** `Makine Süre (100.000 ADET) DK` (Machine duration in minutes)

## Updated Thesis Narrative

1. **Original problem:** Predict manufacturing duration for bolt production
2. **Challenge:** Baseline models have moderate performance (~0.65 correlation)
3. **Failed attempt:** Traditional augmentation (SMOGN) hurts performance
4. **Solution:** Diffusion-based augmentation improves or maintains performance

## Experiment Roadmap

### Experiment 010: Baseline
- Establish baseline model performance
- Features: Çap, Boy (numeric) + Sartname, Kısa tanım, ÜY, İş Yeri (categorical)
- Models: RF, GB, Ridge
- Goal: Confirm this is a "hard" prediction problem

### Experiment 011: Augmentation Comparison
- Train HybridDiffusion model
- Compare: Original vs +Diffusion vs +SMOGN vs +Noise
- Goal: Show diffusion beats SMOGN

## Key Decisions

### Feature Selection
**Use:**
- Çap (diameter) - extracted from material description
- Boy (length) - extracted from material description
- Sartname (specification) - categorical
- Kısa tanım (machine recipe) - categorical
- ÜY (production location) - categorical
- İş Yeri (workplace) - categorical

**Do NOT Use (Data Leakage):**
- All cost columns (Direkt İşicilik, Endirekt Amortisman, etc.)
- These are calculated from machine time, creating circular dependency

### Categorical Encoding
- Group rare categories (< 50 occurrences) into "Other"
- Use one-hot encoding for diffusion training
- Use label encoding for ML models

## Connection to Previous Experiments

| Experiment | Dataset | Target | Baseline | Augmentation Helps? |
|------------|---------|--------|----------|---------------------|
| 007 | Production (numeric) | Quote Amount | R² = 0.92 | No (already good) |
| 008 | Production (full) | Quote Amount | R² = 0.92 | Marginal (-1.3% RMSE) |
| **010** | Manufacturing | Machine Duration | TBD (expect ~0.5-0.7) | TBD |
| **011** | Manufacturing | Machine Duration | - | Expected: Yes |

## Expected Impact on Thesis

If successful, Experiments 010-011 will:
1. Provide the "poor baseline → augmentation helps" narrative
2. Demonstrate diffusion's advantage over SMOGN on complex data
3. Validate the approach on a real business problem
4. Strengthen the thesis conclusion

## Files Created

- `docs/manufacturing-duration-dataset.md` - Dataset analysis
- `todo/010-manufacturing-duration-baseline.md` - Baseline experiment plan
- `todo/011-manufacturing-duration-augmentation.md` - Augmentation experiment plan
- `docs/experiment-plan-manufacturing.md` - This document
