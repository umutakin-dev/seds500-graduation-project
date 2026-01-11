# Experiment 007: Production Data - Numeric Baseline

## Objective
Test Gaussian diffusion on production data using only numeric features as a simplified baseline.

## Dataset
- Source: `6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx` (quotation documents)
- Rows: 5,533 → 5,370 (after removing missing targets)
- Train/Test: 4,296 / 1,074
- Features: 13 numeric + 1 target
- Target: `İlk Girilen Teklif Miktarı` (Initial Quote Amount) - 97% filled

## Results
- **Final loss**: 0.1197 (500 epochs)
- **Baseline R²**: 0.92 (Random Forest)
- **Diffusion vs Noise**: Diffusion wins on 2/3 models (GB, Ridge)
- **Conclusion**: Original data sufficient, augmentation provides marginal benefit

| Model | Original | + Diffusion | + Noise |
|-------|----------|-------------|---------|
| Random Forest | 0.0909 | 0.0914 | 0.0890 |
| Gradient Boosting | 0.0958 | 0.1004 | 0.1042 |
| Ridge | 0.1233 | 0.1236 | 0.1241 |

## Files
- `src/prepare_production_data.py` - Data preparation
- `src/train_production_numeric.py` - Training script
- `src/evaluate_production_numeric.py` - Evaluation script
- `experiments/experiment-007-production-numeric/RESULTS.md` - Full results

## Status
- [x] Create data preparation script
- [x] Extract and clean numeric features
- [x] Train diffusion model (500 epochs)
- [x] Run evaluation
- [x] Document findings
- [x] PR #3 merged (2026-01-11)
