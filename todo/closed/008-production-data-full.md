# Experiment 008: Production Data - Full (HybridDiffusion)

## Objective
Test HybridDiffusion (Gaussian + Multinomial) on production data with mixed feature types.

## Dataset
- Source: `6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx`
- Rows: 5,370 (after removing missing targets)
- Features: 6 numeric + 35 categorical = 117 dimensions (one-hot)
- Target: `İlk Girilen Teklif Miktarı` (Initial Quote Amount)

## Results

**Diffusion beats SMOGN on ALL models:**

| Model | Diffusion RMSE | SMOGN RMSE | Better By |
|-------|----------------|------------|-----------|
| Random Forest | 0.0891 | 0.0985 | **9.5%** |
| Gradient Boosting | 0.0975 | 0.1114 | **12.5%** |
| Ridge | 0.1157 | 0.1265 | **8.5%** |

**Diffusion improves over original data:**
- RF: 0.0903 → 0.0891 (-1.3%) ✅
- GB: 0.0982 → 0.0975 (-0.7%) ✅

**SMOGN fails completely** - makes everything worse

## Key Findings
1. Categorical features improve baseline (vs numeric-only)
2. HybridDiffusion actually improves RF and GB
3. SMOGN fails where diffusion succeeds
4. Thesis validated: diffusion works on real production data

## Files
- `src/train_production_full.py` - Training script
- `src/evaluate_production_full.py` - Evaluation with SMOGN
- `experiments/experiment-008-production-full/RESULTS.md` - Full results

## Status
- [x] Complete multinomial diffusion implementation
- [x] Hybrid diffusion model ready
- [x] Train on production data (500 epochs, loss 0.29)
- [x] Run evaluation with SMOGN comparison
- [x] Compare with baseline (Experiment 007)
- [x] Document findings
- [x] PR #4 merged (2026-01-11)
