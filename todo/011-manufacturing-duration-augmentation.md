# Experiment 011: Manufacturing Duration - Diffusion Augmentation

**Priority:** High
**Depends on:** Experiment 010 (baseline)

## Objective

Test whether HybridDiffusion (Gaussian + Multinomial) can improve model performance on the manufacturing duration prediction task, and compare against SMOGN (traditional augmentation).

## Hypothesis

Based on previous experiments (Exp 008), we expect:
1. SMOGN will fail or hurt performance on this complex manufacturing data
2. Diffusion-based augmentation will improve or maintain performance
3. Diffusion will outperform SMOGN by 8-12%

## Dataset

Same as Experiment 010:
- **Rows:** 17,942
- **Features:** 2 numeric (Çap, Boy) + 4 categorical (Sartname, Kısa tanım, ÜY, İş Yeri)
- **Target:** Machine duration (minutes for 100k units)

## Tasks

### Training
- [ ] Create `src/train_manufacturing.py` script
- [ ] Train HybridDiffusion model:
  - Gaussian diffusion for numeric features (Çap, Boy, target)
  - Multinomial diffusion for categorical features
  - Architecture: HybridMLPDenoiser (256-256-256)
  - Timesteps: 1000, Beta schedule: Linear
  - Epochs: 500

### Evaluation
- [ ] Create `src/evaluate_manufacturing.py` script
- [ ] Generate synthetic samples (1000, 2000, 4000)
- [ ] Compare augmentation methods:
  - Original only (baseline from Exp 010)
  - Original + Diffusion samples
  - Original + Noise (random baseline)
  - Original + SMOGN samples
- [ ] Evaluate with:
  - Random Forest
  - Gradient Boosting
  - Ridge Regression

### Documentation
- [ ] Create `experiments/experiment-011-manufacturing-augmentation/RESULTS.md`
- [ ] Document key findings for thesis

## Metrics

- RMSE (primary - lower is better)
- R² (secondary - higher is better)
- Relative improvement over baseline (%)
- Diffusion vs SMOGN comparison (%)

## Expected Results

| Method | Expected RMSE Change |
|--------|---------------------|
| Original (baseline) | - |
| + Diffusion | -2% to -5% (improvement) |
| + SMOGN | +5% to +15% (degradation) |
| + Noise | +2% to +5% (slight degradation) |

## Success Criteria

1. Diffusion augmentation improves at least one model's RMSE
2. Diffusion outperforms SMOGN on all models
3. Results support thesis: "Diffusion beats traditional augmentation on complex manufacturing data"

## Notes

- This experiment is crucial for the thesis narrative
- If baseline is already very good (R² > 0.9), augmentation may not help much
- Focus on relative comparison: Diffusion vs SMOGN
