# Experiment 017: Three-Way Method Comparison

## Objective

Compare Diffusion, CTGAN, and SMOGN across two use cases:
1. **Augmentation**: Add synthetic data to original training data
2. **Replacement**: Use only synthetic data (for privacy/data sharing)

## Dataset

- **Source**: Ozel Rich (Experiment 013)
- **Train**: 2,136 samples, 28 features (2 numeric + 26 one-hot)
- **Test**: 534 samples
- **Synthetic samples**: 500 for augmentation, full size for replacement

## Methods

| Method | Type | Approach |
|--------|------|----------|
| Diffusion | Generative | HybridDiffusion (Gaussian + Multinomial) |
| CTGAN | GAN | Conditional Tabular GAN |
| SMOGN | Interpolation | SMOTE for Regression |

## Results

### Augmentation Scenario (Original + Synthetic)

| Method | R² | Delta from Baseline | % of Baseline |
|--------|-----|---------------------|---------------|
| Original (baseline) | 0.6451 | - | 100% |
| **Diffusion V6** | **0.6355** | **-0.0095** | **98.5%** |
| CTGAN | 0.6310 | -0.0140 | 97.8% |
| SMOGN | -0.1354 | -0.7805 | N/A |

**Winner: Diffusion** - Maintains baseline quality (98.5%) while SMOGN catastrophically fails.

### Replacement Scenario (Synthetic Only)

| Method | R² | Delta from Baseline | % of Baseline |
|--------|-----|---------------------|---------------|
| Original (baseline) | 0.6451 | - | 100% |
| **CTGAN** | **0.2292** | **-0.4158** | **35.5%** |
| Diffusion V6 (retrained) | 0.1712 | -0.4738 | 26.5% |
| Diffusion (bug fix only) | -0.0032 | -0.6483 | ~0% |
| Diffusion (before fix) | -14.0086 | -14.6536 | N/A |
| SMOGN | N/A | N/A | N/A |

**Winner: CTGAN** - Best method for standalone synthetic data (35.5% of baseline).
**Runner-up: Diffusion V6** - After bug fix and retraining, achieves 26.5% of baseline (75% of CTGAN's performance).

## Key Findings

### 1. Use Case Determines Best Method

| Use Case | Best Method | Recommendation |
|----------|-------------|----------------|
| Augmentation | Diffusion | Use when you have original data and want more |
| Replacement/Privacy | CTGAN | Use when sharing data without originals |

### 2. SMOGN Fails on Complex Data

SMOGN showed catastrophic failure (-0.14 R²) on this 28-dimensional dataset, confirming the pattern from Experiment 013.

### 3. Diffusion Had Bug in Categorical Sampling (FIXED)

**Root Cause Identified**: At high timesteps (t near T), the categorical reverse process had a numerical bug:
- `alpha_t` approaches 0, making `ratio = alpha_t_prev / alpha_t` explode (e.g., ratio=242 at t=999)
- This caused categorical features to extrapolate outside [0,1], corrupting the entire distribution
- After one sampling step, the model saw inputs it never trained on, leading to cascading failure

**Fix Applied** (diffusion.py line 425):
```python
# OLD (buggy): ratio = alpha_t_prev / alpha_t.clamp(min=1e-8)
# FIX: Clamp ratio to [0, 1] to keep it as interpolation
ratio = (alpha_t_prev / alpha_t.clamp(min=1e-8)).clamp(max=1.0)
```

**Results After Bug Fix and Retraining**:
| Metric | Before Fix | After Fix | After Retrain (V6) |
|--------|------------|-----------|-------------------|
| Replacement R² | -14.0086 | -0.0032 | **0.1712** |
| Generated mean direction | Wrong (positive) | Correct (negative) | Correct (negative) |
| At boundary (>0.95) | 99%+ | <15% | <15% |
| % of CTGAN performance | N/A | ~0% | **75%** |

The bug fix enabled proper sampling. Retraining with the fix improved diffusion to achieve 75% of CTGAN's replacement performance.

## Interpretation

### Why Diffusion Works for Augmentation but Not Replacement

When used for **augmentation**:
- Original data (2136 samples) provides the "ground truth"
- 500 synthetic samples add variation without dominating
- Even if some synthetic samples are poor, original data compensates

When used for **replacement**:
- No anchor from original data
- Model collapse becomes apparent
- Generated samples don't capture true distribution

### Why CTGAN Works for Replacement

CTGAN uses:
- Mode-specific normalization for continuous columns
- Conditional generation preserves feature relationships
- Discriminator provides feedback on sample quality

This makes it more robust for standalone synthetic data generation.

## Practical Implications

### For Data Augmentation
Use **Diffusion**:
- Safe default (never worse than baseline by more than 1%)
- Handles mixed numeric/categorical data
- Avoids SMOGN's catastrophic failures

### For Privacy-Preserving Data Sharing
Use **CTGAN**:
- Generates realistic standalone synthetic data
- Models trained on CTGAN synthetic data achieve 35% of original R²
- Partners can train ML models that work on real data

### Avoid SMOGN for Complex Data
SMOGN should not be used for:
- High-dimensional feature spaces (>20 dimensions)
- Complex categorical structures
- Privacy-sensitive applications

## Future Work

1. ~~**Investigate diffusion collapse**: Debug why the model produces extreme values~~ **DONE** - Found and fixed categorical ratio explosion bug
2. **Further hyperparameter tuning**: Test different architectures to close gap with CTGAN
3. **Hybrid approach**: Use CTGAN for replacement, Diffusion for augmentation (recommended based on findings)

## Files

- `src/train_ctgan.py` - CTGAN training script
- `src/compare_augmentation_vs_replacement.py` - Comparison script
- `checkpoints/ctgan/ozel_rich_model.pkl` - Trained CTGAN model
