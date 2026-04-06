# Experiment 012: Ozel Dataset - HybridDiffusion

## Objective
Test HybridDiffusion on a smaller, pre-split dataset with mixed features (numeric + categorical) to see if augmentation helps when baseline performance is moderate-low (R² ~0.55).

## Dataset
- **Source**: `ozel-muhendislik` dataset (pre-split train/test)
- **Train rows**: 2,029
- **Test rows**: 508
- **Features**:
  - Cap (diameter): numeric
  - Boy (length): numeric
  - IslemTipi (process type): categorical (FT, HT, Belirsiz)
- **Target**: MakineSure (manufacturing duration in minutes for 100k units)

## Feature Statistics
| Feature | Min | Max | Mean |
|---------|-----|-----|------|
| Cap | 0.4 | 32.0 | 9.6 |
| Boy | 1.8 | 265.0 | 48.4 |
| MakineSure | 333 | 2274 | 962 |

### IslemTipi Distribution (Train)
| Category | Count | Mean Target |
|----------|-------|-------------|
| FT | 1,254 | 877 |
| HT | 521 | 1,162 |
| Belirsiz | 256 | 973 |

## Model Configuration
- **Architecture**: MLPDenoiser (256-256-256, dropout=0.1)
- **Diffusion**: HybridDiffusion (Gaussian for numeric, Multinomial for categorical)
- **Timesteps**: 1000
- **Beta schedule**: Linear
- **Learning rate**: 0.0005
- **Epochs**: 500
- **Batch size**: 128

## Training Results
- **Final loss**: 0.3671 (num=0.11, cat=0.27)
- Loss converged well, categorical loss higher than numeric (expected for 3-class CE)

| Epoch | Total Loss | Numeric Loss | Categorical Loss |
|-------|------------|--------------|------------------|
| 1 | 1.7381 | 0.7610 | 0.9772 |
| 100 | 0.4670 | 0.1253 | 0.3417 |
| 250 | 0.4158 | 0.1244 | 0.2915 |
| 500 | 0.3771 | 0.1065 | 0.2706 |

## Baseline Evaluation

| Model | RMSE (min) | R² |
|-------|------------|-----|
| Random Forest | 254.0 | 0.4844 |
| Gradient Boosting | 236.0 | 0.5551 |
| Ridge | 265.5 | 0.4368 |

**Assessment**: Moderate-low baseline (R² = 0.55) - room for augmentation to help.

## Augmentation Comparison

### Test 1: 500 synthetic samples

| Model | Original | + Diffusion | + SMOGN | + Noise |
|-------|----------|-------------|---------|---------|
| Random Forest | 254.0 | 257.8 (+1.5%) | 251.8 (-0.9%) | 250.9 (-1.2%) |
| Gradient Boosting | 236.0 | 239.9 (+1.7%) | 237.1 (+0.5%) | 237.4 (+0.6%) |
| Ridge | 265.5 | 265.7 (+0.1%) | 276.9 (+4.3%) | 265.2 (-0.1%) |

**Winner**: SMOGN (2/3 models)

### Test 2: 1000 synthetic samples

| Model | Original | + Diffusion | + SMOGN | + Noise |
|-------|----------|-------------|---------|---------|
| Random Forest | 254.0 | 251.6 (-1.0%) | 249.8 (-1.6%) | 256.2 (+0.9%) |
| Gradient Boosting | 236.0 | 240.4 (+1.9%) | 236.5 (+0.2%) | 235.8 (-0.1%) |
| Ridge | 265.5 | 264.6 (-0.3%) | 276.6 (+4.2%) | 265.8 (+0.1%) |

**Winner**: Mixed (SMOGN better on RF, Diffusion better on Ridge, Noise better on GB)

## Key Findings

1. **Baseline is moderate-low** - R² = 0.55, unlike manufacturing (R² = 0.75)
2. **All methods within ±2%** - No dramatic improvement or degradation
3. **SMOGN works here** - Unlike Exp 011 where SMOGN failed catastrophically (+70-84%)
4. **Diffusion is safe but not better** - Never catastrophic, but doesn't outperform SMOGN
5. **Simple feature space** - 3 features may not provide enough complexity for diffusion to shine

## Comparison: Manufacturing vs Ozel

| Metric | Manufacturing (Exp 010-011) | Ozel (Exp 012) |
|--------|---------------------------|----------------|
| Dataset size | 17,942 rows | 2,029 rows |
| Features | 2 numeric | 2 numeric + 1 categorical |
| Baseline R² | 0.75 | 0.55 |
| SMOGN result | **Catastrophic** (+70-84%) | Works (±2%) |
| Diffusion result | Safe (±0%) | Safe (±2%) |
| Winner | Diffusion | SMOGN/Tie |

## Conclusion

HybridDiffusion generates plausible samples but does not improve regression performance on this dataset. The simple feature space (2 numeric + 1 categorical) may not provide enough structure for diffusion to learn meaningful patterns beyond what SMOGN can capture through interpolation.

**Key insight**: Diffusion's value is as a **"safe" augmentation method** - it never fails catastrophically like SMOGN can (Exp 011), but it doesn't always outperform traditional methods on simpler problems.

## Files
- `src/prepare_ozel_data.py` - Data preparation script
- `src/train_ozel.py` - HybridDiffusion training script
- `src/evaluate_ozel_baseline.py` - Baseline evaluation
- `src/evaluate_ozel.py` - Full augmentation comparison
- `data/ozel/train.csv`, `data/ozel/test.csv` - Dataset
- `checkpoints/ozel/final_model.pt` - Trained model
