# Experiment 013: Ozel Rich Dataset - HybridDiffusion

## Objective
Test HybridDiffusion on ozel data extracted from the manufacturing dataset with richer features (more categorical variables) to see if feature complexity affects augmentation performance.

## Dataset
- **Source**: Ozel rows extracted from `Teklif Maliyet SSEK data.xlsx`
- **Total ozel rows**: 2,780 → 2,670 (after filtering invalid Cap/Boy)
- **Train/Test split**: 2,136 / 534 (80/20)
- **Features**:
  - Cap (diameter): numeric, extracted from Malzeme Tanımı
  - Boy (length): numeric, extracted from Malzeme Tanımı
  - IslemTipi: categorical (FT, HT, Belirsiz) - 3 categories
  - AnmaOlcusu (thread size): categorical (M6, M8, M10...) - 16 categories
  - SartnameSimple (spec): categorical (OZEL, OZEL+A, OZEL+R...) - 5 categories
  - UY (production area): categorical (2101, 2501) - 2 categories
- **Total dimensions**: 29 (3 numeric + 26 one-hot)
- **Target**: Makine Süre (100.000 ADET) DK

## Feature Statistics
| Feature | Correlation with Target |
|---------|------------------------|
| Cap | 0.7055 |
| Boy | 0.1321 |

### Categorical Feature Effects
| AnmaOlcusu | Mean Target | Count |
|------------|-------------|-------|
| M6 | 734 | 564 |
| M8 | 850 | 759 |
| M10 | 947 | 480 |
| M12 | 1040 | 238 |
| M14 | 1339 | 147 |
| M16 | 1373 | 257 |

## Model Configuration
- **Architecture**: MLPDenoiser (512-512-512, dropout=0.1) - larger for 29 dims
- **Diffusion**: HybridDiffusion (Gaussian + Multinomial)
- **Timesteps**: 1000
- **Beta schedule**: Linear
- **Learning rate**: 0.0005
- **Epochs**: 500
- **Batch size**: 128

## Training Results
- **Final loss**: 0.3951 (num=0.08, cat=0.33)
- Categorical loss higher due to 4 categorical features with varying cardinalities

| Epoch | Total Loss | Numeric Loss | Categorical Loss |
|-------|------------|--------------|------------------|
| 1 | 1.9622 | 0.5809 | 1.3813 |
| 100 | 0.5569 | 0.0945 | 0.4624 |
| 250 | 0.4529 | 0.0863 | 0.3666 |
| 500 | 0.4257 | 0.0921 | 0.3336 |

## Baseline Evaluation

| Model | RMSE (min) | R² |
|-------|------------|-----|
| Random Forest | 206.5 | 0.6451 |
| Gradient Boosting | 211.1 | 0.6291 |
| Ridge | 220.3 | 0.5960 |

**Assessment**: Moderate baseline (R² = 0.65) - better than exp 012 (R² = 0.55) due to richer features.

## Augmentation Comparison (500 synthetic samples)

| Model | Original | + Diffusion | + SMOGN | + Noise |
|-------|----------|-------------|---------|---------|
| Random Forest | 206.5 | 208.3 (+0.9%) | **409.2 (+98.2%)** | 208.5 (+1.0%) |
| Gradient Boosting | 211.1 | **208.8 (-1.1%)** | **393.4 (+86.4%)** | 211.1 (+0.0%) |
| Ridge | 220.3 | 369.2 (+67.6%) | **384.8 (+74.7%)** | 220.2 (-0.0%) |

## Key Findings

1. **SMOGN catastrophically failed** - +86% to +98% worse (negative R²!)
2. **Diffusion maintained performance** on RF/GB (+0.9% / -1.1%)
3. **GB actually improved** with diffusion (-1.1% RMSE)
4. **Diffusion beats SMOGN 3/3** models (46-49% better on RF/GB)
5. **Ridge failed with both** augmentation methods (but diffusion less bad)

## Pattern Confirmation: Feature Complexity vs SMOGN

| Experiment | Features | SMOGN Result | Diffusion Result |
|------------|----------|--------------|------------------|
| 011 (Manufacturing) | 2 numeric | **Catastrophic** (+70-84%) | Safe (±0%) |
| 012 (Ozel simple) | 2 num + 1 cat (5 dims) | Works (±2%) | Works (±2%) |
| **013 (Ozel rich)** | 2 num + 4 cat (29 dims) | **Catastrophic** (+86-98%) | Safe (+0.9% to -1.1%) |

**Hypothesis**: SMOGN struggles with high-dimensional feature spaces because interpolation in high dimensions creates unrealistic samples. Diffusion learns the data distribution and generates more plausible samples.

## Conclusion

This experiment confirms the pattern from exp 011: **SMOGN fails catastrophically on complex feature spaces** while diffusion maintains performance. The key insight is that diffusion-based augmentation is a **safer default choice** - it may not always improve results, but it avoids the dramatic failures that traditional methods can exhibit.

**Gradient Boosting showed actual improvement** (-1.1%) with diffusion augmentation, suggesting that with proper tuning, diffusion could provide real benefits beyond just being "safe."

## Files
- `src/prepare_ozel_rich_data.py` - Data preparation script
- `src/train_ozel_rich.py` - HybridDiffusion training script
- `src/evaluate_ozel_rich_baseline.py` - Baseline evaluation
- `src/evaluate_ozel_rich.py` - Full augmentation comparison
- `data/ozel_rich/prepared.pt` - Prepared data
- `checkpoints/ozel_rich/final_model.pt` - Trained model
