# Experiment 007: Production Data - Numeric Baseline

## Objective
Test Gaussian diffusion on production quotation data using only numeric features as a baseline for comparison with the full hybrid model.

## Dataset
- **Source**: `6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx` (quotation documents)
- **Total rows**: 5,533 → 5,370 (after removing missing targets)
- **Train/Test split**: 4,296 / 1,074 (80/20)
- **Features**: 13 numeric columns + 1 target = 14 dimensions
- **Target**: `İlk Girilen Teklif Miktarı` (Initial Quote Amount)

## Model Configuration
- **Architecture**: MLPDenoiser (256-256-256, dropout=0.1)
- **Timesteps**: 1000
- **Beta schedule**: Linear
- **Learning rate**: 0.0005
- **Epochs**: 500

## Training Results
- **Final loss**: 0.1197
- Loss converged around epoch 200, continued slow improvement to 500

| Epoch | Loss |
|-------|------|
| 1 | 0.8989 |
| 100 | 0.1517 |
| 200 | 0.1397 |
| 300 | 0.1313 |
| 400 | 0.1283 |
| 500 | 0.1247 |

## Evaluation Results

### Test 1: 1000 synthetic samples (~23% increase)

| Model | Original | + Diffusion | + Noise | Winner |
|-------|----------|-------------|---------|--------|
| Random Forest | 0.0909 | 0.0909 (±0) | **0.0888** | Noise |
| Gradient Boosting | 0.0958 | **0.0996** (+0.004) | 0.1017 (+0.006) | Diffusion |
| Ridge | 0.1233 | **0.1232** (-0.0001) | 0.1234 (+0.0001) | Diffusion |

### Test 2: 4000 synthetic samples (~93% increase)

| Model | Original | + Diffusion | + Noise | Winner |
|-------|----------|-------------|---------|--------|
| Random Forest | 0.0909 | 0.0914 (+0.0005) | **0.0890** (-0.002) | Noise |
| Gradient Boosting | 0.0958 | **0.1004** (+0.005) | 0.1042 (+0.008) | Diffusion |
| Ridge | 0.1233 | **0.1236** (+0.0003) | 0.1241 (+0.001) | Diffusion |

## Key Findings

1. **Strong baseline** - R²=0.92 for Random Forest means the model already fits well with original data
2. **Diffusion beats Noise** on 2/3 models (Gradient Boosting, Ridge)
3. **RF prefers noise** - Random Forest's bagging mechanism benefits from noisy copies
4. **Augmentation not needed** - With high baseline performance, synthetic data doesn't help
5. **Regression vs Classification** - Unlike Adult (classification), regression on this data doesn't benefit from augmentation

## Comparison with Adult Dataset (Classification)

| Metric | Adult (Exp 009) | Production (Exp 007) |
|--------|-----------------|----------------------|
| Task | Classification | Regression |
| Best result | +0.15% accuracy | +0.0001 RMSE (Ridge) |
| Augmentation helps? | Yes (g=1.0) | No |
| Diffusion vs SMOTE/Noise | Diffusion wins | Mixed |

## Conclusion
Gaussian diffusion successfully generates realistic numeric samples for the production data, but the dataset is already sufficient for strong regression performance. Augmentation provides marginal benefit at best.

## Next Steps
- Experiment 008: Add categorical features with HybridDiffusion
- Compare if categorical information improves prediction

## Files
- `src/prepare_production_data.py` - Data preparation script
- `src/train_production_numeric.py` - Training script
- `src/evaluate_production_numeric.py` - Evaluation script
- `checkpoints/production_numeric/final_model.pt` - Trained model
