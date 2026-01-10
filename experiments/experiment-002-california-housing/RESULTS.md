# Experiment 002: California Housing

## Dataset
- **Source**: sklearn California Housing
- **Training samples**: 16,512
- **Test samples**: 4,128
- **Features**: 8 numeric + 1 target (median house value)
- **Task**: Regression

## Model Configuration
- **Architecture**: MLPDenoiser (256-256-256)
- **Epochs**: 1000
- **Learning rate**: 0.0005
- **Diffusion timesteps**: 1000
- **Beta schedule**: linear

## Training Results
- Final loss: 0.2074
- Mean absolute correlation difference: 0.0213 (good quality)

## ML Efficiency Evaluation

### Ridge Regression
| Method | R² | RMSE | vs Baseline |
|--------|-----|------|-------------|
| Real → Real (baseline) | 0.5759 | 0.7455 | - |
| Diffusion → Real | 0.4103 | 0.8791 | -16.55% |
| Noise → Real | 0.5833 | 0.7390 | +0.74% |
| Augmented-Diffusion | 0.5720 | 0.7489 | -0.39% |
| Augmented-Noise | 0.5802 | 0.7417 | +0.44% |

**Winner**: Noise (+0.82% R² over Diffusion)

### Random Forest
| Method | R² | RMSE | vs Baseline |
|--------|-----|------|-------------|
| Real → Real (baseline) | 0.8051 | 0.5053 | - |
| Diffusion → Real | 0.7447 | 0.5783 | -6.04% |
| Noise → Real | 0.7668 | 0.5529 | -3.84% |
| Augmented-Diffusion | 0.7931 | 0.5206 | -1.20% |
| Augmented-Noise | 0.7935 | 0.5202 | -1.16% |

**Winner**: Tie (similar performance)

## Key Findings

1. **Large dataset = limited augmentation benefit**: With 16K samples, augmentation doesn't help much
2. **Diffusion vs Noise**: Similar performance, neither shows clear advantage
3. **Synthetic-only training hurts**: Training only on synthetic data significantly underperforms
4. **Augmentation is neutral**: Neither helps nor hurts significantly

## Comparison with Experiment 001 (Iris)

| Aspect | Iris | California |
|--------|------|------------|
| Samples | 120 | 16,512 |
| Features | 4 | 9 |
| Task | Classification | Regression |
| Diffusion vs SMOTE/Noise | Tie | Tie |

## Conclusion

On this larger regression dataset, diffusion-based augmentation shows similar performance to simple Gaussian noise augmentation. The baseline with real data alone achieves strong results (R²=0.8051 with Random Forest), leaving little room for augmentation to improve.

This suggests diffusion augmentation may be more valuable in:
- Low-data regimes
- Complex data distributions
- Datasets where simple noise augmentation fails
