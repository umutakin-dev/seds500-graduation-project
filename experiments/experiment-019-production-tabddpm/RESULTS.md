# Experiment 019: TabDDPM on Production Data

## Date
2026-01-16

## Summary

**TabDDPM achieves 98.4% of baseline for replacement scenario on Production data** - outperforming Ozel Rich (87.3%).

## Dataset

- **Source**: Production quotation documents (6 months)
- **Samples**: 5,370 total (4,296 train / 1,074 test)
- **Numerical features**: 6 + target = 7
- **Categorical features**: 30 (after removing 5 constant features)
- **Target**: İlk Girilen Teklif Miktarı (Initial Quote Amount)

## Training

- **Epochs**: 1000
- **Best Loss**: 0.1657 (at epoch 881)
- **Device**: CUDA (RTX 4070 Ti Super)
- **Model**: MLP [512, 512, 512, 512] (larger than Ozel Rich due to 5x more input dims)
- **Training Time**: ~30 minutes

### Convergence

| Epoch | Loss | Gen Mean | At Boundary |
|-------|------|----------|-------------|
| 1 | 1.07 | thousands | [1.0, 1.0, 1.0] |
| 350 | 0.30 | [-15, 7, -13] | [0.97, 0.96, 0.90] |
| 500 | 0.22 | [0.11, -0.95, -0.44] | [0.13, 0.47, 0.04] |
| 650 | 0.19 | [-0.10, -0.77, -0.39] | [0.01, 0.25, 0.00] |
| 881 | 0.17 | converged | low |

## Results

### Main Results

| Scenario | RF R² | vs Baseline | % of Baseline |
|----------|-------|-------------|---------------|
| Baseline | 0.9940 | - | 100% |
| Augmentation | 0.9936 | -0.0004 | **100.0%** |
| Replacement | 0.9785 | -0.0155 | **98.4%** |

### All ML Models

| Scenario | RF R² | GB R² | Ridge R² |
|----------|-------|-------|----------|
| Baseline | 0.9940 | 0.9937 | 0.9887 |
| Replacement | 0.9785 (-0.02) | 0.9780 (-0.02) | 0.9557 (-0.03) |
| Augmentation | 0.9936 (-0.00) | 0.9890 (-0.00) | 0.9783 (-0.01) |

### Comparison with Ozel Rich (Experiment 018)

| Dataset | Baseline R² | Replacement % | Augmentation % |
|---------|-------------|---------------|----------------|
| Ozel Rich | 0.6451 | 87.3% | 99.1% |
| **Production** | 0.9940 | **98.4%** | **100.0%** |

Production outperforms Ozel Rich in the replacement scenario despite being more complex (7 numerical + 30 categorical vs 3 numerical + 4 categorical).

## Key Findings

### 1. Scaling Matters

Initial training failed due to QuantileTransformer + clipping breaking invertibility:

| Scaler | Issue | Result |
|--------|-------|--------|
| QuantileTransformer + clip | Non-linear mapping broken by clipping | Generated targets 30x off |
| MinMaxScaler | Linear, perfectly invertible | Works correctly |

### 2. Outlier Clipping Required

Before outlier clipping (1st-99th percentile):
- Target feature: mean=-0.985, std=0.082 (compressed to boundary)
- Training stuck, poor convergence

After outlier clipping:
- Target feature: mean=-0.902, std=0.274 (well distributed)
- Training converged properly

### 3. Model Capacity Scaling

Production required larger model due to 5x more input dimensions:
- Ozel Rich: 23 input dims → [256, 256, 256]
- Production: 112 input dims → [512, 512, 512, 512]

### 4. High Baseline is Easier

Production's high baseline (0.994) made the replacement task easier:
- Strong patterns in data = easier to learn
- Ozel Rich's lower baseline (0.645) was harder to match

## Lessons Learned

1. **Use MinMaxScaler for diffusion** - QuantileTransformer is incompatible
2. **Clip outliers before scaling** - prevents boundary compression
3. **Scale model with input size** - more features need more capacity
4. **Remove constant features** - categorical features with cardinality=1 add noise

## Files

- `src/prepare_production_data.py` - Data preparation (with outlier clipping)
- `src/train_production_tabddpm.py` - Training script
- `src/test_experiment_019.py` - Evaluation script
- `checkpoints/experiment_019_v2/best_model.pt` - Trained model

## Status

- [x] Data preparation
- [x] Training (1000 epochs)
- [x] Evaluation
- [x] Documentation
- [ ] Privacy test (membership inference attack)
