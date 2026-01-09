# Experiment 001: Iris Baseline (Gaussian Diffusion)

**Date:** 2025-12-01 (initial), 2026-01-08 (re-run with ML efficiency)
**Status:** Completed

## Objective

Validate that our Gaussian diffusion implementation works correctly on a simple, well-understood dataset, and evaluate whether synthetic data improves ML model performance.

## Dataset

- **Name:** Iris
- **Samples:** 150 (120 train, 30 test)
- **Features:** 4 (all numerical)
  - sepal length, sepal width, petal length, petal width
- **Classes:** 3 (setosa, versicolor, virginica)
- **Why this dataset:** Small, fast to train, classification task for ML efficiency evaluation

## Method

- **Model:** Gaussian Diffusion with MLP Denoiser
- **Preprocessing:** QuantileTransformer -> clip to [-1, 1]
- **Architecture:** MLPDenoiser with hidden_dims=[256, 256, 256]
- **Training:** 500 epochs, batch_size=64, lr=1e-3
- **Noise schedule:** Cosine (1000 timesteps)
- **Device:** CUDA (RTX 4070 Ti Super)

## Results

### Statistical Fidelity

| Metric | Value |
|--------|-------|
| Final training loss | 0.1931 |
| Mean correlation difference | 0.0241 |

| Feature | Real Mean | Syn Mean | Real Std | Syn Std |
|---------|-----------|----------|----------|---------|
| sepal length | 0.0055 | -0.0258 | 0.3538 | 0.3357 |
| sepal width | 0.0020 | 0.0406 | 0.3432 | 0.3218 |
| petal length | 0.0048 | -0.0400 | 0.3473 | 0.3004 |
| petal width | 0.0053 | -0.0591 | 0.3560 | 0.3326 |

### ML Efficiency Evaluation

| Scenario | Logistic Regression | Random Forest |
|----------|---------------------|---------------|
| Real -> Real (baseline) | 96.67% | 90.00% |
| Synthetic -> Real | 86.67% (-10.0%) | 90.00% (+0.0%) |
| **Augmented -> Real** | 96.67% (+0.0%) | **96.67% (+6.7%)** |

### Confusion Matrix (Random Forest, Augmented -> Real)

```
              setosa  versicolor  virginica
setosa          10          0          0
versicolor       0          9          1
virginica        0          0         10
```

Accuracy: 96.67% | Precision: 97% | Recall: 97% | F1: 97%

## Conclusions

1. **Gaussian diffusion works** - synthetic data matches real data statistics closely (correlation diff: 0.024)

2. **ML Efficiency varies by model:**
   - Logistic Regression: Synthetic data alone performs worse (-10%), but doesn't hurt when augmented
   - Random Forest: Synthetic data alone matches baseline, augmentation improves by 6.7%

3. **Augmentation shows promise** - Random Forest improved from 90% to 96.67% accuracy when trained on real + synthetic data

4. **Label distribution issue** - Synthetic labels (assigned via 1-NN) are imbalanced [41, 50, 29] vs real [40, 40, 40]. This could be improved with conditional generation.

## Limitations

- Iris is a very small, simple dataset (150 samples, 4 features)
- No categorical features tested (Gaussian diffusion only)
- Label assignment via nearest neighbor is naive

## Figures

Generated visualizations for presentation/report:

| Figure | Description |
|--------|-------------|
| `figures/ml_efficiency.png` | Bar chart comparing Real/Synthetic/Augmented scenarios |
| `figures/confusion_matrix_baseline.png` | Confusion matrix for Real -> Real |
| `figures/confusion_matrix_augmented.png` | Confusion matrix for Augmented -> Real |
| `figures/distributions.png` | Real vs Synthetic feature distributions |
| `figures/correlations.png` | Correlation matrix comparison |

## Files

- **Training script:** `src/train.py`
- **Evaluation script:** `src/evaluate.py`
- **Report generator:** `src/report.py`
- **Notebook:** `notebooks/01_diffusion_explained.ipynb`
- **Checkpoints:** `checkpoints/iris/`
- **Source code:** `src/diffusion.py`, `src/models.py`

## Commands

```bash
# Train
uv run python src/train.py --dataset iris --epochs 500 --device cuda

# Evaluate
uv run python src/evaluate.py --checkpoint checkpoints/iris/final_model.pt --device cuda

# Generate report figures
uv run python src/report.py --experiment experiment-001-iris-baseline --device cuda
```
