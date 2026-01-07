# Experiment 001: Iris Baseline (Gaussian Diffusion)

**Date:** 2025-12-01
**Status:** Completed

## Objective

Validate that our Gaussian diffusion implementation works correctly on a simple, well-understood dataset.

## Dataset

- **Name:** Iris
- **Samples:** 150
- **Features:** 4 (all numerical)
  - sepal length, sepal width, petal length, petal width
- **Why this dataset:** Small, fast to train, easy to verify results visually

## Method

- **Model:** Gaussian Diffusion with MLP Denoiser
- **Preprocessing:** QuantileTransformer → clip to [-1, 1]
- **Architecture:** MLPDenoiser with hidden_dims=[256, 256, 256]
- **Training:** 500 epochs, batch_size=32, lr=1e-3
- **Noise schedule:** Cosine (1000 timesteps)

## Results

| Metric | Value |
|--------|-------|
| Training time | ~7 seconds (GPU) / ~1 min (CPU) |
| Final loss | 0.2089 |
| Mean correlation difference | 0.037 |

### Statistics Comparison

| Feature | Real Mean | Syn Mean | Real Std | Syn Std |
|---------|-----------|----------|----------|---------|
| sepal length | -0.0004 | -0.0112 | 0.3384 | 0.3408 |
| sepal width | 0.0005 | 0.0313 | 0.3390 | 0.3600 |
| petal length | 0.0007 | -0.0473 | 0.3391 | 0.3738 |
| petal width | -0.0015 | -0.0408 | 0.3592 | 0.3878 |

## Conclusions

1. **Gaussian diffusion works** - synthetic data matches real data statistics closely
2. **Correlation structure preserved** - mean absolute difference of 0.037 is excellent
3. **Ready for next steps** - foundation validated, can proceed to more complex experiments

## Files

- **Notebook:** `notebooks/01_diffusion_explained.ipynb` (tutorial + this experiment)
- **Checkpoints:** `checkpoints/iris/` (saved model weights)
- **Source code:** `src/diffusion.py`, `src/models.py`, `src/train.py`

