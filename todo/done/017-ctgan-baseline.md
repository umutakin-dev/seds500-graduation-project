# Add CTGAN Baseline Comparison

**Priority:** MEDIUM
**Effort:** 2-3 days
**Addresses:** "Limited comparison scope - only SMOGN"

## Problem

Currently only comparing Diffusion vs SMOGN. CTGAN (Conditional Tabular GAN) is a popular baseline for synthetic tabular data generation. Adding it strengthens the comparison.

## What is CTGAN?

- GAN-based approach for tabular data (Xu et al., 2019)
- Handles mixed data types (continuous + categorical)
- Mode-specific normalization for continuous columns
- Conditional generator for class imbalance

## Installation

```bash
pip install ctgan
# or
pip install sdv  # Synthetic Data Vault includes CTGAN
```

## Implementation

```python
from ctgan import CTGAN

# Train
ctgan = CTGAN(epochs=300)
ctgan.fit(train_data, discrete_columns=['cat_col1', 'cat_col2'])

# Generate
synthetic_data = ctgan.sample(n=len(train_data))
```

## Experiment Plan

### For Experiment 011 (Manufacturing, 2 features)
- Train CTGAN on same training data
- Generate same amount of synthetic samples
- Evaluate: Train model on CTGAN synthetic, test on real

### For Experiment 013 (Ozel Rich, 29 features)
- Same process
- Compare: Diffusion vs CTGAN vs SMOGN

## Expected Results

| Method | Simple (2-5 feat) | Complex (29 feat) |
|--------|-------------------|-------------------|
| Diffusion | Works | Works |
| CTGAN | Works? | Works? |
| SMOGN | Works | Fails |

If CTGAN also works on complex data, our conclusion becomes:
*"Both diffusion and CTGAN are safer than SMOGN"*

If CTGAN fails on complex data:
*"Diffusion is the most robust method"*

Either result is valuable.

## Potential Issues

1. **Training time**: CTGANs can be slow (epochs needed)
2. **Mode collapse**: GANs may miss parts of distribution
3. **Hyperparameter sensitivity**: May need tuning

## Tasks

- [ ] Install ctgan or sdv
- [ ] Adapt training scripts for CTGAN
- [ ] Run CTGAN on Experiment 011 data
- [ ] Run CTGAN on Experiment 013 data
- [ ] Evaluate with same "train synthetic, test real" methodology
- [ ] Compare results across all three methods
- [ ] Update experiment RESULTS.md files
- [ ] Update thesis comparison section

## References

- Xu et al., "Modeling Tabular data using Conditional GAN" (NeurIPS 2019)
- https://github.com/sdv-dev/CTGAN
