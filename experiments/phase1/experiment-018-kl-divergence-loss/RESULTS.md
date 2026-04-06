# Experiment 018: Results

## Date
2026-01-15

## Summary

**TabDDPM-style implementation achieves 87% of baseline for replacement scenario** - a massive improvement over V6 (26.5%) and CTGAN (35.5%).

## Training

- **Epochs**: 1000
- **Best Loss**: 0.1615
- **Device**: CUDA (RTX 4070 Ti Super)
- **Training Time**: ~5 minutes

### Convergence
The model showed initial instability (epoch 1: mean in thousands) but converged well:

| Epoch | Gen Mean | Train Mean | At Boundary |
|-------|----------|------------|-------------|
| 1 | [12455, 11429, -25435] | [-0.42, -0.75, -0.35] | [1.0, 1.0, 1.0] |
| 300 | [-0.43, -0.86, -0.43] | [-0.42, -0.75, -0.35] | [0.08, 0.34, 0.05] |
| 1000 | [-0.42, -0.78, -0.34] | [-0.42, -0.75, -0.35] | [0.02, 0.18, 0.0] |

## Results

### Replacement Scenario (Synthetic Only)

| Method | Replacement R² | % of Baseline | Improvement |
|--------|----------------|---------------|-------------|
| V6 (simple) | 0.1712 | 26.5% | - |
| CTGAN | 0.2292 | 35.5% | +34% vs V6 |
| **Exp 018 (TabDDPM)** | **0.5628** | **87.3%** | **+229% vs V6, +146% vs CTGAN** |

### Augmentation Scenario (Original + Synthetic)

| Method | Augmentation R² | % of Baseline |
|--------|-----------------|---------------|
| V6 (simple) | 0.6355 | 98.5% |
| **Exp 018 (TabDDPM)** | **0.6395** | **99.1%** |

### All ML Models (Experiment 018)

| Scenario | RF R² | GB R² | Ridge R² |
|----------|-------|-------|----------|
| Baseline | 0.6451 | 0.6291 | 0.5960 |
| Replacement | 0.5628 (-0.08) | 0.5913 (-0.04) | 0.5772 (-0.02) |
| Augmentation | 0.6395 (-0.01) | 0.6294 (+0.00) | 0.5896 (-0.01) |

## Key Improvements from TabDDPM Implementation

1. **Log-space operations**: Numerical stability prevents overflow/underflow
2. **KL divergence loss**: Respects diffusion process structure (predicts posterior, not just x0)
3. **Gumbel-softmax sampling**: Proper categorical sampling instead of direct softmax
4. **Proper posterior computation**: q(x_{t-1}|x_t, x_0) computed correctly

## Generated Sample Quality

```
Generated numeric stats (should match training [-1, 1]):
  Mean: [-0.4235, -0.7577, -0.3706] (train: [-0.416, -0.75, -0.345])
  Std:  [0.3526, 0.2106, 0.2969] (train: [0.382, 0.255, 0.358])
  At boundary (>0.95): [1.5%, 14.4%, 0.5%]

Synthetic target stats:
  Mean: 943.7 (original: 968.4)
  Std:  288.1 (original: 346.9)
```

## Conclusion

**Hypothesis confirmed**: Implementing TabDDPM-style multinomial diffusion significantly improves replacement performance.

The key insight is that the original V6 implementation was using cross-entropy loss which ignores the diffusion process structure. TabDDPM's KL divergence loss on posteriors respects the diffusion process and leads to much better sample quality.

### Practical Implications

| Use Case | Best Method | Performance |
|----------|-------------|-------------|
| **Augmentation** | TabDDPM (Exp 018) | 99.1% of baseline |
| **Replacement/Privacy** | TabDDPM (Exp 018) | 87.3% of baseline |

**TabDDPM-style diffusion is now the best method for both use cases**, surpassing CTGAN for replacement scenarios by a large margin (87% vs 35%).

## Files

- `src/diffusion_tabddpm.py` - TabDDPM-style implementation
- `src/train_experiment_018.py` - Training script
- `src/test_experiment_018.py` - Evaluation script
- `checkpoints/experiment_018/best_model.pt` - Trained model

## Privacy Test (Membership Inference Attack)

| Method | Attack AUC | Status |
|--------|------------|--------|
| V6 Diffusion (Exp 016) | 0.5116 | SAFE |
| SMOGN (Exp 016) | 0.5253 | SAFE |
| **Exp 018 (TabDDPM)** | **0.5103** | **EXCELLENT** |

- AUC ~ 0.5 means attacker cannot determine if a record was in training set
- Exp 018 is slightly MORE private than V6 while achieving 3x better utility

## Status
[x] Implementation
[x] Training
[x] Evaluation
[x] Privacy Test
[x] Documentation
