# Class-Conditional Diffusion for Imbalanced Data

**Source:** Experiment 009

## Task
Implement class-conditional diffusion with classifier-free guidance (CFG) for targeted minority class generation.

## Problem
- KNN labeling fails on imbalanced data (assigns 100% to majority class)
- Need to generate samples with known labels for proper augmentation

## Solution
- Train model with class labels as conditioning input
- Use label dropout (10%) for CFG training
- Generate samples for specific class during inference

## Key Components
- [x] Add `num_classes` parameter to denoiser models
- [x] Implement `label_dropout` in `training_loss()`
- [x] Implement `sample_with_guidance()` for guided sampling
- [x] Create training script (`train_adult_conditional.py`)
- [x] Create evaluation script (`evaluate_adult_conditional.py`)
- [x] Test different guidance scales (1.0, 2.0, 3.0)

## Results (Adult Dataset)
| Guidance | RF Accuracy | LR Accuracy |
|----------|-------------|-------------|
| **1.0** | **86.14% (+0.15%)** | **80.85% (+0.05%)** |
| 2.0 | 85.79% (-0.19%) | 75.33% (-5.47%) |
| 3.0 | 85.94% (-0.05%) | 76.74% (-4.05%) |
| SMOTE | 85.27% (-0.72%) | 74.50% (-6.30%) |

## Key Finding
- **guidance_scale=1.0 is best** for tabular data
- Pure conditional beats CFG (unlike image generation)
- Only method that improves accuracy over original

## Status
- [x] Implementation complete (2026-01-11)
- [x] Experiment 009 documented
- [x] PR #2 merged to main
