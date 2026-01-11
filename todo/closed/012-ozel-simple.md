# Experiment 012: Ozel Simple - HybridDiffusion

**Priority:** High
**Status:** COMPLETED (2026-01-11)

## Objective

Test HybridDiffusion on the standalone ozel dataset with simple features (Cap, Boy, IslemTipi) to see if augmentation helps on a smaller dataset with moderate baseline.

## Dataset

- **Source:** `ozel-muhendislik` dataset (pre-split)
- **Train rows:** 2,029
- **Test rows:** 508
- **Features:** Cap, Boy (numeric) + IslemTipi (categorical, 3 classes)
- **Target:** MakineSure (manufacturing duration)

## Results

### Baseline
| Model | RMSE | R² |
|-------|------|-----|
| Random Forest | 254.0 | 0.4844 |
| Gradient Boosting | 236.0 | 0.5551 |
| Ridge | 265.5 | 0.4368 |

### Augmentation Comparison
| Model | Original | + Diffusion | + SMOGN |
|-------|----------|-------------|---------|
| RF | 254.0 | 251.6 (-1.0%) | 249.8 (-1.6%) |
| GB | 236.0 | 240.4 (+1.9%) | 236.5 (+0.2%) |
| Ridge | 265.5 | 264.6 (-0.3%) | 276.6 (+4.2%) |

## Conclusion

**Tie** - Both methods within ±2%. Simple feature space (5 dims) doesn't differentiate methods.

## Files
- `src/prepare_ozel_data.py`
- `src/train_ozel.py`
- `src/evaluate_ozel_baseline.py`
- `src/evaluate_ozel.py`
- `experiments/experiment-012-ozel-hybrid/RESULTS.md`
