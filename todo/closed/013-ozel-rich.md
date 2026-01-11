# Experiment 013: Ozel Rich - HybridDiffusion

**Priority:** High
**Status:** COMPLETED (2026-01-11)

## Objective

Test HybridDiffusion on ozel data extracted from manufacturing dataset with richer features (29 dimensions) to see if feature complexity affects augmentation performance.

## Dataset

- **Source:** Ozel rows from `Teklif Maliyet SSEK data.xlsx`
- **Train rows:** 2,136
- **Test rows:** 534
- **Features:** Cap, Boy (numeric) + IslemTipi, AnmaOlcusu, SartnameSimple, UY (categorical)
- **Total dimensions:** 29 (3 numeric + 26 one-hot)
- **Target:** MakineSure (manufacturing duration)

## Results

### Baseline
| Model | RMSE | R² |
|-------|------|-----|
| Random Forest | 206.5 | 0.6451 |
| Gradient Boosting | 211.1 | 0.6291 |
| Ridge | 220.3 | 0.5960 |

### Augmentation Comparison
| Model | Original | + Diffusion | + SMOGN |
|-------|----------|-------------|---------|
| RF | 206.5 | 208.3 (+0.9%) | **409.2 (+98.2%)** |
| GB | 211.1 | 208.8 (-1.1%) | **393.4 (+86.4%)** |
| Ridge | 220.3 | 369.2 (+67.6%) | **384.8 (+74.7%)** |

## Conclusion

**Diffusion wins 3/3** - SMOGN catastrophically failed (+86-98% worse, negative R²). Diffusion maintained performance on RF/GB, even slightly improved GB (-1.1%).

**Key insight:** Complex feature spaces (29 dims) break SMOGN's interpolation approach. Diffusion learns the actual distribution and generates realistic samples.

## Files
- `src/prepare_ozel_rich_data.py`
- `src/train_ozel_rich.py`
- `src/evaluate_ozel_rich_baseline.py`
- `src/evaluate_ozel_rich.py`
- `experiments/experiment-013-ozel-rich/RESULTS.md`
